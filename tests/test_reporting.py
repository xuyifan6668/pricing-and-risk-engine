from datetime import date, timedelta
import csv

from market_risk.grid_approx import GridConfig
from market_risk.reporting import build_report_meta
from orchestrator import AgentSpec, CommandRunner
from pricing_engine.api import PricingSettings
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.numerics.base import EngineSettings
from pricing_engine.numerics.monte_carlo import MonteCarloEngine
from pricing_engine.numerics.trees import TreeEngine
from pricing_engine.products.base import Underlying
from pricing_engine.products.vanilla import EuropeanOption
from pricing_engine.reporting import serialize_engine
from pricing_engine.run_pricing_report import (
    TradeSpec,
    build_sample_market,
    build_trade_results,
    write_report_files,
)
from pricing_engine.utils.types import OptionType


def test_serialize_engine_includes_params():
    engine = TreeEngine(steps=123)
    settings = EngineSettings(deterministic=True, max_steps=42)
    payload = serialize_engine(engine, settings)

    assert payload["name"] == "tree"
    assert payload["settings"]["max_steps"] == 42
    assert payload["params"]["steps"] == 123


def test_build_report_meta_handles_dataclass_inputs():
    grid_config = GridConfig(spot_nodes=(0.9, 1.0), vol_shifts=(0.0, 0.01))
    meta = build_report_meta(
        report_type="risk.grid_approx",
        run_id="run-1",
        asof=date(2024, 1, 2),
        inputs={"grid_config": grid_config},
        generator="tests",
    )

    inputs = meta["inputs"]
    assert isinstance(inputs["grid_config"], dict)
    assert inputs["grid_config"]["spot_nodes"] == [0.9, 1.0]
    assert inputs["grid_config"]["vol_shifts"] == [0.0, 0.01]


def test_command_runner_args_preserve_spaces():
    agent = AgentSpec(name="critic", path="some path/agent.md", instructions="")
    args = CommandRunner._format_command_args("python {agent_file} --role {agent}", agent)

    assert args == ["python", "some path/agent.md", "--role", "critic"]


def test_pricing_report_csv_has_report_run_id(tmp_path):
    trades = [
        {
            "trade_id": "T-1",
            "product": {
                "product_type": "EuropeanOption",
                "option_type": "CALL",
                "strike": 100.0,
                "maturity": "2024-06-30",
            },
            "model": {"name": "black_scholes"},
            "engine": {"name": "analytic"},
            "result": {"price": 1.23, "greeks": {}, "diagnostics": {"run_id": "pricing-1"}},
        }
    ]
    payload = {
        "meta": {"run_id": "report-1", "asof": "2024-01-02", "created_at": "2024-01-02T00:00:00"},
        "summary": {"trade_count": 1, "currency": "USD"},
        "market": {"snapshot_id": "snap-1"},
        "trades": trades,
    }

    _, csv_path, _ = write_report_files(
        str(tmp_path),
        date(2024, 1, 2),
        payload,
        trades,
        report_run_id="report-1",
    )

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        row = next(reader)

    assert header[0] == "report_run_id"
    assert row[0] == "report-1"
    assert row[-1] == "pricing-1"


def test_pricing_report_includes_mc_metrics():
    market = build_sample_market(date(2024, 1, 2))
    trade = TradeSpec(
        trade_id="MC-1",
        product=EuropeanOption(
            underlying=Underlying("XYZ"),
            currency="USD",
            maturity=market.asof + timedelta(days=180),
            strike=100.0,
            option_type=OptionType.CALL,
        ),
        model=BlackScholesModel(sigma=0.2),
        engine=MonteCarloEngine(num_paths=1000, timesteps=40, use_antithetic=True),
        settings=PricingSettings(
            engine_settings=EngineSettings(seed=7, deterministic=True),
            compute_greeks=False,
        ),
    )

    trade_results = build_trade_results([trade], market)
    result = trade_results[0]["result"]
    metrics = result["metrics"]

    assert metrics["stderr"] >= 0.0
    assert metrics["ci_lower"] <= result["price"] <= metrics["ci_upper"]
    assert trade_results[0]["engine"]["params"]["num_paths"] == 1000
    assert trade_results[0]["settings"]["engine_settings"]["seed"] == 7
