"""Run a full mock FRTB IMA workflow on synthetic data."""

from __future__ import annotations

import argparse
import os
import sys

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from market_risk.backtesting import Backtester
from market_risk.engine import MarketRiskEngine
from market_risk.es import ExpectedShortfallCalculator
from market_risk.mock_data import MockHistorySettings, MockPortfolioSettings, build_mock_portfolio, generate_market_history
from market_risk.performance import DailyMetrics, PerformanceMonitor
from market_risk.frtb import RTPLCapitalCalculator
from market_risk.pla import PLATester
from market_risk.rniv import RNIVConfig, rniv_addon
from market_risk.rtpl import GridRTPLConfig, compute_rtpl_series_grid
from market_risk.reporting import RunLogger
from market_risk.scenarios import HistoricalScenarioBuilder, default_tenor_grid
from market_risk.grid_config import DEFAULT_GRID_CONFIG_PATH, load_grid_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mock FRTB IMA risk workflow.")
    parser.add_argument("--fast", action="store_true", help="Reduce scenario and MC size for quick runs.")
    args = parser.parse_args()

    history_settings = MockHistorySettings(start=history_start())
    portfolio_settings = MockPortfolioSettings()

    if args.fast:
        history_settings = MockHistorySettings(start=history_start(), days=130, seed=history_settings.seed)
        portfolio_settings = MockPortfolioSettings(quantity_scale=1.0, mc_paths=200, mc_steps=30)

    history = generate_market_history(history_settings)
    base_market = history[-1]

    portfolio = build_mock_portfolio(base_market, portfolio_settings)

    builder = HistoricalScenarioBuilder(tenor_grid=default_tenor_grid(), horizon_days=1)
    engine = MarketRiskEngine(builder)

    scenarios = engine.build_scenarios(history)
    report = engine.compute_risk(portfolio, base_market, scenarios)

    prices = [portfolio.value(market) for market in history]
    actual_pnls = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    var_series = [report.var_result.value for _ in actual_pnls]

    backtest = Backtester().run(actual_pnls, var_series, window=min(250, len(actual_pnls)))

    grid_payload = latest_grid() if not args.fast else None
    grid_config = GridRTPLConfig(
        grid_config=grid_rtpl_config(args.fast, grid_payload),
        dividend_shifts=dividend_shifts(args.fast, grid_payload),
        rate_shifts=rate_shifts(args.fast, grid_payload),
        funding_shifts=funding_shifts(args.fast, grid_payload),
        borrow_shifts=borrow_shifts(args.fast, grid_payload),
    )
    rtpl = compute_rtpl_series_grid(
        portfolio,
        history,
        config=grid_config,
        t_ref=1.0,
        max_days=60 if args.fast else None,
    )
    matched_len = min(len(actual_pnls), len(rtpl))
    actual_pnls_match = actual_pnls[-matched_len:] if matched_len else []
    rtpl_match = rtpl[-matched_len:] if matched_len else []
    pla = PLATester().run(actual_pnls_match, rtpl_match)
    rtpl_es = ExpectedShortfallCalculator().compute(rtpl_match, engine.es_level, scenarios.horizon_days)
    hpl_es = ExpectedShortfallCalculator().compute(actual_pnls_match, engine.es_level, scenarios.horizon_days)
    rtpl_error = (rtpl_es.value - hpl_es.value) / max(abs(hpl_es.value), 1.0)
    rtpl_diff = [r - h for r, h in zip(rtpl_match, actual_pnls_match)]

    rtpl_capital = RTPLCapitalCalculator(es_level=engine.es_level).compute(
        portfolio,
        history,
        grid_config=grid_config,
        max_days=60 if args.fast else None,
    )
    rtpl_capital_error = (
        (rtpl_capital.total_es - report.ima_capital.total_es)
        / max(abs(report.ima_capital.total_es), 1.0)
    )

    rniv = rniv_addon(portfolio, base_market, RNIVConfig())
    capital_with_rniv = report.ima_capital.total_es + rniv

    monitor = PerformanceMonitor(desk="Equity Derivatives")
    monitor.add_metrics(
        DailyMetrics(
            asof=base_market.asof,
            var_result=report.var_result,
            es_result=report.es_result,
            backtest=backtest,
            pla=pla,
        )
    )

    print("=== Mock FRTB IMA Risk Report ===")
    print(f"Base date: {base_market.asof}")
    print(f"Positions: {len(portfolio.positions)}")
    print(f"Scenarios: {len(scenarios.scenarios)}")
    print(f"VaR (99%): {report.var_result.value:,.2f}")
    print(f"ES (97.5%): {report.es_result.value:,.2f}")
    print("Liquidity Horizon ES:")
    for lh, value in sorted(report.ima_capital.es_by_lh.items()):
        print(f"  LH {lh}d: {value:,.2f}")
    print(f"IMA Capital (sqrt sum): {report.ima_capital.total_es:,.2f}")
    print(f"IMA + RNIV add-on: {capital_with_rniv:,.2f} (RNIV={rniv:,.2f})")
    print(f"Backtest traffic light: {backtest.traffic_light} ({backtest.exceptions} exceptions)")
    print(
        "PLA status: "
        f"{pla.status} (corr={pla.correlation:.2f}, std_ratio={pla.std_ratio:.2f}, "
        f"beta={pla.beta:.2f}, r2={pla.r2:.2f}, "
        f"rmse={pla.rmse:,.2f}, explained={pla.explained_ratio:.2f})"
    )
    print(f"RTPL ES (grid): {rtpl_es.value:,.2f} (error vs HPL ES: {rtpl_error:.2%})")
    print(
        f"RTPL Capital (grid): {rtpl_capital.total_es:,.2f} "
        f"(error vs HPL capital: {rtpl_capital_error:.2%})"
    )
    if rtpl_diff:
        max_diff = max(abs(x) for x in rtpl_diff)
        mean_diff = sum(rtpl_diff) / len(rtpl_diff)
        print(f"RTPL vs HPL daily PnL diff: mean={mean_diff:,.2f}, max={max_diff:,.2f}")
    print("Performance summary:", monitor.summary())

    report_path = write_mock_report(
        base_market.asof,
        report=report,
        backtest=backtest,
        pla=pla,
        rtpl_es=rtpl_es.value,
        hpl_es=hpl_es.value,
        rtpl_error=rtpl_error,
        rtpl_capital=rtpl_capital.total_es,
        rtpl_capital_error=rtpl_capital_error,
        rniv=rniv,
        capital_with_rniv=capital_with_rniv,
    )

    logger = RunLogger()
    log_path = logger.log_run(
        name="mock_ima",
        asof=base_market.asof,
        params={
            "fast": args.fast,
            "portfolio_positions": len(portfolio.positions),
            "scenarios": len(scenarios.scenarios),
            "grid_config": grid_config.__dict__,
            "history_days": len(history),
        },
        outputs={"report": report_path},
        notes="mock IMA run with RTPL and grid sensitivities",
    )
    print(f"Run log: {log_path}")



def history_start() -> "date":
    from datetime import date

    return date(2023, 1, 2)


def grid_rtpl_config(fast: bool, payload: dict | None):
    from market_risk.grid_approx import GridConfig

    if fast or not payload:
        return GridConfig(
            spot_nodes=(0.85, 0.95, 1.0, 1.05, 1.15),
            vol_shifts=(-0.03, -0.01, 0.0, 0.01, 0.03),
        )
    return GridConfig(
        spot_nodes=tuple(payload.get("spot_nodes", GridConfig().spot_nodes)),
        vol_shifts=tuple(payload.get("vol_shifts", GridConfig().vol_shifts)),
    )


def dividend_shifts(fast: bool, payload: dict | None):
    if fast or not payload:
        return (-0.002, -0.001, 0.0, 0.001, 0.002)
    return tuple(payload.get("dividend_shifts", (-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005)))


def rate_shifts(fast: bool, payload: dict | None):
    if fast or not payload:
        return (-0.002, -0.001, 0.0, 0.001, 0.002)
    return tuple(payload.get("rate_shifts", (-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005)))


def funding_shifts(fast: bool, payload: dict | None):
    if fast or not payload:
        return (-0.002, -0.001, 0.0, 0.001, 0.002)
    return tuple(payload.get("funding_shifts", (-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005)))


def borrow_shifts(fast: bool, payload: dict | None):
    if fast or not payload:
        return (-0.002, -0.001, 0.0, 0.001, 0.002)
    return tuple(payload.get("borrow_shifts", (-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005)))


def latest_grid() -> dict | None:
    return load_grid_config(DEFAULT_GRID_CONFIG_PATH)


def write_mock_report(
    asof,
    report,
    backtest,
    pla,
    rtpl_es: float,
    hpl_es: float,
    rtpl_error: float,
    rtpl_capital: float,
    rtpl_capital_error: float,
    rniv: float,
    capital_with_rniv: float,
) -> str:
    import json
    import os

    payload = {
        "asof": asof.isoformat(),
        "var": report.var_result.value,
        "es": report.es_result.value,
        "ima_capital": report.ima_capital.total_es,
        "ima_capital_by_lh": report.ima_capital.es_by_lh,
        "rtpl_es": rtpl_es,
        "hpl_es": hpl_es,
        "rtpl_error": rtpl_error,
        "rtpl_capital": rtpl_capital,
        "rtpl_capital_error": rtpl_capital_error,
        "rniv": rniv,
        "capital_with_rniv": capital_with_rniv,
        "backtest": backtest.__dict__,
        "pla": pla.__dict__,
    }
    os.makedirs("market_risk/reports", exist_ok=True)
    path = os.path.join("market_risk/reports", f"ima_report_{asof.strftime('%Y%m%d')}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


if __name__ == "__main__":
    main()
