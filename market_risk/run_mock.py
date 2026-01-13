"""Run a full mock FRTB IMA workflow on synthetic data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from market_risk.reporting import RunLogger, SCHEMA_VERSION, build_report_meta
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

    run_id = build_run_id()
    report_inputs = {
        "fast": args.fast,
        "portfolio_positions": len(portfolio.positions),
        "scenarios": len(scenarios.scenarios),
        "grid_config": asdict(grid_config),
        "history_days": len(history),
    }
    report_paths = write_mock_report(
        base_market.asof,
        run_id=run_id,
        inputs=report_inputs,
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
    print(f"JSON report: {report_paths['json_report']}")
    print(f"CSV report: {report_paths['csv_report']}")
    if "html_report" in report_paths:
        print(f"HTML report: {report_paths['html_report']}")

    logger = RunLogger()
    log_path = logger.log_run(
        name="mock_ima",
        asof=base_market.asof,
        params=report_inputs,
        outputs=report_paths,
        notes="mock IMA run with RTPL and grid sensitivities",
        version=SCHEMA_VERSION,
        run_id=run_id,
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


def build_run_id() -> str:
    from uuid import uuid4

    return str(uuid4())


def write_mock_report(
    asof,
    run_id: str,
    inputs: dict,
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
    output_dir: str = "market_risk/reports",
) -> dict:
    import json
    import os
    import csv

    summary = {
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
    }
    payload = {
        "meta": build_report_meta(
            report_type="risk.ima",
            run_id=run_id,
            asof=asof,
            inputs=inputs,
            generator="market_risk/run_mock.py",
            engine="MarketRiskEngine",
            assumptions=(
                "Synthetic history and portfolio generated from market_risk/mock_data.py.",
                "Historical scenarios use 1-day horizon and tenor grid defaults.",
                "RTPL grid sensitivities use configured spot/vol/dividend/rate shifts.",
            ),
        ),
        "summary": summary,
        "checks": {"backtest": backtest.__dict__, "pla": pla.__dict__},
        "details": {
            "rtpl_error": rtpl_error,
            "rtpl_capital_error": rtpl_capital_error,
            "rniv": rniv,
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    stamp = asof.strftime("%Y%m%d")
    json_path = os.path.join(output_dir, f"ima_report_{stamp}.json")
    csv_path = os.path.join(output_dir, f"ima_report_{stamp}.csv")
    html_path = os.path.join(output_dir, f"ima_report_{stamp}.html")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(render_ima_report_html(payload))
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["run_id", run_id])
        writer.writerow(["asof", asof.isoformat()])
        for key, value in summary.items():
            writer.writerow([key, value])
        writer.writerow(["backtest_traffic_light", backtest.traffic_light])
        writer.writerow(["backtest_exceptions", backtest.exceptions])
        writer.writerow(["pla_status", pla.status])
        writer.writerow(["pla_correlation", pla.correlation])
        writer.writerow(["pla_std_ratio", pla.std_ratio])
        writer.writerow(["pla_beta", pla.beta])
        writer.writerow(["pla_r2", pla.r2])
        writer.writerow(["pla_rmse", pla.rmse])
        writer.writerow(["pla_explained_ratio", pla.explained_ratio])
    return {"json_report": json_path, "csv_report": csv_path, "html_report": html_path}


def render_ima_report_html(payload: dict) -> str:
    meta = payload.get("meta", {})
    summary = payload.get("summary", {})
    checks = payload.get("checks", {})
    backtest = checks.get("backtest", {})
    pla = checks.get("pla", {})

    def fmt(val: object, digits: int = 4) -> str:
        if val is None:
            return ""
        if isinstance(val, float):
            return f"{val:,.{digits}f}"
        return str(val)

    summary_rows = "".join(
        f"<tr><td>{key}</td><td>{fmt(value, 6)}</td></tr>" for key, value in summary.items()
    )
    backtest_rows = "".join(
        f"<tr><td>{key}</td><td>{fmt(value, 6)}</td></tr>" for key, value in backtest.items()
    )
    pla_rows = "".join(f"<tr><td>{key}</td><td>{fmt(value, 6)}</td></tr>" for key, value in pla.items())

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IMA Risk Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; background: #f5f3ef; color: #1f1f1f; }}
    h1 {{ margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #d7d0c5; padding: 6px 8px; font-size: 13px; text-align: left; }}
    th {{ background: #ece7df; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 6px; }}
    .meta-item {{ background: #fff; border: 1px solid #d7d0c5; padding: 8px; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Mock IMA Risk Report</h1>
  <div class="meta">
    <div class="meta-item">Run ID: {meta.get("run_id", "")}</div>
    <div class="meta-item">As-of: {meta.get("asof", "")}</div>
    <div class="meta-item">Created: {meta.get("created_at", "")}</div>
    <div class="meta-item">Generator: {meta.get("generator", "")}</div>
  </div>
  <h2>Summary</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
  <h2>Backtest</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{backtest_rows}</tbody>
  </table>
  <h2>PLA</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{pla_rows}</tbody>
  </table>
</body>
</html>
"""


if __name__ == "__main__":
    main()
