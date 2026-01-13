"""Run grid-based approximation checks on mock portfolio."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
import sys

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from market_risk.grid_approx import GridConfig, PortfolioGridApproximator
from market_risk.mock_data import (
    MockHistorySettings,
    MockPortfolioSettings,
    build_mock_portfolio,
    generate_market_history,
)
from market_risk.scenarios import HistoricalScenarioBuilder, default_tenor_grid
from market_risk.reporting import RunLogger, SCHEMA_VERSION, build_report_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-based approximation demo.")
    parser.add_argument("--fast", action="store_true", help="Use fewer scenarios and MC paths.")
    args = parser.parse_args()

    history_settings = MockHistorySettings(start=history_start())
    portfolio_settings = MockPortfolioSettings()

    if args.fast:
        history_settings = MockHistorySettings(start=history_start(), days=90, seed=history_settings.seed)
        portfolio_settings = MockPortfolioSettings(quantity_scale=1.0, mc_paths=120, mc_steps=24)

    history = generate_market_history(history_settings)
    base_market = history[-1]
    portfolio = build_mock_portfolio(base_market, portfolio_settings)

    if args.fast:
        grid_config = GridConfig(
            spot_nodes=(0.85, 0.95, 1.0, 1.05, 1.15),
            vol_shifts=(-0.03, -0.01, 0.0, 0.01, 0.03),
        )
    else:
        grid_config = GridConfig()
    approx = PortfolioGridApproximator(portfolio, base_market)
    grid = approx.build_grid(grid_config)

    sens = approx.check_sensitivities(grid)
    acc = approx.check_interpolation_accuracy(grid, samples=20)

    builder = HistoricalScenarioBuilder(tenor_grid=default_tenor_grid())
    scenarios = builder.build(history)
    scenario_markets = scenarios.markets(base_market)
    if args.fast:
        scenario_markets = scenario_markets[:40]
    pnls = approx.approximate_pnls(grid, scenario_markets)
    pnl_errors = [approx_pnl - full_pnl for approx_pnl, full_pnl in pnls]
    mean_err = 0.0
    max_err = 0.0

    print("=== Grid Approximation Report ===")
    print(f"Grid nodes: spots={len(grid.spot_grid)}, vols={len(grid.vol_grid)}")
    print(f"Base value: {grid.base_value:,.2f}")
    print("Sensitivity (full vs grid):")
    print(f"  Delta: {sens.full_delta:,.4f} vs {sens.grid_delta:,.4f}")
    print(f"  Gamma: {sens.full_gamma:,.6f} vs {sens.grid_gamma:,.6f}")
    print(f"  Vega : {sens.full_vega:,.2f} vs {sens.grid_vega:,.2f}")
    print("Interpolation accuracy:")
    print(f"  Mean abs error: {acc.mean_abs_error:,.2f}")
    print(f"  RMS error: {acc.rms_error:,.2f}")
    print(f"  Max error: {acc.max_error:,.2f}")
    if pnl_errors:
        mean_err = sum(pnl_errors) / len(pnl_errors)
        max_err = max(abs(e) for e in pnl_errors)
        print("Scenario PnL approximation error:")
        print(f"  Mean: {mean_err:,.2f}")
        print(f"  Max : {max_err:,.2f}")

    run_id = build_run_id()
    report_inputs = {
        "fast": args.fast,
        "history_days": len(history),
        "portfolio_positions": len(portfolio.positions),
        "grid_config": asdict(grid_config),
        "grid_nodes": {"spots": len(grid.spot_grid), "vols": len(grid.vol_grid)},
        "scenario_samples": len(scenario_markets),
    }
    meta = build_report_meta(
        report_type="risk.grid_approx",
        run_id=run_id,
        asof=base_market.asof,
        inputs=report_inputs,
        generator="market_risk/run_grid_approx.py",
        engine="PortfolioGridApproximator",
        assumptions=(
            "Grid approximation uses spot/vol node grid around base snapshot.",
            "Scenario PnL errors compare grid approximation vs full reprice.",
        ),
    )
    report_paths = write_grid_approx_report(
        base_market.asof,
        meta=meta,
        base_value=grid.base_value,
        grid_spot_nodes=len(grid.spot_grid),
        grid_vol_nodes=len(grid.vol_grid),
        sensitivities=sens,
        accuracy=acc,
        scenario_error_mean=mean_err,
        scenario_error_max=max_err,
    )
    print(f"JSON report: {report_paths['json_report']}")
    print(f"CSV report: {report_paths['csv_report']}")
    print(f"HTML report: {report_paths['html_report']}")

    logger = RunLogger()
    log_path = logger.log_run(
        name="grid_approx",
        asof=base_market.asof,
        params=report_inputs,
        outputs=report_paths,
        notes="grid approximation diagnostics",
        version=SCHEMA_VERSION,
        run_id=run_id,
    )
    print(f"Run log: {log_path}")


def history_start() -> "date":
    from datetime import date

    return date(2023, 1, 2)


def build_run_id() -> str:
    from uuid import uuid4

    return str(uuid4())


def write_grid_approx_report(
    asof,
    meta: dict,
    base_value: float,
    grid_spot_nodes: int,
    grid_vol_nodes: int,
    sensitivities,
    accuracy,
    scenario_error_mean: float,
    scenario_error_max: float,
    output_dir: str = "market_risk/reports",
) -> dict:
    import csv
    import json
    import os

    summary = {
        "base_value": base_value,
        "grid_nodes": {"spots": grid_spot_nodes, "vols": grid_vol_nodes},
        "scenario_error_mean": scenario_error_mean,
        "scenario_error_max": scenario_error_max,
    }
    payload = {
        "meta": meta,
        "summary": summary,
        "sensitivities": sensitivities.__dict__,
        "accuracy": accuracy.__dict__,
    }
    os.makedirs(output_dir, exist_ok=True)
    stamp = asof.strftime("%Y%m%d")
    json_path = os.path.join(output_dir, f"grid_approx_report_{stamp}.json")
    csv_path = os.path.join(output_dir, f"grid_approx_report_{stamp}.csv")
    html_path = os.path.join(output_dir, f"grid_approx_report_{stamp}.html")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(render_grid_approx_html(meta, summary, sensitivities, accuracy))
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["run_id", meta.get("run_id", "")])
        writer.writerow(["asof", meta.get("asof", "")])
        writer.writerow(["base_value", base_value])
        writer.writerow(["grid_spot_nodes", grid_spot_nodes])
        writer.writerow(["grid_vol_nodes", grid_vol_nodes])
        writer.writerow(["scenario_error_mean", scenario_error_mean])
        writer.writerow(["scenario_error_max", scenario_error_max])
        writer.writerow(["full_delta", sensitivities.full_delta])
        writer.writerow(["grid_delta", sensitivities.grid_delta])
        writer.writerow(["full_gamma", sensitivities.full_gamma])
        writer.writerow(["grid_gamma", sensitivities.grid_gamma])
        writer.writerow(["full_vega", sensitivities.full_vega])
        writer.writerow(["grid_vega", sensitivities.grid_vega])
        writer.writerow(["accuracy_mean_abs_error", accuracy.mean_abs_error])
        writer.writerow(["accuracy_rms_error", accuracy.rms_error])
        writer.writerow(["accuracy_max_error", accuracy.max_error])
    return {"json_report": json_path, "csv_report": csv_path, "html_report": html_path}


def render_grid_approx_html(meta: dict, summary: dict, sensitivities, accuracy) -> str:
    def fmt(val: object, digits: int = 4) -> str:
        if val is None:
            return ""
        if isinstance(val, float):
            return f"{val:,.{digits}f}"
        return str(val)

    summary_rows = "".join(
        f"<tr><td>{key}</td><td>{fmt(value, 6)}</td></tr>" for key, value in summary.items()
    )
    sens_rows = "".join(
        f"<tr><td>{key}</td><td>{fmt(value, 6)}</td></tr>" for key, value in sensitivities.__dict__.items()
    )
    acc_rows = "".join(
        f"<tr><td>{key}</td><td>{fmt(value, 6)}</td></tr>" for key, value in accuracy.__dict__.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Grid Approximation Report</title>
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
  <h1>Grid Approximation Report</h1>
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
  <h2>Sensitivities</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{sens_rows}</tbody>
  </table>
  <h2>Accuracy</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{acc_rows}</tbody>
  </table>
</body>
</html>
"""


if __name__ == "__main__":
    main()
