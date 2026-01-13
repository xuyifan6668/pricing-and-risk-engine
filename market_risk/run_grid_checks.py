"""Daily grid sensitivity and interpolation checks."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from market_risk.grid_checks import GridCheckConfig, GridCheckRunner, write_report
from market_risk.grid_config import DEFAULT_GRID_CONFIG_PATH, load_grid_config, save_grid_config
from market_risk.mock_data import (
    MockHistorySettings,
    MockPortfolioSettings,
    build_mock_portfolio,
    generate_market_history,
)
from market_risk.reporting import RunLogger, SCHEMA_VERSION, build_report_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily grid sensitivity checks.")
    parser.add_argument("--fast", action="store_true", help="Use lighter grid and fewer paths.")
    parser.add_argument("--grid-config", type=str, default="", help="Path to grid config JSON.")
    parser.add_argument(
        "--ignore-latest-grid",
        action="store_true",
        help="Ignore latest saved grid config.",
    )
    parser.add_argument(
        "--no-auto-grid",
        action="store_true",
        help="Disable writing updated grid config.",
    )
    args = parser.parse_args()

    history_settings = MockHistorySettings(start=history_start())
    portfolio_settings = MockPortfolioSettings()

    if args.fast:
        history_settings = MockHistorySettings(start=history_start(), days=60, seed=history_settings.seed)
        portfolio_settings = MockPortfolioSettings(quantity_scale=1.0, mc_paths=120, mc_steps=24)

    history = generate_market_history(history_settings)
    base_market = history[-1]
    portfolio = build_mock_portfolio(base_market, portfolio_settings)

    if args.fast:
        config = GridCheckConfig(
            spot_nodes=(0.85, 0.95, 1.0, 1.05, 1.15),
            vol_shifts=(-0.03, -0.01, 0.0, 0.01, 0.03),
            dividend_shifts=(-0.002, -0.001, 0.0, 0.001, 0.002),
            rate_shifts=(-0.002, -0.001, 0.0, 0.001, 0.002),
            funding_shifts=(-0.002, -0.001, 0.0, 0.001, 0.002),
            borrow_shifts=(-0.002, -0.001, 0.0, 0.001, 0.002),
            sample_stride=2,
            refine_delta_threshold=0.05,
            refine_gamma_threshold=0.1,
            refine_vega_threshold=0.1,
        )
    else:
        config = GridCheckConfig(
            spot_nodes=(0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3),
            vol_shifts=(-0.08, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.08),
            dividend_shifts=(-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005),
            rate_shifts=(-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005),
            funding_shifts=(-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005),
            borrow_shifts=(-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005),
            sample_stride=1,
            refine_delta_threshold=0.05,
            refine_gamma_threshold=0.1,
            refine_vega_threshold=0.1,
        )

    grid_payload = None
    if args.grid_config:
        grid_payload = load_grid_config(args.grid_config)
    elif not args.ignore_latest_grid and not args.fast:
        grid_payload = load_grid_config(DEFAULT_GRID_CONFIG_PATH)
    if grid_payload:
        config = replace(
            config,
            spot_nodes=tuple(grid_payload.get("spot_nodes", config.spot_nodes)),
            vol_shifts=tuple(grid_payload.get("vol_shifts", config.vol_shifts)),
            dividend_shifts=tuple(grid_payload.get("dividend_shifts", config.dividend_shifts)),
            rate_shifts=tuple(grid_payload.get("rate_shifts", config.rate_shifts)),
            funding_shifts=tuple(grid_payload.get("funding_shifts", config.funding_shifts)),
            borrow_shifts=tuple(grid_payload.get("borrow_shifts", config.borrow_shifts)),
        )

    runner = GridCheckRunner(portfolio, base_market)
    result = runner.run(config)

    run_id = build_run_id()
    report_inputs = {
        "fast": args.fast,
        "history_days": len(history),
        "portfolio_positions": len(portfolio.positions),
        "config": config.__dict__,
    }
    meta = build_report_meta(
        report_type="risk.grid_checks",
        run_id=run_id,
        asof=result.summary.asof,
        inputs=report_inputs,
        generator="market_risk/run_grid_checks.py",
        engine="GridCheckRunner",
        assumptions=(
            "Grid checks use the mock portfolio and synthetic market history.",
            "Interpolation errors are relative to full reprice across shifts.",
        ),
    )
    json_path, html_path = write_report(result, config, meta=meta)

    print("=== Grid Sensitivity Check ===")
    print(f"Date: {result.summary.asof}")
    print(f"Base value: {result.summary.base_value:,.2f}")
    print(f"Mean rel delta error: {result.summary.mean_rel_delta_error:.4f}")
    print(f"Mean rel gamma error: {result.summary.mean_rel_gamma_error:.4f}")
    print(f"Mean rel vega error: {result.summary.mean_rel_vega_error:.4f}")
    print(f"Mean rel cross-gamma error: {result.summary.mean_rel_cross_gamma_error:.4f}")
    print(f"Mean rel dividend error: {result.summary.mean_rel_dividend_error:.4f}")
    print(f"Mean rel dividend cross error: {result.summary.mean_rel_dividend_cross_error:.4f}")
    print(f"Mean rel rate error: {result.summary.mean_rel_rate_error:.4f}")
    print(f"Mean rel rate cross error: {result.summary.mean_rel_rate_cross_error:.4f}")
    print(f"Mean rel funding error: {result.summary.mean_rel_funding_error:.4f}")
    print(f"Mean rel funding cross error: {result.summary.mean_rel_funding_cross_error:.4f}")
    print(f"Mean rel borrow error: {result.summary.mean_rel_borrow_error:.4f}")
    print(f"Mean rel borrow cross error: {result.summary.mean_rel_borrow_cross_error:.4f}")
    if result.refinement.added_spot_nodes or result.refinement.added_vol_nodes:
        print(
            "Adaptive refine added nodes: "
            f"spots={len(result.refinement.added_spot_nodes)}, "
            f"vols={len(result.refinement.added_vol_nodes)}"
        )
    print(f"Interp mean rel error: {result.summary.interp_mean_rel_error:.4f}")
    print(f"Interp max rel error: {result.summary.interp_max_rel_error:.4f}")
    print(
        "Refinement plan: "
        f"flagged={result.refinement.flagged_cells}, "
        f"unrefined={result.refinement.unrefined_cells}, "
        f"rniv_addon={result.refinement.rniv_addon:,.2f}"
    )
    print(f"JSON report: {json_path}")
    print(f"HTML map: {html_path}")

    logger = RunLogger()
    log_path = logger.log_run(
        name="grid_checks",
        asof=result.summary.asof,
        params=report_inputs,
        outputs={"json_report": json_path, "html_report": html_path},
        notes="daily grid sensitivity checks",
        version=SCHEMA_VERSION,
        run_id=run_id,
    )
    print(f"Run log: {log_path}")

    if not args.no_auto_grid:
        next_payload = {
            "asof": result.summary.asof.isoformat(),
            "spot_nodes": result.refinement.spot_nodes,
            "vol_shifts": result.refinement.vol_shifts,
            "dividend_shifts": list(config.dividend_shifts),
            "rate_shifts": list(config.rate_shifts),
            "funding_shifts": list(config.funding_shifts),
            "borrow_shifts": list(config.borrow_shifts),
            "refinement": result.refinement.__dict__,
        }
        latest_path = save_grid_config(DEFAULT_GRID_CONFIG_PATH, next_payload)
        dated_path = save_grid_config(
            os.path.join(
                "market_risk/reports",
                f"grid_config_{result.summary.asof.strftime('%Y%m%d')}.json",
            ),
            next_payload,
        )
        print(f"Saved grid config: {latest_path}")
        print(f"Dated grid config: {dated_path}")


def history_start() -> "date":
    from datetime import date

    return date(2023, 1, 2)


def build_run_id() -> str:
    from uuid import uuid4

    return str(uuid4())


if __name__ == "__main__":
    main()
