"""Run grid-based approximation checks on mock portfolio."""

from __future__ import annotations

import argparse
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
from market_risk.reporting import RunLogger


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

    logger = RunLogger()
    log_path = logger.log_run(
        name="grid_approx",
        asof=base_market.asof,
        params={
            "fast": args.fast,
            "grid_config": grid_config.__dict__,
            "sensitivities": sens.__dict__,
            "accuracy": acc.__dict__,
            "scenario_error_mean": mean_err if pnl_errors else 0.0,
            "scenario_error_max": max_err if pnl_errors else 0.0,
        },
        outputs={},
        notes="grid approximation diagnostics",
    )
    print(f"Run log: {log_path}")


def history_start() -> "date":
    from datetime import date

    return date(2023, 1, 2)


if __name__ == "__main__":
    main()
