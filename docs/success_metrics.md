# Success Metrics

This page defines suggested success metrics for pricing accuracy, risk explainability,
and runtime latency. Treat these as starting points; calibrate thresholds based on
your portfolio, market data quality, and compute budget.

## Pricing accuracy
- Vanilla BS benchmark error: absolute price error below 1e-4 for analytic engines.
- Monte Carlo convergence: RMS error decreases roughly with 1/sqrt(N) and is stable
  across seeds for deterministic settings.
- Greeks stability: bump-based Greeks change by less than 2-5% when bump size is halved.
- Surface sanity: implied vols remain positive and within expected bounds across tenors.

## Risk explainability
- RTPL vs HPL correlation: target >= 0.90 for a stable portfolio window.
- Explained ratio (PLA): target >= 0.85 for daily series.
- ES alignment: RTPL ES vs HPL ES relative error within 10-15%.
- Backtesting exceptions: remain in green zone (<= 4 exceptions in a 250-day window).

## Runtime and latency
- Pricing report runtime: < 5s for the sample portfolio on a laptop.
- Mock IMA workflow: < 60s for the fast run and < 5 minutes for full settings.
- Grid checks: < 90s for fast run and < 6 minutes for full settings.

## How to measure
- Capture metrics from `pricing_engine/run_pricing_report.py` and the reports in
  `pricing_engine/reports/`.
- Use `market_risk/run_mock.py` for RTPL, PLA, and backtesting metrics.
- Use `market_risk/run_grid_approx.py` and `market_risk/run_grid_checks.py` for grid
  approximation and sensitivity diagnostics.
