# Technical Specification

This document summarizes the current architecture, product/model coverage, and
engine compatibility for the pricing and market risk workflows.

## Scope
- Equity derivatives pricing for single-underlying products.
- Market risk workflow with historical VaR/ES, RTPL, and PLA diagnostics.
- Lightweight calibration scaffolding and reporting artifacts.

## Architecture overview
- `pricing_engine/`: pricing models, products, numerics, calibration, and reporting.
- `market_risk/`: scenario generation, RTPL/PLA analytics, grid sensitivities, reports.
- `surface_app/`: static volatility surface calibration demo UI.

Key workflow
1. Build `MarketDataSnapshot` inputs (spot, curves, vol surface, dividends).
2. Price products with `Engine` + `Model` combinations via `pricing_engine.api.price`.
3. Aggregate portfolio values and risk metrics in `market_risk`.
4. Emit reports and run logs for auditability.

## Products
Vanilla
- Spot, forwards, equity swaps.
- European options, American options.

Exotics and volatility products
- Barriers (up/down, in/out), digitals.
- Asian, lookback, cliquet, corridor options.
- Variance and volatility swaps.
- Basket proxy (single-spot approximation).

## Models
- Black-Scholes (lognormal).
- Heston stochastic volatility.
- Merton jump diffusion.
- Local volatility (placeholder grid).
- Local-stochastic volatility (LSV) with leverage surface support.
- Hybrid local-stochastic proxy (placeholder).

## Engines and compatibility
- Analytic:
  - Black-Scholes European options, digitals, forwards, equity swaps, variance/vol swaps.
- Tree (CRR):
  - Black-Scholes European and American options.
- PDE:
  - Black-Scholes European options (finite difference).
  - Barrier options (up/down, in/out) via knock-out PDE and in/out parity when rebate is zero.
- Monte Carlo:
  - Black-Scholes and stochastic-vol models (path simulation).
  - LSM for American options.
  - Brownian bridge support for barrier monitoring.

## Greeks
- Analytic Greeks for Black-Scholes European options.
- Bump-based Greeks for other products/models (spot/vol/rate).
- Default mode is auto: analytic when available, bump otherwise.

## Calibration
- Base calibrator scaffold (uses model initial guesses).
- LSV leverage surface builders:
  - Direct leverage extraction.
  - Particle-based leverage calibration (iterative).

## Reporting and auditability
- Standardized JSON/CSV/HTML outputs for pricing and risk runs.
- Shared `meta` block includes schema version, run id, and generator metadata.
- Run logs captured in `market_risk/reports/runs/`.
- See `docs/reporting.md` for schema details.

## Known limitations
- Single-asset focus; basket handled as proxy only.
- Simplified MC and calibration routines intended for education and prototyping.
- Engine coverage is intentionally limited by product/model compatibility.

## References
- `docs/data_inputs_outputs.md`: inputs and validation checks.
- `docs/success_metrics.md`: accuracy and runtime targets.
- `docs/reporting.md`: report inventory and schema notes.
