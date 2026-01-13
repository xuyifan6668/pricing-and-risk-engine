# Pricing and Product Coverage Roadmap

This file turns the product and model expansion roadmap into trackable milestones
with measurable acceptance criteria. Update statuses as work lands.

## How to use
- Keep phases in order; update status and checkboxes as items complete.
- Tie acceptance criteria to tests and report outputs whenever possible.
- See `docs/success_metrics.md` for thresholds and reporting targets.

## Current baseline (implemented)
- Products: spot, forwards, equity swaps, European and American options, barriers,
  digitals, Asians, lookbacks, cliquets, corridors, variance swaps, vol swaps,
  basket proxy.
- Models: Black-Scholes, Heston, Merton, local vol (placeholder), local-stochastic
  vol (leverage surface), hybrid LSV proxy.
- Engines: analytic, tree, PDE, Monte Carlo (LSM for Americans, Brownian bridge,
  variance reduction toggles, Halton QMC).
- Risk workflow: historical VaR/ES, FRTB IMA liquidity-horizon aggregation,
  PLA/backtesting, RTPL grid sensitivities, grid approximation/checks.
- Reporting: JSON/CSV/HTML reports and run logs with schema metadata.

## Phase 1 - Stabilize core vanilla coverage (4-6 weeks)
Status: complete

Deliverables
- [x] Deterministic Monte Carlo via seeded `EngineSettings`.
- [x] Cross-engine parity tests for European options (analytic vs tree vs PDE).
- [x] Auto Greek method: analytic for BS European options, bump fallback otherwise.
- [x] Greek stability tests (delta/gamma/vega) against bump size halving.

Acceptance criteria
- Parity test passes within abs 1e-2 or rel 1e-3 for BS European calls.
- Deterministic Monte Carlo produces identical prices with fixed seed.
- Bump Greek stability within 2-5 percent when bump size is halved.

## Phase 2 - Barrier and digital coverage (6-8 weeks)
Status: complete

Deliverables
- [x] PDE or analytic pricing for standard barrier types (up/down, in/out).
- [x] Monte Carlo continuous monitoring with Brownian bridge validation cases.
- [x] Digital pricing parity across analytic and MC for BS.

Acceptance criteria
- Barrier PDE/MC prices match analytic benchmarks (where available) within 2 percent.
- Brownian bridge barrier price is less than or equal to discrete monitoring.
- Digital option prices across engines within 2 percent for BS scenarios.

## Phase 3 - Asian, lookback, cliquet, corridor coverage (8-10 weeks)
Status: in progress

Deliverables
- [x] Monte Carlo variance reduction toggles (antithetic, control variate,
  moment matching, Halton QMC).
- [ ] Path-dependent payoff regression tests for Asian, lookback, cliquet, corridor.
- [ ] Monte Carlo stderr and config metadata propagated into pricing results and reports.
- [ ] Convergence sweep utility (paths/seed grid) with report outputs.

Acceptance criteria
- MC standard error shrinks roughly with 1/sqrt(N) across path products.
- Variance reduction reduces stderr by at least 25 percent at fixed paths.
- Pricing reports include stderr, confidence intervals, and MC configuration metadata.
- Regression tests match analytic or benchmark prices within 2-5 percent.

## Phase 4 - Volatility and variance swaps (8-12 weeks)
Status: in progress

Deliverables
- [x] Static replication from surface for variance swap fair strikes.
- [x] Vol swap pricing with variance-to-vol conversion and bias checks.
- [x] Monte Carlo realized variance/vol pricing.
- [ ] Corridor variance swap pricing with surface integration and MC fallback.
- [ ] Benchmark tests for variance/vol swap parity (analytic vs MC).

Acceptance criteria
- Variance swap fair strike matches analytic surface integration benchmarks.
- MC realized variance estimates converge to analytic fair strike within 2 percent.
- Vol swap prices are within 2-3 percent of sqrt-variance benchmarks under BS.

## Phase 5 - Model expansion and calibration depth (12-16 weeks)
Status: in progress

Deliverables
- [x] Local-stochastic vol calibration pipeline (direct leverage + particle).
- [ ] Calibration stability checks (parameter bounds, fit diagnostics, sanity).
- [ ] Local vol surface bootstrapping with arbitrage checks.
- [ ] Heston/Merton calibration to implied vol smiles.
- [ ] Model/engine compatibility matrix with coverage notes in docs.

Acceptance criteria
- LSV calibration produces stable leverage surfaces across tenors.
- Calibration diagnostics saved with run metadata and reproducible seeds.
- Calibration errors meet `docs/success_metrics.md` thresholds.

## Phase 6 - Portfolio and risk workflow enhancements (ongoing)
Status: in progress

Deliverables
- [x] Standardized pricing and risk artifacts (JSON/CSV/HTML) with schema metadata.
- [x] Grid approximation diagnostics and sensitivity check reports.
- [ ] Scenario pricing artifacts aligned with market risk reports.
- [ ] Stress testing library (historical + macro scenarios) with report templates.
- [ ] Performance profiling, caching, and parallelization hooks for portfolio-scale runs.
- [ ] Risk factor mapping and desk approval workflow wiring.

Acceptance criteria
- Pricing and risk reports share consistent schema versioning.
- Scenario pricing artifacts align with `docs/reporting.md` definitions.
- Stress reports include scenario metadata and reproducible inputs.
- Runtime targets meet `docs/success_metrics.md` thresholds.

## Phase 7 - Market data and curve/surface industrialization (12-20 weeks)
Status: planned

Deliverables
- [ ] Curve bootstrapping (OIS, futures, swaps) with multi-curve support.
- [ ] Vol surface calibration with arbitrage checks and extrapolation rules.
- [ ] Corporate actions, calendars, and day-count conventions in market data pipeline.
- [ ] Data quality rules, lineage tracking, and versioned snapshots.

Acceptance criteria
- Bootstrap curves replicate input quotes within tolerance.
- Surface no-arbitrage checks pass across tenors and strikes.
- Snapshot lineage supports reproducible pricing/risk runs.

## Phase 8 - Cross-asset expansion (16-24 weeks)
Status: planned

Deliverables
- [ ] FX, rates, credit, and commodities product libraries.
- [ ] Multi-asset correlation and cross-risk-factor simulation.
- [ ] Multi-curve discounting and collateralization conventions.
- [ ] Basket and hybrid products beyond proxy approximations.

Acceptance criteria
- Cross-asset parity and calibration tests pass within defined tolerances.
- Portfolio risk aggregates across asset classes with consistent risk factors.

## Phase 9 - Counterparty credit and XVA (16-24 weeks)
Status: planned

Deliverables
- [ ] Exposure profiles (EPE/PFE) with netting/CSA terms.
- [ ] CVA/DVA/FVA/MVA/KVA calculations and reporting.
- [ ] Wrong-way risk toggles and stress scenario integration.

Acceptance criteria
- XVA reports reconcile against exposure profiles and market data.
- Stress scenarios produce explainable shifts with audit trails.

## Phase 10 - Productionization and governance (ongoing)
Status: planned

Deliverables
- [ ] Model risk governance (validation evidence, approvals, versioning).
- [ ] Batch orchestration, distributed compute, and storage for large portfolios.
- [ ] Monitoring, SLAs, and alerting for pricing/risk services.
- [ ] Secure APIs, permissioning, and audit logs.

Acceptance criteria
- Reproducible runs with immutable inputs and outputs.
- Performance and availability targets meet business SLAs.
