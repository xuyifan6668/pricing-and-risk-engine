# Equity & Equity Derivatives Pricing Engine - Technical Specification

## Goals and scope
Build an industry-grade pricing engine for equity and equity derivatives (vanilla and exotics) with robust calibration, fast pricing, complete risk, and auditable model governance. The system supports multiple model families, multiple numerical methods, and consistent market data handling (curves, dividends, calendars, and corporate actions).

Key requirements:
- Coverage: spot, forwards, swaps, European/American options; barriers, digitals, Asians (avg price/strike), lookbacks, cliquets, corridors, variance/vol swaps, and optional add-ons (e.g., convertibles via equity optionality components).
- Models: Black-Scholes, local vol, Heston, Merton jump diffusion, hybrid local-stoch.
- Methods: analytic closed forms, PDE/FD, Monte Carlo (QMC), trees; AAD support for Greeks.
- Market data: OIS discounting, equity funding curve, borrow costs; discrete/continuous dividends; corporate actions.
- Engineering: scalable services, reproducible pricing, audit trails, and robust diagnostics.

## Architecture diagram (text)
```
[Market Data Feed] ---> [Ingestion + Normalization] ---> [Market Data Cache]
                               |                             |
                               v                             v
                        [Corporate Actions]            [Curves + Surfaces]
                               |                             |
                               v                             v
[Pricing API] <--> [Product Registry] <--> [Model Registry] <--> [Calibration Service]
      |                     |                       |                 |
      v                     v                       v                 v
[Pricing Service] ---> [Pricers/Engines] <--> [Numerical Backends] ---> [Results + Audit]
      |                                                         |
      v                                                         v
[Greeks/Risk] <---------------------------------------- [Diagnostics + Logs]
```

## Major design choices (with justification)
- **Separation of concerns (Product/Model/Engine)**: Products encode payoffs, models encode dynamics, engines encode numerics. This allows mixing products and models while swapping numerical methods without changing product code.
- **Market data as immutable snapshot**: A pricing run consumes a versioned market snapshot to ensure reproducibility and audit trails.
- **Unified curve abstractions**: Curves (OIS, funding, borrow) and dividend term structures share interfaces to simplify scenario risk and cross-Greeks.
- **Calibration as a service**: Decoupling calibration from pricing allows caching and reusing calibrated parameters, consistent with how desks operate.
- **Multiple numerics**: Closed-form when available for speed and deterministic outputs; PDE/trees for early exercise and barriers; MC/QMC for path-dependent exotics.
- **Diagnostic-rich outputs**: Pricing outputs include metadata (model version, calibration fitness, numerical settings) for audit and stability checks.

## Core data structures and APIs

### Pricing API
Python-first API with an optional C++/Rust backend for numerics:
```
price(product, market_snapshot, model, engine, settings) -> PricingResult
```
- `product`: Product subclass with payoff definition.
- `market_snapshot`: immutable MarketDataSnapshot with curves, vols, dividends.
- `model`: Model instance with parameters.
- `engine`: Engine instance (Analytic, PDE, MC, Tree).
- `settings`: numerical and risk settings, e.g., grids, tolerance, RNG seed.

### Market data
- **Curves**: `Curve` interface providing `df(t)`, `zero_rate(t)`, `fwd_rate(t1,t2)`.
- **Dividend term structures**: discrete schedules (ex-div dates + cash amounts), continuous yield curves, or hybrid schedules.
- **Vol surfaces**: implied vol by (expiry, strike) or (expiry, delta). Arbitrage-checked and smoothly interpolated.
- **Calendars/Corporate actions**: calendars for business-day conventions; splits and special dividends stored as events and applied to underlying adjustments.

### Model registry
- Model families registered with parameter schemas and calibration requirements.
- Supports multiple variants and hybrid models (local-stoch with vol-of-vol term structure).

### Product abstraction
Products are immutable definitions of payoffs and exercise features (European, American, Bermudan). Products may declare required market data (e.g., dividends, borrow curve).

## Product and model class hierarchy (conceptual)
```
Product
├── Spot
├── Forward
├── Swap
├── Option
│   ├── EuropeanOption
│   ├── AmericanOption
│   └── BermudanOption
├── BarrierOption (up/down, in/out)
├── DigitalOption
├── AsianOption (avg price/strike)
├── LookbackOption
├── Cliquet
├── Corridor
├── VarianceSwap
└── VolSwap

Model
├── BlackScholesModel
├── LocalVolModel
├── HestonModel
├── MertonJumpModel
└── HybridLocalStochModel

Engine
├── AnalyticEngine
├── TreeEngine
├── PDEEngine
└── MonteCarloEngine (incl. QMC)
```

## Numerical methods and key algorithms

### Closed-form pricing (where available)
- Black-Scholes closed forms for European calls/puts, digitals, forwards.
- Heston Fourier and Merton characteristic function methods for vanilla options.

### Trees
- Recombining trees for early exercise and discrete dividends.
- Time step aligned with dividend and corporate action dates.

### PDE/FD
- Crank-Nicolson or ADI schemes for local volatility and Heston.
- Fitted boundaries for barrier options.

### Monte Carlo
- Pathwise simulation for local and stochastic vol; jump diffusion via Poisson jumps.
- QMC (Sobol) for variance reduction.
- Longstaff-Schwartz for American style when needed (regression on basis of spot levels).
- Brownian-bridge adjustment for continuous barrier monitoring between MC time steps.
- LSV calibration via particle method: simulate SV paths with leverage surface and set
  `L(t,S)=sigma_loc(t,S)/sqrt(E[v_t|S_t=S])`, with kernel regression for the conditional variance.

## Industry pricing map (typical desk usage)
- **Spot/Forward/Equity swap**: analytic carry model with OIS discounting and funding/borrow adjustments.
- **European options**: Black-Scholes closed form; Heston/Merton via Fourier/FFT; local vol via PDE.
- **American options**: binomial/trinomial trees or PDE; MC + LSM for large dimensional problems.
- **Barriers**: closed-form (Reiner-Rubinstein) for standard barriers; PDE or MC for complex/monitoring.
- **Digitals**: closed-form under BS; PDE for local/stoch vol; MC for exotic monitoring.
- **Asian/Lookback/Cliquet/Corridor**: MC with variance reduction and control variates (geometric Asian).
- **Variance/Vol swaps**: static replication from option surface (log contract) or MC on log returns.
  - Variance fair strike: `K_var = (2/T) * exp(rT) * ∫ OTM(K) / K^2 dK` (OTM price is discounted option PV).
  - Vol swap: use `sqrt(K_var)` with convexity adjustment if vol-of-vol available.

### Reasonable default numerical parameters
- **MC**: 20k–100k paths, 252 time steps, antithetic variates; fixed seed in deterministic mode.
- **Tree**: 300–600 steps for American options; dividend dates aligned to nodes.
- **PDE**: 200–400 spatial points, 200–500 time steps; Crank-Nicolson/implicit.

### Greeks
- **AAD**: for MC and PDE (adjoint). Highest performance, but more complex setup and memory.
- **Bump-and-revalue**: robust for any engine, more expensive.
- **Analytic**: when closed-form formulas exist.

## Calibration pipeline

### Inputs
- Market quotes: spot, forward curve, option implied vols (by strike/delta), dividends, borrow curve.
- Constraints: positivity of variance, Feller condition (Heston), monotonicity and convexity of implied vol.
- Stability: parameter bounds and time smoothing.

### Objective function
- Weighted least squares on implied vol or price space:
  - `min sum w_i * (model_iv_i - market_iv_i)^2`
- Robust penalties for arbitrage violations and parameter extremes.

### Regularization
- Time smoothing for term-structured parameters.
- L2 penalty on deviations from prior parameters.

### Stability checks
- Recalibration with perturbed market data to ensure robustness.
- Reject if implied vol surface violates no-arb (calendar or butterfly).

### Calibration flow (pseudocode)
```
function calibrate(model, market, quotes, settings):
    params0 = model.initial_guess(market)
    constraints = model.constraints(market)
    def objective(params):
        model.set_params(params)
        ivs = model.implied_vol_surface(quotes.expiries, quotes.strikes)
        return weighted_error(ivs, quotes.market_ivs) + regularization(params)

    result = optimizer.minimize(objective, params0, constraints)
    if not result.converged:
        return CalibrationResult(status="failed", diagnostics=result.diag)

    if not arb_checks_pass(model, market):
        return CalibrationResult(status="rejected", diagnostics="arb check fail")

    return CalibrationResult(status="ok", params=result.params, diagnostics=result.diag)
```

## Key algorithms (pseudocode)

### Monte Carlo pricing (generic payoff)
```
function price_mc(product, model, market, settings):
    rng = RNG(seed=settings.seed)
    paths = model.simulate_paths(market, settings.timesteps, settings.num_paths, rng)
    payoffs = [product.payoff(path, market) for path in paths]
    disc = market.discount_curve.df(product.maturity)
    price = disc * mean(payoffs)
    stderr = disc * std(payoffs) / sqrt(num_paths)
    return PriceWithError(price, stderr)
```

### PDE pricing (barrier)
```
function price_pde_barrier(product, model, market, settings):
    grid = build_grid(product, model, market, settings)
    payoff = terminal_payoff(product, grid.S)
    for t in reversed(grid.time):
        payoff = pde_step(payoff, t, model, market, grid)
        payoff = apply_barrier_conditions(payoff, product, grid.S)
    return interpolate(payoff, S0)
```

### Tree pricing (American)
```
function price_tree_american(product, model, market, settings):
    tree = build_tree(model, market, settings.steps)
    values = product.payoff(tree.S[:, -1])
    for t in reversed(tree.times[:-1]):
        cont = discount(expectation(values, tree.probs))
        exc = product.payoff(tree.S[:, t])
        values = max(cont, exc)
    return values[0]
```

## Consistency and arbitrage checks
- Call spread monotonicity and convexity (butterfly) checks for each expiry.
- Calendar spread check across maturities.
- Static no-arb checks on dividend-adjusted forwards.

## System components

### Market data ingestion
- Data adapters to vendor feeds; normalized to internal schema.
- Caching keyed by `(asof, source, instrument)`.

### Calibration service
- Runs scheduled or on-demand calibration for each model.
- Stores calibrated parameters with metadata (fit error, constraints, timestamps).

### Pricing service
- Stateless pricing service with caching (keyed by product + model + market snapshot + settings).
- Concurrency via async requests; heavy numerics offloaded to C++/Rust or separate worker pools.

### Diagnostics and audit
- Each pricing run records model params, calibration id, numerical settings, and checks.

## Performance and reliability
- **Latency targets**: single-vanilla <5ms (analytic), <50ms (PDE/tree), <200ms (MC small); batch throughput >10k options/sec.
- **Deterministic mode**: fixed seeds, consistent grid construction, stable numerical tolerances.
- **Numerical stability**: adaptive grid or time steps; error bounds for MC and PDE.
- **Fallbacks**: if calibration fails, use previous day parameters or a simpler model (e.g., Black-Scholes).

## Validation and testing plan
- **Unit tests**: known analytic formulas (BS, digitals); put-call parity; PDE vs analytic for simple cases.
- **Regression tests**: market snapshot reprice with known outputs.
- **Benchmarks**: compare to vendor prices with tolerance thresholds.
- **Convergence tests**: MC error ~ O(1/sqrt(N)), PDE grid refinement.

## Minimal working skeleton
See `pricing_engine/` package for module stubs and interfaces.

## Implementation status (current)
- Pricing API, product/model/engine abstractions, and market data snapshots are implemented in Python.
- Analytic, tree, and Monte Carlo engines cover core vanilla and select exotic payoffs.
- Greeks support analytic (European BS) and bump-and-reprice paths.
- Risk workflows and a static volatility surface demo are available under `market_risk/` and `surface_app/`.

## PM follow-ups / next goals
- Define MVP product coverage by desk priority (vanilla + barrier + Asian first) and lock acceptance criteria.
- Specify deterministic-mode expectations for each engine (MC, PDE, tree) and required audit fields.
- Confirm calibration scope (Heston, local vol, LSV) and required quote inputs per model.
- Align cache strategy (market snapshot versioning + pricing run cache keys) with audit requirements.
- Provide performance targets by product family and acceptable error tolerances for regression tests.

## PM summary (roadmap + plan)
- **Phase 0: Definition & governance**
  - Finalize MVP product list and acceptance criteria with trading/risk stakeholders.
  - Lock deterministic-mode expectations (seed rules, audit metadata, reproducibility).
  - Define calibration coverage and data dependencies per model family.
- **Phase 1: Core pricing readiness**
  - Harden analytic + tree pricing for vanilla and barrier products.
  - Confirm Monte Carlo determinism requirements and add regression baselines.
  - Establish market snapshot versioning + pricing cache keys for audit trails.
- **Phase 2: Calibration & risk integration**
  - Implement/extend calibration services with persistence + diagnostics.
  - Define benchmark suites (vendor parity, convergence, regression outputs).
  - Integrate risk sensitivities and scenario workflows with pricing outputs.
- **Phase 3: Performance & productionization**
  - Set latency/throughput SLOs by product family and validate.
  - Add monitoring/audit dashboards and release governance gates.
