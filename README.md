# Equity Derivatives Pricing and Market Risk Sandbox

This repository combines a Python pricing engine for equity derivatives, an FRTB IMA-style market risk workflow, and a small browser-based volatility surface calibration demo.

## Highlights
- Pricing models: Black-Scholes, Heston, Merton, local vol, and local-stochastic vol.
- Numerics: analytic, tree, PDE, Monte Carlo (with variance reduction hooks).
- Market risk: historical VaR/ES, FRTB IMA aggregation, RTPL grid sensitivities, PLA/backtesting.
- UI demo: a static "Vol Surface Calibration Studio" with a 3D canvas view.

## Project layout
- `pricing_engine/` core pricing, models, numerics, calibration, market data.
- `market_risk/` FRTB IMA workflow, scenario generation, RTPL, reporting.
- `surface_app/` static HTML/CSS/JS calibration demo.
- `docs/spec.md` technical specification and architecture notes.
- `tests/` unit tests for pricing and market risk components.

## Quickstart
Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip pytest
```

## Run the risk workflows

Mock FRTB IMA run (synthetic history + portfolio):
```bash
python market_risk/run_mock.py --fast
```

Grid approximation diagnostics:
```bash
python market_risk/run_grid_approx.py --fast
```

Daily grid sensitivity checks (JSON + HTML report):
```bash
python market_risk/run_grid_checks.py --fast
```

## Run tests
```bash
pytest -q
```

## Surface app
Serve the static UI locally:
```bash
cd surface_app
python -m http.server 8000
```
Then open `http://localhost:8000` in a browser.

## Notes
- Mock data comes from `market_risk/mock_data.py` and `tests/market_scenarios.py`.
- For architecture details, see `docs/spec.md`.
