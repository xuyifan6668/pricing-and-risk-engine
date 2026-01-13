# Automation and Reproducibility

This document outlines recommended steps for consistent local and CI execution.

## Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip pytest
```

## Reproducibility tips
- Use deterministic engine settings (`EngineSettings(seed=...)`) for MC workflows.
- Use the `--fast` flag for quick local runs, and full settings for baseline runs.
- Capture report artifacts and run logs in CI to track drift over time.

## Suggested CI steps
1. Install dependencies.
2. Run `pytest -q`.
3. Run key workflows (fast mode) to ensure reports still generate:
   ```bash
   python pricing_engine/run_pricing_report.py
   python market_risk/run_mock.py --fast
   python market_risk/run_grid_approx.py --fast
   python market_risk/run_grid_checks.py --fast
   ```
4. Archive `pricing_engine/reports/` and `market_risk/reports/` as build artifacts.

## Agent workflow (optional)
To run the Planner -> Implementer -> Tester -> Reviewer -> Manager loop, see
`agents/README.md` and `agents/run_pipeline.py`.
