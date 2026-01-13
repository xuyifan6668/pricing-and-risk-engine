# Onboarding Guide

This guide provides persona-focused quickstarts for common workflows.

## Pricing analyst
Goal: validate pricing outputs and Greeks for a small trade set.

1. Create a virtual environment and install pytest:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip pytest
   ```
2. Run the sample pricing report:
   ```bash
   python pricing_engine/run_pricing_report.py
   ```
3. Review reports in `pricing_engine/reports/` (JSON, CSV, HTML).
4. Modify the trade list in `pricing_engine/run_pricing_report.py` to add strikes,
   maturities, or alternative engines.

## Market risk analyst
Goal: run the mock FRTB IMA workflow and validate risk explainability.

1. Run the mock IMA workflow:
   ```bash
   python market_risk/run_mock.py --fast
   ```
2. Run grid approximation diagnostics:
   ```bash
   python market_risk/run_grid_approx.py --fast
   ```
3. Run daily grid sensitivity checks:
   ```bash
   python market_risk/run_grid_checks.py --fast
   ```
4. Review reports in `market_risk/reports/` (JSON, CSV, HTML).

## Engineer
Goal: extend models/engines and keep workflows reproducible.

1. Run the test suite:
   ```bash
   pytest -q
   ```
2. Review architecture in `docs/spec.md` and report formats in `docs/reporting.md`.
3. Review the milestone plan in `docs/roadmap.md` for target deliverables.
4. Add new products/models by extending `pricing_engine/` and covering them in `tests/`.
5. Use `market_risk/run_grid_checks.py` to validate sensitivity behavior after changes.
