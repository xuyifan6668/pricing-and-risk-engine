# Reporting Artifacts

This project emits standardized JSON, CSV, and HTML reports for pricing and
market risk workflows. Reports share a common `meta` block that includes the
schema version, run id, as-of date, and generator metadata.

## Report inventory
- Pricing run:
  - Generator: `pricing_engine/run_pricing_report.py`
  - Outputs: `pricing_report_YYYYMMDD.json`, `.csv`, `.html`
- Mock IMA run:
  - Generator: `market_risk/run_mock.py`
  - Outputs: `ima_report_YYYYMMDD.json`, `.csv`, `.html`
- Grid approximation diagnostics:
  - Generator: `market_risk/run_grid_approx.py`
  - Outputs: `grid_approx_report_YYYYMMDD.json`, `.csv`, `.html`
- Grid checks:
  - Generator: `market_risk/run_grid_checks.py`
  - Outputs: `grid_check_YYYYMMDD.json`, `.html`
- Run logs:
  - Generator: `market_risk.reporting.RunLogger`
  - Outputs: `market_risk/reports/runs/YYYYMMDD/run_<uuid>.json`

## Common meta fields
- `schema_version`: shared schema version string.
- `report_type`: e.g., `pricing.run`, `risk.ima`, `risk.grid_approx`.
- `run_id`: report-level UUID.
- `asof`: as-of date of the report.
- `created_at`: UTC timestamp.
- `generator`: script path.
- `inputs`: key parameters used for the run.

## File naming
- Reports are written to `pricing_engine/reports/` and `market_risk/reports/`.
- Filenames are stamped by as-of date (`YYYYMMDD`).
