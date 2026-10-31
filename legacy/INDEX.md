# Legacy archive

Everything in this directory is **superseded code from the pre-rebuild codebase**. Nothing was
deleted; each file was copied here from git history or relocated from the working tree before being
replaced. The live implementation is in the project root.

## Why this exists

The internship brief required a full rebuild of the Nassau Candy Factory Reallocation project. The
pre-rebuild implementation shipped a different data model (it repaired the corrupt ship dates by
reconstructing them from the order year, producing lead times of 0–12 days), a different artifact
schema (`prepared_orders.csv`, `kpis.json`, `pilot_shortlist.csv`, `mode_slopes.csv`, …) and a
single-page tabbed dashboard. That regime is mutually exclusive with the brief's hard assertions
(Row 354 → 906 days, Row 4892 → 1273 days, Row 6141 → 1269 days), which only hold on the raw,
unrepaired distribution. Rather than discard the earlier work, it is archived here.

## Contents

| Path | What it is |
|---|---|
| `app.py` | The pre-rebuild single-page tabbed dashboard (Carto basemap, matplotlib factory map, per-product simulator). |
| `config.py` | Pre-rebuild configuration and the ZIP-centroid geocoding path. |
| `data_prep.py` | Pre-rebuild preparation, including ship-date reconstruction. |
| `features.py` | Pre-rebuild feature lists. |
| `models.py` | Pre-rebuild modelling with temporal validation and ablations. |
| `clustering.py` | Pre-rebuild lane clustering and road-kilometre ablation. |
| `simulation.py` | Pre-rebuild scenario engine with reference data frame and breakeven-speed calculation. |
| `run_pipeline.py` | Pre-rebuild pipeline and report writer. |
| `diagnostics/recommendation_count_py314.py` | Standalone diagnostic script from the pre-rebuild regime. |
| `tests/test_app.py`, `tests/test_pipeline.py`, `tests/test_report_formatting.py` | Tests tied to the old artifact schema (collected `prepared_orders.csv`, `kpis.json`, `pilot_shortlist.csv`, `mode_slopes.csv`, `breakeven_freight_speed_km_day`, …). Superseded by `scripts/smoke_test.py`. |
| `.github/workflows/tests.yml` | CI workflow for the superseded tests. |
| `README.md`, `executive_summary.md`, `research_paper.md` | Stale top-level copies; the current documents live in `docs/`. |

## Verification against the brief

The rebuild is verified by `scripts/smoke_test.py`, which asserts the Phase 8 checks: a boot with
zero exceptions, non-zero duplicate and outlier removal, the three lead-time assertions, 15 products
with no synthetic products, four live sidebar controls, four populated modules, four KPIs, a model
table with RMSE/MAE/R² and cross-validation for all three estimators, and recommendations with
≥90% coverage, ≥3 distinct Top-10 factories and a positive lead-time reduction in every Top-10 row.

Run it with `python -m pytest scripts/smoke_test.py -v` from the project root.