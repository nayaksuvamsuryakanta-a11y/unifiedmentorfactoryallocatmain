# Fix register

Log of defects proven by a failing test, with the fix actually applied. No fix was made
speculatively; every entry below began as a red test.

| # | File | Defect proven by | Symptom | Fix applied | Green tier |
|---|------|------------------|---------|-------------|-----------|
| F-01 | `simulation.py` `candidate_moves` | `test_slider_flips_top_candidate` (T4) | `delta_lead` and `delta_cost` were both pure linear functions of `delta_dist`, so min-max normalisation collapsed `profit_gain` onto `lead_gain_days` and the priority slider never changed the ranking (`corr(delta_lead, delta_cost)` was indistinguishable from 1). | Compute the one-time qualification outlay as a *net* (destination minus incumbent) term inside the per-factory loop, so `delta_cost` varies across candidates independently of `delta_distance`. | T4 |
| F-02 | `data_prep.py` `_calendar_days` (new) | `test_tz_matrix_survives_four_zones` (T1) | `America/New_York` Row 6141 reported 1268 days instead of the anchored 1269; `Australia/Sydney` Row 4892 reported 1272 instead of 1273. Normalising each endpoint to *local* midnight leaves a residual UTC offset, so DST transitions steal or gain days across the subtraction. | New `_calendar_days()` helper: when the columns carry an offset, collapse both endpoints to UTC midnight before differencing. A no-op on the naive source. | T1 |
| F-03 | `data_prep.py` `load_and_audit` | `test_fuzz_200_malformed_rows` (T1) | Cost imputation (`sales - gross_profit`) ran **before** `pd.to_numeric` coercion, so a non-numeric financial field made the subtraction raise `TypeError: operation 'sub' not supported for dtype 'str'`. The exception escaped `load_and_audit_safe`, which only catches `ValueError`/`FileNotFoundError`/`EmptyDataError`/`ParserError`, so a malformed file crashed the load instead of populating the error banner. | Coerce `NUMERIC` columns first, then impute. Every injected defect class is now reported in the audit counts with no unhandled exception. | T1 |

## Anchors that belong to `/legacy` (deliberately NOT fixed)

Three Phase-0 anchor values in the v3 brief describe the superseded implementation and do not
match the rebuilt pipeline. They are recorded here so the discrepancy is visible rather than
silent.

| Anchor | Brief states | Live value | Resolution |
|--------|-------------|------------|------------|
| Model set | LR + Ridge + RF + GB | LR + RF + GB only | Ridge lived in `legacy/models.py`; the rebuild dropped it. |
| Selection rule | mean of CV R² and temporal R² | `cv_r2_mean` only | Temporal R² was a Ridge-era metric; the rebuild selects on CV R² within tolerance. |
| ZIP-centroid coverage | 95.6% ZIP / 4.4% state fallback | 61/61 state centroids | `legacy/data_prep.py` `resolve_customer_centroid` used hardcoded ZIPs; the rebuild uses state centroids for every row. |

See `docs/phase0_audit.md` and `docs/phase8_verification.md` for the live observed values.