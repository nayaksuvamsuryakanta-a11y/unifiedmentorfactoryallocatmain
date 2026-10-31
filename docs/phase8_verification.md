# Phase 8 verification log

Run against the rebuilt tree after the final `python run_pipeline.py`. Every item is the
brief's acceptance letter, the assertion that was checked, and the value actually observed.

## a) App boots without exception

| Check | Assertion | Observed |
|---|---|---|
| `streamlit run app.py` from project root after venv activation, no `ModuleNotFoundError` / `sys.path` failure | `AppTest` completes with an empty exception list | `test_app_boots_without_exception PASSED` in 10.43 s |
| 9 pages registered with distinct routes | `st.navigation` accepts the page set | 9 pages, each with a unique `url_path` from `_page_slug` |
| Sidebar widget inventory | 1 selectbox, 2 multiselects, 1 slider | Confirmed by widget-count assertions |

`AppTest.exception` is an empty `ElementList()` - no uncaught execution. The app is
launched by `st.navigation`, so `streamlit run app.py` is the only command required.

## b) Row counts, duplicates and outliers

| Check | Assertion | Observed |
|---|---|---|
| Input rows ≈ 10,155 | 9,500 ≤ rows ≤ 10,500 | **10,194** ✓ |
| Duplicates removed | > 0 | **313** ✓ |
| Outliers removed | > 0 | **134 (1.356%)** ✓ |
| Lead time never trimmed | no row removed for lead time; no negative lead time | min 904, max 1642, 0 negative; IQR removals by column - Sales 72, Units 0, Gross Profit 60, Cost 122 |
| Final modelling rows | equals dedup minus IQR | **9,747** (9,881 - 134) ✓ |

Deduplication excludes `Row ID`, which is a line number rather than an order identity. The
IQR fence is applied to the four financial columns only.

## c) Hard assertions

| Row ID | Expected days | Observed days | Result |
|---|---:|---:|---|
| 354 | 906 | 906 | **pass** |
| 4892 | 1273 | 1273 | **pass** |
| 6141 | 1269 | 1269 | **pass** |

Measured on the untrimmed distribution, because the brief's assertions must survive the IQR
step. All three pass.

## d) Product count and no synthetic data

| Check | Assertion | Observed |
|---|---|---|
| Product count = 15 | exactly 15 distinct products | **15** ✓ |
| No synthetic products | no `Product_*` rows introduced | **False** ✓ |
| All 15 products traceable to the source | every modelled product name appears in the raw CSV | all 15 present, normalised via `normalize_product_name` |
| Kazookles handled by the Phase 7 fallback, not the explicit map | absent from `PRODUCT_FACTORY` | resolves to **Secret Factory** via the Sugar division; 96 rows use the fallback |

The 15 products are: Everlasting Gobstopper, Fizzy Lifting Drinks, Fun Dip, Hair Toffee,
Kazookles, Laffy Taffy, Lickable Wallpaper, Nerds, SweeTARTS, Wonka Bar - Fudge Mallows,
Wonka Bar - Milk Chocolate, Wonka Bar - Nutty Crunch Surprise, Wonka Bar - Scrumdiddlyumptious,
Wonka Bar - Triple Dazzle Caramel, Wonka Gum.

## e) All four sidebar controls change outputs live

| Control | Type | Assertion | Observed |
|---|---|---|---|
| Product | selectbox | selecting a product filters the recommendation table to that product | changes the rendered rows |
| Region | multiselect | restricting regions recomputes the scored frame against the filtered orders | changes the rendered rows |
| Ship Mode | multiselect | restricting modes recomputes the scored frame against the filtered orders | changes the rendered rows |
| Optimization priority | slider (0.25 / 0.50 / 0.75) | the ranking changes with the weight | **ranking changes** - see §2.4 of `phase0_audit.md` |

The last item was the genuine defect: before the fix the slider was algebraically degenerate.
After adding `Assumptions.setup_freight_cost`, Laffy Taffy's top candidate is The Other Factory
at 0.25 and 0.50 but **Secret Factory** at 0.75, and all 15 products break the
profit_gain/lead_gain collinearity. `test_sidebar_controls_rerender PASSED`.

## f) All four modules render real data; Risk panel populated

| Check | Assertion | Observed |
|---|---|---|
| Module 1 Factory Optimization Simulator | renders real factory portfolio rows | PASSED |
| Module 2 What-If | renders real product/factory counterfactuals | PASSED |
| Module 3 Recommendation Dashboard | renders real recommendations, CSV export enabled | PASSED |
| Module 4 Risk & Impact Panel | renders real risk rows | PASSED |
| Risk panel non-degenerate | all 4 components present on every row; all 3 levels appear; no component is a constant zero | **60 rows, all 4 components present; High 2 / Medium 39 / Low 19** |

The risk engine is `0.40·negProfit + 0.30·predVariance + 0.20·lowSupport + 0.10·distanceShift`
with thresholds High > 0.50, Medium > 0.25. `predVariance` is the best model's held-out
residual standard deviation per product, not `|predicted - current|`.

## g) KPI header shows all four

| KPI | Point | 5th percentile | 95th percentile |
|---|---:|---:|---:|
| Lead Time Reduction (%) | **0.0257** | 0.0203 | 0.0305 |
| Profit Impact Stability (%) | **73.33** | 53.33 | 93.33 |
| Scenario Confidence Score (best-model R²) | **-0.0023** | -0.0023 | -0.0023 |
| Recommendation Coverage (%) | **100.0** | 53.3 | 100.0 |

Intervals come from 500 bootstrap resamples of products (one recommendation each), so a
single high-volume product cannot dominate. `scenario_confidence` is the selected model's R²
and is negative - it is exposed as a confidence measure precisely so the reader sees that the
scenario is assumption-driven, not forecast-driven.

## h) Model table, recommendations and coverage

### h1 - Model table (all 3 models, all 8 fields, with CV)

| model | r2 | rmse | mae | cv_r2_mean | cv_r2_std | cv_rmse_mean | cv_mae_mean | interpretability_tier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear Regression | -0.002314 | 263.293 | 211.545 | **-0.004146** | 0.002224 | 263.091 | 210.994 | 0 |
| Gradient Boosting | -0.001325 | 263.163 | 211.505 | -0.007906 | 0.003622 | 263.580 | 211.146 | 2 |
| Random Forest | -0.110180 | 277.098 | 224.371 | -0.104307 | 0.012856 | 275.900 | 223.718 | 1 |

**Best model justified:** Linear Regression is selected because it is the most interpretable
estimator (tier 0) within the 0.02 R² tolerance of the best CV score. Gradient Boosting has
the least-negative holdout R² (-0.0013) but its CV R² (-0.0079) is worse than Linear
Regression's (-0.0041), and it sits two interpretability tiers higher. All three are negative
because the target is a near-constant; the values are reported verbatim.

### h2 - Recommendations

| Check | Assertion | Observed |
|---|---|---|
| Coverage ≥ 90% | products receiving a recommendation row / catalogue | **15/15 = 100.0%** ✓ |
| ≥ 3 distinct recommended factories in Top-10 | distinct `candidate_factory` values | **3** - Secret Factory, The Other Factory, Wicked Choccy's ✓ |
| Every Top-10 row has positive lead-time reduction | `lead_gain_days > 0` and `delta_distance_km < 0` | **all 10 rows** ✓ |

Top 10:

| product | current_factory | candidate_factory | Orders | Lead gain (d) | Delta distance (km) | Risk | Level |
|---|---|---|---:|---:|---:|---:|---|
| Wonka Bar - Scrumdiddlyumptious | Lot's O' Nuts | Wicked Choccy's | 1702 | 0.290 | -261.332 | 0.168 | Low |
| Wonka Bar - Nutty Crunch Surprise | Lot's O' Nuts | Wicked Choccy's | 1524 | 0.287 | -258.411 | 0.161 | Low |
| Wonka Bar - Fudge Mallows | Lot's O' Nuts | Wicked Choccy's | 1527 | 0.256 | -230.588 | 0.166 | Low |
| Laffy Taffy | Sugar Shack | The Other Factory | 10 | 0.849 | -764.232 | 0.038 | Low |
| SweeTARTS | Sugar Shack | Secret Factory | 10 | 0.543 | -488.944 | 0.177 | Low |
| Fun Dip | Sugar Shack | Secret Factory | 3 | 0.748 | -673.626 | 0.036 | Low |
| Wonka Bar - Milk Chocolate | Wicked Choccy's | Secret Factory | 1768 | 0.426 | -383.422 | 0.382 | Medium |
| Wonka Bar - Triple Dazzle Caramel | Wicked Choccy's | Secret Factory | 1667 | 0.457 | -411.109 | 0.386 | Medium |
| Fizzy Lifting Drinks | Sugar Shack | Secret Factory | 6 | 0.618 | -556.649 | 0.180 | Low |
| Hair Toffee | The Other Factory | Secret Factory | 4 | 0.061 | -54.457 | 0.156 | Low |

Lickable Wallpaper and Wonka Gum have no distance-reducing alternative and receive explicit
retain rows, which is why coverage is 15/15 while only 13 products have a shorter route.

### h3 - `scripts/smoke_test.py` passes

`26 passed` in 18.60 s, one PytestCacheWarning about an existing cache directory (harmless).
The suite covers items a-h above plus documentation, pinning, legacy preservation, source
provenance and argparse safety.

## Supporting artefacts

| Artefact | Content |
|---|---|
| `artifacts/kpis.json` | the four KPIs with bootstrap intervals, written by the pipeline |
| `artifacts/model_metrics.csv` | all 3 models x 8 fields + interpretability tier |
| `artifacts/recommendations.csv` | 60 scored candidate pairs with all 4 risk components |
| `artifacts/per_product_residual_std.csv` | `predVariance` per product (global fallback 263.00) |
| `artifacts/routes.csv` | 7-cluster lane profile, slow-route exposure |
| `artifacts/congested_region_products.csv` | 8 congested pairs |
| `artifacts/best_model.joblib` | persisted pipeline, loaded under `st.cache_resource` |
| `docs/research_paper.md` | full paper including the verbatim verdict |
| `docs/executive_summary.md` | leadership summary with headline numbers |
| `docs/phase0_audit.md` | this brief's Phase 0 audit table and migration plan |
| `docs/phase8_verification.md` | this log |

## Verbatim conclusion (from `docs/research_paper.md`, Section 9)

> Distance differences between the five factories are real and large enough to matter - the
> reducing alternatives save hundreds of kilometres - but the supplied file cannot establish
> that they translate into shorter delivery. Lead time is a near-constant calendar quantity,
> the predictive models score below a mean predictor, and every lead-time delta in the
> scenario engine rests on an assumed freight speed. The analysis therefore supports
> measurement and controlled piloting, not an unqualified network redesign. The actionable
> output is the ranked list of distance-reducing alternatives with their risk decomposition:
> the high-risk moves are high because of prediction uncertainty and capability gaps, and
> those are exactly the quantities a pilot would measure.
>
> **Verdict: retain current assignments for now; pilot the low-risk, distance-reducing moves
> with actual carrier, route and freight-invoice data, and only then retrain on measured
> transit times.**