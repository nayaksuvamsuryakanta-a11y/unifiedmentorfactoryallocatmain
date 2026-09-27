# Factory Reallocation & Shipping Optimization: Nassau Candy Distributor

## Abstract

This decision-support study tests whether moving a product to another of five factories can plausibly shorten delivery without reducing gross profit. In the supplied sample of **10,045 retained order lines**, shipping mode explains most predictable lead-time variation. The distance ablation changes five-fold R² by only a small amount, and within-mode distance slopes are not statistically significant at the 5% level. The simulated moves remain assumption-driven and the configured evidence/materiality gates produce **0 actionable top-ranked products**. The analysis therefore supports investigation and measurement, not an unqualified network redesign.

## Background and problem

Nassau Candy has five fixed factories and a 15-product catalogue. The operational question is whether an alternative factory offers a faster, financially acceptable lane for a product. The source data contains order and ship dates, mode, destination, sales, units, gross profit, and cost, but no actual freight cost or carrier route. Factory coordinates and product incumbencies are supplied reference data.

## Data issues and preparation

The naive `(Ship Date − Order Date)` calculation is invalid: **904 / 1320.84 / 1642 days (min/mean/max)**, with years in ship dates later than the order years. The month/day values parse plausibly. We reconstruct ship year from the order year, roll to the next year if month/day precedes the order date, then subtract the observed minimum residual gap of **173 days**. This data-derived offset avoids guessing a constant. Repaired mode means are {'Same Day': 0.63, 'First Class': 2.84, 'Second Class': 3.87, 'Standard Class': 5.63}, ordered from Same Day through Standard Class; every repaired row is nonnegative and below 30 days.

Right-tail financial outliers use a **3× IQR fence**, wider than the textbook 1.5× fence to retain legitimate bulk orders. Lead time is not trimmed. Order-date calendar variables are derived. Ship mode is encoded ordinally and one-hot. Imputation, scaling, and categorical encoding are fitted within scikit-learn pipelines after each split.

## Geocoding method and limitation

ZIP/postal centroid values from **pgeocode 0.5.0 / GeoNames postal dataset** are checked into `data/reference/zip_centroids.csv` and read locally at runtime. ZIP-level coordinates were available for **95.6%** of retained rows; the remaining **4.4%** use the user-supplied state/province centroids. No runtime geocoding/network request is made. Distances are Haversine great-circle estimates from centroids, not carrier route mileage or measured transit distance. A separate state-only distance feature allows the diagnostics to compare both precisions.

## Modeling methods and results

We compare linear regression, Ridge, random forest, and gradient boosting using a fixed 80/20 random holdout and shuffled five-fold cross-validation. Temporal validation trains on the earlier order year and evaluates on the later year. The random-holdout best model was **Random Forest** (R² 0.657, RMSE 1.063 days, MAE 0.847 days; CV R² 0.663). Best temporal R² came from **Ridge** (R² 0.552). Temporal performance is weaker than the best random holdout, so random-split scores should not be treated as deployment forecasts.

Permutation importance is saved in `artifacts/permutation_importance.csv` and is based on the selected model’s held-out sample. All metrics and folds use the fixed seed.

### Core premise: does factory distance affect lead time?

The nested gradient-boosting ablation is stored in `artifacts/ablation.csv`. The measured ZIP/state-mixed and state-only comparisons are:

        | geocoding_distance | feature_set | cv_r2_mean | cv_r2_std |
| --- | --- | --- | --- |
| ZIP/state mixed | Ship mode | 0.6189 | 0.011 |
| ZIP/state mixed | + calendar | 0.6278 | 0.0104 |
| ZIP/state mixed | + order size | 0.6254 | 0.0112 |
| ZIP/state mixed | + distance | 0.6267 | 0.0113 |
| ZIP/state mixed | + region & factory | 0.6266 | 0.0109 |
| State-only | Ship mode | 0.6189 | 0.011 |
| State-only | + calendar | 0.6278 | 0.0104 |
| State-only | + order size | 0.6254 | 0.0112 |
| State-only | + distance | 0.6257 | 0.0109 |
| State-only | + region & factory | 0.6262 | 0.0107 |

The distance increment is small relative to the ship-mode baseline; switching geocoding precision changes the distance-only CV score by **0.0010 R²**. Within each ship mode, distance slopes/p-values are:

| ship_mode | slope_days_per_km | p_value | n |
| --- | --- | --- | --- |
| Same Day | 3.4e-05 | 0.10248 | 542 |
| First Class | 2.2e-05 | 0.288078 | 1526 |
| Standard Class | -4e-06 | 0.790046 | 6027 |
| Second Class | 6e-06 | 0.836401 | 1950 |

The ship-mode residual-by-distance-decile check is `artifacts/distance_residual_bins.csv`. State identity comparisons control for ship mode and appear in `artifacts/state_distance.csv`. These tests do not establish a reliable causal transit effect from centroid distance; distance-based lead changes in the scenario engine must therefore remain labeled assumptions.

## Route profiles and slow-lane flags

Orders are aggregated to factory/region/division lanes. Low-volume lanes (under 10 orders) are excluded from k-means fitting and assigned to the nearest fitted cluster. The selected k maximizes silhouette score; labels describe relative speed and volume. Problem routes require at least 10 orders and a slower-than-average typical or p90 lead time; exposure ranks excess lead time multiplied by lane order count. See `artifacts/routes.csv`.

## Counterfactual assumptions and optimization

Measured quantities are product-level order/units, observed gross profit and lead baseline, and Haversine distance deltas. **Assumed quantities** are freight speed (900 km/day), freight cost (0.15 currency/unit/1,000 km), capability gap penalty (20%), minimum evidence (10 orders), material lead improvement (1 days), and factory capacity (1.5× current units). Delta lead equals measured distance change divided by assumed speed; delta cost equals measured units times distance change times assumed rate. Candidate factories without historical production in the product division receive a visible capability-gap flag and score discount. Confidence combines evidence volume and lead-time stability, then discounts capability gaps and is capped below 100.

Per-product ranking min-max normalizes lead and profit gain across candidate factories, combines them with the speed/profit weight, and applies capability/confidence discounts. The joint mixed-integer assignment selects one factory per product subject to each factory’s unit capacity. **Solver used: PuLP/CBC.** Joint assignment differs from independent top picks for **6 products**; comparison rows are printed in the pipeline output.

## Actionable KPIs and uncertainty

KPIs are computed only for products meeting both the minimum-order and material-improvement tests. Order-level bootstrap resampling uses 500 iterations. Point estimates and 5th–95th percentile intervals:

```json
{
  "point": {
    "lead_time_reduction_pct": 0.0,
    "profit_stability_pct": 0.0,
    "confidence": 0.0,
    "coverage_pct": 0.0
  },
  "ci": {
    "lead_time_reduction_pct": [
      0.0,
      0.0
    ],
    "profit_stability_pct": [
      0.0,
      0.0
    ],
    "confidence": [
      0.0,
      0.0
    ],
    "coverage_pct": [
      0.0,
      0.0
    ]
  }
}
```

An empty actionable set correctly yields zero estimates and degenerate zero intervals; this is not evidence that every move is harmless, only that none cleared the configured gates.

## Monte Carlo sensitivity

The engine sampled **300** draws over freight speed 400–1,600 km/day and freight cost 0.05–0.30 currency/unit/1,000 km. Lead-time-reduction KPI mean was **7.79%**, median **0.00%**, and 5th–95th range **[0.00%, 30.09%]**. Average top-factory choice agreement with the configured baseline across products was **100.0%**. These measure different things: magnitude sensitivity versus ranking stability.

## Limitations and recommendations

Historical lead time reflects ship mode and observed order behavior, not isolated factory transit time. Destination centroids, missing shipping charges, capability gaps, and assumed speed/cost make scenario predictions exploratory. Temporal performance is worse than random validation. Before operational transfers, run controlled pilots with actual carrier, route, ship date, and freight invoice data. Until evidence and materiality gates are met, retain current assignments and collect those measurements.

## Recommendation table

| Current decision | Recommendation | Evidence |
|---|---|---|
| Retain current factory assignments | No move currently clears both evidence and material-improvement gates | No actionable products at configured thresholds |
