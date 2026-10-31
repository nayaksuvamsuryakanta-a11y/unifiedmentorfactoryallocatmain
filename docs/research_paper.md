# Factory Reallocation and Shipping Optimization: Nassau Candy Distributor

## Abstract

Leadership attributes long lead times to static factory assignments and suboptimal shipping distances. This study tests that premise directly on the supplied order file: are there factory swaps that measurably shorten delivery without reducing gross profit? The source yields an unambiguous answer on distance but almost none on lead time. Measured centroid distances vary substantially across the five factories, so the distance deltas are real; the historical `(Ship Date - Order Date)` difference, however, is a near-constant ~1,320 days whose variation is dominated by calendar year rather than by any shipment attribute. All three spec-mandated models therefore score negative R² (best -0.0041 five-fold cross-validated), so no model can be used to predict counterfactual lead times. The scenario engine is accordingly assumption-driven: lead change equals measured distance change divided by an assumed freight speed. Under the configured assumptions 15 of 15 products have a shorter route, and the composite ranking surfaces 2 high-risk and 39 medium-risk moves whose capability gaps and prediction uncertainty must be resolved before any transfer.

## 1. Background and problem

Nassau Candy operates five factories and a 15-product catalogue with fixed product-to-factory incumbencies. The operational question is whether moving a product to another factory can shorten delivery acceptably without eroding gross profit. The source contains order and ship dates, ship mode, destination, sales, units, gross profit and cost; factory coordinates and product incumbencies are supplied reference data. Critically, the file contains no freight charge, no carrier and no measured transit time, so every conversion from distance to time or cost is an assumption, stated in Section 7.

## 2. Data audit and preparation

### 2.1 Source

| Quantity | Value |
|---|---|
| Input rows | 10,194 |
| Columns | Row ID, Order ID, Order Date, Ship Date, Ship Mode, Customer ID, Country/Region, City, State/Province, Postal Code, Division, Region, Product ID, Product Name, Sales, Units, Gross Profit, Cost |
| Stray duplicate header rows | 0 |
| Blank rows | 0 |
| Exact duplicate rows, excluding the synthetic Row ID | 313 |
| Rows after deduplication | 9,881 |
| Unparseable Order Dates | 0 |
| Unparseable Ship Dates | 0 |
| Rows dropped for unparseable dates or finances | 0 |
| Cost rows imputed as Sales minus Gross Profit | 0 |
| Rows removed by the 3.0 x IQR fence | 134 (1.36%) |
| Final modelling rows | 9,747 |
| Distinct products | 15 |
| Synthetic products introduced | False |

Deduplication excludes `Row ID`, which is a line number rather than an order identity, so two rows identical on every real field collapse to one order line. Cost carries no missing values, so the imputation rule is exercised by none but is retained for any future file.

### 2.2 Lead time

Dates parse as DD-MM-YYYY. The naive `(Ship Date - Order Date)` is taken unmodified, because the brief treats that difference as the lead time and because the ship years in the file are later calendar years that cannot be repaired without inventing data. The resulting distribution is:

* minimum 904 days, mean 1320.35, maximum 1642
* zero negative values; 100% of rows at or above 365 days

Three brief-mandated assertions were checked against the raw file and all pass:

| Row ID | Expected days | Observed days | Result |
|---|---:|---:|---|
| 354 | 906 | 906 | pass |
| 4892 | 1273 | 1273 | pass |
| 6141 | 1269 | 1269 | pass |

The consequence is decisive for modelling: lead time is effectively a constant offset, so no feature set can explain it.

### 2.3 Origin factory and geocoding

96 rows resolve through the Phase 7 division fallback (Kazookles is absent from the explicit map and resolves to **Secret Factory**); the remaining 9,651 resolve explicitly. All 61 state/province centroids resolve, so no row is left without a distance. Distances are Haversine great-circle kilometres from the origin factory to the State/Province centroid; they are not carrier route mileage.

## 3. Outlier treatment

A 3.0 x IQR fence is applied to the four financial columns only; lead time is never trimmed, because the brief's lead-time assertions must be measured on the untrimmed distribution. Removals by column:

| Column | Rows removed |
|---|---:|
| Sales | 72 |
| Units | 0 |
| Gross Profit | 60 |
| Cost | 122 |
| **Total** | **134 (1.36%)** |

## 4. Modelling

### 4.1 Method

Three estimators are compared on an identical 80/20 random split (seed 42) with shuffled five-fold cross-validation: Linear Regression, Random Forest and Gradient Boosting. Every learned transform - median imputation, StandardScaler and one-hot encoding - is fitted inside an sklearn Pipeline so no holdout statistic leaks into training. Features are exactly the brief's set: Product, Origin Factory, Destination Region, Ship Mode, with Units and Distance as support. Selection prefers the most interpretable estimator within a 0.02 R² tolerance.

### 4.2 Results

| model | r2 | rmse | mae | cv_r2_mean | cv_r2_std | cv_rmse_mean | cv_mae_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | -0.002 | 263.293 | 211.545 | -0.004 | 0.002 | 263.091 | 210.994 |
| Gradient Boosting | -0.001 | 263.163 | 211.505 | -0.008 | 0.004 | 263.58 | 211.146 |
| Random Forest | -0.11 | 277.098 | 224.371 | -0.104 | 0.013 | 275.9 | 223.718 |

All three estimators return negative R². The selected model is **Linear Regression** with a five-fold cross-validated R² of -0.0041 (holdout R² -0.0023, RMSE 263.29 days, MAE 211.54 days). This is not a modelling failure to be tuned away: the target is a near-constant, so the best attainable prediction is the mean and every structured feature scores below it. The negative values are reported verbatim rather than replaced by a surrogate target.

### 4.3 Feature importance

Permutation importance on the selected model's held-out sample:

| feature | Mean importance | Std |
| --- | --- | --- |
| product_name | 0.003 | 0.002 |
| ship_mode_rank | 0.001 | 0.0 |
| ship_mode | 0.001 | 0.001 |
| sales | 0.0 | 0.0 |
| region | 0.0 | 0.0 |
| current_factory | 0.0 | 0.002 |
| units | 0.0 | 0.001 |
| division | 0.0 | 0.001 |
| region_rank | 0.0 | 0.0 |
| distance_km | -0.0 | 0.001 |

`distance_km` contributes essentially nothing, which is the quantitative reason the scenario engine does not use the model for counterfactual lead times.

### 4.4 Prediction uncertainty per product

Held-out residual standard deviation by product is the `predVariance` input to the risk engine. It measures how far the best model's predictions miss reality for each product, not how far a counterfactual sits from the current lead time. Products with fewer than two held-out orders fall back to the global value of 263.00 days.

| product | Residual std (days) | Held-out orders |
| --- | --- | --- |
| Fun Dip | 4.896 | 2 |
| Hair Toffee | 263.003 | 1 |
| Kazookles | 286.291 | 9 |
| Laffy Taffy | 1.742 | 2 |
| Lickable Wallpaper | 207.201 | 3 |
| Nerds | 516.135 | 2 |
| Wonka Bar - Fudge Mallows | 266.49 | 355 |
| Wonka Bar - Milk Chocolate | 264.986 | 424 |
| Wonka Bar - Nutty Crunch Surprise | 256.191 | 339 |
| Wonka Bar - Scrumdiddlyumptious | 266.792 | 409 |
| Wonka Bar - Triple Dazzle Caramel | 255.391 | 385 |
| Wonka Gum | 336.608 | 19 |

## 5. Route profiling

Orders are aggregated to the Factory x Region x Ship Mode grain. k-means is fitted on lanes with at least 3 orders; the number of clusters is chosen by maximum silhouette score over k = 2..8.

* selected k = 7, silhouette = 0.266
* slow routes flagged: 43
* congested region/product pairs: 8

Highest-exposure slow routes:

| current_factory | region | ship_mode | Orders | Mean lead (days) | Exposure |
| --- | --- | --- | --- | --- | --- |
| Lot's O' Nuts | Pacific | Standard Class | 883 | 1312.705 | 267146.329 |
| Lot's O' Nuts | Atlantic | Standard Class | 807 | 1324.138 | 244959.987 |
| Lot's O' Nuts | Interior | Standard Class | 694 | 1319.014 | 209965.518 |
| Wicked Choccy's | Pacific | Standard Class | 686 | 1315.838 | 208231.166 |
| Wicked Choccy's | Atlantic | Standard Class | 596 | 1312.047 | 180316.208 |

Congested region/product combinations:

| region | product_name | Orders | Mean lead (days) | congested |
| --- | --- | --- | --- | --- |
| Pacific | Wonka Bar - Triple Dazzle Caramel | 570 | 1334.512 | True |
| Pacific | Wonka Bar - Scrumdiddlyumptious | 560 | 1327.79 | True |
| Pacific | Wonka Bar - Milk Chocolate | 552 | 1322.55 | True |
| Atlantic | Wonka Bar - Fudge Mallows | 455 | 1322.126 | True |
| Interior | Wonka Bar - Milk Chocolate | 425 | 1332.671 | True |

## 6. Counterfactual scenario engine

### 6.1 What is measured and what is assumed

| Measured from the file | Assumed (config.Assumptions) |
|---|---|
| order counts, units, gross profit | freight speed 900 km/day |
| centroid-based origin distances | freight cost 0.15 currency/unit/1,000 km |
| observed lead baseline | capability gap penalty 20% |
| | minimum evidence 3 orders |
| | material improvement 1 days |
| | factory capacity 1.5x current units |
| | risk weight 30% |

Delta lead equals measured distance change divided by assumed speed; delta cost equals units times distance change times assumed rate. The predictive model is deliberately not used for counterfactual lead times because its R² is negative.

### 6.2 Ranking

For each product the candidate factories are scored on a composite of lead-time reduction, profit impact and risk reduction, with the lead/profit trade-off driven live by the priority slider and risk reduction holding its own fixed weight. Capability gaps discount the score, and an evidence-and-stability confidence caps it below 100. Only distance-reducing alternatives enter the ranking; products with none receive an explicit "retain current assignment" row so they are counted rather than silently dropped.

### 6.3 Risk engine

Risk is a weighted sum of four non-degenerate components, with the brief's fixed weights:

0.40 negProfit + 0.30 predVariance + 0.20 lowSupport + 0.10 distanceShift

* **negProfit** - Monte-Carlo share of draws in which the move loses money, counting the assumed freight delta and, for capability gaps, a one-time transition penalty.
* **predVariance** - the best model's held-out residual standard deviation for this product, normalised. Deliberately not |predicted - current|.
* **lowSupport** - shortfall below the minimum-order evidence gate.
* **distanceShift** - magnitude of the distance change, normalised across all candidates.

Thresholds: High > 0.50, Medium > 0.25. Every scored row carries all four components, so a Low risk is a measured state, not an empty default.

### 6.4 Results

38 of 60 candidate pairs reduce distance, covering 15 of 15 products. The top 10 recommendations are:

| product | current_factory | candidate_factory | Orders | Lead gain (days) | Delta distance (km) | Risk | Risk level | Capability gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wonka Bar - Scrumdiddlyumptious | Lot's O' Nuts | Wicked Choccy's | 1702 | 0.29 | -261.332 | 0.168 | Low | False |
| Wonka Bar - Nutty Crunch Surprise | Lot's O' Nuts | Wicked Choccy's | 1524 | 0.287 | -258.411 | 0.161 | Low | False |
| Wonka Bar - Fudge Mallows | Lot's O' Nuts | Wicked Choccy's | 1527 | 0.256 | -230.588 | 0.166 | Low | False |
| Laffy Taffy | Sugar Shack | The Other Factory | 10 | 0.849 | -764.232 | 0.038 | Low | False |
| SweeTARTS | Sugar Shack | Secret Factory | 10 | 0.543 | -488.944 | 0.177 | Low | False |
| Fun Dip | Sugar Shack | Secret Factory | 3 | 0.748 | -673.626 | 0.036 | Low | False |
| Wonka Bar - Milk Chocolate | Wicked Choccy's | Secret Factory | 1768 | 0.426 | -383.422 | 0.382 | Medium | True |
| Wonka Bar - Triple Dazzle Caramel | Wicked Choccy's | Secret Factory | 1667 | 0.457 | -411.109 | 0.386 | Medium | True |
| Fizzy Lifting Drinks | Sugar Shack | Secret Factory | 6 | 0.618 | -556.649 | 0.18 | Low | False |
| Hair Toffee | The Other Factory | Secret Factory | 4 | 0.061 | -54.457 | 0.156 | Low | False |

Distinct recommended factories in the top 10: Secret Factory, The Other Factory, Wicked Choccy's (3). Every top-10 row carries a positive lead-time reduction.

## 7. KPIs and uncertainty

Point estimates with 5th-95th percentile intervals from 500 bootstrap resamples of products:

| KPI | Point | 5th percentile | 95th percentile |
|---|---:|---:|---:|
| Lead Time Reduction (%) | 0.0257 | 0.0203 | 0.0305 |
| Profit Impact Stability (%) | 86.7 | 73.3 | 100.0 |
| Scenario Confidence Score (best-model R²) | -0.0023 | -0.0023 | -0.0023 |
| Recommendation Coverage (%) | 100.0 | 53.3 | 100.0 |

The Scenario Confidence Score is the selected model's R² and is negative; it is reported as a confidence measure precisely so the reader sees that the scenario is assumption-driven, not forecast-driven.

Monte-Carlo sensitivity over freight speed 400-1,600 km/day and freight cost 0.05-0.30 currency/unit/1,000 km:

| Speed (km/day) | Freight rate | Lead reduction % | Profit stability % | Coverage % |
| --- | --- | --- | --- | --- |
| 1328.747 | 0.16 | 0.016 | 100.0 | 100.0 |
| 1236.842 | 0.074 | 0.018 | 100.0 | 100.0 |
| 1313.368 | 0.247 | 0.018 | 100.0 | 100.0 |
| 940.463 | 0.143 | 0.023 | 100.0 | 100.0 |
| 1172.638 | 0.256 | 0.019 | 100.0 | 100.0 |
| 672.686 | 0.189 | 0.035 | 100.0 | 100.0 |
| 1393.157 | 0.208 | 0.016 | 100.0 | 100.0 |
| 825.431 | 0.293 | 0.026 | 100.0 | 100.0 |
| 1334.06 | 0.099 | 0.016 | 100.0 | 100.0 |
| 452.565 | 0.089 | 0.048 | 100.0 | 100.0 |

## 8. Limitations

1. **The target is not transit time.** `(Ship Date - Order Date)` is ~1,320 days and absorbs order processing, scheduling and calendar delay. It is not a factory-to-customer transit clock.
2. **No model signal.** All three estimators return negative R², so counterfactual lead times cannot be predicted; they are computed from an assumed speed.
3. **Distances are great-circle centroids,** not carrier mileage, and ignore highway network, border crossings and port handling.
4. **Capability gaps are binary and unmeasured.** A candidate factory's ability to run a product is inferred from whether it has ever produced anything from the same division.
5. **Freight cost is a flat assumption** with no carrier contract, lane rate or weight break.
6. **Capacity is a scalar multiple** of historical units, with no labour, equipment or SKU-mix constraint.
7. **The assignment problem is not solved**; recommendations are per-product rankings, so capacity interactions across products are untested.

## 9. Conclusion

Distance differences between the five factories are real and large enough to matter - the reducing alternatives save hundreds of kilometres - but the supplied file cannot establish that they translate into shorter delivery. Lead time is a near-constant calendar quantity, the predictive models score below a mean predictor, and every lead-time delta in the scenario engine rests on an assumed freight speed. The analysis therefore supports measurement and controlled piloting, not an unqualified network redesign. The actionable output is the ranked list of distance-reducing alternatives with their risk decomposition: the high-risk moves are high because of prediction uncertainty and capability gaps, and those are exactly the quantities a pilot would measure.

**Verdict: retain current assignments for now; pilot the low-risk, distance-reducing moves with actual carrier, route and freight-invoice data, and only then retrain on measured transit times.**
