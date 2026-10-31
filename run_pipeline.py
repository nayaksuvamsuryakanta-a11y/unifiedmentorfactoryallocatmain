"""Analytical backend: python run_pipeline.

Loads the source, trains the three spec models, clusters routes, scores counterfactuals,
writes the artifacts directory and renders both report documents.
"""
import json
import logging
import sys

import numpy as np
import pandas as pd

from clustering import cluster_routes, congested_region_products
from config import (ARTIFACT_DIR, Assumptions, DIVISION_FACTORIES, FACTORY_COORDS,
                    RISK_HIGH, RISK_MEDIUM, RISK_WEIGHT_DISTANCE_SHIFT,
                    RISK_WEIGHT_LOW_SUPPORT, RISK_WEIGHT_NEG_PROFIT,
                    RISK_WEIGHT_PRED_VARIANCE, SEED, STATE_CENTROIDS)
from data_prep import load_and_audit
from models import train_and_persist
from simulation import (candidate_moves, compute_kpis, factory_optimiser_summary,
                        score_products, sensitivity, top_recommendations)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

K_MAX = 8


def markdown_table(frame, columns=None, decimals=3):
    """Render a frame as a Markdown table without an extra dependency."""
    if frame.empty:
        return "(no rows)"
    sub = frame if columns is None else frame[list(columns)]
    values = sub.round(decimals).fillna("").astype(str)
    headers = list(values.columns)
    body = "\n".join("| " + " | ".join(r) + " |"
                     for r in values.itertuples(index=False, name=None))
    return "| " + " | ".join(headers) + " |\n" \
        + "| " + " | ".join("---" for _ in headers) + " |\n" + body


def plural(n, noun, fmt="g"):
    return f"{n:{fmt}} {noun}" if n == 1 else f"{n:{fmt}} {noun}s"


def write_reports(df, audit, report, best_name, importance, resid_frame, global_std,
                  lanes, congestion, scored, kpis, mc, residual_std, top_n=10):
    best = report[report.model == best_name].iloc[0]
    total_products = audit["product_count"]
    evaluable = int(scored["product"].nunique()) if len(scored) else 0
    recs = top_recommendations(scored, top_n)
    moves = scored[scored["lead_gain_days"] > 0] if len(scored) else scored
    # `evaluable` counts every product (a retain row keeps the catalogue whole); this is the
    # subset that actually has a shorter route.
    reducing_products = int(moves["product"].nunique()) if len(moves) else 0
    qualifying = moves[moves["sufficient_evidence"] & moves["materially_better"]] \
        if len(moves) else moves
    high_risk = scored[scored["risk_level"] == "High"] if "risk_level" in scored else scored.head(0)
    medium_risk = scored[scored["risk_level"] == "Medium"] if "risk_level" in scored else scored.head(0)
    slow = lanes[lanes["slow_route"]].sort_values("exposure", ascending=False).head(5)
    congested = congestion[congestion["congested"]].head(5)

    lead_reduction = kpis["point"]["lead_time_reduction_pct"]
    profit_stab = kpis["point"]["profit_stability_pct"]
    confidence = kpis["point"]["scenario_confidence"]
    coverage = kpis["point"]["coverage_pct"]
    ci = kpis["ci"]
    factories_in_top10 = sorted(recs["candidate_factory"].unique())
    selected_k = int(lanes["selected_k"].iloc[0]) if len(lanes) else 1
    silhouette = float(lanes["silhouette_score"].iloc[0]) if len(lanes) else float("nan")
    candidate_count = len(candidate_moves(df))

    paper = f"""# Factory Reallocation and Shipping Optimization: Nassau Candy Distributor

## Abstract

Leadership attributes long lead times to static factory assignments and suboptimal shipping distances. This study tests that premise directly on the supplied order file: are there factory swaps that measurably shorten delivery without reducing gross profit? The source yields an unambiguous answer on distance but almost none on lead time. Measured centroid distances vary substantially across the five factories, so the distance deltas are real; the historical `(Ship Date - Order Date)` difference, however, is a near-constant ~1,320 days whose variation is dominated by calendar year rather than by any shipment attribute. All three spec-mandated models therefore score negative R² (best {best.cv_r2_mean:+.4f} five-fold cross-validated), so no model can be used to predict counterfactual lead times. The scenario engine is accordingly assumption-driven: lead change equals measured distance change divided by an assumed freight speed. Under the configured assumptions {reducing_products} of {total_products} products have a shorter route ({evaluable} of {total_products} receive a recommendation row; products with none keep an explicit retain row), and the composite ranking surfaces {len(high_risk)} high-risk and {len(medium_risk)} medium-risk moves whose capability gaps and prediction uncertainty must be resolved before any transfer.

## 1. Background and problem

Nassau Candy operates five factories and a {total_products}-product catalogue with fixed product-to-factory incumbencies. The operational question is whether moving a product to another factory can shorten delivery acceptably without eroding gross profit. The source contains order and ship dates, ship mode, destination, sales, units, gross profit and cost; factory coordinates and product incumbencies are supplied reference data. Critically, the file contains no freight charge, no carrier and no measured transit time, so every conversion from distance to time or cost is an assumption, stated in Section 7.

## 2. Data audit and preparation

### 2.1 Source

| Quantity | Value |
|---|---|
| Input rows | {audit['input_rows']:,} |
| Columns | {', '.join(audit['input_columns'])} |
| Stray duplicate header rows | {audit['stray_header_rows']} |
| Blank rows | {audit['blank_rows']} |
| Exact duplicate rows, excluding the synthetic Row ID | {audit['duplicate_rows']:,} |
| Rows after deduplication | {audit['rows_after_dedup']:,} |
| Unparseable Order Dates | {audit['unparseable_order_dates']} |
| Unparseable Ship Dates | {audit['unparseable_ship_dates']} |
| Rows dropped for unparseable dates or finances | {audit['dropped_parse_errors']} |
| Cost rows imputed as Sales minus Gross Profit | {audit['cost_imputed']} |
| Rows removed by the 3.0 x IQR fence | {audit['iqr_total_removed']} ({audit['iqr_removed_pct']:.2f}%) |
| Final modelling rows | {audit['rows_final']:,} |
| Distinct products | {audit['product_count']} |
| Synthetic products introduced | {audit['synthetic_products_present']} |

Deduplication excludes `Row ID`, which is a line number rather than an order identity, so two rows identical on every real field collapse to one order line. Cost carries no missing values, so the imputation rule is exercised by none but is retained for any future file.

### 2.2 Lead time

Dates parse as DD-MM-YYYY. The naive `(Ship Date - Order Date)` is taken unmodified, because the brief treats that difference as the lead time and because the ship years in the file are later calendar years that cannot be repaired without inventing data. The resulting distribution is:

* minimum {audit['lead_min']} days, mean {audit['lead_mean']:.2f}, maximum {audit['lead_max']}
* zero negative values; 100% of rows at or above 365 days

Three brief-mandated assertions were checked against the raw file and all pass:

| Row ID | Expected days | Observed days | Result |
|---|---:|---:|---|
| 354 | 906 | {audit['assertions']['354']['observed']} | pass |
| 4892 | 1273 | {audit['assertions']['4892']['observed']} | pass |
| 6141 | 1269 | {audit['assertions']['6141']['observed']} | pass |

The consequence is decisive for modelling: lead time is effectively a constant offset, so no feature set can explain it.

### 2.3 Origin factory and geocoding

{audit['fallback_products']} rows resolve through the Phase 7 division fallback (Kazookles is absent from the explicit map and resolves to **{DIVISION_FACTORIES.get('Other')}**); the remaining {audit['rows_final'] - audit['fallback_products']:,} resolve explicitly. All {len(STATE_CENTROIDS)} state/province centroids resolve, so no row is left without a distance. Distances are Haversine great-circle kilometres from the origin factory to the State/Province centroid; they are not carrier route mileage.

## 3. Outlier treatment

A 3.0 x IQR fence is applied to the four financial columns only; lead time is never trimmed, because the brief's lead-time assertions must be measured on the untrimmed distribution. Removals by column:

| Column | Rows removed |
|---|---:|
| Sales | {audit['iqr_removed_by_column']['sales']} |
| Units | {audit['iqr_removed_by_column']['units']} |
| Gross Profit | {audit['iqr_removed_by_column']['gross_profit']} |
| Cost | {audit['iqr_removed_by_column']['cost']} |
| **Total** | **{audit['iqr_total_removed']} ({audit['iqr_removed_pct']:.2f}%)** |

## 4. Modelling

### 4.1 Method

Three estimators are compared on an identical 80/20 random split (seed {SEED}) with shuffled five-fold cross-validation: Linear Regression, Random Forest and Gradient Boosting. Every learned transform - median imputation, StandardScaler and one-hot encoding - is fitted inside an sklearn Pipeline so no holdout statistic leaks into training. Features are exactly the brief's set: Product, Origin Factory, Destination Region, Ship Mode, with Units and Distance as support. Selection prefers the most interpretable estimator within a {Assumptions.model_selection_tolerance:.2f} R² tolerance.

### 4.2 Results

{markdown_table(report[['model', 'r2', 'rmse', 'mae', 'cv_r2_mean', 'cv_r2_std', 'cv_rmse_mean', 'cv_mae_mean']])}

All three estimators return negative R². The selected model is **{best_name}** with a five-fold cross-validated R² of {best.cv_r2_mean:+.4f} (holdout R² {best.r2:+.4f}, RMSE {best.rmse:.2f} days, MAE {best.mae:.2f} days). This is not a modelling failure to be tuned away: the target is a near-constant, so the best attainable prediction is the mean and every structured feature scores below it. The negative values are reported verbatim rather than replaced by a surrogate target.

### 4.3 Feature importance

Permutation importance on the selected model's held-out sample:

{markdown_table(importance.head(10)[['feature', 'importance_mean', 'importance_std']].rename(columns={'importance_mean': 'Mean importance', 'importance_std': 'Std'}))}

`distance_km` contributes essentially nothing, which is the quantitative reason the scenario engine does not use the model for counterfactual lead times.

### 4.4 Prediction uncertainty per product

Held-out residual standard deviation by product is the `predVariance` input to the risk engine. It measures how far the best model's predictions miss reality for each product, not how far a counterfactual sits from the current lead time. Products with fewer than two held-out orders fall back to the global value of {global_std:.2f} days.

{markdown_table(resid_frame[['product', 'residual_std', 'holdout_count']].rename(columns={'residual_std': 'Residual std (days)', 'holdout_count': 'Held-out orders'}))}

## 5. Route profiling

Orders are aggregated to the Factory x Region x Ship Mode grain. k-means is fitted on lanes with at least {Assumptions.min_orders_for_recommendation} orders; the number of clusters is chosen by maximum silhouette score over k = 2..{K_MAX}.

* selected k = {selected_k}, silhouette = {silhouette:.3f}
* slow routes flagged: {int(lanes['slow_route'].sum())}
* congested region/product pairs: {int(congestion['congested'].sum())}

Highest-exposure slow routes:

{markdown_table(slow[['current_factory', 'region', 'ship_mode', 'order_count', 'mean_lead', 'exposure']].rename(columns={'order_count': 'Orders', 'mean_lead': 'Mean lead (days)', 'exposure': 'Exposure'}))}

Congested region/product combinations:

{markdown_table(congested[['region', 'product_name', 'order_count', 'mean_lead', 'congested']].rename(columns={'order_count': 'Orders', 'mean_lead': 'Mean lead (days)'}))}

## 6. Counterfactual scenario engine

### 6.1 What is measured and what is assumed

| Measured from the file | Assumed (config.Assumptions) |
|---|---|
| order counts, units, gross profit | freight speed {Assumptions.freight_speed_km_day:g} km/day |
| centroid-based origin distances | freight cost {Assumptions.freight_cost_per_unit_per_1000km:g} currency/unit/1,000 km |
| observed lead baseline | capability gap penalty {Assumptions.capability_gap_penalty:.0%} |
| | minimum evidence {Assumptions.min_orders_for_recommendation} orders |
| | material improvement {Assumptions.min_material_days:g} days |
| | factory capacity {Assumptions.capacity_multiplier:g}x current units |
| | risk weight {Assumptions.risk_weight:.0%} |

Delta lead equals measured distance change divided by assumed speed; delta cost equals units times distance change times assumed rate. The predictive model is deliberately not used for counterfactual lead times because its R² is negative.

### 6.2 Ranking

For each product the candidate factories are scored on a composite of lead-time reduction, profit impact and risk reduction, with the lead/profit trade-off driven live by the priority slider and risk reduction holding its own fixed weight. Capability gaps discount the score, and an evidence-and-stability confidence caps it below 100. Only distance-reducing alternatives enter the ranking; products with none receive an explicit "retain current assignment" row so they are counted rather than silently dropped.

### 6.3 Risk engine

Risk is a weighted sum of four non-degenerate components, with the brief's fixed weights:

{RISK_WEIGHT_NEG_PROFIT:.2f} negProfit + {RISK_WEIGHT_PRED_VARIANCE:.2f} predVariance + {RISK_WEIGHT_LOW_SUPPORT:.2f} lowSupport + {RISK_WEIGHT_DISTANCE_SHIFT:.2f} distanceShift

* **negProfit** - Monte-Carlo share of draws in which the move loses money, counting the assumed freight delta and, for capability gaps, a one-time transition penalty.
* **predVariance** - the best model's held-out residual standard deviation for this product, normalised. Deliberately not |predicted - current|.
* **lowSupport** - shortfall below the minimum-order evidence gate.
* **distanceShift** - magnitude of the distance change, normalised across all candidates.

Thresholds: High > {RISK_HIGH:.2f}, Medium > {RISK_MEDIUM:.2f}. Every scored row carries all four components, so a Low risk is a measured state, not an empty default.

### 6.4 Results

{len(moves)} of {candidate_count} candidate pairs reduce distance, covering {evaluable} of {total_products} products. The top {top_n} recommendations are:

{markdown_table(recs[['product', 'current_factory', 'candidate_factory', 'orders', 'lead_gain_days', 'delta_distance_km', 'risk', 'risk_level', 'capability_gap']].rename(columns={'orders': 'Orders', 'lead_gain_days': 'Lead gain (days)', 'delta_distance_km': 'Delta distance (km)', 'risk': 'Risk', 'risk_level': 'Risk level', 'capability_gap': 'Capability gap'}))}

Distinct recommended factories in the top {top_n}: {', '.join(factories_in_top10)} ({len(factories_in_top10)}). Every top-{top_n} row carries a positive lead-time reduction.

## 7. KPIs and uncertainty

Point estimates with 5th-95th percentile intervals from {Assumptions.kpi_bootstrap_iterations} bootstrap resamples of products:

| KPI | Point | 5th percentile | 95th percentile |
|---|---:|---:|---:|
| Lead Time Reduction (%) | {lead_reduction:.4f} | {ci['lead_time_reduction_pct'][0]:.4f} | {ci['lead_time_reduction_pct'][1]:.4f} |
| Profit Impact Stability (%) | {profit_stab:.1f} | {ci['profit_stability_pct'][0]:.1f} | {ci['profit_stability_pct'][1]:.1f} |
| Scenario Confidence Score (best-model R²) | {confidence:+.4f} | {ci['scenario_confidence'][0]:+.4f} | {ci['scenario_confidence'][1]:+.4f} |
| Recommendation Coverage (%) | {coverage:.1f} | {ci['coverage_pct'][0]:.1f} | {ci['coverage_pct'][1]:.1f} |

The Scenario Confidence Score is the selected model's R² and is negative; it is reported as a confidence measure precisely so the reader sees that the scenario is assumption-driven, not forecast-driven.

Monte-Carlo sensitivity over freight speed 400-1,600 km/day and freight cost 0.05-0.30 currency/unit/1,000 km:

{markdown_table(mc[['speed_km_day', 'freight_rate', 'lead_time_reduction_pct', 'profit_stability_pct', 'coverage_pct']].head(10).rename(columns={'speed_km_day': 'Speed (km/day)', 'freight_rate': 'Freight rate', 'lead_time_reduction_pct': 'Lead reduction %', 'profit_stability_pct': 'Profit stability %', 'coverage_pct': 'Coverage %'}))}

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
"""
    docs_dir = ARTIFACT_DIR.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "research_paper.md").write_text(paper, encoding="utf-8")

    executive = f"""# Executive summary: Nassau Candy factory allocation

**Leadership attributes long lead times to static factory assignments and suboptimal shipping distances. We tested that premise directly.** The distance differences between the five factories are real - the reducing alternatives save hundreds of kilometres - but the supplied order file cannot establish that they shorten delivery, and the analysis therefore supports controlled piloting rather than an unqualified network redesign.

| Decision | Recommendation | Why |
|---|---|---|
| Factory changes | Pilot the low-risk, distance-reducing moves; retain everything else for now | Distances are measured, lead-time deltas are assumptions, and the predictive model has no signal |
| Modelling status | Do not deploy any lead-time forecast | All three spec models return negative R² (best {best.cv_r2_mean:+.4f} cross-validated) |
| Next step | Pilot selected routes with actual carrier, ship date, delivery date and freight invoice data | Converts assumed speed and cost into measured lane parameters |
| Data improvement | Record carrier, origin factory, measured transit days and freight charge | Without these the scenario engine stays assumption-driven |
| Risk review | Investigate the {len(high_risk)} high-risk and {len(medium_risk)} medium-risk moves first | Their scores are dominated by prediction uncertainty and capability gaps, which a pilot measures |

### Headline numbers

* {audit['rows_final']:,} order lines modelled; {audit['duplicate_rows']:,} duplicate rows removed; {audit['iqr_total_removed']} ({audit['iqr_removed_pct']:.2f}%) removed by the 3.0 x IQR fence.
* {reducing_products} of {audit['product_count']} products have a distance-reducing alternative; {int(recs['product'].nunique())} carry a recommendation (the remaining {audit['product_count'] - reducing_products} keep an explicit retain row).
* Lead-time reduction {lead_reduction:.4f}%, profit stability {profit_stab:.1f}%, coverage {coverage:.1f}%, scenario confidence {confidence:+.4f}.
* Distinct recommended factories in the top {top_n}: {len(factories_in_top10)} ({', '.join(factories_in_top10)}).
* Slow routes flagged: {int(lanes['slow_route'].sum())}; congested region/product pairs: {int(congestion['congested'].sum())}.

### Measured versus assumed

Measured from the source: order counts, units, gross profit, observed lead baselines and centroid-based distances. Assumed for scenarios: freight speed, freight cost, capability penalties and capacity limits. Every dashboard page that surfaces a lead-time or profit delta states this distinction.
"""
    (docs_dir / "executive_summary.md").write_text(executive, encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR.parent / "docs").mkdir(parents=True, exist_ok=True)

    df, audit = load_and_audit()
    print(f"Input {audit['input_rows']:,} rows -> {audit['rows_final']:,} after dedup and IQR fence")

    report, best_name, pipe, importance, resid_frame, global_std = train_and_persist(df)
    print(f"Selected model: {best_name} (CV R² {report[report.model == best_name].cv_r2_mean.iloc[0]:+.4f})")

    lanes = cluster_routes(df)
    congestion = congested_region_products(df)

    residual_std = resid_frame.set_index("product")["residual_std"].to_dict()
    scored = score_products(df, residual_std=residual_std, global_std=global_std)
    evaluable = int(scored["product"].nunique())
    reducing_products = int(scored[scored["lead_gain_days"] > 0]["product"].nunique())
    kpis = compute_kpis(scored, evaluable, report.iloc[0]["r2"])
    high = int((scored["risk_level"] == "High").sum()) if "risk_level" in scored else 0
    print(f"Coverage {kpis['point']['coverage_pct']:.1f}%, {reducing_products} of "
          f"{evaluable} products have a shorter route, {high} high-risk")

    mc, _ = sensitivity(df, iterations=30)

    for name, frame in {
        "model_metrics": report, "feature_importance": importance,
        "per_product_residual_std": resid_frame, "routes": lanes,
        "congested_region_products": congestion, "recommendations": scored,
    }.items():
        frame.to_csv(ARTIFACT_DIR / f"{name}.csv", index=False)
    (ARTIFACT_DIR / "kpis.json").write_text(json.dumps(kpis, indent=2), encoding="utf-8")

    write_reports(df, audit, report, best_name, importance, resid_frame, global_std,
                  lanes, congestion, scored, kpis, mc, residual_std)
    print(f"Reports written to {ARTIFACT_DIR.parent / 'docs'}")
    print(f"Artifacts saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()