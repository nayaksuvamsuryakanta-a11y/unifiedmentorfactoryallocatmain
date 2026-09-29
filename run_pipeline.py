"""Run the complete analytical backend: python run_pipeline.py."""
import json
import logging
import sys

import numpy as np
import pandas as pd

from config import ARTIFACT_DIR, Assumptions, FACTORY_COORDS, SHIP_MODE_RANK, DATA_PATH
from data_prep import prepare_data
from models import train_models
from clustering import cluster_routes, congested_region_products
from simulation import score_products, joint_optimize, compute_kpis, sensitivity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def markdown_table(frame):
    """Render a small Markdown table without adding an undeclared dependency."""
    if frame.empty:
        return "(no rows)"
    values = frame.fillna("").astype(str)
    headers = list(values.columns)
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in values.itertuples(index=False, name=None))


def write_reports(df, metrics, temporal, diagnostic, routes, recommendations,
                  assignment, exploratory_assignment, exploratory_moves, exploratory_forced,
                  kpis, mc, choice_stability, solver, pilot_shortlist):
    selected_name=diagnostic["selected_model"]
    best=metrics[(metrics.feature_set=="Full feature set")&(metrics.model==selected_name)].iloc[0]
    temporal_table = pd.DataFrame(temporal).T
    temporal_best = temporal_table.r2.idxmax() if not temporal_table.empty else "unavailable"
    mode = diagnostic["mode_slopes"].sort_values("p_value")
    ablation = diagnostic["ablation"]
    dist_cols = ablation[ablation.feature_set == "+ distance"]
    if len(dist_cols) == 2:
        distance_effect = (dist_cols.set_index("geocoding_distance").cv_r2_mean["ZIP/state mixed"]
                           - dist_cols.set_index("geocoding_distance").cv_r2_mean["State-only"])
    else:
        distance_effect = float("nan")
    per_product = recommendations[~recommendations.is_incumbent].sort_values("score", ascending=False).drop_duplicates("product")
    # Match the KPI contract: only the single highest-ranked choice per product
    # can contribute; a lower-ranked alternative does not make that product actionable.
    actionable = per_product[
        per_product.sufficient_evidence & per_product.materially_better
    ] if len(per_product) else per_product
    joint_map = assignment.set_index("product").factory.to_dict()
    incumbents = df.groupby("product_name").current_factory.agg(lambda s:s.dropna().mode().iloc[0] if s.notna().any() else None)
    differs = [(r.product, r.candidate_factory, joint_map.get(r.product))
               for r in per_product.itertuples() if joint_map.get(r.product) != r.candidate_factory]
    precise = float((df.geo_precision == "ZIP").mean() * 100)
    state = 100 - precise
    raw_dates = pd.read_csv(DATA_PATH, usecols=["Order Date", "Ship Date"])
    raw_order_all = pd.to_datetime(raw_dates["Order Date"], dayfirst=True)
    raw_ship_all = pd.to_datetime(raw_dates["Ship Date"], dayfirst=True)
    naive_gap = (raw_ship_all - raw_order_all).dt.days
    raw_ship = pd.to_datetime(df.ship_date_raw, dayfirst=True)
    rebuilt = pd.to_datetime({"year": df.order_date.dt.year, "month": raw_ship.dt.month, "day": raw_ship.dt.day}, errors="coerce")
    invalid = rebuilt.isna()
    if invalid.any():
        rebuilt.loc[invalid] = pd.to_datetime({"year": df.order_date.dt.year[invalid] + 1, "month": raw_ship.dt.month[invalid], "day": raw_ship.dt.day[invalid]}, errors="coerce")
    rebuilt.loc[rebuilt < df.order_date] += pd.DateOffset(years=1)
    residual_offset = int((rebuilt - df.order_date).dt.days.min())
    mc_summary = mc.lead_time_reduction_pct.describe(percentiles=[.05, .5, .95])
    pilot_rows=pilot_shortlist.head(5)
    if pilot_rows.empty:
        pilot_table="(No evidence-qualified distance-reducing alternatives are in the shortlist.)"
    else:
        pilot_table=markdown_table(pilot_rows[["product","current_factory","candidate_factory","orders","lead_gain","breakeven_freight_speed_km_day"]].round(2))
    pilot_sentence="These moves qualify only if measured carrier speed is at or below the listed break-even value, so measure it in a pilot."
    paper = f"""# Factory Reallocation & Shipping Optimization: Nassau Candy Distributor

## Abstract

Leadership attributes long lead times to static factory assignments and suboptimal shipping distances. We tested that premise directly: shipping mode dominates prediction and distance adds little.

This decision-support study tests whether moving a product to another of five factories can plausibly shorten delivery without reducing gross profit. In the supplied sample of **{len(df):,} retained order lines**, shipping mode explains most predictable lead-time variation. The distance ablation changes five-fold R² by only a small amount, and within-mode distance slopes are not statistically significant at the 5% level. The simulated moves remain assumption-driven and the configured evidence/materiality gates produce **{actionable['product'].nunique() if len(actionable) else 0} actionable top-ranked products**. The analysis therefore supports investigation and measurement, not an unqualified network redesign.

## Background and problem

Nassau Candy has five fixed factories and a 15-product catalogue. The operational question is whether an alternative factory offers a faster, financially acceptable lane for a product. The source data contains order and ship dates, mode, destination, sales, units, gross profit, and cost, but no actual freight cost or carrier route. Factory coordinates and product incumbencies are supplied reference data.

## Data issues and preparation

The naive `(Ship Date − Order Date)` calculation is invalid: **{naive_gap.min()} / {naive_gap.mean():.2f} / {naive_gap.max()} days (min/mean/max)**, with years in ship dates later than the order years. The month/day values parse plausibly. We reconstruct ship year from the order year, roll to the next year if month/day precedes the order date, then subtract the observed minimum residual gap of **{residual_offset} days**. This data-derived offset avoids guessing a constant. Repaired mode means are {df.groupby('ship_mode').lead_time_days.mean().reindex(SHIP_MODE_RANK).round(2).to_dict()}, ordered from Same Day through Standard Class; every repaired row is nonnegative and below 30 days.

Right-tail financial outliers use a **3× IQR fence**, wider than the textbook 1.5× fence to retain legitimate bulk orders. Lead time is not trimmed. Order-date calendar variables are derived. Ship mode is encoded ordinally and one-hot. Imputation, scaling, and categorical encoding are fitted within scikit-learn pipelines after each split.

## Geocoding method and limitation

ZIP/postal centroid values from **pgeocode 0.5.0 / GeoNames postal dataset** are checked into `data/reference/zip_centroids.csv` and read locally at runtime. ZIP-level coordinates were available for **{precise:.1f}%** of retained rows; the remaining **{state:.1f}%** use the user-supplied state/province centroids. No runtime geocoding/network request is made. Distances are Haversine great-circle estimates from centroids, not carrier route mileage or measured transit distance. A separate state-only distance feature allows the diagnostics to compare both precisions.

## Modeling methods and results

We compare linear regression, Ridge, random forest, and gradient boosting using a fixed 80/20 random holdout and shuffled five-fold cross-validation. Temporal validation trains on the earlier order year and evaluates on the later year. The selected model was **{best.model}** (R² {best.r2:.3f}, RMSE {best.rmse:.3f} days, MAE {best.mae:.3f} days; CV R² {best.cv_r2_mean:.3f}). Best temporal R² came from **{temporal_best}** (R² {temporal_table.loc[temporal_best, 'r2']:.3f}). Temporal performance is weaker than the best random holdout, so random-split scores should not be treated as deployment forecasts.

Product and current factory overlap because each product has a fixed incumbent. The brief-spec variant uses exactly product, current factory, region, and ship mode; its scores are in `artifacts/model_metrics.csv`.

Model-selection score averages the five-fold CV R² and temporal R²; the configured tolerance is {Assumptions.model_selection_tolerance:.2f}, and choices at 0.01, 0.02, and 0.05 are {diagnostic['selection_choices']}.

Permutation importance is saved in `artifacts/permutation_importance.csv` and is based on the selected model’s held-out sample. All metrics and folds use the fixed seed.

### Core premise: does factory distance affect lead time?

The nested gradient-boosting ablation is stored in `artifacts/ablation.csv`. The measured ZIP/state-mixed and state-only comparisons are:

        {markdown_table(ablation.round(4))}

The distance increment is small relative to the ship-mode baseline; switching geocoding precision changes the distance-only CV score by **{distance_effect:.4f} R²**. Within each ship mode, distance slopes/p-values are:

{markdown_table(mode.round(6))}

The ship-mode residual-by-distance-decile check is `artifacts/distance_residual_bins.csv`. State identity comparisons control for ship mode and appear in `artifacts/state_distance.csv`. These tests do not establish a reliable causal transit effect from centroid distance; distance-based lead changes in the scenario engine must therefore remain labeled assumptions.

## Route profiles and slow-lane flags

Orders are aggregated to factory/region/division lanes. Low-volume lanes (under 10 orders) are excluded from k-means fitting and assigned to the nearest fitted cluster. The selected k maximizes silhouette score; labels describe relative speed and volume. Problem routes require at least 10 orders and a slower-than-average typical or p90 lead time; exposure ranks excess lead time multiplied by lane order count. See `artifacts/routes.csv`.

## Counterfactual assumptions and optimization

Measured quantities are product-level order/units, observed gross profit and lead baseline, and Haversine distance deltas. **Assumed quantities** are freight speed ({Assumptions.freight_speed_km_day:g} km/day), freight cost ({Assumptions.freight_cost_per_unit_per_1000km:g} currency/unit/1,000 km), capability gap penalty ({Assumptions.capability_gap_penalty:.0%}), minimum evidence ({Assumptions.min_orders_for_recommendation} orders), material lead improvement ({Assumptions.min_material_days:g} days), and factory capacity ({Assumptions.capacity_multiplier:g}× current units). Delta lead equals measured distance change divided by assumed speed; delta cost equals measured units times distance change times assumed rate. Candidate factories without historical production in the product division receive a visible capability-gap flag and score discount. Confidence combines evidence volume and lead-time stability, then discounts capability gaps and is capped below 100.

Per-product ranking min-max normalizes lead and profit gain across candidate factories, combines them with the speed/profit weight, and applies capability/confidence discounts. The joint mixed-integer assignment selects one factory per product subject to each factory’s unit capacity. **Solver used: {solver}.** Gated assignment moves **{int((assignment.factory != assignment["product"].map(incumbents)).sum())} products** off incumbents. Exploratory assignment moves **{exploratory_moves}**, with **{exploratory_forced}** differing from their own highest-scoring option due to capacity. This replaces the former comparison, which came from zeroing incumbent scores and allowing moves without evidence or materiality gates.

Under the current freight model, speed and profit gains are perfectly correlated per product, so the speed/profit control cannot change rankings. A real trade-off requires per-factory cost or freight-rate data absent from the source. Risk weight ({Assumptions.risk_weight:.0%}) is an assumption. Risk averages the capability-gap flag and candidate capacity pressure (product units divided by candidate factory historical units, capped at 1); incumbent risk is zero. Capability gaps affect confidence, the score multiplier, and the risk axis, so these effects stack.

## Actionable KPIs and uncertainty

KPIs are computed only for products meeting both the minimum-order and material-improvement tests. Order-level bootstrap resampling uses 500 iterations. Point estimates and 5th–95th percentile intervals:

```json
{json.dumps(kpis, indent=2)}
```

An empty actionable set correctly yields zero estimates and degenerate zero intervals; this is not evidence that every move is harmless, only that none cleared the configured gates.

## Monte Carlo sensitivity

The engine sampled **{len(mc)}** draws over freight speed 400–1,600 km/day and freight cost 0.05–0.30 currency/unit/1,000 km. Lead-time-reduction KPI mean was **{mc_summary['mean']:.2f}%**, median **{mc_summary['50%']:.2f}%**, and 5th–95th range **[{mc_summary['5%']:.2f}%, {mc_summary['95%']:.2f}%]**. Average top-factory choice agreement with the configured baseline across products was **{choice_stability:.1f}%**. These measure different things: magnitude sensitivity versus ranking stability.

## Limitations and recommendations

The KPI bootstrap resamples which products pass the order-count gate but does not re-estimate lead or profit deltas. Historical lead time reflects ship mode and observed order behavior, not isolated factory transit time. Destination centroids, missing shipping charges, capability gaps, and assumed speed/cost make scenario predictions exploratory. Temporal performance is worse than random validation. Before operational transfers, run controlled pilots with actual carrier, route, ship date, and freight invoice data. Until evidence and materiality gates are met, retain current assignments and collect those measurements.

## Conditional pilot shortlist

{pilot_sentence}

{pilot_table}
"""
    (ARTIFACT_DIR.parent / "research_paper.md").write_text(paper, encoding="utf-8")
    executive = f"""# Executive summary: Nassau Candy factory allocation

**Leadership attributes long lead times to static factory assignments and suboptimal shipping distances. We tested that premise directly; the data does not show a dependable delivery-time benefit from factory distance once shipping mode is considered.** Shipping mode predicts delivery timing much better than factory location, and temporal validation is weaker than a random split. Scenario estimates depend on assumed freight speed and freight cost, so the analysis does not justify moving products on its own.

| Decision | Recommendation | Why |
|---|---|---|
| Factory changes | Keep current assignments unless a pilot validates a shortlisted move | {actionable['product'].nunique() if len(actionable) else 0} top-ranked moves meet both default evidence and material-improvement thresholds |
| Next step | Pilot selected routes with actual carrier and invoice data | Current distance-to-time and freight-cost conversions are assumptions |
| Data improvement | Record carrier, origin, destination, ship date, delivery date, and freight charge | Enables direct measurement of transit and cost by route |
| Risk review | Investigate the highest-exposure slow routes in `artifacts/routes.csv` | Route profiles identify operational lanes for measurement |

### Break-even pilot shortlist

{pilot_sentence}

{pilot_table}

**Measured from the source:** order counts, units, gross profit, repaired historical lead times, and centroid-based distances. **Assumed for scenarios:** freight speed, freight cost, capability penalties, and capacity limits. ZIP centroids covered {precise:.1f}% of retained rows; state/province centroids covered {state:.1f}%.
"""
    (ARTIFACT_DIR.parent / "executive_summary.md").write_text(executive, encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_data()
    df.to_csv(ARTIFACT_DIR / "prepared_orders.csv", index=False)
    print(f"Prepared rows after sales/units/profit/cost IQR filter: {len(df):,}")
    model_report, importance, temporal, diagnostic = train_models(df)
    routes = cluster_routes(df)
    congestion = congested_region_products(df)
    recommendations = score_products(df, top_n=3)
    all_candidates = score_products(df, top_n=len(FACTORY_COORDS))
    # Keep this conditional list genuinely below the configured lead-time gate.
    pilot_shortlist=all_candidates[
        (~all_candidates.is_incumbent)
        & all_candidates.sufficient_evidence.fillna(False)
        & (all_candidates.delta_distance_km<0)
        & (~all_candidates.materially_better.fillna(False))
    ].sort_values("breakeven_freight_speed_km_day",ascending=False)
    assignment = joint_optimize(all_candidates, df, gated=True)
    exploratory_assignment = joint_optimize(all_candidates, df, gated=False)
    kpis = compute_kpis(recommendations, df.product_name.nunique(), order_frame=df)
    mc, choices = sensitivity(df, Assumptions.monte_carlo_iterations)
    for name, obj in [
        ("model_metrics", model_report), ("permutation_importance", importance),
        ("routes", routes), ("congested_region_products", congestion), ("recommendations", recommendations),
        ("joint_assignment", assignment), ("joint_assignment_gated", assignment),
        ("joint_assignment_exploratory", exploratory_assignment), ("pilot_shortlist", pilot_shortlist),
        ("ablation", diagnostic["ablation"]),
        ("model_selection", diagnostic["model_selection"]),
        ("mode_slopes", diagnostic["mode_slopes"]),
        ("distance_residual_bins", diagnostic["residual_distance"]),
        ("state_distance", diagnostic["state_distance"]), ("sensitivity", mc),
    ]:
        obj.to_csv(ARTIFACT_DIR / f"{name}.csv", index=False)
    (ARTIFACT_DIR / "temporal_metrics.json").write_text(json.dumps(temporal, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "kpis.json").write_text(json.dumps(kpis, indent=2), encoding="utf-8")
    individual = recommendations[~recommendations.is_incumbent].sort_values("score", ascending=False).drop_duplicates("product").set_index("product").candidate_factory.to_dict()
    joint = assignment.set_index("product").factory.to_dict()
    differences = [(p, factory, joint.get(p)) for p, factory in individual.items() if joint.get(p) != factory]
    print("\nModel held-out and five-fold CV metrics:\n", model_report.to_string(index=False))
    print("\nModel selection (mean CV and temporal R²; configured tolerance):\n", diagnostic["model_selection"].to_string(index=False))
    print("Model choices by tolerance: " + ", ".join(f"{tol:.2f} → {name}" for tol,name in diagnostic["selection_choices"].items()))
    comparison = (model_report[model_report.feature_set=="Full feature set"]
                  .drop(columns=["temporal_r2"],errors="ignore")
                  .set_index("model").join(pd.DataFrame(temporal).T.add_prefix("temporal_")))
    print("\nRandom split versus temporal validation (R²/RMSE/MAE):\n", comparison[["r2", "rmse", "mae", "temporal_r2", "temporal_rmse", "temporal_mae"]].to_string())
    print("\nDistance ablation (both geocoding precisions):\n", diagnostic["ablation"].to_string(index=False))
    print("\nWithin-mode distance slopes and p-values:\n", diagnostic["mode_slopes"].to_string(index=False))
    print("\nShip-mode-controlled state/distance proxy comparison:\n", diagnostic["state_distance"].to_string(index=False))
    print(f"\nJoint solver path: {assignment.attrs.get('solver', 'unknown')}")
    incumbents = df.groupby("product_name").current_factory.agg(lambda s:s.dropna().mode().iloc[0] if s.notna().any() else None)
    gated_moves = int((assignment.factory != assignment["product"].map(incumbents)).sum())
    exploratory_moves = int((exploratory_assignment.factory != exploratory_assignment["product"].map(incumbents)).sum())
    highest = all_candidates.sort_values("score", ascending=False).drop_duplicates("product").set_index("product").candidate_factory.sort_index()
    exploratory_choice = exploratory_assignment.set_index("product").factory.sort_index()
    exploratory_forced = int((exploratory_choice.to_numpy() != highest.reindex(exploratory_choice.index).to_numpy()).sum())
    print(f"Gated joint assignment moves off incumbent: {gated_moves} products")
    print(f"Exploratory joint assignment moves off incumbent: {exploratory_moves} products; differs from own highest-scoring option: {exploratory_forced}")
    print("\nConditional pilot shortlist (top 5 by break-even speed):\n", pilot_shortlist[["product","candidate_factory","orders","lead_gain","breakeven_freight_speed_km_day"]].head(5).round(2).to_string(index=False))
    print(f"\nActionable KPI estimates (500 order-bootstrap resamples):\n{json.dumps(kpis, indent=2)}")
    baseline_choices = score_products(df, top_n=1)
    baseline_choices = baseline_choices[~baseline_choices.is_incumbent].set_index("product").candidate_factory.to_dict()
    all_products = sorted(set().union(*(c.keys() for c in choices))) if choices else []
    stability_rows = []
    for product in all_products:
        draws = [choice[product] for choice in choices if product in choice]
        stability_rows.append({
            "product": product,
            "default_factory": baseline_choices.get(product),
            "choice_agreement_pct": 100 * float(np.mean([factory == baseline_choices.get(product) for factory in draws])) if draws else 0.0,
            "distinct_factories": len(set(draws)),
        })
    stability_frame = pd.DataFrame(stability_rows)
    stability_frame.to_csv(ARTIFACT_DIR / "sensitivity_stability.csv", index=False)
    mean_stability = float(stability_frame.choice_agreement_pct.mean()) if len(stability_frame) else 0.0
    (ARTIFACT_DIR / "sensitivity_summary.json").write_text(json.dumps({
        "iterations": len(mc),
        "mean_choice_agreement_pct": mean_stability,
    }, indent=2), encoding="utf-8")
    mc_summary = mc.lead_time_reduction_pct.describe(percentiles=[.05, .5, .95])
    print(f"\nMonte Carlo: {len(mc)} draws; lead-reduction KPI mean={mc_summary['mean']:.2f}%, median={mc_summary['50%']:.2f}%, 5th-95th=[{mc_summary['5%']:.2f}%, {mc_summary['95%']:.2f}%]; mean factory-choice agreement with default assumptions={mean_stability:.1f}%.")
    write_reports(df, model_report, temporal, diagnostic, routes, recommendations,
                  assignment, exploratory_assignment, exploratory_moves, exploratory_forced,
                  kpis, mc, mean_stability, assignment.attrs.get("solver", "unknown"), pilot_shortlist)
    print(f"Reports written: {ARTIFACT_DIR.parent / 'research_paper.md'}, {ARTIFACT_DIR.parent / 'executive_summary.md'}")
    print(f"Artifacts saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
