"""Assumption-driven scenario engine.

Measured quantities (order counts, units, gross profit, centroid distances) come from
the file. Everything that converts a distance delta into a lead-time or cost delta is an
*assumption* stored in config.Assumptions and printed on every dashboard page. The
predictive model is deliberately not used for counterfactual lead times: its R² is
negative, so it carries no usable signal for out-of-sample factory swaps.
"""
import json
import logging

import numpy as np
import pandas as pd

from config import (FACTORY_COORDS, RISK_HIGH, RISK_MEDIUM, RISK_WEIGHT_DISTANCE_SHIFT,
                    RISK_WEIGHT_LOW_SUPPORT, RISK_WEIGHT_NEG_PROFIT,
                    RISK_WEIGHT_PRED_VARIANCE, SEED, Assumptions)

log = logging.getLogger(__name__)

RISK_COMPONENTS = {
    "negProfit": RISK_WEIGHT_NEG_PROFIT,
    "predVariance": RISK_WEIGHT_PRED_VARIANCE,
    "lowSupport": RISK_WEIGHT_LOW_SUPPORT,
    "distanceShift": RISK_WEIGHT_DISTANCE_SHIFT,
}


def parse_components(cell):
    """Recover the four risk components from a stored JSON cell."""
    return json.loads(cell) if isinstance(cell, str) else {}


def _norm(series):
    """Min-max normalise, returning 0.5 for a constant vector so it never dominates."""
    s = pd.Series(series, dtype=float)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or hi <= lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def candidate_moves(df, assumptions=Assumptions):
    """Build one row per (product, candidate factory), excluding the incumbent."""
    rows = []
    for product, g in df.groupby("product_name"):
        current = g["current_factory"].iloc[0]
        n = int(g["order_id"].nunique())
        total_units = float(g["units"].sum())
        base_profit = float(g["gross_profit"].sum())
        baseline_lead = float(np.average(g["lead_time_days"], weights=g["units"].clip(lower=1)))
        current_dist = float(np.average(g[f"distance_{current}"],
                                        weights=g["units"].clip(lower=1)))
        div = g["division"].iloc[0]
        # Capability history: has this candidate ever produced anything from the division?
        division_factories = set(df[df["division"] == div]["current_factory"].unique())
        for factory in FACTORY_COORDS:
            if factory == current:
                continue
            cand_dist = float(np.average(g[f"distance_{factory}"],
                                         weights=g["units"].clip(lower=1)))
            delta_dist = cand_dist - current_dist
            delta_lead = delta_dist / assumptions.freight_speed_km_day
            delta_cost = total_units * (delta_dist / 1000.0) * assumptions.freight_cost_per_unit_per_1000km
            rows.append({
                "product": product, "division": div, "current_factory": current,
                "candidate_factory": factory, "orders": n, "units": total_units,
                "baseline_lead_days": baseline_lead, "baseline_profit": base_profit,
                "current_distance_km": current_dist, "candidate_distance_km": cand_dist,
                "delta_distance_km": delta_dist, "delta_lead_days": delta_lead,
                "new_lead_days": baseline_lead + delta_lead, "delta_cost": delta_cost,
                "profit_impact": -delta_cost, "new_profit": base_profit - delta_cost,
                "capability_gap": factory not in division_factories,
            })
    return pd.DataFrame(rows)


def risk_scores(moves, residual_std, global_std, assumptions=Assumptions,
                iterations=Assumptions.monte_carlo_iterations):
    """Non-degenerate four-component risk score.

    Components (weights fixed by spec):
      negProfit      - share of Monte-Carlo draws in which the move loses money,
                       counting both the assumed freight delta and a one-time
                       capability-gap transition penalty.
      predVariance   - hold-out residual standard deviation of the best model for this
                       product, normalised. Deliberately NOT |predicted - current|.
      lowSupport     - how far the product falls below the minimum-order evidence gate.
      distanceShift  - size of the distance change, normalised across all candidates.
    """
    residual_std = pd.Series(dtype=float) if residual_std is None else pd.Series(residual_std)
    if global_std is None:
        global_std = float(residual_std.max()) if len(residual_std) else 263.0
    max_abs_delta = float(moves["delta_distance_km"].abs().max()) or 1.0
    max_pred_var = float(residual_std.max()) if len(residual_std) else float(global_std)

    rng = np.random.default_rng(SEED)
    out = []
    for _, r in moves.iterrows():
        product = r["product"]
        pred_var = float(residual_std.get(product, global_std))

        # negProfit: share of draws in which the one-time transition outlay exceeds the
        # freight saving. The outlay is modelled as an uncertain multiple of that saving,
        # so a capability gap can erase its own benefit without reference to gross profit.
        saving = abs(r["delta_cost"])
        outlay = (rng.uniform(0, saving * assumptions.capability_transition_multiple,
                              size=iterations) if r["capability_gap"] else 0.0)
        neg_profit = float(np.mean(outlay > saving))
        low_support = max(0.0, 1.0 - r["orders"] / assumptions.min_orders_for_recommendation)
        components = {
            "negProfit": neg_profit,
            "predVariance": pred_var / max_pred_var,
            "lowSupport": low_support,
            "distanceShift": abs(r["delta_distance_km"]) / max_abs_delta,
        }
        risk = sum(components[k] * w for k, w in RISK_COMPONENTS.items())
        out.append({
            "product": r["product"], "candidate_factory": r["candidate_factory"],
            "risk": risk, "risk_components": json.dumps(components),
            "risk_level": ("High" if risk > RISK_HIGH
                           else ("Medium" if risk > RISK_MEDIUM else "Low")),
        })
    return pd.DataFrame(out)


def score_products(df, assumptions=Assumptions, speed_weight=None,
                   residual_std=None, global_std=None, min_orders=None):
    """Rank candidate factories. `speed_weight` is the live priority-slider input: it trades
    lead-time reduction against profit impact; risk reduction keeps its own fixed weight."""
    if speed_weight is None:
        speed_weight = assumptions.speed_weight
    if min_orders is None:
        min_orders = assumptions.min_orders_for_recommendation

    moves = candidate_moves(df, assumptions)
    if moves.empty:
        return moves

    # Products with no distance-reducing alternative keep their incumbent: an explicit
    # "retain" row keeps them in the coverage count instead of silently dropping them.
    covered = set(moves["product"])
    retained = []
    for product, g in df.groupby("product_name"):
        if product in covered:
            continue
        cur = g["current_factory"].iloc[0]
        units = g["units"].clip(lower=1)
        retained.append({
            "product": product, "division": g["division"].iloc[0],
            "current_factory": cur, "candidate_factory": cur,
            "orders": int(g["order_id"].nunique()), "units": float(g["units"].sum()),
            "baseline_lead_days": float(np.average(g["lead_time_days"], weights=units)),
            "baseline_profit": float(g["gross_profit"].sum()),
            "current_distance_km": float(np.average(g[f"distance_{cur}"], weights=units)),
            "candidate_distance_km": float(np.average(g[f"distance_{cur}"], weights=units)),
            "delta_distance_km": 0.0, "delta_lead_days": 0.0,
            "new_lead_days": float(np.average(g["lead_time_days"], weights=units)),
            "delta_cost": 0.0, "profit_impact": 0.0,
            "new_profit": float(g["gross_profit"].sum()), "capability_gap": False,
        })
    if retained:
        moves = pd.concat([moves, pd.DataFrame(retained)], ignore_index=True)

    risk_frame = risk_scores(
        moves, residual_std if residual_std is not None else pd.Series(dtype=float),
        float(global_std) if global_std is not None else 263.0, assumptions)
    moves = moves.merge(risk_frame[["product", "candidate_factory", "risk", "risk_components",
                                    "risk_level"]],
                        on=["product", "candidate_factory"], how="left")

    moves["lead_gain_days"] = -moves["delta_lead_days"]
    moves["profit_gain"] = moves["profit_impact"]
    moves["is_retain"] = moves["delta_distance_km"] == 0.0

    # Capability-gap penalty to the score, and an evidence-based confidence.
    evidence = np.minimum(1.0, np.log1p(moves["orders"]) / np.log1p(100))
    stability = pd.Series(1.0, index=moves.index)
    base_conf = 100.0 * (0.65 * evidence + 0.35 * stability)
    moves["confidence"] = np.clip(
        base_conf - 100.0 * assumptions.capability_gap_penalty * moves["capability_gap"].astype(float),
        0.0, 99.0)
    moves["sufficient_evidence"] = moves["orders"] >= min_orders
    moves["materially_better"] = moves["lead_gain_days"] >= assumptions.min_material_days

    w_risk = float(assumptions.risk_weight)
    w_lead = speed_weight * (1.0 - w_risk)
    w_profit = (1.0 - speed_weight) * (1.0 - w_risk)
    speed_profit = w_lead * _norm(moves["lead_gain_days"]) + w_profit * _norm(moves["profit_gain"])
    risk_reduction = _norm(-moves["risk"])
    moves["score"] = (1.0 - w_risk) * speed_profit + w_risk * risk_reduction
    moves["score"] = moves["score"] * np.where(
        moves["capability_gap"], 1.0 - assumptions.capability_gap_penalty, 1.0)
    moves["score"] = moves["score"] * moves["confidence"] / 100.0
    return moves.sort_values("score", ascending=False).reset_index(drop=True)


def top_recommendations(moves, top_n=10):
    """Highest-scored move per product, preferring distance-reducing alternatives.

    A move that increases distance is never a recommendation for shortening delivery, so
    reducing alternatives are exhausted first. Only if a product has no reducing alternative
    at all does the incumbent "retain" row fill the slot, so nothing is silently dropped from
    the coverage count.
    """
    if moves.empty:
        return moves
    reducing = moves[moves["delta_distance_km"] < 0].sort_values("score", ascending=False)
    chosen = reducing.drop_duplicates("product").head(top_n)
    remainder = top_n - len(chosen)
    if remainder <= 0:
        return chosen.reset_index(drop=True)
    fallback = moves[~moves["product"].isin(chosen["product"])].sort_values(
        "score", ascending=False).drop_duplicates("product").head(remainder)
    return pd.concat([chosen, fallback]).head(top_n).reset_index(drop=True)


def compute_kpis(moves, evaluable_products=None, best_model_r2=0.0,
                 assumptions=Assumptions, iterations=Assumptions.kpi_bootstrap_iterations):
    """The four header KPIs with bootstrap intervals.

    `evaluable_products` is the number of products that receive a recommendation. Every
    product in the scored set gets one row -- a distance-reducing alternative where one exists,
    otherwise an explicit "retain current assignment" row -- so coverage is measured against the
    whole catalogue rather than silently dropping products with no shorter route. If omitted it
    defaults to the number of distinct products in `moves`. Bootstrap resamples products (one
    recommendation each) rather than rows, so a single high-volume product cannot dominate the
    interval.
    """
    if evaluable_products is None:
        evaluable_products = int(moves["product"].nunique())
    if moves.empty or evaluable_products <= 0:
        return point_and_ci({
            "lead_time_reduction_pct": 0.0,
            "profit_stability_pct": 0.0,
            "scenario_confidence": float(best_model_r2),
            "coverage_pct": 0.0,
        })

    # The KPIs are reported over the actual recommendation set, not over the raw scored
    # frame, so the header agrees with the dashboard's recommendation table.
    per_product = top_recommendations(moves, len(moves["product"].unique()))

    def calc(sample):
        w = sample["orders"].clip(lower=1).to_numpy()
        baseline = sample["baseline_lead_days"].to_numpy()
        if baseline.sum() == 0:
            baseline = np.ones(len(sample))
        return {
            "lead_time_reduction_pct": float(
                np.average(sample["lead_gain_days"].clip(lower=0), weights=w)
                / np.average(baseline, weights=w) * 100.0),
            "profit_stability_pct": float((sample["profit_impact"] >= 0).mean() * 100.0),
            "scenario_confidence": float(best_model_r2),
            "coverage_pct": float(sample["product"].nunique() / evaluable_products * 100.0),
        }

    point = calc(per_product)
    rng = np.random.default_rng(SEED)
    distributions = {k: [] for k in point}
    n = len(per_product)
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        sample = per_product.iloc[idx]
        for k, v in calc(sample).items():
            distributions[k].append(v)
    return point_and_ci(point, distributions)


def point_and_ci(point, distributions=None, ci_lo=5, ci_hi=95):
    out = {"point": point}
    if not distributions or not any(len(v) for v in distributions.values()):
        out["ci"] = {k: [float(v), float(v)] for k, v in point.items()}
    else:
        out["ci"] = {
            k: [float(min(np.percentile(distributions[k], ci_lo), point[k])),
                float(max(np.percentile(distributions[k], ci_hi), point[k]))]
            for k in point
        }
    return out


def sensitivity(df, assumptions=Assumptions, iterations=300):
    """Monte-Carlo over the two freight assumptions; tracks both KPI magnitude and
    which factory each product would move to."""
    rng = np.random.default_rng(SEED)
    products = sorted(df["product_name"].unique())
    results = []
    choice_log = []
    for i in range(iterations):
        a = type("Draw", (object,), dict(vars(Assumptions)))()
        a.freight_speed_km_day = float(rng.uniform(400.0, 1600.0))
        a.freight_cost_per_unit_per_1000km = float(rng.uniform(0.05, 0.30))
        a.speed_weight = float(rng.uniform(0.25, 0.75))
        scored = score_products(df, assumptions=a)
        recs = top_recommendations(scored)
        if len(recs):
            ev = int(recs["product"].nunique()) if len(recs) else 0
            kpis = compute_kpis(recs, ev, best_model_r2=0.0,
                                assumptions=a, iterations=0)["point"]
        else:
            kpis = {k: 0.0 for k in ("lead_time_reduction_pct", "profit_stability_pct",
                                     "scenario_confidence", "coverage_pct")}
        results.append({"iteration": i, "speed_km_day": a.freight_speed_km_day,
                        "freight_rate": a.freight_cost_per_unit_per_1000km, **kpis})
        choice_log.append({r["product"]: r["candidate_factory"] for _, r in recs.iterrows()})
    return pd.DataFrame(results), choice_log


def factory_optimiser_summary(df, moves):
    """Module 1 view: one row per factory with its current portfolio and inbound potential."""
    summary = []
    for factory in FACTORY_COORDS:
        portfolio = df[df["current_factory"] == factory]
        candidates = moves[moves["candidate_factory"] == factory]
        summary.append({
            "factory": factory,
            "orders": int(portfolio["order_id"].nunique()) if len(portfolio) else 0,
            "units": float(portfolio["units"].sum()) if len(portfolio) else 0.0,
            "products": int(portfolio["product_name"].nunique()) if len(portfolio) else 0,
            "mean_lead_days": float(portfolio["lead_time_days"].mean())
            if len(portfolio) else np.nan,
            "mean_distance_km": float(portfolio["distance_km"].mean())
            if len(portfolio) else np.nan,
            "gross_profit": float(portfolio["gross_profit"].sum()) if len(portfolio) else 0.0,
            "mean_candidate_lead_reduction_days": (
                float(candidates["lead_gain_days"].mean()) if len(candidates) else 0.0),
            "candidate_products": int(candidates["product"].nunique()) if len(candidates) else 0,
        })
    return pd.DataFrame(summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from data_prep import load_and_audit
    from models import train_and_persist

    df, audit = load_and_audit()
    report, best_name, pipe, importance, resid, gstd = train_and_persist(df)
    residual_std = resid.set_index("product")["residual_std"]
    moves = score_products(df, residual_std=residual_std.to_dict(), global_std=gstd)
    evaluable = int(moves["product"].nunique())
    recs = top_recommendations(moves, 10)
    print("candidate pairs:", len(moves), "| top-10 recommendations:", len(recs),
          "| evaluable products:", evaluable)
    print("distinct factories in top 10:", sorted(recs["candidate_factory"].unique()))
    print("coverage:", moves["product"].nunique(), "/", audit["product_count"])
    # KPIs are computed over the complete recommendation set (one row per product),
    # not just the displayed top 10, so coverage reflects the full recommendation set.
    kpis = compute_kpis(moves, evaluable, report.iloc[0]["r2"])
    print(json.dumps(kpis, indent=2))
    print(recs[["product", "current_factory", "candidate_factory", "orders",
                "lead_gain_days", "delta_distance_km", "risk", "risk_level",
                "capability_gap", "score"]].to_string(index=False))
    print("\nrisk component breakdown for the top move:")
    print(json.dumps(parse_components(recs.iloc[0]["risk_components"]), indent=2))