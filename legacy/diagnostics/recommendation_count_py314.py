"""Read-only recommendation diagnostics for the Python 3.14 environment.

Loads the prepared-order artifact and calls the production score_products
function directly. It writes only labeled diagnostic artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ARTIFACT_DIR, Assumptions, FACTORY_COORDS  # noqa: E402
from simulation import score_products  # noqa: E402


def score_all(frame: pd.DataFrame, *, weight: float | None = None,
              assumptions: object = Assumptions) -> pd.DataFrame:
    """Return every alternative candidate for each product."""
    if weight is None:
        return score_products(frame, assumptions=assumptions,
                              top_n=len(FACTORY_COORDS))
    local = SimpleNamespace(**{
        name: value for name, value in vars(assumptions).items()
        if not name.startswith("__")
    })
    local.speed_weight = float(weight)
    return score_products(frame, assumptions=local,
                          top_n=len(FACTORY_COORDS))


def top_picks(candidates: pd.DataFrame) -> pd.DataFrame:
    return (candidates.sort_values("score", ascending=False)
            .drop_duplicates("product").copy())


def gates_summary(candidates: pd.DataFrame) -> dict[str, object]:
    picks = top_picks(candidates)
    actionable = picks[picks.sufficient_evidence.fillna(False)
                       & picks.materially_better.fillna(False)]
    materiality_only = picks[picks.sufficient_evidence.fillna(False)
                             & ~picks.materially_better.fillna(False)]
    evidence_only = picks[~picks.sufficient_evidence.fillna(False)
                          & picks.materially_better.fillna(False)]
    both = picks[~picks.sufficient_evidence.fillna(False)
                 & ~picks.materially_better.fillna(False)]
    return {
        "product_count": int(picks["product"].nunique()),
        "actionable_count": int(actionable["product"].nunique()),
        "materiality_only_count": int(materiality_only["product"].nunique()),
        "evidence_only_count": int(evidence_only["product"].nunique()),
        "both_gates_fail_count": int(both["product"].nunique()),
        "actionable_products": "; ".join(sorted(actionable["product"].tolist())),
        "actionable_details": [
            {
                "product": row.product,
                "candidate_factory": row.candidate_factory,
                "orders": int(row.orders),
                "lead_gain_days": float(row.lead_gain),
                "profit_impact": float(row.profit_impact),
            }
            for row in actionable.itertuples()
        ],
        "top_factories": dict(zip(picks["product"], picks["candidate_factory"])),
        "actionable_product_set": set(actionable["product"].tolist()),
    }


def gate_label(row: object) -> str:
    evidence = bool(row.sufficient_evidence)
    material = bool(row.materially_better)
    if evidence and material:
        return "actionable"
    if evidence:
        return "materiality_only"
    if material:
        return "evidence_only"
    return "both"


def write_state_only(frame: pd.DataFrame) -> pd.DataFrame:
    state = frame.copy()
    for factory in FACTORY_COORDS:
        state[f"distance_{factory}"] = state[f"distance_state_{factory}"]
    all_candidates = score_all(state)
    picks = top_picks(all_candidates)
    picks_by_product = dict(zip(picks["product"], picks["candidate_factory"]))
    result = all_candidates.copy()
    result["top_ranked_state_only"] = result.apply(
        lambda row: picks_by_product.get(row["product"]) == row["candidate_factory"],
        axis=1,
    )
    result["gate_result"] = result.apply(gate_label, axis=1)
    result["min_orders_required"] = Assumptions.min_orders_for_recommendation
    result["min_material_gain_days"] = Assumptions.min_material_days
    cols = [
        "product", "current_factory", "candidate_factory", "orders",
        "top_ranked_state_only", "delta_distance_km", "delta_lead_days",
        "lead_gain", "profit_impact", "sufficient_evidence",
        "materially_better", "gate_result", "score",
        "min_orders_required", "min_material_gain_days",
    ]
    return result[cols].sort_values(["product", "score"], ascending=[True, False])


def run_weight_sweep(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    default = gates_summary(score_all(frame, weight=Assumptions.speed_weight))
    rows = []
    found_five = False
    for weight in [round(i / 20, 2) for i in range(21)]:
        result = gates_summary(score_all(frame, weight=weight))
        changed = sorted(
            product for product in set(default["top_factories"]) | set(result["top_factories"])
            if default["top_factories"].get(product) != result["top_factories"].get(product)
        )
        current_actionable = set(result["actionable_product_set"])
        default_actionable = set(default["actionable_product_set"])
        is_five = result["actionable_count"] == 5
        rows.append({
            "speed_weight": weight,
            "actionable_count": result["actionable_count"],
            "actionable_products": result["actionable_products"],
            "products_with_top_factory_flip_vs_default": "; ".join(changed),
            "flip_count_vs_default": len(changed),
            "products_entering_actionable_vs_default": "; ".join(sorted(current_actionable - default_actionable)),
            "products_leaving_actionable_vs_default": "; ".join(sorted(default_actionable - current_actionable)),
            "matches_live_count_5": is_five,
        })
        found_five = found_five or is_five
    return pd.DataFrame(rows), found_five


def run_freight_speed_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the full freight-speed slider range at its configured step."""
    rows = []
    for speed in range(400, 1601, 50):
        local = copy_assumptions(freight_speed_km_day=speed)
        result = gates_summary(score_all(frame, assumptions=local))
        rows.append({
            "freight_speed_km_day": speed,
            "actionable_count": result["actionable_count"],
            "actionable_products": result["actionable_products"],
            "actionable_details": json.dumps(result["actionable_details"]),
            "matches_live_count_5": result["actionable_count"] == 5,
        })
    return pd.DataFrame(rows)


def copy_assumptions(**updates: object) -> object:
    local = SimpleNamespace(**{
        name: value for name, value in vars(Assumptions).items()
        if not name.startswith("__")
    })
    for name, value in updates.items():
        setattr(local, name, value)
    return local


def sweep_until_five(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Sweep scoring assumptions one at a time, stopping at the first five."""
    trials: list[dict[str, object]] = []
    # The dashboard exposes speed (400..1600 by 50) and cost (.05..30 by .01).
    # They are checked here too because they change per-product candidate scores.
    sweep_specs: list[tuple[str, list[object], str]] = [
        ("freight_speed_km_day", list(range(400, 1601, 50)), "dashboard_slider"),
        ("freight_cost_per_unit_per_1000km",
         [round(i / 100, 2) for i in range(5, 31)], "dashboard_slider"),
        # The dashboard carries these config values in its assumptions object,
        # though it does not expose direct controls for them.
        ("min_orders_for_recommendation", list(range(1, 21)), "config_gate_assumption"),
        ("min_material_days", [round(i / 20, 2) for i in range(0, 31)],
         "config_gate_assumption"),
        ("capability_gap_penalty", [round(i / 20, 2) for i in range(0, 21)],
         "config_scoring_assumption"),
    ]
    first = "none"
    for name, values, category in sweep_specs:
        for value in values:
            local = copy_assumptions(**{name: value})
            result = gates_summary(score_all(frame, assumptions=local))
            hit = result["actionable_count"] == 5
            trials.append({
                "assumption": name,
                "value": value,
                "category": category,
                "actionable_count": result["actionable_count"],
                "actionable_products": result["actionable_products"],
                "actionable_details": json.dumps(result["actionable_details"]),
                "matches_live_count_5": hit,
            })
            if hit:
                first = f"{name}={value} ({category})"
                return pd.DataFrame(trials), first
    return pd.DataFrame(trials), first


def main() -> None:
    frame = pd.read_csv(ARTIFACT_DIR / "prepared_orders.csv")
    if frame.product_name.nunique() != 15:
        raise RuntimeError(f"Expected 15 products; found {frame.product_name.nunique()}")

    mixed = score_all(frame)
    summary = gates_summary(mixed)
    near_miss_path = ARTIFACT_DIR / "near_miss_diagnostic.csv"
    hand = pd.read_csv(near_miss_path)
    hand_top = hand.drop_duplicates("product")
    hand_counts = {
        "products": int(hand_top["product"].nunique()),
        "actionable": int((hand_top["failed_gates"] == "neither (actionable)").sum()),
        "materiality_only": int((hand_top["failed_gates"] == "materiality only").sum()),
        "both": int((hand_top["failed_gates"] == "both").sum()),
        "closest_materiality_gap_days": float(hand.material_gap_days.max()),
    }
    actual_top = top_picks(mixed)
    actual_top["gate_result"] = actual_top.apply(gate_label, axis=1)
    actual_top["material_gap_days"] = (
        actual_top.lead_gain - Assumptions.min_material_days
    )
    actual_top["matched_hand_table_gate"] = actual_top.apply(
        lambda row: row["gate_result"] == hand_top.loc[
            hand_top["product"] == row["product"], "failed_gates"
        ].iloc[0].replace("materiality only", "materiality_only").replace(
            "evidence only", "evidence_only"
        ).replace("neither (actionable)", "actionable"), axis=1,
    )
    actual_top.to_csv(ARTIFACT_DIR / "recommendation_runtime_check_py314.csv", index=False)

    saved_recs = pd.read_csv(ARTIFACT_DIR / "recommendations.csv")
    saved_top = top_picks(saved_recs).set_index("product")
    runtime_top = actual_top.set_index("product")
    saved_pick_match = all(
        runtime_top.loc[product, "candidate_factory"] == saved_top.loc[product, "candidate_factory"]
        and abs(float(runtime_top.loc[product, "lead_gain"])
                - float(saved_top.loc[product, "lead_gain"])) < 1e-10
        for product in saved_top.index
    )

    state_result = write_state_only(frame)
    state_actionable = state_result[
        state_result.sufficient_evidence.fillna(False)
        & state_result.materially_better.fillna(False)
    ]
    state_top_actionable = state_actionable[
        state_actionable.top_ranked_state_only
    ]
    state_result.to_csv(ARTIFACT_DIR / "state_only_diagnostic.csv", index=False)
    weights, weight_hit = run_weight_sweep(frame)
    weights.to_csv(ARTIFACT_DIR / "speed_weight_sweep.csv", index=False)
    freight_speeds = run_freight_speed_sweep(frame)
    freight_speeds.to_csv(ARTIFACT_DIR / "freight_speed_sweep.csv", index=False)
    if weight_hit:
        fallback = pd.DataFrame(columns=[
            "assumption", "value", "category", "actionable_count",
            "actionable_products", "actionable_details", "matches_live_count_5",
        ])
        first_five = "speed_weight sweep produced one or more five-count values"
    else:
        fallback, first_five = sweep_until_five(frame)
    fallback.to_csv(ARTIFACT_DIR / "assumption_fallback_sweep.csv", index=False)

    max_material_gap = float(actual_top.material_gap_days.max())
    sanity = {
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "mixed_distance_scoring": {
            key: value for key, value in summary.items()
            if key not in {"top_factories", "actionable_details", "actionable_product_set"}
        },
        "hand_near_miss_reference": hand_counts,
        "user_context_claimed_counts": {
            "actionable": 0,
            "materiality_only": 11,
            "both": 4,
        },
        "user_context_claim_matches_saved_hand_table": (
            hand_counts["actionable"] == 0
            and hand_counts["materiality_only"] == 11
            and hand_counts["both"] == 4
        ),
        "sanity_check_matches_counts": (
            summary["product_count"] == hand_counts["products"]
            and summary["actionable_count"] == hand_counts["actionable"]
            and summary["materiality_only_count"] == hand_counts["materiality_only"]
            and summary["both_gates_fail_count"] == hand_counts["both"]
        ),
        "runtime_closest_materiality_gap_days": max_material_gap,
        "runtime_top_pick_gates_match_hand_table": bool(
            actual_top.matched_hand_table_gate.all()
        ),
        "runtime_top_picks_match_saved_recommendations": bool(saved_pick_match),
        "state_only_actionable_candidate_count": int(state_actionable["product"].nunique()),
        "state_only_actionable_candidates": state_actionable[[
            "product", "candidate_factory", "orders", "lead_gain", "profit_impact"
        ]].to_dict(orient="records"),
        "state_only_actionable_top_pick_count": int(state_top_actionable["product"].nunique()),
        "speed_weight_has_five_count": weight_hit,
        "first_five_count_assumption": first_five,
        "monte_carlo_iterations_effect": (
            "not swept: it affects compute_kpis uncertainty iterations, not "
            "score_products ranking or the gated top-pick recommendation count"
        ),
        "capacity_multiplier_effect": (
            "not swept: it affects joint_optimize capacity constraints, not "
            "the per-product recommendation count"
        ),
    }
    (ARTIFACT_DIR / "recommendation_diagnostic_summary_py314.json").write_text(
        json.dumps(sanity, indent=2), encoding="utf-8"
    )

    print("MIXED-DISTANCE SCORING SANITY CHECK")
    print(json.dumps(sanity, indent=2))
    print("\nMIXED-DISTANCE TOP PICKS (runtime versus saved artifact)")
    print(actual_top[[
        "product", "candidate_factory", "orders", "delta_lead_days",
        "material_gap_days", "sufficient_evidence", "materially_better",
        "gate_result", "matched_hand_table_gate",
    ]].to_string(index=False))
    print("\nSTATE-ONLY CANDIDATE GATES (all alternatives)")
    print(state_result.to_string(index=False))
    print("\nSPEED-WEIGHT SWEEP")
    print(weights.to_string(index=False))
    print("\nFULL DASHBOARD FREIGHT-SPEED SWEEP")
    print(freight_speeds[[
        "freight_speed_km_day", "actionable_count", "actionable_products",
        "matches_live_count_5",
    ]].to_string(index=False))
    print("\nOTHER ASSUMPTION SWEEP")
    print(fallback.to_string(index=False) if not fallback.empty else "Not run: a speed_weight value already produced 5.")
    print(f"\nFIRST EXACTLY-FIVE RESULT: {first_five}")


if __name__ == "__main__":
    main()
