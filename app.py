"""Interactive Streamlit decision support for factory allocation scenarios."""
from types import SimpleNamespace
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from config import ARTIFACT_DIR, FACTORY_COORDS, PRODUCT_FACTORY, Assumptions
from simulation import score_products

ROOT = Path(__file__).resolve().parent
BG = "#0e0e0e"
PANEL = "#1c1c1c"
TEXT = "#f5f5f5"
ACCENT = "#F2B84B"
FACTORY_COLORS = {
    "Lot's O' Nuts": [255, 154, 61],
    "Wicked Choccy's": [57, 213, 220],
    "Sugar Shack": [167, 243, 112],
    "Secret Factory": [247, 120, 196],
    "The Other Factory": [255, 211, 77],
}
MAP_STYLE_URL = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

st.set_page_config(page_title="Nassau Candy | Factory Allocation", page_icon="🍬", layout="wide")


def stretch_width():
    """Use the layout keyword supported by the pinned and newer Streamlit APIs."""
    major_minor = tuple(int(part) for part in st.__version__.split(".")[:2])
    return {"width": "stretch"} if major_minor >= (1, 50) else {"use_container_width": True}


@st.cache_data(show_spinner=False)
def load_orders():
    path = ARTIFACT_DIR / "prepared_orders.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python run_pipeline.py` before launching the dashboard.")
    return pd.read_csv(path, parse_dates=["order_date", "ship_date"])


@st.cache_data(show_spinner=False)
def load_table(name):
    return pd.read_csv(ARTIFACT_DIR / name)


def make_assumptions(speed, cost, weight):
    return SimpleNamespace(
        freight_speed_km_day=float(speed),
        freight_cost_per_unit_per_1000km=float(cost),
        capability_gap_penalty=Assumptions.capability_gap_penalty,
        min_orders_for_recommendation=Assumptions.min_orders_for_recommendation,
        min_material_days=Assumptions.min_material_days,
        capacity_multiplier=Assumptions.capacity_multiplier,
        speed_weight=float(weight),
        monte_carlo_iterations=Assumptions.monte_carlo_iterations,
    )


def display_frame(frame, height=350, dim_thin=False):
    """Format numeric cells for legibility while keeping status cells as plain text."""
    view = frame.copy()
    for col in view.columns:
        if pd.api.types.is_bool_dtype(view[col]) or not pd.api.types.is_numeric_dtype(view[col]):
            continue
        low = col.lower()
        is_delta = "delta" in low or col.startswith("Δ")
        if any(token in low for token in ("profit", "cost", "sales", "gross")):
            view[col] = view[col].map(lambda x: "—" if pd.isna(x) else (f"${x:+,.2f}" if is_delta or "impact" in low else f"${x:,.2f}"))
        elif "distance" in low or low.endswith("_km"):
            view[col] = view[col].map(lambda x: "—" if pd.isna(x) else (f"{x:+,.0f} km" if is_delta else f"{x:,.0f} km"))
        elif is_delta or "gain" in low:
            view[col] = view[col].map(lambda x: "—" if pd.isna(x) else f"{x:+,.2f}")
        elif any(token in low for token in ("confidence", "score", "pct", "margin", "stability")):
            if "margin" in low:
                view[col] = view[col].map(lambda x: "—" if pd.isna(x) else f"{x * 100:,.1f}%")
            else:
                view[col] = view[col].map(lambda x: "—" if pd.isna(x) else f"{x:,.1f}{'%' if any(t in low for t in ('confidence', 'pct')) else ''}")
        elif any(token in low for token in ("orders", "units", "volume", "count")):
            view[col] = view[col].map(lambda x: "—" if pd.isna(x) else f"{x:,.0f}")
        else:
            view[col] = view[col].map(lambda x: "—" if pd.isna(x) else f"{x:,.2f}")
    rendered = view
    if dim_thin and "status" in view.columns:
        def thin_row(row):
            if "Thin evidence" in str(row.get("status", "")):
                return ["color: #b8b8b8; background-color: #1c1c1c"] * len(row)
            return [""] * len(row)
        # Style the row while keeping cell values (especially status labels) plain text.
        rendered = view.style.apply(thin_row, axis=1)
    st.dataframe(rendered, hide_index=True, height=height, **stretch_width())


def filter_orders(orders, product, region, ship_mode):
    subset = orders[orders.product_name == product]
    if region != "All regions":
        subset = subset[subset.region == region]
    if ship_mode != "All ship modes":
        subset = subset[subset.ship_mode == ship_mode]
    return subset


def scenario_rows(subset, all_orders, assumptions):
    if subset.empty:
        return pd.DataFrame()
    all_candidates = score_products(
        subset, assumptions=assumptions, top_n=len(FACTORY_COORDS), reference_df=all_orders
    )
    if all_candidates.empty:
        return all_candidates
    return all_candidates


def top_factory_chart(scenarios, current_factory):
    data = scenarios.sort_values("new_lead_days", ascending=True).copy()
    alternatives = data[data.candidate_factory != current_factory]
    best_factory = alternatives.iloc[0].candidate_factory if not alternatives.empty else None
    colors = []
    for factory in data.candidate_factory:
        if factory == current_factory:
            colors.append("#F2B84B")
        elif factory == best_factory:
            colors.append("#49D6C7")
        else:
            colors.append("#6F7C8A")
    fig, ax = plt.subplots(figsize=(10, 3.4), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.barh(data.candidate_factory, data.new_lead_days, color=colors, edgecolor="#0e0e0e", height=.66)
    ax.invert_yaxis()
    ax.set_xlabel("Projected lead time (days; assumptions applied to observed baseline)", color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.grid(axis="x", color="#343434", linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    right = max(float(data.new_lead_days.max()), 0.2)
    ax.set_xlim(min(0, float(data.new_lead_days.min()) * 1.08), right * 1.22)
    for bar, value in zip(bars, data.new_lead_days):
        ax.text(value + max(right * .015, .03), bar.get_y() + bar.get_height()/2, f"{value:.2f} d", color=TEXT, va="center", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, **stretch_width())
    plt.close(fig)
    st.caption("Gold: current factory · Teal: fastest alternative · Gray: other alternatives")


def factory_map_data():
    return pd.DataFrame([
        {"factory": name, "latitude": lat, "longitude": lon, "color": FACTORY_COLORS[name]}
        for name, (lat, lon) in FACTORY_COORDS.items()
    ])


def build_factory_deck():
    map_data = factory_map_data()
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_line_color=[245, 245, 245, 210],
        get_radius=62000,
        line_width_min_pixels=2,
        stroked=True,
        filled=True,
        pickable=True,
    )
    return pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=39.0, longitude=-98.0, zoom=3, pitch=0),
        map_provider="carto",
        map_style=MAP_STYLE_URL,
        tooltip={"text": "{factory}"},
    )


def factory_map():
    deck = build_factory_deck()
    st.pydeck_chart(deck, **stretch_width())
    st.caption("Factory markers: 🟠 Lot's O' Nuts · 🔹 Wicked Choccy's · 🟢 Sugar Shack · 🟣 Secret Factory · 🟡 The Other Factory")


def speed_curve(row, selected_speed):
    speeds = np.linspace(400, 1600, 60)
    projected = row.baseline_lead_days + row.delta_distance_km / speeds
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(speeds, projected, color="#49D6C7", linewidth=2.5)
    ax.axvline(selected_speed, color=ACCENT, linestyle="--", linewidth=1.5, label=f"Selected: {selected_speed:,.0f} km/day")
    ax.set_xlabel("Assumed freight speed (km/day)", color=TEXT)
    ax.set_ylabel("Projected lead time (days)", color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.grid(color="#343434", alpha=.8)
    ax.legend(facecolor=PANEL, edgecolor="#444", labelcolor=TEXT)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    fig.tight_layout()
    st.pyplot(fig, **stretch_width())
    plt.close(fig)


def sensitivity_chart(samples, point):
    values = samples.lead_time_reduction_pct.dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(9, 3.4), facecolor=BG)
    ax.set_facecolor(BG)
    if len(values):
        ax.hist(values, bins=24, color="#49D6C7", edgecolor=BG, alpha=.9)
        ax.axvline(point, color=ACCENT, linewidth=2.4, label=f"Configured point estimate: {point:.2f}%")
        ax.axvline(np.percentile(values, 5), color="#A88BFA", linestyle="--", label="5th / 95th percentiles")
        ax.axvline(np.percentile(values, 95), color="#A88BFA", linestyle="--")
    ax.set_xlabel("Actionable lead-time reduction (%)", color=TEXT)
    ax.set_ylabel("Monte Carlo draws", color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.grid(axis="y", color="#343434", alpha=.7)
    ax.legend(facecolor=PANEL, edgecolor="#444", labelcolor=TEXT)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    fig.tight_layout()
    st.pyplot(fig, **stretch_width())
    plt.close(fig)


def add_recommendation_status(frame):
    result = frame.copy()
    status = []
    for row in result.itertuples():
        if not bool(getattr(row, "sufficient_evidence", False)):
            status.append("⚠️ Thin evidence")
        elif bool(getattr(row, "capability_gap", False)):
            status.append("⚠️ Capability gap")
        elif bool(getattr(row, "materially_better", False)) and getattr(row, "profit_impact", 0) >= 0:
            status.append("✅ Actionable")
        elif getattr(row, "profit_impact", 0) < 0:
            status.append("⚠️ Profit reduction")
        else:
            status.append("— Below materiality")
    result["status"] = status
    result["evidence_status"] = np.where(result.sufficient_evidence.fillna(False), "✅ Sufficient evidence", "⚠️ Thin evidence")
    result["improvement_status"] = np.where(result.materially_better.fillna(False), "✅ Materially better", "— Below materiality")
    return result


def main():
    try:
        orders = load_orders()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    st.title("Nassau Candy · Factory Reallocation")
    st.caption("Observed history is measured; alternative-factory lead-time and freight-cost changes depend on the assumptions shown.")
    with st.expander("Click to read the statistical case"):
        st.markdown(
            """The source ship years are corrupt: the naive gap spans hundreds to over a thousand days. """
            """Dates are reconstructed from the order year and raw ship month/day, then corrected by the data-derived minimum residual. """
            """Shipping mode carries most predictive signal; temporal validation is weaker than a random split, and within-mode distance slopes are not significant. """
            """The scenario engine uses measured centroid-distance deltas but assumes freight speed and cost. ZIP centroids cover 95.6% of retained rows; supplied state/province centroids cover the rest. """
            """Treat suggestions as hypotheses for route pilots, not as measured savings."""
        )

    with st.sidebar:
        st.markdown("### Filters")
        products = sorted(orders.product_name.dropna().unique())
        product = st.selectbox("Product", products, index=0)
        regions = ["All regions"] + sorted(orders.region.dropna().unique())
        region = st.selectbox("Region", regions, index=0)
        modes = ["All ship modes"] + sorted(orders.ship_mode.dropna().unique())
        ship_mode = st.selectbox("Ship mode", modes, index=0)
        st.markdown("### Assumptions")
        speed = st.slider("Freight speed (assumed km/day)", 400, 1600, int(Assumptions.freight_speed_km_day), 50)
        cost = st.slider("Freight cost (assumed /unit/1,000 km)", 0.05, 0.30, float(Assumptions.freight_cost_per_unit_per_1000km), 0.01, format="%.2f")
        weight = st.slider("Speed vs profit priority", 0.0, 1.0, float(Assumptions.speed_weight), 0.05, format="%.2f")
        st.caption("Under the current distance-only freight model, speed and profit gains are perfectly correlated, so this control does not change rankings.")

    assumptions = make_assumptions(speed, cost, weight)
    subset = filter_orders(orders, product, region, ship_mode)
    scenarios = scenario_rows(subset, orders, assumptions)
    low_evidence = subset.order_id.nunique() < Assumptions.min_orders_for_recommendation

    tab_sim, tab_whatif, tab_recs, tab_risk, tab_evidence = st.tabs([
        "Factory Optimization Simulator",
        "What-If Scenario Analysis",
        "Recommendation Dashboard",
        "Risk & Impact Panel",
        "Evidence & Diagnostics",
    ])

    with tab_sim:
        st.subheader(product)
        if low_evidence:
            st.warning(
                f"⚠️ Low evidence: this product/region/ship-mode filter contains {subset.order_id.nunique():,} historical orders; "
                f"the minimum for recommendations is {Assumptions.min_orders_for_recommendation}. Figures below are indicative only."
            )
        if scenarios.empty:
            st.info("No historical rows match this filter combination.")
        else:
            display_frame(scenarios[["candidate_factory", "orders", "units", "new_lead_days", "candidate_distance_km", "new_profit", "profit_impact", "confidence", "capability_gap"]].rename(columns={
                "candidate_factory": "Factory", "orders": "Historical orders", "units": "Units", "new_lead_days": "Projected lead (days)",
                "candidate_distance_km": "Distance (km)", "new_profit": "Projected gross profit", "profit_impact": "Profit impact", "confidence": "Confidence (%)", "capability_gap": "Capability gap",
            }), height=245)
            top_factory_chart(scenarios, PRODUCT_FACTORY.get(product, ""))
            current = scenarios.iloc[0]
            st.caption(
                f"Baseline is weighted by observed units from {current.orders:,.0f} orders. "
                f"Candidate lead-time and profit changes use assumed speed {speed:,.0f} km/day and cost ${cost:.2f}/unit/1,000 km."
            )

    with tab_whatif:
        st.subheader("Compare one factory move")
        if scenarios.empty:
            st.info("No historical rows match this filter combination.")
        else:
            candidates = scenarios[scenarios.candidate_factory != PRODUCT_FACTORY.get(product)]
            if candidates.empty:
                st.info("No alternative factory is available for this product.")
            else:
                default_factory = candidates.sort_values("score", ascending=False).iloc[0].candidate_factory
                chosen_factory = st.selectbox("Alternative factory", candidates.candidate_factory.tolist(), index=candidates.candidate_factory.tolist().index(default_factory), key="whatif_factory")
                chosen = candidates[candidates.candidate_factory == chosen_factory].iloc[0]
                incumbent = scenarios[scenarios.candidate_factory == PRODUCT_FACTORY.get(product)].iloc[0]
                left, right = st.columns(2)
                with left:
                    st.markdown(f"#### Current · {incumbent.candidate_factory}")
                    st.metric("Observed baseline lead", f"{incumbent.baseline_lead_days:.2f} days")
                    st.metric("Observed gross profit", f"${incumbent.baseline_profit:,.2f}")
                    st.metric("Weighted distance", f"{incumbent.current_distance_km:,.0f} km")
                with right:
                    st.markdown(f"#### Candidate · {chosen.candidate_factory}")
                    st.metric("Projected lead", f"{chosen.new_lead_days:.2f} days", f"{chosen.delta_lead_days:+.2f} days")
                    st.metric("Projected gross profit", f"${chosen.new_profit:,.2f}", f"${chosen.profit_impact:+,.2f}")
                    st.metric("Weighted distance", f"{chosen.candidate_distance_km:,.0f} km", f"{chosen.delta_distance_km:+,.0f} km")
                st.caption("Distance delta is calculated from supplied factory and customer centroids. Lead and freight-cost deltas are assumption-driven.")
                st.markdown("##### Projected saving across assumed freight speeds")
                speed_curve(chosen, speed)

    with tab_recs:
        st.subheader("Ranked product assignments")
        mode = st.radio("Assignment mode", ["Per-product best option", "Globally optimal joint assignment"], horizontal=True)
        actionable_only = st.checkbox("Actionable only", value=False)
        if mode == "Per-product best option":
            candidates = score_products(orders, assumptions=assumptions, top_n=len(FACTORY_COORDS))
            candidates = candidates[~candidates.is_incumbent]
            recommendations = candidates.sort_values("score", ascending=False).drop_duplicates("product")
            view = add_recommendation_status(recommendations)
        else:
            candidates = score_products(orders, assumptions=assumptions, top_n=len(FACTORY_COORDS))
            exploratory = st.toggle("Exploratory (ignores evidence and materiality gates)", value=False)
            if exploratory:
                st.warning("Exploratory assignment ignores evidence and materiality gates; treat moves as ranking scenarios only.")
            artifact = "joint_assignment_exploratory.csv" if exploratory else "joint_assignment_gated.csv"
            assignment = load_table(artifact)
            view = assignment.merge(candidates, left_on=["product", "factory"], right_on=["product", "candidate_factory"], how="left", suffixes=("_assigned", "_candidate"))
            view["units"] = view["units_assigned"].fillna(view["units_candidate"])
            view["candidate_factory"] = view.factory
            view["current_factory"] = view.current_factory.fillna(view.factory)
            order_counts = orders.groupby("product_name").order_id.nunique()
            view["orders"] = view["orders"].fillna(view["product"].map(order_counts))
            view["sufficient_evidence"] = view.sufficient_evidence.fillna(view.orders >= Assumptions.min_orders_for_recommendation)
            view["materially_better"] = view.materially_better.fillna(False)
            view["capability_gap"] = view.capability_gap.fillna(False)
            view["profit_impact"] = view.profit_impact.fillna(0.0)
            view = add_recommendation_status(view)
        if actionable_only:
            view = view[view.sufficient_evidence.fillna(False) & view.materially_better.fillna(False)]
        cols = [c for c in ["product", "current_factory", "candidate_factory", "orders", "units", "delta_distance_km", "delta_lead_days", "profit_impact", "confidence", "score", "evidence_status", "improvement_status", "capability_gap", "status"] if c in view]
        display_frame(view[cols].rename(columns={
            "product": "Product", "current_factory": "Current factory", "candidate_factory": "Recommended factory", "orders": "Orders", "units": "Units",
            "delta_distance_km": "Δ distance (km)", "delta_lead_days": "Δ lead (days)", "profit_impact": "Profit impact", "confidence": "Confidence (%)",
            "score": "Objective score", "capability_gap": "Capability gap",
            "evidence_status": "Evidence", "improvement_status": "Materiality",
        }), height=500, dim_thin=True)
        st.download_button("Download current table as CSV", view[cols].to_csv(index=False).encode("utf-8"), "factory_recommendations.csv", "text/csv")

    with tab_risk:
        st.subheader("Operational risks and route exposure")
        candidates = score_products(orders, assumptions=assumptions, top_n=len(FACTORY_COORDS))
        candidates = candidates[~candidates.is_incumbent]
        gaps = candidates[candidates.capability_gap].copy()
        if len(gaps):
            gaps["status"] = "⚠️ Capability gap"
            st.markdown("#### Capability-gap alternatives")
            display_frame(gaps[["product", "candidate_factory", "orders", "confidence", "score", "status"]], height=220)
        else:
            st.success("No capability-gap alternatives appear in the current candidate set.")
        losses = candidates[candidates.profit_impact < 0].copy()
        st.markdown("#### Moves with modeled profit reductions")
        if len(losses):
            losses["status"] = "⚠️ Profit reduction"
            display_frame(losses[["product", "current_factory", "candidate_factory", "delta_distance_km", "profit_impact", "status"]], height=220)
        else:
            st.info("No candidate move reduces modeled profit under the selected assumptions.")
        routes = load_table("routes.csv")
        problem = routes[routes.problem_route.astype(bool)].sort_values("exposure", ascending=False)
        st.markdown("#### Slow routes flagged by clustering")
        display_frame(problem[["current_factory", "region", "division", "order_count", "mean_lead", "p90_lead", "mean_distance", "mean_margin", "cluster_label", "exposure"]].head(20), height=300)
        st.markdown("#### Factory locations")
        factory_map()

    with tab_evidence:
        st.subheader("Evidence and diagnostics")
        st.markdown("#### Nested feature ablation · five-fold CV R²")
        display_frame(load_table("ablation.csv"), height=260)
        st.markdown("#### Distance slope within each ship mode")
        display_frame(load_table("mode_slopes.csv"), height=200)
        st.markdown("#### State identity vs distance, controlling for ship mode")
        display_frame(load_table("state_distance.csv"), height=180)
        st.markdown("#### Historical residual by distance decile")
        display_frame(load_table("distance_residual_bins.csv"), height=260)
        st.markdown("#### Random split and temporal holdout")
        model_metrics = load_table("model_metrics.csv").rename(columns={"r2": "r2_random", "rmse": "rmse_random", "mae": "mae_random"})
        temporal_values = json.loads((ARTIFACT_DIR / "temporal_metrics.json").read_text(encoding="utf-8"))
        temporal = pd.DataFrame.from_dict(temporal_values, orient="index").rename_axis("model").reset_index()
        temporal = temporal.rename(columns={"r2": "r2_temporal", "rmse": "rmse_temporal", "mae": "mae_temporal"})
        joined = model_metrics.merge(temporal, on="model")
        columns = [c for c in ["model", "r2_random", "rmse_random", "mae_random", "r2_temporal", "rmse_temporal", "mae_temporal"] if c in joined]
        display_frame(joined[columns], height=200)
        kpi_data = json.loads((ARTIFACT_DIR / "kpis.json").read_text(encoding="utf-8"))
        point = kpi_data["point"]["lead_time_reduction_pct"]
        samples = load_table("sensitivity.csv")
        st.markdown(f"#### Monte Carlo sensitivity · {len(samples)} draws")
        sensitivity_chart(samples, point)
        stability = load_table("sensitivity_stability.csv")
        stability_summary = json.loads((ARTIFACT_DIR / "sensitivity_summary.json").read_text(encoding="utf-8"))
        mean_agreement = stability_summary.get("mean_choice_agreement_pct", 0.0)
        st.markdown(f"#### Factory choice stability · mean agreement {mean_agreement:.1f}%")
        display_frame(stability, height=260)
        st.caption(
            f"Assumptions sampled: speed 400–1,600 km/day and cost 0.05–0.30 per unit per 1,000 km. "
            f"Configured point estimate: {point:.2f}%; simulated median: {samples.lead_time_reduction_pct.median():.2f}%; "
            f"5th–95th percentile: {samples.lead_time_reduction_pct.quantile(.05):.2f}%–{samples.lead_time_reduction_pct.quantile(.95):.2f}%. "
            "Recommendation stability is separate from the magnitude distribution; the validated report gives the cross-draw factory-choice agreement."
        )


if __name__ == "__main__":
    main()
