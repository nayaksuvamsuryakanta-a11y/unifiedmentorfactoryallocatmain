r"""Nassau Candy Distributor -- Factory reallocation dashboard.

Launch:  streamlit run app.py
Run from the project root with the virtual environment activated, e.g.

    venv Scripts\activate   (Windows)   or   source venv/bin/activate   (Unix)
    streamlit run app.py

Every stochastic step uses config.SEED (42): the train/holdout split, the 5-fold shuffled
KFold, the KMeans init and the bootstrap/Monte-Carlo loops. Counterfactual lead times are
computed from an assumption-driven distance delta, not from the predictive model, because the
best model's R2 is negative (see the Models page). That assumption is printed wherever it is
used.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (DEFAULT_TOP_N, FACTORY_COORDS, PRODUCT_FACTORY, RISK_HIGH, RISK_MEDIUM,
                    RISK_WEIGHT_DISTANCE_SHIFT, RISK_WEIGHT_LOW_SUPPORT,
                    RISK_WEIGHT_NEG_PROFIT, RISK_WEIGHT_PRED_VARIANCE, SEED, Assumptions,
                    _PRODUCT_DIVISION, resolve_factory)
from clustering import cluster_routes, congested_region_products
from data_prep import MIN_VALID_FRACTION, load_and_audit_safe
from models import load_model, train_and_persist
from simulation import (candidate_moves, compute_kpis, factory_optimiser_summary,
                        parse_components, score_products, sensitivity,
                        top_recommendations)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Nassau Candy Factory Reallocation",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COMPONENT_WEIGHTS = {
    "negProfit": RISK_WEIGHT_NEG_PROFIT,
    "predVariance": RISK_WEIGHT_PRED_VARIANCE,
    "lowSupport": RISK_WEIGHT_LOW_SUPPORT,
    "distanceShift": RISK_WEIGHT_DISTANCE_SHIFT,
}
RISK_LABELS = ["High", "Medium", "Low"]
RISK_COLOUR = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}


# ---------------------------------------------------------------------------
# Cached data and model loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Loading and auditing the source CSV...")
def load_source():
    """(frame, audit, error). On a fatal parse failure the frame is None."""
    return load_and_audit_safe()


@st.cache_resource(show_spinner="Training / loading the best predictive model...")
def load_best_model(df):
    """Returns (pipeline, model_name). Trains once and reuses the joblib artefact."""
    path = Path("artifacts/best_model.joblib")
    if not path.exists():
        _, name, pipe = train_and_persist(df)[:3]
        return pipe, name
    return load_model(path)[:2]


def _residual_std(df):
    """{product: residual_std}, global_std -- the predVariance input to the risk engine."""
    path = Path("artifacts/per_product_residual_std.csv")
    if path.exists():
        resid = pd.read_csv(path)
    else:
        _, _, _, resid, _gstd = train_and_persist(df)
    return resid.set_index("product")["residual_std"].to_dict(), float(resid["residual_std"].max())


@st.cache_data(ttl=3600, show_spinner="Clustering factory -> region -> ship-mode lanes...")
def cluster_routes_cached(df):
    return cluster_routes(df)


@st.cache_data(ttl=3600, show_spinner="Detecting congested region-product pairs...")
def congested_cached(df):
    return congested_region_products(df)


@st.cache_data(show_spinner="Scoring alternative factories...")
def score_for(speed_weight: float, df):
    residual_std, global_std = _residual_std(df)
    return score_products(df, speed_weight=speed_weight,
                          residual_std=residual_std, global_std=global_std)


@st.cache_data(show_spinner="Running Monte-Carlo scenario sensitivity...")
def sensitivity_cached(df):
    return sensitivity(df)


# ---------------------------------------------------------------------------
# Shared UI
# ---------------------------------------------------------------------------

def dark_template(fig):
    """Match streamlit's dark theme so charts do not render on a white card."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e0e0e",
                      plot_bgcolor="#1c1c1c", font=dict(color="#f5f5f5"))
    return fig


def render_assumption_badge():
    """Every counterfactual in this app is assumption driven, not model forecast.

    The label names the assumption set; the hash is over its numeric values, so a change to
    freight speed, freight rate or the capability-gap penalty changes the hash and is visible
    on every page.
    """
    st.badge("assumption:A1", icon="🧪")
    st.caption(f"Lead-time and cost deltas come from `assumption:A1` "
               f"(version `{Assumptions.assumption_version()[:16]}…`). "
               "The predictive model is not used for counterfactuals: its R² is negative.")


def render_sidebar(df):
    """Always-visible sidebar. The four controls are session-state keyed, so a change on any
    page is reflected everywhere and survives page navigation."""
    with st.sidebar:
        st.header("🎛 Optimization controls", divider=True)
        render_assumption_badge()

        products = ["All"] + sorted(df["product_name"].unique().tolist())
        st.selectbox("Product selector", products, index=0, key="sel_product")

        regions = sorted(df["region"].unique().tolist())
        st.multiselect("Region selector", regions, default=regions, key="sel_regions")

        ship_modes = sorted(df["ship_mode"].unique().tolist())
        st.multiselect("Ship Mode filter", ship_modes, default=ship_modes, key="sel_modes")

        st.slider("Optimization priority", 0.00, 1.00, 0.50, 0.05,
                  help=("Balances lead-time reduction against profit impact. "
                        "Risk reduction keeps its own fixed weight (0.30)."),
                  key="opt_slider")

        filtered = df[df["region"].isin(st.session_state.sel_regions)
                      & df["ship_mode"].isin(st.session_state.sel_modes)]
        if st.session_state.sel_product != "All":
            filtered = filtered[filtered["product_name"] == st.session_state.sel_product]

        st.metric("Filtered order lines", f"{len(filtered):,}")
        st.caption(f"{len(df):,} order lines in the cleaned source.")
        st.divider()
        st.caption("Nassau Candy Factory Reallocation  ·  CodeAlpha internship")
        return (st.session_state.sel_product, st.session_state.sel_regions,
                st.session_state.sel_modes, st.session_state.opt_slider, filtered)


def render_error_state(audit, err):
    """Hard error below 95% valid rows; otherwise a warning with parse diagnostics."""
    valid = float(audit.get("valid_fraction", 0.0))
    fatal = err is not None or valid < MIN_VALID_FRACTION
    (st.error if fatal else st.warning)("⚠️ The source file could not be used as-is." if fatal
                                       else "⚠️ The source file has parse defects.")
    if err:
        st.code(err, language="text")
    st.markdown("| Metric | Value |\n|---|---|")
    st.markdown(f"| Input rows | {audit.get('input_rows', 0):,} |")
    if audit.get("stray_header_rows"):
        st.markdown(f"| Stray duplicate header rows | {audit['stray_header_rows']} |")
    if audit.get("blank_rows"):
        st.markdown(f"| Blank rows | {audit['blank_rows']} |")
    st.markdown(f"| Exact duplicate rows dropped | {audit.get('duplicate_rows', 0):,} |")
    st.markdown(f"| Unparseable Order Dates | {audit.get('unparseable_order_dates', 0)} |")
    st.markdown(f"| Unparseable Ship Dates | {audit.get('unparseable_ship_dates', 0)} |")
    st.markdown(f"| Rows with unparseable financials | "
                f"{audit.get('unparseable_financial_rows', 0)} |")
    st.markdown(f"| **Valid rows** | **{valid * 100:.1f}%** "
                f"(threshold {MIN_VALID_FRACTION * 100:.0f}%) |")
    reasons = audit.get("parse_error_reasons") or []
    if reasons:
        st.markdown("**First five parse-error reasons**")
        for reason in reasons:
            st.markdown(f"- {reason}")
    if fatal:
        st.stop()


def _page_slug(title):
    """Turn a page title into a unique, URL-safe route path.

    `functools.partial` hides the callable name from Streamlit, so every partial-wrapped page
    would otherwise resolve to the empty route and collide with `st.navigation`'s uniqueness
    check. Passing `url_path` explicitly pins each page to a distinct pathname.
    """
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def kpi_header(scored, best_model_r2):
    """The four header KPIs with bootstrap intervals.

    Coverage is measured against the whole catalogue -- products with no distance-reducing
    alternative keep an explicit "retain" row -- so the header agrees with the recommendation
    table. `compute_kpis` defaults `evaluable_products` to the full product count, so we pass it
    explicitly only to be unambiguous.
    """
    kpis = compute_kpis(scored, int(scored["product"].nunique()), best_model_r2)
    p, ci = kpis["point"], kpis["ci"]
    specs = [
        ("Lead Time Reduction", p["lead_time_reduction_pct"], ci["lead_time_reduction_pct"],
         "%", "share of baseline lead days saved across recommendations"),
        ("Profit Impact Stability", p["profit_stability_pct"], ci["profit_stability_pct"],
         "%", "share of recommendations that do not erode profit"),
        ("Scenario Confidence", p["scenario_confidence"], ci["scenario_confidence"], "%",
         "best-model 5-fold cross-validated R2"),
        ("Recommendation Coverage", p["coverage_pct"], ci["coverage_pct"], "%",
         "products with a recommended distance-reducing alternative"),
    ]
    for col, (label, val, interval, unit, sub) in zip(st.columns(4), specs):
        with col:
            lo, hi = interval
            st.metric(label, f"{val:.1f}{unit}", f"{lo:.1f}{unit} – {hi:.1f}{unit} (90% bootstrap CI)")
            st.caption(sub)
    return p


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_recommendations(df, scored, best_model_r2):
    st.title("Module 3: Recommendation Dashboard")
    st.markdown("Ranked factory reassignment for the catalogue. Recommendations are scored from "
                "measured distance deltas, not from the predictive model, because the model "
                "carries no usable lead-time signal (see the Models page).")

    product, regions, modes, speed, filtered = render_sidebar(df)
    if filtered.empty:
        st.warning("No order lines match the selected Region and Ship Mode filters; relax them "
                   "to see recommendations.")
        st.stop()
    scored = score_for(speed, filtered)

    st.subheader("KPI header")
    kpi_header(scored, best_model_r2)

    recs = top_recommendations(scored, DEFAULT_TOP_N)
    if product != "All":
        recs = recs[recs["product"] == product]

    st.subheader(f"Top-{DEFAULT_TOP_N} recommendations")
    st.caption(f"Optimization priority (lead-time weight) = {speed:.2f}. "
               "Risk reduction carries a fixed weight of 0.30.")
    display = recs[[
        "product", "current_factory", "candidate_factory", "orders", "units",
        "lead_gain_days", "delta_distance_km", "profit_impact", "risk", "risk_level",
        "confidence", "score", "gain_basis", "assumption_version",
    ]].copy()
    for col in ["lead_gain_days", "profit_impact", "risk", "score"]:
        display[col] = display[col].round(3)
    display["delta_distance_km"] = display["delta_distance_km"].round(1)
    display["assumption_version"] = display["assumption_version"].str.slice(0, 16)
    st.dataframe(display, hide_index=True,
                 column_config={
                     "orders": st.column_config.Column("Order count"),
                     "units": st.column_config.Column("Units"),
                     "risk_level": st.column_config.Column(
                         "Risk level",
                         help="High > 0.50, Medium > 0.25, Low otherwise"),
                     "gain_basis": st.column_config.Column(
                         "Provenance",
                         help="The assumption set that produced every lead-time delta"),
                     "assumption_version": st.column_config.Column(
                         "Assumption version",
                         help="sha256 of the assumption values that produced these deltas"),
                 })

    csv = recs.to_csv(index=False).encode()
    st.download_button("📥 Export recommendations (CSV)", csv, "recommendations.csv",
                       "text/csv")

    distinct = sorted(recs["candidate_factory"].unique())
    st.caption(f"Distinct factories in the Top-{DEFAULT_TOP_N}: {', '.join(distinct)} "
               f"({len(distinct)} of 5).")


def page_factory_optimizer(df, scored):
    st.title("Module 1: Factory Optimization Simulator")
    st.markdown("Every factory, its current inbound portfolio and the lead-time reduction it "
                "could absorb from distance-reducing alternatives.")

    _, _, _, _, filtered = render_sidebar(df)
    if filtered.empty:
        st.warning("No order lines match the selected Region and Ship Mode filters; relax them "
                   "to continue.")
        st.stop()
    scored = score_for(st.session_state.opt_slider, filtered)

    summary = factory_optimiser_summary(df, scored)
    st.subheader("Factory portfolio and inbound potential")
    st.dataframe(summary.round(3), hide_index=True)

    st.subheader("Mean lead-time reduction available per candidate factory")
    fig = px.bar(summary, x="factory", y="mean_candidate_lead_reduction_days",
                 text="mean_candidate_lead_reduction_days",
                 labels={"factory": "Factory",
                         "mean_candidate_lead_reduction_days": "Mean lead reduction (days)"})
    fig.update_traces(texttemplate="%{text:.1f}d", textposition="outside")
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide",
                      yaxis=dict(range=[0, max(summary["mean_candidate_lead_reduction_days"].max() * 1.2, 1)]))
    dark_template(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Current mean lead time by factory")
    lead_by = summary.dropna(subset=["mean_lead_days"])
    fig = px.bar(lead_by, x="factory", y="mean_lead_days", text="mean_lead_days",
                 labels={"factory": "Factory", "mean_lead_days": "Mean lead time (days)"})
    fig.update_traces(texttemplate="%{text:.0f}d", textposition="outside")
    dark_template(fig)
    st.plotly_chart(fig, use_container_width=True)


def page_what_if(df, scored):
    st.title("Module 2: What-If Scenario Analysis")
    st.markdown("Monte-Carlo over the two freight assumptions. Every numeric that turns a "
                "distance delta into a lead-time or cost delta is an **assumption**, not a "
                "fitted coefficient.")

    _, _, _, _, filtered = render_sidebar(df)
    if filtered.empty:
        st.warning("No order lines match the selected Region and Ship Mode filters; relax them "
                   "to continue.")
        st.stop()
    scored = score_for(st.session_state.opt_slider, filtered)

    with st.expander("Non-empirical assumptions used below", expanded=False):
        st.markdown("| Assumption | Value |")
        st.markdown("|---|---:|")
        for name, val in [("Freight speed (km/day)", Assumptions.freight_speed_km_day),
                          ("Freight cost per unit per 1000 km", Assumptions.freight_cost_per_unit_per_1000km),
                          ("Capability gap penalty", Assumptions.capability_gap_penalty),
                          ("Capability transition multiple", Assumptions.capability_transition_multiple),
                          ("Minimum orders for recommendation", Assumptions.min_orders_for_recommendation),
                          ("Minimum material days", Assumptions.min_material_days),
                          ("Capacity multiplier", Assumptions.capacity_multiplier)]:
            st.markdown(f"| {name} | {val} |")

    mc, _choice_log = sensitivity_cached(df)
    kpi_cols = ["lead_time_reduction_pct", "profit_stability_pct", "coverage_pct"]
    st.subheader("KPI stability across 300 assumption draws")
    melted = mc.melt("iteration", value_vars=kpi_cols, varname="kpi", value_name="value")
    fig = px.line(melted, x="iteration", y="value", color="kpi",
                  title="KPI value under sampled freight assumptions")
    dark_template(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("KPI distribution summary")
    st.dataframe(mc[kpi_cols].describe().T.round(3), use_container_width=True)

    st.subheader("Per-product sensitivity")
    product = st.selectbox("Product", sorted(df["product_name"].unique()), key="whatif_product")
    sub = scored[scored["product"] == product].sort_values("score", ascending=False)
    if sub.empty:
        st.write("No scored rows for this product.")
        return
    st.dataframe(sub[[
        "current_factory", "candidate_factory", "orders", "lead_gain_days",
        "delta_distance_km", "profit_impact", "risk", "risk_level", "confidence", "score",
    ]].round(3).sort_values("score", ascending=False), hide_index=True)


def page_risk_impact(df, scored):
    st.title("Module 4: Risk & Impact Panel")
    st.markdown("Risk is a weighted blend of four independent signals. The weights are fixed by "
                "specification, not tuned.")

    _, _, _, _, filtered = render_sidebar(df)
    if filtered.empty:
        st.warning("No order lines match the selected Region and Ship Mode filters; relax them "
                   "to continue.")
        st.stop()
    scored = score_for(st.session_state.opt_slider, filtered)

    st.subheader("Risk engine composition")
    for col, (k, w) in zip(st.columns(4), RISK_COMPONENT_WEIGHTS.items()):
        with col:
            st.metric(k, f"{w:.2f}")
    st.caption("risk = 0.40·negProfit + 0.30·predVariance + 0.20·lowSupport + 0.10·distanceShift  |  "
               "High > 0.50, Medium > 0.25")

    counts = scored["risk_level"].value_counts().reindex(RISK_LABELS, fill_value=0)
    st.subheader("Risk distribution")
    for col, lvl in zip(st.columns(3), RISK_LABELS):
        with col:
            st.metric(lvl, f"{int(counts[lvl])} recommendations",
                      f"{counts[lvl] / len(scored) * 100:.1f}%")
    fig = px.pie(values=counts.values, names=counts.index, color=counts.index,
                 color_discrete_map=RISK_COLOUR, title="Recommendation risk levels")
    dark_template(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Component breakdown by risk level")
    components = scored["risk_components"].apply(parse_components)
    comp_df = pd.DataFrame(components.tolist(), index=scored.index)
    comp_df["risk_level"] = scored["risk_level"]
    comp_mean = comp_df.groupby("risk_level")[list(RISK_COMPONENT_WEIGHTS)].mean().reindex(RISK_LABELS)
    st.dataframe(comp_mean.round(3).style.format("{:.3f}"), use_container_width=True)
    fig = px.bar(comp_mean.reset_index().melt("risk_level", varname="component", valuename="value"),
                 x="risk_level", y="value", color="component", barmode="group",
                 title="Mean risk component by level")
    dark_template(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Alerts")
    high = scored[scored["risk_level"] == "High"].sort_values("risk", ascending=False)
    if not high.empty:
        st.warning(f"{len(high)} recommendation(s) are high risk and should be reviewed.")
        disp = high[[
            "product", "current_factory", "candidate_factory", "risk", "risk_level",
            "lead_gain_days", "profit_impact",
        ]].copy()
        for k in RISK_COMPONENT_WEIGHTS:
            disp[k] = high["risk_components"].apply(lambda c, k=k: parse_components(c).get(k))
        st.dataframe(disp.round(3), hide_index=True)
    else:
        st.success("No high-risk recommendations. Risk is concentrated in the medium band.")

    medium = scored[scored["risk_level"] == "Medium"]
    with st.expander(f"{len(medium)} medium-risk recommendations"):
        if not medium.empty:
            st.dataframe(medium[[
                "product", "current_factory", "candidate_factory", "orders", "risk",
                "lead_gain_days", "profit_impact", "capability_gap",
            ]].round(3), hide_index=True)


def page_eda_models(df, scored, best_name, best_model_r2):
    st.title("EDA & Models")
    render_sidebar(df)
    scored = score_for(st.session_state.opt_slider, df)

    with st.expander("Assumptions converting distance into lead time and cost", expanded=False):
        st.markdown(f"- freight_speed_km_day = {Assumptions.freight_speed_km_day}")
        st.markdown(f"- freight_cost_per_unit_per_1000km = {Assumptions.freight_cost_per_unit_per_1000km}")
        st.markdown("- Everything else that converts a measured distance into a lead-time or "
                    "cost delta lives in `config.Assumptions` and is non-empirical.")

    st.subheader("Lead-time distribution")
    fig = px.histogram(df, x="lead_time_days", nbins=40,
                       title="Naive lead time (Ship Date - Order Date), days")
    fig.add_vline(x=df["lead_time_days"].mean(), line_dash="dash",
                  line_annotation=dict(text=f"mean = {df['lead_time_days'].mean():.0f} d",
                                       font=dict(color="#f5f5f5")))
    dark_template(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Lead time by origin factory")
    by_factory = df.groupby("current_factory")["lead_time_days"].agg(
        ["count", "mean", "std"]).reset_index()
    st.dataframe(by_factory.round(2), hide_index=True)
    fig = px.box(df, x="current_factory", y="lead_time_days", points="outliers",
                 title="Lead-time spread by origin factory")
    dark_template(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Region x Ship Mode mean lead time")
    pivot = df.pivot_table(values="lead_time_days", index="region", columns="ship_mode",
                           aggfunc="mean").reindex(["Atlantic", "Gulf", "Interior", "Pacific"])
    st.dataframe(pivot.round(1).style.format("{:.1f}"), use_container_width=True)
    fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="plasma",
                    title="Mean lead time (days) by region and ship mode",
                    labels=dict(color="Mean lead days"))
    dark_template(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model comparison (5-fold cross-validation, seed 42)")
    try:
        metrics = pd.read_csv("artifacts/model_metrics.csv")
    except Exception:
        metrics = pd.DataFrame()
    if not metrics.empty and "model" in metrics:
        disp = metrics.copy()
        disp["cv_rmse_std"] = disp["cv_rmse_std"].abs()
        disp["cv_mae_std"] = disp["cv_mae_std"].abs()
        disp["best"] = disp["model"] == best_name
        st.dataframe(disp.round(4), hide_index=True)
        st.info(f"**Selected: {best_name}.** All three estimators score at or below zero "
                "because the lead time in this file is a near-constant; the linear model is "
                "preferred on interpretability when accuracy is tied within tolerance.")
    else:
        st.write("No model metrics artefact found; run `python run_pipeline.py`.")

    st.subheader("Feature importance (permutation, best model)")
    try:
        imp = pd.read_csv("artifacts/feature_importance.csv")
    except Exception:
        imp = pd.DataFrame()
    if not imp.empty:
        fig = px.bar(imp, x="importance_mean", y="feature", orientation="h",
                     error_x="importance_std", title="Permutation importance (R2 drop)")
        dark_template(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Clustered factory-region-ship-mode lanes")
    try:
        routes = pd.read_csv("artifacts/routes.csv")
    except Exception:
        routes = pd.DataFrame()
    if not routes.empty:
        st.dataframe(routes[[
            "current_factory", "region", "ship_mode", "order_count", "mean_lead", "cluster",
            "cluster_label", "slow_route", "exposure",
        ]].round(2), hide_index=True)
        fig = px.scatter(routes, x="order_count", y="mean_lead", color="cluster",
                         hover_data=["current_factory", "region", "ship_mode",
                                     "cluster_label", "slow_route"],
                         title="Lanes by volume and mean lead time (KMeans clusters)")
        dark_template(fig)
        st.plotly_chart(fig, use_container_width=True)
        sel_k = int(routes["selected_k"].iloc[0]) if "selected_k" in routes else None
        sil = float(routes["silhouette_score"].iloc[0]) if "silhouette_score" in routes else None
        st.caption(f"Selected k from elbow + silhouette: {sel_k}, silhouette = {sil:.3f}")


def page_factory_intelligence(df):
    st.title("Factory Intelligence")
    render_sidebar(df)

    st.subheader("Network map (north-up)")
    fact = pd.DataFrame([{"factory": f, "lat": c[0], "lon": c[1]}
                         for f, c in FACTORY_COORDS.items()])
    fig = px.scatter_geo(fact, lat="lat", lon="lon", text="factory",
                         hover_name="factory", hover_data={"lat": ":.5f", "lon": ":.5f"})
    fig.update_traces(textposition="top right",
                      marker=dict(size=16, color="#F2B84B", line=dict(width=2, color="#fff")))
    fig.update_geos(projection_type="mercator", visible=False, showcoastlines=True,
                    coastlinecolor="#555", showland=True, landcolor="#181818",
                    showocean=True, oceancolor="#0e0e0e",
                    center=dict(lat=40.0, lon=-98.0), projection_scale=5.5,
                    lataxis=dict(showgrid=False), lonaxis=dict(showgrid=False))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#0e0e0e",
                      plot_bgcolor="#0e0e0e", font=dict(color="#f5f5f5"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Factory coordinates")
    coord_rows = [{"Factory": f, "Latitude": f"{lat:.6f}", "Longitude": f"{lon:.6f}"}
                  for f, (lat, lon) in FACTORY_COORDS.items()]
    st.dataframe(pd.DataFrame(coord_rows), hide_index=True)

    st.subheader("Product -> factory assignment grid")
    dist_cols = [f"distance_{f}" for f in FACTORY_COORDS]
    grid = df.groupby("product_name")[dist_cols].mean()
    grid["current_factory"] = df.groupby("product_name")["current_factory"].first()
    grid = grid.reset_index()
    grid["provenance"] = df.groupby("product_name")["factory_provenance"].first()
    st.dataframe(grid.round(1), hide_index=True)
    fallback = grid[grid["provenance"] == "division-fallback"]
    if not fallback.empty:
        st.info(f"{len(fallback)} product(s) resolve via the division-based fallback: "
                + ", ".join(fallback["product_name"].tolist()) + ".")
    unmapped = grid[grid["provenance"] == "unmapped"]
    if not unmapped.empty:
        st.warning(f"{len(unmapped)} product(s) could not be assigned to any factory: "
                   + ", ".join(unmapped["product_name"].tolist()) + ".")

    st.subheader("Kazookles (Phase 7)")
    st.markdown("`Kazookles` is deliberately absent from the explicit spec map. It resolves "
                "through the division-based fallback: its division's modal factory is the "
                "destination. That fallback is derived from the explicit map by majority vote, "
                "so it is data-driven rather than a second independently maintained table.")
    factory, prov = resolve_factory("Kazookles", _PRODUCT_DIVISION.get("Kazookles", "Other"))
    st.markdown(f"- Division: `Other` → nearest factory: **{factory}** "
                f"(provenance: `{prov}`)")
    st.caption(f"Rows resolved by the division fallback in this file: "
               f"{int((df['factory_provenance'] == 'division-fallback').sum())}.")

    st.subheader("Factory statistics")
    stats = df.groupby("current_factory").agg(
        order_lines=("row_id", "count"), orders=("order_id", "nunique"), units=("units", "sum"),
        products=("product_name", "nunique"), mean_lead=("lead_time_days", "mean"),
        mean_distance=("distance_km", "mean"), gross_profit=("gross_profit", "sum")).reset_index()
    st.dataframe(stats.round(2), hide_index=True)


def page_research_paper():
    st.title("Research Paper")
    path = Path("docs/research_paper.md")
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.write("docs/research_paper.md not found. Run `python run_pipeline.py` first.")


def page_executive_summary():
    st.title("Executive Summary")
    path = Path("docs/executive_summary.md")
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.write("docs/executive_summary.md not found. Run `python run_pipeline.py` first.")


def page_methodology():
    st.title("Methodology & Model Card")
    st.markdown("""
## 1. Data preparation

- **Dates.** Parsed strictly as DD-MM-YYYY with `errors="coerce"`. No date reconstruction is
  performed: the raw difference `(Ship Date - Order Date)` is used as the lead time.
- **Stray duplicate header rows** are detected by comparing the first cell of every row to the
  header and dropped. Blank rows and rows whose field count drifts from the header are diagnosed
  before parsing.
- **Deduplication** uses every column *except* `Row ID` as the identity key: `Row ID` is a
  synthetic line number, so two rows identical on all real fields are one order line.
- **Cost imputation:** missing `Cost` = `Sales - Gross Profit`.
- **Outlier removal:** a 3.0x IQR fence is applied to the four financial columns only
  (`Sales`, `Units`, `Gross Profit`, `Cost`). Lead time is **never** trimmed.
- **Distance** is the Haversine great-circle distance from the incumbent origin factory to the
  State/Province centroid. 61 centroids are embedded (51 US states incl. Alaska/Hawaii + DC + 10
  Canadian provinces/territories present in the source).
- **Origin factory:** the explicit spec map for 14 products, with a division-based fallback for
  unlisted products (Kazookles).

## 2. Modelling

Target: expected lead time given Product, Origin Factory, Destination Region and Ship Mode
(+ Units, Distance as support). Three estimators are trained on an 80/20 split (seed 42) and
5-fold shuffled cross-validation:

| Estimator | Configuration |
|---|---|
| Linear Regression | sklearn `LinearRegression` |
| Random Forest | `n_estimators=100`, `min_samples_leaf=3`, seed 42 |
| Gradient Boosting | `n_estimators=100`, seed 42 |

Metrics are RMSE, MAE and R2, reported for both hold-out and cross-validation. The estimator is
selected on accuracy *and* interpretability: estimators within `model_selection_tolerance` of the
best CV R2 trade accuracy for interpretability, so a linear model that ties a forest wins.

**Honest result:** all three estimators score at or below zero. The lead time in this file is a
near-constant (~1320 days), so there is no usable signal to predict. Counterfactual lead times are
therefore computed from an assumption-driven distance delta, and the assumption is stated wherever
it is used.

## 3. Route aggregation and scenario engine

- **KMeans** on Factory x Region x Ship Mode lanes, with elbow + silhouette selecting k.
- Slow routes (high mean lead) and high-exposure lanes are flagged; congested region-product pairs
  are those whose volume exceeds the threshold.
- **Scenario engine:** for each product x alternative factory, recompute distance, predict the new
  lead time as `baseline + delta_distance / freight_speed`, estimate the operational improvement
  and measure profit sensitivity. All conversion constants live in `config.Assumptions`.
- **Optimization ranking** blends lead-time reduction, profit impact and risk reduction. The lead
  / profit blend is driven live by the priority slider; risk reduction keeps its fixed weight.
- **Risk engine:** `risk = 0.40*negProfit + 0.30*predVariance + 0.20*lowSupport + 0.10*distanceShift`,
  with `predVariance` = hold-out residual standard deviation of the best model per product (NOT
  `|predicted - current|`). Thresholds: High > 0.50, Medium > 0.25.
- **KPIs:** Lead Time Reduction (%), Profit Impact Stability (%), Scenario Confidence Score
  (best-model R2), Recommendation Coverage (%), each with a bootstrap interval.

## 4. Model card

| Field | Value |
|---|---|
| Task | Regression: expected lead time from product, factory, region, ship mode |
| Training data | Cleaned source, 80/20 split, seed 42 |
| Validation | 5-fold shuffled CV, seed 42 |
| Features | product_name, current_factory, region, ship_mode, division, ship_mode_rank, region_rank, units, distance_km, sales |
| Preprocessing | median imputation + standard scaling numerics, one-hot categoricals in a sklearn Pipeline |
| Selected model | Linear Regression |
| Cross-validated R2 | negative (see the Models page) |
| Intended use | Diagnostic comparison of factory assignments, **not** prediction |
| Limitations | Lead time is a near-constant; `factory` is collinear with `product`; any alternative-factory prediction is out-of-sample extrapolation |
| Ethical note | Recommendations affect routing and cost, not staffing. Every distance-to-lead conversion is an explicit assumption, never presented as measured. |
""")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def main():
    with st.spinner("Loading and auditing the source data..."):
        df, audit, err = load_source()

    if err is not None or float(audit.get("valid_fraction", 0.0)) < MIN_VALID_FRACTION:
        render_error_state(audit, err)
        return
    if float(audit.get("valid_fraction", 0.0)) < 1.0:
        render_error_state(audit, err)

    with st.spinner("Loading the predictive model..."):
        best_model, best_name = load_best_model(df)

    # The model is cache_resource'd, so a warm rerun reuses it. Counting re-renders lets the
    # dashboard prove that a page refresh is not a retrain.
    st.session_state.setdefault("warm_reruns", 0)
    st.session_state.warm_reruns += 1
    st.caption(f"Warm reruns without retraining: {st.session_state.warm_reruns}")

    with st.spinner("Scoring alternative factories..."):
        scored = score_for(0.5, df)

    try:
        metrics = pd.read_csv("artifacts/model_metrics.csv")
        best_model_r2 = float(metrics.loc[metrics["model"] == best_name, "cv_r2_mean"].iloc[0])
    except Exception:
        best_model_r2 = 0.0

    PAGES = {
        "Optimization Lab": [
            st.Page(functools.partial(page_recommendations, df=df, scored=scored,
                                      best_model_r2=best_model_r2),
                    title="Recommendation Dashboard", icon="📋",
                    url_path=_page_slug("Recommendation Dashboard")),
            st.Page(functools.partial(page_factory_optimizer, df=df, scored=scored),
                    title="Factory Optimization Simulator", icon="🏭",
                    url_path=_page_slug("Factory Optimization Simulator")),
            st.Page(functools.partial(page_what_if, df=df, scored=scored),
                    title="What-If Scenario Analysis", icon="🔮",
                    url_path=_page_slug("What-If Scenario Analysis")),
            st.Page(functools.partial(page_risk_impact, df=df, scored=scored),
                    title="Risk & Impact Panel", icon="⚠️",
                    url_path=_page_slug("Risk & Impact Panel")),
        ],
        "Analysis": [
            st.Page(functools.partial(page_eda_models, df=df, scored=scored,
                                      best_model=best_model, best_name=best_name,
                                      best_model_r2=best_model_r2),
                    title="EDA & Models", icon="📊",
                    url_path=_page_slug("EDA & Models")),
            st.Page(functools.partial(page_factory_intelligence, df=df),
                    title="Factory Intelligence", icon="🗺",
                    url_path=_page_slug("Factory Intelligence")),
        ],
        "Documentation": [
            st.Page(page_research_paper, title="Research Paper", icon="📄"),
            st.Page(page_executive_summary, title="Executive Summary", icon="📝"),
            st.Page(page_methodology, title="Methodology & Model Card", icon="📐"),
        ],
    }
    pg = st.navigation(PAGES)
    pg.run()


if __name__ == "__main__":
    main()