"""Streamlit dashboard for factory reallocation and shipping optimization."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from data_prep import FACTORY_COORDINATES, load_and_prepare_data
from models import FEATURES, MODEL_PATH
from simulation import load_proxy_model, simulate_product_reallocation

DATA_PATH = Path(__file__).resolve().parent / "Nassau Candy Distributor.csv"

st.set_page_config(
    page_title="Nassau Candy Factory Optimization",
    page_icon="🍬",
    layout="wide",
)


@st.cache_data(show_spinner="Loading and preparing order data...")
def load_data() -> pd.DataFrame:
    return load_and_prepare_data(DATA_PATH)


@st.cache_resource(show_spinner="Loading the transit-time proxy model...")
def load_model():
    return load_proxy_model(MODEL_PATH)


def _filtered_product_data(
    data: pd.DataFrame,
    product: str,
    regions: list[str],
    ship_modes: list[str],
) -> pd.DataFrame:
    filtered = data[
        data["Product Name"].eq(product)
        & data["Region"].isin(regions)
        & data["Ship Mode"].isin(ship_modes)
    ].copy()
    if filtered.empty:
        raise ValueError("No order rows match the selected product and filters.")
    return filtered


def _scenario_data(data: pd.DataFrame, model, product: str) -> pd.DataFrame:
    return simulate_product_reallocation(product, data, model)


def _all_factory_transit_days(
    product_data: pd.DataFrame, model, scenarios: pd.DataFrame
) -> pd.DataFrame:
    current_factory = str(product_data["Origin_Factory"].iloc[0])
    baseline = float(model.predict(product_data[FEATURES]).mean())
    rows = [{"Factory": current_factory, "Estimated Transit Days": baseline}]
    rows.extend(
        {
            "Factory": row["Candidate Factory"],
            "Estimated Transit Days": row["Estimated New Transit Days"],
        }
        for _, row in scenarios.iterrows()
    )
    return pd.DataFrame(rows)


def _rank_scenarios(scenarios: pd.DataFrame, priority: float) -> pd.DataFrame:
    ranked = scenarios.copy()
    ranked["Score"] = (
        priority * ranked["Transit Days Reduced"]
        + (1 - priority) * ranked["Profit Impact"]
    )
    return ranked.sort_values("Score", ascending=False).reset_index(drop=True)


def _display_value(value: float) -> str:
    return f"{value:,.2f}"


def main() -> None:
    data = load_data()
    model = load_model()

    st.title("Factory Reallocation & Shipping Optimization")
    st.caption(
        "Nassau Candy Distributor · scenario recommendations powered by an "
        "estimated transit-time proxy model"
    )

    with st.sidebar:
        st.header("Optimization controls")
        products = sorted(data["Product Name"].unique())
        product = st.selectbox("Product selector", products)
        regions = st.multiselect(
            "Region selector",
            sorted(data["Region"].unique()),
            default=sorted(data["Region"].unique()),
        )
        ship_modes = st.multiselect(
            "Ship mode filter",
            sorted(data["Ship Mode"].unique()),
            default=sorted(data["Ship Mode"].unique()),
        )
        priority = st.slider(
            "Optimization priority",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0.0 prioritizes profit impact; 1.0 prioritizes transit-time reduction.",
        )

    try:
        product_data = _filtered_product_data(data, product, regions, ship_modes)
        scenarios = _scenario_data(product_data, model, product)
    except ValueError as exc:
        st.error(str(exc))
        return

    current_factory = str(product_data["Origin_Factory"].iloc[0])
    current_transit = float(model.predict(product_data[FEATURES]).mean())
    current_profit = float(product_data["Gross Profit"].sum())
    ranked = _rank_scenarios(scenarios, priority)
    all_factory_days = _all_factory_transit_days(product_data, model, scenarios)

    tabs = st.tabs(
        [
            "Factory Optimization Simulator",
            "What-If Scenario Analysis",
            "Recommendation Dashboard",
            "Risk & Impact Panel",
        ]
    )

    with tabs[0]:
        st.subheader(f"Factory optimization for {product}")
        metric_columns = st.columns(3)
        metric_columns[0].metric("Current factory", current_factory)
        metric_columns[1].metric("Current estimated transit", f"{current_transit:.2f} days")
        metric_columns[2].metric("Orders in selection", f"{len(product_data):,}")
        chart = px.bar(
            all_factory_days,
            x="Factory",
            y="Estimated Transit Days",
            color="Factory",
            title="Estimated transit days by factory",
            labels={"Estimated Transit Days": "Estimated transit days"},
        )
        chart.update_layout(showlegend=False)
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(
            scenarios[
                [
                    "Candidate Factory",
                    "Candidate Distance_km",
                    "Estimated New Transit Days",
                    "Transit Days Reduced",
                    "Profit Impact",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tabs[1]:
        st.subheader("What-if scenario analysis")
        candidate = st.selectbox(
            "Alternate Factory",
            scenarios["Candidate Factory"].tolist(),
            key="what_if_factory",
        )
        selected = scenarios.loc[scenarios["Candidate Factory"].eq(candidate)].iloc[0]
        metric_columns = st.columns(4)
        metric_columns[0].metric("Current transit", f"{current_transit:.2f} days")
        metric_columns[1].metric(
            "New transit", f"{selected['Estimated New Transit Days']:.2f} days",
            delta=f"{selected['Transit Days Reduced']:.2f} days",
        )
        metric_columns[2].metric("Current profit", f"${current_profit:,.2f}")
        metric_columns[3].metric(
            "New profit", f"${current_profit + selected['Profit Impact']:,.2f}",
            delta=f"${selected['Profit Impact']:,.2f}",
        )
        distance_chart = px.bar(
            pd.DataFrame(
                {
                    "Factory": [current_factory, candidate],
                    "Distance (km)": [
                        selected["Current Distance_km"],
                        selected["Candidate Distance_km"],
                    ],
                }
            ),
            x="Factory",
            y="Distance (km)",
            color="Factory",
            title="Current versus alternate factory distance",
        )
        distance_chart.update_layout(showlegend=False)
        st.plotly_chart(distance_chart, use_container_width=True)

    with tabs[2]:
        st.subheader("Ranked factory recommendations")
        top = ranked.iloc[0]
        st.success(
            f"Top recommendation: move **{product}** to **{top['Candidate Factory']}** "
            f"(score {_display_value(top['Score'])})."
        )
        st.caption(
            f"Score = ({priority:.1f} × transit reduction) + "
            f"({1 - priority:.1f} × profit impact)"
        )
        st.dataframe(
            ranked[
                [
                    "Candidate Factory",
                    "Transit Days Reduced",
                    "Profit Impact",
                    "Score",
                    "Candidate Distance_km",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tabs[3]:
        st.subheader("Risk & impact panel")
        negative = scenarios.loc[scenarios["Profit Impact"] < 0]
        if negative.empty:
            st.success("No evaluated factory move has a negative profit impact.")
        else:
            st.warning("These factory moves have a negative estimated profit impact:")
            st.dataframe(
                negative[["Candidate Factory", "Profit Impact", "Transit Days Reduced"]],
                use_container_width=True,
                hide_index=True,
            )
        st.info(
            "Note: Scenario simulations utilize a synthetic transit-time proxy model "
            "to account for source data date anomalies."
        )


if __name__ == "__main__":
    main()
