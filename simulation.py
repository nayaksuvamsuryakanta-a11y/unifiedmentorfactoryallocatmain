"""Factory-reallocation scenarios powered by the synthetic transit-time proxy."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_prep import FACTORY_COORDINATES, load_and_prepare_data
from models import FEATURES, MODEL_PATH

FREIGHT_COST_PER_UNIT_PER_1000_KM = 0.15


def load_proxy_model(model_path: str | Path = MODEL_PATH):
    """Load the pipeline produced by ``train_proxy_model``."""
    return joblib.load(model_path)


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    safe_weights = weights.clip(lower=1).astype(float)
    return float(np.average(values.astype(float), weights=safe_weights))


def simulate_product_reallocation(
    product_name: str,
    prepared_data: pd.DataFrame | None = None,
    model=None,
    model_path: str | Path = MODEL_PATH,
) -> pd.DataFrame:
    """Estimate transit-time and cost impact of each alternate factory.

    The returned rows compare the current factory with every other factory for the
    selected product. Predictions come from the explicitly synthetic proxy model,
    never from the corrupted observed ``Lead_Time`` target.
    """
    frame = prepared_data if prepared_data is not None else load_and_prepare_data()
    product_rows = frame.loc[frame["Product Name"] == product_name].copy()
    if product_rows.empty:
        raise ValueError(f"Product not found in prepared data: {product_name!r}")
    if model is None:
        model = load_proxy_model(model_path)

    current_factory = str(product_rows["Origin_Factory"].iloc[0])
    weights = product_rows["Units"]
    baseline_predictions = model.predict(product_rows[FEATURES])
    baseline_transit_days = _weighted_average(pd.Series(baseline_predictions), weights)
    baseline_distance = _weighted_average(product_rows["Distance_km"], weights)
    total_units = float(product_rows["Units"].sum())
    gross_profit = float(product_rows["Gross Profit"].sum())
    rows = []

    for factory, (factory_lat, factory_lon) in FACTORY_COORDINATES.items():
        if factory == current_factory:
            continue
        candidate = product_rows.copy()
        candidate["Origin_Factory"] = factory
        candidate["Distance_km"] = [
            _distance_for_customer(factory_lat, factory_lon, latitude, longitude)
            for latitude, longitude in zip(
                candidate["Customer_Latitude"], candidate["Customer_Longitude"]
            )
        ]
        predicted = model.predict(candidate[FEATURES])
        candidate_transit_days = _weighted_average(pd.Series(predicted), weights)
        distance_delta = _weighted_average(candidate["Distance_km"], weights) - baseline_distance
        shipping_penalty = (
            total_units * distance_delta / 1000 * FREIGHT_COST_PER_UNIT_PER_1000_KM
        )
        rows.append(
            {
                "Product Name": product_name,
                "Current Factory": current_factory,
                "Candidate Factory": factory,
                "Current Distance_km": baseline_distance,
                "Candidate Distance_km": _weighted_average(candidate["Distance_km"], weights),
                "Baseline Transit Days": baseline_transit_days,
                "Estimated New Transit Days": candidate_transit_days,
                "Transit Days Reduced": baseline_transit_days - candidate_transit_days,
                "Gross Profit": gross_profit,
                "Profit Impact": -shipping_penalty,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "Transit Days Reduced", ascending=False
    ).reset_index(drop=True)


def _distance_for_customer(
    factory_latitude: float,
    factory_longitude: float,
    customer_latitude: float,
    customer_longitude: float,
) -> float:
    """Calculate a candidate-factory distance without changing prepared source data."""
    from data_prep import haversine_km

    return haversine_km(
        factory_latitude,
        factory_longitude,
        customer_latitude,
        customer_longitude,
    )


if __name__ == "__main__":
    from models import train_proxy_model

    _, _, proxy = train_proxy_model()
    data = load_and_prepare_data()
    first_product = str(data["Product Name"].iloc[0])
    print(simulate_product_reallocation(first_product, data, proxy).to_string(index=False))
