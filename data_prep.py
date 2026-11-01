"""Prepare Nassau Candy order data for modeling and scenario analysis."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "Nassau Candy Distributor.csv"
EARTH_RADIUS_KM = 6371.0088

FACTORY_COORDINATES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.97107),
}

PRODUCT_FACTORIES = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Hair Toffee": "The Other Factory",
}

DIVISION_FACTORIES = {"Other": "Secret Factory"}
SHIP_MODE_MULTIPLIERS = {
    "Same Day": 0.1,
    "First Class": 1.0,
    "Second Class": 1.5,
    "Standard Class": 2.0,
}
RANDOM_STATE = 42

STATE_CENTROIDS = {
    "Alabama": (32.806671, -86.791130),
    "Alberta": (53.933271, -116.576504),
    "Arizona": (33.729759, -111.431221),
    "Arkansas": (34.969704, -92.373123),
    "British Columbia": (53.726669, -127.647621),
    "California": (36.116203, -119.681564),
    "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371),
    "Delaware": (39.318523, -75.507141),
    "District of Columbia": (38.897438, -77.026817),
    "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074),
    "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927),
    "Manitoba": (53.760861, -98.813873),
    "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374),
    "New Brunswick": (46.565314, -66.461914),
    "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051),
    "Newfoundland and Labrador": (53.135509, -57.660435),
    "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012),
    "Nova Scotia": (44.681987, -63.744311),
    "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917),
    "Ontario": (51.253775, -85.323214),
    "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755),
    "Prince Edward Island": (46.510712, -63.416814),
    "Quebec": (52.939916, -73.549136),
    "Rhode Island": (41.680893, -71.511780),
    "Saskatchewan": (52.939916, -106.450864),
    "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954453),
    "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
}


def haversine_km(
    latitude_1: float | np.ndarray,
    longitude_1: float | np.ndarray,
    latitude_2: float | np.ndarray,
    longitude_2: float | np.ndarray,
) -> float | np.ndarray:
    """Return great-circle distance in kilometers between two coordinates."""
    lat_1, lon_1, lat_2, lon_2 = (
        np.radians(np.asarray(value, dtype=float))
        for value in (latitude_1, longitude_1, latitude_2, longitude_2)
    )
    delta_latitude = lat_2 - lat_1
    delta_longitude = lon_2 - lon_1
    haversine = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(lat_1) * np.cos(lat_2) * np.sin(delta_longitude / 2) ** 2
    )
    distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    return float(distance) if np.ndim(distance) == 0 else distance


def _factory_for_product(product_name: str) -> str:
    """Return the incumbent factory for a product, failing clearly if unmapped."""
    try:
        return PRODUCT_FACTORIES[product_name]
    except KeyError as exc:
        raise ValueError(f"No factory mapping is configured for {product_name!r}") from exc


def _normalise_product_name(product_name: str) -> str:
    """Normalize spacing around the hyphen used in Wonka Bar product names."""
    return " - ".join(part.strip() for part in str(product_name).split("-"))


def load_and_prepare_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and add parsed dates, lead time, centroids, and distances."""
    frame = pd.read_csv(path)
    required_columns = {
        "Order Date",
        "Ship Date",
        "Product Name",
        "State/Province",
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame["Order Date"] = pd.to_datetime(
        frame["Order Date"], format="%d-%m-%Y", errors="raise"
    )
    frame["Ship Date"] = pd.to_datetime(
        frame["Ship Date"], format="%d-%m-%Y", errors="raise"
    )
    frame["Lead_Time"] = (frame["Ship Date"] - frame["Order Date"]).dt.days

    centroids = frame["State/Province"].map(STATE_CENTROIDS)
    unresolved = centroids.isna()
    if unresolved.any():
        states = sorted(frame.loc[unresolved, "State/Province"].unique())
        raise ValueError(f"No centroid is configured for states or provinces: {states}")
    frame["Customer_Latitude"] = centroids.map(lambda point: point[0])
    frame["Customer_Longitude"] = centroids.map(lambda point: point[1])

    frame["Product Name"] = frame["Product Name"].map(_normalise_product_name)
    frame["Origin_Factory"] = frame["Product Name"].map(PRODUCT_FACTORIES)
    fallback = frame["Division"].map(DIVISION_FACTORIES)
    frame["Origin_Factory"] = frame["Origin_Factory"].fillna(fallback)
    if frame["Origin_Factory"].isna().any():
        products = sorted(frame.loc[frame["Origin_Factory"].isna(), "Product Name"].unique())
        raise ValueError(f"No factory mapping is configured for products: {products}")
    frame["Distance_km"] = [
        haversine_km(
            FACTORY_COORDINATES[factory][0],
            FACTORY_COORDINATES[factory][1],
            latitude,
            longitude,
        )
        for factory, latitude, longitude in zip(
            frame["Origin_Factory"],
            frame["Customer_Latitude"],
            frame["Customer_Longitude"],
        )
    ]
    frame["Estimated_Transit_Days"] = np.maximum(
        1,
        np.rint(
            frame["Distance_km"] / 800
            * frame["Ship Mode"].map(SHIP_MODE_MULTIPLIERS)
            + np.random.default_rng(RANDOM_STATE).uniform(-0.5, 0.5, len(frame))
        ),
    ).astype(int)
    return frame


if __name__ == "__main__":
    prepared = load_and_prepare_data()
    print(prepared.head())
