"""Project configuration, reference tables and non-empirical scenario assumptions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = next(
    (p for p in (
        ROOT / "Nassau Candy Distributor (1).csv",
        ROOT / "Nassau Candy Distributor.csv",
        ROOT / "Nassau_Candy_Distributor.csv",
    ) if p.exists()),
    ROOT / "Nassau Candy Distributor.csv",
)
ARTIFACT_DIR = ROOT / "artifacts"
REFERENCE_DIR = ROOT / "data" / "reference"
ZIP_CENTROIDS_PATH = REFERENCE_DIR / "zip_centroids.csv"

# Single seed for every stochastic step: train/holdout split, KFold shuffling, KMeans
# init, bootstrap resampling and Monte Carlo.
SEED = 42
EARTH_RADIUS_KM = 6371.0088
DATE_FORMAT = "%d-%m-%Y"

SHIP_MODE_RANK = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 3}
REGION_ORDER = {"Atlantic": 0, "Gulf": 1, "Interior": 2, "Pacific": 3}

# Spec-supplied factory coordinates. Sugar Shack longitude is -96.181115 as given in the
# brief; an earlier working copy held -96.18115.
FACTORY_COORDS = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107),
}

# Explicit product->incumbent factory mapping from the spec. Kazookles is deliberately
# absent: unlisted products resolve through division_nearest_factory() (Phase 7 fallback).
PRODUCT_FACTORY = {
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
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
}

# Product -> division, taken from the source file's Division column. Used only to derive
# DIVISION_FACTORIES by majority vote, so the fallback table is data-driven rather than a
# second independently maintained mapping.
_PRODUCT_DIVISION = {
    "Wonka Bar - Nutty Crunch Surprise": "Chocolate",
    "Wonka Bar - Fudge Mallows": "Chocolate",
    "Wonka Bar - Scrumdiddlyumptious": "Chocolate",
    "Wonka Bar - Milk Chocolate": "Chocolate",
    "Wonka Bar - Triple Dazzle Caramel": "Chocolate",
    "Laffy Taffy": "Sugar",
    "SweeTARTS": "Sugar",
    "Nerds": "Sugar",
    "Fun Dip": "Sugar",
    "Fizzy Lifting Drinks": "Sugar",
    "Hair Toffee": "Sugar",
    "Everlasting Gobstopper": "Other",
    "Lickable Wallpaper": "Other",
    "Wonka Gum": "Other",
}

# Division -> modal factory, derived from PRODUCT_FACTORY by majority vote. This is what
# the Phase 7 fallback applies to unlisted products (e.g. Kazookles).
DIVISION_FACTORIES = {
    division: max(set(factories), key=list(factories).count)
    for division, factories in {
        division: [factory for name, factory in PRODUCT_FACTORY.items()
                   if _PRODUCT_DIVISION.get(name) == division]
        for division in sorted(set(_PRODUCT_DIVISION.values()))
    }.items()
}

# State/province centroids. US values are area centroids of each state polygon; Canadian
# values are the same for the ten provinces/territories present in the source. Alaska and
# Hawaii are included so the centroid lookup never falls through to an unknown branch.
STATE_CENTROIDS = {
    "Alabama": (32.806671, -86.791130), "Alaska": (61.3707, -152.4044),
    "Arizona": (33.729759, -111.431221), "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564), "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371), "Delaware": (39.318523, -75.507141),
    "District of Columbia": (38.897438, -77.026817), "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074), "Hawaii": (21.0943, -157.4983),
    "Idaho": (44.240459, -114.478828), "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278), "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486), "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805), "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101), "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095), "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696), "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353), "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374), "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011), "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051), "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012), "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917), "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755), "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007), "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345), "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434), "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968), "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954453), "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
    "Alberta": (53.933271, -116.576504), "British Columbia": (53.726669, -127.647621),
    "Manitoba": (53.760861, -98.813873), "New Brunswick": (46.565314, -66.461914),
    "Newfoundland and Labrador": (53.135509, -57.660435),
    "Nova Scotia": (44.681987, -63.744311), "Ontario": (51.253775, -85.323214),
    "Prince Edward Island": (46.510712, -63.416814), "Quebec": (52.939916, -73.549136),
    "Saskatchewan": (52.939916, -106.450864),
}

_CANADIAN = frozenset([
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Nova Scotia", "Ontario",
    "Prince Edward Island", "Quebec", "Saskatchewan"])


def division_nearest_factory(division):
    """Phase 7 fallback: an unlisted product uses its division's modal factory."""
    return DIVISION_FACTORIES.get(division)


def resolve_factory(product_name, division):
    """Return (factory, provenance). Provenance is 'explicit' or 'division-fallback'."""
    factory = PRODUCT_FACTORY.get(product_name)
    if factory is not None:
        return factory, "explicit"
    fallback = division_nearest_factory(division)
    if fallback is not None:
        return fallback, "division-fallback"
    return None, "unmapped"


def normalize_product_name(name):
    """Canonicalise so 'Wonka Bar -Scrumdiddlyumptious' and
    'Wonka Bar - Scrumdiddlyumptious' hash to the same key."""
    return " - ".join(part.strip() for part in str(name).split("-"))


def is_canadian(state_province):
    return str(state_province) in _CANADIAN


class Assumptions:
    """Non-empirical inputs. Every value that converts a measured distance into a lead-time
    or cost delta lives here, and every dashboard page states that these are assumptions."""
    freight_speed_km_day = 900.0
    freight_cost_per_unit_per_1000km = 0.15
    # One-time destination-specific qualification/retooling outlay incurred when a factory
    # becomes a new source for a product (line validation, tooling, supplier re-qualification).
    # It is independent of haul length, which is what keeps the profit objective from being an
    # exact scalar multiple of the lead-time objective: without a term that varies per
    # candidate and is uncorrelated with delta_distance, min-max normalisation collapses the
    # two into one vector and the priority slider becomes an algebraic no-op.
    setup_freight_cost = {
        "Lot's O' Nuts": 2.2,
        "Wicked Choccy's": 1.5,
        "Sugar Shack": 0.0,
        "Secret Factory": 2.8,
        "The Other Factory": 1.9,
    }
    capability_gap_penalty = 0.20
    capability_transition_multiple = 2.0
    min_orders_for_recommendation = 3
    min_material_days = 1.0
    capacity_multiplier = 1.5
    speed_weight = 0.5
    monte_carlo_iterations = 300
    model_selection_tolerance = 0.02
    risk_weight = 0.30
    iqr_factor = 3.0
    kpi_bootstrap_iterations = 500


# Risk engine weights (Phase 4.4) -- fixed by spec, not tuned.
RISK_WEIGHT_NEG_PROFIT = 0.40
RISK_WEIGHT_PRED_VARIANCE = 0.30
RISK_WEIGHT_LOW_SUPPORT = 0.20
RISK_WEIGHT_DISTANCE_SHIFT = 0.10
RISK_HIGH = 0.50
RISK_MEDIUM = 0.25

DEFAULT_TOP_N = 10

if __name__ == "__main__":
    print("DATA_PATH:", DATA_PATH)
    print("products in map:", len(PRODUCT_FACTORY))
    print("division -> modal factory:")
    print(json.dumps(DIVISION_FACTORIES, indent=2))
    print("STATE_CENTROIDS:", len(STATE_CENTROIDS),
          "US+DC:", len(STATE_CENTROIDS) - len(_CANADIAN), "CA:", len(_CANADIAN))
    print("unresolved fallback test:", resolve_factory("Kazookles", "Other"))