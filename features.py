"""Leakage-safe feature assembly. Every learned transform (impute, scale, one-hot) is
fitted inside an sklearn Pipeline so nothing from the holdout set leaks into training."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Spec Phase 3 feature set: Product, Origin Factory, Destination Region, Ship Mode,
# with Units and Distance as support features.
NUMERIC = ["ship_mode_rank", "region_rank", "units", "distance_km", "sales"]
CATEGORICAL = ["product_name", "current_factory", "region", "ship_mode", "division"]


def make_preprocessor(numeric=NUMERIC, categorical=CATEGORICAL):
    """ColumnTransformer that imputes, scales and encodes within the pipeline."""
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )