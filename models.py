"""Train lead-time and explicitly synthetic transit-time proxy models."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import load_and_prepare_data

RANDOM_STATE = 42
MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"
CATEGORICAL_FEATURES = ["Ship Mode", "Region", "Product Name", "Origin_Factory"]
NUMERICAL_FEATURES = ["Distance_km"]
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def _build_pipeline(regressor: object) -> Pipeline:
    return Pipeline(
        [("preprocessor", build_preprocessor()), ("regressor", regressor)]
    )


def _train_target(
    frame: pd.DataFrame,
    target: str,
    model_path: str | Path,
) -> tuple[pd.DataFrame, str, Pipeline]:
    missing = sorted(set(FEATURES + [target]).difference(frame.columns))
    if missing:
        raise ValueError(f"Prepared data is missing required columns: {missing}")
    X_train, X_test, y_train, y_test = train_test_split(
        frame[FEATURES],
        frame[target],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    fitted = {}
    rows = []
    for name, regressor in build_models().items():
        pipeline = _build_pipeline(regressor)
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        rows.append(
            {
                "Model": name,
                "RMSE": mean_squared_error(y_test, prediction) ** 0.5,
                "MAE": mean_absolute_error(y_test, prediction),
                "R2": r2_score(y_test, prediction),
            }
        )
        fitted[name] = pipeline
    metrics = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["Model"])
    best_pipeline = fitted[best_name]
    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, output_path)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Best model by R2: {best_name}")
    print(f"Model saved to: {output_path.resolve()}")
    return metrics, best_name, best_pipeline


def train_proxy_model(
    data_path: str | Path | None = None,
    model_path: str | Path = MODEL_PATH,
) -> tuple[pd.DataFrame, str, Pipeline]:
    """Train the dashboard proxy against explicitly synthetic transit days."""
    frame = load_and_prepare_data(data_path or Path(__file__).resolve().parent / "Nassau Candy Distributor.csv")
    print("Proxy model evaluation metrics (target: Estimated_Transit_Days):")
    result = _train_target(frame, "Estimated_Transit_Days", model_path)
    print(
        "⚠️ NOTE: This model predicts ESTIMATED TRANSIT DAYS, "
        "not the corrupted observed Lead_Time."
    )
    return result


def train_and_save_model(
    data_path: str | Path | None = None,
    model_path: str | Path = MODEL_PATH,
) -> tuple[pd.DataFrame, str, Pipeline]:
    """Train the observed Lead_Time model without filtering or rewriting its target."""
    frame = load_and_prepare_data(data_path or Path(__file__).resolve().parent / "Nassau Candy Distributor.csv")
    print("Observed Lead_Time evaluation metrics:")
    return _train_target(frame, "Lead_Time", model_path)


if __name__ == "__main__":
    train_proxy_model()
