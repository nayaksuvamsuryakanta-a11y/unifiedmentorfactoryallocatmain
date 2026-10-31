"""Predictive models for expected lead time.

Three estimators are trained and cross-validated on the same split and folds so the
comparison is apples-to-apples. Because the raw lead time in this file is dominated by a
near-constant calendar offset, every R² is at or below zero; the selection rule still
applies and the honest values are reported rather than replaced by a fitted surrogate.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from config import ARTIFACT_DIR, SEED
from features import CATEGORICAL, NUMERIC, make_preprocessor

FEATURES = CATEGORICAL + NUMERIC
# Lower tier number means more interpretable.
INTERPRETABILITY_TIER = {"Linear Regression": 0, "Random Forest": 1, "Gradient Boosting": 2}
MODEL_ARTIFACT = ARTIFACT_DIR / "best_model.joblib"
METRICS_ARTIFACT = ARTIFACT_DIR / "model_metrics.csv"
IMPORTANCE_ARTIFACT = ARTIFACT_DIR / "feature_importance.csv"
RESIDUAL_STD_ARTIFACT = ARTIFACT_DIR / "per_product_residual_std.csv"


def metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def build_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                               random_state=SEED, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=SEED),
    }


SCORING = {
    "r2": "r2",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
}


def train_models(df, out_dir=ARTIFACT_DIR):
    """Fit, cross-validate and select. Returns (report, best_name, best_pipe)."""
    X = df[FEATURES]
    y = df["lead_time_days"]
    tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2,
                                      random_state=SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    fitted = {}
    for name, estimator in build_models().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", clone(estimator))])
        cv_res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=1)
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = pipe.predict(X.iloc[te_idx])
        holdout = metrics(y.iloc[te_idx], pred)
        rows.append({
            "model": name,
            "r2": holdout["r2"],
            "rmse": holdout["rmse"],
            "mae": holdout["mae"],
            "cv_r2_mean": float(cv_res["test_r2"].mean()),
            "cv_r2_std": float(cv_res["test_r2"].std()),
            "cv_rmse_mean": float(-cv_res["test_rmse"].mean()),
            "cv_rmse_std": float(-cv_res["test_rmse"].std()),
            "cv_mae_mean": float(-cv_res["test_mae"].mean()),
            "cv_mae_std": float(-cv_res["test_mae"].std()),
            "interpretability_tier": INTERPRETABILITY_TIER[name],
        })
        fitted[name] = pipe

    report = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)
    best_name = select_best(report)
    return report, best_name, fitted[best_name]


def select_best(report, tolerance=0.02):
    """Highest CV R² wins; estimators within the tolerance trade accuracy for
    interpretability, so a linear model that ties a forest is preferred."""
    best = report.cv_r2_mean.max()
    eligible = report[report.cv_r2_mean >= best - tolerance]
    return eligible.sort_values(["interpretability_tier", "cv_r2_mean"],
                                ascending=[True, False]).iloc[0].model


def persist_model(pipe, name, out_dir=ARTIFACT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_model.joblib"
    joblib.dump({"model": pipe, "name": name, "features": FEATURES}, path, compress=3)
    return path


def load_model(path=MODEL_ARTIFACT):
    """Load a joblib-persisted pipeline. Streamlit pages wrap this in st.cache_resource."""
    payload = joblib.load(path)
    return payload["model"], payload["name"], payload["features"]


def permutation_importance_for(pipe, X_te, y_te, n_repeats=10, seed=SEED):
    imp = permutation_importance(pipe, X_te, y_te, n_repeats=n_repeats,
                                 random_state=seed, scoring="r2", n_jobs=1)
    return (pd.DataFrame({"feature": FEATURES, "importance_mean": imp.importances_mean,
                          "importance_std": imp.importances_std})
            .sort_values("importance_mean", ascending=False))


def per_product_residual_std(pipe, X_te, y_te, feature_names):
    """Held-out residual standard deviation per product.

    This is the predVariance input to the risk engine: it measures how much the best
    model's predictions vary from reality for a given product, not how far a counterfactual
    sits from the current lead time.
    """
    resid = y_te.to_numpy() - pipe.predict(X_te)
    frame = pd.DataFrame({"product": X_te["product_name"].to_numpy(), "residual": resid})
    grouped = frame.groupby("product").residual.agg(["std", "count"]).rename(
        columns={"std": "residual_std", "count": "holdout_count"})
    global_std = float(frame["residual"].std(ddof=0))
    grouped["residual_std"] = grouped["residual_std"].where(
        grouped["holdout_count"] >= 2, global_std)
    grouped["residual_std"] = grouped["residual_std"].fillna(global_std)
    return grouped.reset_index(), global_std


def train_and_persist(df, out_dir=ARTIFACT_DIR):
    """Single entry point: train, select, persist and export all model artefacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X = df[FEATURES]
    y = df["lead_time_days"]
    tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2,
                                      random_state=SEED)
    report, best_name, best_pipe = train_models(df, out_dir)
    persist_model(best_pipe, best_name, out_dir)
    report.to_csv(out_dir / "model_metrics.csv", index=False)

    importance = permutation_importance_for(best_pipe, X.iloc[te_idx], y.iloc[te_idx],
                                        n_repeats=5)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)

    resid_frame, global_std = per_product_residual_std(best_pipe, X.iloc[te_idx],
                                                       y.iloc[te_idx], FEATURES)
    resid_frame.to_csv(out_dir / "per_product_residual_std.csv", index=False)
    return report, best_name, best_pipe, importance, resid_frame, global_std


if __name__ == "__main__":
    from data_prep import load_and_audit

    df, _ = load_and_audit()
    report, name, pipe, importance, resid, gstd = train_and_persist(df)
    print(report.to_string(index=False))
    print("\nbest:", name)
    print("\nfeature importance:\n", importance.to_string(index=False))
    print("\nper-product residual std (global fallback %.2f):\n" % gstd,
          resid.to_string(index=False))