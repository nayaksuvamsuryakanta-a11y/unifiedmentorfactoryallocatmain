import json

import pandas as pd

r = pd.read_csv("artifacts/recommendations.csv")
print("rows:", len(r))
print("distinct factories:", sorted(r["candidate_factory"].unique()))
print("lead_gain_days min:", r["lead_gain_days"].min())
print("delta_distance_km max:", r["delta_distance_km"].max())
print("products covered:", r["product"].nunique())
m = pd.read_csv("artifacts/model_metrics.csv")
print(m[["model", "r2", "rmse", "mae", "cv_r2_mean", "cv_r2_std", "cv_rmse_mean", "cv_mae_mean"]].to_string(index=False))
print("kpis:", json.dumps(json.load(open("artifacts/kpis.json"))["point"], indent=2))
print("risk levels:", r["risk_level"].value_counts().to_dict())