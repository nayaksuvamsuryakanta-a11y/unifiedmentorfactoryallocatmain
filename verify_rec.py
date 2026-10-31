import json

import pandas as pd

from config import Assumptions
from data_prep import load_and_audit
from simulation import score_products, top_recommendations

df, audit = load_and_audit()
resid = (pd.read_csv("artifacts/per_product_residual_std.csv")
         .set_index("product")["residual_std"].to_dict())
moves = score_products(df, residual_std=resid,
                       global_std=float(pd.read_csv("artifacts/per_product_residual_std.csv")
                                        ["residual_std"].max()))
recs = top_recommendations(moves, 10)
reducing = moves[moves["delta_distance_km"] < 0]
print("products in file:", audit["product_count"])
print("products with >=1 distance-reducing alternative:", reducing["product"].nunique())
print("products covered by a recommendation row (incl. retain):", moves["product"].nunique())
print("reduce-only products:", sorted(reducing["product"].unique()))
print("retain products:", sorted(set(moves["product"]) - set(reducing["product"])))
print("top-10 distinct factories:", sorted(recs["candidate_factory"].unique()))
print("top-10 rows:", len(recs))
print("retain rows in top-10:", int(recs["is_retain"].sum()))
print("lead gain > 0 in top-10:", int((recs["lead_gain_days"] > 0).sum()))
print("risk levels:", recs["risk_level"].value_counts().to_dict())
print("all 4 risk components present in every row:",
      all(set(["negProfit", "predVariance", "lowSupport", "distanceShift"])
          <= set(r.keys()) for r in recs["risk_components"].map(json.loads)))