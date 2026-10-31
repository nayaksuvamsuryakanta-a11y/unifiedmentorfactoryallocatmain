"""Route profiling on the Factory x Region x Ship Mode grain.

Elbow and silhouette jointly choose k; labels describe the dominant lane character of
each cluster. Slow routes and congested region/product pairs are flagged separately.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import SEED

CLUSTER_FEATURES = ["order_count", "mean_lead", "p90_lead", "lead_std",
                    "mean_distance", "mean_units"]


def cluster_routes(df, min_orders=10, k_max=8):
    """Aggregate to factory/region/ship mode, cluster lanes, and flag slow ones."""
    lanes = (df.groupby(["current_factory", "region", "ship_mode"], dropna=False)
             .agg(order_count=("order_id", "nunique"),
                  mean_lead=("lead_time_days", "mean"),
                  p90_lead=("lead_time_days", lambda s: s.quantile(0.9)),
                  lead_std=("lead_time_days", "std"),
                  mean_distance=("distance_km", "mean"),
                  mean_units=("units", "mean"),
                  mean_profit=("gross_profit", "mean"))
             .reset_index())
    lanes["lead_std"] = lanes["lead_std"].fillna(0)
    for col in CLUSTER_FEATURES:
        lanes[col] = lanes[col].replace([np.inf, -np.inf], np.nan)
    eligible = lanes["order_count"] >= min_orders

    chosen_k = 1
    silhouette = np.nan
    cluster_labels = None

    if eligible.sum() >= 3:
        X = StandardScaler().fit_transform(lanes.loc[eligible, CLUSTER_FEATURES])
        candidates = []
        for k in range(2, min(k_max + 1, int(eligible.sum()))):
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
            if len(set(km.labels_)) > 1:
                candidates.append((silhouette_score(X, km.labels_), k, km))
        if candidates:
            best_sil, chosen_k, best_km = max(candidates, key=lambda c: c[0])
            silhouette = float(best_sil)
            scaler = StandardScaler().fit(lanes.loc[eligible, CLUSTER_FEATURES])
            cluster_labels = best_km.predict(scaler.transform(lanes[CLUSTER_FEATURES]))
        else:
            cluster_labels = np.zeros(len(lanes))
    else:
        cluster_labels = np.zeros(len(lanes))

    lanes["cluster"] = cluster_labels
    if chosen_k > 1:
        stats = lanes[eligible].groupby("cluster").agg(
            volume=("order_count", "mean"), slow=("mean_lead", "mean")).reset_index()
        v_med, t_med = stats.volume.median(), stats.slow.median()
        label_map = {r.cluster: ("Slow" if r.slow >= t_med else "Fast") + " / "
                     + ("high volume" if r.volume >= v_med else "low volume")
                     for r in stats.itertuples()}
    else:
        label_map = {0: "Single route cluster"}
    lanes["cluster_label"] = lanes["cluster"].map(label_map).fillna("Unclustered lane")

    avg_lead = lanes["mean_lead"].mean()
    lanes["slow_route"] = (lanes["order_count"] >= min_orders) & (
        (lanes["mean_lead"] > avg_lead) | (lanes["p90_lead"] > avg_lead))
    lanes["exposure"] = (lanes[["mean_lead", "p90_lead"]].max(axis=1) - avg_lead
                         ).clip(lower=0) * lanes["order_count"]
    lanes["selected_k"] = chosen_k
    lanes["silhouette_score"] = silhouette
    lanes["min_orders_threshold"] = min_orders
    return lanes.sort_values("exposure", ascending=False).reset_index(drop=True)


def congested_region_products(df):
    """Region/product pairs with above-average volume and above-average lead time."""
    lanes = (df.groupby(["region", "product_name"], dropna=False)
             .agg(order_count=("order_id", "nunique"),
                  mean_lead=("lead_time_days", "mean"),
                  p90_lead=("lead_time_days", lambda s: s.quantile(0.9)),
                  mean_units=("units", "mean"), mean_profit=("gross_profit", "mean"))
             .reset_index())
    volume_cutoff = float(lanes["order_count"].quantile(0.75)) if len(lanes) else 0.0
    overall_lead = float(df["lead_time_days"].mean()) if len(df) else 0.0
    lanes["volume_threshold"] = volume_cutoff
    lanes["overall_mean_lead"] = overall_lead
    lanes["congested"] = (lanes["order_count"] > volume_cutoff) & (
        lanes["mean_lead"] > overall_lead)
    return lanes.sort_values(["congested", "order_count", "mean_lead"],
                             ascending=[False, False, False]).reset_index(drop=True)


if __name__ == "__main__":
    from data_prep import load_and_audit

    df, audit = load_and_audit()
    lanes = cluster_routes(df)
    print("selected_k:", lanes.selected_k.iloc[0], "silhouette:",
          lanes.silhouette_score.iloc[0])
    print(lanes[["current_factory", "region", "ship_mode", "order_count", "mean_lead",
                 "cluster", "cluster_label", "slow_route", "exposure"]]
          .sort_values("exposure", ascending=False).to_string(index=False))
    print("\ncongested:")
    print(congested_region_products(df).head(10).to_string(index=False))