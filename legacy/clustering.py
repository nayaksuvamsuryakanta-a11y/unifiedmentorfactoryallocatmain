"""Factory-region-division route profiling and cluster labels."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from config import SEED

def cluster_routes(df,min_orders=10):
    lanes=df.groupby(["current_factory","region","division"],dropna=False).agg(order_count=("order_id","nunique"),mean_lead=("lead_time_days","mean"),p90_lead=("lead_time_days",lambda x:x.quantile(.9)),lead_std=("lead_time_days","std"),mean_distance=("distance_km","mean"),mean_margin=("gross_margin", "mean")).reset_index()
    lanes["lead_std"]=lanes.lead_std.fillna(0)
    features=["order_count","mean_lead","p90_lead","lead_std","mean_distance","mean_margin"]
    # Unknown centroids and occasional missing margins should not discard an entire lane.
    lanes[features]=lanes[features].replace([np.inf,-np.inf],np.nan)
    lanes[features]=lanes[features].fillna(lanes[features].median()).fillna(0)
    fit=lanes.order_count>=min_orders
    if fit.sum()<3:
        lanes["cluster"]=0; lanes["cluster_label"]="Insufficient route history"; chosen_k=1; chosen_silhouette=np.nan
    else:
        X=StandardScaler().fit_transform(lanes.loc[fit,features])
        candidates=[]
        for k in range(2,min(6,int(fit.sum()))):
            km=KMeans(n_clusters=k,random_state=SEED,n_init=10).fit(X)
            if len(set(km.labels_))>1: candidates.append((silhouette_score(X,km.labels_),km))
        chosen_silhouette,km=max(candidates,key=lambda x:x[0]) if candidates else (np.nan,KMeans(n_clusters=1,random_state=SEED,n_init=10).fit(X))
        chosen_k=km.n_clusters
        scaler=StandardScaler().fit(lanes.loc[fit,features]); scaled=scaler.transform(lanes[features])
        lanes["cluster"]=km.predict(scaled)
        stats=lanes[fit].groupby("cluster").agg(volume=("order_count","mean"),slow=("mean_lead","mean"))
        vmed,tmed=stats.volume.median(),stats.slow.median()
        labelmap={i:("Slow" if r.slow>=tmed else "Fast")+" / "+("high volume" if r.volume>=vmed else "low volume") for i,r in stats.iterrows()}
        lanes["cluster_label"]=lanes.cluster.map(labelmap).fillna("Assigned to nearest route cluster")
    avg=lanes.mean_lead.mean(); lanes["problem_route"]= (lanes.order_count>=min_orders)&((lanes.mean_lead>avg)|(lanes.p90_lead>avg))
    lanes["exposure"]=(lanes[["mean_lead","p90_lead"]].max(axis=1)-avg).clip(lower=0)*lanes.order_count
    lanes["selected_k"]=chosen_k; lanes["silhouette_score"]=chosen_silhouette
    return lanes.sort_values("exposure",ascending=False)

def congested_region_products(df):
    """Find region/product pairs with unusually high volume and above-average lead."""
    lanes=df.groupby(["region","product_name"],dropna=False).agg(
        order_count=("order_id","nunique"),mean_lead=("lead_time_days","mean"),
        p90_lead=("lead_time_days",lambda x:x.quantile(.9))).reset_index()
    volume_cutoff=float(lanes.order_count.quantile(.75)) if len(lanes) else 0.0
    overall_lead=float(df.lead_time_days.mean()) if len(df) else 0.0
    lanes["volume_threshold"]=volume_cutoff
    lanes["overall_mean_lead"]=overall_lead
    lanes["congested"]=(lanes.order_count>volume_cutoff)&(lanes.mean_lead>overall_lead)
    return lanes.sort_values(["congested","order_count","mean_lead"],ascending=[False,False,False])
