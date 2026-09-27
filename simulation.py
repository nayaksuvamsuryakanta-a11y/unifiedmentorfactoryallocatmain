"""Counterfactual ranking. Distance deltas are measured; time/cost deltas are assumptions."""
import numpy as np, pandas as pd
from config import FACTORY_COORDS, PRODUCT_FACTORY, Assumptions, SEED

def _norm(s):
    lo,hi=s.min(),s.max()
    return (s-lo)/(hi-lo) if hi>lo else pd.Series(.5,index=s.index)

def score_products(df,assumptions=Assumptions,top_n=3,reference_df=None):
    rows=[]
    for product,g in df.groupby("product_name"):
        current=PRODUCT_FACTORY.get(product)
        if not current: continue
        n=g.order_id.nunique(); baseline=float(np.average(g.lead_time_days,weights=g.units.clip(lower=1)))
        total_units=float(g.units.sum()); base_profit=float(g.gross_profit.sum()); cv=float(g.lead_time_days.std(ddof=0)/(g.lead_time_days.mean() or 1))
        evidence=min(1,np.log1p(n)/np.log1p(100)); stability=1/(1+max(cv,0))
        base_conf=100*(.65*evidence+.35*stability)
        division=g.division.mode().iloc[0]
        # Filtered what-if views can use a product/region/mode subset for observed
        # baseline metrics, while capability history must still use all orders.
        history=reference_df if reference_df is not None else df
        prod_div=history[history.division==division].groupby("current_factory").product_name.nunique()
        current_dist=float(np.average(g[f"distance_{current}"],weights=g.units.clip(lower=1)))
        candidates=[]
        for factory in FACTORY_COORDS:
            dist=float(np.average(g[f"distance_{factory}"],weights=g.units.clip(lower=1)))
            delta_dist=dist-current_dist # measured from supplied locations and customer coordinates
            delta_lead=delta_dist/assumptions.freight_speed_km_day # assumed freight-speed conversion
            delta_cost=total_units*delta_dist/1000*assumptions.freight_cost_per_unit_per_1000km # assumed cost
            cap_gap=factory!=current and prod_div.get(factory,0)==0
            # Confidence is evidence/stability minus the explicit capability-gap penalty.
            conf=max(0,min(99,base_conf-(100*assumptions.capability_gap_penalty if cap_gap else 0)))
            candidates.append(dict(product=product,current_factory=current,candidate_factory=factory,orders=n,units=total_units,baseline_lead_days=baseline,baseline_profit=base_profit,current_distance_km=current_dist,candidate_distance_km=dist,delta_distance_km=delta_dist,delta_lead_days=delta_lead,new_lead_days=baseline+delta_lead,delta_cost=delta_cost,profit_impact=-delta_cost,new_profit=base_profit-delta_cost,confidence=conf,capability_gap=bool(cap_gap),evidence_score=evidence,stability_score=stability))
        c=pd.DataFrame(candidates); c=c[c.candidate_factory!=current].copy()
        c["lead_gain"]=-c.delta_lead_days; c["profit_gain"]=c.profit_impact
        c["objective"]=(assumptions.speed_weight*_norm(c.lead_gain)+(1-assumptions.speed_weight)*_norm(c.profit_gain))
        c["score"]=c.objective*np.where(c.capability_gap,1-assumptions.capability_gap_penalty,1)*c.confidence/100
        c["sufficient_evidence"]=n>=assumptions.min_orders_for_recommendation
        c["materially_better"]=(c.lead_gain>=assumptions.min_material_days)&(c.profit_impact>=0)
        rows.extend(c.sort_values("score",ascending=False).head(top_n).to_dict("records"))
    return pd.DataFrame(rows)

def joint_optimize(recs,df,assumptions=Assumptions):
    """Solve one assignment per product under unit capacities; PuLP then SciPy MILP."""
    products=sorted(df.product_name.unique()); factories=list(FACTORY_COORDS)
    units=df.groupby("product_name").units.sum().to_dict()
    capacity=df.groupby("current_factory").units.sum().reindex(factories,fill_value=0)*assumptions.capacity_multiplier
    table=recs.sort_values("score",ascending=False).drop_duplicates(["product","candidate_factory"])
    options={(r.product,r.candidate_factory):float(r.score) for r in table.itertuples()}
    # incumbent always available with neutral objective
    incumbents=df.groupby("product_name").current_factory.agg(lambda s:s.dropna().mode().iloc[0] if s.notna().any() else None).to_dict()
    for p in products:
        incumbent=incumbents.get(p) or PRODUCT_FACTORY.get(p)
        if incumbent is None: raise ValueError(f"No incumbent factory configured for product {p!r}")
        options.setdefault((p,incumbent),0.0)
    try:
        import pulp
        prob=pulp.LpProblem("factory_assignment",pulp.LpMaximize)
        x={(p,f):pulp.LpVariable(f"x_{i}_{j}",cat="Binary") for i,(p,f) in enumerate(options) for j in [0]}
        prob += pulp.lpSum(options[k]*x[k] for k in x)
        for p in products: prob += pulp.lpSum(x[p,f] for f in factories if (p,f) in x)==1
        for f in factories: prob += pulp.lpSum(units[p]*x[p,f] for p in products if (p,f) in x)<=capacity[f]
        status=prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status]!="Optimal": raise RuntimeError(f"PuLP solve status {pulp.LpStatus[status]}")
        selected=[(p,f) for (p,f),v in x.items() if v.value()>.5]
        solver="PuLP/CBC"
    except Exception as exc:
        from scipy.optimize import milp, LinearConstraint, Bounds
        keys=list(options); c=-np.array([options[k] for k in keys]); A=[]; lb=[]; ub=[]
        for p in products: A.append([1 if k[0]==p else 0 for k in keys]); lb.append(1); ub.append(1)
        for f in factories: A.append([units[k[0]] if k[1]==f else 0 for k in keys]); lb.append(0); ub.append(capacity[f])
        result=milp(c,integrality=np.ones(len(keys)),bounds=Bounds(0,1),constraints=LinearConstraint(np.array(A),lb,ub),options={"time_limit":60})
        if not result.success: raise RuntimeError(f"PuLP failed ({exc}); SciPy MILP failed: {result.message}")
        selected=[keys[i] for i,v in enumerate(result.x) if v>.5]; solver=f"SciPy MILP fallback (PuLP unavailable/failed: {exc})"
    assignment=pd.DataFrame([{"product":p,"factory":f,"units":units[p]} for p,f in selected])
    assignment.attrs["solver"]=solver; assignment.attrs["capacity"]=capacity.to_dict()
    return assignment

def compute_kpis(recs,products,iterations=500,seed=SEED,order_frame=None,assumptions=Assumptions):
    cols=["product","candidate_factory","sufficient_evidence","materially_better","lead_gain","profit_impact","confidence","orders"]
    if recs is None or recs.empty: recs=pd.DataFrame(columns=cols)
    top=recs.sort_values("score",ascending=False).drop_duplicates("product") if "score" in recs else recs
    actionable=top[top.sufficient_evidence.fillna(False)&top.materially_better.fillna(False)] if len(top) else top
    catalogue=max(int(products),1)
    def calc(sample):
        if sample.empty:return {"lead_time_reduction_pct":0.,"profit_stability_pct":0.,"confidence":0.,"coverage_pct":0.}
        w=sample.orders.clip(lower=1)
        baseline=sample.baseline_lead_days if "baseline_lead_days" in sample else pd.Series(1.,index=sample.index)
        return {"lead_time_reduction_pct":float(np.average(sample.lead_gain.clip(lower=0),weights=w)/np.average(baseline,weights=w)*100),"profit_stability_pct":float((sample.profit_impact>=0).mean()*100),"confidence":float(sample.confidence.mean()),"coverage_pct":float(sample["product"].nunique()/catalogue*100)}
    point=calc(actionable); rng=np.random.default_rng(seed); distributions={k:[] for k in point}
    for _ in range(iterations):
        if order_frame is not None and not order_frame.empty:
            order_products=order_frame[["order_id","product_name"]].drop_duplicates()
            ids=order_products.order_id.drop_duplicates().to_numpy()
            sampled=rng.choice(ids,size=len(ids),replace=True)
            multiplicity=pd.Series(sampled).value_counts()
            per_order=order_products.groupby("order_id").product_name.agg(list)
            counts={}
            for oid,mult in multiplicity.items():
                for product in per_order.get(oid,[]): counts[product]=counts.get(product,0)+int(mult)
            s=top.copy()
            s["orders"]=s["product"].map(counts).fillna(0)
            s=s[(s.orders>=assumptions.min_orders_for_recommendation)&s.materially_better.fillna(False)]
        else:
            s=actionable.iloc[rng.integers(0,len(actionable),len(actionable))] if len(actionable) else actionable
        for k,v in calc(s).items(): distributions[k].append(v)
    # Keep the point estimate inside the reported percentile interval, including
    # when eligibility thresholds make bootstrap replicates discontinuous.
    ci={}
    for key,values in distributions.items():
        lo,hi=(np.percentile(values,[5,95]) if values else (point[key],point[key]))
        ci[key]=[float(min(lo,point[key])),float(max(hi,point[key]))]
    return {"point":point,"ci":ci}

def sensitivity(df,iterations=300):
    rng=np.random.default_rng(SEED); results=[]; choices=[]
    for i in range(iterations):
        a=type("Draw",(object,),dict(vars(Assumptions)))
        a.freight_speed_km_day=float(rng.uniform(400,1600)); a.freight_cost_per_unit_per_1000km=float(rng.uniform(.05,.30))
        rec=score_products(df,a,top_n=1); k=compute_kpis(rec,df.product_name.nunique(),iterations=0,order_frame=df,assumptions=a)["point"]
        results.append({"iteration":i,**k}); choices.append(rec.set_index("product").candidate_factory.to_dict() if len(rec) else {})
    return pd.DataFrame(results), choices
