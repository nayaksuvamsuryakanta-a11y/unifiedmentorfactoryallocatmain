import numpy as np
import pandas as pd
from data_prep import repair_ship_dates, haversine_km, resolve_customer_centroid, prepare_data
from config import STATE_CENTROIDS, FACTORY_COORDS, PRODUCT_FACTORY, DATA_PATH
from simulation import compute_kpis, joint_optimize, score_products
from models import select_model
from clustering import congested_region_products

def test_date_repair_is_nonnegative_small_and_mode_ordered():
    orders=pd.Series(pd.to_datetime(["2024-01-01"]*4))
    ships=pd.Series(pd.to_datetime(["1900-01-01","1900-01-02","1900-01-05","1900-01-10"]))
    fixed=repair_ship_dates(orders,ships); days=(fixed-orders).dt.days
    assert days.min()>=0 and days.max()<30
    assert list(days)==sorted(days) and days[0]<days[1]<days[2]<days[3]

def test_supplied_data_date_repair_is_valid_for_every_retained_row():
    frame=prepare_data(DATA_PATH)
    assert frame.lead_time_days.ge(0).all()
    assert frame.lead_time_days.lt(30).all()
    means=frame.groupby("ship_mode").lead_time_days.mean()
    assert means["Same Day"]<means["First Class"]<means["Second Class"]<means["Standard Class"]

def test_haversine():
    assert haversine_km(0,0,0,0)==0
    assert abs(haversine_km(0,0,0,1)-111.195)<.2

def test_kpi_coverage_and_empty():
    result=compute_kpis(pd.DataFrame(),15,iterations=20)
    assert result["point"]["coverage_pct"]==0
    assert 0<=result["point"]["coverage_pct"]<=100

def test_state_fallback_table_available():
    assert STATE_CENTROIDS["Ontario"]==(51.253775,-85.323214)

def test_unknown_postal_uses_state_centroid():
    point,precision=resolve_customer_centroid("ZZZ-NOT-A-POSTAL-CODE","Ontario",{})
    assert point==STATE_CENTROIDS["Ontario"] and precision=="state"

def test_joint_optimizer_capacity_and_one_factory_per_product():
    products=list(PRODUCT_FACTORY)[:2]
    rows=[]
    for p in products:
        for f in FACTORY_COORDS:
            rows.append({"product":p,"candidate_factory":f,"score":1.0 if f==PRODUCT_FACTORY[p] else .5})
    recs=pd.DataFrame(rows)
    orders=pd.DataFrame({"product_name":products,"units":[10,10],"current_factory":[PRODUCT_FACTORY[p] for p in products]})
    result=joint_optimize(recs,orders,gated=False)
    assert result["product"].nunique()==len(products) and len(result)==len(products)
    limits=orders.groupby("current_factory").units.sum().reindex(FACTORY_COORDS,fill_value=0)*1.5
    used=result.groupby("factory").units.sum().reindex(FACTORY_COORDS,fill_value=0)
    assert (used<=limits+1e-9).all()

def test_joint_optimizer_falls_back_when_pulp_import_fails(monkeypatch):
    import builtins
    original_import=builtins.__import__
    def without_pulp(name,*args,**kwargs):
        if name=="pulp": raise ImportError("simulated missing optional solver")
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",without_pulp)
    products=list(PRODUCT_FACTORY)[:2]
    rows=[{"product":p,"candidate_factory":f,"score":1.0 if f==PRODUCT_FACTORY[p] else .5,
           "is_incumbent":f==PRODUCT_FACTORY[p],"sufficient_evidence":True,"materially_better":True}
          for p in products for f in FACTORY_COORDS]
    orders=pd.DataFrame({"product_name":products,"units":[10,10],"current_factory":[PRODUCT_FACTORY[p] for p in products]})
    result=joint_optimize(pd.DataFrame(rows),orders)
    assert result["product"].nunique()==len(products)
    assert result.attrs["solver"].startswith("SciPy MILP fallback")

def test_joint_optimizer_falls_back_when_pulp_solver_raises(monkeypatch):
    import pulp
    def failed_solve(*args,**kwargs): raise RuntimeError("simulated CBC failure")
    monkeypatch.setattr(pulp.LpProblem,"solve",failed_solve)
    products=list(PRODUCT_FACTORY)[:2]
    rows=[{"product":p,"candidate_factory":f,"score":1.0 if f==PRODUCT_FACTORY[p] else .5,
           "is_incumbent":f==PRODUCT_FACTORY[p],"sufficient_evidence":True,"materially_better":True}
          for p in products for f in FACTORY_COORDS]
    orders=pd.DataFrame({"product_name":products,"units":[10,10],"current_factory":[PRODUCT_FACTORY[p] for p in products]})
    result=joint_optimize(pd.DataFrame(rows),orders)
    assert result["product"].nunique()==len(products)
    assert result.attrs["solver"].startswith("SciPy MILP fallback")

def test_bootstrap_bounds_contain_point():
    rec=pd.DataFrame([dict(product="p",candidate_factory="f",sufficient_evidence=True,materially_better=True,lead_gain=2.,baseline_lead_days=5.,profit_impact=1.,confidence=60.,orders=20,score=1.)])
    result=compute_kpis(rec,1,iterations=100)
    for key,value in result["point"].items(): assert result["ci"][key][0]<=value<=result["ci"][key][1]

def _scoring_fixture():
    product=list(PRODUCT_FACTORY)[0]
    rows=[]
    for i in range(10):
        row={"product_name":product,"order_id":i,"units":1,"lead_time_days":2+i%2,"gross_profit":10.,"division":"Candy","current_factory":PRODUCT_FACTORY[product]}
        for j,factory in enumerate(FACTORY_COORDS): row[f"distance_{factory}"]=100.+j*300.
        row[f"distance_{PRODUCT_FACTORY[product]}"]=1000.
        row["distance_Wicked Choccy's"]=100.
        rows.append(row)
    return pd.DataFrame(rows)

def _assumptions(weight):
    from types import SimpleNamespace
    return SimpleNamespace(freight_speed_km_day=900.,freight_cost_per_unit_per_1000km=.15,
        capability_gap_penalty=.2,min_orders_for_recommendation=10,min_material_days=1.,
        capacity_multiplier=1.5,speed_weight=weight,risk_weight=.2,monte_carlo_iterations=1)

def test_score_products_keeps_one_incumbent_row_and_invariant_weights():
    frame=_scoring_fixture()
    speed=score_products(frame,_assumptions(1.),top_n=len(FACTORY_COORDS))
    profit=score_products(frame,_assumptions(0.),top_n=len(FACTORY_COORDS))
    assert speed.is_incumbent.sum()==1
    assert speed[speed.is_incumbent].candidate_factory.iloc[0]==PRODUCT_FACTORY[frame.product_name.iloc[0]]
    speed_order=speed[~speed.is_incumbent].sort_values("score",ascending=False).candidate_factory.tolist()
    profit_order=profit[~profit.is_incumbent].sort_values("score",ascending=False).candidate_factory.tolist()
    assert speed_order==profit_order
    a=speed.set_index("candidate_factory").score.sort_index().to_numpy()
    b=profit.set_index("candidate_factory").score.sort_index().to_numpy()
    assert np.allclose(a,b)

def test_breakeven_speed_reaches_material_lead_gain():
    frame=_scoring_fixture()
    scored=score_products(frame,_assumptions(.5),top_n=len(FACTORY_COORDS))
    move=scored[(~scored.is_incumbent)&(scored.candidate_factory=="Wicked Choccy's")].iloc[0]
    assert np.isclose(move.breakeven_freight_speed_km_day,-move.delta_distance_km/1.0)
    at_break_even=_assumptions(.5)
    at_break_even.freight_speed_km_day=move.breakeven_freight_speed_km_day
    verified=score_products(frame,at_break_even,top_n=len(FACTORY_COORDS))
    verified_move=verified[(~verified.is_incumbent)&(verified.candidate_factory=="Wicked Choccy's")].iloc[0]
    assert np.isclose(verified_move.lead_gain,at_break_even.min_material_days)
    non_improving=scored[scored.delta_distance_km>=0]
    assert non_improving.breakeven_freight_speed_km_day.isna().all()

def test_gated_joint_optimizer_keeps_incumbents_when_no_move_is_material():
    frame=_scoring_fixture()
    recs=score_products(frame,_assumptions(.5),top_n=len(FACTORY_COORDS))
    recs["materially_better"]=False
    result=joint_optimize(recs,frame,gated=True)
    assert result.factory.tolist()==[PRODUCT_FACTORY[frame.product_name.iloc[0]]]

def test_gated_joint_optimizer_treats_missing_or_nan_flags_as_not_passing():
    products=list(PRODUCT_FACTORY)[:2]
    rows=[]
    for p in products:
        for f in FACTORY_COORDS:
            incumbent=f==PRODUCT_FACTORY[p]
            rows.append({"product":p,"candidate_factory":f,"is_incumbent":incumbent,
                "score":1.0 if not incumbent else .1,
                "sufficient_evidence":True,"materially_better":True})
    recs=pd.DataFrame(rows)
    recs["sufficient_evidence"]=recs.sufficient_evidence.astype(object)
    recs["materially_better"]=recs.materially_better.astype(object)
    # NaN in either gate blocks that product's alternatives.
    alt=recs.candidate_factory!=recs["product"].map(PRODUCT_FACTORY)
    recs.loc[alt&(recs["product"]==products[0]),"sufficient_evidence"]=np.nan
    recs.loc[alt&(recs["product"]==products[1]),"materially_better"]=np.nan
    orders=pd.DataFrame({"product_name":products,"units":[10,10],
        "current_factory":[PRODUCT_FACTORY[p] for p in products]})
    result=joint_optimize(recs,orders,gated=True)
    assigned=result.set_index("product").factory
    assert all(assigned[p]==PRODUCT_FACTORY[p] for p in products)
    missing_column=joint_optimize(recs.drop(columns="sufficient_evidence"),orders,gated=True).set_index("product").factory
    assert all(missing_column[p]==PRODUCT_FACTORY[p] for p in products)

def test_joint_optimizer_capacity_respected_in_both_modes():
    products=list(PRODUCT_FACTORY)[:2]
    rows=[{"product":p,"candidate_factory":f,"score":1. if f==PRODUCT_FACTORY[p] else .5,
           "is_incumbent":f==PRODUCT_FACTORY[p],"sufficient_evidence":True,"materially_better":True}
          for p in products for f in FACTORY_COORDS]
    recs=pd.DataFrame(rows)
    orders=pd.DataFrame({"product_name":products,"units":[10,10],"current_factory":[PRODUCT_FACTORY[p] for p in products]})
    for gated in (True,False):
        result=joint_optimize(recs,orders,gated=gated)
        limits=orders.groupby("current_factory").units.sum().reindex(FACTORY_COORDS,fill_value=0)*1.5
        used=result.groupby("factory").units.sum().reindex(FACTORY_COORDS,fill_value=0)
        assert (used<=limits+1e-9).all()

def test_model_selection_prefers_interpretable_model_within_tolerance():
    report=pd.DataFrame({"feature_set":["Full feature set"]*4,
        "model":["Linear Regression","Ridge","Random Forest","Gradient Boosting"],
        "cv_r2_mean":[.78,.80,.81,.80]})
    temporal={"Linear Regression":{"r2":.78},"Ridge":{"r2":.80},
        "Random Forest":{"r2":.81},"Gradient Boosting":{"r2":.80}}
    selection,chosen=select_model(report,temporal,.02)
    assert chosen=="Ridge"
    assert selection.chosen.sum()==1

def test_risk_scores_include_capability_and_capacity_pressure():
    frame=_scoring_fixture()
    scored=score_products(frame,_assumptions(.5),top_n=len(FACTORY_COORDS))
    assert scored.loc[scored.is_incumbent,"risk_score"].eq(0).all()
    assert scored.risk_score.between(0,1).all()
    assert scored.loc[~scored.is_incumbent,"risk_reduction"].eq(-scored.loc[~scored.is_incumbent,"risk_score"]).all()

def test_region_product_congestion_requires_high_volume_and_lead():
    frame=pd.DataFrame({"region":["A"]*4+["B"]*2,"product_name":["P"]*6,
        "order_id":range(6),"lead_time_days":[5.,5.,5.,5.,1.,1.]})
    result=congested_region_products(frame)
    assert result.loc[result.region=="A","congested"].iloc[0]
    assert not result.loc[result.region=="B","congested"].iloc[0]
