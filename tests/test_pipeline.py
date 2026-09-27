import numpy as np
import pandas as pd
from data_prep import repair_ship_dates, haversine_km, resolve_customer_centroid, prepare_data
from config import STATE_CENTROIDS, FACTORY_COORDS, PRODUCT_FACTORY, DATA_PATH
from simulation import compute_kpis, joint_optimize

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
    result=joint_optimize(recs,orders)
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
    rows=[{"product":p,"candidate_factory":f,"score":1.0 if f==PRODUCT_FACTORY[p] else .5}
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
    rows=[{"product":p,"candidate_factory":f,"score":1.0 if f==PRODUCT_FACTORY[p] else .5}
          for p in products for f in FACTORY_COORDS]
    orders=pd.DataFrame({"product_name":products,"units":[10,10],"current_factory":[PRODUCT_FACTORY[p] for p in products]})
    result=joint_optimize(pd.DataFrame(rows),orders)
    assert result["product"].nunique()==len(products)
    assert result.attrs["solver"].startswith("SciPy MILP fallback")

def test_bootstrap_bounds_contain_point():
    rec=pd.DataFrame([dict(product="p",candidate_factory="f",sufficient_evidence=True,materially_better=True,lead_gain=2.,baseline_lead_days=5.,profit_impact=1.,confidence=60.,orders=20,score=1.)])
    result=compute_kpis(rec,1,iterations=100)
    for key,value in result["point"].items(): assert result["ci"][key][0]<=value<=result["ci"][key][1]
