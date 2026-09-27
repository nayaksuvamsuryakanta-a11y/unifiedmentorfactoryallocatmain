"""Predictive benchmarks and the core distance diagnostic."""
import numpy as np, pandas as pd
from scipy.stats import linregress
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from features import make_preprocessor
from config import SEED

def metrics(y,p): return {"r2":float(r2_score(y,p)),"rmse":float(mean_squared_error(y,p)**.5),"mae":float(mean_absolute_error(y,p))}
def train_models(df):
    features=[c for c in ["ship_mode_rank","order_month","order_quarter","order_weekday","order_weekend","sales","units","gross_profit","cost","distance_km","ship_mode","region","division","current_factory"] if c in df]
    X,y=df[features],df.lead_time_days
    tr,te=train_test_split(np.arange(len(df)),test_size=.2,random_state=SEED)
    estimators={"Linear Regression":LinearRegression(),"Ridge":Ridge(alpha=1.0),"Random Forest":RandomForestRegressor(n_estimators=100,min_samples_leaf=3,random_state=SEED,n_jobs=1),"Gradient Boosting":GradientBoostingRegressor(random_state=SEED,n_estimators=60)}
    rows=[]; fitted={}
    cv=KFold(5,shuffle=True,random_state=SEED)
    for name,est in estimators.items():
        pipe=Pipeline([("prep",make_preprocessor([c for c in features if c not in ("ship_mode","region","division","current_factory")],[c for c in features if c in ("ship_mode","region","division","current_factory")])),("model",est)])
        # Windows joblib process fan-out can create dozens of workers and spend
        # more time serializing the same frame than fitting these small models.
        # Serial CV is predictable and keeps the full pipeline runnable locally/CI.
        cv_scores=cross_validate(pipe,X,y,cv=cv,scoring={"r2":"r2","rmse":"neg_root_mean_squared_error","mae":"neg_mean_absolute_error"},n_jobs=1)
        pipe.fit(X.iloc[tr],y.iloc[tr]); pred=pipe.predict(X.iloc[te])
        rows.append({"model":name,**metrics(y.iloc[te],pred),"cv_r2_mean":float(cv_scores["test_r2"].mean()),"cv_r2_std":float(cv_scores["test_r2"].std()),"cv_rmse_mean":float(-cv_scores["test_rmse"].mean()),"cv_rmse_std":float(cv_scores["test_rmse"].std()),"cv_mae_mean":float(-cv_scores["test_mae"].mean()),"cv_mae_std":float(cv_scores["test_mae"].std())})
        fitted[name]=pipe
    report=pd.DataFrame(rows)
    best=fitted[report.sort_values("r2",ascending=False).iloc[0].model]
    imp=permutation_importance(best,X.iloc[te],y.iloc[te],n_repeats=10,random_state=SEED,scoring="r2",n_jobs=1)
    importance=pd.DataFrame({"feature":features,"importance_mean":imp.importances_mean,"importance_std":imp.importances_std}).sort_values("importance_mean",ascending=False)
    temporal={}
    years=sorted(df.order_date.dt.year.unique())
    if len(years)>=2:
        early=df.order_date.dt.year==years[0]; late=df.order_date.dt.year==years[-1]
        for name,model in fitted.items():
            if early.sum() and late.sum():
                m=clone(model).fit(X.loc[early],y.loc[early]); temporal[name]=metrics(y.loc[late],m.predict(X.loc[late]))
    diagnostic=distance_diagnostics(df,features)
    return report,importance,temporal,diagnostic

def distance_diagnostics(df,features):
    # Repeat the nested ablation with mixed ZIP/state distances and state-only
    # distances, so the precision comparison uses identical folds and features.
    base_blocks=[["ship_mode_rank"],["order_month","order_quarter","order_weekday","order_weekend"],["sales","units","gross_profit","cost"]]
    names=["Ship mode","+ calendar","+ order size","+ distance","+ region & factory"]
    rows=[]
    for distance_col,precision in (("distance_km","ZIP/state mixed"),("distance_state_km","State-only")):
        accumulated=[]
        blocks=base_blocks+[[distance_col],["region","division","current_factory"]]
        for name,block in zip(names,blocks):
            accumulated += block
            cat=[c for c in accumulated if c in ("ship_mode","region","division","current_factory")]
            num=[c for c in accumulated if c not in cat]
            pipe=Pipeline([("prep",make_preprocessor(num,cat)),("model",GradientBoostingRegressor(random_state=SEED,n_estimators=60))])
            score=cross_val_score(pipe,df[accumulated],df.lead_time_days,cv=KFold(5,shuffle=True,random_state=SEED),scoring="r2",n_jobs=1)
            rows.append({"geocoding_distance":precision,"feature_set":name,"cv_r2_mean":score.mean(),"cv_r2_std":score.std()})
    slopes=[]
    for mode,g in df.groupby("ship_mode"):
        z=g[["distance_km","lead_time_days"]].dropna()
        if len(z)>2:
            res=linregress(z.distance_km,z.lead_time_days); slopes.append({"ship_mode":mode,"slope_days_per_km":res.slope,"p_value":res.pvalue,"n":len(z)})
    # Ship-mode-only residual monotonicity, distance deciles.
    mode_means=df.groupby("ship_mode").lead_time_days.transform("mean")
    residual=df.lead_time_days-mode_means
    decile=pd.qcut(df.distance_km,10,duplicates="drop")
    residual_bins=pd.DataFrame({"distance_bin":decile.astype(str),"mean_residual_days":residual}).groupby("distance_bin",observed=False).mean().reset_index()
    # State dummy vs distance comparisons; categorical destination identity is a proxy test.
    from sklearn.preprocessing import OneHotEncoder
    state=df.state.fillna("Unknown").to_frame()
    dist=df[["distance_km"]]
    def state_compare(cols):
        enc=OneHotEncoder(handle_unknown="ignore")
        from sklearn.compose import ColumnTransformer
        prep=ColumnTransformer([("cat",enc,[c for c in cols if c in ("ship_mode","state")]),("num",make_preprocessor([c for c in cols if c=="distance_km"],[]),[c for c in cols if c=="distance_km"])],remainder="drop")
        p=Pipeline([("prep",prep),("model",GradientBoostingRegressor(random_state=SEED))])
        return float(cross_val_score(p,df[cols],df.lead_time_days,cv=KFold(5,shuffle=True,random_state=SEED),scoring="r2",n_jobs=1).mean())
    state_rows=[]
    for label,cols in [("Ship mode + state dummies",["ship_mode","state"]),("Ship mode + distance",["ship_mode","distance_km"]),("Ship mode + state + distance",["ship_mode","state","distance_km"])]:
        try: state_rows.append({"comparison":label,"cv_r2_mean":state_compare(cols)})
        except ValueError: state_rows.append({"comparison":label,"cv_r2_mean":np.nan})
    return {"ablation":pd.DataFrame(rows),"mode_slopes":pd.DataFrame(slopes),"residual_distance":residual_bins,"state_distance":pd.DataFrame(state_rows)}
