"""Leakage-safe preprocessing; all learned transforms live in sklearn pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUMERIC=["ship_mode_rank","order_month","order_quarter","order_weekday","order_weekend","sales","units","gross_profit","cost","distance_km"]
CATEGORICAL=["ship_mode","region","division","current_factory"]
def make_preprocessor(numeric=NUMERIC,categorical=CATEGORICAL):
    return ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
                              ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)],remainder="drop")
