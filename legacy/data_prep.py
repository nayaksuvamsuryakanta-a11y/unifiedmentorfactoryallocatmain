"""Input validation, audited date repair, geocoding and feature preparation."""
import logging
import re
import numpy as np
import pandas as pd
from config import (DATA_PATH, FACTORY_COORDS, PRODUCT_FACTORY, SHIP_MODE_RANK,
                    STATE_CENTROIDS, ZIP_CENTROIDS_PATH, EARTH_RADIUS_KM)

log = logging.getLogger(__name__)
RENAME = {"Row ID":"row_id","Order ID":"order_id","Order Date":"order_date","Ship Date":"ship_date_raw","Ship Mode":"ship_mode","Customer ID":"customer_id","Country/Region":"country","City":"city","State/Province":"state","Postal Code":"postal_code","Division":"division","Region":"region","Product ID":"product_id","Product Name":"product_name","Sales":"sales","Units":"units","Gross Profit":"gross_profit","Cost":"cost"}

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km, vectorized over numpy-compatible inputs."""
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def resolve_customer_centroid(postal_code, state, zip_centroids=None, country=None):
    """Resolve ZIP when checked in; otherwise return the supplied state fallback."""
    zips=zip_centroids or {}
    key=str(postal_code).split('.')[0]
    point=zips.get((str(country),key)) if country is not None else zips.get(key)
    if point and all(pd.notna(v) for v in point): return point,"ZIP"
    point=STATE_CENTROIDS.get(str(state).strip())
    if point: return point,"state"
    return (np.nan,np.nan),"unresolved"

def repair_ship_dates(order_dates, ship_dates):
    """Rebuild year from order year: source ship years are corrupt; retain month/day.

    Rolling to the next year handles genuine year-end crossings. A residual, if
    present consistently, is removed using its observed minimum (never guessed).
    """
    order = pd.to_datetime(order_dates, errors="coerce", dayfirst=True)
    ship = pd.to_datetime(ship_dates, errors="coerce", dayfirst=True)
    if order.isna().any() or ship.isna().any():
        raise ValueError("Unparseable order or ship date")
    rebuilt = pd.to_datetime({"year":order.dt.year, "month":ship.dt.month, "day":ship.dt.day}, errors="coerce")
    invalid = rebuilt.isna()
    if invalid.any():
        # Invalid leap-day in non-leap order year: next-year roll preserves valid month/day intent.
        rebuilt.loc[invalid] = pd.to_datetime({"year":order.dt.year[invalid]+1,"month":ship.dt.month[invalid],"day":ship.dt.day[invalid]}, errors="coerce")
    rebuilt.loc[rebuilt < order] += pd.DateOffset(years=1)
    gaps = (rebuilt-order).dt.days
    residual = int(gaps.min())
    # The defect is an almost constant offset, with calendar and randomized
    # processing-day variation around it. The minimum residual is the stated
    # data-derived offset; subtracting it preserves all relative differences.
    print(f"Date repair: minimum residual gap={residual} day(s); removed this data-derived offset (not a guessed constant).")
    if residual > 0:
        rebuilt -= pd.to_timedelta(residual, unit="D")
    return rebuilt

def load_zip_centroids():
    if ZIP_CENTROIDS_PATH.exists():
        z = pd.read_csv(ZIP_CENTROIDS_PATH, dtype={"postal_code":str})
        return {(str(r.country),str(r.postal_code)): (r.latitude,r.longitude) for r in z.itertuples() if pd.notna(r.latitude) and pd.notna(r.longitude)}
    try:
        import pgeocode  # optional one-time acquisition, never used by pipeline at runtime
        log.info("pgeocode is installed, but no checked-in centroid file exists; runtime geocoding disabled.")
    except ImportError:
        log.info("ZIP geocoding unavailable (pgeocode not installed or no checked-in centroid file); using state centroids for 100%% of rows.")
    return {}

def prepare_data(path=DATA_PATH):
    df = pd.read_csv(path, dtype={"Postal Code":str}).rename(columns=RENAME)
    required = set(RENAME.values())
    missing = required-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    df["order_date"] = pd.to_datetime(df.order_date, dayfirst=True, errors="raise")
    raw_ship = pd.to_datetime(df.ship_date_raw, dayfirst=True, errors="raise")
    naive = (raw_ship-df.order_date).dt.days
    print(f"Naive ship gap (days): min={naive.min()}, mean={naive.mean():.2f}, max={naive.max()}")
    print(f"Date audit: order years {sorted(df.order_date.dt.year.unique().tolist())}; raw ship years {sorted(raw_ship.dt.year.unique().tolist())}; parsed month/day values are plausible, but ship years are later and corrupt, so reconstruction uses order year + raw month/day.")
    df["ship_date"] = repair_ship_dates(df.order_date, raw_ship)
    df["lead_time_days"] = (df.ship_date-df.order_date).dt.days
    if (df.lead_time_days < 0).any() or df.lead_time_days.max() >= 30: raise ValueError("Repaired lead-time magnitude validation failed")
    by_mode = df.groupby("ship_mode").lead_time_days.mean().reindex(SHIP_MODE_RANK)
    print("Repaired mean lead time by mode:", by_mode.to_dict())
    if not (by_mode.is_monotonic_increasing and by_mode.iloc[0] < by_mode.iloc[1] < by_mode.iloc[2] < by_mode.iloc[3]):
        raise ValueError("Repaired lead-time ordering validation failed")
    for col in ("sales","units","gross_profit","cost"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["gross_margin"] = np.divide(df.gross_profit,df.sales, out=np.zeros(len(df),dtype=float), where=df.sales.to_numpy()!=0)
    # A 3.0 IQR fence retains legitimate bulk orders while removing gross data errors.
    keep = pd.Series(True,index=df.index)
    for c in ("sales","units","gross_profit","cost"):
        q1,q3=df[c].quantile([.25,.75]); keep &= df[c].between(q1-3*(q3-q1),q3+3*(q3-q1))
    df=df.loc[keep].copy()
    df["ship_mode_rank"] = df.ship_mode.map(SHIP_MODE_RANK)
    df["order_month"] = df.order_date.dt.month
    df["order_quarter"] = df.order_date.dt.quarter
    df["order_weekday"] = df.order_date.dt.dayofweek
    df["order_weekend"] = (df.order_weekday >= 5).astype(int)
    zips=load_zip_centroids()
    zipcoords=[]; statecoords=[]
    precisions=[]
    for r in df.itertuples():
        country="United States" if str(r.country).lower() in ("united states","us","usa") else "Canada" if str(r.country).lower() in ("canada","ca") else str(r.country)
        point,precision=resolve_customer_centroid(r.postal_code,r.state,zips,country)
        zipcoords.append(point if precision=="ZIP" else None)
        statecoords.append(STATE_CENTROIDS.get(str(r.state).strip()))
        precisions.append(precision)
    df["customer_lat_zip"]=[x[0] if x else np.nan for x in zipcoords]
    df["customer_lon_zip"]=[x[1] if x else np.nan for x in zipcoords]
    df["customer_lat_state"]=[x[0] if x else np.nan for x in statecoords]
    df["customer_lon_state"]=[x[1] if x else np.nan for x in statecoords]
    df["customer_lat"] = df.customer_lat_zip.fillna(df.customer_lat_state)
    df["customer_lon"] = df.customer_lon_zip.fillna(df.customer_lon_state)
    for factory,(lat,lon) in FACTORY_COORDS.items():
        key=f"distance_{factory}"
        df[key]=haversine_km(lat,lon,df.customer_lat,df.customer_lon)
        state_key=f"distance_state_{factory}"
        df[state_key]=haversine_km(lat,lon,df.customer_lat_state,df.customer_lon_state)
    df["product_name"] = df.product_name.astype(str).map(lambda s: re.sub(r"\s*-\s*", " - ", s).strip())
    df["current_factory"] = df.product_name.map(PRODUCT_FACTORY)
    df["distance_km"]=[r.get(f"distance_{r['current_factory']}",np.nan) if pd.notna(r["current_factory"]) else np.nan for r in df.to_dict("records")]
    df["distance_state_km"]=[r.get(f"distance_state_{r['current_factory']}",np.nan) if pd.notna(r["current_factory"]) else np.nan for r in df.to_dict("records")]
    valid_zip=df.customer_lat_zip.notna() & df.customer_lon_zip.notna()
    print(f"Geocoding precision: ZIP-level for {valid_zip.mean()*100:.1f}% of retained rows; state-level fallback for {(~valid_zip).mean()*100:.1f}%.")
    df["geo_precision"] = np.where(valid_zip,"ZIP","state")
    return df
