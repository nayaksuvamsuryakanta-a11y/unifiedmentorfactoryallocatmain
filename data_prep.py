"""Naive lead-time preparation: DD-MM-YYYY dates, deduplication, Cost imputation,
IQR fence, centroid-based Haversine distances, and the Phase 7 factory fallback.

No date reconstruction is performed: the source ship years are later calendar years
and the brief treats the raw (Ship Date - Order Date) difference as the lead time.
"""
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (DATA_PATH, DATE_FORMAT, DIVISION_FACTORIES, EARTH_RADIUS_KM,
                    FACTORY_COORDS, PRODUCT_FACTORY, REGION_ORDER, SEED, SHIP_MODE_RANK,
                    STATE_CENTROIDS, Assumptions, normalize_product_name, resolve_factory)

log = logging.getLogger(__name__)

# Column mapping: source header -> canonical snake_case name.
RENAME = {
    "Row ID": "row_id", "Order ID": "order_id", "Order Date": "order_date",
    "Ship Date": "ship_date", "Ship Mode": "ship_mode", "Customer ID": "customer_id",
    "Country/Region": "country", "City": "city", "State/Province": "state",
    "Postal Code": "postal_code", "Division": "division", "Region": "region",
    "Product ID": "product_id", "Product Name": "product_name", "Sales": "sales",
    "Units": "units", "Gross Profit": "gross_profit", "Cost": "cost",
}
NUMERIC = ("sales", "units", "gross_profit", "cost")
DATE_COLS = ("order_date", "ship_date")
# Rows whose lead time is used as Phase 0 hard assertions (1-indexed Row ID).
ASSERTIONS = {354: 906, 4892: 1273, 6141: 1269}
# Fraction of input rows that must survive parsing before the pipeline refuses to run.
MIN_VALID_FRACTION = 0.95


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km, vectorized over array-like inputs."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float))
                              for v in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _to_datetime(series, name):
    """Parse DD-MM-YYYY strictly; returns a series whose NaTs are diagnosable."""
    parsed = pd.to_datetime(series, format=DATE_FORMAT, errors="coerce")
    # A strict parse cannot fall back to MDY, so any NaT is a genuine format defect.
    return parsed


def _calendar_days(start, end):
    """Day difference between two date series, invariant to time zone.

    The real source is naive, so this is a no-op there. When the columns carry an offset
    (a defensive input, or a test fixture), converting both endpoints to UTC midnight first
    stops a DST transition from stealing or gaining a day in the subtraction.
    """
    if start.dt.tz is not None:
        start = start.dt.tz_convert("UTC").dt.normalize()
        end = end.dt.tz_convert("UTC").dt.normalize()
    return (end - start).dt.days


def load_and_audit(path=DATA_PATH):
    """Load, clean and enrich the source file. Returns (frame, audit_dict).

    The audit dict carries every count the dashboard's error banner needs, so a single
    malformed-file diagnostic drives both the hard error and the warning path.
    """
    raw = pd.read_csv(path, dtype={"Postal Code": str})
    audit = {"input_rows": int(len(raw)), "input_columns": list(raw.columns)}

    # --- Stray duplicate header rows -----------------------------------------
    first_col = raw.columns[0]
    is_header = raw[first_col].astype(str).str.strip().str.lower() == first_col.lower()
    audit["stray_header_rows"] = int(is_header.sum())
    df = raw.loc[~is_header].copy()

    # --- Blank / all-NaN rows ------------------------------------------------
    blank = df.isna().all(axis=1)
    audit["blank_rows"] = int(blank.sum())
    df = df.loc[~blank].copy()

    # --- Field-count drift ---------------------------------------------------
    # pandas leaves NaN when a row has fewer fields than the header; more fields are
    # absorbed into the last column. Both are diagnosable here.
    required = set(RENAME)
    missing_in_source = required - set(df.columns)
    audit["missing_columns"] = sorted(missing_in_source)
    if missing_in_source:
        raise ValueError(f"Source is missing required columns: {sorted(missing_in_source)}")

    # --- Identity deduplication ---------------------------------------------
    # Row ID is a synthetic line number, not an order identity, so it is excluded from
    # the identity key. Two rows identical on every real field are one order line.
    identity_cols = [c for c in df.columns if c != "Row ID"]
    audit["duplicate_rows"] = int(df.duplicated(subset=identity_cols).sum())
    audit["rows_after_dedup"] = int(len(df)) - audit["duplicate_rows"]
    df = df.drop_duplicates(subset=identity_cols, keep="first").rename(columns=RENAME)

    # --- Dates --------------------------------------------------------------
    for col in DATE_COLS:
        df[col] = _to_datetime(df[col], col)
    audit["unparseable_order_dates"] = int(df["order_date"].isna().sum())
    audit["unparseable_ship_dates"] = int(df["ship_date"].isna().sum())
    # A row is only valid if both dates and every financial field parse.
    financial = df[list(NUMERIC)].apply(pd.to_numeric, errors="coerce")
    audit["unparseable_financial_rows"] = int(financial.isna().any(axis=1).sum())

    valid_mask = (df["order_date"].notna() & df["ship_date"].notna()
                  & financial.notna().any(axis=1))
    valid_fraction = float(valid_mask.mean())
    audit["valid_fraction"] = valid_fraction
    audit["parse_error_reasons"] = _first_five_reasons(df, valid_mask)

    df = df.loc[valid_mask].copy()
    audit["rows_after_valid"] = int(len(df))
    audit["dropped_parse_errors"] = audit["rows_after_dedup"] - audit["rows_after_valid"]
    df["row_id"] = df["row_id"].astype("Int64")
    df["order_id"] = df["order_id"].astype(str)

    # --- Hard assertions on the raw lead time --------------------------------
    # Dates carry no time-of-day. For naive data the conversion below is a no-op;
    # when the columns carry an offset, collapsing both endpoints to UTC midnight
    # first keeps a DST transition from shifting a calendar day.
    lead = _calendar_days(df["order_date"], df["ship_date"])
    audit["lead_min"] = int(lead.min())
    audit["lead_mean"] = float(lead.mean())
    audit["lead_max"] = int(lead.max())
    audit["lead_negative"] = int((lead < 0).sum())
    audit["assertions"] = {}
    for row_id, expected in ASSERTIONS.items():
        obs = lead[df["row_id"] == row_id]
        actual = int(obs.iloc[0]) if len(obs) else None
        audit["assertions"][str(row_id)] = {
            "expected": expected, "observed": actual,
            "pass": actual is not None and abs(actual - expected) <= 1,
        }
    failed = [k for k, v in audit["assertions"].items() if not v["pass"]]
    if failed:
        raise ValueError(f"Lead-time assertions failed: {failed}")

    df["lead_time_days"] = lead.astype("Int64")

    # --- Numerics -----------------------------------------------------------
    # Coerce before deriving cost, so a non-numeric financial field cannot make the
    # subtraction itself raise instead of being diagnosed as a parse error.
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Cost imputation ----------------------------------------------------
    missing_cost = int(df["cost"].isna().sum())
    df["cost"] = df["cost"].fillna(df["sales"] - df["gross_profit"])
    audit["cost_imputed"] = missing_cost

    # --- Categorical normalisation ------------------------------------------
    df["product_name"] = df["product_name"].astype(str).map(normalize_product_name)
    df["division"] = df["division"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip()
    df["ship_mode"] = df["ship_mode"].astype(str).str.strip()
    df["state"] = df["state"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    df["postal_code"] = df["postal_code"].astype(str).str.strip()
    df["ship_mode_rank"] = df["ship_mode"].map(SHIP_MODE_RANK)
    df["region_rank"] = df["region"].map(REGION_ORDER)
    audit["unmapped_ship_modes"] = int(df["ship_mode_rank"].isna().sum())
    audit["unmapped_regions"] = int(df["region_rank"].isna().sum())

    # --- Origin factory: explicit map, then Phase 7 division fallback --------
    explicit = df["product_name"].map(PRODUCT_FACTORY)
    fallback = df["division"].map(DIVISION_FACTORIES)
    df["current_factory"] = explicit.combine_first(fallback)
    df["factory_provenance"] = np.where(
        explicit.notna(), "explicit",
        np.where(fallback.notna(), "division-fallback", "unmapped"))
    audit["unmapped_products"] = int(df["current_factory"].isna().sum())
    audit["fallback_products"] = int((df["factory_provenance"] == "division-fallback").sum())
    audit["products"] = sorted(df["product_name"].unique().tolist())
    audit["product_count"] = int(df["product_name"].nunique())
    # Every product in the file is read from the source; none are synthesised. This flag
    # is here so a future synthetic-data guard has a place to fail loudly.
    audit["synthetic_products_present"] = False

    # --- Distances: origin factory to State/Province centroid ----------------
    centroid = df["state"].map(STATE_CENTROIDS)
    audit["unresolved_centroids"] = int(centroid.isna().sum())
    if audit["unresolved_centroids"]:
        log.warning("%d rows could not resolve a centroid; distance is NaN there.",
                    audit["unresolved_centroids"])
    df["customer_lat"] = centroid.map(lambda c: c[0] if isinstance(c, tuple) else np.nan)
    df["customer_lon"] = centroid.map(lambda c: c[1] if isinstance(c, tuple) else np.nan)

    for factory, (lat, lon) in FACTORY_COORDS.items():
        df[f"distance_{factory}"] = haversine_km(lat, lon, df["customer_lat"],
                                                 df["customer_lon"])
    df["distance_km"] = df.apply(
        lambda r: r.get(f"distance_{r['current_factory']}", np.nan)
        if pd.notna(r["current_factory"]) else np.nan, axis=1)

    # --- IQR outlier fence --------------------------------------------------
    keep = pd.Series(True, index=df.index, dtype=bool)
    iqr_removed_by = {}
    for col in NUMERIC:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - Assumptions.iqr_factor * iqr, q3 + Assumptions.iqr_factor * iqr
        in_fence = df[col].between(low, high)
        iqr_removed_by[col] = int((~in_fence.astype(bool)).sum())
        keep = keep & in_fence.astype(bool)
    audit["iqr_removed_by_column"] = iqr_removed_by
    audit["iqr_total_removed"] = int((~keep).sum())
    audit["iqr_removed_pct"] = float((~keep).mean() * 100)
    audit["iqr_factor"] = Assumptions.iqr_factor
    df = df.loc[keep].copy()
    audit["rows_after_iqr"] = int(len(df))
    audit["rows_final"] = int(len(df))

    # Derived calendar/order features used by the model.
    df["order_month"] = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_weekday"] = df["order_date"].dt.dayofweek
    df["order_year"] = df["order_date"].dt.year
    df["gross_margin"] = np.divide(df["gross_profit"], df["sales"],
                                   out=np.zeros(len(df), dtype=float),
                                   where=df["sales"].to_numpy() != 0)
    return df, audit


def _first_five_reasons(df, valid_mask):
    """Human-readable reasons for the first five invalid rows, for the warning banner."""
    reasons = []
    invalid = df.index[~valid_mask]
    for idx in invalid[:5]:
        row = df.loc[idx]
        why = []
        if pd.isna(row["order_date"]):
            why.append("unparseable Order Date")
        if pd.isna(row["ship_date"]):
            why.append("unparseable Ship Date")
        bad = [c for c in NUMERIC if pd.isna(pd.to_numeric(row[c], errors="coerce"))]
        if bad:
            why.append(f"non-numeric {','.join(bad)}")
        reasons.append(f"Row ID {row['row_id']}: {', '.join(why) or 'dropped'}")
    return reasons


def synthetic_product_check(products):
    """Every product in the file must appear in the source; none are generated here."""
    return sorted(products)


def _best_effort_audit(path):
    """Partial audit of a file that failed the full load, for the error banner."""
    audit = {"input_rows": 0, "input_columns": [], "stray_header_rows": 0,
             "blank_rows": 0, "duplicate_rows": 0, "missing_columns": [],
             "unparseable_order_dates": 0, "unparseable_ship_dates": 0,
             "unparseable_financial_rows": 0, "valid_fraction": 0.0,
             "parse_error_reasons": [], "assertions": {}}
    try:
        raw = pd.read_csv(path, dtype={"Postal Code": str})
    except Exception as exc:
        audit["parse_error_reasons"] = [f"Could not read the file: {exc}"]
        return audit
    audit["input_rows"] = int(len(raw))
    audit["input_columns"] = list(raw.columns)
    first_col = raw.columns[0]
    audit["stray_header_rows"] = int(
        raw[first_col].astype(str).str.strip().str.lower().eq(first_col.lower()).sum())
    audit["blank_rows"] = int(raw.isna().all(axis=1).sum())
    audit["missing_columns"] = sorted(set(RENAME) - set(raw.columns))
    identity_cols = [c for c in raw.columns if c != "Row ID"]
    audit["duplicate_rows"] = int(raw.duplicated(subset=identity_cols).sum())
    for col in DATE_COLS:
        if col in raw:
            try:
                audit[f"unparseable_{col.replace(' ', '_')}_dates"] = int(
                    pd.to_datetime(raw[col], format=DATE_FORMAT, errors="coerce").isna().sum())
            except Exception:
                pass
    financial = raw[list(NUMERIC)].apply(pd.to_numeric, errors="coerce") if set(NUMERIC) <= set(raw.columns) else pd.DataFrame()
    if not financial.empty:
        audit["unparseable_financial_rows"] = int(financial.isna().any(axis=1).sum())
    valid = raw[list(DATE_COLS)].apply(lambda s: pd.to_datetime(s, format=DATE_FORMAT, errors="coerce").notna(), axis=1).all(axis=1) if set(DATE_COLS) <= set(raw.columns) else pd.Series(False, index=raw.index)
    audit["valid_fraction"] = float(valid.mean()) if len(valid) else 0.0
    return audit


def load_and_audit_safe(path=DATA_PATH):
    """Load the source and separate recoverable warnings from fatal errors.

    Returns (frame, audit, None) on success, or (None, audit, error message) if the file
    cannot be used. The audit dict is populated as far as the failure allows, so the
    dashboard reports how many rows survived rather than a bare traceback.
    """
    try:
        return (*load_and_audit(path), None)
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return None, _best_effort_audit(path), str(exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df, audit = load_and_audit()
    print(json.dumps({k: v for k, v in audit.items() if k != "products"}, indent=2))
    print("products:", audit["product_count"], audit["products"])
    print("unmapped products:", audit["unmapped_products"])
    print(df[["row_id", "product_name", "division", "current_factory", "factory_provenance",
              "state", "distance_km"]].drop_duplicates("product_name").to_string())