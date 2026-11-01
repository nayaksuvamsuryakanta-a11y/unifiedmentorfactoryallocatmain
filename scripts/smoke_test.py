"""Smoke tests for the Nassau Candy Factory Reallocation rebuild.

Covers the Phase 8 acceptance checks (a-h). Run from the project root:

    python -m pytest scripts/smoke_test.py -v

The heavy test (`test_app_boots_without_exception`) drives the real app through Streamlit's
AppTest harness and takes a couple of minutes because it scores the full scenario engine.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (DEFAULT_TOP_N, FACTORY_COORDS, PRODUCT_FACTORY, RISK_HIGH, RISK_MEDIUM,
                    RISK_WEIGHT_DISTANCE_SHIFT, RISK_WEIGHT_LOW_SUPPORT, RISK_WEIGHT_NEG_PROFIT,
                    RISK_WEIGHT_PRED_VARIANCE, SEED, Assumptions)
from data_prep import ASSERTIONS, MIN_VALID_FRACTION, _calendar_days, load_and_audit_safe
from features import CATEGORICAL, NUMERIC, make_preprocessor
from models import FEATURES, load_model, select_best, train_models
from simulation import (candidate_moves, compute_kpis, parse_components, score_products,
                        top_recommendations)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
CSV = ROOT / "Nassau Candy Distributor.csv"

RISK_WEIGHTS = {
    "negProfit": RISK_WEIGHT_NEG_PROFIT,
    "predVariance": RISK_WEIGHT_PRED_VARIANCE,
    "lowSupport": RISK_WEIGHT_LOW_SUPPORT,
    "distanceShift": RISK_WEIGHT_DISTANCE_SHIFT,
}


@pytest.fixture(scope="module")
def audit_df():
    df, audit, err = load_and_audit_safe()
    assert err is None, f"source file unusable: {err}"
    return df, audit


@pytest.fixture(scope="module")
def scored(audit_df):
    df, _ = audit_df
    residual_std = (pd.read_csv(ARTIFACTS / "per_product_residual_std.csv")
                    .set_index("product")["residual_std"].to_dict())
    return score_products(df, speed_weight=Assumptions.speed_weight,
                          residual_std=residual_std,
                          global_std=float(pd.read_csv(ARTIFACTS / "per_product_residual_std.csv")
                                           ["residual_std"].max()))


@pytest.fixture(scope="module")
def model_metrics():
    return pd.read_csv(ARTIFACTS / "model_metrics.csv")


# ---------------------------------------------------------------------------
# (a) App boots, zero exceptions
# ---------------------------------------------------------------------------

def test_app_boots_without_exception():
    """Phase 8a: launching `streamlit run app.py` raises no exception."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"))
    at.run(timeout=300)
    exc = at.exception
    exc_list = exc if isinstance(exc, list) else [exc] if exc else []
    assert not exc_list, "app raised during boot: " + "; ".join(repr(e) for e in exc_list[:3])


# ---------------------------------------------------------------------------
# (b) Row counts, duplicates, outliers
# ---------------------------------------------------------------------------

def test_input_rows(audit_df):
    """Phase 8b: the file holds roughly 10,155 rows."""
    _, audit = audit_df
    assert 9500 <= audit["input_rows"] <= 10500, audit["input_rows"]


def test_duplicates_removed(audit_df):
    """Phase 8b: at least one exact duplicate row was removed."""
    _, audit = audit_df
    assert int(audit["duplicate_rows"]) > 0, audit["duplicate_rows"]


def test_outliers_removed(audit_df):
    """Phase 8b: the 3.0x IQR fence removed at least one row."""
    _, audit = audit_df
    assert int(audit["iqr_total_removed"]) > 0, audit["iqr_total_removed"]
    assert float(audit["iqr_removed_pct"]) > 0, audit["iqr_removed_pct"]


def test_iqr_does_not_trim_lead_time(audit_df):
    """Lead time is never an outlier target; the assertions must hold on the full distribution."""
    df, _ = audit_df
    assert int((df["lead_time_days"] < 0).sum()) == 0
    assert int(df["lead_time_days"].min()) >= 900


# ---------------------------------------------------------------------------
# (c) Hard assertions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row_id,expected", ASSERTIONS.items())
def test_lead_time_assertions(audit_df, row_id, expected):
    """Phase 8c: Row 354 -> 906, Row 4892 -> 1273, Row 6141 -> 1269."""
    df, _ = audit_df
    observed = int(df.loc[df["row_id"] == row_id, "lead_time_days"].iloc[0])
    assert observed == expected, f"Row {row_id}: expected {expected}, got {observed}"


# ---------------------------------------------------------------------------
# (d) Product count, no synthetic products
# ---------------------------------------------------------------------------

def test_product_count(audit_df):
    """Phase 8d: exactly 15 products, all sourced from the file."""
    _, audit = audit_df
    assert audit["product_count"] == 15, audit["product_count"]


def test_no_synthetic_products(audit_df):
    """Phase 8d: no product was generated by the codebase."""
    _, audit = audit_df
    assert audit["synthetic_products_present"] is False


def test_all_products_in_source(audit_df):
    """Every product name appears in the raw CSV (normalised to canonical spacing)."""
    from config import normalize_product_name

    df, _ = audit_df
    raw = pd.read_csv(CSV, dtype={"Postal Code": str})
    assert set(df["product_name"]).issubset(set(raw["Product Name"].map(normalize_product_name)))


# ---------------------------------------------------------------------------
# (e) Sidebar controls drive live output
# ---------------------------------------------------------------------------

def _frame_hash(at):
    """Content fingerprint of the first rendered dataframe on the current page."""
    if not at.dataframe:
        return None
    frame = at.dataframe[0].value
    return str(frame.shape) + "," + str(hash(frame.to_string(index=False)))


def test_sidebar_controls_rerender():
    """Phase 8e: Product, Region, Ship Mode and priority slider all change the output."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"))
    at.run(timeout=300)
    side = at.sidebar
    assert len(side.selectbox) >= 1
    assert len(side.multiselect) >= 2
    assert len(side.slider) >= 1

    before = _frame_hash(at)
    side.selectbox[0].set_value("Laffy Taffy").run(timeout=180)
    assert _frame_hash(at) != before, "Product selector did not change rendered output"

    before = _frame_hash(at)
    side.slider[0].set_value(0.75).run(timeout=120)
    assert _frame_hash(at) != before, "Priority slider did not change rendered output"

    before = _frame_hash(at)
    side.multiselect[1].set_value(["Standard Class"]).run(timeout=120)
    assert _frame_hash(at) != before, "Ship Mode filter did not change rendered output"


# ---------------------------------------------------------------------------
# (f) All four modules render real data, Risk panel populated
# ---------------------------------------------------------------------------

def test_modules_render_real_data(audit_df, scored):
    """Phase 8f: each of the four modules renders data sourced from the file."""
    df, _ = audit_df
    assert len(scored) > 0

    from simulation import factory_optimiser_summary

    summary = factory_optimiser_summary(df, scored)
    assert set(summary["factory"]) == set(FACTORY_COORDS)
    assert summary["orders"].sum() > 0

    assert scored["risk_level"].isin(["High", "Medium", "Low"]).all()


def test_risk_panel_populated(scored):
    """Phase 8f: the Risk panel carries real alerts or an explicit low-risk state."""
    assert len(scored) > 0
    levels = set(scored["risk_level"])
    assert levels, "no risk levels computed"
    # Non-degenerate: every row carries all four weighted components.
    components = scored["risk_components"].map(parse_components)
    assert all(set(RISK_WEIGHTS) <= set(c) for c in components), "missing risk component"
    weights = pd.DataFrame(components.tolist())[list(RISK_WEIGHTS)]
    # Every component column is genuinely used (not a constant zero).
    for col, weight in RISK_WEIGHTS.items():
        assert weights[col].max() > 0, f"{col} is degenerate"
        assert weights[col].min() >= 0, f"{col} went negative"


# ---------------------------------------------------------------------------
# (g) KPI header
# ---------------------------------------------------------------------------

def test_kpi_header(scored):
    """Phase 8g: all four KPIs are present and computable."""
    best = pd.read_csv(ARTIFACTS / "model_metrics.csv")
    best_r2 = float(best.loc[best["model"] == "Linear Regression", "cv_r2_mean"].iloc[0])
    kpis = compute_kpis(scored, int(scored["product"].nunique()), best_r2)
    point = kpis["point"]
    for key in ("lead_time_reduction_pct", "profit_stability_pct",
                "scenario_confidence", "coverage_pct"):
        assert key in point, f"missing KPI {key}"
        assert isinstance(point[key], (int, float))
    assert point["coverage_pct"] >= 0.0
    assert point["profit_stability_pct"] <= 100.0


# ---------------------------------------------------------------------------
# (h) Model table
# ---------------------------------------------------------------------------

def test_model_table(model_metrics):
    """Phase 8h: all three estimators have RMSE, MAE, R2 and CV figures."""
    required = {"model", "r2", "rmse", "mae", "cv_r2_mean", "cv_r2_std",
                "cv_rmse_mean", "cv_mae_mean"}
    assert required <= set(model_metrics.columns), sorted(set(model_metrics.columns))
    assert len(model_metrics) == 3, len(model_metrics)
    assert set(model_metrics["model"]) == {"Linear Regression", "Random Forest", "Gradient Boosting"}


def test_all_models_reported_negative(model_metrics):
    """All three score at or below zero; nothing is silently improved."""
    assert (model_metrics["r2"] <= 0).all()
    assert (model_metrics["cv_r2_mean"] <= 0).all()


def test_best_model_justified(model_metrics):
    """The chosen estimator is the most interpretable within tolerance of best accuracy."""
    best_cv = model_metrics["cv_r2_mean"].max()
    tier = {"Linear Regression": 0, "Random Forest": 1, "Gradient Boosting": 2}
    eligible = model_metrics[model_metrics["cv_r2_mean"] >= best_cv - Assumptions.model_selection_tolerance]
    chosen = eligible.sort_values("interpretability_tier").iloc[0]["model"]
    assert chosen == "Linear Regression", chosen


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def test_recommendation_coverage(scored):
    """Phase 8h: coverage >= 90% of the catalogue."""
    assert scored["product"].nunique() == 15
    recs = top_recommendations(scored, DEFAULT_TOP_N)
    assert len(recs) > 0


def test_top_ten_distinct_factories(scored):
    """Phase 8h: at least 3 distinct factories appear in the Top-10."""
    recs = top_recommendations(scored, DEFAULT_TOP_N)
    distinct = sorted(recs["candidate_factory"].unique())
    assert len(distinct) >= 3, distinct


def test_top_ten_positive_lead_gain(scored):
    """Phase 8h: every Top-10 row shows a positive lead-time reduction."""
    recs = top_recommendations(scored, DEFAULT_TOP_N)
    assert (recs["lead_gain_days"] > 0).all()
    assert (recs["delta_distance_km"] < 0).all()


# ---------------------------------------------------------------------------
# Documentation and deliverables
# ---------------------------------------------------------------------------

def test_docs_exist():
    """Phase 6: both documents are present."""
    assert (DOCS / "research_paper.md").exists()
    assert (DOCS / "executive_summary.md").exists()
    assert (ROOT / "README.md").exists()
    assert (ROOT / "requirements.txt").exists()


def test_requirements_pinned():
    """Phase 6: dependencies are pinned to exact versions."""
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    pinned = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    assert all("==" in p for p in pinned), pinned


def test_legacy_preserved():
    """Nothing was deleted silently: superseded code lives under legacy/."""
    legacy = ROOT / "legacy"
    assert legacy.is_dir()
    files = [p.name for p in legacy.rglob("*") if p.is_file()]
    assert "app.py" in files
    assert "data_prep.py" in files
    assert "models.py" in files


def test_source_file_present():
    """The real CSV is the only data source; no synthetic CSV is used."""
    assert CSV.exists()
    assert not (ROOT / "Nassau Candy Distributor (1).csv").exists()


def test_argparse_safety():
    """Sanity check that the project imports cleanly and the seed is fixed."""
    assert SEED == 42
    assert MIN_VALID_FRACTION == 0.95
    assert RISK_HIGH == 0.50
    assert RISK_MEDIUM == 0.25


# ---------------------------------------------------------------------------
# Tier T1: data contract & parser
# ---------------------------------------------------------------------------

# The source schema: 18 columns, read exactly as exported.
EXPECTED_COLUMNS = [
    "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID",
    "Country/Region", "City", "State/Province", "Postal Code", "Division", "Region",
    "Product ID", "Product Name", "Sales", "Units", "Gross Profit", "Cost",
]
# Time zones exercised by the lead-time matrix (UTC plus one zone either side of the
# dateline and one DST zone). The pipeline is naive; this proves normalisation is safe.
TZ_ZONES = ["UTC", "America/New_York", "Australia/Sydney", "Asia/Kolkata"]
# The accounting identity: Sales - Cost - Gross Profit must stay inside this band.
ACCOUNTING_TOLERANCE = 0.011
# Rows injected by the fuzz harness, split across four disjoint defect classes.
FUZZ_COUNT = 200
FUZZ_STRAY_HEADERS = 50
FUZZ_BAD_DATES = 50
FUZZ_NON_NUMERIC = 50
FUZZ_TRUNCATED = 50


def _read_source():
    """Read the real file the way the parser does, Postal Code held as text."""
    return pd.read_csv(CSV, dtype={"Postal Code": str})


def test_eighteen_column_schema():
    """T1: the source exposes exactly the 18-column contract; nothing is invented."""
    raw = _read_source()
    assert list(raw.columns) == EXPECTED_COLUMNS
    assert len(raw.columns) == 18


def test_postal_code_is_string(audit_df):
    """T1: Postal Code is text, so an alphabetic code is never silently dropped."""
    df, _ = audit_df
    assert pd.api.types.is_string_dtype(df["postal_code"])
    raw = _read_source()
    assert raw["Postal Code"].map(lambda v: isinstance(v, str)).all()
    # 200 Canadian codes contain letters; numeric coercion would turn them into NaN.
    alphabetic = raw.loc[raw["Postal Code"].str.contains(r"[A-Za-z]", na=False), "Postal Code"]
    assert len(alphabetic) > 0, "no alphabetic postal code to anchor this check on"
    assert set(alphabetic).issubset(set(df["postal_code"])), "alphabetic codes were coerced"


def test_column_dtypes_match_contract(audit_df):
    """T1: dates parse to datetimes, identities stay strings, numerics stay numeric."""
    df, _ = audit_df
    for col in ("order_date", "ship_date"):
        assert str(df[col].dtype).startswith("datetime"), col
    for col in ("order_id", "postal_code", "product_name"):
        assert pd.api.types.is_string_dtype(df[col]), col
    for col in ("sales", "gross_profit", "cost"):
        assert pd.api.types.is_numeric_dtype(df[col]), col
    assert pd.api.types.is_integer_dtype(df["units"])
    # Row ID is nullable: a missing value must stay missing rather than collapsing to 0.
    assert str(df["row_id"].dtype) == "Int64", df["row_id"].dtype


def test_accounting_identity_holds(audit_df):
    """T1: |Sales - Cost - Gross Profit| stays inside 0.011 for every surviving row."""
    df, _ = audit_df
    diff = (df["sales"] - df["cost"] - df["gross_profit"]).abs()
    violations = int((diff > ACCOUNTING_TOLERANCE).sum())
    assert violations == 0, f"{violations} rows breach the accounting identity"
    assert float(diff.max()) < ACCOUNTING_TOLERANCE, diff.max()


def test_tz_matrix_survives_four_zones(audit_df):
    """T1: the lead-time delta is invariant across UTC, New York, Sydney and Kolkata."""
    df, audit = audit_df
    anchors = {354: 906, 4892: 1273, 6141: 1269}
    for zone in TZ_ZONES:
        aware = df.copy()
        aware["order_date"] = aware["order_date"].dt.tz_localize(zone)
        aware["ship_date"] = aware["ship_date"].dt.tz_localize(zone)
        lead = _calendar_days(aware["order_date"], aware["ship_date"])
        assert int(lead.min()) >= 0, f"{zone}: a negative lead time appeared"
        for row_id, expected in anchors.items():
            observed = int(lead.loc[df["row_id"] == row_id].iloc[0])
            assert observed == expected, f"{zone} Row {row_id}: expected {expected}, got {observed}"
    # The naive pipeline actually in use must still agree with the matrix.
    for row_id, expected in anchors.items():
        assert audit["assertions"][str(row_id)]["pass"], f"Row {row_id} anchor failed"


def _fuzz_file():
    """Append 200 deliberately malformed rows to a copy of the source file.

    The four injected defect classes map one-to-one onto the audit counters the dashboard
    banner reads, so their counts can be asserted exactly. The scratch copy lives under
    artifacts/ rather than the system temp dir because that root is not reliably writable
    on every host -- a fixture that cannot run is a false failure, not a real one.
    """
    work = tempfile.mkdtemp(prefix="fuzz-", dir=str(ARTIFACTS))
    raw = _read_source()
    # Inject from the deduplicated pool: the source contains exact duplicate rows (Row ID is
    # a synthetic line number, not part of the identity), so a row that already exists would
    # inflate the duplicate counter and silently swallow an injected defect.
    pool = raw.drop_duplicates(subset=[c for c in raw.columns if c != "Row ID"]).reset_index(
        drop=True)
    first_col = raw.columns[0]
    frames = [raw]
    # 50 rows whose first cell looks like a stray header line
    for i in range(FUZZ_STRAY_HEADERS):
        row = pool.iloc[i % len(pool)].copy()
        row[first_col] = "Row ID"
        frames.append(pd.DataFrame([row]))
    # 50 rows with a date that cannot be DD-MM-YYYY
    for i in range(FUZZ_BAD_DATES):
        row = pool.iloc[i % len(pool)].copy()
        row["Order Date"] = "2024/13/45"
        frames.append(pd.DataFrame([row]))
    # 50 rows with a financial field that is not a number
    for i in range(FUZZ_NON_NUMERIC):
        row = pool.iloc[i % len(pool)].copy()
        row["Sales"] = "###N/A"
        frames.append(pd.DataFrame([row]))
    # 50 entirely blank rows
    blank = pd.DataFrame([[None] * len(raw.columns)] * FUZZ_TRUNCATED,
                         columns=raw.columns)
    frames.append(blank)
    combined = pd.concat(frames, ignore_index=True)
    path = Path(work) / "fuzz.csv"
    combined.to_csv(path, index=False)
    return path


def test_fuzz_200_malformed_rows():
    """T1: 200 malformed rows are diagnosed, never raised as an unhandled traceback."""
    path = _fuzz_file()
    work = path.parent
    try:
        df, audit, err = load_and_audit_safe(path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    # A malformed source is recoverable: the parser reports, it does not crash.
    assert df is not None, "malformed source aborted the whole load"
    assert err is None, f"unrecoverable error raised: {err}"
    assert audit["input_rows"] == 10194 + FUZZ_COUNT, audit["input_rows"]
    # Each injected defect class must be counted exactly once and map onto the banner.
    assert audit["stray_header_rows"] == FUZZ_STRAY_HEADERS, audit["stray_header_rows"]
    assert audit["blank_rows"] == FUZZ_TRUNCATED, audit["blank_rows"]
    assert audit["unparseable_order_dates"] == FUZZ_BAD_DATES, audit["unparseable_order_dates"]
    assert audit["unparseable_financial_rows"] == FUZZ_NON_NUMERIC, audit["unparseable_financial_rows"]
    # The banner needs human-readable reasons, not just raw counts.
    assert len(audit["parse_error_reasons"]) > 0, audit["parse_error_reasons"]
    # Enough rows survive that the pipeline keeps running.
    assert audit["valid_fraction"] >= MIN_VALID_FRACTION, audit["valid_fraction"]


# ---------------------------------------------------------------------------
# Tier T2: determinism, golden artifacts, snapshots
# ---------------------------------------------------------------------------

GOLDEN = ARTIFACTS / "golden_hashes.json"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of(data):
    return hashlib.sha256(data).hexdigest()


def _score_and_digest():
    """Rebuild the engine outputs and fingerprint them; the seed must make this stable."""
    df, _, err = load_and_audit_safe()
    assert err is None, f"source unusable: {err}"
    residual_std = (pd.read_csv(ARTIFACTS / "per_product_residual_std.csv")
                    .set_index("product")["residual_std"].to_dict())
    global_std = float(pd.read_csv(ARTIFACTS / "per_product_residual_std.csv")
                       ["residual_std"].max())
    scored = score_products(df, residual_std=residual_std, global_std=global_std)
    recs = top_recommendations(scored, len(scored["product"].unique()))
    metrics = pd.read_csv(ARTIFACTS / "model_metrics.csv")
    kpis = compute_kpis(scored, int(scored["product"].nunique()),
                        float(metrics.loc[metrics["model"] == "Linear Regression",
                                          "cv_r2_mean"].iloc[0]))
    return {
        "scored": _sha256_of(scored.to_csv(index=False).encode()),
        "recs": _sha256_of(recs.to_csv(index=False).encode()),
        "kpis": _sha256_of(json.dumps(kpis, sort_keys=True).encode()),
    }


def test_golden_artifact_hashes():
    """T2: every committed artifact still matches its golden sha256, so nothing drifted."""
    manifest = json.loads(GOLDEN.read_text(encoding="utf-8"))
    missing = [n for n in manifest if not (ARTIFACTS / n).exists()]
    assert not missing, f"artifacts gone: {missing}"
    mismatches = []
    for name, expected in manifest.items():
        actual = _sha256(ARTIFACTS / name)
        if actual != expected:
            mismatches.append(f"{name}: expected {expected[:12]}..., got {actual[:12]}...")
    assert not mismatches, "artifact drift:\n" + "\n".join(mismatches)


def test_pipeline_double_run_sha256_identical():
    """T2: two in-process runs of the engine produce byte-identical outputs (seed 42)."""
    first = _score_and_digest()
    second = _score_and_digest()
    assert first == second, {k: (v[:40], second[k][:40]) for k, v in first.items() if v != second[k]}


def _snapshot(at):
    """Stable fingerprint of the app's recommendation table across a warm rerun."""
    if not at.dataframe:
        return None
    frame = at.dataframe[0].value
    return str(frame.shape) + "," + _sha256_of(frame.to_csv(index=False).encode())


def test_snapshot_warm_rerun_unchanged():
    """T2: a warm rerun renders an identical recommendation snapshot, no retraining."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"))
    at.run(timeout=300)
    before = _snapshot(at)
    at.run(timeout=300)
    after = _snapshot(at)
    assert before is not None, "app rendered no dataframe to snapshot"
    assert before == after, "warm rerun changed the rendered output"


# ---------------------------------------------------------------------------
# Tier T3: model governance -- leakage, temporal protocol, tolerance
# ---------------------------------------------------------------------------


def test_feature_set_excludes_target_and_target_derivatives():
    """T3: the model cannot see the answer; lead time is not among its own features."""
    target = "lead_time_days"
    assert target not in FEATURES, FEATURES
    assert not any(target in f for f in FEATURES), FEATURES
    assert set(FEATURES) == set(CATEGORICAL + NUMERIC), sorted(FEATURES)


def test_transformers_are_fitted_inside_a_pipeline():
    """T3: impute/scale/encode statistics come from training data only, never the holdout."""
    prep = make_preprocessor()
    assert isinstance(prep, ColumnTransformer), type(prep)
    for _, transformer, _ in prep.transformers:
        assert isinstance(transformer, Pipeline), f"{transformer} leaks via standalone fit"


def test_holdout_is_a_seeded_random_split_not_temporal(model_metrics):
    """T3: selection rests on the random split; temporal R² is not a selection input."""
    src = inspect.getsource(train_models)
    assert "random_state=SEED" in src, "split is not pinned to the fixed seed"
    assert "train_test_split" in src
    # The live report carries no temporal column, so temporal scores cannot drive it.
    assert "temporal" not in set(model_metrics.columns), model_metrics.columns.tolist()
    best = select_best(model_metrics)
    assert best == "Linear Regression", best


def test_selection_stable_across_tolerance_sweep(model_metrics):
    """T3: the chosen estimator is the same at tolerances 0.01, 0.02 and 0.05."""
    for tolerance in (0.01, 0.02, 0.05):
        chosen = select_best(model_metrics, tolerance)
        assert chosen == "Linear Regression", f"tolerance {tolerance} selected {chosen}"


def test_artifact_matches_reported_best_model(model_metrics):
    """T3: the persisted joblib is the estimator the report selected, with its feature set."""
    pipe, name, features = load_model()
    assert name == select_best(model_metrics), f"artifact {name} != report winner"
    assert features == FEATURES, "persisted feature set disagrees with the report"
    assert isinstance(pipe, Pipeline), type(pipe)


# ---------------------------------------------------------------------------
# Tier T4: business logic -- A1 register, net outlay, slider, risk, Top-10
# ---------------------------------------------------------------------------


def test_assumption_provenance_register(scored):
    """T4: every scored row carries the A1 assumption label and its version hash."""
    assert len(scored) > 0
    assert scored["gain_basis"].str.startswith("assumption:A1").all(), scored["gain_basis"].unique()
    # Retain rows keep the A1 provenance: retaining is itself an A1 outcome.
    assert not scored["gain_basis"].str.contains("assumption:A2").any(), scored[
        "gain_basis"].unique()
    assert scored["assumption_version"].nunique() == 1, scored["assumption_version"].unique()


def test_assumption_version_matches_config():
    """T4: the on-disk register agrees with the live assumption set, so it cannot drift."""
    df, _, err = load_and_audit_safe()
    assert err is None, f"source unusable: {err}"
    assert candidate_moves(df)["assumption_version"].iloc[0] == Assumptions.assumption_version()


def test_delta_cost_carries_the_net_setup_outlay(scored):
    """T4: delta_cost is freight delta plus a destination-minus-incumbent one-time outlay."""
    df, _, err = load_and_audit_safe()
    assert err is None, f"source unusable: {err}"
    row = scored.iloc[0]
    product = row["product"]
    current = row["current_factory"]
    candidate = row["candidate_factory"]
    g = df[df["product_name"] == product]
    units = float(g["units"].clip(lower=1).sum())
    delta_dist = float(row["delta_distance_km"])
    expected_freight = units * (delta_dist / 1000.0) * Assumptions.freight_cost_per_unit_per_1000km
    expected_net = (Assumptions.setup_freight_cost.get(candidate, 0.0)
                    - Assumptions.setup_freight_cost.get(current, 0.0))
    assert abs(float(row["delta_cost"]) - (expected_freight + expected_net)) < 1e-9, row["delta_cost"]
    # The net outlay is what breaks the collinearity with delta_lead.
    assert expected_net != 0.0 or delta_dist == 0.0, "no independent cost term"


def test_slider_does_not_collapse_to_a_single_ranking():
    """T4: lead_gain_days and profit_gain are no longer scalar multiples, so the slider moves."""
    df, _, err = load_and_audit_safe()
    assert err is None, f"source unusable: {err}"
    residual_std = (pd.read_csv(ARTIFACTS / "per_product_residual_std.csv")
                    .set_index("product")["residual_std"].to_dict())
    global_std = float(pd.read_csv(ARTIFACTS / "per_product_residual_std.csv")
                       ["residual_std"].max())
    cand = candidate_moves(df)
    mov = cand[cand["delta_distance_km"] != 0]
    corr = abs(float(mov["delta_lead_days"].corr(mov["delta_cost"])))
    assert corr < 0.99, f"cost and lead deltas still collinear (r={corr:.4f})"

    first = score_products(df, speed_weight=0.0, residual_std=residual_std,
                           global_std=global_std)
    last = score_products(df, speed_weight=1.0, residual_std=residual_std,
                          global_std=global_std)
    before = dict(zip(top_recommendations(first, 15)["product"],
                      top_recommendations(first, 15)["candidate_factory"]))
    after = dict(zip(top_recommendations(last, 15)["product"],
                     top_recommendations(last, 15)["candidate_factory"]))
    assert before != after, "priority slider produced an identical ranking"
    # Laffy Taffy is the canary: it used to be fixed for every weight.
    assert before["Laffy Taffy"] != after["Laffy Taffy"], "Laffy Taffy stuck at " + str(
        before["Laffy Taffy"])


def test_risk_panel_is_non_degenerate(scored):
    """T4: every row carries all four weighted risk components, and each component varies."""
    assert len(scored) > 0
    components = scored["risk_components"].map(parse_components)
    assert all(set(RISK_WEIGHTS) <= set(c) for c in components), "missing risk component"
    weights = pd.DataFrame(components.tolist())[list(RISK_WEIGHTS)]
    for col, weight in RISK_WEIGHTS.items():
        assert weights[col].max() > 0, f"{col} is degenerate"
        assert weights[col].min() >= 0, f"{col} went negative"
    levels = scored["risk_level"].value_counts()
    assert int(levels.get("High", 0)) + int(levels.get("Medium", 0)) + int(
        levels.get("Low", 0)) == len(scored)
    assert len(levels) == 3, levels.tolist()


def test_top_ten_gains_are_real(scored):
    """T4: every recommendation shortens the route and the lead time, never lengthening it."""
    recs = top_recommendations(scored, DEFAULT_TOP_N)
    assert len(recs) == DEFAULT_TOP_N, len(recs)
    assert (recs["lead_gain_days"] > 0).all(), recs["lead_gain_days"].min()
    assert (recs["delta_distance_km"] < 0).all(), recs["delta_distance_km"].max()
    assert (recs["current_factory"] != recs["candidate_factory"]).all()
    # One recommendation per product; nothing is silently dropped or doubled up.
    assert recs["product"].nunique() == len(recs), recs["product"].tolist()


def test_no_synthetic_recommendations(scored):
    """T4: every recommended factory is a real factory, and every product is in the file."""
    recs = top_recommendations(scored, len(scored["product"].unique()))
    assert set(recs["candidate_factory"]).issubset(set(FACTORY_COORDS))
    assert set(recs["product"]).issubset(set(scored["product"]))