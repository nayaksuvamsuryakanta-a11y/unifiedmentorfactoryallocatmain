from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
REQUIRED_ARTIFACTS = [
    "prepared_orders.csv",
    "recommendations.csv",
    "kpis.json",
    "joint_assignment_gated.csv",
    "joint_assignment_exploratory.csv",
    "routes.csv",
    "congested_region_products.csv",
    "ablation.csv",
    "mode_slopes.csv",
    "distance_residual_bins.csv",
    "state_distance.csv",
    "model_selection.csv",
    "sensitivity.csv",
    "sensitivity_stability.csv",
    "pilot_shortlist.csv",
]


def run_app():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    return at


def test_default_launch_raises_no_exceptions():
    at = run_app()
    assert len(at.exception) == 0


def test_low_evidence_warning_fires_and_then_silences_when_evidence_is_sufficient():
    at = run_app()
    assert any("Low evidence" in warning.value for warning in at.warning)
    at.selectbox[1].set_value("Wonka Bar - Fudge Mallows")
    at.run(timeout=30)
    assert not any("Low evidence" in warning.value for warning in at.warning)


def test_recommendation_dashboard_pilot_shortlist_renders_with_real_rows():
    at = run_app()
    shortlist = next(
        df.value for df in at.dataframe if hasattr(df.value, "columns") and "breakeven_freight_speed_km_day" in df.value.columns
    )
    assert len(shortlist) >= 1


def test_simulator_candidate_table_includes_risk_score_column():
    at = run_app()
    sim = next(df.value for df in at.dataframe if hasattr(df.value, "columns") and "Risk score" in df.value.columns)
    assert "Risk score" in sim.columns


def test_speed_profit_collinearity_caption_is_present():
    at = run_app()
    assert any(
        "Under the current distance-only freight model, speed and profit gains are perfectly correlated" in caption.value
        for caption in at.caption
    )


def test_risk_impact_panel_and_map_section_build_without_exception():
    at = run_app()
    assert len(at.exception) == 0
    assert any("Factory markers" in caption.value for caption in at.caption)


def test_toggling_assignment_mode_renders_different_data():
    at = run_app()
    before = at.dataframe[2].value.copy()
    at.radio[0].set_value("Globally optimal joint assignment")
    at.run(timeout=30)
    gated = at.dataframe[2].value.copy()
    assert not before.equals(gated)
    at.toggle[0].set_value(True)
    at.run(timeout=30)
    exploratory = at.dataframe[2].value.copy()
    assert not gated.equals(exploratory)


def test_required_artifacts_exist_with_expected_schema():
    required = {name: ARTIFACT_DIR / name for name in REQUIRED_ARTIFACTS}
    missing = sorted(str(path.relative_to(ARTIFACT_DIR)) for path in required.values() if not path.exists())
    assert not missing, f"Missing required artifacts: {missing}"

    expected_columns = {
        "prepared_orders.csv": ["order_id", "product_name", "ship_mode", "region", "lead_time_days"],
        "recommendations.csv": ["product", "current_factory", "candidate_factory", "score"],
        "kpis.json": ["point", "ci"],
        "joint_assignment_gated.csv": ["product", "factory", "units"],
        "joint_assignment_exploratory.csv": ["product", "factory", "units"],
        "routes.csv": ["current_factory", "region", "problem_route", "exposure"],
        "congested_region_products.csv": ["region", "product_name", "congested"],
        "ablation.csv": ["geocoding_distance", "feature_set", "cv_r2_mean"],
        "mode_slopes.csv": ["ship_mode", "slope_days_per_km", "p_value"],
        "distance_residual_bins.csv": ["distance_bin", "mean_residual_days"],
        "state_distance.csv": ["comparison", "cv_r2_mean"],
        "model_selection.csv": ["model", "selection_score", "chosen"],
        "sensitivity.csv": ["iteration", "lead_time_reduction_pct"],
        "sensitivity_stability.csv": ["product", "default_factory", "choice_agreement_pct"],
        "pilot_shortlist.csv": ["product", "candidate_factory", "breakeven_freight_speed_km_day"],
    }
    for name, expected in expected_columns.items():
        path = ARTIFACT_DIR / name
        if path.suffix == ".csv":
            columns = list(__import__("pandas").read_csv(path).columns)
            for item in expected:
                assert item in columns, f"{name} missing required column {item}"
        else:
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
            for item in expected:
                assert item in payload, f"{name} missing required key {item}"


def test_missing_required_artifact_renders_cleanup_error():
    missing_file = ARTIFACT_DIR / "prepared_orders.csv"
    backup = ARTIFACT_DIR / "prepared_orders.csv.bak"
    missing_file.rename(backup)
    try:
        at = AppTest.from_file(str(APP_PATH))
        at.run(timeout=30)
        assert any("Missing required artifact files" in error.value for error in at.error)
    finally:
        backup.rename(missing_file)
