# Technical documentation

## 1. Purpose and architecture

The project reads `Nassau Candy Distributor.csv`, audits and cleans order lines, derives
lead time and factory-to-destination distances, trains three regression models, clusters
routes, and scores alternative factory assignments. `run_pipeline.py` writes CSV, JSON, and
Joblib outputs under `artifacts/` and renders the two report Markdown files. `app.py` reads
the source CSV, the persisted model and selected artifacts, then renders a Streamlit
dashboard. Scenario lead-time and cost changes are assumption-driven. There is no database.
The checked-in dashboard has no REST endpoint or authentication implementation.

## 2. Dependency table

Installed versions below were obtained with `importlib.metadata.version()` from
`%TEMP%\factoryalloc-test-20261101\Scripts\python.exe`, after installing
`requirements.txt`.

| Library | Pinned | Installed | Role | Imported by | Pinned but never imported? |
|---|---:|---:|---|---|---|
| `pandas` | 3.0.6 | 3.0.6 | tabular cleaning, grouping, metrics, CSV I/O | `app.py`, `data_prep.py`, `models.py`, `clustering.py`, `simulation.py`, `run_pipeline.py`, tests | No |
| `numpy` | 2.5.3 | 2.5.3 | arrays, math, random draws | `app.py`, `data_prep.py`, `models.py`, `clustering.py`, `simulation.py`, `run_pipeline.py` | No |
| `scikit-learn` | 1.9.1 | 1.9.1 | preprocessing, regression, clustering, validation | `features.py`, `models.py`, `clustering.py` | No |
| `scipy` | 1.18.1 | 1.18.1 | legacy `linregress`/`milp` code only | `legacy/models.py`, `legacy/simulation.py` | No, but not used by active pipeline |
| `matplotlib` | 3.11.2 | 3.11.2 | legacy plotting only | `legacy/app.py` | No, but not used by active app |
| `seaborn` | 0.13.2 | 0.13.2 | no direct import found | none | **Yes** |
| `streamlit` | 1.64.0 | 1.64.0 | dashboard widgets, pages, cache, charts | `app.py`, tests, smoke scripts | No |
| `plotly` | 5.24.1 | 5.24.1 | interactive charts and factory map | `app.py` | No |
| `tenacity` | 9.2.1 | 9.2.1 | no direct import found | none | **Yes** |
| `joblib` | 1.6.0 | 1.6.0 | persist and load the fitted model | `models.py` | No |
| `PuLP` | 4.0.0 | 4.0.0 | legacy/test optimization import only | `legacy/simulation.py`, `legacy/tests/test_pipeline.py`, `tests/test_pipeline.py` | No, but not used by active pipeline |
| `pytest` | 9.1.1 | 9.1.1 | test runner and test assertions | `scripts/smoke_test.py`, tests | No |

`pgeocode` is not pinned or imported. The checked-in ZIP file metadata says its values came
from `pgeocode 0.5.0 / GeoNames postal dataset`, but active `data_prep.py` uses the embedded
state/province centroid table instead.

## 3. Direct third-party API reference

### `pandas`

- `pd.read_csv(path, dtype=...)` reads the source and artifact CSVs (`data_prep.py:74`,
  `data_prep.py:264`, `app.py:85`, `app.py:490`, `app.py:507`, `app.py:518`).
- `DataFrame.groupby`, `agg`, `quantile`, `merge`-style frame operations, `drop_duplicates`,
  `rename`, `assign`-style column operations, `fillna`, `astype`, `apply`, `melt`,
  `sort_values`, `reset_index`, `value_counts`, `nunique`, `to_numpy`, and `to_csv` perform
  the audit, route aggregation, risk summaries, and artifact export. They are used throughout
  `data_prep.py`, `clustering.py`, `models.py`, `simulation.py`, `app.py`, and
  `run_pipeline.py`.
- `pd.to_datetime(..., format="%d-%m-%Y", errors="coerce")` strictly parses dates
  (`data_prep.py:49`, `data_prep.py:78`).
- `pd.to_numeric(..., errors="coerce")` diagnoses and converts financial columns
  (`data_prep.py:95`, `data_prep.py:132`).
- `pd.Series`, `pd.isna`, and `pd.notna` support typed series, missing-value checks and
  defensive input handling (`data_prep.py:38`, `simulation.py:32`, `app.py`).

### `numpy`

- `np.asarray`, `np.radians`, `np.sin`, `np.cos`, `np.arcsin`, `np.sqrt`, and `np.clip`
  implement vectorized Haversine distance (`data_prep.py:38-45`).
- `np.average` weights baseline and candidate distances by units (`simulation.py:48-52`).
- `np.arange` creates train/test indices (`models.py:60`, `models.py:142`); `np.random.default_rng`
  creates deterministic Monte Carlo draws (`simulation.py:108`); `np.random`-compatible
  numerical operations support bootstrap calculations (`simulation.py:233-282`).
- `np.inf`, `np.nan`, `np.zeros`, and `np.divide` handle route and derived-feature edge
  cases (`data_prep.py:185`, `data_prep.py:221`, `clustering.py:39`).

### `scikit-learn`

- `ColumnTransformer`, `Pipeline`, `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`
  build leakage-safe preprocessing: median numeric imputation, scaling, most-frequent
  categorical imputation, and unknown-safe one-hot encoding (`features.py:14-44`).
- `LinearRegression()` is the interpretable baseline; `RandomForestRegressor` receives
  `n_estimators=100`, `min_samples_leaf=3`, `random_state=42`, `n_jobs=1`; and
  `GradientBoostingRegressor` receives `n_estimators=100`, `random_state=42`
  (`models.py:39-46`).
- `train_test_split(..., test_size=0.2, random_state=42)`, `KFold(n_splits=5,
  shuffle=True, random_state=42)`, and `cross_validate(..., scoring=SCORING, n_jobs=1)`
  create the fixed holdout and five-fold comparison (`models.py:60-68`).
- `mean_absolute_error`, `mean_squared_error`, and `r2_score` calculate MAE, RMSE and R2
  (`models.py:31-35`); `permutation_importance(..., n_repeats=5 or 10, random_state=42,
  scoring="r2", n_jobs=1)` produces feature importance (`models.py:113-118`).
- `KMeans(n_clusters=k, random_state=42, n_init=10)` and `silhouette_score` select route
  clusters after standardization (`clustering.py:38-53`).

### `streamlit`

`st.set_page_config` configures the page (`app.py:44`); `st.cache_data` and
`st.cache_resource` cache source data, model, route work and scores (`app.py:65-110`).
`st.sidebar`, `st.selectbox`, `st.multiselect`, `st.slider`, `st.metric`, `st.warning`,
`st.error`, `st.dataframe`, `st.plotly_chart`, `st.download_button`, `st.expander`,
`st.columns`, `st.tabs`, `st.title`, `st.header`, `st.subheader`, `st.caption`, `st.markdown`,
`st.spinner`, `st.stop`, `st.Page`, and `st.navigation` implement the dashboard and its
navigation (`app.py:124-759`). `streamlit.testing.v1.AppTest` was used for the non-browser
boot check.

### `plotly`

`plotly.express.pie`, `bar`, `line`, `scatter`, `scatter_geo`, `histogram`, and `box` create
dashboard figures. `plotly.graph_objects.Figure` and `go.Figure` support custom figures.
`Figure.update_layout` applies the dark theme and `Figure.update_geos` configures the factory
map (`app.py:117`, `app.py:543-548`).

### `joblib`

`joblib.dump(..., compress=3)` persists the pipeline and metadata; `joblib.load` restores
them (`models.py:100-110`). The artifact is `artifacts/best_model.joblib`.

### `pytest`

`pytest` collection discovers `tests/`, `legacy/tests/`, and root test probes. Test modules
use `pytest` fixtures and assertions in `tests/` and `scripts/smoke_test.py`. The exact
requested command was run; its result is recorded in section 10 and the final report.

### Legacy-only APIs

`scipy.stats.linregress`, `scipy.optimize.milp`, `LinearConstraint`, and `Bounds` occur only
under `legacy/`. `pulp` occurs only in legacy/test code. They are pinned for repository
compatibility but are not called by the active `run_pipeline.py` path.

## 4. Standard-library modules used

| Module | Use | Files |
|---|---|---|
| `__future__` | postponed annotations | `app.py`, `diagnostics/recommendation_count_py314.py` |
| `functools` | partial page callbacks | `app.py` |
| `hashlib` | assumption fingerprint | `config.py` |
| `json` | artifact, risk and report serialization | `app.py`, `config.py`, `data_prep.py`, `run_pipeline.py`, `simulation.py` |
| `logging` | diagnostic logging | `app.py`, `data_prep.py`, `run_pipeline.py`, `simulation.py` |
| `pathlib.Path` | repository and artifact paths | `app.py`, `config.py`, `data_prep.py` |
| `re` | text parsing/helpers | `app.py`, `data_prep.py` |
| `sys` | diagnostic path/runtime handling | `run_pipeline.py`, diagnostics |
| `os`, `time` | repository probe scripts | root probe scripts |

## 5. External services and data sources

### Runtime

The runtime source is the local `Nassau Candy Distributor.csv`. The app also reads local
`artifacts/` and Markdown reports. `data/reference/zip_centroids.csv` is checked in and
identifies `pgeocode 0.5.0 / GeoNames postal dataset` as its source, but active code does not
read it. `data/reference/natural_earth_admin0_110m.geojson` and
`data/reference/natural_earth_us_admin1_110m.geojson` are Natural Earth boundary files; active
`app.py` does not read them. Its factory map uses Plotly geographic rendering and
`FACTORY_COORDS` (`app.py:543-548`).

The required network grep over active Python/config/docs found no runtime `requests`,
`urllib`, `httpx`, `socket`, or `fetch` calls. URL strings occur only in documentation and
legacy source comments. Therefore the running app has no application-level runtime network
calls; Streamlit itself still serves its local HTTP/WebSocket interface.

### Build time

`pip install -r requirements.txt` obtains Python packages from the package index. The pipeline
reads the local CSV and writes local artifacts; it does not call a geocoder or remote data
service. `run_pipeline.py` must not be run as part of this test because it rewrites committed
outputs.

### Development and hosting

Streamlit Community Cloud hosts the deployed app from the GitHub repository and installs
`requirements.txt`. GitHub stores source and GitHub Actions runs `.github/workflows/tests.yml`
on push and pull request. The workflow tests Python `3.11`, `3.12`, and `3.14`, installs the
requirements, then runs `pytest -q`. The deployed URL redirected to Streamlit authentication
in the available non-browser fetch tool, so its visual behavior was not verified here.

## 6. Project API

Signatures below are the public functions/classes in the active modules. Internal underscore
helpers are omitted.

| Module | Public API | Inputs, output, files and callers |
|---|---|---|
| `config.py` | `division_nearest_factory(division)`; `resolve_factory(product_name, division)`; `normalize_product_name(name)`; `is_canadian(state_province)` | Resolve factory provenance, normalize names, and classify Canada. Called by `data_prep.py` and configuration checks. `Assumptions.assumption_version()` hashes non-empirical constants; called by `simulation.py` and `app.py`. |
| `data_prep.py` | `haversine_km(lat1, lon1, lat2, lon2)` | Returns vectorized kilometres; used by `load_and_audit`. |
| `data_prep.py` | `load_and_audit(path=DATA_PATH)` | Reads the source CSV; returns `(DataFrame, audit dict)` with cleaned rows, distances, ranks, dates and diagnostics. Called by `run_pipeline.py`, the app, and tests. |
| `data_prep.py` | `synthetic_product_check(products)` | Returns sorted source products; used by audit/test checks. |
| `data_prep.py` | `load_and_audit_safe(path=DATA_PATH)` | Returns `(frame, audit, error)` and is called by `app.py` to render recoverable errors. |
| `features.py` | `make_preprocessor(numeric=NUMERIC, categorical=CATEGORICAL)` | Returns the sklearn `ColumnTransformer` pipeline used by `models.py`. |
| `models.py` | `metrics(y_true, y_pred)` | Returns numeric R2/RMSE/MAE; called by `train_models`. |
| `models.py` | `build_models()` | Returns the three configured estimators; called by `train_models`. |
| `models.py` | `train_models(df, out_dir=ARTIFACT_DIR)` | Fits and cross-validates models; returns report, selected name and pipeline. |
| `models.py` | `select_best(report, tolerance=0.02)` | Applies CV-R2 then interpretability selection. |
| `models.py` | `persist_model(pipe, name, out_dir=ARTIFACT_DIR)` / `load_model(path=MODEL_ARTIFACT)` | Write/read `best_model.joblib`. Called by pipeline and app. |
| `models.py` | `permutation_importance_for(...)`; `per_product_residual_std(...)` | Produce feature importance and residual uncertainty frames for risk. |
| `models.py` | `train_and_persist(df, out_dir=ARTIFACT_DIR)` | Orchestrates training and writes model metrics, importance and residual CSVs. |
| `clustering.py` | `cluster_routes(df, min_orders=10, k_max=8)` | Returns route-lane cluster/exposure DataFrame; called by pipeline and cached app page. |
| `clustering.py` | `congested_region_products(df)` | Returns region/product congestion DataFrame; called by pipeline and app. |
| `simulation.py` | `parse_components(cell)` | Decodes stored risk JSON. |
| `simulation.py` | `candidate_moves(df, assumptions=Assumptions)` | Returns product/factory counterfactual rows using measured distances and assumption A1. |
| `simulation.py` | `risk_scores(...)` | Returns weighted risk rows from residual variance, support and distance components. |
| `simulation.py` | `score_products(...)` | Adds retain rows, risk, gains and ranking fields; called by app and pipeline. |
| `simulation.py` | `top_recommendations(moves, top_n=10)` | Returns the ranked shortlist. |
| `simulation.py` | `compute_kpis(...)`; `point_and_ci(...)` | Calculate KPI point values and bootstrap intervals. |
| `simulation.py` | `sensitivity(df, assumptions=Assumptions, iterations=300)` | Returns scenario sensitivity and stability frames. |
| `simulation.py` | `factory_optimiser_summary(df, moves)` | Summarizes factory capacity/assignment impact for the simulator. |
| `run_pipeline.py` | `markdown_table(frame, columns=None, decimals=3)`; `plural(n, noun, fmt="g")` | Report formatting helpers. |
| `run_pipeline.py` | `write_reports(...)` | Reads computed frames and writes `docs/research_paper.md` and `docs/executive_summary.md`. |
| `run_pipeline.py` | `main()` | Loads, trains, clusters, scores, writes all artifacts and reports. Not run per instruction. |
| `app.py` | `main()` and `page_*` functions | Streamlit entry point and page renderers; read source/artifacts and render UI. |

## 7. Pipeline-to-app contract

| App input | Required columns/keys | Writer |
|---|---|---|
| `Nassau Candy Distributor.csv` | Source headers mapped by `RENAME` in `data_prep.py` | External shipped source |
| `artifacts/best_model.joblib` | Joblib dict keys `model`, `name`, `features` | `models.persist_model` |
| `artifacts/per_product_residual_std.csv` | `product`, `residual_std`, `holdout_count` | `models.train_and_persist` |
| `artifacts/model_metrics.csv` | `model`, `cv_r2_mean` and reported metric columns | `models.train_and_persist` |
| `artifacts/feature_importance.csv` | `feature`, `importance_mean`, `importance_std` | `models.train_and_persist` |
| `artifacts/routes.csv` | route aggregation and cluster columns | `run_pipeline.main` |
| `artifacts/recommendations.csv` and related scenario CSVs | recommendation, gain, profit, risk and assumption columns | `run_pipeline.main` |
| `artifacts/kpis.json` and sensitivity JSON/CSV files | KPI point/CI and scenario sensitivity keys | `run_pipeline.main` |
| `data/reference/*` | No active `app.py` startup read. `config.py` defines `REFERENCE_DIR` and `ZIP_CENTROIDS_PATH`, but active preparation uses `STATE_CENTROIDS`. | Checked-in reference data |

The app startup guard is `load_and_audit_safe()` plus `MIN_VALID_FRACTION` in
`app.py:65-70` and `app.py:704-709`; it requires a usable source frame, then loads/trains
the model. Missing report artifacts are handled by page-level “not found” text rather than a
startup traceback.

## 8. Configuration

`config.Assumptions` (`config.py:158-202`) contains `freight_speed_km_day=900.0`,
`freight_cost_per_unit_per_1000km=0.15`, setup costs per factory, capability-gap values,
`min_orders_for_recommendation=3`, `min_material_days=1.0`, `capacity_multiplier=1.5`,
`speed_weight=0.5`, `monte_carlo_iterations=300`, `model_selection_tolerance=0.02`,
`risk_weight=0.30`, `iqr_factor=3.0`, and `kpi_bootstrap_iterations=500`.
`assumption_version()` fingerprints the distance-to-time/cost inputs.

`.streamlit/config.toml` selects the dark theme, disables usage statistics, sets CORS/XSRF
options, disables run-on-save, and sets a 200 MB upload limit. The CI workflow uses Python
3.11/3.12/3.14, installs requirements, and runs pytest. The README setup example uses `.venv`
and says to select the requirements-pinned or closest available Python on Community Cloud; it
does not state one single Python version. Its verification command is narrower than the
requested `pytest -q -p no:cacheprovider`.

## 9. Findings

1. `seaborn` and `tenacity` are pinned but never imported.
2. `pgeocode` is described in the ZIP metadata/research paper but is absent from
   `requirements.txt`; active code does not import it, so this is provenance/documentation
   drift rather than a current import failure.
3. The README describes a broader “Modules 1–4” dashboard, while current `app.py` exposes
   four Optimization Lab pages, two Analysis pages, and three Documentation pages. There is
   no `Evidence & Diagnostics` page, and the map is on Factory Intelligence rather than the
   Risk page. The README also says `python -m pytest scripts/smoke_test.py -v`, while CI runs
   `pytest -q`.

No active imports or calls implement a REST API, database client, socket client, or
authentication. The only HTTP service observed during testing was Streamlit's local server.

## 10. NOT VERIFIED

- Browser visual checks were not run: no browser tool/page was available in this session, and
  the deployed URL redirected through Streamlit authentication in `web_fetch`.
- Deployed-app checks (tabs, Gobstopper warning, Wonka Bar behavior, and map) are not verified.
- The exact requested pytest command failed during collection; it was not a passing suite.
- AppTest cannot judge visual overlap, truncation, chart appearance, borders, or load perception.
- `run_pipeline.py` was not run, as instructed; artifact freshness was not regenerated.
- The five-tab requirement, freight-speed/freight-cost sliders, and Evidence & Diagnostics page
  were not found in the active source and therefore cannot be reported as passing.
