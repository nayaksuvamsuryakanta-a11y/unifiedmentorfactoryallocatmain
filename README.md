# Nassau Candy Distributor — Factory Reallocation

Rebuilt from the source file `Nassau Candy Distributor.csv` (10,194 rows, 18 columns) for the
CodeAlpha internship task. The single source of truth is the shipped CSV; **no synthetic data is
generated anywhere in this project**.

The deliverable is a Streamlit app plus supporting documents.

## Layout

| Path | Purpose |
|---|---|
| `app.py` | Multi-page Streamlit dashboard (Modules 1–4, plus EDA & Models, Factory Intelligence, Research Paper, Executive Summary, Methodology & Model Card) |
| `.streamlit/config.toml` | Professional dark theme |
| `config.py` | Constants, factory coordinates, centroids, product→factory map, division fallback, risk weights, scenario assumptions |
| `data_prep.py` | Load, audit, clean, impute, geocode, IQR fence |
| `features.py` | Feature lists and sklearn preprocessing pipeline |
| `models.py` | Linear / Random Forest / Gradient Boosting, 5-fold CV, joblib persistence |
| `clustering.py` | KMeans lane clustering + congestion detection |
| `simulation.py` | Candidate moves, non-degenerate risk engine, scoring, KPIs, sensitivity |
| `run_pipeline.py` | Batch run that writes `artifacts/` and `docs/` |
| `scripts/smoke_test.py` | pytest suite asserting the Phase 8 acceptance checks |
| `docs/research_paper.md` | Full research paper |
| `docs/executive_summary.md` | Executive summary |
| `requirements.txt` | Pinned dependencies |
| `legacy/` | Superseded pre-rebuild code, tests and stale docs (nothing was deleted silently) |

## Windows setup

1. Open `factoryallocatmain` as the working directory.
2. Create and activate the virtual environment (once):

   ```cmd
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the pipeline so every artefact is current:

   ```cmd
   python run_pipeline.py
   ```

4. Launch the app from the project root:

   ```cmd
   streamlit run app.py
   ```

The app must be started from the project root so that relative paths such as
`artifacts/best_model.joblib` and `docs/research_paper.md` resolve.

If `artifacts/` is absent, `app.py` trains the model on first launch (cached with
`st.cache_resource`), so `streamlit run app.py` still works — but running the pipeline first
guarantees the research paper and executive summary are present.

## Streamlit Community Cloud deploy

1. Push this repository to a GitHub branch (`legacy/`, `.venv/`, `__pycache__/`,
   `.pytest_cache/` and `.gitignore` keep them out of the repo).
2. In the [Streamlit Community Cloud](https://share.streamlit.io) console create a new app and
   point it at the repository and branch, setting **Main file path** to `app.py`.
3. Set **Python version** to the one pinned in `requirements.txt` (or the closest available).
4. Under **Runtime** add a **Secrets** entry only if you want to pin the data path; otherwise the
   app discovers `Nassau Candy Distributor.csv` automatically.
5. Commit `requirements.txt` — Community Cloud installs it verbatim.
6. Deploy. The first build regenerates `artifacts/` because the cached model is absent, which can
   take a few minutes; subsequent visits hit the cache.

Caveat: Community Cloud containers are ephemeral, so `artifacts/` is rebuilt on cold start. Keep
`requirements.txt` pinned exactly as shipped — drifting versions (notably pandas 3.x and PuLP 4.x)
broke the codebase during development.

## Methodology

1. **Dates** are parsed strictly as DD-MM-YYYY. No reconstruction is performed — the raw
   `(Ship Date - Order Date)` difference, in days, *is* the lead time.
2. **Cleaning:** stray duplicate header rows, blank rows and field-count drift are diagnosed;
   exact duplicate rows are dropped using every column except `Row ID` as the identity key (the row
   number is a synthetic line number, not order identity); missing `Cost` is imputed as
   `Sales - Gross Profit`; `Postal Code` stays a string.
3. **Outliers:** a 3.0× IQR fence is applied to the four financial columns only. Lead time is
   never trimmed, because the brief's lead-time assertions must be measured on the untrimmed
   distribution. The percentage removed is reported.
4. **Distance:** Haversine great-circle distance from the incumbent origin factory to the
   State/Province centroid, over 61 embedded centroids (51 US states incl. Alaska/Hawaii + DC + 10
   Canadian provinces/territories present in the source).
5. **Origin factory:** the explicit spec map for 14 products; unlisted products such as Kazookles
   resolve through a division-based fallback (division → modal factory by majority vote over the
   explicit map), surfaced on the Factory Intelligence page.
6. **Modelling:** target = expected lead time from Product, Origin Factory, Destination Region and
   Ship Mode (+ Units, Distance). 80/20 split, seed 42, 5-fold shuffled CV. All three estimators
   score at or below zero because lead time in this file is a near-constant (~1320 days); Linear
   Regression is selected on interpretability. The negative values are reported verbatim rather
   than replaced.
7. **Counterfactuals:** because the model carries no predictive signal, alternative-factory lead
   time is `baseline + delta_distance / assumed_speed`, with every assumption printed wherever it
   is used.
8. **Risk:** `risk = 0.40·negProfit + 0.30·predVariance + 0.20·lowSupport + 0.10·distanceShift`,
   where `predVariance` is the best model's per-product hold-out residual standard deviation —
   deliberately *not* `|predicted − current|`. Thresholds: High > 0.50, Medium > 0.25.
9. **KPIs:** Lead Time Reduction (%), Profit Impact Stability (%), Scenario Confidence Score
   (best-model R²), Recommendation Coverage (%), each with a 90% bootstrap interval.

## Verification

```cmd
python -m pytest scripts/smoke_test.py -v
```

The suite asserts: the app boots without exception; row counts, duplicate removal and outlier
removal are all non-zero; the three lead-time assertions hold; the product count is 15 with no
synthetic products; all four sidebar controls re-render output; all four modules render real data
with a populated Risk panel; the KPI header shows all four metrics; the model table carries
RMSE/MAE/R² with CV for all three estimators and a justified best model; recommendations show
≥90% coverage, ≥3 distinct factories in the Top-10 and a positive lead-time reduction in every
Top-10 row.

## Submission checklist

Four URLs are required by the brief. Fill them in from your actual deployment environment — the
placeholders below are not real endpoints.

| # | Deliverable | URL (fill in) |
|---|---|---|
| 1 | Live Streamlit app | `_streamlit_app_url_` |
| 2 | Source repository | `_repository_url_` |
| 3 | Research paper (`docs/research_paper.md`) | `_paper_url_` |
| 4 | Executive summary (`docs/executive_summary.md`) | `_executive_summary_url_` |

## Honest findings

All three spec-mandated models score negative R² (best: Linear Regression, five-fold CV R²
−0.0041, holdout −0.0023, RMSE 263.29 days, MAE 211.54 days). The target is a near-constant, so
the best attainable prediction is the mean and every structured feature scores below it. This is
reported verbatim throughout — on the Models page, in the KPI header's Scenario Confidence Score,
in the research paper and in the executive summary — rather than concealed behind a fitted
surrogate. Every lead-time delta in the scenario engine is therefore labelled as assumption-driven
and shows the assumed freight speed it rests on.