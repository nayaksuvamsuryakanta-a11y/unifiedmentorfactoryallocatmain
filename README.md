# Nassau Candy Factory Reallocation & Shipping Optimization

Decision support for testing whether assigning products to alternative factories could reduce delivery time without materially reducing gross profit. The backend separates facts calculated from the source data from scenario assumptions.

## Requirements and run

Pinned dependencies support Python 3.11, 3.12, and 3.14. Local verification used Python 3.14.0; CI covers all three versions. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_pipeline.py
python -m pytest -q
streamlit run app.py
```

The pipeline reads `Nassau Candy Distributor.csv` beside `run_pipeline.py`, validates and repairs the ship dates, trains the models, runs diagnostics and both optimization modes, and writes CSV/JSON outputs to `artifacts/`, plus `research_paper.md` and `executive_summary.md` at the project root. The dashboard reads checked-in artifacts directly and does not run the pipeline on launch. Regenerate artifacts with `python run_pipeline.py` after changing analysis code or assumptions.

The dashboard includes the filtered factory simulator, one-move comparison, catalogue and joint-assignment views, risk/route map, and evidence diagnostics. Its near-black palette and Carto dark basemap are configured locally. Monte Carlo draws and per-product choice-stability results are read from saved artifacts so dashboard interactions do not rerun the sensitivity simulation.

## Geocoding

ZIP/postal centroids are stored in `data/reference/zip_centroids.csv` with source metadata (`pgeocode 0.5.0 / GeoNames postal dataset`). The pipeline reads this static file only and makes no network calls. Unresolved postal codes use the user-supplied state/province centroid table in `config.py`. If rebuilding the reference file, pgeocode is optional and should only be used for that one-time offline acquisition; it is not a runtime dependency. Distances use Haversine great-circle calculations from centroids, not actual carrier routes.

## Interpretation

Measured/calculated from the source: order count, units, gross profit, repaired historical lead time, ship mode, destination centroid, and distance from the supplied factory coordinates. Assumptions configured in `config.py`: freight speed and cost, capability-gap penalty, evidence/materiality thresholds, capacity multiplier, ranking weight, and Monte Carlo iterations. A scenario's predicted lead/profit deltas depend on those assumptions and should be treated as planning estimates pending route pilots.

## Outputs

`artifacts/` includes prepared orders, random/CV/temporal model metrics, permutation importance, distance ablations and residual bins, mode-specific significance tests, route clusters, per-product recommendations, joint assignments, bootstrap KPI intervals, and Monte Carlo draws. The report files explain the findings and their limitations in plain language.
