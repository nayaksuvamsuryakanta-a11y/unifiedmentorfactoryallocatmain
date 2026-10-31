# Executive summary: Nassau Candy factory allocation

**Leadership attributes long lead times to static factory assignments and suboptimal shipping distances. We tested that premise directly; the data does not show a dependable delivery-time benefit from factory distance once shipping mode is considered.** Shipping mode predicts delivery timing much better than factory location, and temporal validation is weaker than a random split. Scenario estimates depend on assumed freight speed and freight cost, so the analysis does not justify moving products on its own.

| Decision | Recommendation | Why |
|---|---|---|
| Factory changes | Keep current assignments unless a pilot validates a shortlisted move | 0 top-ranked moves meet both default evidence and material-improvement thresholds |
| Next step | Pilot selected routes with actual carrier and invoice data | Current distance-to-time and freight-cost conversions are assumptions |
| Data improvement | Record carrier, origin, destination, ship date, delivery date, and freight charge | Enables direct measurement of transit and cost by route |
| Risk review | Investigate the highest-exposure slow routes in `artifacts/routes.csv` | Route profiles identify operational lanes for measurement |

### Break-even pilot shortlist

These moves qualify only if measured carrier speed is at or below the listed break-even value, so measure it in a pilot.

| product | current_factory | candidate_factory | orders | lead_gain | breakeven_freight_speed_km_day |
| --- | --- | --- | --- | --- | --- |
| Laffy Taffy | Sugar Shack | The Other Factory | 10 | 0.82 | 737.46 |
| Laffy Taffy | Sugar Shack | Secret Factory | 10 | 0.72 | 644.5 |
| Wonka Bar - Nutty Crunch Surprise | Lot's O' Nuts | Secret Factory | 1524 | 0.71 | 637.78 |
| Wonka Bar - Scrumdiddlyumptious | Lot's O' Nuts | Secret Factory | 1702 | 0.69 | 617.95 |
| Wonka Bar - Fudge Mallows | Lot's O' Nuts | Secret Factory | 1525 | 0.68 | 615.07 |

The single best-ranked move per product clears both configured gates for 0 products; any move that independently clears both gates for 1 product.

**Measured from the source:** order counts, units, gross profit, repaired historical lead times, and centroid-based distances. **Assumed for scenarios:** freight speed, freight cost, capability penalties, and capacity limits. ZIP centroids covered 95.6% of retained rows; state/province centroids covered 4.4%.
