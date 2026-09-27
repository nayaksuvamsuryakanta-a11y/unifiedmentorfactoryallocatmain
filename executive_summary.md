# Executive summary: Nassau Candy factory allocation

**The data does not show a dependable delivery-time benefit from factory distance once shipping mode is considered.** Shipping mode predicts delivery timing much better than factory location, and temporal validation is weaker than a random split. Scenario estimates depend on assumed freight speed and freight cost, so the analysis does not justify moving products on its own.

| Decision | Recommendation | Why |
|---|---|---|
| Factory changes | Keep current assignments for now | 1 products met both the evidence and material-improvement thresholds |
| Next step | Pilot selected routes with actual carrier and invoice data | Current distance-to-time and freight-cost conversions are assumptions |
| Data improvement | Record carrier, origin, destination, ship date, delivery date, and freight charge | Enables direct measurement of transit and cost by route |
| Risk review | Investigate the highest-exposure slow routes in `artifacts/routes.csv` | Route profiles identify operational lanes for measurement |

**Measured from the source:** order counts, units, gross profit, repaired historical lead times, and centroid-based distances. **Assumed for scenarios:** freight speed, freight cost, capability penalties, and capacity limits. ZIP centroids covered 95.6% of retained rows; state/province centroids covered 4.4%.
