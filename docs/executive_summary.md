# Executive summary: Nassau Candy factory allocation

**Leadership attributes long lead times to static factory assignments and suboptimal shipping distances. We tested that premise directly.** The distance differences between the five factories are real - the reducing alternatives save hundreds of kilometres - but the supplied order file cannot establish that they shorten delivery, and the analysis therefore supports controlled piloting rather than an unqualified network redesign.

| Decision | Recommendation | Why |
|---|---|---|
| Factory changes | Pilot the low-risk, distance-reducing moves; retain everything else for now | Distances are measured, lead-time deltas are assumptions, and the predictive model has no signal |
| Modelling status | Do not deploy any lead-time forecast | All three spec models return negative R² (best -0.0041 cross-validated) |
| Next step | Pilot selected routes with actual carrier, ship date, delivery date and freight invoice data | Converts assumed speed and cost into measured lane parameters |
| Data improvement | Record carrier, origin factory, measured transit days and freight charge | Without these the scenario engine stays assumption-driven |
| Risk review | Investigate the 2 high-risk and 39 medium-risk moves first | Their scores are dominated by prediction uncertainty and capability gaps, which a pilot measures |

### Headline numbers

* 9,747 order lines modelled; 313 duplicate rows removed; 134 (1.36%) removed by the 3.0 x IQR fence.
* 15 of 15 products have a distance-reducing alternative; 10 carry a recommendation.
* Lead-time reduction 0.0257%, profit stability 86.7%, coverage 100.0%, scenario confidence -0.0023.
* Distinct recommended factories in the top 10: 3 (Secret Factory, The Other Factory, Wicked Choccy's).
* Slow routes flagged: 43; congested region/product pairs: 8.

### Measured versus assumed

Measured from the source: order counts, units, gross profit, observed lead baselines and centroid-based distances. Assumed for scenarios: freight speed, freight cost, capability penalties and capacity limits. Every dashboard page that surfaces a lead-time or profit delta states this distinction.
