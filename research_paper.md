# Factory Reallocation & Shipping Optimization Recommendation System
## Nassau Candy Distributor

## 1. Executive Summary

Nassau Candy Distributor operates a multi-factory network in which products are
assigned to incumbent factories serving customers across the United States and
Canada. Static assignments can create avoidable distance, transit-time, and
profit trade-offs, particularly when product demand, destination mix, or shipping
mode changes.

This project delivers a decision-support application that combines data
preparation, predictive modeling, factory reallocation simulation, and
priority-based recommendations. The system calculates factory-to-customer
distances, estimates operational transit time, evaluates each product at the four
alternate factories, and ranks moves according to the user's balance between
speed and profit. A Streamlit dashboard makes the analysis accessible to
operations and leadership users without requiring code execution.

The result moves the organization from descriptive reporting toward prescriptive
analytics: instead of only showing historical orders, the system evaluates
specific factory alternatives and their expected operational and financial
impact.

## 2. Data Preparation and Exploratory Analysis

The source dataset contains **10,194 order rows** with order and ship dates,
shipping mode, destination state or province, region, product, sales, units,
gross profit, and cost. Dates are parsed with the required `DD-MM-YYYY` format.
The original observed lead-time field is calculated as:

```text
Lead_Time = Ship Date - Order Date
```

The data contains a material date anomaly. **The observed Ship Dates in the raw
data were corrupted (offset by years). To maintain methodological integrity, we
retained the observed data for reporting but engineered a deterministic,
distance-based 'Estimated Transit Days' proxy to power the operational
simulation.** The observed `Lead_Time` is therefore preserved as a data-quality
reference and is not presented as a reliable operational transit prediction.

Customer locations are represented by state or province centroids. For each
order, the system maps the destination to latitude and longitude and calculates
great-circle distance from the applicable factory using the Haversine formula.
The five factory coordinates are:

- Lot's O' Nuts
- Wicked Choccy's
- Sugar Shack
- Secret Factory
- The Other Factory

Product names are normalized to handle source spelling variation, including
inconsistent spacing in one Wonka Bar name. Product-to-factory mappings are
applied from the project specification; `Kazookles` is handled through its
division fallback. The prepared data contains no unresolved customer distances.

For operational what-if analysis, `Estimated_Transit_Days` is generated from
physical and service features rather than corrupted dates:

```text
Base_Days = Distance_km / 800
Estimated_Transit_Days =
    round(Base_Days × Ship_Mode_Multiplier + small seeded noise)
```

The ship-mode multipliers are 0.1 for Same Day, 1.0 for First Class, 1.5 for
Second Class, and 2.0 for Standard Class. Values are bounded to a minimum of
one day. This is explicitly a scenario proxy, not a reconstruction of actual
historical transit performance.

## 3. Predictive Modeling

The proxy model uses the following features:

- Ship Mode
- Region
- Product Name
- Origin Factory
- Distance in kilometers

Categorical variables are one-hot encoded with unknown-category handling.
`Distance_km` is passed as a numerical feature. The preprocessing transformer
and regressor are combined in one scikit-learn `Pipeline`, preventing training
and inference from using inconsistent transformations.

The data is split into training and testing partitions using an 80/20 split with
`random_state=42`. Three regressors are compared on the held-out test set using
RMSE, MAE, and R²:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.8353 | 0.6338 | 0.8984 |
| Random Forest | 0.3937 | 0.2854 | 0.9774 |
| Gradient Boosting | 0.3874 | 0.2962 | 0.9782 |

Gradient Boosting is selected because it has the highest R². The fitted
preprocessor and model are persisted together with `joblib` in
`best_model.pkl`, allowing the Streamlit application to load the model without
retraining on every interaction.

The high proxy-model scores should be interpreted correctly: they demonstrate
that the engineered operational estimate is internally predictable from its
generating features. They do not prove that the corrupted historical dates
could have been predicted accurately.

## 4. Simulation and Optimization Logic

For a selected product, the scenario engine identifies its current factory and
creates four counterfactual rows, one for each alternate factory. For every
candidate, it:

1. Recalculates the customer distance from the candidate factory.
2. Rebuilds the feature row with the candidate `Origin_Factory` and
   `Distance_km`.
3. Uses the persisted proxy pipeline to estimate new transit days.
4. Calculates transit-day reduction relative to the current assignment.
5. Estimates profit impact from the distance-related shipping-cost change.

The application exposes an **Optimization Priority** slider from 0.0 to 1.0.
The recommendation score is:

```text
Score = (Priority × Transit_Days_Reduced)
      + ((1 - Priority) × Profit_Impact)
```

A value of 1.0 emphasizes speed, while 0.0 emphasizes profit impact. Intermediate
values allow users to explore balanced strategies. Recommendations are sorted
by the resulting score, and moves with negative estimated profit impact are
highlighted in the Risk & Impact Panel.

## 5. Business Recommendations for Nassau Candy Leadership

### 5.1 Pilot distance-reducing chocolate reallocations

Use the Recommendation Dashboard to identify chocolate products whose alternate
factory is materially closer to their customer mix. In the validated example,
moving `Wonka Bar - Milk Chocolate` from Wicked Choccy's to Secret Factory
reduced estimated transit time by approximately **0.82 days** and produced a
positive estimated profit impact under the scenario cost assumption. Leadership
should pilot the highest-ranked moves in a controlled product-region lane before
making a network-wide change.

### 5.2 Prioritize high-volume Standard Class lanes

Standard Class shipments receive the largest service-time multiplier in the
proxy, while high-volume lanes create the greatest aggregate opportunity for
distance and cost improvement. Operations should filter the dashboard to
Standard Class orders, review products with substantial unit volume, and use the
priority slider to compare speed-led and profit-led decisions.

### 5.3 Use risk flags and staged implementation

Do not approve every speed-improving move automatically. A longer candidate
distance can create a negative profit impact even when the proxy predicts a
service benefit. Leadership should review every warning in the Risk & Impact
Panel, validate freight rates and factory capacity with operations, and begin
with a limited pilot. Actual carrier transit times, freight invoices, service
levels, and post-move performance should then be collected to replace the proxy
with an empirically validated target.

## 6. Conclusion

The solution provides Nassau Candy with a modular path from descriptive to
prescriptive analytics. Data preparation standardizes dates, factory mappings,
centroids, and distances. The modeling layer supplies a reusable operational
transit-time proxy, while the simulation engine converts product and factory
choices into comparable transit and profit scenarios. The Streamlit dashboard
then makes those scenarios interactive through persistent filters, four focused
analysis tabs, and a speed-versus-profit control.

Because the raw ship dates are corrupted, the current recommendations are
decision-support scenarios rather than claims about historical actual transit
performance. The next production improvement is to obtain corrected operational
dates and validated freight-cost data, retrain against observed transit outcomes,
and compare realized pilot results with dashboard predictions.
