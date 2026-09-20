# 🚚 GRIDPOINT

Demand-Aware Warehouse Optimization & Network Resilience Platform.

# GRIDPOINT

**Live Demo:** [https://your-actual-gridpoint-url.streamlit.app](https://gridpoint.streamlit.app/)

> Demand-aware warehouse optimization and network resilience platform.

## What it solves

GridPoint helps an e-commerce company choose warehouse locations and assign neighborhoods while minimizing order-weighted delivery distance.

The optimization objective is:

`Total weighted delivery cost = Σ (daily_orders × distance_to_assigned_warehouse)`

## Current prototype

- CSV upload
- Sample Bengaluru neighborhood dataset
- Demand-weighted K-Means candidate warehouse placement
- Neighborhood-to-warehouse assignment
- Capacity constraints
- Maximum delivery radius
- Before vs after comparison
- Warehouse utilization
- What-If Lab:
  - demand change
  - warehouse failure
- Interactive map
- Deterministic AI Analyst starter tab

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit.

## CSV format

The uploaded CSV must contain:

- `neighborhood`
- `latitude`
- `longitude`
- `daily_orders`

See `data/neighborhoods.csv`.

## Important hackathon note

The current AI Analyst tab is a deterministic starter. During the hackathon, if the team adds a public AI API, update this README to name the API/model and explain exactly where it is used.

The hackathon rules require the AI component to be disclosed in the README and state that the team's core logic, integration, and problem-solving must be its own work.

## Suggested future improvements

1. Add a monetary delivery-rate model (₹/km/order).
2. Add explicit infrastructure cost per warehouse.
3. Improve constrained optimization beyond the weighted K-Means baseline.
4. Add scenario comparison charts.
5. Connect an allowed public AI API to explain computed results.
6. Add a true resilience score based on multiple failure scenarios.

