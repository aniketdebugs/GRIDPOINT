# app_v3.py
"""
GRIDPOINT v3
Demand-Aware Warehouse Optimization & Network Resilience

Clean, self-contained Streamlit implementation covering:
Core:
- Sample CSV / Upload CSV / Manual entry
- Neighborhood visualization
- 1..K warehouse optimization
- Demand-weighted delivery distance/cost
- Assignment table and baseline comparison

Stage A:
- Multiple warehouses
- Warehouse capacity
- Maximum service radius
- Demand-change what-if
- Infrastructure-cost trade-off

Stage B:
- Vehicle types
- Fuel-cost model
- Traffic-dependent delivery time
- Network resilience / single-warehouse failure simulation
- Warehouse utilization
- Cost breakdown
- What-if stress test

Expected CSV columns:
latitude, longitude, daily_orders
Optional:
neighborhood, name, id
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="GRIDPOINT | Logistics Intelligence",
    page_icon="📍",
    layout="wide",
)

st.title("📍 GRIDPOINT")
st.caption("Demand-Aware Warehouse Optimization & Network Resilience")


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
VEHICLES = {
    "Bike": {"capacity": 20, "cost_per_km": 3.0, "speed_kmph": 30.0},
    "Van": {"capacity": 100, "cost_per_km": 8.0, "speed_kmph": 35.0},
    "Truck": {"capacity": 500, "cost_per_km": 15.0, "speed_kmph": 30.0},
}

TRAFFIC_MULTIPLIER = {
    "Low": 1.0,
    "Medium": 1.3,
    "High": 1.7,
}


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def sample_data() -> pd.DataFrame:
    rows = [
        ("Indiranagar", 12.9784, 77.6408, 140),
        ("Koramangala", 12.9352, 77.6245, 180),
        ("HSR Layout", 12.9116, 77.6389, 120),
        ("BTM Layout", 12.9166, 77.6101, 155),
        ("Jayanagar", 12.9250, 77.5938, 110),
        ("JP Nagar", 12.9063, 77.5857, 95),
        ("Banashankari", 12.9255, 77.5468, 90),
        ("Rajajinagar", 12.9910, 77.5530, 130),
        ("Malleshwaram", 13.0035, 77.5700, 105),
        ("Yeshwanthpur", 13.0280, 77.5400, 125),
        ("Hebbal", 13.0358, 77.5970, 100),
        ("Whitefield", 12.9698, 77.7500, 200),
        ("Marathahalli", 12.9591, 77.6974, 175),
        ("Bellandur", 12.9250, 77.6764, 190),
        ("Electronic City", 12.8452, 77.6602, 160),
        ("Kengeri", 12.9141, 77.4827, 75),
        ("Peenya", 13.0320, 77.5270, 85),
        ("RT Nagar", 13.0196, 77.5946, 80),
        ("Frazer Town", 13.0005, 77.6135, 70),
        ("Shivajinagar", 12.9857, 77.6057, 115),
    ]
    return pd.DataFrame(
        rows,
        columns=["neighborhood", "latitude", "longitude", "daily_orders"],
    )


def normalize_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    aliases = {
        "lat": "latitude",
        "latitude": "latitude",
        "lon": "longitude",
        "lng": "longitude",
        "long": "longitude",
        "longitude": "longitude",
        "orders": "daily_orders",
        "daily_order": "daily_orders",
        "daily_orders": "daily_orders",
        "demand": "daily_orders",
        "name": "neighborhood",
        "neighbourhood": "neighborhood",
        "neighborhood": "neighborhood",
    }

    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in aliases:
            rename[col] = aliases[key]

    df = df.rename(columns=rename)

    required = {"latitude", "longitude", "daily_orders"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    if "neighborhood" not in df.columns:
        df["neighborhood"] = [f"N{i + 1}" for i in range(len(df))]

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["daily_orders"] = pd.to_numeric(df["daily_orders"], errors="coerce")

    df = df.dropna(
        subset=["latitude", "longitude", "daily_orders"]
    ).copy()

    df = df[df["daily_orders"] >= 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows remain after cleaning the dataset.")

    df["neighborhood"] = df["neighborhood"].astype(str)
    df["id"] = np.arange(len(df))

    return df[
        ["id", "neighborhood", "latitude", "longitude", "daily_orders"]
    ]


# ---------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------
def haversine_matrix(df: pd.DataFrame) -> np.ndarray:
    lat = np.radians(df["latitude"].to_numpy(dtype=float))
    lon = np.radians(df["longitude"].to_numpy(dtype=float))

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat[:, None])
        * np.cos(lat[None, :])
        * np.sin(dlon / 2.0) ** 2
    )

    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def distance_to_warehouses(
    df: pd.DataFrame,
    warehouse_indices: list[int],
) -> np.ndarray:
    lat1 = np.radians(df["latitude"].to_numpy(dtype=float))
    lon1 = np.radians(df["longitude"].to_numpy(dtype=float))

    lat2 = lat1[warehouse_indices]
    lon2 = lon1[warehouse_indices]

    dlat = lat1[:, None] - lat2[None, :]
    dlon = lon1[:, None] - lon2[None, :]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1[:, None])
        * np.cos(lat2[None, :])
        * np.sin(dlon / 2.0) ** 2
    )

    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------
# Warehouse optimization
# ---------------------------------------------------------------------
def choose_initial_warehouses(
    df: pd.DataFrame,
    k: int,
    distances: np.ndarray,
) -> list[int]:
    weights = df["daily_orders"].to_numpy(dtype=float)
    first = int(np.argmax(weights))
    selected = [first]

    while len(selected) < min(k, len(df)):
        nearest = distances[:, selected].min(axis=1)
        score = nearest * (weights + 1.0)
        score[selected] = -1
        selected.append(int(np.argmax(score)))

    return selected


def optimize_warehouses(
    df: pd.DataFrame,
    k: int,
    distances: np.ndarray,
    iterations: int = 5,
) -> list[int]:
    k = max(1, min(int(k), len(df)))

    selected = choose_initial_warehouses(df, k, distances)
    weights = df["daily_orders"].to_numpy(dtype=float)

    for _ in range(iterations):
        current = selected.copy()
        changed = False

        current_distances = distances[:, current]
        labels = np.argmin(current_distances, axis=1)

        for warehouse_no in range(k):
            members = np.where(labels == warehouse_no)[0]

            if len(members) == 0:
                continue

            best = current[warehouse_no]
            best_cost = float("inf")

            for candidate in members:
                cost = float(
                    np.sum(
                        weights[members]
                        * distances[members, candidate]
                    )
                )

                if cost < best_cost:
                    best_cost = cost
                    best = int(candidate)

            if best != current[warehouse_no]:
                current[warehouse_no] = best
                changed = True

        # Ensure warehouse locations are unique.
        if len(set(current)) < k:
            used = set()
            fixed = []

            for idx in current:
                if idx not in used:
                    fixed.append(idx)
                    used.add(idx)
                else:
                    nearest_existing = (
                        distances[:, fixed].min(axis=1)
                        if fixed
                        else np.zeros(len(df))
                    )
                    score = nearest_existing * (weights + 1.0)
                    score[list(used)] = -1
                    replacement = int(np.argmax(score))
                    fixed.append(replacement)
                    used.add(replacement)

            current = fixed
            changed = True

        selected = current

        if not changed:
            break

    return selected


# ---------------------------------------------------------------------
# Assignment with capacity + radius
# ---------------------------------------------------------------------
@dataclass
class AssignmentResult:
    assignments: np.ndarray
    distance_km: np.ndarray
    unserved: np.ndarray
    used_capacity: np.ndarray


def assign_neighborhoods(
    df: pd.DataFrame,
    warehouse_indices: list[int],
    capacities: list[float | None],
    radius_km: float | None,
) -> AssignmentResult:
    n = len(df)
    k = len(warehouse_indices)

    distances = distance_to_warehouses(
        df,
        warehouse_indices,
    )

    orders = df["daily_orders"].to_numpy(dtype=float)

    assignments = np.full(n, -1, dtype=int)
    used = np.zeros(k, dtype=float)

    # Large-demand neighborhoods are assigned first.
    order_indices = np.argsort(-orders)

    for i in order_indices:
        feasible = []

        for w in range(k):
            distance = float(distances[i, w])

            if radius_km is not None and distance > radius_km:
                continue

            capacity = capacities[w]

            if (
                capacity is not None
                and used[w] + orders[i] > capacity + 1e-9
            ):
                continue

            feasible.append(w)

        if feasible:
            chosen = min(
                feasible,
                key=lambda w: float(distances[i, w]),
            )
            assignments[i] = chosen
            used[chosen] += orders[i]

    selected_distance = np.full(n, np.nan)

    for i in range(n):
        if assignments[i] >= 0:
            selected_distance[i] = distances[
                i,
                assignments[i],
            ]

    return AssignmentResult(
        assignments=assignments,
        distance_km=selected_distance,
        unserved=assignments < 0,
        used_capacity=used,
    )


# ---------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------
def delivery_cost(
    df: pd.DataFrame,
    result: AssignmentResult,
    cost_per_km: float,
) -> float:
    served = ~result.unserved
    orders = df["daily_orders"].to_numpy(dtype=float)

    return float(
        np.sum(
            orders[served]
            * result.distance_km[served]
            * float(cost_per_km)
        )
    )


def fuel_cost(
    optimized_distance: float,
    fuel_price: float,
    efficiency: float,
) -> float:
    return (
        float(optimized_distance)
        / max(float(efficiency), 0.1)
        * float(fuel_price)
    )


def weighted_centroid_baseline(df: pd.DataFrame) -> float:
    weights = df["daily_orders"].to_numpy(dtype=float)
    total = max(float(weights.sum()), 1.0)

    centroid_lat = float(
        np.sum(df["latitude"].to_numpy(dtype=float) * weights)
        / total
    )
    centroid_lon = float(
        np.sum(df["longitude"].to_numpy(dtype=float) * weights)
        / total
    )

    lat1 = np.radians(df["latitude"].to_numpy(dtype=float))
    lon1 = np.radians(df["longitude"].to_numpy(dtype=float))
    lat2 = math.radians(centroid_lat)
    lon2 = math.radians(centroid_lon)

    dlat = lat1 - lat2
    dlon = lon1 - lon2

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * math.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    distances = (
        6371.0
        * 2.0
        * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )

    return float(
        np.sum(
            df["daily_orders"].to_numpy(dtype=float)
            * distances
        )
    )


def demand_scenario(
    df: pd.DataFrame,
    change_pct: float,
) -> pd.DataFrame:
    out = df.copy()
    multiplier = 1.0 + float(change_pct) / 100.0

    out["daily_orders"] = np.maximum(
        0.0,
        np.round(
            out["daily_orders"] * multiplier,
            2,
        ),
    )

    return out


# ---------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------
def failure_analysis(
    df: pd.DataFrame,
    warehouse_indices: list[int],
    capacities: list[float | None],
    radius_km: float | None,
    cost_per_km: float,
) -> pd.DataFrame:
    rows = []

    base_result = assign_neighborhoods(
        df,
        warehouse_indices,
        capacities,
        radius_km,
    )

    base_cost = delivery_cost(
        df,
        base_result,
        cost_per_km,
    )

    for failed in range(len(warehouse_indices)):
        survivors = [
            i
            for i in range(len(warehouse_indices))
            if i != failed
        ]

        if not survivors:
            rows.append(
                {
                    "Scenario": f"Warehouse {failed + 1} fails",
                    "Scenario cost (₹)": np.nan,
                    "Extra cost (₹)": np.nan,
                    "Unserved orders": float(
                        df["daily_orders"].sum()
                    ),
                }
            )
            continue

        surviving_warehouses = [
            warehouse_indices[i]
            for i in survivors
        ]
        surviving_capacities = [
            capacities[i]
            for i in survivors
        ]

        result = assign_neighborhoods(
            df,
            surviving_warehouses,
            surviving_capacities,
            radius_km,
        )

        scenario_cost = delivery_cost(
            df,
            result,
            cost_per_km,
        )

        unserved = float(
            df.loc[
                result.unserved,
                "daily_orders",
            ].sum()
        )

        rows.append(
            {
                "Scenario": f"Warehouse {failed + 1} fails",
                "Scenario cost (₹)": scenario_cost,
                "Extra cost (₹)": scenario_cost - base_cost,
                "Unserved orders": unserved,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Network Configuration")

    input_mode = st.radio(
        "Data input method",
        [
            "Sample Dataset",
            "Upload CSV",
            "Manual Entry",
        ],
    )

    uploaded_file = None

    if input_mode == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload neighborhood CSV",
            type=["csv"],
        )

    st.subheader("🏭 Warehouse Model")

    warehouse_count = st.slider(
        "Number of warehouses",
        min_value=1,
        max_value=8,
        value=2,
    )

    capacity_enabled = st.checkbox(
        "Enable warehouse capacity",
        value=True,
    )

    if capacity_enabled:
        capacity_value = st.number_input(
            "Capacity per warehouse (orders/day)",
            min_value=1.0,
            max_value=100000.0,
            value=1800.0,
            step=50.0,
        )
        capacities = [
            float(capacity_value)
            for _ in range(warehouse_count)
        ]
    else:
        capacities = [
            None
            for _ in range(warehouse_count)
        ]

    radius_enabled = st.checkbox(
        "Enable maximum service radius",
        value=True,
    )

    if radius_enabled:
        radius_km = st.number_input(
            "Maximum service radius (km)",
            min_value=0.1,
            max_value=1000.0,
            value=15.0,
            step=0.5,
        )
    else:
        radius_km = None

    st.subheader("🚚 Vehicle Model")

    vehicle_type = st.selectbox(
        "Vehicle type",
        list(VEHICLES.keys()),
    )

    vehicle = VEHICLES[vehicle_type]

    st.caption(
        f"Capacity: {vehicle['capacity']:.0f} orders/trip • "
        f"Operating cost: ₹{vehicle['cost_per_km']:.1f}/km • "
        f"Average speed: {vehicle['speed_kmph']:.0f} km/h"
    )

    st.subheader("⛽ Fuel Model")

    fuel_price = st.number_input(
        "Fuel price (₹/L)",
        min_value=50.0,
        max_value=200.0,
        value=100.0,
        step=1.0,
    )

    fuel_efficiency = st.number_input(
        "Vehicle efficiency (km/L)",
        min_value=1.0,
        max_value=100.0,
        value=15.0,
        step=1.0,
    )

    st.subheader("🚦 Traffic Conditions")

    traffic_level = st.select_slider(
        "Traffic",
        options=["Low", "Medium", "High"],
        value="Medium",
    )

    traffic_multiplier = TRAFFIC_MULTIPLIER[traffic_level]

    st.subheader("📈 Demand Scenario")

    demand_change = st.slider(
        "Demand change (%)",
        min_value=-50,
        max_value=100,
        value=0,
        step=5,
    )

    st.subheader("🏗️ Infrastructure")

    infrastructure_cost = st.number_input(
        "Infrastructure cost per warehouse (₹)",
        min_value=0.0,
        max_value=10000000.0,
        value=5000.0,
        step=500.0,
    )


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
try:
    if input_mode == "Upload CSV":
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV loaded.")
        else:
            raw_df = sample_data()
            st.sidebar.info(
                "No CSV selected. Using the sample dataset."
            )

    elif input_mode == "Manual Entry":
        raw_df = sample_data().head(5).copy()

        raw_df = st.data_editor(
            raw_df,
            num_rows="dynamic",
            width="stretch",
        )

    else:
        raw_df = sample_data()

    df = normalize_dataframe(raw_df)

except Exception as exc:
    st.error(f"Data loading error: {exc}")
    st.stop()


# ---------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------
scenario_df = demand_scenario(
    df,
    demand_change,
)

warehouse_count = min(
    warehouse_count,
    len(scenario_df),
)

capacities = capacities[:warehouse_count]

distance_matrix = haversine_matrix(
    scenario_df,
)

warehouse_indices = optimize_warehouses(
    scenario_df,
    warehouse_count,
    distance_matrix,
)

assignment = assign_neighborhoods(
    scenario_df,
    warehouse_indices,
    capacities,
    radius_km,
)

served = ~assignment.unserved

total_orders = float(
    scenario_df["daily_orders"].sum()
)

served_orders = float(
    scenario_df.loc[
        served,
        "daily_orders",
    ].sum()
)

unserved_orders = float(
    scenario_df.loc[
        assignment.unserved,
        "daily_orders",
    ].sum()
)

optimized_distance = float(
    np.sum(
        scenario_df.loc[
            served,
            "daily_orders",
        ].to_numpy()
        * assignment.distance_km[served]
    )
)

optimized_delivery_cost = delivery_cost(
    scenario_df,
    assignment,
    vehicle["cost_per_km"],
)

optimized_fuel_cost = fuel_cost(
    optimized_distance,
    fuel_price,
    fuel_efficiency,
)

optimized_infrastructure_cost = (
    float(warehouse_count)
    * float(infrastructure_cost)
)

optimized_total_cost = (
    optimized_delivery_cost
    + optimized_fuel_cost
    + optimized_infrastructure_cost
)

baseline_distance = weighted_centroid_baseline(
    scenario_df,
)

baseline_delivery_cost = (
    baseline_distance
    * float(vehicle["cost_per_km"])
)

base_time_hours = (
    optimized_distance
    / max(float(vehicle["speed_kmph"]), 0.1)
)

traffic_adjusted_minutes = (
    base_time_hours
    * traffic_multiplier
    * 60.0
)


# ---------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Daily orders",
    f"{total_orders:,.0f}",
)

m2.metric(
    "Served orders",
    f"{served_orders:,.0f}",
)

m3.metric(
    "Unserved orders",
    f"{unserved_orders:,.0f}",
)

m4.metric(
    "Total network cost",
    f"₹{optimized_total_cost:,.0f}",
)

if demand_change != 0:
    st.info(
        f"Demand scenario: {demand_change:+.0f}% "
        "from the original dataset."
    )


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tabs = st.tabs(
    [
        "🗺️ Network",
        "📊 Optimization",
        "🚚 Operations",
        "🛡️ Resilience",
        "🧪 What-if Lab",
    ]
)


# =====================================================================
# Network
# =====================================================================
with tabs[0]:
    st.subheader("Network Map")

    map_df = scenario_df[
        ["latitude", "longitude", "daily_orders"]
    ].copy()

    st.map(
        map_df,
        latitude="latitude",
        longitude="longitude",
        size="daily_orders",
        zoom=10,
    )

    warehouse_rows = []

    for w, idx in enumerate(warehouse_indices):
        assigned = float(
            assignment.used_capacity[w]
        )

        capacity = capacities[w]

        if capacity is None:
            utilization = 0.0
            capacity_label = "Unlimited"
        else:
            utilization = (
                assigned
                / max(float(capacity), 1.0)
                * 100.0
            )
            capacity_label = f"{float(capacity):,.0f}"

        warehouse_rows.append(
            {
                "Warehouse": f"W{w + 1}",
                "Location": scenario_df.loc[
                    idx,
                    "neighborhood",
                ],
                "Latitude": scenario_df.loc[
                    idx,
                    "latitude",
                ],
                "Longitude": scenario_df.loc[
                    idx,
                    "longitude",
                ],
                "Assigned orders": assigned,
                "Capacity": capacity_label,
                "Utilization %": utilization,
            }
        )

    st.markdown("### Warehouse utilization")

    st.dataframe(
        pd.DataFrame(
            warehouse_rows
        ).style.format(
            {
                "Assigned orders": "{:,.0f}",
                "Utilization %": "{:.1f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    if unserved_orders > 0:
        st.warning(
            "Some demand is unserved because the selected "
            "capacity/radius constraints are restrictive."
        )
    else:
        st.success(
            "All demand is assigned under the selected constraints."
        )


# =====================================================================
# Optimization
# =====================================================================
with tabs[1]:
    st.subheader("Optimization Results")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Optimized weighted distance",
        f"{optimized_distance:,.1f}",
    )

    c2.metric(
        "Baseline weighted distance",
        f"{baseline_distance:,.1f}",
    )

    improvement = (
        100.0
        * (
            1.0
            - optimized_distance
            / max(baseline_distance, 1e-9)
        )
    )

    c3.metric(
        "Distance change vs baseline",
        f"{improvement:.1f}%",
    )

    rows = []

    for i, row in scenario_df.iterrows():
        warehouse_no = int(
            assignment.assignments[i]
        )

        if warehouse_no >= 0:
            warehouse_label = (
                f"W{warehouse_no + 1}"
            )
            distance = float(
                assignment.distance_km[i]
            )
            status = "Served"
        else:
            warehouse_label = "Unserved"
            distance = np.nan
            status = "Unserved"

        rows.append(
            {
                "Neighborhood": row["neighborhood"],
                "Daily orders": row["daily_orders"],
                "Warehouse": warehouse_label,
                "Distance (km)": distance,
                "Status": status,
            }
        )

    st.markdown("### Neighborhood assignments")

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Baseline vs optimized cost")

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Baseline delivery cost",
                "Optimized delivery cost",
                "Delivery-cost difference",
            ],
            "Cost (₹)": [
                baseline_delivery_cost,
                optimized_delivery_cost,
                baseline_delivery_cost
                - optimized_delivery_cost,
            ],
        }
    )

    st.dataframe(
        comparison.style.format(
            {"Cost (₹)": "₹{:,.2f}"}
        ),
        width="stretch",
        hide_index=True,
    )


# =====================================================================
# Operations / Stage B
# =====================================================================
with tabs[2]:
    st.subheader("🚚 Vehicle, Fuel & Traffic Model")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Delivery cost",
        f"₹{optimized_delivery_cost:,.0f}",
    )

    c2.metric(
        "Fuel cost",
        f"₹{optimized_fuel_cost:,.0f}",
    )

    c3.metric(
        "Estimated delivery time",
        f"{traffic_adjusted_minutes:,.0f} min",
    )

    st.markdown("### Cost breakdown")

    breakdown = pd.DataFrame(
        {
            "Component": [
                "Delivery",
                "Fuel",
                "Infrastructure",
            ],
            "Cost (₹)": [
                optimized_delivery_cost,
                optimized_fuel_cost,
                optimized_infrastructure_cost,
            ],
        }
    )

    st.dataframe(
        breakdown.style.format(
            {"Cost (₹)": "₹{:,.2f}"}
        ),
        width="stretch",
        hide_index=True,
    )

    st.info(
        f"Traffic = {traffic_level}; "
        f"time multiplier = ×{traffic_multiplier:.1f}."
    )

    st.markdown("### Vehicle types")

    vehicle_table = pd.DataFrame(
        [
            {
                "Vehicle": name,
                "Capacity (orders/trip)": data["capacity"],
                "Operating cost (₹/km)": data["cost_per_km"],
                "Average speed (km/h)": data["speed_kmph"],
            }
            for name, data in VEHICLES.items()
        ]
    )

    st.dataframe(
        vehicle_table,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Fuel cost = optimized distance ÷ efficiency × fuel price."
    )


# =====================================================================
# Resilience / Stage B
# =====================================================================
with tabs[3]:
    st.subheader("🛡️ Network Resilience Lab")

    st.write(
        "Each scenario removes one warehouse and attempts to "
        "reassign its demand to the surviving warehouses."
    )

    if warehouse_count < 2:
        st.warning(
            "Select at least 2 warehouses to run failure scenarios."
        )
    else:
        resilience_df = failure_analysis(
            scenario_df,
            warehouse_indices,
            capacities,
            radius_km,
            vehicle["cost_per_km"],
        )

        st.dataframe(
            resilience_df.style.format(
                {
                    "Scenario cost (₹)": "₹{:,.2f}",
                    "Extra cost (₹)": "₹{:,.2f}",
                    "Unserved orders": "{:,.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        max_unserved = float(
            resilience_df[
                "Unserved orders"
            ].fillna(0).max()
        )

        if max_unserved > 0:
            st.warning(
                "At least one single-warehouse failure "
                "leaves some demand unserved."
            )
        else:
            st.success(
                "All tested single-warehouse failures remain "
                "serviceable under the selected constraints."
            )

    st.markdown("### Resilience interpretation")

    st.caption(
        "The resilience module is a deterministic stress test. "
        "It does not claim to predict real-world disruptions."
    )


# =====================================================================
# What-if Lab
# =====================================================================
with tabs[4]:
    st.subheader("🧪 What-if Lab")

    what_if_demand = st.slider(
        "Demand change for stress test (%)",
        min_value=-50,
        max_value=100,
        value=20,
        step=5,
    )

    what_if_traffic = st.select_slider(
        "Traffic scenario",
        options=["Low", "Medium", "High"],
        value="High",
    )

    stress_df = demand_scenario(
        df,
        what_if_demand,
    )

    stress_distance_matrix = haversine_matrix(
        stress_df,
    )

    stress_warehouses = optimize_warehouses(
        stress_df,
        warehouse_count,
        stress_distance_matrix,
    )

    stress_assignment = assign_neighborhoods(
        stress_df,
        stress_warehouses,
        capacities,
        radius_km,
    )

    stress_delivery_cost = delivery_cost(
        stress_df,
        stress_assignment,
        vehicle["cost_per_km"],
    )

    stress_served = (
        ~stress_assignment.unserved
    )

    stress_distance = float(
        np.sum(
            stress_df.loc[
                stress_served,
                "daily_orders",
            ].to_numpy()
            * stress_assignment.distance_km[
                stress_served
            ]
        )
    )

    stress_fuel = fuel_cost(
        stress_distance,
        fuel_price,
        fuel_efficiency,
    )

    stress_infrastructure = (
        float(warehouse_count)
        * float(infrastructure_cost)
    )

    stress_total = (
        stress_delivery_cost
        + stress_fuel
        + stress_infrastructure
    )

    stress_time = (
        stress_distance
        / max(float(vehicle["speed_kmph"]), 0.1)
        * TRAFFIC_MULTIPLIER[what_if_traffic]
        * 60.0
    )

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Stress delivery cost",
        f"₹{stress_delivery_cost:,.0f}",
        f"₹{stress_delivery_cost - optimized_delivery_cost:,.0f}",
    )

    q2.metric(
        "Stress total cost",
        f"₹{stress_total:,.0f}",
        f"₹{stress_total - optimized_total_cost:,.0f}",
    )

    q3.metric(
        "Stress unserved orders",
        f"{stress_df.loc[stress_assignment.unserved, 'daily_orders'].sum():,.0f}",
    )

    q4.metric(
        "Stress ETA",
        f"{stress_time:,.0f} min",
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Demand change",
                "Traffic",
                "Delivery cost",
                "Fuel cost",
                "Infrastructure cost",
                "Total cost",
                "Unserved orders",
            ],
            "Scenario": [
                f"{what_if_demand:+.0f}%",
                what_if_traffic,
                f"₹{stress_delivery_cost:,.2f}",
                f"₹{stress_fuel:,.2f}",
                f"₹{stress_infrastructure:,.2f}",
                f"₹{stress_total:,.2f}",
                f"{stress_df.loc[stress_assignment.unserved, 'daily_orders'].sum():,.0f}",
            ],
        }
    )

    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
    )


st.divider()

st.caption(
    "GRIDPOINT v3 • Multiple warehouses • Capacity • Radius • "
    "Vehicles • Fuel • Traffic • Demand scenarios • "
    "Infrastructure trade-off • Resilience"
)
