import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from itertools import combinations

# ============================================================
# GRIDPOINT — Demand-Aware Warehouse Optimization
# ============================================================

st.set_page_config(
    page_title="GridPoint | Logistics Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

EARTH_RADIUS_KM = 6371.0

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.metric-card {
    padding: 0.8rem 1rem;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.25);
    background: rgba(128,128,128,.06);
}
.small-note {font-size: 0.82rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Geography / mathematics
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def distance_matrix(df, centers):
    return np.column_stack([
        haversine_km(
            df["latitude"].to_numpy(float),
            df["longitude"].to_numpy(float),
            lat, lon
        )
        for lat, lon in centers
    ])

def weighted_centroid(df):
    w = df["daily_orders"].to_numpy(float)
    return (
        float(np.average(df["latitude"], weights=w)),
        float(np.average(df["longitude"], weights=w))
    )

def normalize_coordinates(df):
    x = df[["latitude", "longitude"]].to_numpy(float).copy()
    mean_lat = np.radians(df["latitude"].mean())
    x[:, 1] *= np.cos(mean_lat)
    return x, mean_lat

# -----------------------------
# Demand-aware candidate centers
# -----------------------------
def demand_weighted_centers(df, k):
    """Weighted K-Means gives demand-aware candidate warehouse locations."""
    X, mean_lat = normalize_coordinates(df)
    weights = df["daily_orders"].to_numpy(float)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=25,
        max_iter=500
    )
    model.fit(X, sample_weight=weights)

    centers = []
    for lat, lon_scaled in model.cluster_centers_:
        centers.append(
            (float(lat), float(lon_scaled / np.cos(mean_lat)))
        )
    return centers

# -----------------------------
# Constrained assignment
# -----------------------------
def constrained_assignment(
    df, centers, capacities=None, max_radius=None, failed=None
):
    n = len(df)
    k = len(centers)
    dmat = distance_matrix(df, centers)

    assignment = np.full(n, -1, dtype=int)
    remaining = np.array(
        capacities if capacities is not None else [np.inf] * k,
        dtype=float
    )

    active = np.ones(k, dtype=bool)
    if failed:
        for f in failed:
            if 0 <= f < k:
                active[f] = False

    # High-demand neighborhoods first.
    order = np.argsort(-df["daily_orders"].to_numpy(float))
    unassigned = []

    for i in order:
        candidates = np.argsort(dmat[i])
        chosen = None

        for j in candidates:
            if not active[j]:
                continue
            if max_radius is not None and dmat[i, j] > max_radius:
                continue
            if remaining[j] < df.iloc[i]["daily_orders"]:
                continue
            chosen = int(j)
            break

        if chosen is None:
            unassigned.append(int(i))
        else:
            assignment[i] = chosen
            remaining[chosen] -= df.iloc[i]["daily_orders"]

    return assignment, dmat, remaining, unassigned

def weighted_distance_cost(df, assignment, dmat, rate_per_order_km):
    valid = assignment >= 0
    if not np.any(valid):
        return 0.0

    idx = np.arange(len(df))[valid]
    orders = df["daily_orders"].to_numpy(float)[valid]
    distances = dmat[idx, assignment[valid]]
    return float(np.sum(orders * distances * rate_per_order_km))

def optimize_network(
    df, k, rate_per_order_km, capacities=None, max_radius=None, failed=None
):
    centers = demand_weighted_centers(df, k)
    assignment, dmat, remaining, unassigned = constrained_assignment(
        df, centers, capacities, max_radius, failed
    )

    delivery_cost = weighted_distance_cost(
        df, assignment, dmat, rate_per_order_km
    )

    assigned_orders = []
    for j in range(k):
        assigned_orders.append(
            float(df.loc[assignment == j, "daily_orders"].sum())
        )

    return {
        "centers": centers,
        "assignment": assignment,
        "distance_matrix": dmat,
        "remaining_capacity": remaining,
        "assigned_orders": assigned_orders,
        "delivery_cost": delivery_cost,
        "unassigned": unassigned,
        "feasible": len(unassigned) == 0,
    }

def baseline_benchmark(df, rate_per_order_km):
    """Benchmark: one demand-weighted central depot.
    This is a benchmark, not a claim about an actual company's existing network."""
    center = weighted_centroid(df)
    distances = haversine_km(
        df["latitude"].to_numpy(float),
        df["longitude"].to_numpy(float),
        center[0],
        center[1]
    )
    cost = float(
        np.sum(
            df["daily_orders"].to_numpy(float)
            * distances
            * rate_per_order_km
        )
    )
    return cost, center

def system_cost(delivery_cost, warehouse_count, infrastructure_per_warehouse):
    return float(delivery_cost + warehouse_count * infrastructure_per_warehouse)

# -----------------------------
# Resilience / scenario analysis
# -----------------------------
def failure_analysis(df, result, rate, capacities, radius):
    k = len(result["centers"])
    rows = []

    for failed in range(k):
        scenario = optimize_network(
            df,
            k,
            rate,
            capacities=capacities,
            max_radius=radius,
            failed=[failed]
        )

        normal_cost = result["delivery_cost"]
        extra = scenario["delivery_cost"] - normal_cost
        unassigned_orders = float(
            df.iloc[scenario["unassigned"]]["daily_orders"].sum()
        ) if scenario["unassigned"] else 0.0

        rows.append({
            "Failed warehouse": f"W{failed+1}",
            "Scenario cost (₹)": scenario["delivery_cost"],
            "Extra cost (₹)": extra,
            "Unserved orders": unassigned_orders,
            "Feasible": "Yes" if scenario["feasible"] else "No"
        })

    return pd.DataFrame(rows)

# -----------------------------
# Visualization
# -----------------------------
def network_map(df, result, title="Optimized warehouse network"):
    fig = go.Figure()

    # Neighborhoods
    fig.add_trace(go.Scattergeo(
        lat=df["latitude"],
        lon=df["longitude"],
        mode="markers",
        marker=dict(size=9, symbol="circle"),
        text=[
            f"<b>{n}</b><br>Daily orders: {o:,.0f}"
            for n, o in zip(df["neighborhood"], df["daily_orders"])
        ],
        hoverinfo="text",
        name="Neighborhoods"
    ))

    # Assignment lines
    for i, w in enumerate(result["assignment"]):
        if w >= 0:
            c = result["centers"][w]
            fig.add_trace(go.Scattergeo(
                lat=[df.iloc[i]["latitude"], c[0]],
                lon=[df.iloc[i]["longitude"], c[1]],
                mode="lines",
                line=dict(width=1),
                hoverinfo="skip",
                showlegend=False
            ))

    # Warehouses
    centers = result["centers"]
    fig.add_trace(go.Scattergeo(
        lat=[c[0] for c in centers],
        lon=[c[1] for c in centers],
        mode="markers+text",
        marker=dict(size=18, symbol="star"),
        text=[f"W{i+1}" for i in range(len(centers))],
        textposition="top center",
        hovertext=[
            f"<b>Warehouse {i+1}</b><br>"
            f"Assigned orders: {result['assigned_orders'][i]:,.0f}"
            for i in range(len(centers))
        ],
        hoverinfo="text",
        name="Warehouses"
    ))

    fig.update_geos(
        projection_type="mercator",
        center=dict(
            lat=float(df["latitude"].mean()),
            lon=float(df["longitude"].mean())
        ),
        lataxis_range=[
            float(df["latitude"].min()) - 0.08,
            float(df["latitude"].max()) + 0.08
        ],
        lonaxis_range=[
            float(df["longitude"].min()) - 0.08,
            float(df["longitude"].max()) + 0.08
        ],
        showcountries=True,
        showland=True
    )

    fig.update_layout(
        title=title,
        height=600,
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(orientation="h")
    )
    return fig

# -----------------------------
# Load data
# -----------------------------
st.title("🚚 GRIDPOINT")
st.caption("Demand-Aware Warehouse Optimization & Network Resilience")

with st.sidebar:
    st.header("⚙️ Network Configuration")

    uploaded = st.file_uploader(
        "Upload neighborhood CSV",
        type=["csv"],
        help="Columns required: neighborhood, latitude, longitude, daily_orders"
    )

    if uploaded is None:
        df = pd.read_csv("data/neighborhoods.csv")
        st.info("Using the included sample Bengaluru dataset.")
    else:
        df = pd.read_csv(uploaded)

    required = {"neighborhood", "latitude", "longitude", "daily_orders"}
    missing = required - set(df.columns)
    if missing:
        st.error("Missing columns: " + ", ".join(sorted(missing)))
        st.stop()

    for col in ["latitude", "longitude", "daily_orders"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "daily_orders"])

    if len(df) < 2:
        st.error("Please provide at least 2 neighborhoods.")
        st.stop()

    k = st.slider(
        "Number of warehouses",
        1, min(5, len(df)), min(2, len(df))
    )

    st.subheader("💰 Cost Model")
    rate = st.number_input(
        "Delivery rate (₹ / order / km)",
        min_value=0.1,
        value=2.50,
        step=0.10
    )

    infra = st.number_input(
        "Infrastructure cost (₹ / warehouse / day)",
        min_value=0.0,
        value=15000.0,
        step=1000.0
    )

    st.subheader("🛡️ Constraints")
    use_capacity = st.checkbox("Warehouse capacity", True)
    if use_capacity:
        default_capacity = max(
            1000,
            int(np.ceil(df["daily_orders"].sum() / k * 1.15 / 100) * 100)
        )
        cap_value = st.number_input(
            "Capacity per warehouse (orders/day)",
            min_value=1,
            value=default_capacity,
            step=100
        )
        capacities = [float(cap_value)] * k
    else:
        capacities = None

    use_radius = st.checkbox("Maximum service radius", False)
    if use_radius:
        radius = st.slider("Maximum radius (km)", 1, 100, 20)
    else:
        radius = None

    st.subheader("🔮 What-If Lab")
    demand_change = st.slider("Demand change (%)", -50, 100, 0)

    failed_choice = st.selectbox(
        "Simulate warehouse failure",
        ["None"] + [f"Warehouse {i+1}" for i in range(k)]
    )

    st.divider()
    run = st.button("🚀 Run GridPoint", use_container_width=True)

# Demand scenario
scenario_df = df.copy()
scenario_df["daily_orders"] = np.maximum(
    0,
    np.round(
        scenario_df["daily_orders"] * (1 + demand_change / 100)
    )
)

failed = []
if failed_choice != "None":
    failed = [int(failed_choice.split()[-1]) - 1]

# Run / cache result
settings_key = (
    len(df), k, float(rate), float(infra),
    tuple(capacities) if capacities else None,
    radius, demand_change, tuple(failed)
)

if "last_key" not in st.session_state or st.session_state.last_key != settings_key or run:
    result = optimize_network(
        df, k, rate, capacities, radius
    )
    scenario_result = optimize_network(
        scenario_df, k, rate, capacities, radius, failed
    )
    st.session_state.result = result
    st.session_state.scenario_result = scenario_result
    st.session_state.last_key = settings_key
else:
    result = st.session_state.result
    scenario_result = st.session_state.scenario_result

# Baseline benchmark
baseline_delivery, baseline_center = baseline_benchmark(df, rate)

optimized_delivery = result["delivery_cost"]
baseline_total = system_cost(baseline_delivery, 1, infra)
optimized_total = system_cost(optimized_delivery, k, infra)

savings = baseline_total - optimized_total
savings_pct = savings / baseline_total * 100 if baseline_total else 0

# -----------------------------
# KPI row
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Neighborhoods", f"{len(df):,}")
c2.metric("Daily orders", f"{df['daily_orders'].sum():,.0f}")
c3.metric("Optimized system cost", f"₹{optimized_total:,.0f}")
c4.metric(
    "vs central-depot benchmark",
    f"{savings_pct:.1f}%",
    delta=f"₹{savings:,.0f}"
)

if result["unassigned"]:
    st.error(
        f"⚠️ {len(result['unassigned'])} neighborhood(s) are unassigned "
        "under the current capacity/radius constraints. Increase capacity, "
        "radius, or warehouse count."
    )

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "🗺️ Network Map",
    "📊 Optimization",
    "🛡️ Resilience",
    "🔮 What-If Lab",
    "💡 AI Analyst",
])

# 1. Map
with tabs[0]:
    st.subheader("Optimized Warehouse Network")
    st.plotly_chart(
        network_map(df, result),
        use_container_width=True
    )

    display = df[["neighborhood", "daily_orders"]].copy()
    display["warehouse"] = [
        f"W{x+1}" if x >= 0 else "Unassigned"
        for x in result["assignment"]
    ]

    # Distance and delivery cost per neighborhood
    distances = []
    costs = []
    for i, w in enumerate(result["assignment"]):
        if w >= 0:
            d = result["distance_matrix"][i, w]
            distances.append(d)
            costs.append(d * df.iloc[i]["daily_orders"] * rate)
        else:
            distances.append(np.nan)
            costs.append(np.nan)

    display["distance_km"] = np.round(distances, 2)
    display["delivery_cost_₹"] = np.round(costs, 2)

    st.dataframe(display, use_container_width=True, hide_index=True)

# 2. Optimization
with tabs[1]:
    st.subheader("📊 Before vs Optimized Network")

    a, b, c = st.columns(3)
    a.metric("Benchmark total cost", f"₹{baseline_total:,.0f}")
    b.metric("Optimized total cost", f"₹{optimized_total:,.0f}")
    c.metric("Net difference", f"₹{savings:,.0f}")

    chart = pd.DataFrame({
        "Configuration": ["Central-depot benchmark", "GridPoint"],
        "Total system cost": [baseline_total, optimized_total]
    })
    st.bar_chart(chart.set_index("Configuration"))

    st.markdown("### Cost breakdown")
    breakdown = pd.DataFrame({
        "Component": ["Delivery", "Infrastructure"],
        "Benchmark": [
            baseline_delivery,
            infra
        ],
        "GridPoint": [
            optimized_delivery,
            k * infra
        ]
    })
    st.dataframe(
        breakdown.style.format({
            "Benchmark": "₹{:,.0f}",
            "GridPoint": "₹{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Warehouse utilization")
    utilization_rows = []
    for j in range(k):
        assigned = result["assigned_orders"][j]
        cap = capacities[j] if capacities else np.inf
        utilization = assigned / cap * 100 if np.isfinite(cap) and cap else 0
        utilization_rows.append({
            "Warehouse": f"W{j+1}",
            "Assigned orders": assigned,
            "Capacity": cap if np.isfinite(cap) else "Unlimited",
            "Utilization %": utilization
        })

    util_df = pd.DataFrame(utilization_rows)
    st.dataframe(
        util_df.style.format({
            "Assigned orders": "{:,.0f}",
            "Utilization %": "{:.1f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The benchmark is a demand-weighted central-depot benchmark. "
        "It is used to quantify improvement when an existing warehouse "
        "network is not supplied in the input data."
    )

# 3. Resilience
with tabs[2]:
    st.subheader("🛡️ Network Resilience Lab")
    st.write(
        "What happens if one warehouse becomes unavailable? "
        "GridPoint reassigns demand and measures the resulting impact."
    )

    failure_df = failure_analysis(
        df, result, rate, capacities, radius
    )

    if not failure_df.empty:
        st.dataframe(
            failure_df.style.format({
                "Scenario cost (₹)": "₹{:,.0f}",
                "Extra cost (₹)": "₹{:,.0f}",
                "Unserved orders": "{:,.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        max_extra = max(0.0, failure_df["Extra cost (₹)"].max())
        max_unserved = failure_df["Unserved orders"].max()

        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Worst extra cost",
            f"₹{max_extra:,.0f}"
        )
        r2.metric(
            "Worst unserved orders",
            f"{max_unserved:,.0f}"
        )
        r3.metric(
            "Failure scenarios tested",
            f"{len(failure_df)}"
        )

        if max_unserved > 0:
            st.warning(
                "The current network has at least one single-warehouse "
                "failure scenario that leaves some demand unserved."
            )
        else:
            st.success(
                "Every single-warehouse failure scenario remained serviceable "
                "under the selected constraints."
            )

# 4. What-if
with tabs[3]:
    st.subheader("🔮 What-If Lab")
    st.write(
        "Stress-test the network without changing the original optimization."
    )

    q1, q2, q3 = st.columns(3)
    q1.metric("Demand shock", f"{demand_change:+d}%")
    q2.metric("Scenario delivery cost", f"₹{scenario_result['delivery_cost']:,.0f}")
    q3.metric(
        "Scenario unserved orders",
        f"{scenario_df.iloc[scenario_result['unassigned']]['daily_orders'].sum():,.0f}"
        if scenario_result["unassigned"] else "0"
    )

    if failed:
        st.warning(f"🚨 {failed_choice} is offline in this scenario.")

    if scenario_result["unassigned"]:
        st.error(
            "The network cannot serve all demand under this scenario. "
            "Try increasing capacity, increasing radius, or adding a warehouse."
        )
    else:
        st.success("All neighborhoods remain serviceable in this scenario.")

    scenario_compare = pd.DataFrame({
        "Metric": [
            "Daily orders",
            "Delivery cost",
            "Unserved orders"
        ],
        "Normal": [
            df["daily_orders"].sum(),
            result["delivery_cost"],
            df.iloc[result["unassigned"]]["daily_orders"].sum()
            if result["unassigned"] else 0
        ],
        "Scenario": [
            scenario_df["daily_orders"].sum(),
            scenario_result["delivery_cost"],
            scenario_df.iloc[scenario_result["unassigned"]]["daily_orders"].sum()
            if scenario_result["unassigned"] else 0
        ]
    })
    st.dataframe(
        scenario_compare.style.format({
            "Normal": "{:,.0f}",
            "Scenario": "{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

# 5. AI Analyst
with tabs[4]:
    st.subheader("💡 GridPoint Decision Analyst")
    st.info(
        "This starter uses deterministic insights from the optimization results. "
        "During the hackathon, you can connect an allowed public AI API to "
        "generate natural-language explanations from these computed facts."
    )

    max_util = 0
    if capacities:
        max_util = max(
            (result["assigned_orders"][j] / capacities[j] * 100)
            for j in range(k)
        )

    insights = []

    if savings > 0:
        insights.append(
            f"GridPoint reduces the benchmark system cost by "
            f"₹{savings:,.0f} ({savings_pct:.1f}%)."
        )
    else:
        insights.append(
            "The current warehouse count does not beat the central-depot "
            "benchmark after infrastructure cost; compare another warehouse count."
        )

    if capacities and max_util >= 85:
        insights.append(
            f"At least one warehouse is operating above 85% utilization "
            f"({max_util:.1f}%), leaving limited capacity headroom."
        )
    elif capacities:
        insights.append(
            f"The highest warehouse utilization is {max_util:.1f}%."
        )

    if result["unassigned"]:
        insights.append(
            f"{len(result['unassigned'])} neighborhoods are unassigned "
            "under the selected constraints."
        )
    else:
        insights.append("All neighborhoods are currently serviceable.")

    if not failure_df.empty:
        worst = failure_df.loc[failure_df["Extra cost (₹)"].idxmax()]
        insights.append(
            f"The most expensive single-warehouse failure scenario is "
            f"{worst['Failed warehouse']}, adding approximately "
            f"₹{worst['Extra cost (₹)']:,.0f}."
        )

    st.markdown("### Current network insights")
    for item in insights:
        st.write("• " + item)

    st.markdown("### Model inputs used")
    st.code(
        f"""Warehouses: {k}
Delivery rate: ₹{rate:.2f} / order / km
Infrastructure: ₹{infra:,.0f} / warehouse / day
Capacity: {"Enabled" if capacities else "Disabled"}
Max radius: {f"{radius} km" if radius else "Disabled"}
Demand shock: {demand_change:+d}%"""
    )

st.divider()
st.caption(
    "GridPoint is a hackathon prototype. The monetary cost model is an "
    "explicit user-configured assumption, not a market-price claim."
)
