import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="GridPoint",
    page_icon="🚚",
    layout="wide"
)

# -----------------------------
# Core calculations
# -----------------------------
EARTH_RADIUS_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

def weighted_centroid(df):
    w = df["daily_orders"].to_numpy(dtype=float)
    lat = np.average(df["latitude"], weights=w)
    lon = np.average(df["longitude"], weights=w)
    return float(lat), float(lon)

def distance_matrix(df, centers):
    return np.column_stack([
        haversine_km(
            df["latitude"].to_numpy(),
            df["longitude"].to_numpy(),
            lat, lon
        )
        for lat, lon in centers
    ])

def assign_nearest(df, centers, capacities=None, max_radius=None, failed=None):
    """Greedy assignment with capacity/radius constraints.
    Returns assignment array, feasibility flag, and messages."""
    n = len(df)
    k = len(centers)
    dmat = distance_matrix(df, centers)
    assignment = np.full(n, -1, dtype=int)
    remaining = np.array(capacities if capacities is not None else [np.inf] * k, dtype=float)
    active = np.ones(k, dtype=bool)
    if failed:
        for f in failed:
            if 0 <= f < k:
                active[f] = False

    # High-demand neighborhoods first makes the greedy capacity assignment
    # less likely to strand large demand at the end.
    order = np.argsort(-df["daily_orders"].to_numpy())

    messages = []
    feasible = True

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
            feasible = False
            continue

        assignment[i] = chosen
        remaining[chosen] -= df.iloc[i]["daily_orders"]

    if not feasible:
        messages.append(
            "Some neighborhoods could not be assigned under the current "
            "capacity/radius/failure constraints."
        )

    return assignment, dmat, remaining, feasible, messages

def total_delivery_cost(df, assignment, dmat):
    valid = assignment >= 0
    if not np.any(valid):
        return 0.0
    orders = df["daily_orders"].to_numpy(dtype=float)
    return float(np.sum(orders[valid] * dmat[np.arange(len(df))[valid], assignment[valid]]))

def optimize_network(df, k, capacities=None, max_radius=None, failed=None):
    """Weighted K-Means provides demand-aware candidate warehouse locations.
    Then a constrained assignment step enforces capacity/radius/failure rules."""
    if k > len(df):
        raise ValueError("Number of warehouses cannot exceed number of neighborhoods.")

    X = df[["latitude", "longitude"]].to_numpy()
    weights = df["daily_orders"].to_numpy(dtype=float)

    # Scale longitude by cos(mean latitude) so K-Means is less distorted
    # by latitude/longitude geometry.
    mean_lat = np.radians(df["latitude"].mean())
    X_scaled = X.copy()
    X_scaled[:, 1] *= np.cos(mean_lat)

    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    model.fit(X_scaled, sample_weight=weights)

    centers_scaled = model.cluster_centers_
    centers = []
    for lat, lon_scaled in centers_scaled:
        centers.append((float(lat), float(lon_scaled / np.cos(mean_lat))))

    assignment, dmat, remaining, feasible, messages = assign_nearest(
        df, centers, capacities, max_radius, failed
    )
    cost = total_delivery_cost(df, assignment, dmat)

    return {
        "centers": centers,
        "assignment": assignment,
        "distance_matrix": dmat,
        "remaining_capacity": remaining,
        "feasible": feasible,
        "messages": messages,
        "cost": cost
    }

def baseline_cost(df):
    center = weighted_centroid(df)
    d = haversine_km(
        df["latitude"].to_numpy(),
        df["longitude"].to_numpy(),
        center[0], center[1]
    )
    orders = df["daily_orders"].to_numpy(dtype=float)
    return float(np.sum(orders * d)), center

def make_map(df, result):
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lat=df["latitude"],
        lon=df["longitude"],
        mode="markers",
        marker=dict(size=10, symbol="circle"),
        text=[
            f"{n}<br>Orders: {o}"
            for n, o in zip(df["neighborhood"], df["daily_orders"])
        ],
        hoverinfo="text",
        name="Neighborhoods"
    ))

    assignment = result["assignment"]
    centers = result["centers"]

    # Draw neighborhood -> warehouse links
    for i, w in enumerate(assignment):
        if w >= 0:
            fig.add_trace(go.Scattergeo(
                lat=[df.iloc[i]["latitude"], centers[w][0]],
                lon=[df.iloc[i]["longitude"], centers[w][1]],
                mode="lines",
                line=dict(width=1),
                hoverinfo="skip",
                showlegend=False
            ))

    if centers:
        fig.add_trace(go.Scattergeo(
            lat=[c[0] for c in centers],
            lon=[c[1] for c in centers],
            mode="markers+text",
            marker=dict(size=18, symbol="star"),
            text=[f"W{i+1}" for i in range(len(centers))],
            textposition="top center",
            hovertext=[
                f"Warehouse {i+1}<br>"
                f"Remaining capacity: "
                f"{result['remaining_capacity'][i]:,.0f}"
                for i in range(len(centers))
            ],
            hoverinfo="text",
            name="Warehouses"
        ))

    fig.update_geos(
        projection_type="mercator",
        center=dict(lat=float(df["latitude"].mean()), lon=float(df["longitude"].mean())),
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
        height=620,
        margin=dict(l=0, r=0, t=30, b=0),
        title="GridPoint Network"
    )
    return fig

# -----------------------------
# UI
# -----------------------------
st.title("🚚 GRIDPOINT")
st.caption("Demand-Aware Warehouse Optimization & Network Resilience")

with st.sidebar:
    st.header("⚙️ Network Controls")

    uploaded = st.file_uploader(
        "Upload neighborhood CSV",
        type=["csv"],
        help="Required columns: neighborhood, latitude, longitude, daily_orders"
    )

    if uploaded is None:
        df = pd.read_csv("data/neighborhoods.csv")
        st.info("Using the included sample Bengaluru dataset.")
    else:
        df = pd.read_csv(uploaded)

    required = {"neighborhood", "latitude", "longitude", "daily_orders"}
    missing = required - set(df.columns)

    if missing:
        st.error(f"Missing columns: {', '.join(sorted(missing))}")
        st.stop()

    df = df.copy()
    df["daily_orders"] = pd.to_numeric(df["daily_orders"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "daily_orders"])

    k = st.slider(
        "Number of warehouses",
        min_value=1,
        max_value=min(5, len(df)),
        value=min(2, len(df))
    )

    use_capacity = st.checkbox("Enable warehouse capacity", value=True)
    capacity = None
    if use_capacity:
        capacity_value = st.number_input(
            "Capacity per warehouse (orders/day)",
            min_value=1,
            value=max(5000, int(df["daily_orders"].sum() / max(k, 1) * 1.15)),
            step=100
        )
        capacity = [float(capacity_value)] * k

    use_radius = st.checkbox("Enable maximum service radius", value=False)
    radius = None
    if use_radius:
        radius = st.slider("Maximum delivery radius (km)", 1, 100, 20)

    st.divider()
    st.markdown("### 🔮 What-If Lab")
    demand_change = st.slider("Demand change (%)", -50, 100, 0)
    failed_warehouse = st.selectbox(
        "Simulate warehouse failure",
        ["None"] + [f"Warehouse {i+1}" for i in range(k)]
    )

    run = st.button("🚀 Run Optimization", use_container_width=True)

if "result" not in st.session_state or run:
    try:
        result = optimize_network(
            df,
            k,
            capacities=capacity,
            max_radius=radius
        )
        st.session_state.result = result
    except Exception as e:
        st.error(f"Optimization error: {e}")
        st.stop()
else:
    result = st.session_state.result

# What-if demand scenario
scenario_df = df.copy()
scenario_df["daily_orders"] = np.maximum(
    0, np.round(scenario_df["daily_orders"] * (1 + demand_change / 100))
)

scenario_failed = []
if failed_warehouse != "None":
    scenario_failed = [int(failed_warehouse.split()[-1]) - 1]

scenario_result = optimize_network(
    scenario_df,
    k,
    capacities=capacity,
    max_radius=radius,
    failed=scenario_failed
)

base_cost, base_center = baseline_cost(df)
optimized_cost = result["cost"]
scenario_cost = scenario_result["cost"]

valid_base = optimized_cost > 0
savings_pct = ((base_cost - optimized_cost) / base_cost * 100) if base_cost else 0

# -----------------------------
# Metrics
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Neighborhoods", len(df))
c2.metric("Daily Orders", f"{df['daily_orders'].sum():,.0f}")
c3.metric("Optimized Cost", f"{optimized_cost:,.0f}")
c4.metric("Baseline Savings", f"{savings_pct:.1f}%")

if not result["feasible"]:
    st.warning(
        "The current constraints make some neighborhoods infeasible. "
        "Increase capacity, radius, or warehouse count."
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Network Map", "📊 Optimization", "🔮 What-If Lab", "🤖 AI Analyst"]
)

with tab1:
    st.subheader("Optimized Warehouse Network")
    st.plotly_chart(make_map(df, result), use_container_width=True)

    assignment_display = df[["neighborhood", "daily_orders"]].copy()
    assignment_display["warehouse"] = [
        f"W{x+1}" if x >= 0 else "Unassigned"
        for x in result["assignment"]
    ]
    st.dataframe(assignment_display, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Before vs After")
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline weighted cost", f"{base_cost:,.0f}")
    m2.metric("Optimized weighted cost", f"{optimized_cost:,.0f}")
    m3.metric("Savings", f"{base_cost - optimized_cost:,.0f}")

    chart_df = pd.DataFrame({
        "Configuration": ["Baseline", "Optimized"],
        "Weighted Delivery Cost": [base_cost, optimized_cost]
    })
    st.bar_chart(
        chart_df.set_index("Configuration"),
        y="Weighted Delivery Cost"
    )

    st.markdown("### Warehouse utilization")
    orders = df["daily_orders"].to_numpy()
    util = []
    for j in range(k):
        assigned_orders = orders[result["assignment"] == j].sum()
        cap = capacity[j] if capacity else np.inf
        util.append({
            "Warehouse": f"W{j+1}",
            "Assigned orders": assigned_orders,
            "Capacity": cap if np.isfinite(cap) else "Unlimited",
            "Utilization %": (assigned_orders / cap * 100) if np.isfinite(cap) and cap else 0
        })
    st.dataframe(pd.DataFrame(util), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("🔮 What-If Lab")
    st.write(
        "Stress-test the optimized network by changing demand or taking a warehouse offline."
    )

    s1, s2, s3 = st.columns(3)
    s1.metric("Demand change", f"{demand_change:+d}%")
    s2.metric("Scenario cost", f"{scenario_cost:,.0f}")
    delta = scenario_cost - optimized_cost
    s3.metric("Cost impact", f"{delta:+,.0f}")

    if scenario_failed:
        st.warning(
            f"Warehouse {scenario_failed[0] + 1} is simulated as unavailable."
        )

    if not scenario_result["feasible"]:
        st.error(
            "⚠️ Network stress test found neighborhoods that cannot be served "
            "under the current constraints."
        )
    else:
        st.success("All neighborhoods remain serviceable in this scenario.")

    scenario_table = scenario_df[["neighborhood", "daily_orders"]].copy()
    scenario_table["scenario_warehouse"] = [
        f"W{x+1}" if x >= 0 else "Unassigned"
        for x in scenario_result["assignment"]
    ]
    st.dataframe(scenario_table, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("🤖 AI Logistics Analyst")
    st.info(
        "This tab is intentionally API-free in the starter version. "
        "It summarizes the optimization using deterministic rules. "
        "During the hackathon, you can connect an allowed public AI API "
        "to turn these computed facts into a natural-language analysis."
    )

    assigned = df["daily_orders"].to_numpy()
    utilization_lines = []
    for j in range(k):
        assigned_orders = assigned[result["assignment"] == j].sum()
        if capacity:
            utilization_lines.append(
                f"W{j+1}: {assigned_orders:,.0f}/{capacity[j]:,.0f} orders "
                f"({assigned_orders/capacity[j]*100:.1f}%)"
            )
        else:
            utilization_lines.append(
                f"W{j+1}: {assigned_orders:,.0f} assigned orders"
            )

    st.markdown("### GridPoint Network Insight")
    st.write(
        f"The optimized configuration uses **{k} warehouse(s)** and produces "
        f"a weighted delivery cost of **{optimized_cost:,.0f}**, compared with "
        f"a baseline of **{base_cost:,.0f}**."
    )

    if savings_pct >= 0:
        st.success(f"The optimization reduces the baseline cost by {savings_pct:.1f}%.")
    else:
        st.warning("The current optimized configuration is more expensive than the baseline.")

    st.markdown("**Warehouse utilization:**")
    for line in utilization_lines:
        st.write("- " + line)

    if not scenario_result["feasible"]:
        st.error(
            "Resilience warning: the selected what-if scenario creates an "
            "infeasible network under the current constraints."
        )
    else:
        st.success(
            "Resilience check: the selected what-if scenario remains feasible."
        )

st.divider()
st.caption(
    "GridPoint is a hackathon prototype. Cost values are relative weighted "
    "delivery-distance units unless a monetary delivery-rate model is supplied."
)
