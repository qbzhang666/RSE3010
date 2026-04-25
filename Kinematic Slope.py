import math
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Geometry and stereonet helper functions
# ============================================================

def acute_angle(angle_deg):
    """
    Return the acute orientation difference in degrees, between 0 and 180.
    """
    angle_deg = abs(angle_deg) % 360
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    return angle_deg


def circular_diff(a_deg, b_deg):
    """
    Difference between two azimuths, returned as 0–180 degrees.
    """
    return acute_angle(a_deg - b_deg)


def plane_normal(dip_dir_deg, dip_deg):
    """
    Calculate the normal vector of a plane from dip direction and dip.

    Coordinate system:
    x = East
    y = North
    z = Up

    Input format:
    dip / dip direction, e.g. 40 / 160.
    """

    alpha = math.radians(dip_dir_deg)
    beta = math.radians(dip_deg)

    # Unit vector in the down-dip direction
    dip_vector = np.array([
        math.sin(alpha) * math.cos(beta),
        math.cos(alpha) * math.cos(beta),
        -math.sin(beta)
    ])

    # Strike direction = dip direction - 90 degrees
    strike_az = math.radians(dip_dir_deg - 90)

    strike_vector = np.array([
        math.sin(strike_az),
        math.cos(strike_az),
        0.0
    ])

    normal = np.cross(strike_vector, dip_vector)
    return normal / np.linalg.norm(normal)


def trend_plunge(vector):
    """
    Convert a 3D vector into trend and plunge.

    The returned line is always plotted in the lower hemisphere.
    """

    vector = np.asarray(vector, dtype=float)
    vector = vector / np.linalg.norm(vector)

    # Lower hemisphere convention
    if vector[2] > 0:
        vector = -vector

    horizontal_length = math.hypot(vector[0], vector[1])

    trend = (math.degrees(math.atan2(vector[0], vector[1])) + 360) % 360
    plunge = math.degrees(math.atan2(-vector[2], horizontal_length))

    return trend, plunge


def intersection_line(dd1, dip1, dd2, dip2):
    """
    Calculate the intersection line of two planes.

    Output:
    trend, plunge
    """

    n1 = plane_normal(dd1, dip1)
    n2 = plane_normal(dd2, dip2)

    line = np.cross(n1, n2)

    return trend_plunge(line)


def equal_angle_project(trend_deg, plunge_deg):
    """
    Wulff net projection:
    equal-angle lower-hemisphere stereographic projection.

    This is NOT equal-area.

    Formula:
    r = tan((90 - plunge) / 2)
    """

    trend_rad = math.radians(trend_deg)

    r = math.tan(math.radians(90 - plunge_deg) / 2)

    x = r * math.sin(trend_rad)
    y = r * math.cos(trend_rad)

    return x, y


def great_circle_points(dip_dir_deg, dip_deg, npts=721):
    """
    Generate equal-angle projected points for the great circle of a plane.
    """

    normal = plane_normal(dip_dir_deg, dip_deg)

    # Create two orthogonal unit vectors lying in the plane
    reference = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(reference, normal)) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])

    u = np.cross(normal, reference)
    u = u / np.linalg.norm(u)

    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    xs = []
    ys = []

    for t in np.linspace(0, 2 * np.pi, npts):
        line_vector = math.cos(t) * u + math.sin(t) * v

        # Lower hemisphere only
        if line_vector[2] <= 1e-9:
            trend, plunge = trend_plunge(line_vector)
            x, y = equal_angle_project(trend, plunge)
            xs.append(x)
            ys.append(y)

    return np.array(xs), np.array(ys)


def draw_wulff_net_grid(ax, interval=10):
    """
    Draw a teaching-style Wulff net background.

    This is an equal-angle stereographic net.
    The background is drawn using projected great-circle families,
    so it does not look like a simple polar-coordinate plot.
    """

    theta = np.linspace(0, 2 * np.pi, 720)

    # Primitive circle
    ax.plot(
        np.sin(theta),
        np.cos(theta),
        color="black",
        linewidth=1.4
    )

    # Great-circle families
    # These give the Wulff-net visual structure
    for dip in range(interval, 90, interval):
        for dip_dir in [0, 90, 180, 270]:
            x, y = great_circle_points(dip_dir, dip)
            ax.plot(
                x,
                y,
                color="0.88",
                linewidth=0.55,
                zorder=0
            )

    # Constant plunge reference circles
    # These are also projected with equal-angle geometry
    for plunge in range(interval, 90, interval):
        r = math.tan(math.radians(90 - plunge) / 2)
        ax.plot(
            r * np.sin(theta),
            r * np.cos(theta),
            color="0.92",
            linewidth=0.45,
            zorder=0
        )

    # Cardinal directions
    for label, azimuth in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
        ax.text(
            1.11 * np.sin(np.radians(azimuth)),
            1.11 * np.cos(np.radians(azimuth)),
            label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold"
        )


def default_joint_dataframe():
    """
    Default Assignment 3 teaching dataset.
    Format: dip / dip direction.
    """

    return pd.DataFrame({
        "Joint": ["J1", "J2", "J3"],
        "Dip β (deg)": [40.0, 55.0, 80.0],
        "Dip direction α (deg)": [160.0, 110.0, 340.0],
    })


# ============================================================
# Streamlit app
# ============================================================

st.set_page_config(
    page_title="Week 11 Kinematic Analysis with Wulff Net",
    layout="wide"
)

st.title("RSE3010 Week 11 — Kinematic Analysis with Wulff Net")
st.caption(
    "Standalone teaching app: equal-angle stereonet for kinematic analysis — What can fail?"
)

# ============================================================
# Sidebar inputs
# ============================================================

with st.sidebar:
    st.header("Slope and friction inputs")

    slope_dip = st.number_input(
        "Slope angle ψf / βs (deg)",
        min_value=0.0,
        max_value=90.0,
        value=60.0,
        step=1.0
    )

    slope_dd = st.number_input(
        "Slope dip direction αs (deg)",
        min_value=0.0,
        max_value=360.0,
        value=170.0,
        step=1.0
    )

    dry_phi = st.number_input(
        "Dry friction angle φdry (deg)",
        min_value=0.0,
        max_value=90.0,
        value=45.0,
        step=1.0
    )

    wet_phi = st.number_input(
        "Wet friction angle φwet (deg)",
        min_value=0.0,
        max_value=90.0,
        value=28.0,
        step=1.0
    )

    tolerance = st.number_input(
        "Directional tolerance (deg)",
        min_value=0.0,
        max_value=90.0,
        value=20.0,
        step=1.0
    )

    grid_interval = st.selectbox(
        "Wulff net grid interval",
        [5, 10, 15, 20],
        index=1
    )


# ============================================================
# Joint input table
# ============================================================

st.subheader("Joint-set input")

st.write(
    "Use **dip / dip direction** format. "
    "In this app, α = dip direction or trend, and β = dip or plunge."
)

jdf = st.data_editor(
    default_joint_dataframe(),
    num_rows="dynamic",
    use_container_width=True
)

jdf = jdf.dropna()
jdf["Dip β (deg)"] = pd.to_numeric(jdf["Dip β (deg)"], errors="coerce")
jdf["Dip direction α (deg)"] = pd.to_numeric(
    jdf["Dip direction α (deg)"],
    errors="coerce"
)
jdf = jdf.dropna()


# ============================================================
# Kinematic condition functions
# ============================================================

def planar_possible(alpha, beta, phi):
    """
    Planar sliding condition.

    Teaching form:
    - joint direction approximately parallel to slope direction
    - joint dip is lower than slope angle, so it daylights
    - joint dip is greater than friction angle
    """

    direction_ok = circular_diff(alpha, slope_dd) <= tolerance
    daylight_ok = beta < slope_dip
    friction_ok = beta > phi

    return direction_ok and daylight_ok and friction_ok


def wedge_possible(trend, plunge, phi):
    """
    Wedge sliding condition.

    Teaching form:
    - intersection trend approximately exits through slope face
    - plunge is lower than slope angle, so it daylights
    - plunge is greater than friction angle
    """

    direction_ok = circular_diff(trend, slope_dd) <= tolerance
    daylight_ok = plunge < slope_dip
    friction_ok = plunge > phi

    return direction_ok and daylight_ok and friction_ok


def toppling_possible(alpha, beta, phi):
    """
    Simplified toppling condition for teaching.

    Joint should dip steeply in the opposite direction to the slope face.
    """

    opposite_direction_ok = circular_diff(alpha - slope_dd - 180, 0) <= tolerance

    # Simplified Goodman-Bray style condition
    steep_enough = beta >= (90 - slope_dip + phi)

    return opposite_direction_ok and steep_enough


# ============================================================
# Perform kinematic checks
# ============================================================

results = []

# Planar and toppling for each joint
for _, row in jdf.iterrows():

    name = row["Joint"]
    beta = float(row["Dip β (deg)"])
    alpha = float(row["Dip direction α (deg)"])

    results.append([
        "Planar",
        name,
        beta,
        alpha,
        "Dry",
        planar_possible(alpha, beta, dry_phi)
    ])

    results.append([
        "Planar",
        name,
        beta,
        alpha,
        "Wet",
        planar_possible(alpha, beta, wet_phi)
    ])

    results.append([
        "Toppling",
        name,
        beta,
        alpha,
        "Dry",
        toppling_possible(alpha, beta, dry_phi)
    ])

    results.append([
        "Toppling",
        name,
        beta,
        alpha,
        "Wet",
        toppling_possible(alpha, beta, wet_phi)
    ])

# Wedge intersections for joint pairs
for (_, r1), (_, r2) in combinations(jdf.iterrows(), 2):

    trend, plunge = intersection_line(
        float(r1["Dip direction α (deg)"]),
        float(r1["Dip β (deg)"]),
        float(r2["Dip direction α (deg)"]),
        float(r2["Dip β (deg)"])
    )

    pair_label = f"{r1['Joint']}–{r2['Joint']}"

    results.append([
        "Wedge",
        pair_label,
        plunge,
        trend,
        "Dry",
        wedge_possible(trend, plunge, dry_phi)
    ])

    results.append([
        "Wedge",
        pair_label,
        plunge,
        trend,
        "Wet",
        wedge_possible(trend, plunge, wet_phi)
    ])

res_df = pd.DataFrame(
    results,
    columns=[
        "Mode",
        "Joint / pair",
        "Dip or plunge β (deg)",
        "Direction / trend α (deg)",
        "Condition",
        "Kinematically possible"
    ]
)


# ============================================================
# Layout
# ============================================================

col1, col2 = st.columns([1.15, 1])


# ============================================================
# Plot Wulff net
# ============================================================

with col1:

    st.subheader("Model stereonet — Wulff net (equal-angle)")

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw proper Wulff-style grid
    draw_wulff_net_grid(ax, interval=grid_interval)

    theta = np.linspace(0, 2 * np.pi, 720)

    # Slope and joint great circles
    items = [
        ("Slope", slope_dd, slope_dip)
    ]

    for _, row in jdf.iterrows():
        items.append((
            row["Joint"],
            float(row["Dip direction α (deg)"]),
            float(row["Dip β (deg)"])
        ))

    for label, dip
