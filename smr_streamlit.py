import io
from itertools import combinations

import matplotlib.pyplot as plt
import mplstereonet
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# Utility helpers
# =========================================================
def angular_difference_deg(a1: float, a2: float) -> float:
    """Smallest absolute angular difference in degrees."""
    return abs((a1 - a2 + 180) % 360 - 180)


def first_scalar(value, default=np.nan) -> float:
    """Convert scalar/array-like output to a single float safely."""
    try:
        arr = np.asarray(value, dtype=float).flatten()
        if arr.size == 0:
            return float(default)
        return float(arr[0])
    except Exception:
        return float(default)


# =========================================================
# SMR calculation functions
# =========================================================
def calculate_F1(alpha_j, alpha_s, method, alpha_i=None):
    method = method.lower()

    if method == "planar":
        A = angular_difference_deg(alpha_j, alpha_s)
    elif method == "toppling":
        A = angular_difference_deg(alpha_j, alpha_s + 180)
    elif method == "wedge" and alpha_i is not None:
        A = angular_difference_deg(alpha_i, alpha_s)
    else:
        return None

    if A > 30:
        return 0.15
    return (1 - np.sin(np.radians(A))) ** 2


def calculate_F2(beta_j, method, beta_i=None):
    method = method.lower()

    if method == "toppling":
        return 1.0
    elif method == "planar":
        value = abs(beta_j)
    elif method == "wedge" and beta_i is not None:
        value = abs(beta_i)  # wedge uses plunge
    else:
        value = abs(beta_j)

    if value < 20:
        return 0.15
    elif 20 <= value <= 30:
        return 0.40
    elif 30 < value <= 35:
        return 0.70
    elif 35 < value <= 45:
        return 0.85
    return 1.0


def calculate_F3(method, beta_j, beta_s, alpha_j, alpha_s, beta_i=None):
    method = method.lower()

    if method == "planar":
        C = beta_j - beta_s
    elif method == "wedge" and beta_i is not None:
        C = beta_i - beta_s  # wedge uses plunge
    elif method == "toppling":
        C = beta_j + beta_s
    else:
        return 0

    if method in ["planar", "wedge"]:
        if C > 10:
            return 0
        elif 0 < C <= 10:
            return -6
        elif np.isclose(C, 0):
            return -25
        elif -5 <= C < 0:
            return -50
        return -60

    if method == "toppling":
        if C < 110:
            return 0
        elif 110 <= C <= 120:
            return -6
        return -25

    return 0


def calculate_F4(excavation_method):
    return {
        "natural": 15,
        "pre-split blasting": 10,
        "smooth blasting": 8,
        "mechanical/blasting": 0,
        "poor blasting": -8,
    }.get(excavation_method.lower(), 0)


def calculate_SMR(
    RMRb,
    alpha_j,
    beta_j,
    alpha_s,
    beta_s,
    method,
    excavation,
    alpha_i=None,
    beta_i=None,
):
    F1 = calculate_F1(alpha_j, alpha_s, method, alpha_i=alpha_i)
    F2 = calculate_F2(beta_j, method, beta_i=beta_i)
    F3 = calculate_F3(method, beta_j, beta_s, alpha_j, alpha_s, beta_i=beta_i)
    F4 = calculate_F4(excavation)
    SMR = RMRb + (F1 * F2 * F3) + F4
    return SMR, F1, F2, F3, F4


def interpret_SMR(SMR):
    if SMR > 80:
        return "Class I", "Very good - Completely stable"
    elif SMR > 60:
        return "Class II", "Good - Stable"
    elif SMR > 40:
        return "Class III", "Fair - Partially stable"
    elif SMR > 20:
        return "Class IV", "Poor - Unstable"
    return "Class V", "Very poor - Completely unstable"


# =========================================================
# Stereonet / intersection helpers
# =========================================================
def compute_plane_intersection(dipdir1, dip1, dipdir2, dip2):
    """
    Returns (trend, plunge) for the intersection of two planes.
    mplstereonet.plane_intersection typically returns (plunge, bearing/trend),
    but we still normalise robustly.
    """
    strike1 = (dipdir1 - 90) % 360
    strike2 = (dipdir2 - 90) % 360

    result = mplstereonet.plane_intersection(strike1, dip1, strike2, dip2)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("Unexpected output from mplstereonet.plane_intersection")

    plunge_raw, trend_raw = result
    plunge = first_scalar(plunge_raw)
    trend = first_scalar(trend_raw) % 360

    # normalise to positive plunge
    if plunge < 0:
        plunge = -plunge
        trend = (trend + 180) % 360

    return trend, plunge


def add_plane_to_stereonet(ax, dipdir, dip, color, linewidth=1.5, linestyle="-"):
    strike = (dipdir - 90) % 360
    ax.plane(strike, dip, color=color, linewidth=linewidth, linestyle=linestyle)


# =========================================================
# Streamlit app
# =========================================================
st.set_page_config(page_title="Extended SMR Tool", layout="wide")
st.title("⛰️ Extended Slope Mass Rating (SMR) Calculator")

with st.sidebar:
    st.header("Global Parameters")
    RMRb = st.slider("Basic Rock Mass Rating (RMRb)", 0, 100, 60)
    method = st.selectbox("Failure mechanism", ["Planar", "Toppling", "Wedge"])
    excavation = st.selectbox(
        "Excavation method",
        [
            "Natural",
            "Pre-split blasting",
            "Smooth blasting",
            "Mechanical/Blasting",
            "Poor blasting",
        ],
    )
    n_joints = st.number_input("Number of Joint Sets", 1, 10, 2)
    n_slopes = st.number_input("Number of Slope Faces", 1, 5, 1)

st.subheader("📌 Input Data")

joint_sets = []
for i in range(n_joints):
    with st.expander(f"Joint Set {i + 1}", expanded=(i == 0)):
        alpha_j = st.number_input(
            f"αⱼ (Joint dip direction °) [Set {i + 1}]",
            min_value=0.0,
            max_value=360.0,
            value=120.0,
            step=1.0,
            key=f"aj_{i}",
        )
        beta_j = st.number_input(
            f"βⱼ (Joint dip angle °) [Set {i + 1}]",
            min_value=0.0,
            max_value=90.0,
            value=30.0,
            step=1.0,
            key=f"bj_{i}",
        )
        joint_sets.append((float(alpha_j), float(beta_j)))

slope_faces = []
for i in range(n_slopes):
    with st.expander(f"Slope Face {i + 1}", expanded=(i == 0)):
        alpha_s = st.number_input(
            f"αₛ (Slope dip direction °) [Face {i + 1}]",
            min_value=0.0,
            max_value=360.0,
            value=110.0,
            step=1.0,
            key=f"as_{i}",
        )
        beta_s = st.number_input(
            f"βₛ (Slope dip angle °) [Face {i + 1}]",
            min_value=0.0,
            max_value=90.0,
            value=60.0,
            step=1.0,
            key=f"bs_{i}",
        )
        slope_faces.append((float(alpha_s), float(beta_s)))

st.subheader("📊 SMR Results Table")

records = []
intersection_records = []

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "stereonet"})
joint_colors = ["g", "r", "c", "m", "y", "k", "orange", "purple", "brown", "b"]

# Plot joint sets
for j_id, (aj, bj) in enumerate(joint_sets):
    color = joint_colors[j_id % len(joint_colors)]
    add_plane_to_stereonet(ax, aj, bj, color=color, linewidth=1.5)
    ax.text(
        1.08,
        1.00 - j_id * 0.06,
        f"Joint Set {j_id + 1} ({bj:.0f}°/{aj:03.0f}°)",
        color=color,
        transform=ax.transAxes,
        fontsize=9,
    )

# Plot slopes
base_offset = len(joint_sets)
for s_id, (as_, bs) in enumerate(slope_faces):
    add_plane_to_stereonet(ax, as_, bs, color="blue", linewidth=2.5)
    ax.text(
        1.08,
        1.00 - (base_offset + s_id) * 0.06,
        f"Slope Face {s_id + 1} ({bs:.0f}°/{as_:03.0f}°)",
        color="blue",
        transform=ax.transAxes,
        fontsize=9,
    )

# Compute intersections robustly
if len(joint_sets) >= 2:
    for (i, (az1_dd, dip1)), (j, (az2_dd, dip2)) in combinations(enumerate(joint_sets), 2):
        try:
            trend, plunge = compute_plane_intersection(az1_dd, dip1, az2_dd, dip2)
            intersection_records.append(
                {
                    "Joint Pair": f"JS{i + 1} & JS{j + 1}",
                    "Trend (°)": round(trend, 1),
                    "Plunge (°)": round(plunge, 1),
                }
            )
        except Exception as e:
            st.warning(f"Could not compute intersection for JS{i + 1} & JS{j + 1}: {e}")

ax.grid(True)
ax.set_azimuth_ticks(np.arange(0, 360, 30))
st.pyplot(fig)

if method.lower() == "wedge" and len(joint_sets) < 2:
    st.warning("⚠️ At least 2 joint sets are needed for wedge failure analysis.")

# ---------------------------------------------------------
# SMR calculations
# ---------------------------------------------------------
if method.lower() in ["planar", "toppling"]:
    for j_id, (aj, bj) in enumerate(joint_sets):
        for s_id, (as_, bs) in enumerate(slope_faces):
            smr, f1, f2, f3, f4 = calculate_SMR(
                RMRb=RMRb,
                alpha_j=aj,
                beta_j=bj,
                alpha_s=as_,
                beta_s=bs,
                method=method,
                excavation=excavation,
            )
            cls, desc = interpret_SMR(smr)

            records.append(
                {
                    "Feature": f"Joint Set {j_id + 1}",
                    "Slope Face": s_id + 1,
                    "αⱼ (Dip direction °)": aj,
                    "βⱼ (Dip angle °)": bj,
                    "αₛ (Slope dip dir °)": as_,
                    "βₛ (Slope dip angle °)": bs,
                    "Failure Mode": method,
                    "F₁": round(f1, 4),
                    "F₂": round(f2, 4),
                    "F₃": f3,
                    "F₁×F₂×F₃": round(f1 * f2 * f3, 2),
                    "F₄": f4,
                    "SMR": round(smr, 2),
                    "Class": cls,
                    "Description": desc,
                }
            )

elif method.lower() == "wedge" and intersection_records:
    for intersection in intersection_records:
        trend = float(intersection["Trend (°)"])    # alpha_i
        plunge = float(intersection["Plunge (°)"])  # beta_i
        pair_label = intersection["Joint Pair"]

        for s_id, (as_, bs) in enumerate(slope_faces):
            smr, f1, f2, f3, f4 = calculate_SMR(
                RMRb=RMRb,
                alpha_j=0.0,
                beta_j=0.0,
                alpha_s=as_,
                beta_s=bs,
                method=method,
                excavation=excavation,
                alpha_i=trend,
                beta_i=plunge,
            )
            cls, desc = interpret_SMR(smr)

            records.append(
                {
                    "Feature": f"Intersection {pair_label}",
                    "Slope Face": s_id + 1,
                    "αᵢ (Trend °)": trend,
                    "βᵢ (Plunge °)": plunge,
                    "αₛ (Slope dip dir °)": as_,
                    "βₛ (Slope dip angle °)": bs,
                    "Failure Mode": "Wedge",
                    "F₁": round(f1, 4),
                    "F₂": round(f2, 4),
                    "F₃": f3,
                    "F₁×F₂×F₃": round(f1 * f2 * f3, 2),
                    "F₄": f4,
                    "SMR": round(smr, 2),
                    "Class": cls,
                    "Description": desc,
                }
            )

# ---------------------------------------------------------
# Results table
# ---------------------------------------------------------
st.subheader("📄 SMR Calculations")
df_results = pd.DataFrame(records)


def highlight_class(row):
    color = ""
    if row["Class"] == "Class I":
        color = "background-color: lightgreen"
    elif row["Class"] == "Class II":
        color = "background-color: palegreen"
    elif row["Class"] == "Class III":
        color = "background-color: khaki"
    elif row["Class"] == "Class IV":
        color = "background-color: lightsalmon"
    elif row["Class"] == "Class V":
        color = "background-color: lightcoral"

    return [color] * len(row)


if df_results.empty:
    st.info("No results to display yet.")
else:
    styled_df = df_results.style.apply(highlight_class, axis=1).format(precision=2)
    st.dataframe(styled_df, use_container_width=True)

# ---------------------------------------------------------
# Intersection table
# ---------------------------------------------------------
if intersection_records:
    st.subheader("🧭 Intersection Orientations")
    df_intersections = pd.DataFrame(intersection_records)

    if method.lower() == "wedge":
        df_intersections = df_intersections.rename(
            columns={
                "Trend (°)": "αᵢ (Trend °)",
                "Plunge (°)": "βᵢ (Plunge °)",
            }
        )

    st.dataframe(df_intersections, use_container_width=True)

# ---------------------------------------------------------
# Downloads
# ---------------------------------------------------------
png_buffer = io.BytesIO()
fig.savefig(png_buffer, format="png", bbox_inches="tight", dpi=300)
png_buffer.seek(0)

st.download_button(
    "📥 Download Stereonet as PNG",
    png_buffer,
    file_name="stereonet_smr.png",
    mime="image/png",
)

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    df_results.to_excel(writer, sheet_name="SMR Calculations", index=False)
    if intersection_records:
        export_intersections = pd.DataFrame(intersection_records)
        export_intersections.to_excel(
            writer, sheet_name="Intersection Orientations", index=False
        )

excel_buffer.seek(0)

st.download_button(
    label="📥 Download Results as Excel",
    data=excel_buffer,
    file_name="smr_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown(
    """
### 📖 SMR Interpretation Classes

| SMR Value | Class | Description |
|---|---|---|
| 81–100 | I | Very good - Completely stable |
| 61–80 | II | Good - Stable |
| 41–60 | III | Fair - Partially stable |
| 21–40 | IV | Poor - Unstable |
| 0–20 | V | Very poor - Completely unstable |
"""
)
