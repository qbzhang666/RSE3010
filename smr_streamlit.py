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
# SMR factor calculations
# =========================================================
def calculate_A(mode, alpha_j=None, alpha_s=None, alpha_i=None):
    mode = mode.lower()

    if mode == "planar":
        return angular_difference_deg(alpha_j, alpha_s)
    elif mode == "wedge":
        return angular_difference_deg(alpha_i, alpha_s)
    elif mode == "toppling":
        return angular_difference_deg(alpha_j, alpha_s + 180)
    return None


def calculate_B(mode, beta_j=None, beta_i=None):
    mode = mode.lower()

    if mode == "toppling":
        return None
    elif mode == "planar":
        return abs(beta_j)
    elif mode == "wedge":
        return abs(beta_i)
    return None


def calculate_C(mode, beta_j=None, beta_s=None, beta_i=None):
    mode = mode.lower()

    if mode == "planar":
        return beta_j - beta_s
    elif mode == "wedge":
        return beta_i - beta_s
    elif mode == "toppling":
        return beta_j + beta_s
    return None


def calculate_F1_from_A(A):
    if A is None:
        return None
    if A > 30:
        return 0.15
    return (1 - np.sin(np.radians(A))) ** 2


def calculate_F2_from_B(mode, B):
    mode = mode.lower()

    if mode == "toppling":
        return 1.0

    if B is None:
        return None

    value = abs(B)

    if value < 20:
        return 0.15
    elif 20 <= value <= 30:
        return 0.40
    elif 30 < value <= 35:
        return 0.70
    elif 35 < value <= 45:
        return 0.85
    return 1.0


def calculate_F3_from_C(mode, C):
    mode = mode.lower()

    if C is None:
        return 0

    if mode in ["planar", "wedge"]:
        if C > 10:
            return 0
        elif 0 < C <= 10:
            return -6
        elif np.isclose(C, 0):
            return -25
        elif -5 <= C < 0:
            return -50
        else:
            return -60

    elif mode == "toppling":
        if C < 110:
            return 0
        elif 110 <= C <= 120:
            return -6
        else:
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


def classify_smr(smr):
    if smr > 80:
        return "I", "Very good", "Completely stable", "None"
    elif smr > 60:
        return "II", "Good", "Stable", "Occasional spot bolting"
    elif smr > 40:
        return "III", "Fair", "Partially stable", "Systematic bolting"
    elif smr > 20:
        return "IV", "Poor", "Unstable", "Corrective measures"
    else:
        return "V", "Very poor", "Completely unstable", "Re-excavation"


# =========================================================
# Stereonet helpers
# =========================================================
def compute_plane_intersection(dipdir1, dip1, dipdir2, dip2):
    """
    Returns (trend, plunge) for the intersection of two planes.
    mplstereonet.plane_intersection typically returns (plunge, bearing/trend).
    """
    strike1 = (dipdir1 - 90) % 360
    strike2 = (dipdir2 - 90) % 360

    result = mplstereonet.plane_intersection(strike1, dip1, strike2, dip2)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("Unexpected output from mplstereonet.plane_intersection")

    plunge_raw, trend_raw = result
    plunge = first_scalar(plunge_raw)
    trend = first_scalar(trend_raw) % 360

    if plunge < 0:
        plunge = -plunge
        trend = (trend + 180) % 360

    return trend, plunge


def add_plane_to_stereonet(ax, dipdir, dip, color, linewidth=1.5, linestyle="-"):
    strike = (dipdir - 90) % 360
    ax.plane(strike, dip, color=color, linewidth=linewidth, linestyle=linestyle)


# =========================================================
# Record builders
# =========================================================
def build_planar_record(case_name, alpha_j, beta_j, alpha_s, beta_s, rmrb, excavation, slope_face_id):
    A = calculate_A("planar", alpha_j=alpha_j, alpha_s=alpha_s)
    B = calculate_B("planar", beta_j=beta_j)
    C = calculate_C("planar", beta_j=beta_j, beta_s=beta_s)

    F1 = calculate_F1_from_A(A)
    F2 = calculate_F2_from_B("planar", B)
    F3 = calculate_F3_from_C("planar", C)
    F4 = calculate_F4(excavation)

    product = F1 * F2 * F3
    smr = rmrb + product + F4
    cls, quality, stability, support = classify_smr(smr)

    return {
        "Slope Face": slope_face_id,
        "Case": case_name,
        "Mode": "Planar",
        "A (°)": round(A, 1),
        "B (°)": round(B, 1),
        "C (°)": round(C, 1),
        "F1": round(F1, 2),
        "F2": round(F2, 2),
        "F3": F3,
        "F1×F2×F3": round(product, 1),
        "F4": F4,
        "SMR": round(smr, 1),
        "Class": cls,
        "Quality": quality,
        "Stability": stability,
        "Support": support,
    }


def build_toppling_record(case_name, alpha_j, beta_j, alpha_s, beta_s, rmrb, excavation, slope_face_id):
    A = calculate_A("toppling", alpha_j=alpha_j, alpha_s=alpha_s)
    B = calculate_B("toppling")
    C = calculate_C("toppling", beta_j=beta_j, beta_s=beta_s)

    F1 = calculate_F1_from_A(A)
    F2 = calculate_F2_from_B("toppling", B)
    F3 = calculate_F3_from_C("toppling", C)
    F4 = calculate_F4(excavation)

    product = F1 * F2 * F3
    smr = rmrb + product + F4
    cls, quality, stability, support = classify_smr(smr)

    return {
        "Slope Face": slope_face_id,
        "Case": case_name,
        "Mode": "Toppling",
        "A (°)": round(A, 1),
        "B (°)": "—",
        "C (°)": round(C, 1),
        "F1": round(F1, 2),
        "F2": round(F2, 2),
        "F3": F3,
        "F1×F2×F3": round(product, 1),
        "F4": F4,
        "SMR": round(smr, 1),
        "Class": cls,
        "Quality": quality,
        "Stability": stability,
        "Support": support,
    }


def build_wedge_record(case_name, trend, plunge, alpha_s, beta_s, rmrb, excavation, slope_face_id):
    A = calculate_A("wedge", alpha_i=trend, alpha_s=alpha_s)
    B = calculate_B("wedge", beta_i=plunge)
    C = calculate_C("wedge", beta_i=plunge, beta_s=beta_s)

    F1 = calculate_F1_from_A(A)
    F2 = calculate_F2_from_B("wedge", B)
    F3 = calculate_F3_from_C("wedge", C)
    F4 = calculate_F4(excavation)

    product = F1 * F2 * F3
    smr = rmrb + product + F4
    cls, quality, stability, support = classify_smr(smr)

    return {
        "Slope Face": slope_face_id,
        "Case": case_name,
        "Mode": "Wedge",
        "A (°)": round(A, 1),
        "B (°)": round(B, 1),
        "C (°)": round(C, 1),
        "F1": round(F1, 2),
        "F2": round(F2, 2),
        "F3": F3,
        "F1×F2×F3": round(product, 1),
        "F4": F4,
        "SMR": round(smr, 1),
        "Class": cls,
        "Quality": quality,
        "Stability": stability,
        "Support": support,
    }


# =========================================================
# Streamlit app
# =========================================================
st.set_page_config(page_title="Extended SMR Tool", layout="wide")
st.title("⛰️ Extended Slope Mass Rating (SMR) Calculator")

with st.sidebar:
    st.header("Global Parameters")
    RMRb = st.slider("Basic Rock Mass Rating (RMRb)", 0, 100, 45)
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
    n_joints = st.number_input("Number of Joint Sets", 1, 10, 3)
    n_slopes = st.number_input("Number of Slope Faces", 1, 5, 1)

st.subheader("📌 Input Data")

joint_sets = []
for i in range(n_joints):
    with st.expander(f"Joint Set {i + 1}", expanded=(i == 0)):
        alpha_j = st.number_input(
            f"αⱼ (Joint dip direction °) [Set {i + 1}]",
            min_value=0.0,
            max_value=360.0,
            value=[137.0, 60.0, 322.0][i] if i < 3 else 120.0,
            step=1.0,
            key=f"aj_{i}",
        )
        beta_j = st.number_input(
            f"βⱼ (Joint dip angle °) [Set {i + 1}]",
            min_value=0.0,
            max_value=90.0,
            value=[44.0, 49.0, 85.0][i] if i < 3 else 30.0,
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
            value=138.0 if i == 0 else 110.0,
            step=1.0,
            key=f"as_{i}",
        )
        beta_s = st.number_input(
            f"βₛ (Slope dip angle °) [Face {i + 1}]",
            min_value=0.0,
            max_value=90.0,
            value=65.0 if i == 0 else 60.0,
            step=1.0,
            key=f"bs_{i}",
        )
        slope_faces.append((float(alpha_s), float(beta_s)))

# =========================================================
# Stereonet plot
# =========================================================
st.subheader("🧭 Stereonet")

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "stereonet"})
joint_colors = ["g", "r", "c", "m", "y", "k", "orange", "purple", "brown", "b"]

for j_id, (aj, bj) in enumerate(joint_sets):
    color = joint_colors[j_id % len(joint_colors)]
    add_plane_to_stereonet(ax, aj, bj, color=color, linewidth=1.5)
    ax.text(
        1.08,
        1.00 - j_id * 0.06,
        f"J{j_id + 1} ({bj:.0f}°/{aj:03.0f}°)",
        color=color,
        transform=ax.transAxes,
        fontsize=9,
    )

base_offset = len(joint_sets)
for s_id, (as_, bs) in enumerate(slope_faces):
    add_plane_to_stereonet(ax, as_, bs, color="blue", linewidth=2.5)
    ax.text(
        1.08,
        1.00 - (base_offset + s_id) * 0.06,
        f"S{s_id + 1} ({bs:.0f}°/{as_:03.0f}°)",
        color="blue",
        transform=ax.transAxes,
        fontsize=9,
    )

ax.grid(True)
ax.set_azimuth_ticks(np.arange(0, 360, 30))
st.pyplot(fig)

# =========================================================
# Intersections
# =========================================================
intersection_records = []
intersection_map = {}

if len(joint_sets) >= 2:
    for (i, (az1_dd, dip1)), (j, (az2_dd, dip2)) in combinations(enumerate(joint_sets), 2):
        try:
            trend, plunge = compute_plane_intersection(az1_dd, dip1, az2_dd, dip2)
            label = f"J{i + 1}∩J{j + 1}"
            intersection_records.append(
                {
                    "Case": label,
                    "Trend (°)": round(trend, 1),
                    "Plunge (°)": round(plunge, 1),
                }
            )
            intersection_map[label] = (trend, plunge)
        except Exception as e:
            st.warning(f"Could not compute intersection for J{i + 1} & J{j + 1}: {e}")

# =========================================================
# Full SMR table: planar + wedge + toppling
# =========================================================
st.subheader("📊 Full SMR Results Table")

records = []

for s_id, (as_, bs) in enumerate(slope_faces, start=1):
    # Planar
    for j_id, (aj, bj) in enumerate(joint_sets, start=1):
        records.append(
            build_planar_record(
                case_name=f"J{j_id}",
                alpha_j=aj,
                beta_j=bj,
                alpha_s=as_,
                beta_s=bs,
                rmrb=RMRb,
                excavation=excavation,
                slope_face_id=s_id,
            )
        )

    # Wedge
    for label, (trend, plunge) in intersection_map.items():
        records.append(
            build_wedge_record(
                case_name=label,
                trend=trend,
                plunge=plunge,
                alpha_s=as_,
                beta_s=bs,
                rmrb=RMRb,
                excavation=excavation,
                slope_face_id=s_id,
            )
        )

    # Toppling
    for j_id, (aj, bj) in enumerate(joint_sets, start=1):
        records.append(
            build_toppling_record(
                case_name=f"J{j_id}",
                alpha_j=aj,
                beta_j=bj,
                alpha_s=as_,
                beta_s=bs,
                rmrb=RMRb,
                excavation=excavation,
                slope_face_id=s_id,
            )
        )

df_results = pd.DataFrame(records)

mode_order = {"Planar": 1, "Wedge": 2, "Toppling": 3}
df_results["_mode_order"] = df_results["Mode"].map(mode_order)
df_results["_case_sort"] = df_results["Case"]
df_results = df_results.sort_values(
    by=["Slope Face", "_mode_order", "_case_sort"], ascending=[True, True, True]
).drop(columns=["_mode_order", "_case_sort"])


def highlight_class(row):
    class_color = {
        "I": "background-color: lightgreen",
        "II": "background-color: palegreen",
        "III": "background-color: khaki",
        "IV": "background-color: lightsalmon",
        "V": "background-color: lightcoral",
    }.get(row["Class"], "")
    return [class_color] * len(row)


styled_df = df_results.style.apply(highlight_class, axis=1)
st.dataframe(styled_df, use_container_width=True)

# =========================================================
# Optional filtered view
# =========================================================
st.subheader("🔎 Filtered View")
selected_modes = st.multiselect(
    "Show modes",
    options=["Planar", "Wedge", "Toppling"],
    default=["Planar", "Wedge", "Toppling"],
)

df_filtered = df_results[df_results["Mode"].isin(selected_modes)].copy()
st.dataframe(df_filtered, use_container_width=True)

# =========================================================
# Intersection orientation table
# =========================================================
if intersection_records:
    st.subheader("📐 Intersection Orientations")
    df_intersections = pd.DataFrame(intersection_records).rename(
        columns={
            "Trend (°)": "αᵢ (Trend °)",
            "Plunge (°)": "βᵢ (Plunge °)",
        }
    )
    st.dataframe(df_intersections, use_container_width=True)

# =========================================================
# Downloads
# =========================================================
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
    df_results.to_excel(writer, sheet_name="Full SMR Results", index=False)
    df_filtered.to_excel(writer, sheet_name="Filtered Results", index=False)
    if intersection_records:
        df_intersections.to_excel(writer, sheet_name="Intersections", index=False)

excel_buffer.seek(0)

st.download_button(
    label="📥 Download Results as Excel",
    data=excel_buffer,
    file_name="smr_results_full.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown(
    """
### 📖 SMR Interpretation Classes

| SMR Value | Class | Quality | Stability | Typical Support |
|---|---|---|---|---|
| 81–100 | I | Very good | Completely stable | None |
| 61–80 | II | Good | Stable | Occasional spot bolting |
| 41–60 | III | Fair | Partially stable | Systematic bolting |
| 21–40 | IV | Poor | Unstable | Corrective measures |
| 0–20 | V | Very poor | Completely unstable | Re-excavation |
"""
)
