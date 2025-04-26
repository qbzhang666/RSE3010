import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet
import io
from itertools import combinations

# ---- SMR Calculation Functions ---- #
def calculate_F1(alpha_j, alpha_s):
    A = abs((alpha_j - alpha_s + 180) % 360 - 180)
    F1 = (1 - np.sin(np.radians(A))) ** 2
    return F1

def calculate_F2(beta_j, method):
    if method.lower() == 'toppling':
        return 1.0
    else:
        return np.tan(np.radians(beta_j)) ** 2

def calculate_F3(method, beta_j, beta_s, alpha_j, alpha_s):
    if method.lower() == 'planar':
        C = beta_j - beta_s
        if C > 10:
            return 0
        elif 0 < C <= 10:
            return -6
        elif C == 0:
            return -25
        elif -5 <= C < 0:
            return -50
        else:
            return -60
    elif method.lower() == 'wedge':
        return -50
    elif method.lower() == 'toppling':
        C = beta_j + beta_s
        if C < 110:
            return 0
        elif 110 <= C <= 120:
            return -6
        else:
            return -25
    else:
        return 0

def calculate_F4(excavation_method):
    return {
        'natural': 15,
        'pre-split blasting': 10,
        'smooth blasting': 8,
        'mechanical': 0,
        'poor blasting': -8
    }.get(excavation_method.lower(), 0)

def calculate_SMR(RMRb, alpha_j, beta_j, alpha_s, beta_s, method, excavation):
    F1 = calculate_F1(alpha_j, alpha_s)
    F2 = calculate_F2(beta_j, method)
    F3 = calculate_F3(method, beta_j, beta_s, alpha_j, alpha_s)
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
    else:
        return "Class V", "Very poor - Completely unstable"

# ---- Streamlit App ---- #
st.set_page_config(page_title="Extended SMR Tool", layout="wide")
st.title("⛰️ Extended Slope Mass Rating (SMR) Calculator")

with st.sidebar:
    st.header("Global Parameters")
    RMRb = st.slider("Basic Rock Mass Rating (RMRb)", 0, 100, 60)
    method = st.selectbox("Failure mechanism", ['Planar', 'Toppling', 'Wedge'])
    excavation = st.selectbox("Excavation method", ['Natural', 'Pre-split blasting', 'Smooth blasting', 'Mechanical', 'Poor blasting'])
    n_joints = st.number_input("Number of Joint Sets", 1, 10, 2)
    n_slopes = st.number_input("Number of Slope Faces", 1, 5, 1)

st.subheader("📌 Input Data")
joint_sets = []
for i in range(n_joints):
    with st.expander(f"Joint Set {i+1}"):
        alpha_j = st.number_input(f"αⱼ (Joint dip direction °) [Set {i+1}]", 0, 360, 120, key=f"aj_{i}")
        beta_j = st.number_input(f"βⱼ (Joint dip angle °) [Set {i+1}]", 0, 90, 30, key=f"bj_{i}")
        joint_sets.append((alpha_j, beta_j))

slope_faces = []
for i in range(n_slopes):
    with st.expander(f"Slope Face {i+1}"):
        alpha_s = st.number_input(f"αₛ (Slope dip direction °) [Face {i+1}]", 0, 360, 110, key=f"as_{i}")
        beta_s = st.number_input(f"βₛ (Slope dip angle °) [Face {i+1}]", 0, 90, 60, key=f"bs_{i}")
        slope_faces.append((alpha_s, beta_s))

# ---- Results Calculation ---- #
st.subheader("📊 SMR Results Table")

records = []
joint_colors = ['g', 'r', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'stereonet'})
intersection_records = []

for j_id, (aj, bj) in enumerate(joint_sets):
    color = joint_colors[j_id % len(joint_colors)]
    strike_j = (aj - 90) % 360
    ax.plane(strike_j, bj, color+'-', linewidth=1.5)
    ax.text(aj, bj, f'JS{j_id+1}', color=color, fontsize=8, ha='center', va='center')

for s_id, (as_, bs) in enumerate(slope_faces):
    strike_s = (as_ - 90) % 360
    ax.plane(strike_s, bs, 'b--', linewidth=2)
    ax.text(as_, bs, f'SF{s_id+1}', color='blue', fontsize=8, ha='center', va='center')

# ---- Calculate intersections for all joint pairs ---- #
if len(joint_sets) >= 2:
    for (i, (az1_dd, dip1)), (j, (az2_dd, dip2)) in combinations(enumerate(joint_sets), 2):
        strike1 = (az1_dd - 90) % 360
        strike2 = (az2_dd - 90) % 360
        trend_arr, plunge_arr = mplstereonet.plane_intersection(strike1, dip1, strike2, dip2)
        trend = float(trend_arr)
        plunge = float(plunge_arr)
        if plunge < 0:
            trend = (trend + 180) % 360
            plunge = -plunge
        intersection_records.append({"Joint Pair": f"JS{i+1} & JS{j+1}", "Trend (°)": round(trend,1), "Plunge (°)": round(plunge,1)})

ax.grid(True)
ax.set_azimuth_ticks(np.arange(0, 360, 30))
st.pyplot(fig)

# ---- Display Intersection Table ---- #
if intersection_records:
    st.subheader("🧭 Intersection Orientations")
    df_intersections = pd.DataFrame(intersection_records)
    st.dataframe(df_intersections, use_container_width=True)

# ---- Export Plot Button ---- #
buffer = io.BytesIO()
fig.savefig(buffer, format="png")
buffer.seek(0)
st.download_button("📥 Download Stereonet as PNG", buffer, file_name="stereonet_smr.png")

# ---- Legend ---- #
st.markdown("""
### 📖 SMR Interpretation Classes
| SMR Value | Class    | Description                          |
|-----------|----------|--------------------------------------|
| 81-100    | I        | Very good - Completely stable        |
| 61-80     | II       | Good - Stable                        |
| 41-60     | III      | Fair - Partially stable              |
| 21-40     | IV       | Poor - Unstable                      |
| 0-20      | V        | Very poor - Completely unstable      |
""")
