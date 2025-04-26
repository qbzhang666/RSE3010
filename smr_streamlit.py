import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet
import io

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
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'stereonet'})
intersection_text = ""

for j_id, (aj, bj) in enumerate(joint_sets):
    color = joint_colors[j_id % len(joint_colors)]
    for s_id, (as_, bs) in enumerate(slope_faces):
        smr, f1, f2, f3, f4 = calculate_SMR(RMRb, aj, bj, as_, bs, method, excavation)
        f_product = round(f1 * f2 * f3, 2)
        cls, desc = interpret_SMR(smr)
        records.append({
            "Joint Set": j_id+1,
            "Slope Face": s_id+1,
            "αⱼ": aj, "βⱼ": bj,
            "αₛ": as_, "βₛ": bs,
            "F₁": round(f1, 4), "F₂": round(f2, 4), "F₃": f3, "F₁×F₂×F₃": f_product, "F₄": f4,
            "SMR": round(smr, 2),
            "Class": cls,
            "Description": desc
        })
    strike_j = (aj - 90) % 360
    ax.plane(strike_j, bj, color+'-', linewidth=1.5)
    pole_az = strike_j
    pole_plunge = 90 - bj
    x, y = mplstereonet.stereonet_math.pole(pole_az, pole_plunge)
    ax.text(x, y, f'JS{j_id+1}', fontsize=6, ha='center', va='center', color=color)

for s_id, (as_, bs) in enumerate(slope_faces):
    strike_s = (as_ - 90) % 360
    ax.plane(strike_s, bs, 'b--', linewidth=2)
    pole_az = strike_s
    pole_plunge = 90 - bs
    x, y = mplstereonet.stereonet_math.pole(pole_az, pole_plunge)
    ax.text(x, y, f'SF{s_id+1}', fontsize=6, ha='center', va='center', color='blue')

# ---- Calculate and plot intersection if 2 joints ---- #
if len(joint_sets) == 2:
    az1_dd, dip1 = joint_sets[0]
    az2_dd, dip2 = joint_sets[1]
    strike1 = (az1_dd - 90) % 360
    strike2 = (az2_dd - 90) % 360
    trend_plunge = mplstereonet.plane_intersection(strike1, dip1, strike2, dip2)
    if trend_plunge:
        trend, plunge = trend_plunge
        if plunge < 0:
            trend = (trend + 180) % 360
            plunge = -plunge
        ax.pole(trend, plunge, 'ko', markersize=5)
        intersection_text = f"**Intersection orientation**: Trend = {np.round(trend, 1)}°, Plunge = {np.round(plunge, 1)}°"

ax.grid(True)
ax.set_azimuth_ticks(np.arange(0, 360, 30))
st.pyplot(fig)

if intersection_text:
    st.markdown(f"### 🧭 {intersection_text}")

# ---- SMR Table ---- #
df = pd.DataFrame(records)
st.dataframe(df, use_container_width=True)

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
