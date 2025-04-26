import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet
import io

# ---- SMR Calculation Functions ---- #
def calculate_F1(alpha_j, alpha_s):
    A = abs(alpha_j - alpha_s)
    if A <= 5:
        return 1.0
    elif A <= 15:
        return 0.85
    elif A <= 30:
        return 0.70
    elif A <= 60:
        return 0.40
    else:
        return 0.15

def calculate_F2(beta_j):
    if beta_j >= 45:
        return 1.0
    elif beta_j >= 35:
        return 0.85
    elif beta_j >= 25:
        return 0.70
    elif beta_j >= 15:
        return 0.50
    elif beta_j >= 10:
        return 0.20
    else:
        return 0.15

def calculate_F3(method):
    return {'planar': -60, 'toppling': -25, 'wedge': -50}.get(method.lower(), 0)

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
    F2 = calculate_F2(beta_j)
    F3 = calculate_F3(method)
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

for j_id, (aj, bj) in enumerate(joint_sets):
    color = joint_colors[j_id % len(joint_colors)]
    for s_id, (as_, bs) in enumerate(slope_faces):
        smr, f1, f2, f3, f4 = calculate_SMR(RMRb, aj, bj, as_, bs, method, excavation)
        cls, desc = interpret_SMR(smr)
        records.append({
            "Joint Set": j_id+1,
            "Slope Face": s_id+1,
            "αⱼ": aj, "βⱼ": bj,
            "αₛ": as_, "βₛ": bs,
            "SMR": round(smr, 2),
            "Class": cls,
            "Description": desc
        })
    ax.plane(aj, bj, color+'-', linewidth=1.5, label=f'Joint Set {j_id+1}')
    ax.pole(aj, bj, color+'o', markersize=5)

for s_id, (as_, bs) in enumerate(slope_faces):
    ax.plane(as_, bs, 'b--', linewidth=1.5, label=f'Slope Face {s_id+1}')

ax.grid(True)
ax.legend(fontsize='small', loc='upper right', bbox_to_anchor=(1.3, 1))
st.pyplot(fig)

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
