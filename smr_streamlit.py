import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet
import io
from itertools import combinations

# ---- SMR Calculation Functions ---- #
def calculate_F1(alpha_j, alpha_s, method, alpha_i=None):
    if method.lower() == 'planar':
        A = abs((alpha_j - alpha_s + 180) % 360 - 180)
    elif method.lower() == 'toppling':
        A = abs((alpha_j - alpha_s - 180 + 180) % 360 - 180)
    elif method.lower() == 'wedge' and alpha_i is not None:
        A = abs((alpha_i - alpha_s + 180) % 360 - 180)
    else:
        return None

    if A > 30:
        return 0.15
    else:
        return (1 - np.sin(np.radians(A))) ** 2

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

def calculate_SMR(RMRb, alpha_j, beta_j, alpha_s, beta_s, method, excavation, alpha_i=None):
    F1 = calculate_F1(alpha_j, alpha_s, method, alpha_i)
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

# Sidebar for input
with st.sidebar:
    st.header("Global Parameters")
    RMRb = st.slider("Basic Rock Mass Rating (RMRb)", 0, 100, 60)
    method = st.selectbox("Failure mechanism", ['Planar', 'Toppling', 'Wedge'])
    excavation = st.selectbox("Excavation method", ['Natural', 'Pre-split blasting', 'Smooth blasting', 'Mechanical', 'Poor blasting'])
    n_joints = st.number_input("Number of Joint Sets", 1, 10, 2)
    n_slopes = st.number_input("Number of Slope Faces", 1, 5, 1)

st.subheader("📌 Input Data")
# Joint and Slope input
# (input section remains unchanged)

st.subheader("📊 SMR Results Table")
records = []
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'stereonet'})
intersection_records = []
legend_labels = []

# (stereonet plotting remains unchanged)

st.pyplot(fig)

if method.lower() == 'wedge' and len(joint_sets) < 2:
    st.warning("⚠️ At least 2 joint sets are needed for wedge failure analysis.")

# Corrected SMR Calculation Logic
if method.lower() in ['planar', 'toppling']:
    for j_id, (aj, bj) in enumerate(joint_sets):
        for s_id, (as_, bs) in enumerate(slope_faces):
            smr, f1, f2, f3, f4 = calculate_SMR(RMRb, aj, bj, as_, bs, method, excavation)
            cls, desc = interpret_SMR(smr)
            records.append({
                "Feature": f"Joint Set {j_id+1}",
                "Slope Face": s_id+1,
                "αⱼ / Trend (°)": aj,
                "βⱼ / Plunge (°)": bj,
                "αₛ (Slope dip dir °)": as_,
                "βₛ (Slope dip angle °)": bs,
                "Failure Mode": method,
                "F₁": round(f1, 4),
                "F₂": round(f2, 4),
                "F₃": f3,
                "F₁×F₂×F₃": round(f1*f2*f3, 2),
                "F₄": f4,
                "SMR": round(smr, 2),
                "Class": cls,
                "Description": desc
            })

if method.lower() == 'wedge' and intersection_records:
    for intersection in intersection_records:
        trend = intersection["Trend (°)"]
        plunge = intersection["Plunge (°)"]
        pair_label = intersection["Joint Pair"]
        for s_id, (as_, bs) in enumerate(slope_faces):
            smr, f1, f2, f3, f4 = calculate_SMR(RMRb, 0, plunge, as_, bs, method, excavation, alpha_i=trend)
            cls, desc = interpret_SMR(smr)
            records.append({
                "Feature": f"Intersection {pair_label}",
                "Slope Face": s_id+1,
                "αⱼ / Trend (°)": trend,
                "βⱼ / Plunge (°)": plunge,
                "αₛ (Slope dip dir °)": as_,
                "βₛ (Slope dip angle °)": bs,
                "Failure Mode": "Wedge",
                "F₁": round(f1, 4),
                "F₂": round(f2, 4),
                "F₃": f3,
                "F₁×F₂×F₃": round(f1*f2*f3, 2),
                "F₄": f4,
                "SMR": round(smr, 2),
                "Class": cls,
                "Description": desc
            })

st.subheader("📄 SMR Calculations")
df_results = pd.DataFrame(records)

# Highlighting function remains unchanged
styled_df = df_results.style.apply(highlight_class, axis=1)

st.dataframe(styled_df, use_container_width=True, height=700)

if intersection_records:
    st.subheader("🧭 Intersection Orientations")
    df_intersections = pd.DataFrame(intersection_records)
    st.dataframe(df_intersections, use_container_width=True)

buffer = io.BytesIO()
fig.savefig(buffer, format="png")
buffer.seek(0)
st.download_button("📥 Download Stereonet as PNG", buffer, file_name="stereonet_smr.png")

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
