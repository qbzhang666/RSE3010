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

def calculate_F2(beta_j, method, beta_i=None):
    if method.lower() == 'toppling':
        return 1.0
    elif method.lower() == 'planar':
        value = abs(beta_j)
    elif method.lower() == 'wedge' and beta_i is not None:
        value = abs(beta_i)
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
    else:
        return 1.0

def calculate_F3(method, beta_j, beta_s, alpha_j, alpha_s, beta_i=None):
    if method.lower() == 'planar':
        C = beta_j - beta_s
    elif method.lower() == 'wedge' and beta_i is not None:
        C = beta_i - beta_s
    elif method.lower() == 'toppling':
        C = beta_j + beta_s
    else:
        return 0

    if method.lower() in ['planar', 'wedge']:
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
    elif method.lower() == 'toppling':
        if C < 110:
            return 0
        elif 110 <= C <= 120:
            return -6
        else:
            return -25

def calculate_F4(excavation_method):
    return {
        'natural': 15,
        'pre-split blasting': 10,
        'smooth blasting': 8,
        'mechanical/blasting': 0,
        'poor blasting': -8
    }.get(excavation_method.lower(), 0)

def calculate_SMR(RMRb, alpha_j, beta_j, alpha_s, beta_s, method, excavation, alpha_i=None):
    F1 = calculate_F1(alpha_j, alpha_s, method, alpha_i)
    F2 = calculate_F2(beta_j, method, beta_i=alpha_i)
    F3 = calculate_F3(method, beta_j, beta_s, alpha_j, alpha_s, beta_i=alpha_i)
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
    excavation = st.selectbox("Excavation method", ['Natural', 'Pre-split blasting', 'Smooth blasting', 'Mechanical/Blasting', 'Poor blasting'])
    n_joints = st.number_input("Number of Joint Sets", 1, 10, 2)
    n_slopes = st.number_input("Number of Slope Faces", 1, 5, 1)

st.subheader("📌 Input Data")
# Joint and Slope input
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

st.subheader("📊 SMR Results Table")
records = []
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'stereonet'})
intersection_records = []
legend_labels = []

joint_colors = ['g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'b']

for j_id, (aj, bj) in enumerate(joint_sets):
    color = joint_colors[j_id % len(joint_colors)]
    strike_j = (aj - 90) % 360
    ax.plane(strike_j, bj, color=color, linestyle='-', linewidth=1.5)
    legend_labels.append((f"Joint Set {j_id+1} ({bj:.0f}°/{aj:03.0f}°)", color))

for s_id, (as_, bs) in enumerate(slope_faces):
    strike_s = (as_ - 90) % 360
    ax.plane(strike_s, bs, color='blue', linewidth=3)
    legend_labels.append((f"Slope Face {s_id+1} ({bs:.0f}°/{as_:03.0f}°)", 'blue'))

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
for idx, (label, color) in enumerate(legend_labels):
    ax.text(1.1, 1.0-idx*0.07, label, color=color, transform=ax.transAxes, fontsize=9)

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
                "αⱼ / Plunge (°)": aj,
                "βⱼ / Trend (°)": bj,
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
                "αᵢ (plunge °)": plunge,
                "βᵢ (trend °)": trend,
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

def highlight_class(row):
    color = ''
    if row['Class'] == 'Class I':
        color = 'background-color: lightgreen'
    elif row['Class'] == 'Class II':
        color = 'background-color: palegreen'
    elif row['Class'] == 'Class III':
        color = 'background-color: khaki'
    elif row['Class'] == 'Class IV':
        color = 'background-color: lightsalmon'
    elif row['Class'] == 'Class V':
        color = 'background-color: lightcoral'
    return ['' for _ in row.index[:-2]] + [color, '']

styled_df = df_results.style.apply(highlight_class, axis=1)
styled_df = styled_df.format(precision=2)

st.dataframe(styled_df, use_container_width=True)

if intersection_records:
    st.subheader("🧭 Intersection Orientations")
    df_intersections = pd.DataFrame(intersection_records)
    
    # Rename columns for Wedge failure
    if method.lower() == 'wedge':
        df_intersections = df_intersections.rename(columns={
            "Trend (°)": "βᵢ (Trend °)",
            "Plunge (°)": "αᵢ (Plunge °)"
        })
    
    st.dataframe(df_intersections, use_container_width=True)

buffer = io.BytesIO()
fig.savefig(buffer, format="png")
buffer.seek(0)
st.download_button("📥 Download Stereonet as PNG", buffer, file_name="stereonet_smr.png")

# --- Export to Excel --- #

excel_buffer = io.BytesIO()

with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
    df_results.to_excel(writer, sheet_name='SMR Calculations', index=False)
    if intersection_records:
        df_intersections.to_excel(writer, sheet_name='Intersection Orientations', index=False)

excel_buffer.seek(0)

st.download_button(
    label="📥 Download Results as Excel",
    data=excel_buffer,
    file_name="smr_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

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
