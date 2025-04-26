import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet as mpl
import io

# ---- CORRECTED SMR CALCULATION FUNCTIONS ---- #
def calculate_F1(alpha_j, alpha_s):
    """Calculate parallelism between joint and slope strike (0°-180°)"""
    angular_diff = abs((alpha_j - alpha_s + 180) % 360 - 180)
    return (1 - np.sin(np.radians(angular_diff))) ** 2

def calculate_F2(beta_j, failure_type):
    """Calculate joint dip factor according to Romana's SMR"""
    failure_type = failure_type.lower()
    if failure_type == 'toppling':
        return 1.0
    
    # Planar/Wedge failure F2 values
    if beta_j > 45:
        return 1.0
    elif 30 < beta_j <= 45:
        return 0.85
    elif 20 < beta_j <= 30:
        return 0.70
    elif 10 < beta_j <= 20:
        return 0.50
    else:
        return 0.30

def calculate_F3(failure_type, beta_j, beta_s):
    """Calculate slope-joint relationship factor"""
    failure_type = failure_type.lower()
    
    if failure_type == 'planar':
        if beta_j > beta_s:
            return -60
        elif beta_j == beta_s:
            return -50
        elif (beta_s - 10) < beta_j < beta_s:
            return -50
        else:
            return 0
            
    elif failure_type == 'toppling':
        total = beta_j + beta_s
        if total > 110:
            return -25
        elif 100 < total <= 110:
            return -6
        else:
            return 0
            
    elif failure_type == 'wedge':
        return -50  # Simplified assumption
        
    return 0

def calculate_F4(excavation_method):
    """Excavation method adjustment factors"""
    return {
        'natural': 0,
        'pre-split blasting': -5,
        'smooth blasting': -8,
        'mechanical': -10,
        'poor blasting': -12
    }.get(excavation_method.lower(), 0)

def calculate_SMR(RMRb, alpha_j, beta_j, alpha_s, beta_s, failure_type, excavation):
    """Calculate final Slope Mass Rating"""
    F1 = calculate_F1(alpha_j, alpha_s)
    F2 = calculate_F2(beta_j, failure_type)
    F3 = calculate_F3(failure_type, beta_j, beta_s)
    F4 = calculate_F4(excavation)
    return RMRb + (F1 * F2 * F3) + F4, F1, F2, F3, F4

def interpret_SMR(SMR):
    """Classification according to Romana"""
    if SMR > 80: return "I", "Very good - Completely stable"
    if SMR > 60: return "II", "Good - Stable"
    if SMR > 40: return "III", "Fair - Partially stable"
    if SMR > 20: return "IV", "Poor - Unstable"
    return "V", "Very poor - Completely unstable"

# ---- STREAMLIT INTERFACE ---- #
st.set_page_config(page_title="SMR Calculator", layout="wide")
st.title("⛰️ Slope Mass Rating Calculator (Romana, 1985)")

# Sidebar Controls
with st.sidebar:
    st.header("Global Parameters")
    RMRb = st.slider("Basic RMR", 0, 100, 60)
    failure_type = st.selectbox("Failure Mechanism", ["Planar", "Toppling", "Wedge"])
    excavation = st.selectbox("Excavation Method", [
        "Natural", "Pre-split Blasting", "Smooth Blasting", 
        "Mechanical", "Poor Blasting"
    ])
    n_joints = st.number_input("Joint Sets", 1, 5, 2)
    n_slopes = st.number_input("Slope Faces", 1, 3, 1)

# Joint/Slope Input Sections
joints = []
for i in range(n_joints):
    with st.expander(f"Joint Set {i+1}"):
        col1, col2 = st.columns(2)
        with col1:
            alpha_j = st.number_input(f"αⱼ (°)", 0, 360, 120, key=f"aj{i}")
        with col2:
            beta_j = st.number_input(f"βⱼ (°)", 0, 90, 45, key=f"bj{i}")
        joints.append((alpha_j, beta_j))

slopes = []
for i in range(n_slopes):
    with st.expander(f"Slope Face {i+1}"):
        col1, col2 = st.columns(2)
        with col1:
            alpha_s = st.number_input(f"αₛ (°)", 0, 360, 90, key=f"as{i}")
        with col2:
            beta_s = st.number_input(f"βₛ (°)", 0, 90, 60, key=f"bs{i}")
        slopes.append((alpha_s, beta_s))

# Results Calculation
records = []
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'stereonet'})

# Plot Joints
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, (aj, bj) in enumerate(joints):
    color = colors[i % len(colors)]
    
    # Plot joint plane
    ax.plane(aj, bj, color=color, linestyle='-', 
            label=f'Joint {i+1} (αⱼ={aj}°, βⱼ={bj}°)')
    
    # Add pole
    pole_azi, pole_ang = mpl.pole(aj, bj)
    ax.pole(pole_azi, pole_ang, marker='o', color=color, markersize=8)

# Plot Slopes
for i, (as_, bs) in enumerate(slopes):
    # Plot slope plane
    ax.plane(as_, bs, color='#2ca02c', linestyle='--', linewidth=2,
            label=f'Slope {i+1} (αₛ={as_}°, βₛ={bs}°)')
    
    # Add text annotation
    mid_azi = (as_ + 180) % 360
    mid_dip = bs * 0.5
    x, y = mpl.stereonet_math.pole(mid_azi, 90 - mid_dip)
    ax.text(x, y, f'S{i+1}', fontsize=10, color='#2ca02c', 
          ha='center', va='center')

# Calculate SMR for all combinations
for j_idx, (aj, bj) in enumerate(joints):
    for s_idx, (as_, bs) in enumerate(slopes):
        smr, F1, F2, F3, F4 = calculate_SMR(RMRb, aj, bj, as_, bs, failure_type, excavation)
        cls, desc = interpret_SMR(smr)
        
        records.append({
            "Joint": j_idx+1,
            "Slope": s_idx+1,
            "αⱼ": aj, "βⱼ": bj,
            "αₛ": as_, "βₛ": bs,
            "F₁": round(F1, 2),
            "F₂": F2,
            "F₃": F3,
            "F₄": F4,
            "SMR": round(smr, 1),
            "Class": cls,
            "Stability": desc
        })

# Plot Formatting
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_azimuth_ticks(range(0, 360, 30))
ax.legend(bbox_to_anchor=(1.3, 1), fontsize=9)
plt.tight_layout()

# Display Results
col1, col2 = st.columns([1, 2])
with col1:
    st.pyplot(fig)
with col2:
    df = pd.DataFrame(records)
    st.dataframe(df.set_index(["Joint", "Slope"]), use_container_width=True)

# Download Button
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150)
st.download_button("📥 Download Stereonet", buf.getvalue(), 
                 file_name="smr_stereonet.png", mime="image/png")

# Classification Legend
st.markdown("""
**SMR Classification**
| Class | SMR Range | Stability Description          |
|-------|-----------|--------------------------------|
| I     | 81-100    | Very good - Completely stable  |
| II    | 61-80     | Good - Stable                  |
| III   | 41-60     | Fair - Partially stable        |
| IV    | 21-40     | Poor - Unstable                |
| V     | 0-20      | Very poor - Completely unstable|
""")
