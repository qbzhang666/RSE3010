import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet

# ---- SMR Calculation Functions ---- #

def calculate_F1(alpha_j, alpha_slope):
    A = abs(alpha_j - alpha_slope)
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

def calculate_F2(joint_dip):
    if joint_dip >= 45:
        return 1.0
    elif joint_dip >= 35:
        return 0.85
    elif joint_dip >= 25:
        return 0.70
    elif joint_dip >= 15:
        return 0.50
    elif joint_dip >= 10:
        return 0.20
    else:
        return 0.15

def calculate_F3(method):
    F3_values = {'planar': -60, 'toppling': -25, 'wedge': -50}
    return F3_values.get(method.lower(), 0)

def calculate_F4(excavation_method):
    methods = {
        'natural': 15,
        'pre-split blasting': 10,
        'smooth blasting': 8,
        'mechanical': 0,
        'poor blasting': -8
    }
    return methods.get(excavation_method.lower(), 0)

def calculate_SMR(RMRb, alpha_j, alpha_slope, joint_dip, method, excavation_method):
    F1 = calculate_F1(alpha_j, alpha_slope)
    F2 = calculate_F2(joint_dip)
    F3 = calculate_F3(method)
    F4 = calculate_F4(excavation_method)
    
    SMR = RMRb + (F1 * F2 * F3) + F4
    return SMR, (F1, F2, F3, F4)

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

st.set_page_config(page_title="SMR Tool", layout="wide")
st.title("⛰️ Slope Mass Rating (SMR) Calculator")

# Sidebar Inputs
with st.sidebar:
    st.header("Input Parameters")

    RMRb = st.slider("Basic Rock Mass Rating (RMRb)", 0, 100, 60)

    alpha_j = st.number_input("Joint dip direction (αᵢ, °)", min_value=0, max_value=360, value=120)
    joint_dip = st.number_input("Joint dip angle (βᵢ, °)", min_value=0, max_value=90, value=30)

    alpha_slope = st.number_input("Slope dip direction (αₛ, °)", min_value=0, max_value=360, value=110)
    slope_dip = st.number_input("Slope dip angle (βₛ, °)", min_value=0, max_value=90, value=60)

    method = st.selectbox("Failure mechanism", ['Planar', 'Toppling', 'Wedge'])
    excavation_method = st.selectbox("Excavation method", ['Natural', 'Pre-split blasting', 'Smooth blasting', 'Mechanical', 'Poor blasting'])

# Calculate SMR
SMR, factors = calculate_SMR(RMRb, alpha_j, alpha_slope, joint_dip, method, excavation_method)
F1, F2, F3, F4 = factors
SMR_class, SMR_desc = interpret_SMR(SMR)

# Display Stereonet and Results
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(4,4), subplot_kw={'projection':'stereonet'})
    ax.plane(alpha_j, joint_dip, 'g-', linewidth=2, label='Joint Plane (αᵢ, βᵢ)')
    ax.pole(alpha_j, joint_dip, 'ro', markersize=8, label='Joint Pole')
    ax.plane(alpha_slope, slope_dip, 'b--', linewidth=2, label='Slope Face (αₛ, βₛ)')
    ax.grid(True)
    ax.legend(loc='upper right')

    st.pyplot(fig)

    st.markdown("### 📌 Calculation Results")
    st.metric(label="Calculated SMR", value=f"{SMR:.2f}")
    st.write(f"**SMR Class:** {SMR_class}")
    st.write(f"**Description:** {SMR_desc}")
    st.write(f"**F₁:** {F1}, **F₂:** {F2}, **F₃:** {F3}, **F₄:** {F4}")

with col2:
    st.info("""
    ### ℹ️ How to Use:
    Adjust parameters in the sidebar to calculate the Slope Mass Rating (SMR).  
    The stereonet visually shows joint and slope orientations.
    """)

# SMR Interpretation Guideline at bottom
st.markdown("---")
st.markdown("### 📖 SMR Interpretation Guideline")

smr_interpretation = """
| SMR Value | Stability Class  | Description / Stability                |
|-----------|------------------|----------------------------------------|
| 81 - 100  | Class I          | Very good - Completely stable          |
| 61 - 80   | Class II         | Good - Stable                          |
| 41 - 60   | Class III        | Fair - Partially stable                |
| 21 - 40   | Class IV         | Poor - Unstable                        |
| 0 - 20    | Class V          | Very poor - Completely unstable        |
"""

st.markdown(smr_interpretation)
