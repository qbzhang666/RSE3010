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

# ---- Streamlit App ---- #

st.set_page_config(page_title="SMR Tool", layout="wide")

st.title("⛰️ Slope Mass Rating (SMR) Calculator")

# Sidebar Inputs
with st.sidebar:
    st.header("Input Parameters")

    RMRb = st.slider("Basic RMR (RMRb)", 0, 100, 60)

    alpha_j = st.number_input("Joint dip-direction (°)", min_value=0, max_value=360, value=120)
    alpha_slope = st.number_input("Slope face dip-direction (°)", min_value=0, max_value=360, value=110)
    joint_dip = st.number_input("Joint dip (°)", min_value=0, max_value=90, value=30)

    method = st.selectbox("Failure mechanism", ['Planar', 'Toppling', 'Wedge'])
    excavation_method = st.selectbox("Excavation method", ['Natural', 'Pre-split blasting', 'Smooth blasting', 'Mechanical', 'Poor blasting'])

# Calculate SMR
SMR, factors = calculate_SMR(RMRb, alpha_j, alpha_slope, joint_dip, method, excavation_method)
F1, F2, F3, F4 = factors

# Display Results
st.subheader("📌 Results")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Calculated SMR", value=f"{SMR:.2f}")
    st.write(f"**F1:** {F1}")
    st.write(f"**F2:** {F2}")
    st.write(f"**F3:** {F3}")
    st.write(f"**F4:** {F4}")

with col2:
    fig, ax = plt.subplots(subplot_kw={'projection':'stereonet'})
    ax.plane(alpha_j, joint_dip, 'g-', linewidth=2, label='Joint Plane')
    ax.pole(alpha_j, joint_dip, 'ro', markersize=8, label='Pole')
    ax.plane(alpha_slope, 90, 'b--', linewidth=1, label='Slope Face')
    ax.grid(True)
    ax.legend(loc='upper right')

    st.pyplot(fig)

st.markdown("---")

# Interpretation of SMR (optional helpful guideline)
st.subheader("📖 SMR Interpretation Guideline")

smr_interpretation = """
| SMR Value | Stability Class  | Description / Stability |
|-----------|------------------|-------------------------|
| 81 - 100  | Class I          | Very good - Completely stable |
| 61 - 80   | Class II         | Good - Stable |
| 41 - 60   | Class III        | Fair - Partially stable |
| 21 - 40   | Class IV         | Poor - Unstable |
| 0 - 20    | Class V          | Very poor - Completely unstable |
"""

st.markdown(smr_interpretation)
