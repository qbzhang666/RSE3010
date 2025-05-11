import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

st.set_page_config(page_title="CCM Analyzer Pro", layout="wide")
st.title("Advanced Convergence-Confinement Method Analysis")

# ================================================
# Sidebar Inputs
# ================================================
with st.sidebar:
    st.header("⚙️ Input Parameters")
    
    # Tunnel Parameters
    with st.expander("🏗️ Tunnel Parameters", expanded=True):
        r0 = st.number_input("Tunnel Radius (m)", 1.0, 10.0, 5.0)
        p0 = st.number_input("In-situ Stress (MPa)", 1.0, 50.0, 10.0)
        depth = st.number_input("Depth (m)", 50.0, 1000.0, 500.0)
    
    # Rock Parameters
    with st.expander("🪨 Rock Parameters"):
        failure_criterion = st.selectbox(
            "Rock Mass Failure Criterion",
            ["Mohr-Coulomb (Duncan-Fama)", "Hoek-Brown (Carranza-Torres)", "Generalized HB"]
        )
        
        E = st.number_input("Young's Modulus (MPa)", 500.0, 100000.0, 30000.0)
        nu = st.slider("Poisson's Ratio", 0.1, 0.49, 0.3)
        c = st.number_input("Cohesion (MPa)", 0.1, 10.0, 1.5)
        phi_deg = st.number_input("Friction Angle (°)", 5.0, 60.0, 30.0)
        
        if "Hoek-Brown" in failure_criterion:
            GSI = st.slider("GSI", 10, 100, 50)
            mi = st.number_input("Intact Rock Constant (mi)", 5.0, 30.0, 10.0)
            D = st.slider("Disturbance Factor (D)", 0.0, 1.0, 0.0)
    
    # LDP Parameters
    with st.expander("📏 LDP Parameters"):
        ldp_model = st.selectbox("LDP Model", ["Vlachopoulos", "Hoek", "Panet"])
        alpha = st.slider("Alpha (α)", 0.6, 0.95, 0.85)
        R_star = st.slider("Plastic Radius R*", 1.0, 5.0, 2.5)
    
    # Support Parameters
    with st.expander("🔧 Support System"):
        support_criteria = st.selectbox("Support Criteria", [
            "Distance from Face", 
            "Target Displacement", 
            "Convergence Percentage"
        ])
        
        if support_criteria == "Distance from Face":
            support_pos = st.slider("x/r₀ from Face", 0.0, 10.0, 1.5)
        elif support_criteria == "Target Displacement":
            u_install = st.number_input("uₛ₀ (mm)", 0.0, 500.0, 30.0) / 1000
        else:
            convergence_pct = st.slider("Convergence (%)", 0.0, 10.0, 1.0)
        
        k = st.number_input("Support Stiffness (MPa/m)", 100, 2000, 650)
        p_max = st.number_input("Max Support Pressure (MPa)", 0.5, 10.0, 3.0)

# ================================================
# Calculation Functions
# ================================================
@st.cache_data
def calculate_GRC():
    phi_rad = np.radians(phi_deg)
    G = E / (2 * (1 + nu))
    
    if "Mohr-Coulomb" in failure_criterion:
        sin_phi = np.sin(phi_rad)
        sigma_cm = (2 * c * np.cos(phi_rad)) / (1 - sin_phi)
        p_cr = (2 * p0 - sigma_cm) / (1 + (1 + sin_phi)/(1 - sin_phi))
    else:  # Hoek-Brown
        a = 0.5 + (np.exp(-GSI/15) - np.exp(-20/3)) / 6
        mb = mi * np.exp((GSI - 100)/(28 - 14*D))
        s = np.exp((GSI - 100)/(9 - 3*D))
        sigma_cm = (p0 * (mb + 4*s - a)) / (mb * (1 + a))
        p_cr = sigma_cm  # Simplified for demonstration
    
    p = np.linspace(0.1, p0, 500)
    u_r = np.zeros_like(p)
    
    for i, p_i in enumerate(p):
        if p_i >= p_cr:
            u_r[i] = (p0 - p_i) * r0 / (2 * G)
        else:
            k_rock = (1 + sin_phi) / (1 - sin_phi) if "Mohr" in failure_criterion else 2.0
            exponent = (k_rock - 1) / 2
            u_elastic = (p0 - p_cr) * r0 / (2 * G)
            u_r[i] = u_elastic * (p_cr / p_i) ** exponent
    
    return p, u_r, p_cr

def ldp_profile(x_star):
    if ldp_model == "Panet":
        return np.where(x_star <= 0,
                        1 - alpha * np.exp(-1.5 * x_star),
                        np.exp(-1.5 * x_star))
    elif ldp_model == "Hoek":
        return np.where(x_star <= 0,
                        0.25 * np.exp(2.5 * x_star),
                        1 - 0.75 * np.exp(-0.5 * x_star))
    elif ldp_model == "Vlachopoulos":
        return np.where(x_star <= 0,
                        (1/3) * np.exp(2 * x_star - 0.15 * x_star / alpha),
                        1 - (1 - (1/3) * np.exp(-0.15 * x_star)) * np.exp(-3 * x_star / R_star))

def calculate_SCC(u_values, u_install):
    scc = np.zeros_like(u_values)
    for i, u in enumerate(u_values):
        if u >= u_install:
            scc[i] = min(k * (u - u_install), p_max)
    return scc

# ================================================
# Main Calculations
# ================================================
p, u_r, p_cr = calculate_GRC()
G = E / (2 * (1 + nu))
u_max = np.max(u_r)

# Determine installation displacement
if support_criteria == "Distance from Face":
    u_install = ldp_profile(np.array([support_pos]))[0] * u_max
elif support_criteria == "Convergence Percentage":
    u_install = (convergence_pct / 100) * (2 * r0)
    
scc = calculate_SCC(u_r, u_install)

# Find intersection
intersection_points = np.argwhere(np.diff(np.sign(p - scc))).flatten()
if intersection_points.any():
    idx = intersection_points[0]
    u_int = u_r[idx]
    p_int = p[idx]
    fos = p_max / p_int
else:
    u_int = p_int = fos = None

# ================================================
# Visualization
# ================================================
col1, col2 = st.columns(2)

with col1:
    # GRC + SCC Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_r*1000, y=p, name="GRC", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=u_r*1000, y=scc, name="SCC", line=dict(dash='dash')))
    
    if u_int:
        fig.add_trace(go.Scatter(x=[u_int*1000], y=[p_int], 
                             mode='markers', marker=dict(size=12),
                             name=f"FoS = {fos:.2f}"))
    
    fig.update_layout(
        title="GRC-SCC Interaction",
        xaxis_title="Radial Displacement [mm]",
        yaxis_title="Pressure [MPa]",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # LDP Plot
    ldp_x = np.linspace(-5, 10, 500)
    ldp_y = ldp_profile(ldp_x)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ldp_x, y=ldp_y*u_max*1000, line=dict(width=2)))
    
    if support_criteria == "Distance from Face":
        fig2.add_vline(x=support_pos, line_dash="dot", 
                     annotation_text=f"Support @ {support_pos}r₀")
    
    fig2.update_layout(
        title="Longitudinal Displacement Profile",
        xaxis_title="Distance from Face (x/r₀)",
        yaxis_title="Displacement [mm]",
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)

# ================================================
# Results Panel
# ================================================
st.subheader("📊 Analysis Results")
cols = st.columns(4)
with cols[0]:
    st.metric("Critical Pressure", f"{p_cr:.2f} MPa")
with cols[1]:
    st.metric("Max Displacement", f"{u_max*1000:.1f} mm")
with cols[2]:
    value = f"{fos:.2f}" if fos else "N/A"
    st.metric("Factor of Safety", value)
with cols[3]:
    status = "✅ Stable" if fos and fos > 1.0 else "⚠️ Unstable"
    st.metric("Stability Status", status)

# ================================================
# Technical Documentation
# ================================================
with st.expander("📚 Theory & References"):
    st.markdown("""
    **Failure Criteria:**
    - Mohr-Coulomb (Duncan-Fama): Traditional shear strength model
    - Hoek-Brown (Carranza-Torres): Empirical rock mass failure criterion
    
    **LDP Models:**
    - Panet (1995): Exponential decay model
    - Vlachopoulos (2009): Combined elastic-plastic model
    - Hoek (2006): Empirical tunnel face effect model
    
    **Support Interaction:**
    - SCC calculated using support stiffness (k) and maximum capacity (p_max)
    - Installation timing determined by LDP position
    """)

# ================================================
# Data Export
# ================================================
if st.button("💾 Export Data"):
    import pandas as pd
    df = pd.DataFrame({
        'Displacement (m)': u_r,
        'GRC Pressure (MPa)': p,
        'SCC Pressure (MPa)': scc
    })
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "ccm_results.csv", "text/csv")
