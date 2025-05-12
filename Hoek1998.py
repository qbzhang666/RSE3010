import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import pandas as pd

# Page configuration
st.set_page_config(page_title="GRC and Plastic Zone Tool", layout="wide", page_icon="🏗️")
st.title("🛠️ Advanced Ground Reaction Curve (GRC) Analysis")

# Sidebar Inputs
with st.sidebar:
    st.header("Project Configuration")
    project_name = st.text_input("Analysis Name", "Tunnel Analysis 01")
    
    with st.expander("🚧 Tunnel Geometry", expanded=True):
        r0 = st.number_input("Tunnel radius $r_0$ (m)", 1.0, 20.0, 5.0, 0.1)
        p0 = st.number_input("In-situ stress $p_0$ (MPa)", 1.0, 100.0, 7.0, 0.1)

    with st.expander("📦 Material Properties", expanded=True):
        c = st.number_input("Cohesion $c$ (MPa)", 0.1, 20.0, 1.5, 0.1)
        phi_deg = st.slider("Friction angle $\\phi$ (°)", 0.0, 45.0, 23.0, 0.5)
        E = st.number_input("Young’s Modulus $E$ (MPa)", 500.0, 100000.0, 1800.0, 100.0)
        nu = st.slider("Poisson’s ratio $\\nu$", 0.1, 0.49, 0.3, 0.01)

# Calculations
try:
    phi = np.radians(phi_deg)
    sin_phi = np.sin(phi)
    k = (1 + sin_phi) / (1 - sin_phi)
    sigma_cm = max((2 * c * np.cos(phi)) / (1 - sin_phi), 1e-6)
    p_cr = max((2 * p0 - sigma_cm) / (1 + k), 0)
    G = E / (2 * (1 + nu))
    u_ie = (p0 - p_cr) * r0 / (2 * G) if p0 > p_cr else 0

    p = np.linspace(1e-6, p0, 500)
    elastic_mask = p >= p_cr
    plastic_mask = ~elastic_mask
    
    u_r = np.zeros_like(p)
    R_p = np.zeros_like(p)

    u_r[elastic_mask] = (p0 - p[elastic_mask]) * r0 / (2 * G)
    R_p[elastic_mask] = r0

    if np.any(plastic_mask):
        numerator = 2 * (p0 * (k - 1) + sigma_cm)
        denominator = (1 + k) * ((k - 1) * p[plastic_mask] + sigma_cm)
        R_p[plastic_mask] = r0 * (numerator / denominator) ** (1 / (k - 1))
        u_r[plastic_mask] = u_ie * (p_cr / p[plastic_mask]) ** ((k - 1) / 2)

except Exception as e:
    st.error(f"🚨 Calculation Error: {str(e)}")
    st.stop()

# Tabs for Output
tab1, tab2, tab3 = st.tabs(["📈 Interactive Plot", "📝 Derived Values", "📦 Data Export"])

with tab1:
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(11, 6.5), facecolor='white')
    
    grc_line, = ax1.plot(u_r, p, '#1f77b4', lw=2.5, label="Ground Reaction Curve")
    crit_line = ax1.axhline(p_cr, color='#d62728', ls='--', lw=2, label=f"$p_{{cr}}$ = {p_cr:.2f} MPa")
    trans_line = ax1.axvline(u_ie, color='#228b22', ls='-.', lw=2, label=f"$u_{{ie}}$ = {u_ie:.3f} m")

    ax1.set_xlabel("Radial Displacement $u_r$ (m)")
    ax1.set_ylabel("Support Pressure $p_i$ (MPa)")
    ax1.set_ylim(0, max(p0, p_cr)*1.15)
    ax1.set_xlim(0, 0.08)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    rp_line, = ax2.plot(u_r, R_p, color='#ff7f0e', lw=2.5, label="Plastic Radius $R_p$")
    ax2.set_ylabel("Plastic Radius $R_p$ (m)")
    ax2.set_ylim(r0, max(R_p)*1.1 if max(R_p) > r0 else r0*1.1)

    lines = [grc_line, crit_line, trans_line, rp_line]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    plt.title(f"GRC Analysis: {project_name}", fontsize=14, pad=20)
    st.pyplot(fig)

with tab2:
    st.markdown("### 🧮 Derived Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Critical Pressure", f"{p_cr:.2f} MPa")
        st.metric("Shear Modulus (G)", f"{G:.0f} MPa")
        st.metric("Sigma_cm", f"{sigma_cm:.2f} MPa")
    with col2:
        st.metric("Transition Displacement", f"{u_ie:.4f} m")
        st.metric("k-value", f"{k:.2f}")
        st.metric("Max Plastic Radius", f"{max(R_p):.1f} m")

    with st.expander("🔍 Analysis Validation", expanded=True):
        if p_cr <= 0:
            st.error("Critical pressure ≤ 0 - Plastic zone exists for all p_i")
        elif p0 < p_cr:
            st.warning("In-situ stress < Critical pressure - Elastic behavior only")
        else:
            st.success("Valid elasto-plastic analysis domain")

with tab3:
    # Data export system
    st.markdown("### 📥 Export Results")
    
    # Create dataframe with formatted values
    df = pd.DataFrame({
        "Support Pressure (MPa)": np.round(p, 2),
        "Radial Displacement (m)": np.round(u_r, 4),
        "Plastic Radius (m)": np.round(R_p, 2),
        "Zone Type": np.where(p >= p_cr, "Elastic", "Plastic")
    })
    
    # SINGLE SET OF EXPORT CONTROLS
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📄 Download CSV Data",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"grc_data_{project_name}.csv",
            mime='text/csv',
            key="csv_export_tab3"
        )
        
    with col2:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor='white')
        st.download_button(
            label="🖼️ Download Plot",
            data=buf.getvalue(),
            file_name=f"grc_plot_{project_name}.png",
            mime="image/png",
            key="plot_export_tab3"
        )
    
    # Display parameters and equations
    st.markdown("**Analysis Parameters:**")
    st.code(f"""Project Name: {project_name}
Tunnel Radius: {r0:.2f} m
In-Situ Stress: {p0:.2f} MPa
Cohesion: {c:.2f} MPa
Friction Angle: {phi_deg:.1f}°
Young's Modulus: {E:.0f} MPa
Poisson's Ratio: {nu:.2f}""", language="text")

    st.markdown("**Fundamental Equations:**")
    st.latex(r"""
    \begin{aligned}
    1.\ & \text{Critical Pressure:} \\
    & p_{cr} = \frac{2p_0 - \sigma_{cm}}{1 + k} \\
    2.\ & \sigma_{cm}: \\
    & \sigma_{cm} = \frac{2c\cos\phi}{1 - \sin\phi} \\
    3.\ & \text{k-value:} \\
    & k = \frac{1 + \sin\phi}{1 - \sin\phi} \\
    4.\ & \text{Shear Modulus:} \\
    & G = \frac{E}{2(1+\nu)} \\
    5.\ & \text{Transition Displacement:} \\
    & u_{ie} = \frac{(p_0 - p_{cr}) r_0}{2G}
    \end{aligned}
    """)

# Warnings
if max(u_r) > 0.1:
    st.warning("⚠️ Significant displacements detected (>0.1m) - Consider support system design")
if max(R_p) > 2 * r0:
    st.warning("⚠️ Large plastic zone (R_p > 2r₀) - Reevaluate reinforcement strategy")
if p_cr < 0.2 * p0:
    st.warning("⚠️ Low critical pressure ratio (p_cr/p₀ < 0.2) - Verify material parameters")
