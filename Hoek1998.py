import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="GRC & Plastic Zone Tool", layout="wide", page_icon="🏗️")
st.title("🛠️ Ground Reaction Curve & Plastic Zone Analyzer")

# ---------------- Sidebar: Input Parameters ----------------
with st.sidebar:
    st.header("Project Details")
    project_name = st.text_input("Project Name", "Tunnel Analysis 01")
    analysis_type = st.selectbox("Analysis Type", ["Elasto-Plastic", "Elastic Only"], index=0)

    st.header("Geometric Parameters")
    r0 = st.number_input("Tunnel Radius r₀ (m)", 1.0, 20.0, 5.0, step=0.1,
                         help="Radius of the excavated tunnel")
    p0 = st.number_input("In-situ Stress p₀ (MPa)", 1.0, 100.0, 7.0, step=0.1,
                         help="Initial geostatic stress")

    st.header("Material Properties")
    c = st.number_input("Cohesion c (MPa)", 0.1, 20.0, 1.5, step=0.1,
                        help="Cohesion of rock/soil")
    phi_deg = st.slider("Friction Angle φ (°)", 0.0, 45.0, 23.0, step=0.5,
                        help="Internal friction angle")
    E = st.number_input("Young’s Modulus E (MPa)", 500.0, 100000.0, 1800.0, step=100.0,
                        help="Elastic modulus of the ground")
    nu = st.slider("Poisson’s Ratio ν", 0.1, 0.49, 0.3, step=0.01,
                   help="Poisson’s ratio of the material")

# ---------------- Calculations with Error Handling ----------------
try:
    phi = np.radians(phi_deg)
    sin_phi = np.sin(phi)
    k = (1 + sin_phi) / (1 - sin_phi)
    sigma_cm = (2 * c * np.cos(phi)) / (1 - sin_phi)
    p_cr = max((2 * p0 - sigma_cm) / (1 + k), 0)
    G = E / (2 * (1 + nu))
    u_ie = (p0 - p_cr) * r0 / (2 * G) if p0 > p_cr else 0

    p = np.linspace(1e-6, p0, 300)
    u_r = np.zeros_like(p)
    R_p = np.zeros_like(p)

    elastic_mask = p >= p_cr
    plastic_mask = ~elastic_mask

    u_r[elastic_mask] = (p0 - p[elastic_mask]) * r0 / (2 * G)
    R_p[elastic_mask] = r0

    numerator = 2 * (p0 * (k - 1) + sigma_cm)
    denominator = (1 + k) * ((k - 1) * p[plastic_mask] + sigma_cm)
    R_p[plastic_mask] = r0 * (numerator / denominator) ** (1 / (k - 1))
    u_r[plastic_mask] = u_ie * (p_cr / p[plastic_mask]) ** ((k - 1) / 2)

except Exception as e:
    st.error(f"Error in calculations: {str(e)}")
    st.stop()

# ---------------- Plotting ----------------
fig, ax1 = plt.subplots(figsize=(12, 7))

# Style handling: fallback if seaborn style not available
try:
    plt.style.use('seaborn-darkgrid')
except OSError:
    plt.style.use('ggplot')

x_buffer = max(u_r) * 0.1
y_buffer = max(p0, p_cr) * 0.1

ax1.plot(u_r, p, 'b-', lw=2.5, label="Ground Reaction Curve (GRC)")
ax1.axhline(p_cr, color='r', ls='--', lw=2, label=f"$p_{{cr}}$ = {p_cr:.2f} MPa")
ax1.axvline(u_ie, color='purple', ls='-.', lw=2, label=f"$u_{{ie}}$ = {u_ie:.3f} m")

ax1.set_xlabel("Radial Displacement $u_r$ (m)", fontsize=12)
ax1.set_ylabel("Support Pressure $p_i$ (MPa)", fontsize=12)
ax1.set_ylim(0, max(p0, p_cr) + y_buffer)
ax1.set_xlim(0, max(u_r) + x_buffer)
ax1.tick_params(axis='both', which='major', labelsize=10)

ax2 = ax1.twinx()
ax2.plot(u_r, R_p, 'darkorange', lw=2.5, label="Plastic Radius $R_p$")
ax2.set_ylabel("Plastic Radius $R_p$ (m)", fontsize=12)
ax2.set_ylim(r0, max(R_p) * 1.1)
ax2.tick_params(axis='y', labelcolor='darkorange', labelsize=10)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

plt.title(f"Ground Reaction Analysis: {project_name}", fontsize=14, pad=20)

# ---------------- Layout ----------------
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.pyplot(fig)
    if st.button("📥 Export Data"):
        csv_data = StringIO()
        np.savetxt(csv_data, np.column_stack((u_r, p, R_p)), delimiter=',',
                   header="u_r(m),p_i(MPa),R_p(m)", comments='')
        st.download_button("Download CSV", data=csv_data.getvalue(),
                           file_name="grc_results.csv", mime="text/csv")

with col2:
    st.subheader("🔢 Key Metrics")
    st.metric("Critical Pressure $p_{cr}$", f"{p_cr:.2f} MPa")
    st.metric("Plastic Radius Range", f"{r0:.1f} - {max(R_p):.1f} m")
    st.metric("Max Displacement", f"{max(u_r):.3f} m")

    st.subheader("📌 Material Constants")
    st.write(f"K = {k:.2f}")
    st.write(f"G = {G:.0f} MPa")
    st.write(f"$\\sigma_{{cm}}$ = {sigma_cm:.2f} MPa")

with col3:
    with st.expander("📘 Model Theory"):
        st.write("""
        **Based on:**
        - Mohr-Coulomb yield criterion  
        - Closed-form solution for circular tunnels  
        - Kastner's equation for critical pressure

        **Equations:**
        $$
        p_{cr} = \\frac{2p_0 - \\sigma_{cm}}{1 + K}
        $$
        $$
        R_p = r_0 \\left[\\frac{2(p_0(K-1) + \\sigma_{cm})}{(1+K)((K-1)p_i + \\sigma_{cm})}\\right]^{1/(K-1)}
        $$
        """)

    with st.expander("⚠️ Analysis Check"):
        if p_cr <= 0:
            st.error("Critical pressure ≤ 0 → plastic zone forms for all $p_i$")
        elif p0 < p_cr:
            st.warning("In-situ stress < critical pressure → elastic behaviour only")
        else:
            st.success("Valid elasto-plastic analysis range")

# ---------------- Alerts ----------------
if max(u_r) > 0.1:
    st.warning("⚠️ Large displacements detected – consider support system design.")
if max(R_p) > 3 * r0:
    st.warning("⚠️ Extensive plastic zone – reevaluate reinforcement strategy.")
