import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="GRC & Plastic Zone Tool", layout="wide", page_icon="🏗️")
st.title("🛠️ Ground Reaction Curve, Plastic Zone & Support System Analyzer")

# ---------------- Sidebar: Input Parameters ----------------
with st.sidebar:
    st.header("Project Details")
    project_name = st.text_input("Project Name", "Tunnel Analysis 01")
    analysis_type = st.selectbox("Analysis Type", ["Elasto-Plastic", "Elastic Only"], index=0)

    st.header("Geometric Parameters")
    r0 = st.number_input("Tunnel Radius r₀ (m)", 1.0, 20.0, 5.0, step=0.1)
    p0 = st.number_input("In-situ Stress p₀ (MPa)", 1.0, 100.0, 7.0, step=0.1)

    st.header("Material Properties")
    c = st.number_input("Cohesion c (MPa)", 0.1, 20.0, 1.5, step=0.1)
    phi_deg = st.slider("Friction Angle φ (°)", 0.0, 45.0, 23.0, step=0.5)
    E = st.number_input("Young’s Modulus E (MPa)", 500.0, 100000.0, 1800.0, step=100.0)
    nu = st.slider("Poisson’s Ratio ν", 0.1, 0.49, 0.3, step=0.01)

# ---------------- Core Calculations ----------------
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

    if analysis_type == "Elastic Only":
        u_r[:] = (p0 - p) * r0 / (2 * G)
        R_p[:] = r0  # no plastic zone
    else:
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


# ---------------- Main Plot ----------------
fig, ax1 = plt.subplots(figsize=(12, 7))
plt.style.use('default')  # Clean, no background

x_buffer = max(u_r) * 0.1
y_buffer = max(p0, p_cr) * 0.1

ax1.plot(u_r, p, 'b-', lw=2.5, label="Ground Reaction Curve (GRC)")
ax1.axhline(p_cr, color='r', ls='--', lw=2, label=f"$p_{{cr}}$ = {p_cr:.2f} MPa")
ax1.axvline(u_ie, color='purple', ls='-.', lw=2, label=f"$u_{{ie}}$ = {u_ie:.3f} m")

ax1.set_xlabel("Radial Displacement $u_r$ (m)")
ax1.set_ylabel("Support Pressure $p_i$ (MPa)")
ax1.set_ylim(0, max(p0, p_cr) + y_buffer)
x_max = st.sidebar.slider("Max Displacement Axis (m)", 0.02, 0.2, 0.08, step=0.01)
ax1.set_xlim(0, x_max)
ax1.tick_params(axis='both', which='major', labelsize=10)

ax2 = ax1.twinx()
ax2.plot(u_r, R_p, 'darkorange', lw=2.5, label="Plastic Radius $R_p$")
ax2.set_ylabel("Plastic Radius $R_p$ (m)")
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
        Based on:
        - Mohr-Coulomb yield criterion  
        - Kastner's equation  
        - Closed-form solution for circular tunnels

        $$p_{cr} = \\frac{2p_0 - \\sigma_{cm}}{1 + K}$$  
        $$R_p = r_0 \\left[\\frac{2(p_0(K-1) + \\sigma_{cm})}{(1+K)((K-1)p_i + \\sigma_{cm})}\\right]^{1/(K-1)}$$
        """)

    with st.expander("⚠️ Analysis Check"):
        if p_cr <= 0:
            st.error("Critical pressure ≤ 0 → plastic zone forms for all $p_i$")
        elif p0 < p_cr:
            st.warning("In-situ stress < critical pressure → elastic behaviour only")
        else:
            st.success("Valid elasto-plastic analysis range")

# ---------------- Module 1: Time-Dependent Support ----------------
with st.expander("⏳ Time-Dependent Support Pressure Simulation"):
    st.subheader("Support Pressure vs. Time")

    colA, colB, colC = st.columns(3)
    with colA:
        p_final = st.number_input("Final Support Pressure p_final (MPa)", 0.5, 20.0, 3.0, step=0.1)
    with colB:
        tau = st.number_input("Time Constant τ (days)", 0.1, 30.0, 5.0, step=0.1)
    with colC:
        t_max = st.number_input("Max Simulation Time (days)", 1.0, 100.0, 30.0, step=1.0)

    t_vals = np.linspace(0, t_max, 200)
    p_support = p_final * (1 - np.exp(-t_vals / tau))

    plt.style.use('default')
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_vals, p_support, 'g-', lw=2)
    ax.set_title("Time-Dependent Support Pressure", fontsize=12)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Support Pressure $p_s(t)$ (MPa)")
    ax.grid(True)
    st.pyplot(fig2)

    st.markdown(
        f"- Support pressure at time τ (t = {tau}): **{p_final * (1 - np.exp(-1)):.2f} MPa**\\n"
        f"- Final support pressure at {t_max:.0f} days: **{p_support[-1]:.2f} MPa**"
    )

# ---------------- Module 2: Support Characteristic Curve ----------------
with st.expander("🔄 Support Characteristic Curve (SCC)"):
    st.subheader("Support Pressure vs Radial Displacement")

    colA, colB = st.columns(2)
    with colA:
        k_supp = st.number_input("Support Stiffness k (MPa/m)", 10.0, 10000.0, 500.0, step=50.0)
    with colB:
        u_install = st.number_input("Installation Displacement $u_{install}$ (m)", 0.001, 0.1, 0.02, step=0.001)

    u_vals = np.linspace(0, max(u_r)*1.1, 300)
    p_scc = np.where(u_vals >= u_install, k_supp * (u_vals - u_install), 0)

    plt.style.use('default')
    fig3, ax = plt.subplots(figsize=(8, 4))
    ax.plot(u_vals, p_scc, 'm-', lw=2.5)
    ax.set_title("Support Characteristic Curve", fontsize=12)
    ax.set_xlabel("Radial Displacement $u$ (m)")
    ax.set_ylabel("Support Pressure $p_s$ (MPa)")
    ax.grid(True)
    st.pyplot(fig3)

    st.markdown(f"- Peak support pressure at max displacement ({max(u_vals):.3f} m): **{max(p_scc):.2f} MPa**")

# ---------------- Module 3: TBM / NATM Parameters ----------------
with st.expander("📐 TBM / NATM Parameters"):
    st.subheader("Advance Rate & Convergence Control")

    colA, colB = st.columns(2)
    with colA:
        advance_rate = st.number_input("TBM Advance Rate (m/day)", 0.1, 50.0, 5.0, step=0.5)
    with colB:
        lag_distance = st.number_input("Support Installation Lag (m)", 0.1, 20.0, 2.0, step=0.5)

    time_lag = lag_distance / advance_rate
    convergence_rate = st.number_input("Expected Convergence per Day (mm/day)", 0.1, 20.0, 1.5, step=0.1)
    max_days = st.slider("Monitor Duration (days)", 1, 100, 30)

    time_days = np.linspace(0, max_days, 200)
    convergence_mm = np.minimum(time_days * convergence_rate, 100)

    plt.style.use('default')
    fig4, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_days, convergence_mm, 'c-', lw=2)
    ax.axvline(time_lag, color='red', ls='--', label=f"Support Lag = {time_lag:.1f} days")
    ax.set_title("Face Convergence Monitoring", fontsize=12)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Convergence (mm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig4)

    st.markdown(f"- Support installation at: **{time_lag:.1f} days**\\n"
                f"- Max monitored convergence: **{convergence_mm[-1]:.1f} mm**")
