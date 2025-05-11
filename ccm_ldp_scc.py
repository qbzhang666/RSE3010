# Streamlit App: CCM with Corrected GRC Transition (MC + Hoek-Brown)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CCM Analysis Tool", layout="wide")
st.title("Convergence-Confinement Method Analysis")

# Constants
GRAVITY = 9.81  # m/s²

# 1. Input Parameters
with st.sidebar:
    st.header("1. Tunnel Parameters")
    r0 = st.number_input("Tunnel Radius [m]", 1.0, 10.0, 5.0)
    diameter = 2 * r0
    depth = st.number_input("Tunnel Depth [m]", 10.0, 5000.0, 500.0)

    st.header("2. Rock Parameters")
    density = st.number_input("Rock Density [kg/m³]", 1500.0, 3500.0, 2650.0)
    p0 = (depth * density * GRAVITY) / 1e6  # MPa
    st.metric("In-situ Stress p₀ [MPa]", f"{p0:.2f}")
    E = st.number_input("Young's Modulus E [MPa]", 500.0, 1e5, 10000.0)
    nu = st.number_input("Poisson's Ratio ν", 0.10, 0.49, 0.30, step=0.01)

    st.subheader("Failure Criterion")
    criterion = st.selectbox("Select Criterion", ["Mohr-Coulomb", "Hoek-Brown"])

    if criterion == "Mohr-Coulomb":
        c = st.number_input("Cohesion c [MPa]", 0.1, 10.0, 1.5)
        phi_deg = st.number_input("Friction Angle φ [°]", 5.0, 60.0, 30.0)
        phi_rad = np.radians(phi_deg)
    else:
        sigma_ci = st.number_input("σ_ci [MPa]", 1.0, 100.0, 30.0)
        GSI = st.slider("GSI", 10, 100, 75)
        mi = st.number_input("mᵢ", 1.0, 50.0, 15.0)
        D = st.slider("Disturbance Factor D", 0.0, 1.0, 0.0)

        mb = mi * np.exp((GSI - 100) / (28 - 14 * D))
        s_val = np.exp((GSI - 100) / (9 - 3 * D))
        a_val = 0.5 + (1 / 6) * (np.exp(-GSI / 15) - np.exp(-20 / 3))

        st.markdown(f"**Calculated Parameters:**  \n"
                    f"- m_b = {mb:.2f}  \n"
                    f"- s = {s_val:.4f}  \n"
                    f"- a = {a_val:.3f}")

# 2. GRC Calculation
def calculate_GRC():
    G = E / (2 * (1 + nu))
    p = np.linspace(p0, 0.1, 500)
    u = np.zeros_like(p)

    if criterion == "Mohr-Coulomb":
        sin_phi = np.sin(phi_rad)
        k = (1 + sin_phi) / (1 - sin_phi)
        sigma_cm = (2 * c * np.cos(phi_rad)) / (1 - sin_phi)
        p_cr = (2 * p0 - sigma_cm) / (1 + k)
        u_elastic = (p0 - p_cr) * r0 / (2 * G)

        for i, pi in enumerate(p):
            if pi >= p_cr:
                u[i] = (p0 - pi) * r0 / (2 * G)
            else:
                exponent = (k - 1) / 2
                u[i] = u_elastic * (p_cr / pi) ** exponent

    else:  # Hoek-Brown
        sigma_cm = (sigma_ci / 2) * ((mb + 4 * s_val) ** a_val - mb ** a_val)
        p_cr = p0 - sigma_cm / 2
        k_HB = (2 * (1 - nu) * (mb + 4 * s_val) ** a_val) / (1 + nu)
        R_pl = r0 * ((2 * p0 / sigma_cm) + 1) ** (1 / k_HB)
        u_elastic = (p0 - p_cr) * r0 / (2 * G)

        for i, pi in enumerate(p):
            if pi >= p_cr:
                u[i] = (p0 - pi) * r0 / (2 * G)
            else:
                u[i] = u_elastic * (p_cr / pi) ** k_HB

    return p, u, p_cr

p_grc, u_grc, p_cr = calculate_GRC()

# 3. LDP & Support System
with st.sidebar:
    st.header("3. LDP and Support Criteria")
    ldp_model = st.selectbox("LDP Model", ["Vlachopoulos", "Panet", "Hoek"])
    alpha = st.slider("α", 0.6, 0.95, 0.85)
    R_star = st.slider("R*", 1.0, 5.0, 2.5)
    install_criteria = st.selectbox("Installation Criteria", [
        "Distance from face", "Displacement threshold", "Convergence %"
    ])
    if install_criteria == "Distance from face":
        x_install = st.slider("x/r₀", 0.0, 10.0, 1.5)
    elif install_criteria == "Displacement threshold":
        u_install = st.number_input("uₛ₀ [mm]", 1.0, 500.0, 30.0) / 1000
    elif install_criteria == "Convergence %":
        conv_pct = st.slider("Convergence [%]", 0.01, 1.0, 0.05)
        u_install = (conv_pct / 100) * diameter

# LDP calculation
ldp_x = np.linspace(-5, 10, 500)
def ldp_profile(x, model, alpha, R_star):
    if model == "Panet":
        return np.where(x <= 0, 1 - alpha * np.exp(-1.5 * x), np.exp(-1.5 * x))
    elif model == "Hoek":
        return np.where(x <= 0, 0.25 * np.exp(2.5 * x), 1 - 0.75 * np.exp(-0.5 * x))
    elif model == "Vlachopoulos":
        return np.where(x <= 0,
                        (1 / 3) * np.exp(2 * x - 0.15 * x / alpha),
                        1 - (1 - (1 / 3) * np.exp(-0.15 * x)) * np.exp(-3 * x / R_star))

ldp_y = ldp_profile(ldp_x, ldp_model, alpha, R_star)
u_max = 0.04  # 40 mm
u_ldp = ldp_y * u_max

if install_criteria == "Distance from face":
    u_install = np.interp(x_install, ldp_x, u_ldp)

# 4. SCC
with st.sidebar:
    st.header("4. Support System & Threshold")
    k_supp = st.number_input("Support Stiffness k [MPa/m]", 100, 5000, 650)
    p_max = st.number_input("Max Support Pressure pₛₘ [MPa]", 0.1, 10.0, 3.0)
    disp_thresh = st.slider("Threshold [mm]", 10, 100, 40)

u_scc = np.linspace(0, np.max(u_grc) * 1.2, 500)
scc_vals = np.where(u_scc >= u_install, np.minimum(k_supp * (u_scc - u_install), p_max), 0)
scc_on_grc = np.where(u_grc >= u_install, np.minimum(k_supp * (u_grc - u_install), p_max), 0)

def find_intersection(u_vals, grc_vals, scc_vals):
    for i in range(1, len(u_vals)):
        if (grc_vals[i] - scc_vals[i]) * (grc_vals[i - 1] - scc_vals[i - 1]) < 0:
            u_eq = np.interp(0, [grc_vals[i - 1] - scc_vals[i - 1], grc_vals[i] - scc_vals[i]],
                             [u_vals[i - 1], u_vals[i]])
            p_eq = np.interp(u_eq, u_vals, grc_vals)
            return u_eq, p_eq
    return None, None

u_eq, p_eq = find_intersection(u_grc, p_grc, scc_on_grc)
FoS = p_max / p_eq if p_eq else float("inf")

# 5. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# GRC + SCC
ax1.plot(u_grc * 1000, p_grc, label="GRC", lw=2)
ax1.plot(u_scc * 1000, scc_vals, '--', color='orange', label="SCC", lw=2)
ax1.axvline(disp_thresh, linestyle=':', color='red', label=f"Threshold = {disp_thresh} mm")
ax1.set_xlim(0, 100)
ax1.set_xlabel("Tunnel Wall Displacement [mm]")
ax1.set_ylabel("Radial Pressure [MPa]")
ax1.set_title("GRC + SCC Interaction")
ax1.grid(True, color='lightgrey', alpha=0.4)
if u_eq:
    ax1.legend(title=f"FoS = pₛₘ / P_eq = {p_max:.2f} / {p_eq:.2f} = {FoS:.2f}")
else:
    ax1.legend()

# LDP Plot
ax2.plot(ldp_x, u_ldp * 1000, label=f"{ldp_model} LDP", lw=2)
if install_criteria == "Distance from face":
    ax2.axvline(x_install, color='r', linestyle='--', label=f'Support @ x/r₀ = {x_install:.2f}')
ax2.set_xlabel("Normalized Distance x/r₀")
ax2.set_ylabel("Radial Displacement [mm]")
ax2.set_title("Longitudinal Deformation Profile")
ax2.grid(True, color='lightgrey', alpha=0.4)
ax2.legend()

st.pyplot(fig)

# 6. Safety Summary
st.markdown("### Safety Summary")
st.write(f"- Critical Pressure $p_{{cr}}$: **{p_cr:.2f} MPa**")
st.write(f"- Installation Displacement $u_{{install}}$: **{u_install*1000:.2f} mm**")
if u_eq:
    st.success(f"✅ Intersection at {u_eq*1000:.2f} mm → Pressure = {p_eq:.2f} MPa → FoS = {FoS:.2f}")
else:
    st.warning("⚠️ No intersection found – support system may be insufficient.")

# Documentation
with st.expander("📘 Geotechnical Parameters"):
    st.markdown(f"""
    - \( p_0 = \\frac{{\\rho g z}}{{10^6}} = \\frac{{{density:.0f} \cdot 9.81 \cdot {depth:.0f}}}{{10^6}} = {p0:.2f} \text{{ MPa}} \)
    """)
