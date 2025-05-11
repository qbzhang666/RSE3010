import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CCM Analysis Tool", layout="wide")
st.title("Convergence-Confinement Method (CCM) – Interactive Analysis")

# 1. Tunnel Parameters
st.sidebar.header("1. Tunnel Parameters")
r0 = st.sidebar.number_input("Tunnel Radius (m)", 1.0, 10.0, 5.0)

# 2. Rock / Soil Parameters
st.sidebar.header("2. Rock / Soil Parameters")
p0 = st.sidebar.number_input("In-situ Stress (MPa)", 1.0, 50.0, 10.0)
E = st.sidebar.number_input("Young's Modulus (MPa)", 500.0, 100000.0, 30000.0)
nu = st.sidebar.slider("Poisson's Ratio", 0.1, 0.49, 0.3)
c = st.sidebar.number_input("Cohesion (MPa)", 0.1, 10.0, 1.5)
phi_deg = st.sidebar.number_input("Friction Angle (°)", 5.0, 60.0, 30.0)

# Derived parameters
phi_rad = np.radians(phi_deg)
sin_phi = np.sin(phi_rad)
k_rock = (1 + sin_phi) / (1 - sin_phi)
sigma_cm_MC = (2 * c * np.cos(phi_rad)) / (1 - sin_phi)
p_cr = (2 * p0 - sigma_cm_MC) / (1 + k_rock)
G = E / (2 * (1 + nu))
diameter = 2 * r0

# GRC
p_vals = np.linspace(0.1, p0, 500)
u_grc = np.zeros_like(p_vals)
for i, p_i in enumerate(p_vals):
    if p_i >= p_cr:
        u_grc[i] = (p0 - p_i) * r0 / (2 * G)
    else:
        exponent = (k_rock - 1) / 2
        u_elastic = (p0 - p_cr) * r0 / (2 * G)
        u_grc[i] = u_elastic * (p_cr / p_i) ** exponent

# 3. LDP
st.sidebar.header("3. LDP Model")
ldp_model = st.sidebar.selectbox("LDP Model", ["Panet", "Hoek", "Vlachopoulos"])
alpha = st.sidebar.slider("Alpha (α)", 0.6, 0.95, 0.85)
R_star = st.sidebar.slider("Plastic Radius R*", 1.0, 5.0, 2.5)
ldp_x = np.linspace(-5, 10, 500)

def ldp_profile(x_star, model, alpha, R_star):
    if model == "Panet":
        return np.where(x_star <= 0,
                        1 - alpha * np.exp(-1.5 * x_star),
                        np.exp(-1.5 * x_star))
    elif model == "Hoek":
        return np.where(x_star <= 0,
                        0.25 * np.exp(2.5 * x_star),
                        1 - 0.75 * np.exp(-0.5 * x_star))
    elif model == "Vlachopoulos":
        return np.where(x_star <= 0,
                        (1/3) * np.exp(2 * x_star - 0.15 * x_star / alpha),
                        1 - (1 - (1/3) * np.exp(-0.15 * x_star)) * np.exp(-3 * x_star / R_star))

ldp_y = ldp_profile(ldp_x, ldp_model, alpha, R_star)
u_max = np.max(u_grc)
ldp_u = ldp_y * u_max

# 4. SCC Input
st.sidebar.header("4. Support System & SCC")
k_supp = st.sidebar.number_input("Support Stiffness (MPa/m)", 100, 2000, 650)
p_max = st.sidebar.number_input("Max Support Pressure (MPa)", 0.5, 10.0, 3.0)

# Support criteria
install_criteria = st.sidebar.selectbox("Support Installation Criteria", [
    "Distance from face", "Convergence %", "Target displacement (uₛ₀)"
])
if install_criteria == "Distance from face":
    x_install = st.sidebar.slider("Support Distance from Face (x/r₀)", -5.0, 10.0, 1.5)
    u_install = np.interp(x_install, ldp_x, ldp_u)
elif install_criteria == "Convergence %":
    conv_pct = st.sidebar.slider("Convergence (%)", 0.1, 10.0, 1.0)
    u_install = (conv_pct / 100) * diameter
elif install_criteria == "Target displacement (uₛ₀)":
    u_install_mm = st.sidebar.number_input("Target Displacement uₛ₀ (mm)", 0.0, 100.0, 30.0)
    u_install = u_install_mm / 1000

# Threshold input
st.sidebar.header("5. Displacement Threshold")
threshold_mm = st.sidebar.number_input("Threshold (mm)", 1.0, 100.0, 30.0)
threshold_m = threshold_mm / 1000

# SCC calculation
def calculate_SCC(u_vals, k, u0, p_max):
    return np.clip(k * (u_vals - u0), 0, p_max)

scc_vals = calculate_SCC(u_grc, k_supp, u_install, p_max)

# Intersection
def find_intersection(u_vals, grc_vals, scc_vals):
    for i in range(1, len(u_vals)):
        if (grc_vals[i] - scc_vals[i]) * (grc_vals[i - 1] - scc_vals[i - 1]) < 0:
            u_int = np.interp(0, [scc_vals[i - 1] - grc_vals[i - 1], scc_vals[i] - grc_vals[i]],
                              [u_vals[i - 1], u_vals[i]])
            p_eq = np.interp(u_int, u_vals, grc_vals)
            return u_int, p_eq
    return None, None

u_eq, p_eq = find_intersection(u_grc, p_vals, scc_vals)
fos = p_max / p_eq if p_eq and p_eq > 0 else None

# GRC + SCC plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(u_grc * 1000, p_vals, label="GRC", lw=2)
ax.plot(u_grc * 1000, scc_vals, '--', color='orange', lw=2, label="SCC")
ax.axvline(threshold_mm, color='red', linestyle=':', lw=1.5, label=f"Threshold = {threshold_mm:.0f} mm")
if u_eq and p_eq:
    ax.legend(title=f"FoS = pₛₘ / pₑq = {p_max:.2f} / {p_eq:.2f} = {fos:.2f}")
else:
    ax.legend()
ax.set_xlabel("Tunnel Wall Displacement [mm]")
ax.set_ylabel("Radial Stress [MPa]")
ax.set_title("GRC + SCC Interaction")
ax.grid(True)
st.pyplot(fig)

# LDP plot
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(ldp_x, ldp_u * 1000, lw=2, label=f"LDP: {ldp_model}")
if install_criteria == "Distance from face":
    ax2.axvline(x_install, color='r', linestyle='--', label=f"Support at x/r₀ = {x_install}")
ax2.set_xlabel("Distance to Tunnel Face (x/r₀)")
ax2.set_ylabel("Radial Displacement [mm]")
ax2.set_title("Longitudinal Deformation Profile (LDP)")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# Summary
st.markdown("### Summary")
st.write(f"- Critical Pressure $p_{{cr}}$: **{p_cr:.2f} MPa**")
st.write(f"- Support installed at displacement: **{u_install * 1000:.2f} mm**")
if u_eq and p_eq:
    st.success(f"✅ GRC and SCC intersect at displacement = {u_eq*1000:.2f} mm, pressure = {p_eq:.2f} MPa → FoS = {fos:.2f}")
    if u_eq > threshold_m:
        st.error(f"⚠️ Displacement exceeds threshold: {u_eq*1000:.2f} mm > {threshold_mm:.2f} mm")
else:
    st.warning("⚠️ No intersection between GRC and SCC.")
