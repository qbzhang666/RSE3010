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

# 3. Rock Mass Failure Criterion
st.sidebar.header("3. Rock Mass Failure Criterion (GRC)")
failure_criterion = st.sidebar.selectbox(
    "Select Rock Mass Failure Criterion",
    [
        "Mohr-Coulomb (Duncan-Fama)",
        "Hoek-Brown (Carranza-Torres)",
        "Generalized HB with dilation",
        "MC with dilation",
        "Variable criteria with softening",
        "Non-linear generalized HB"
    ]
)

# Apply defaults based on model
if "Hoek-Brown" in failure_criterion:
    phi_deg = 35.0
    c = 2.0
elif "Mohr-Coulomb" in failure_criterion:
    phi_deg = 30.0
    c = 1.5

# Derived parameters
phi_rad = np.radians(phi_deg)
sin_phi = np.sin(phi_rad)
k_rock = (1 + sin_phi) / (1 - sin_phi)
sigma_cm_MC = (2 * c * np.cos(phi_rad)) / (1 - sin_phi)
p_cr = (2 * p0 - sigma_cm_MC) / (1 + k_rock)
G = E / (2 * (1 + nu))

# Ground Reaction Curve (GRC)
p = np.linspace(0.1, p0, 500)
u_r = np.zeros_like(p)
for i, p_i in enumerate(p):
    if p_i >= p_cr:
        u_r[i] = (p0 - p_i) * r0 / (2 * G)
    else:
        exponent = (k_rock - 1) / 2
        u_elastic = (p0 - p_cr) * r0 / (2 * G)
        u_r[i] = u_elastic * (p_cr / p_i) ** exponent

# 4. SCC Support System
st.sidebar.header("4. Support System & SCC")
support_criteria = st.sidebar.selectbox("Support Criteria", [
    "When Tunnel Wall Displacement = uₛ₀",
    "Distance from Tunnel Face (L)",
    "When Convergence = ε × Diameter"
])
k = st.sidebar.number_input("Support Stiffness (MPa/m)", 100, 2000, 650)
p_max = st.sidebar.number_input("Max Support Pressure (MPa)", 0.5, 10.0, 3.0)
diameter = 2 * r0

# Support criteria input
if support_criteria == "When Tunnel Wall Displacement = uₛ₀":
    u_install = st.sidebar.number_input("Target Displacement uₛ₀ (mm)", 0.0, 500.0, 30.0) / 1000
elif support_criteria == "When Convergence = ε × Diameter":
    convergence_pct = st.sidebar.slider("Convergence (%)", 0.0, 10.0, 1.0)
    u_install = (convergence_pct / 100) * diameter
else:
    u_install = 0.03  # Default for LDP-based cases (will revise if LDP section enabled)

# SCC Function
def calculate_scc(u_values, k, u_install, p_max):
    return np.where(u_values >= u_install, np.minimum(k * (u_values - u_install), p_max), 0)

scc_vals = calculate_scc(u_r, k, u_install, p_max)

# Find intersection between GRC and SCC
def find_intersection(u_vals, grc_vals, scc_vals):
    for i in range(1, len(u_vals)):
        if (grc_vals[i] - scc_vals[i]) * (grc_vals[i-1] - scc_vals[i-1]) < 0:
            u_int = np.interp(0, [scc_vals[i-1] - grc_vals[i-1], scc_vals[i] - grc_vals[i]], [u_vals[i-1], u_vals[i]])
            p_eq = np.interp(u_int, u_vals, grc_vals)
            return u_int, p_eq
    return None, None

u_int, p_eq = find_intersection(u_r, p, scc_vals)
fos = p_max / p_eq if p_eq and p_eq > 0 else float("inf")

# Plot GRC + SCC
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(u_r * 1000, p, label="GRC", lw=2)
ax.plot(u_r * 1000, scc_vals, '--', lw=2, color='orange',
        label=f"SCC\nFoS = pₛₘ / pₑq = {p_max:.2f} / {p_eq:.2f} = {fos:.2f}")
ax.set_xlabel("Tunnel Wall Displacement [mm]", fontsize=14)
ax.set_ylabel("Radial Stress [MPa]", fontsize=14)
ax.set_title("GRC + SCC Interaction", fontsize=16)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=12)
st.pyplot(fig)

# Summary output
st.markdown("### Summary")
st.write(f"- Critical Pressure $p_{{cr}}$: **{p_cr:.2f} MPa**")
st.write(f"- Installation Displacement: **{u_install * 1000:.2f} mm**")

if u_int and p_eq:
    st.success(f"✅ GRC and SCC intersect at displacement = {u_int*1000:.2f} mm, pressure = {p_eq:.2f} MPa → FoS = {fos:.2f}")
else:
    st.warning("⚠️ No intersection between GRC and SCC (support insufficient)")
