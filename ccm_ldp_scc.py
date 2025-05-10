# Streamlit App: CCM with Rock Mass Failure, GRC, LDP, SCC
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CCM Analysis Tool", layout="wide")
st.title("Convergence-Confinement Method (CCM) – Interactive Analysis")

# Sidebar Input Sections
st.sidebar.header("1. Tunnel Parameters")
r0 = st.sidebar.number_input("Tunnel Radius (m)", 1.0, 10.0, 5.0)

st.sidebar.header("2. Rock / Soil Parameters")
p0 = st.sidebar.number_input("In-situ Stress (MPa)", 1.0, 50.0, 10.0)
E = st.sidebar.number_input("Young's Modulus (MPa)", 500.0, 100000.0, 30000.0)
nu = st.sidebar.slider("Poisson's Ratio", 0.1, 0.49, 0.3)
c = st.sidebar.number_input("Cohesion (MPa)", 0.1, 10.0, 1.5)
phi_deg = st.sidebar.number_input("Friction Angle (°)", 5.0, 60.0, 30.0)

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

# Assign values or model-dependent adjustment (placeholder logic)
if "Hoek-Brown" in failure_criterion:
    phi_deg = 35.0
    c = 2.0
elif "Mohr-Coulomb" in failure_criterion:
    phi_deg = 30.0
    c = 1.5
# Add more mappings if needed

# Derived parameters
phi_rad = np.radians(phi_deg)
sin_phi = np.sin(phi_rad)
k_rock = (1 + sin_phi) / (1 - sin_phi)
sigma_cm_MC = (2 * c * np.cos(phi_rad)) / (1 - sin_phi)
p_cr = (2 * p0 - sigma_cm_MC) / (1 + k_rock)
G = E / (2 * (1 + nu))
u_ie = (p0 - p_cr) * r0 / (2 * G)

# Compute GRC
p = np.linspace(0.1, p0, 500)
u_r = np.zeros_like(p)
for i, p_i in enumerate(p):
    if p_i >= p_cr:
        u_r[i] = (p0 - p_i) * r0 / (2 * G)
    else:
        exponent = (k_rock - 1) / 2
        u_elastic = (p0 - p_cr) * r0 / (2 * G)
        u_r[i] = u_elastic * (p_cr / p_i) ** exponent

# LDP profile options
st.sidebar.header("4. LDP Curve Selection")
ldp_model = st.sidebar.selectbox("Select LDP Model", ["Vlachopoulos", "Hoek", "Panet"])
alpha = st.sidebar.slider("Alpha (α): deformation behind face", 0.6, 0.95, 0.85)
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
u_max = np.max(u_r)
u_ldp_actual = ldp_y * u_max

# SCC definition
st.sidebar.header("5. Support System & SCC")
k = st.sidebar.number_input("Support Stiffness (MPa/m)", 100, 2000, 650)
p_max = st.sidebar.number_input("Max Support Pressure (MPa)", 0.5, 10.0, 3.0)
support_pos_x = st.sidebar.slider("Select Support Distance from Tunnel Face (x/r₀)", 0.0, 10.0, 1.5)

u_install = ldp_profile(np.array([support_pos_x]), ldp_model, alpha, R_star)[0] * u_max
def calculate_scc(u_values):
    scc = np.zeros_like(u_values)
    for i, u in enumerate(u_values):
        if u >= u_install:
            scc[i] = min(k * (u - u_install), p_max)
    return scc

scc = calculate_scc(u_r)

# Find GRC-SCC intersection

def find_intersection(u_gr, p_gr, u_sc, p_sc):
    idx = np.where((p_sc[1:] >= p_gr[1:]) & (p_sc[:-1] < p_gr[:-1]))[0]
    if len(idx) == 0:
        return None, None
    i = idx[0]
    x = np.interp(0, p_sc[i:i+2] - p_gr[i:i+2], u_gr[i:i+2])
    y = np.interp(x, u_gr[i:i+2], p_gr[i:i+2])
    return x, y

u_int, p_int = find_intersection(u_r, p, u_r, scc)

# Main Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(u_r * 1000, p, label="GRC", lw=2)
ax.plot(u_r * 1000, scc, label="SCC", linestyle='--', lw=2)
if u_int is not None:
    ax.plot(u_int * 1000, p_int, 'ro', label=f"Intersection\nFoS = {p_max / p_int:.2f}")
ax.set_xlabel("Tunnel Wall Displacement [mm]", fontsize=14)
ax.set_ylabel("Radial Stress [MPa]", fontsize=14)
ax.set_title("GRC + SCC Interaction", fontsize=16)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()
st.pyplot(fig)

# LDP Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(ldp_x, u_ldp_actual * 1000, lw=2, label=f"LDP Model: {ldp_model}")
ax2.axvline(support_pos_x, color='r', linestyle='--', label=fr'Support Location $x/r_0$ = {support_pos_x}')
ax2.set_xlabel("Distance to Tunnel Face $x/r_0$", fontsize=14)
ax2.set_ylabel("Radial Displacement [mm]", fontsize=14)
ax2.set_title("Longitudinal Deformation Profile (LDP)", fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()
st.pyplot(fig2)

# Results Summary
st.markdown("### Summary")
st.write(f"- Rock Mass Criterion: **{failure_criterion}**")
st.write(f"- Critical Pressure $p_{{cr}}$: **{p_cr:.2f} MPa**")
st.write(f"- Installation Displacement from LDP: **{u_install*1000:.2f} mm**")
if u_int:
    st.success(f"✅ GRC and SCC intersect at {u_int*1000:.2f} mm, pressure = {p_int:.2f} MPa → FoS = {p_max/p_int:.2f}")
else:
    st.warning("⚠️ No intersection between GRC and SCC (support insufficient)")
