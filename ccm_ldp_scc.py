import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Streamlit config
st.set_page_config(page_title="CCM + LDP Tool", layout="wide")
st.title("Convergence-Confinement Method (CCM) with LDP Analysis")

# --------------------------
# Sidebar Inputs
# --------------------------
st.sidebar.header("Rock Mass & Tunnel Properties")
r0 = st.sidebar.number_input("Tunnel Radius (m)", 1.0, 10.0, 5.0)
p0 = st.sidebar.number_input("In-situ Stress (MPa)", 1.0, 20.0, 7.0)
c = st.sidebar.number_input("Cohesion (MPa)", 0.1, 10.0, 1.5)
phi_deg = st.sidebar.number_input("Friction Angle (°)", 5.0, 60.0, 23.0)
E = st.sidebar.number_input("Young's Modulus (MPa)", 500.0, 10000.0, 3000.0)
nu = st.sidebar.slider("Poisson's Ratio", 0.1, 0.49, 0.3)

st.sidebar.header("Support System")
k = st.sidebar.number_input("Support Stiffness k (MPa/m)", 100, 2000, 650)
p_max = st.sidebar.number_input("Max Support Pressure (MPa)", 0.5, 10.0, 3.0)
u_install_mm = st.sidebar.number_input("Installation Displacement (mm)", 0.0, 100.0, 30.0)

# --------------------------
# Derived parameters
# --------------------------
phi_rad = np.radians(phi_deg)
sin_phi = np.sin(phi_rad)
k_rock = (1 + sin_phi) / (1 - sin_phi)
sigma_cm_MC = (2 * c * np.cos(phi_rad)) / (1 - np.sin(phi_rad))
p_cr = (2 * p0 - sigma_cm_MC) / (1 + k_rock)
G = E / (2 * (1 + nu))
u_ie = (p0 - p_cr) * r0 / (2 * G)

# --------------------------
# Ground Reaction Curve (GRC)
# --------------------------
p = np.linspace(0.1, p0, 500)
u_r = np.zeros_like(p)
for i, p_i in enumerate(p):
    if p_i >= p_cr:
        u_r[i] = (p0 - p_i) * r0 / (2 * G)
    else:
        exponent = (k_rock - 1) / 2
        u_r_elastic_at_p_cr = (p0 - p_cr) * r0 / (2 * G)
        u_r[i] = u_r_elastic_at_p_cr * (p_cr / p_i) ** exponent

# --------------------------
# SCC and Intersection
# --------------------------
def calculate_scc(u_values):
    u_install = u_install_mm / 1000
    scc = np.zeros_like(u_values)
    for i, u in enumerate(u_values):
        if u >= u_install:
            scc[i] = min(k * (u - u_install), p_max)
    return scc

def find_intersection(u_gr, p_gr, u_sc, p_sc):
    cross_from_below = np.where((p_sc[1:] >= p_gr[1:]) & (p_sc[:-1] < p_gr[:-1]))[0]
    cross_from_above = np.where((p_sc[1:] <= p_gr[1:]) & (p_sc[:-1] > p_gr[:-1]))[0]
    crossings = np.concatenate((cross_from_below, cross_from_above))
    crossings.sort()
    if len(crossings) == 0:
        return None, None
    i = crossings[0]
    if i in cross_from_below:
        x = np.interp(0, p_sc[i:i+2] - p_gr[i:i+2], u_gr[i:i+2])
    else:
        x = np.interp(0, p_gr[i:i+2] - p_sc[i:i+2], u_gr[i:i+2])
    y = np.interp(x, u_gr[i:i+2], p_gr[i:i+2])
    return x, y

scc = calculate_scc(u_r)
u_int, p_int = find_intersection(u_r, p, u_r, scc)

# --------------------------
# LDP Calculation
# --------------------------
def ldp_profile(x, a=1.5):
    return 1 - np.exp(-a * x)

ldp_x = np.linspace(0, 10, 500)
ldp_y = ldp_profile(ldp_x)
u_max = max(u_r)
u_target = u_install_mm / 1000
u_ratio = u_target / u_max
if u_ratio >= 1.0:
    x_support = None
else:
    x_support = -np.log(1 - u_ratio) / 1.5

# --------------------------
# Plot: GRC + SCC
# --------------------------
fig1, ax1 = plt.subplots(figsize=(10, 7))
ax1.plot(u_r * 1000, p, label="GRC", lw=2)
ax1.plot(u_r * 1000, scc, label="SCC", linestyle='--', lw=2)
if u_int is not None:
    ax1.plot(u_int * 1000, p_int, 'ro', label=f"Intersection\nFoS = {p_max / p_int:.2f}")
ax1.set_xlabel("Radial Displacement [mm]", fontsize=14)
ax1.set_ylabel("Radial Stress [MPa]", fontsize=14)
ax1.set_xlim(0, 40)
ax1.set_ylim(0, p0 * 1.1)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=12)
st.pyplot(fig1)

# --------------------------
# Plot: LDP
# --------------------------
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(ldp_x, ldp_y, 'k-', lw=2, label=r'LDP: $u(x)/u_{\max}$')
if x_support is not None:
    ax2.axvline(x_support, color='r', linestyle='--', lw=2,
                label=fr'Suggested Support Location: $x/r_0$ = {x_support:.2f}')
ax2.set_xlabel(r'Normalized Distance from Face $x/r_0$', fontsize=14)
ax2.set_ylabel(r'Normalized Displacement $u(x)/u_{\max}$', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=12)
st.pyplot(fig2)

# --------------------------
# Results Summary
# --------------------------
st.markdown("### Results Summary")
st.write(f"- Critical Pressure $p_{{cr}}$ = **{p_cr:.2f} MPa**")
st.write(f"- Elastic Limit Displacement $u^{{ie}}$ = **{u_ie*1000:.2f} mm**")
if u_int is not None:
    st.success(f"✅ Intersection at {u_int*1000:.2f} mm, {p_int:.2f} MPa → **FoS = {p_max / p_int:.2f}**")
else:
    st.error("❌ No intersection — support is insufficient.")

if x_support is not None:
    st.info(f"📌 Install support at **{x_support:.2f} × r₀ = {x_support * r0:.2f} m** from tunnel face")
else:
    st.warning("⚠️ Support installation displacement exceeds predicted maximum deformation. No LDP recommendation.")
