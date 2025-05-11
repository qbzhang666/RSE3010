# Streamlit App: CCM with In-Situ Stress from Tunnel Depth & Density
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CCM Analysis Tool", layout="wide")
st.title("Convergence-Confinement Method Analysis")

# Constants
GRAVITY = 9.81  # m/s²

# -------------------------------
# 1. Tunnel & Rock Parameters
# -------------------------------
with st.sidebar:
    st.header("1. Tunnel Parameters")
    r0 = st.number_input("Tunnel Radius [m]", 1.0, 10.0, 5.0)
    diameter = 2 * r0
    tunnel_depth = st.number_input("Tunnel Depth [m]", 10.0, 5000.0, 500.0)

    st.header("2. Rock Parameters")
    density = st.number_input("Rock Density [kg/m³]", 1500.0, 3500.0, 2650.0)
    p0 = (tunnel_depth * density * GRAVITY) / 1e6  # MPa
    st.metric("In-situ Stress p₀ [MPa]", f"{p0:.2f}")
    E = st.number_input("Young's Modulus E [MPa]", 500.0, 100000.0, 10000.0)
    nu = st.slider("Poisson's Ratio ν", 0.1, 0.49, 0.3)
    c = st.number_input("Cohesion c [MPa]", 0.1, 10.0, 1.5)
    phi_deg = st.number_input("Friction Angle φ [°]", 5.0, 60.0, 30.0)

# -------------------------------
# 2. GRC Computation (Mohr-Coulomb)
# -------------------------------
phi_rad = np.radians(phi_deg)
sin_phi = np.sin(phi_rad)
k_rock = (1 + sin_phi) / (1 - sin_phi)
sigma_cm = (2 * c * np.cos(phi_rad)) / (1 - sin_phi)
p_cr = (2 * p0 - sigma_cm) / (1 + k_rock)
G = E / (2 * (1 + nu))

p_grc = np.linspace(p0, 0.1, 500)
u_grc = np.zeros_like(p_grc)

for i, p in enumerate(p_grc):
    if p >= p_cr:
        u_grc[i] = (p0 - p) * r0 / (2 * G)
    else:
        exponent = (k_rock - 1) / 2
        u_elastic = (p0 - p_cr) * r0 / (2 * G)
        u_grc[i] = u_elastic * (p_cr / p) ** exponent

# -------------------------------
# 3. LDP
# -------------------------------
with st.sidebar:
    st.header("3. LDP Curve & Support Criteria")
    ldp_model = st.selectbox("LDP Model", ["Vlachopoulos", "Hoek", "Panet"])
    alpha = st.slider("Alpha α", 0.6, 0.95, 0.85)
    R_star = st.slider("Plastic Radius R*", 1.0, 5.0, 2.5)

    install_criteria = st.selectbox("Installation Criteria", [
        "Distance from face",
        "When Tunnel Wall Displacement = uₛ₀",
        "Convergence %"
    ])

    if install_criteria == "Distance from face":
        x_install = st.slider("Support Distance x/r₀", 0.0, 10.0, 1.5)
    elif install_criteria == "When Tunnel Wall Displacement = uₛ₀":
        u_install_mm = st.number_input("Target Displacement uₛ₀ [mm]", 1.0, 500.0, 30.0)
        u_install = u_install_mm / 1000
    elif install_criteria == "Convergence %":
        conv_pct = st.slider("Convergence [%]", 0.01, 1.0, 0.05)
        u_install = (conv_pct / 100) * diameter

# LDP displacement profile
ldp_x = np.linspace(-5, 10, 500)
def ldp_profile(x_star, model, alpha, R_star):
    if model == "Panet":
        return np.where(x_star <= 0, 1 - alpha * np.exp(-1.5 * x_star), np.exp(-1.5 * x_star))
    elif model == "Hoek":
        return np.where(x_star <= 0, 0.25 * np.exp(2.5 * x_star), 1 - 0.75 * np.exp(-0.5 * x_star))
    elif model == "Vlachopoulos":
        return np.where(x_star <= 0,
                        (1/3) * np.exp(2 * x_star - 0.15 * x_star / alpha),
                        1 - (1 - (1/3) * np.exp(-0.15 * x_star)) * np.exp(-3 * x_star / R_star))
ldp_y = ldp_profile(ldp_x, ldp_model, alpha, R_star)
u_max = 0.04  # 40 mm in meters
u_ldp = ldp_y * u_max

if install_criteria == "Distance from face":
    u_install = np.interp(x_install, ldp_x, u_ldp)

# -------------------------------
# 4. SCC
# -------------------------------
with st.sidebar:
    st.header("4. Support System & Threshold")
    k_supp = st.number_input("Support Stiffness k [MPa/m]", 100, 5000, 650)
    p_max = st.number_input("Max Support Pressure pₛₘ [MPa]", 0.1, 10.0, 3.0)
    disp_thresh = st.slider("Display Threshold [mm]", 10, 100, 40)

u_scc = np.linspace(0, np.max(u_grc) * 1.2, 500)
scc_vals = np.zeros_like(u_scc)
for i, u in enumerate(u_scc):
    if u >= u_install:
        scc_vals[i] = min(k_supp * (u - u_install), p_max)

scc_on_grc = np.zeros_like(u_grc)
for i, u in enumerate(u_grc):
    if u >= u_install:
        scc_on_grc[i] = min(k_supp * (u - u_install), p_max)

# Intersection
def find_intersection(u_vals, p1, p2):
    for i in range(1, len(u_vals)):
        if (p1[i] - p2[i]) * (p1[i-1] - p2[i-1]) < 0:
            u_eq = np.interp(0, [p1[i-1]-p2[i-1], p1[i]-p2[i]], [u_vals[i-1], u_vals[i]])
            p_eq = np.interp(u_eq, u_vals, p1)
            return u_eq, p_eq
    return None, None

u_eq, p_eq = find_intersection(u_grc, p_grc, scc_on_grc)
FoS = p_max / p_eq if p_eq and p_eq > 0 else float("inf")

# -------------------------------
# 5. Plotting
# -------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# GRC + SCC
ax1.plot(u_grc * 1000, p_grc, label="GRC", lw=2)
ax1.plot(u_scc * 1000, scc_vals, '--', color='orange', label="SCC", lw=2)
ax1.axvline(disp_thresh, linestyle=':', color='red', label=f"Threshold = {disp_thresh} mm")
ax1.set_xlabel("Tunnel Wall Displacement [mm]")
ax1.set_ylabel("Radial Pressure [MPa]")
ax1.set_title("GRC + SCC Interaction")
ax1.set_xlim(0, 100)  # cap to 100 mm (adjust as needed)
ax1.grid(True, alpha=0.3)

ax1.grid(True, color='lightgrey', alpha=0.4)
if u_eq:
    ax1.legend(title=f"FoS = pₛₘ / p_eq = {p_max:.2f} / {p_eq:.2f} = {FoS:.2f}")
else:
    ax1.legend()

# LDP
ax2.plot(ldp_x, u_ldp * 1000, label=f"{ldp_model} LDP", lw=2)
if install_criteria == "Distance from face":
    ax2.axvline(x_install, color='r', linestyle='--', label=f'Support @ x/r₀ = {x_install}')
ax2.set_xlabel("Normalized Distance x/r₀")
ax2.set_ylabel("Radial Displacement [mm]")
ax2.set_title("Longitudinal Deformation Profile")
ax2.grid(True, color='lightgrey', alpha=0.4)
ax2.legend()

st.pyplot(fig)

# -------------------------------
# 6. Results
# -------------------------------
st.markdown("### Safety Summary")
st.write(f"- Critical Pressure $p_{{cr}}$: **{p_cr:.2f} MPa**")
st.write(f"- Installation Displacement $u_{{install}}$: **{u_install*1000:.2f} mm**")
if u_eq:
    st.success(f"✅ GRC and SCC intersect at displacement = {u_eq*1000:.2f} mm, pressure = {p_eq:.2f} MPa → FoS = {FoS:.2f}")
else:
    st.warning("⚠️ No intersection found – support system insufficient")

# Optional Geotechnical Summary
with st.expander("📘 Geotechnical Parameters"):
    st.markdown(f"""
    - **In-situ Stress Calculation**:  
      \( p_0 = \\frac{{\\rho \cdot g \cdot z}}{{10^6}} = \\frac{{{density:.0f} \cdot 9.81 \cdot {tunnel_depth:.0f}}}{{10^6}} = {p0:.2f}\ \text{{MPa}} \)
    - **Tunnel Depth**: {tunnel_depth:.0f} m  
    - **Rock Density**: {density:.0f} kg/m³  
    """)
