# Streamlit App: CCM with Dynamic Threshold Input
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CCM Analysis Tool", layout="wide")
st.title("Convergence-Confinement Method Analysis")

# 1. Tunnel Parameters
st.sidebar.header("1. Tunnel Parameters")
r0 = st.sidebar.number_input("Tunnel Radius (m)", 1.0, 10.0, 5.0)
diameter = 2 * r0

# 2. Rock Parameters
st.sidebar.header("2. Rock Parameters")
p0 = st.sidebar.number_input("In-situ Stress (MPa)", 1.0, 50.0, 10.0)
E = st.sidebar.number_input("Young's Modulus (MPa)", 500.0, 1e5, 3e4)
nu = st.sidebar.slider("Poisson's Ratio", 0.1, 0.49, 0.3)

# 3. Failure Criterion
st.sidebar.header("3. Failure Criterion")
criterion = st.sidebar.selectbox(
    "Select Criterion",
    ["Mohr-Coulomb", "Hoek-Brown"]
)

if "Mohr" in criterion:
    c = st.sidebar.number_input("Cohesion (MPa)", 0.1, 10.0, 1.5)
    phi_deg = st.sidebar.number_input("Friction Angle (°)", 5.0, 60.0, 30.0)
    phi = np.radians(phi_deg)
else:
    sigma_ci = st.sidebar.number_input("σ_ci (MPa)", 1.0, 100.0, 30.0)
    m_b = st.sidebar.number_input("m_b (-)", 0.1, 35.0, 15.0)
    s = st.sidebar.number_input("s (-)", 0.0, 1.0, 0.1)
    a = st.sidebar.number_input("a (-)", 0.3, 1.0, 0.5)

# 4. LDP Parameters
st.sidebar.header("4. LDP Parameters")
ldp_model = st.sidebar.selectbox(
    "LDP Model",
    ["Vlachopoulos", "Panet", "Hoek"]
)

if ldp_model == "Vlachopoulos":
    R_star = st.sidebar.number_input("Plastic Radius R*", 1.0, 5.0, 2.5)
elif ldp_model == "Panet":
    alpha = st.sidebar.slider("α Parameter", 0.6, 0.95, 0.85)

# 5. Support System (Dynamically Updated)
st.sidebar.header("5. Support System")
install_criteria = st.sidebar.selectbox(
    "Installation Criteria",
    ["Distance from Face", "Displacement Threshold", "Convergence %"]
)

if install_criteria == "Distance from Face":
    x_install = st.sidebar.slider("Installation Distance (x/r₀)", 0.0, 5.0, 1.0)
elif install_criteria == "Displacement Threshold":
    u_install = st.sidebar.number_input("Installation Displacement (mm)", 0.0, 500.0, 30.0) / 1000
elif install_criteria == "Convergence %":
    convergence_pct = st.sidebar.slider("Convergence Percentage (%)", 0.0, 10.0, 1.0)
    u_install = (convergence_pct / 100) * diameter

k = st.sidebar.number_input("Support Stiffness (MPa/m)", 100, 5000, 650)
p_max = st.sidebar.number_input("Max Support Pressure (MPa)", 0.1, 10.0, 3.0)

# GRC Calculations
G = E / (2 * (1 + nu))
p = np.linspace(0.1, p0, 500)
u_r = np.zeros_like(p)

if "Mohr" in criterion:
    sin_phi = np.sin(phi)
    k_rock = (1 + sin_phi) / (1 - sin_phi)
    sigma_cm = (2 * c * np.cos(phi)) / (1 - sin_phi)
    p_cr = (2 * p0 - sigma_cm) / (1 + k_rock)
else:
    sigma_cm = (sigma_ci/(m_b*s)) * (m_b*p0/sigma_ci + s)**a - sigma_ci/m_b
    p_cr = p0 - sigma_cm

# Calculate displacements
elastic_mask = p >= p_cr
u_elastic = (p0 - p_cr) * r0 / (2 * G)
u_r[elastic_mask] = (p0 - p[elastic_mask]) * r0 / (2 * G)
u_r[~elastic_mask] = u_elastic * (p_cr / p[~elastic_mask])**0.65

# LDP Calculations
def calculate_ldp():
    x = np.linspace(-3, 10, 500)
    if ldp_model == "Vlachopoulos":
        y = np.where(x <= 0,
                    (1/3) * np.exp(2*x - 0.15*R_star),
                    1 - (1 - (1/3)*np.exp(-0.15*R_star)) * np.exp(-3*x/R_star))
    elif ldp_model == "Panet":
        y = np.where(x <= 0, 
                    1 - alpha*np.exp(1.5*x),
                    np.exp(-1.5*x))
    elif ldp_model == "Hoek":
        y = np.where(x <= 0,
                    0.25*np.exp(2.5*x),
                    1 - 0.75*np.exp(-0.5*x))
    return x, y * u_r.max()

ldp_x, ldp_u = calculate_ldp()

# Support Installation Calculation
if install_criteria == "Distance from Face":
    u_install = np.interp(x_install, ldp_x, ldp_u)
elif install_criteria == "Convergence %":
    mask = ldp_u >= u_install
    if np.any(mask):
        x_install = ldp_x[np.argmax(mask)]

# SCC Calculations
scc_vals = np.clip(k * (u_r - u_install), 0, p_max)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# GRC + SCC Plot
ax1.plot(u_r*1000, p, label='GRC', lw=2)
ax1.plot(u_r*1000, scc_vals, '--', label='SCC', lw=2)
ax1.set_xlabel("Radial Displacement [mm]")
ax1.set_ylabel("Support Pressure [MPa]")
ax1.grid(True, alpha=0.3)
ax1.legend()

# LDP Plot
ax2.plot(ldp_x, ldp_u*1000, label=f'{ldp_model} LDP', lw=2)
if install_criteria != "Displacement Threshold":
    ax2.axvline(x_install, color='r', ls='--', 
               label=f'Installation @ x/r₀ = {x_install:.1f}')
ax2.set_xlabel("Normalized Distance x/r₀")
ax2.set_ylabel("Displacement [mm]")
ax2.grid(True, alpha=0.3)
ax2.legend()

st.pyplot(fig)

# Results Summary
st.subheader("Analysis Results")
cols = st.columns(3)
with cols[0]:
    st.metric("Critical Pressure", f"{p_cr:.2f} MPa")
with cols[1]:
    st.metric("Max Displacement", f"{u_r.max()*1000:.1f} mm")
with cols[2]:
    st.metric("Installation Displacement", f"{u_install*1000:.1f} mm")

# Safety Factor Calculation
diff = scc_vals - p
crossings = np.where(np.diff(np.sign(diff)))[0]
if len(crossings) > 0:
    u_int = np.interp(0, [diff[crossings[0]], [u_r[crossings[0]]])
    p_int = np.interp(u_int, u_r, p)
    st.success(f"Safety Factor: {p_max/p_int:.2f}")
else:
    st.error("No intersection - support system inadequate!")
