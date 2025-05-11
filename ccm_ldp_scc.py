import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CCM Analysis Tool", layout="wide")
st.title("Convergence-Confinement Method (CCM) – Interactive Analysis")

# -------------------------------
# 1. Tunnel Parameters
# -------------------------------
st.sidebar.header("1. Tunnel Parameters")
r0 = st.sidebar.number_input("Tunnel Radius (m)", 1.0, 10.0, 5.0)

# -------------------------------
# 2. Rock / Soil Parameters
# -------------------------------
st.sidebar.header("2. Rock / Soil Parameters")
p0 = st.sidebar.number_input("In-situ Stress (MPa)", 1.0, 50.0, 10.0)
E = st.sidebar.number_input("Young's Modulus (MPa)", 500.0, 100000.0, 30000.0)
nu = st.sidebar.slider("Poisson's Ratio", 0.1, 0.49, 0.3)
c = st.sidebar.number_input("Cohesion (MPa)", 0.1, 10.0, 1.5)
phi_deg = st.sidebar.number_input("Friction Angle (°)", 5.0, 60.0, 30.0)

# -------------------------------
# 3. Rock Mass Failure Criterion (GRC)
# -------------------------------
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

if "Hoek-Brown" in failure_criterion:
    phi_deg = 35.0
    c = 2.0
elif "Mohr-Coulomb" in failure_criterion:
    phi_deg = 30.0
    c = 1.5

phi_rad = np.radians(phi_deg)
sin_phi = np.sin(phi_rad)
k_rock = (1 + sin_phi) / (1 - sin_phi)
sigma_cm_MC = (2 * c * np.cos(phi_rad)) / (1 - sin_phi)
p_cr = (2 * p0 - sigma_cm_MC) / (1 + k_rock)
G = E / (2 * (1 + nu))
u_ie = (p0 - p_cr) * r0 / (2 * G)

p = np.linspace(0.1, p0, 500)
u_r = np.zeros_like(p)
for i, p_i in enumerate(p):
    if p_i >= p_cr:
        u_r[i] = (p0 - p_i) * r0 / (2 * G)
    else:
        exponent = (k_rock - 1) / 2
        u_elastic = (p0 - p_cr) * r0 / (2 * G)
        u_r[i] = u_elastic * (p_cr / p_i) ** exponent

# -------------------------------
# 4. LDP
# -------------------------------
st.sidebar.header("4. LDP Curve Selection")
ldp_model = st.sidebar.selectbox("Select LDP Model", ["Vlachopoulos", "Hoek", "Panet"])
alpha = st.sidebar.slider("Alpha (α): deformation behind face", 0.6, 0.95, 0.85)
R_star = st.sidebar.slider("Plastic Radius R*", 1.0, 5.0, 2.5)

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
u_max = np.max(u_r)
u_ldp_actual = ldp_y * u_max

# -------------------------------
# 5. SCC Support System
# -------------------------------
st.sidebar.header("5. Support System & SCC")
install_criteria = st.sidebar.selectbox("LDP Support Criteria", [
    "Distance from face",
    "When Tunnel Wall Displacement = uₛ₀",
    "Convergence %"
])

k_supp = st.sidebar.number_input("Support Stiffness k (MPa/m)", 100, 2000, 650)
p_max = st.sidebar.number_input("Max Support Pressure pₛₘ (MPa)", 0.5, 10.0, 3.0)
diameter = 2 * r0

if install_criteria == "Distance from face":
    x_install = st.sidebar.slider("Support Distance x/r₀", 0.0, 10.0, 1.5)
    u_install = np.interp(x_install, ldp_x, u_ldp_actual)
elif install_criteria == "When Tunnel Wall Displacement = uₛ₀":
    u_install = st.sidebar.number_input("uₛ₀ (mm)", 0.0, 200.0, 30.0) / 1000
elif install_criteria == "Convergence %":
    conv_pct = st.sidebar.slider("Convergence (%)", 0.1, 10.0, 1.0)
    u_install = (conv_pct / 100) * diameter

u_scc = np.linspace(0, np.max(u_r) * 1.2, 500)
def calculate_SCC(u_vals, k, u0, p_max):
    scc = np.zeros_like(u_vals)
    for i, u in enumerate(u_vals):
        if u >= u0:
            scc[i] = min(k * (u - u0), p_max)
    return scc

scc_vals = calculate_SCC(u_scc, k_supp, u_install, p_max)
scc_on_grc = calculate_SCC(u_r, k_supp, u_install, p_max)

# -------------------------------
# Find Intersection and FoS
# -------------------------------
def find_intersection(u_vals, grc_vals, scc_vals):
    for i in range(1, len(u_vals)):
        if (grc_vals[i] - scc_vals[i]) * (grc_vals[i-1] - scc_vals[i-1]) < 0:
            u_eq = np.interp(0, [scc_vals[i-1] - grc_vals[i-1], scc_vals[i] - grc_vals[i]], [u_vals[i-1], u_vals[i]])
            p_eq = np.interp(u_eq, u_vals, grc_vals)
            return u_eq, p_eq
    return None, None

u_eq, p_eq = find_intersection(u_r, p, scc_on_grc)
fos = p_max / p_eq if p_eq and p_eq > 0 else float("inf")

# -------------------------------
# Plot GRC + SCC
# -------------------------------
threshold = st.sidebar.slider("Display threshold line at (mm)", 10, 100, 30)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(u_r * 1000, p, label="GRC", lw=2)
ax.plot(u_scc * 1000, scc_vals, linestyle='--', lw=2, color='orange', label="SCC")
ax.axvline(threshold, linestyle=':', color='red', label=f"Threshold = {threshold} mm")

if u_eq is not None and p_eq is not None:
    ax.legend(title=f"FoS = pₛₘ / pₑq = {p_max:.2f} / {p_eq:.2f} = {fos:.2f}")
else:
    ax.legend()

ax.set_xlabel("Tunnel Wall Displacement [mm]", fontsize=14)
ax.set_ylabel("Radial Stress [MPa]", fontsize=14)
ax.set_title("GRC + SCC Interaction", fontsize=16)
ax.grid(True)
st.pyplot(fig)

# -------------------------------
# Results
# -------------------------------
st.markdown("### Summary")
st.write(f"- Rock Mass Criterion: **{failure_criterion}**")
st.write(f"- Critical Pressure $p_{{cr}}$: **{p_cr:.2f} MPa**")
st.write(f"- Installation Displacement $u_{{install}}$: **{u_install*1000:.2f} mm**")
if u_eq and p_eq:
    st.success(f"✅ GRC and SCC intersect at displacement = {u_eq*1000:.2f} mm, pressure = {p_eq:.2f} MPa → FoS = {fos:.2f}")
else:
    st.warning("⚠️ No intersection found – support system insufficient")
