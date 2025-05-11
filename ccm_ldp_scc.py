# Streamlit App: CCM with GRC, LDP, SCC – Final Version
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
u_ie = (p0 - p_cr) * r0 / (2 * G)

# 3. GRC
st.sidebar.header("3. Rock Mass Failure Criterion (GRC)")
failure_criterion = st.sidebar.selectbox("Select Rock Mass Failure Criterion", [
    "Mohr-Coulomb (Duncan-Fama)",
    "Hoek-Brown (Carranza-Torres)",
    "Generalized HB with dilation"
])
p = np.linspace(0.1, p0, 500)
u_r = np.zeros_like(p)
for i, p_i in enumerate(p):
    if p_i >= p_cr:
        u_r[i] = (p0 - p_i) * r0 / (2 * G)
    else:
        exponent = (k_rock - 1) / 2
        u_elastic = (p0 - p_cr) * r0 / (2 * G)
        u_r[i] = u_elastic * (p_cr / p_i) ** exponent

# 4. LDP
st.sidebar.header("4. LDP Model")
ldp_model = st.sidebar.selectbox("LDP Model", ["Panet", "Hoek", "Vlachopoulos"])
alpha = st.sidebar.slider("α: deformation behind face", 0.6, 0.95, 0.85)
R_star = st.sidebar.slider("Plastic Radius R*", 1.0, 5.0, 2.5)
support_pos_x = st.sidebar.slider("Support Location (x/r₀)", 0.0, 10.0, 1.5)

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
u_install = ldp_profile(np.array([support_pos_x]), ldp_model, alpha, R_star)[0] * u_max

# 5. SCC
st.sidebar.header("5. Support System")
k = st.sidebar.number_input("Support Stiffness k (MPa/m)", 100, 2000, 650)
p_sm = st.sidebar.number_input("Max Support Pressure pₛₘ (MPa)", 0.5, 10.0, 3.0)

def calculate_scc(u_vals, k, u_install, p_sm):
    return np.clip(k * (u_vals - u_install), 0, p_sm)

u_scc = np.sort(np.unique(np.append(np.linspace(0, np.max(u_r), 499), u_install)))
scc_vals = calculate_scc(u_scc, k, u_install, p_sm)
scc_on_grc = calculate_scc(u_r, k, u_install, p_sm)

# Intersection
def find_intersection(u_vals, grc_vals, scc_vals):
    for i in range(1, len(u_vals)):
        if (grc_vals[i] - scc_vals[i]) * (grc_vals[i-1] - scc_vals[i-1]) < 0:
            u_int = np.interp(0, [scc_vals[i-1] - grc_vals[i-1], scc_vals[i] - grc_vals[i]], [u_vals[i-1], u_vals[i]])
            p_eq = np.interp(u_int, u_vals, grc_vals)
            return u_int, p_eq
    return None, None

u_int, p_eq = find_intersection(u_r, p, scc_on_grc)
fos_val = p_sm / p_eq if p_eq and p_eq > 0 else None

# Plot GRC + SCC
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(u_r * 1000, p, label="GRC", lw=2)
ax.plot(u_scc * 1000, scc_vals, linestyle='--', color='orange', lw=2, label="SCC")
if u_int and p_eq:
    ax.legend(title=f"FoS = pₛₘ / pₑq = {p_sm:.2f} / {p_eq:.2f} = {fos_val:.2f}")
else:
    ax.legend(title="FoS: No intersection")
ax.set_xlabel("Tunnel Wall Displacement [mm]")
ax.set_ylabel("Radial Stress [MPa]")
ax.set_title("GRC + SCC Interaction")
ax.grid(True)
st.pyplot(fig)

# Plot LDP
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(ldp_x, u_ldp_actual * 1000, lw=2, label=f"LDP: {ldp_model}")
ax2.axvline(support_pos_x, color='r', linestyle='--', label=f"Support at x/r₀ = {support_pos_x}")
ax2.set_xlabel("Distance to Tunnel Face (x/r₀)")
ax2.set_ylabel("Radial Displacement [mm]")
ax2.set_title("Longitudinal Deformation Profile (LDP)")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# Summary
st.markdown("### Summary")
st.write(f"- Rock Mass Criterion: **{failure_criterion}**")
st.write(f"- Critical Pressure $p_{{cr}}$: **{p_cr:.2f} MPa**")
st.write(f"- Support installed at displacement: **{u_install * 1000:.2f} mm**")
if u_int and p_eq:
    st.success(f"✅ GRC and SCC intersect at displacement = {u_int*1000:.2f} mm, pressure = {p_eq:.2f} MPa → FoS = {fos_val:.2f}")
else:
    st.warning("⚠️ No intersection between GRC and SCC. Support insufficient.")
