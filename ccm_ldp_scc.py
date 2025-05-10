# Streamlit App: CCM with LDP Model Selection
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CCM Analysis Tool", layout="wide")
st.title("Convergence-Confinement Method (CCM) – Interactive Analysis")

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
# View Selection
# --------------------------
view_option = st.radio("Select Analysis View:", ["Ground Reaction Curve (GRC)",
                                                  "Support Characteristic Curve (SCC)",
                                                  "Longitudinal Deformation Profile (LDP)"])

# --------------------------
# Calculations
# --------------------------
phi_rad = np.radians(phi_deg)
sin_phi = np.sin(phi_rad)
k_rock = (1 + sin_phi) / (1 - sin_phi)
sigma_cm_MC = (2 * c * np.cos(phi_rad)) / (1 - sin_phi)
p_cr = (2 * p0 - sigma_cm_MC) / (1 + k_rock)
G = E / (2 * (1 + nu))
u_ie = (p0 - p_cr) * r0 / (2 * G)

# GRC
p = np.linspace(0.1, p0, 500)
u_r = np.zeros_like(p)
for i, p_i in enumerate(p):
    if p_i >= p_cr:
        u_r[i] = (p0 - p_i) * r0 / (2 * G)
    else:
        exponent = (k_rock - 1) / 2
        u_elastic = (p0 - p_cr) * r0 / (2 * G)
        u_r[i] = u_elastic * (p_cr / p_i) ** exponent

# SCC

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

# LDP Models

def ldp_profile(x_star, model="Vlachopoulos", alpha=0.85, R_star=2.5):
    x_star = np.array(x_star)
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
    else:
        raise ValueError("Unsupported LDP model. Choose from 'Panet', 'Hoek', 'Vlachopoulos'.")

# --------------------------
# Output: GRC
# --------------------------
if view_option == "Ground Reaction Curve (GRC)":
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(u_r * 1000, p, 'b-', lw=2)
    ax.set_xlabel("Radial Displacement [mm]", fontsize=14)
    ax.set_ylabel("Radial Stress [MPa]", fontsize=14)
    ax.set_title("Ground Reaction Curve (GRC)", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# --------------------------
# Output: SCC
# --------------------------
elif view_option == "Support Characteristic Curve (SCC)":
    scc = calculate_scc(u_r)
    u_int, p_int = find_intersection(u_r, p, u_r, scc)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(u_r * 1000, p, label="GRC", lw=2)
    ax.plot(u_r * 1000, scc, label="SCC", linestyle='--', lw=2)
    if u_int is not None:
        ax.plot(u_int * 1000, p_int, 'ro', label=f"Intersection\nFoS = {p_max / p_int:.2f}")
    ax.set_xlabel("Radial Displacement [mm]", fontsize=14)
    ax.set_ylabel("Radial Stress [MPa]", fontsize=14)
    ax.set_title("GRC + SCC Interaction", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    st.pyplot(fig)

    st.markdown("### SCC Result Summary")
    st.write(f"- Critical Pressure $p_{{cr}}$ = **{p_cr:.2f} MPa**")
    st.write(f"- Elastic Limit Displacement $u^{{ie}}$ = **{u_ie*1000:.2f} mm**")
    if u_int is not None:
        st.success(f"✅ Intersection at {u_int*1000:.2f} mm, {p_int:.2f} MPa → FoS = **{p_max/p_int:.2f}**")
    else:
        st.error("❌ No intersection — support is insufficient.")

# --------------------------
# Output: LDP
# --------------------------
elif view_option == "Longitudinal Deformation Profile (LDP)":
    st.subheader("Select LDP Equation")
    model = st.selectbox("LDP Model", ["Vlachopoulos", "Hoek", "Panet"])
    alpha = st.slider("Alpha (α) – deformation behind face", 0.6, 0.95, 0.85)
    R_star = st.slider("R* (Plastic Radius)", 1.0, 5.0, 2.5)

    ldp_x = np.linspace(0, 10, 500)
    u_star = ldp_profile(ldp_x, model=model, alpha=alpha, R_star=R_star)
    u_max = np.max(u_r)
    u_actual = u_star * u_max

    u_target = u_install_mm / 1000
    u_ratio = u_target / u_max
    if u_ratio >= 1.0:
        x_support = None
    else:
        x_support = -np.log(1 - u_ratio) / 1.5

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ldp_x, u_actual * 1000, lw=2, label=f"LDP: {model}")
    if x_support is not None:
        ax.axvline(x_support, color='r', linestyle='--',
                   label=fr'Suggested Support at $x/r_0$ = {x_support:.2f}')
    ax.set_xlabel("Distance from Tunnel Face $x/r_0$", fontsize=14)
    ax.set_ylabel("Radial Displacement [mm]", fontsize=14)
    ax.set_title("Longitudinal Deformation Profile (LDP)", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"### {model} LDP Support Recommendation")
    if x_support is not None:
        st.success(f"📌 Install support at **{x_support:.2f} × r₀ = {x_support * r0:.2f} m** from tunnel face")
    else:
        st.warning("⚠️ Installation displacement exceeds predicted max deformation. No LDP suggestion.")
