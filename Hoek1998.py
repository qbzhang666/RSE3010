import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🛠️ Ground Reaction Curve and Plastic Zone Analysis")

# Sidebar inputs
st.sidebar.header("Input Parameters")
r0 = st.sidebar.number_input("Tunnel radius r₀ (m)", 1.0, 20.0, 5.0)
p0 = st.sidebar.number_input("In-situ stress p₀ (MPa)", 1.0, 50.0, 7.0)
c = st.sidebar.number_input("Cohesion c (MPa)", 0.1, 10.0, 1.5)
phi_deg = st.sidebar.slider("Friction angle φ (°)", 0.0, 45.0, 23.0)
E = st.sidebar.number_input("Young’s Modulus E (MPa)", 500.0, 50000.0, 1800.0)
nu = st.sidebar.slider("Poisson’s ratio ν", 0.1, 0.49, 0.3)

# Derived parameters
phi = np.radians(phi_deg)
sin_phi = np.sin(phi)
k = (1 + sin_phi) / (1 - sin_phi)
sigma_cm = (2 * c * np.cos(phi)) / (1 - sin_phi)
p_cr = (2 * p0 - sigma_cm) / (1 + k)
G = E / (2 * (1 + nu))
u_ie = (p0 - p_cr) * r0 / (2 * G)

# Calculations
p = np.linspace(1e-6, p0, 100)
u_r = np.zeros_like(p)
R_p = np.zeros_like(p)

for i, p_i in enumerate(p):
    if p_i >= p_cr:
        u_r[i] = (p0 - p_i) * r0 / (2 * G)
        R_p[i] = r0
    else:
        numerator = 2 * (p0 * (k - 1) + sigma_cm)
        denominator = (1 + k) * ((k - 1) * p_i + sigma_cm)
        R_p[i] = r0 * (numerator / denominator) ** (1 / (k - 1))
        exponent = (k - 1) / 2
        u_r[i] = u_ie * (p_cr / p_i) ** exponent

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(u_r, p, 'b-', lw=2, label="Ground Reaction Curve (GRC)")
ax1.axhline(p_cr, color='r', ls='--', label=f"Critical Pressure $p_{{cr}}$ = {p_cr:.2f} MPa")
ax1.axvline(u_ie, color='g', ls=':', lw=2, label=f"Transition ($u_{{ie}}$ = {u_ie:.3f} m)")

ax1.set_xlabel("Radial Displacement $u_i$ (m)")
ax1.set_ylabel("Support Pressure $p_i$ (MPa)")
ax1.set_ylim(0, 9)
ax1.set_xlim(0, 0.08)

ax2 = ax1.twinx()
ax2.plot(u_r, R_p, 'orange', lw=2, label="Plastic Radius $R_p$")
ax2.set_ylabel("Plastic Radius $R_p$ (m)")

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("Ground Reaction Curve and Plastic Zone", pad=20)

st.pyplot(fig)
