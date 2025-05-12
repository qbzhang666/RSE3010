import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares

st.set_page_config(page_title="Mohr-Coulomb Strength Analysis", layout="wide")

# --- Functions ---
def calculate_insitu_stresses(h, K, unit_weight):
    unit_weight_mpa = unit_weight / 1000
    sigma_v = unit_weight_mpa * h
    sigma_h = K * sigma_v
    if sigma_v >= sigma_h:
        return sigma_v, sigma_h, sigma_v, sigma_h, "Vertical"
    else:
        return sigma_v, sigma_h, sigma_h, sigma_v, "Horizontal"

def fit_mohr_coulomb_tangent(sigma1_exp, sigma3_exp):
    centers = (sigma1_exp + sigma3_exp) / 2
    radii = (sigma1_exp - sigma3_exp) / 2

    def distance_residuals(params):
        c, phi_deg = params
        phi_rad = np.radians(phi_deg)
        tan_phi = np.tan(phi_rad)
        distances = []
        for sn, r in zip(centers, radii):
            tau_pred = c + sn * tan_phi
            distance = np.abs(tau_pred) - r * np.sqrt(1 + tan_phi**2)
            distances.append(distance)
        return distances

    tau = (sigma1_exp - sigma3_exp) / 2
    sn = (sigma1_exp + sigma3_exp) / 2
    lin_reg = LinearRegression().fit(sn.reshape(-1, 1), tau)
    init_c = lin_reg.intercept_
    init_phi = np.degrees(np.arctan(lin_reg.coef_[0]))

    result = least_squares(distance_residuals, [init_c, init_phi], bounds=(0, [np.inf, 90]))
    return result.x[0], result.x[1]

def tensile_strength(c, phi_deg):
    phi_rad = np.radians(phi_deg)
    return (2 * c * np.cos(phi_rad)) / (1 + np.sin(phi_rad))

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")
h = st.sidebar.number_input("Tunnel Depth (m)", 10.0, 2000.0, 180.0)
K = st.sidebar.number_input("Horizontal Stress Ratio (K)", 0.1, 5.0, 2.0)
unit_weight = st.sidebar.number_input("Unit Weight (kN/m³)", 10.0, 35.0, 27.0)

# Tensile cutoff input
st.sidebar.markdown("### Tensile Cut-off")
apply_tensile_cutoff = st.sidebar.checkbox("Apply Tensile Cut-off", value=True)
tensile_cutoff_ratio = st.sidebar.number_input("Tensile Cut-off Ratio (fraction of σ_c)", 0.05, 0.2, 0.1)

# Manual Experimental Data
st.sidebar.markdown("### Manual Input of Experimental Data")
manual_data = st.sidebar.text_area("Enter σ₃ and σ₁ pairs (comma separated, one pair per line):", value="0,5\n2,10\n4,16\n6,21\n7,25")

# Parse input data
data_lines = manual_data.strip().split("\n")
sigma3_list, sigma1_list = [], []
for line in data_lines:
    parts = line.split(',')
    sigma3_list.append(float(parts[0]))
    sigma1_list.append(float(parts[1]))

sigma3_values = np.array(sigma3_list)
sigma1_values = np.array(sigma1_list)

# --- Computation ---
sigma_v, sigma_h, sigma_1, sigma_3, direction = calculate_insitu_stresses(h, K, unit_weight)
cohesion, friction_angle = fit_mohr_coulomb_tangent(sigma1_values, sigma3_values)

# Tensile strength calculation
sig_c = (2 * cohesion * np.cos(np.radians(friction_angle))) / (1 - np.sin(np.radians(friction_angle)))
sig_t = tensile_strength(cohesion, friction_angle)

# Apply tensile cut-off if selected
if apply_tensile_cutoff:
    tensile_cutoff = tensile_cutoff_ratio * sig_c
    sig_t_cutoff = -tensile_cutoff
else:
    tensile_cutoff = sig_t
    sig_t_cutoff = -sig_t

# Original Mohr-Coulomb line without cutoff
x_fit_original = np.linspace(0, max(sigma3_values) * 1.2, 100)
y_fit_original = cohesion + np.tan(np.radians(friction_angle)) * x_fit_original

# Plotting parameters with cutoff
x_fit_cutoff = np.linspace(sig_t_cutoff, max(sigma3_values) * 1.2, 100)
y_fit_cutoff = cohesion + np.tan(np.radians(friction_angle)) * x_fit_cutoff

# --- Output ---
st.subheader("Mohr-Coulomb Parameters")
st.markdown(f"""
- **Cohesion (c):** {cohesion:.2f} MPa  
- **Friction angle** $\phi$: {friction_angle:.2f}°  
- **Uniaxial Compressive Strength** $\sigma_c$: {sig_c:.2f} MPa  
- **Calculated Tensile Strength** $\sigma_t$: {sig_t:.2f} MPa  
- **Applied Tensile Cut-off**: {tensile_cutoff:.2f} MPa
""")

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_fit_original, y_fit_original, 'k--', label='Original Mohr-Coulomb Criterion')
ax.plot(x_fit_cutoff, y_fit_cutoff, 'r-', label='Mohr-Coulomb Criterion with Tensile Cut-off')

# Experimental data plotting with safeguard
colors = plt.cm.viridis(np.linspace(0, 1, len(sigma3_values)))
for σ3, σ1, color in zip(sigma3_values, sigma1_values, colors):
    center = (σ1 + σ3) / 2
    radius = (σ1 - σ3) / 2
    if radius > 0:
        arc = Arc((center, 0), 2*radius, 2*radius, 0, 0, 180, color=color, alpha=0.6)
        ax.add_patch(arc)

# Axis adjustments
ax.set_xlim(sig_t_cutoff * 1.2, max(sigma1_values)*1.2)
ax.set_ylim(0, max(sigma1_values)*0.7)
ax.set_aspect('equal')
ax.set_xlabel(r'Normal Stress $\sigma_n$ (MPa)')
ax.set_ylabel(r'Shear Stress $\tau$ (MPa)')
ax.grid(True)
ax.legend()

st.pyplot(fig)
