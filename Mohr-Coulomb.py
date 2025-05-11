import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

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

def fit_mohr_coulomb(sigma1_values, sigma3_values):
    normal_stress = (sigma1_values + sigma3_values)/2
    shear_stress = (sigma1_values - sigma3_values)/2
    A = np.vstack([normal_stress, np.ones(len(normal_stress))]).T
    slope, intercept = np.linalg.lstsq(A, shear_stress, rcond=None)[0]
    phi = np.degrees(np.arctan(slope))
    cohesion = intercept
    return cohesion, phi

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")
h = st.sidebar.number_input("Tunnel Depth (m)", 10.0, 2000.0, 180.0)
K = st.sidebar.number_input("Horizontal Stress Ratio (K)", 0.1, 5.0, 2.0)
unit_weight = st.sidebar.number_input("Unit Weight (kN/m³)", 10.0, 35.0, 27.0)

# --- Computation ---
sigma_v, sigma_h, sigma_1, sigma_3, direction = calculate_insitu_stresses(h, K, unit_weight)
sigma3_values = np.linspace(0, sigma_3*2, 6)
sigma1_values = 2.5 * sigma3_values + 5
cohesion, friction_angle = fit_mohr_coulomb(sigma1_values, sigma3_values)

# Regression lines
x_fit = np.linspace(0, sigma3_values.max(), 100)
y_fit = cohesion + np.tan(np.radians(friction_angle)) * x_fit

# Mohr-Coulomb envelope in principal stress space
sin_phi = np.sin(np.radians(friction_angle))
term1 = (2 * cohesion * np.cos(np.radians(friction_angle))) / (1 - sin_phi)
term2 = (1 + sin_phi) / (1 - sin_phi)
mc_sig3 = np.linspace(0, sigma3_values.max(), 100)
mc_sig1 = term1 + term2 * mc_sig3

# --- Text Outputs ---
st.subheader("In-situ Stress Analysis")
st.markdown(f"""
- **Unit weight:** {unit_weight} kN/m³  
- **Vertical stress (\(\sigma_v\))**: {sigma_v:.2f} MPa  
- **Horizontal stress (\(\sigma_h\))**: {sigma_h:.2f} MPa  
- **Major Principal Stress (\(\sigma_1\))**: {sigma_1:.2f} MPa ({direction})  
- **Minor Principal Stress (\(\sigma_3\))**: {sigma_3:.2f} MPa  
""")

st.subheader("Mohr-Coulomb Parameters")
st.markdown(f"""
- **Cohesion (c):** {cohesion:.2f} MPa  
- **Friction angle (\(\phi\))**: {friction_angle:.2f}°  
""")

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Mohr-Coulomb Strength Analysis\nDepth: {h} m, K: {K}', fontsize=16)

# Principal Stress Plot
ax1.plot(mc_sig3, mc_sig1, 'g--', lw=2,
         label=r'Mohr-Coulomb: $\sigma_1 = \frac{2c\cos\phi}{1-\sin\phi} + \frac{1+\sin\phi}{1-\sin\phi}\sigma_3$')
ax1.scatter(sigma3_values, sigma1_values, c='b', s=80, label='Synthetic Stress States')
ax1.scatter(sigma_3, sigma_1, c='r', s=120, label='In-situ Stress', marker='*')
ax1.set_xlabel(r'Minor Principal Stress ($\sigma_3$) [MPa]')
ax1.set_ylabel(r'Major Principal Stress ($\sigma_1$) [MPa]')
ax1.grid(True)
ax1.legend()

# Shear-Normal Plot
mc_label = rf"Mohr-Coulomb: $\tau = c + \sigma_n \tan\phi$\n(c = {cohesion:.2f} MPa, $\phi$ = {friction_angle:.1f}\u00b0)"
ax2.plot(x_fit,
