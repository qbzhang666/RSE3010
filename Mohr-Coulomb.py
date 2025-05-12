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
tensile_cutoff_ratio = st.sidebar.number_input("Tensile Cut-off Ratio (fraction of σ_c)", min_value=0.0, value=0.1)

# Manual Experimental Data
st.sidebar.markdown("### Manual Input of Experimental Data")
manual_data = st.sidebar.text_area("Enter σ₃ and σ₁ pairs (comma separated, one pair per line):", value="0,5\n2,10\n4,16\n6,21\n7,25")

# Parse input data
data_lines = manual_data.strip().split("\n")
sigma3_list, sigma1_list = [], []
try:
    for line in data_lines:
        parts = line.split(',')
        if len(parts) == 2:
            sigma3_list.append(float(parts[0]))
            sigma1_list.append(float(parts[1]))
except:
    st.error("Invalid format. Please enter numeric σ₃ and σ₁ pairs, separated by a comma.")
    st.stop()

# File Upload
st.sidebar.markdown("### Upload Experimental Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file with σ₃ and σ₁ columns", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if "sigma3" in data.columns and "sigma1" in data.columns:
        sigma3_values = data["sigma3"].values
        sigma1_values = data["sigma1"].values
    else:
        st.error("CSV must contain 'sigma3' and 'sigma1' columns.")
        st.stop()
else:
    sigma3_values = np.array(sigma3_list)
    sigma1_values = np.array(sigma1_list)

# --- Computation ---
sigma_v, sigma_h, sigma_1, sigma_3, direction = calculate_insitu_stresses(h, K, unit_weight)
cohesion, friction_angle = fit_mohr_coulomb_tangent(sigma1_values, sigma3_values)

# Tensile strength calculation
sig_c = (2 * cohesion * np.cos(np.radians(friction_angle))) / (1 - np.sin(np.radians(friction_angle)))
sig_t = tensile_strength(cohesion, friction_angle)

# Mohr-Coulomb line with and without cutoff
x_fit_original = np.linspace(-sig_c, max(sigma3_values) * 1.2, 100)
y_fit_original = cohesion + np.tan(np.radians(friction_angle)) * x_fit_original

if apply_tensile_cutoff:
    tensile_cutoff = tensile_cutoff_ratio * sig_c
    sig_t_cutoff = -tensile_cutoff
else:
    tensile_cutoff = sig_t
    sig_t_cutoff = -sig_t

x_fit_cutoff = np.linspace(sig_t_cutoff, max(sigma3_values) * 1.2, 100)
y_fit_cutoff = cohesion + np.tan(np.radians(friction_angle)) * x_fit_cutoff

# --- Output ---
st.subheader("In-situ Stress Analysis")
st.markdown(f"""
- **Unit weight:** {unit_weight} kN/m³  
- **Vertical stress** $\sigma_v$: {sigma_v:.2f} MPa  
- **Horizontal stress** $\sigma_h$: {sigma_h:.2f} MPa  
- **Major Principal Stress** $\sigma_1$: {sigma_1:.2f} MPa ({direction})  
- **Minor Principal Stress** $\sigma_3$: {sigma_3:.2f} MPa  
""")

st.subheader("Mohr-Coulomb Parameters")
st.markdown(f"""
- **Cohesion (c):** {cohesion:.2f} MPa  
- **Friction angle** $\phi$: {friction_angle:.2f}°  
""")

# (Same content up to the plotting section)

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Mohr-Coulomb Strength Analysis\nDepth: {h:.1f} m, K: {K}', fontsize=16)

# Principal Stress Plot
sin_phi = np.sin(np.radians(friction_angle))
term1 = (2 * cohesion * np.cos(np.radians(friction_angle))) / (1 - sin_phi)
term2 = (1 + sin_phi) / (1 - sin_phi)
mc_sig3 = np.linspace(0, max(sigma3_values) * 1.1, 100)
mc_sig1 = term1 + term2 * mc_sig3
ax1.plot(mc_sig3, mc_sig1, 'g--', lw=2, label=fr'Mohr-Coulomb: $\sigma_1 = \frac{{2c\cos\phi}}{{1-\sin\phi}} + \frac{{1+\sin\phi}}{{1-\sin\phi}}\sigma_3$')
ax1.scatter(sigma3_values, sigma1_values, c='b', s=80, label='Experimental Data')
ax1.scatter(sigma_3, sigma_1, c='r', s=120, label='In-situ Stress', marker='*')
ax1.set_xlabel(r'Minor Principal Stress ($\sigma_3$) [MPa]')
ax1.set_ylabel(r'Major Principal Stress ($\sigma_1$) [MPa]')
ax1.grid(True)
ax1.legend()

# Shear-Normal Plot
mc_label = fr"Mohr-Coulomb: $\tau = c + \sigma_n \tan\phi$\n$(c = {cohesion:.2f}\ \mathrm{{MPa}},\ \phi = {friction_angle:.1f}^\circ)$"
cutoff_label = fr"Tensile Cut-off: $\sigma_t = {tensile_cutoff_ratio:.2f} \times \sigma_c = {tensile_cutoff:.2f}\ \mathrm{{MPa}}$"

# Mohr-Coulomb (black) extended to horizontal axis
x_mc_extended = np.linspace(-cohesion / np.tan(np.radians(friction_angle)), max(sigma3_values) * 1.2, 200)
y_mc_extended = cohesion + np.tan(np.radians(friction_angle)) * x_mc_extended
ax2.plot(x_mc_extended, y_mc_extended, 'k--', lw=2, label=mc_label)

# Cut-off line shown only from tensile_cutoff to zero
x_cutoff = np.linspace(sig_t_cutoff, 0, 100)
y_cutoff = cohesion + np.tan(np.radians(friction_angle)) * x_cutoff
ax2.plot(x_cutoff, y_cutoff, 'r-', lw=2.5, label=cutoff_label)

# Vertical line at cut-off
ax2.vlines(x=sig_t_cutoff, ymin=0, ymax=cohesion + np.tan(np.radians(friction_angle)) * sig_t_cutoff,
           colors='red', linestyles='-', lw=2)

# Experimental Mohr circles
colors = plt.cm.viridis(np.linspace(0, 1, len(sigma3_values)))
for σ3, σ1, color in zip(sigma3_values, sigma1_values, colors):
    center = (σ1 + σ3) / 2
    radius = (σ1 - σ3) / 2
    if radius > 0:
        arc = Arc(xy=(center, 0), width=2*radius, height=2*radius, angle=0, theta1=0, theta2=180, color=color, alpha=0.6)
        ax2.add_patch(arc)

max_limit = max((sigma1_values + sigma3_values)/2 + (sigma1_values - sigma3_values)/2) * 1.1
left_limit = min(-cohesion / np.tan(np.radians(friction_angle)) * 1.1, -1.5)
ax2.set_xlim(left_limit, max_limit)
ax2.set_ylim(0, max_limit)
ax2.set_aspect('equal')
ax2.set_xlabel(r'Normal Stress ($\sigma_n$) [MPa]')
ax2.set_ylabel(r'Shear Stress ($\tau$) [MPa]')
ax2.grid(True)
ax2.legend()

st.pyplot(fig)
