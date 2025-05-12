# Hoek-Brown & Mohr-Coulomb with Experimental Data Input (Full Streamlit App)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Hoek-Brown & Mohr-Coulomb Fit", layout="wide")

def calculate_insitu_stresses(h, K, unit_weight):
    unit_weight_mpa = unit_weight / 1000
    sigma_v = unit_weight_mpa * h
    sigma_h = K * sigma_v
    return sigma_v, sigma_h, max(sigma_v, sigma_h), min(sigma_v, sigma_h), "Vertical" if sigma_v >= sigma_h else "Horizontal"

def calculate_hb_parameters(GSI, mi, D):
    mb = mi * np.exp((GSI - 100) / (28 - 14 * D))
    s = np.exp((GSI - 100) / (9 - 3 * D))
    a = 0.5 + (1 / 6) * (np.exp(-GSI / 15) - np.exp(-20 / 3))
    return mb, s, a

def hoek_brown(sigci, mb, s, a, sig3_exp):
    sig1_exp = sig3_exp + sigci * (mb * (sig3_exp / sigci) + s) ** a
    return sig1_exp

def fit_mohr_coulomb_exp(sigma3, sigma1):
    sigma_n = (sigma1 + sigma3) / 2
    tau = (sigma1 - sigma3) / 2
    A = np.vstack([sigma_n, np.ones_like(sigma_n)]).T
    slope, intercept = np.linalg.lstsq(A, tau, rcond=None)[0]
    phi = np.degrees(np.arctan(slope))
    return intercept, phi

# Sidebar inputs
st.sidebar.header("Input Parameters")
h = st.sidebar.number_input("Tunnel Depth (m)", 10.0, 2000.0, 250.0)
K = st.sidebar.number_input("Horizontal Stress Ratio (K)", 0.1, 5.0, 1.5)
unit_weight = st.sidebar.number_input("Unit Weight (kN/m³)", 10.0, 35.0, 27.0)
GSI = st.sidebar.slider("Geological Strength Index (GSI)", 10, 100, 45)
D = st.sidebar.slider("Disturbance Factor (D)", 0.0, 1.0, 1.0, step=0.1)
sigci = st.sidebar.number_input("UCS of Intact Rock (σci) [MPa]", 5.0, 250.0, 25.0)
mi = st.sidebar.number_input("Intact rock constant (mi)", 1.0, 100.0, 20.0)

st.sidebar.markdown("### Enter Experimental Mohr Data")
exp_data = st.sidebar.text_area("Enter σ₃, σ₁ pairs (comma separated)", "1,4\n3,10\n5,17\n7,25")
exp_lines = exp_data.strip().split('\n')
sigma3_list, sigma1_list = [], []
for line in exp_lines:
    try:
        parts = line.strip().split(',')
        sigma3_list.append(float(parts[0]))
        sigma1_list.append(float(parts[1]))
    except:
        continue

sigma3_arr = np.array(sigma3_list)
sigma1_arr = np.array(sigma1_list)

# Core calculations
mb, s, a = calculate_hb_parameters(GSI, mi, D)
sigma1_hb = hoek_brown(sigci, mb, s, a, sigma3_arr)
cohesion, phi = fit_mohr_coulomb_exp(sigma3_arr, sigma1_arr)

# Mohr circle properties
centers = (sigma1_arr + sigma3_arr) / 2
radii = (sigma1_arr - sigma3_arr) / 2
sigma_n = centers
tau_exp = radii
tau_mc_fit = cohesion + sigma_n * np.tan(np.radians(phi))
tau_hb_fit = (sigma1_hb - sigma3_arr) / 2  # Approximate τ from HB sig1-sig3

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Hoek-Brown & Mohr-Coulomb Envelopes", fontsize=16)

# Left: σ1–σ3
ax1.plot(sigma3_arr, sigma1_hb, 'b-', lw=2, label=r'Hoek-Brown: $\sigma_1 = \sigma_3 + \sigma_{ci}(m_b \frac{\sigma_3}{\sigma_{ci}} + s)^a$')
mc_sig1 = (2 * cohesion * np.cos(np.radians(phi)) / (1 - np.sin(np.radians(phi))) +
           (1 + np.sin(np.radians(phi))) / (1 - np.sin(np.radians(phi))) * sigma3_arr)
ax1.plot(sigma3_arr, mc_sig1, 'g--', lw=2, label=r'Mohr-Coulomb: $\sigma_1 = \frac{2c \cos\phi}{1 - \sin\phi} + \frac{1 + \sin\phi}{1 - \sin\phi} \cdot \sigma_3$')
ax1.scatter(sigma3_arr, sigma1_arr, c='black', marker='x', label='Experimental Data')
ax1.set_xlabel(r'$\sigma_3$ [MPa]')
ax1.set_ylabel(r'$\sigma_1$ [MPa]')
ax1.grid(True)
ax1.legend(loc="upper left")

# Right: τ–σₙ
ax2.plot(sigma_n, tau_exp, 'ro', label='Experimental τ–σₙ')
ax2.plot(sigma_n, tau_hb_fit, 'r-', lw=2, label='Hoek-Brown τ (approx)')
ax2.plot(sigma_n, tau_mc_fit, 'k--', lw=2, label=fr'Mohr-Coulomb: $\tau = c + \sigma_n \tan\phi$\n$(c = {cohesion:.2f} MPa, \phi = {phi:.1f}^\circ)$')
for c, r in zip(centers, radii):
    arc = Arc((c, 0), 2*r, 2*r, theta1=0, theta2=180, color='grey', alpha=0.3)
    ax2.add_patch(arc)
lim = max((centers + radii).max(), tau_mc_fit.max()) * 1.1
ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)
ax2.set_aspect('equal')
ax2.set_xlabel(r'$\sigma_n$ [MPa]')
ax2.set_ylabel(r'$\tau$ [MPa]')
ax2.grid(True)
ax2.legend(loc="upper left")

st.pyplot(fig)

# Equations
with st.expander("📘 Show All Equations Used"):
    st.markdown("#### Hoek-Brown and Mohr-Coulomb Strength Criteria")
    st.latex(r"\sigma_1 = \sigma_3 + \sigma_{ci} \left( m_b \frac{\sigma_3}{\sigma_{ci}} + s \right)^a")
    st.latex(r"\sigma_1 = \frac{2c \cos \phi}{1 - \sin \phi} + \frac{1 + \sin \phi}{1 - \sin \phi} \cdot \sigma_3")
    st.latex(r"\tau = c + \sigma_n \tan \phi")
    st.markdown("#### Hoek-Brown Parameters")
    st.latex(r"m_b = m_i \cdot \exp\left(\frac{\text{GSI} - 100}{28 - 14D}\right)")
    st.latex(r"s = \exp\left(\frac{\text{GSI} - 100}{9 - 3D}\right)")
    st.latex(r"a = 0.5 + \frac{1}{6} \left( \exp\left(-\frac{\text{GSI}}{15}\right) - \exp\left(-\frac{20}{3} \right) \right)")
