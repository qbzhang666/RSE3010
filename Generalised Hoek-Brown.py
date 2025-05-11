import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Generalised Hoek-Brown Hybrid", layout="wide")

# --- Functions ---
def calculate_insitu_stresses(h, K, unit_weight):
    unit_weight_mpa = unit_weight / 1000
    sigma_v = unit_weight_mpa * h
    sigma_h = K * sigma_v
    if sigma_v >= sigma_h:
        return sigma_v, sigma_h, sigma_v, sigma_h, "Vertical"
    else:
        return sigma_v, sigma_h, sigma_h, sigma_v, "Horizontal"

def calculate_hb_parameters(GSI, mi, D):
    mb = mi * np.exp((GSI - 100) / (28 - 14 * D))
    s = np.exp((GSI - 100) / (9 - 3 * D))
    a = 0.5 + (1 / 6) * (np.exp(-GSI / 15) - np.exp(-20 / 3))
    return mb, s, a

def hoek_brown(sigci, mb, s, a, min_sig3, max_sig3, num_points=100):
    sig3 = np.linspace(min_sig3, max_sig3, num_points)
    term = mb * (sig3 / sigci) + s
    valid = term >= 0
    sig1 = np.where(valid, sig3 + sigci * term ** a, np.nan)
    df = pd.DataFrame({'sig3': sig3, 'sig1': sig1})
    df.dropna(inplace=True)
    df['ds1ds3'] = 1 + a * mb * (mb * (df.sig3 / sigci) + s) ** (a - 1)
    df['sign'] = ((df.sig1 + df.sig3) / 2 -
                  (df.sig1 - df.sig3) / 2 * (df.ds1ds3 - 1) / (df.ds1ds3 + 1))
    df['tau'] = ((df.sig1 - df.sig3) *
                 np.sqrt(df.ds1ds3) / (df.ds1ds3 + 1))
    return df

def fit_mohr_coulomb(df):
    x = df['sign'].values
    y = df['tau'].values
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    phi = np.degrees(np.arctan(slope))
    cohesion = intercept
    return cohesion, phi

# --- Rock Types ---
rock_type_dict = {
    "Igneous": {"Granite": 32, "Basalt": 16, "Diorite": 25},
    "Sedimentary": {"Sandstone": 17, "Shale": 6, "Limestone": 12},
    "Metamorphic": {"Gneiss": 28, "Schist": 12, "Marble": 9}
}

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")
h = st.sidebar.number_input("Tunnel Depth (m)", 10.0, 2000.0, 250.0, step=10.0)
K = st.sidebar.number_input("Horizontal Stress Ratio (K)", 0.1, 5.0, 1.5, step=0.1)
unit_weight = st.sidebar.number_input("Unit Weight (kN/m³)", 10.0, 35.0, 27.0)
GSI = st.sidebar.slider("Geological Strength Index (GSI)", 10, 100, 45)
D = st.sidebar.slider("Disturbance Factor (D)", 0.0, 1.0, 1.0, step=0.1)
sigci = st.sidebar.number_input("UCS of Intact Rock (σci) [MPa]", 5.0, 250.0, 25.0)

st.sidebar.markdown("### Rock Type Selection")
category = st.sidebar.selectbox("Rock Category", list(rock_type_dict.keys()))
rock = st.sidebar.selectbox("Rock Type", list(rock_type_dict[category].keys()))
mi = rock_type_dict[category][rock]
st.sidebar.write(f"**Selected mi value:** {mi}")

# Mohr Circle Selection
num_circles = st.sidebar.slider("Number of Mohr Circles", 1, 20, 10, step=1)

# --- Computation ---
sigma_v, sigma_h, sigma_1, sigma_3, direction = calculate_insitu_stresses(h, K, unit_weight)
mb, s, a = calculate_hb_parameters(GSI, mi, D)
df = hoek_brown(sigci, mb, s, a, 0.8 * sigma_3, 1.2 * sigma_1)
cohesion, phi_deg = fit_mohr_coulomb(df)

x_fit = np.linspace(0, df['sign'].max(), 100)
y_fit = cohesion + np.tan(np.radians(phi_deg)) * x_fit

mc_sig3 = np.linspace(0, df.sig3.max(), 100)
mc_sig1 = ((2 * cohesion * np.cos(np.radians(phi_deg))) / (1 - np.sin(np.radians(phi_deg))) +
           ((1 + np.sin(np.radians(phi_deg))) / (1 - np.sin(np.radians(phi_deg)))) * mc_sig3)

# Mohr Circle selection
circle_indices = np.linspace(0, len(df)-1, num_circles, dtype=int)
circle_data = df.iloc[circle_indices]

# --- Results Display ---
st.subheader("In-situ Stress Analysis")
st.markdown(f"""
- **Unit weight:** {unit_weight} kN/m³  
- **Vertical stress (σ_v):** {sigma_v:.2f} MPa  
- **Horizontal stress (σ_h):** {sigma_h:.2f} MPa  
- **σ₁:** {sigma_1:.2f} MPa ({direction})  
- **σ₃:** {sigma_3:.2f} MPa  
""")

st.subheader("Hoek-Brown Parameters")
st.markdown(f"**mb:** {mb:.4f}, **s:** {s:.4f}, **a:** {a:.4f}")

st.subheader("Mohr-Coulomb Parameters")
st.markdown(f"**Cohesion (c):** {cohesion:.2f} MPa, **Friction angle (φ):** {phi_deg:.2f}°")

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Hoek-Brown & Mohr-Coulomb Envelopes", fontsize=16)

# Plot σ₁–σ₃
ax1.plot(df.sig3, df.sig1, 'b-', lw=2, label=r'Hoek-Brown: $\sigma_1 = \sigma_3 + \sigma_{ci}(m_b\frac{\sigma_3}{\sigma_{ci}} + s)^a$')
ax1.plot(mc_sig3, mc_sig1, 'g--', lw=2, label=r'Mohr-Coulomb: $\sigma_1 = \frac{2c\cos\phi}{1-\sin\phi} + \frac{1+\sin\phi}{1-\sin\phi}\sigma_3$')
ax1.scatter(sigma_3, sigma_1, c='r', s=100, label='In-situ Stress')
ax1.set_xlabel(r'$\sigma_3$ [MPa]')
ax1.set_ylabel(r'$\sigma_1$ [MPa]')
ax1.set_xlim(0, None)
ax1.set_ylim(0, None)
ax1.grid(True)
ax1.legend()

# Plot τ–σₙ
ax2.plot(df['sign'], df['tau'], 'r-', lw=2, label=r'Hoek-Brown: $\tau = \frac{(\sigma_1-\sigma_3)\sqrt{d\sigma_1/d\sigma_3}}{d\sigma_1/d\sigma_3+1}$')
ax2.plot(x_fit, y_fit, 'k--', lw=2, label=fr'Mohr-Coulomb: $\tau = c + \sigma_n \tan\phi$\n(c = {cohesion:.2f} MPa, φ = {phi_deg:.1f}°)')

# Mohr Circles
for _, row in circle_data.iterrows():
    center = (row.sig1 + row.sig3)/2
    radius = (row.sig1 - row.sig3)/2
    arc = Arc((center, 0), 2*radius, 2*radius, theta1=0, theta2=180, color='grey', alpha=0.4)
    ax2.add_patch(arc)

# Get limits from full Mohr circle data
circle_centers = (circle_data.sig1 + circle_data.sig3) / 2
circle_radii = (circle_data.sig1 - circle_data.sig3) / 2
x_max = (circle_centers + circle_radii).max() * 1.1
y_max = circle_radii.max() * 1.1
lim = max(x_max, y_max)

ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)

ax2.set_aspect('equal')
ax2.set_xlabel(r'$\sigma_n$ [MPa]')
ax2.set_ylabel(r'$\tau$ [MPa]')
ax2.grid(True)
ax2.legend()

st.pyplot(fig)

# --- Reference Table ---
with st.expander("📘 Suggested $m_i$ Values for Rock Types (Hoek & Marinos, 2000)", expanded=False):
    st.image("mi_reference.png", caption="Suggested $m_i$ values for various rock types", use_container_width=True)

# --- Data Table ---
with st.expander("View Failure Envelope Data"):
    st.dataframe(df.reset_index(drop=True))
