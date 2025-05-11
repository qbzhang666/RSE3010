import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Generalised Hoek-Brown", layout="wide")

# --- Functions ---
def calculate_insitu_stresses(h, K, unit_weight):
    unit_weight_mpa = unit_weight / 1000
    sigma_v = unit_weight_mpa * h
    sigma_h = K * sigma_v
    if sigma_v >= sigma_h:
        sigma_1, sigma_3, direction = sigma_v, sigma_h, "Vertical"
    else:
        sigma_1, sigma_3, direction = sigma_h, sigma_v, "Horizontal"
    return sigma_v, sigma_h, sigma_1, sigma_3, direction

def calculate_hb_parameters(GSI, mi, D):
    """Generalised Hoek-Brown Criterion (Hoek, Carranza-Torres & Corkum, 2002)"""
    mb = mi * np.exp((GSI - 100) / (28 - 14 * D))
    s = np.exp((GSI - 100) / (9 - 3 * D))
    a = 0.5 + (1/6) * (np.exp(-GSI / 15) - np.exp(-20 / 3))
    return mb, s, a

def hoek_brown(sigci, mb, s, a, min_sig3, max_sig3, num_points=100):
    sig3 = np.linspace(min_sig3, max_sig3, num_points)
    term = mb * (sig3 / sigci) + s
    valid = term >= 0
    sig1 = np.full_like(sig3, np.nan)
    sig1[valid] = sig3[valid] + sigci * term[valid] ** a
    df = pd.DataFrame({'sig3': sig3, 'sig1': sig1})
    df.dropna(inplace=True)
    df['ds1ds3'] = 1 + a * mb * (mb * (df.sig3/sigci) + s) ** (a-1)
    df['sign'] = ((df.sig1 + df.sig3)/2 - (df.sig1 - df.sig3)/2 * (df.ds1ds3 - 1)/(df.ds1ds3 + 1))
    df['tau'] = ((df.sig1 - df.sig3) * np.sqrt(df.ds1ds3)/(df.ds1ds3 + 1))
    return df

def fit_mohr_coulomb(df):
    X = df['sign'].values.reshape(-1, 1)
    y = df['tau'].values
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    phi_deg = np.degrees(np.arctan(slope))
    cohesion = intercept
    return cohesion, phi_deg, model

# --- Rock Types Dictionary ---
rock_type_dict = {
    "Igneous": {
        "Granite": 32, "Granodiorite": 29, "Diorite": 25, "Gabbro": 27, "Norite": 22, "Peridotite": 25,
        "Dolerite": 16, "Rhyolite": 16, "Andesite": 25, "Basalt": 16, "Diabase": 16, "Porphyries": 20,
        "Agglomerate": 19, "Tuff": 13
    },
    "Sedimentary": {
        "Conglomerate": 4, "Breccia": 6, "Sandstone": 17, "Siltstone": 7, "Mudstone": 4, "Shale": 6,
        "Marl": 7, "Crystalline limestone": 12, "Sparitic limestone": 10, "Micritic limestone": 9,
        "Dolomite": 9, "Gypsum": 8, "Anhydrite": 12, "Coal": 8, "Chalk": 7
    },
    "Metamorphic": {
        "Gneiss": 28, "Schist": 12, "Phyllites": 7, "Slate": 7, "Migmatite": 29, "Amphibolite": 26,
        "Quartzite": 20, "Meta-sandstone": 19, "Hornfels": 19, "Marble": 9
    }
}

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")
h = st.sidebar.number_input("Tunnel Depth (m)", value=250.0, step=10.0, format="%.1f")
K = st.sidebar.number_input("Horizontal Stress Ratio (K)", value=1.5, step=0.1, format="%.1f")
unit_weight = st.sidebar.number_input("Unit Weight (kN/m³)", value=27.0, step=1.0, format="%.1f")
GSI = st.sidebar.slider("Geological Strength Index (GSI)", 10, 100, 45)
D = st.sidebar.slider("Disturbance Factor (D)", 0.0, 1.0, 1.0, 0.1)
sigci = st.sidebar.number_input("UCS of Intact Rock (σci) [MPa]", value=25.0, step=1.0, format="%.1f")

# --- Rock Selection for mi ---
st.sidebar.markdown("### Rock Type Selection")
rock_category = st.sidebar.selectbox("Rock Category", list(rock_type_dict.keys()))
rock_name = st.sidebar.selectbox("Rock Type", list(rock_type_dict[rock_category].keys()))
mi = rock_type_dict[rock_category][rock_name]
st.sidebar.write(f"**Selected mi value:** {mi}")

# --- Reference Table ---
with st.sidebar.expander("📘 Reference Table: Suggested $m_i$ Values"):
    st.image("mi_reference.png", caption="Hoek & Marinos (2000): Suggested mi Values", use_column_width=True)

# --- Computation ---
sigma_v, sigma_h, sigma_1, sigma_3, direction = calculate_insitu_stresses(h, K, unit_weight)
mb, s, a = calculate_hb_parameters(GSI, mi, D)
df = hoek_brown(sigci, mb, s, a, min_sig3=0.8*sigma_3, max_sig3=1.2*sigma_1)
cohesion, phi_deg, mc_model = fit_mohr_coulomb(df)
circle_indices = np.linspace(0, len(df)-1, 10, dtype=int)
circle_data = df.iloc[circle_indices]

# --- Results ---
st.subheader("In-situ Stress Analysis")
st.markdown(f"""
- **Unit weight:** {unit_weight} kN/m³  
- **Vertical stress (σ_v):** {sigma_v:.2f} MPa  
- **Horizontal stress (σ_h):** {sigma_h:.2f} MPa  
- **Major Principal Stress (σ₁):** {sigma_1:.2f} MPa ({direction})  
- **Minor Principal Stress (σ₃):** {sigma_3:.2f} MPa  
""")

st.subheader("Hoek-Brown Parameters (2002)")
st.markdown(f"""
- **mb:** {mb:.4f}  
- **s:** {s:.4f}  
- **a:** {a:.4f}  
""")

st.subheader("Mohr-Coulomb Parameters (from τ–σₙ)")
st.markdown(f"""
- **Cohesion (c):** {cohesion:.3f} MPa  
- **Friction angle (φ):** {phi_deg:.2f}°  
""")

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Hoek-Brown Failure Criterion", fontsize=16)

# Principal Stress Plot
ax1.plot(df.sig3, df.sig1, 'b-', lw=2, 
         label=r'$\sigma_1 = \sigma_3 + \sigma_{ci} (m_b \frac{\sigma_3}{\sigma_{ci}} + s)^a$')
ax1.scatter(sigma_3, sigma_1, c='r', s=80, label='In-situ Stress')
ax1.set_xlabel(r'$\sigma_3$ [MPa]')
ax1.set_ylabel(r'$\sigma_1$ [MPa]')
ax1.set_title("Principal Stress Space")
ax1.grid(True)
ax1.legend()

# Shear vs Normal Stress
ax2.plot(df.sign, df.tau, 'r-', lw=2, 
         label=r'$\tau = \frac{(\sigma_1-\sigma_3)\sqrt{d\sigma_1/d\sigma_3}}{d\sigma_1/d\sigma_3+1}$')

# Mohr-Coulomb Fit Line
x_fit = np.linspace(0, df.sign.max(), 100)
y_fit = mc_model.predict(x_fit.reshape(-1, 1))
ax2.plot(x_fit, y_fit, 'k--', lw=2, label='Mohr-Coulomb Fit')

# Add cohesion and friction angle to the legend
ax2.plot([], [], ' ', label=f"c = {cohesion:.2f} MPa, φ = {phi_deg:.1f}°")


# Calculate limits before using them
x_max = df.sign.max() * 1.1
y_max = df.tau.max() * 1.1
max_limit = max(x_max, y_max)

# Annotate cohesion and phi
ax2.text(
    0.05 * max_limit,
    0.95 * max_limit,
    f"c = {cohesion:.2f} MPa\nφ = {phi_deg:.1f}°",
    fontsize=11,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4')
)

ax2.set_xlim(0, max_limit)
ax2.set_ylim(0, max_limit)
ax2.set_aspect('equal')

for _, row in circle_data.iterrows():
    center = (row.sig1 + row.sig3) / 2
    radius = (row.sig1 - row.sig3) / 2
    if center + radius <= max_limit:
        arc = Arc((center, 0), 2*radius, 2*radius, angle=0, theta1=0, theta2=180, color='grey', alpha=0.5)
        ax2.add_patch(arc)

ax2.legend()
st.pyplot(fig)

# --- Data Output ---
with st.expander("View Failure Envelope Data"):
    st.dataframe(df.reset_index(drop=True))
