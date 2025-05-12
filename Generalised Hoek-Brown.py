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

def hoek_brown(sigci, mb, s, a, sig3_values):
    sig1_values = sig3_values + sigci * (mb * (sig3_values / sigci) + s) ** a
    df = pd.DataFrame({'sig3': sig3_values, 'sig1': sig1_values})
    df['ds1ds3'] = 1 + a * mb * (mb * (df.sig3 / sigci) + s) ** (a - 1)
    df['sign'] = ((df.sig1 + df.sig3) / 2 - (df.sig1 - df.sig3) / 2 * (df.ds1ds3 - 1) / (df.ds1ds3 + 1))
    df['tau'] = ((df.sig1 - df.sig3) * np.sqrt(df.ds1ds3) / (df.ds1ds3 + 1))
    return df

def fit_mohr_coulomb(df):
    x = df['sign'].values
    y = df['tau'].values
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    phi = np.degrees(np.arctan(slope))
    cohesion = intercept
    return cohesion, phi

# --- Extended Rock Type Dictionary ---
rock_type_dict = {
    "Igneous": {
        "Granite": 32, "Granodiorite": 29, "Diorite": 25, "Dolerite": 16,
        "Gabbro": 27, "Norite": 22, "Peridotite": 25, "Rhyolite": 16,
        "Andesite": 25, "Basalt": 16, "Diabase": 16, "Porphyry": 20,
        "Agglomerate": 19, "Tuff": 13
    },
    "Sedimentary": {
        "Conglomerate": 4, "Breccia": 4, "Sandstone": 17, "Siltstone": 7,
        "Marl": 7, "Mudstone": 4, "Shale": 6, "Crystalline limestone": 12,
        "Sparitic limestone": 10, "Micritic limestone": 9, "Dolomite": 9,
        "Gypsum": 8, "Anhydrite": 12, "Coal": 8, "Chalk": 7
    },
    "Metamorphic": {
        "Gneiss": 28, "Schist": 12, "Phyllites": 7, "Slate": 7,
        "Migmatite": 29, "Amphibolite": 26, "Quartzite": 20,
        "Meta-sandstone": 19, "Hornfels": 19, "Marble": 9
    }
}

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")
h = st.sidebar.number_input("Tunnel Depth (m)", 10.0, 2000.0, 250.0)
K = st.sidebar.number_input("Horizontal Stress Ratio (K)", 0.1, 5.0, 1.5)
unit_weight = st.sidebar.number_input("Unit Weight (kN/m³)", 10.0, 35.0, 27.0)
GSI = st.sidebar.slider("Geological Strength Index (GSI)", 10, 100, 45)
D = st.sidebar.slider("Disturbance Factor (D)", 0.0, 1.0, 1.0, step=0.1)
sigci = st.sidebar.number_input("UCS of Intact Rock (σci) [MPa]", 5.0, 250.0, 25.0)

st.sidebar.markdown("### Rock Type Selection")
category = st.sidebar.selectbox("Rock Category", list(rock_type_dict.keys()))
rock = st.sidebar.selectbox("Rock Type", list(rock_type_dict[category].keys()))
mi = rock_type_dict[category][rock]
st.sidebar.write(f"**Selected mi value:** {mi}")

# --- Manual Input of Experimental Data ---
st.sidebar.markdown("### Manual Input of Experimental Data")
manual_data = st.sidebar.text_area("Enter σ₃ and σ₁ pairs (comma separated, one pair per line):", value="0,5\n2,10\n4,16\n6,21\n7,25")

sigma3_list, sigma1_list = [], []
data_lines = manual_data.strip().split("\n")
try:
    for line in data_lines:
        parts = line.split(',')
        if len(parts) == 2:
            sigma3_list.append(float(parts[0]))
            sigma1_list.append(float(parts[1]))
except:
    st.sidebar.error("Invalid format. Please enter numeric σ₃ and σ₁ pairs, separated by a comma.")

# --- Upload CSV Alternative ---
st.sidebar.markdown("### Upload Experimental Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file with σ₃ and σ₁ columns", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if "sigma3" in data.columns and "sigma1" in data.columns:
        sigma3_values = data["sigma3"].values
        sigma1_values = data["sigma1"].values
    else:
        st.sidebar.error("CSV must contain 'sigma3' and 'sigma1' columns.")
        st.stop()
else:
    sigma3_values = np.array(sigma3_list)
    sigma1_values = np.array(sigma1_list)

# --- In-situ Parameters and Computation ---
sigma_v, sigma_h, sigma_1, sigma_3, direction = calculate_insitu_stresses(h, K, unit_weight)
mb, s, a = calculate_hb_parameters(GSI, mi, D)
df = hoek_brown(sigci, mb, s, a, sigma3_values)
cohesion, phi_deg = fit_mohr_coulomb(df)

x_fit = np.linspace(0, df['sign'].max(), 100)
y_fit = cohesion + np.tan(np.radians(phi_deg)) * x_fit
mc_sig3 = np.linspace(0, df['sig3'].max(), 100)
mc_sig1 = ((2 * cohesion * np.cos(np.radians(phi_deg))) / (1 - np.sin(np.radians(phi_deg))) +
           ((1 + np.sin(np.radians(phi_deg))) / (1 - np.sin(np.radians(phi_deg)))) * mc_sig3)

# --- Output (Formatted like Mohr-Coulomb version) ---
st.subheader("In-situ Stress Analysis")
st.markdown(f"""
- **Unit weight:** {unit_weight:.1f} kN/m³  
- **Vertical stress** $\sigma_v$: {sigma_v:.2f} MPa  
- **Horizontal stress** $\sigma_h$: {sigma_h:.2f} MPa  
- **Major Principal Stress** $\sigma_1$: {sigma_1:.2f} MPa ({direction})  
- **Minor Principal Stress** $\sigma_3$: {sigma_3:.2f} MPa  
""")

st.subheader("Hoek-Brown Parameters (Hoek & Brown, 2002)")
st.markdown(f"""
- **mb:** {mb:.4f}  
- **s:** {s:.4f}  
- **a:** {a:.4f}  
""")

st.subheader("Mohr-Coulomb Parameters")
st.markdown(f"""
- **Cohesion** $(c)$: {cohesion:.2f} MPa  
- **Friction angle** $\phi$: {phi_deg:.2f}°  
""")
