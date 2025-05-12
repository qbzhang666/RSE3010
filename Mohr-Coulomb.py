import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from sklearn.linear_model import RANSACRegressor, LinearRegression, TheilSenRegressor
from scipy.optimize import curve_fit

st.set_page_config(page_title="Mohr-Coulomb Lab", layout="wide")

# --- Enhanced Functions ---
def calculate_insitu_stresses(h, K, unit_weight):
    unit_weight_mpa = unit_weight / 1000
    sigma_v = unit_weight_mpa * h
    sigma_h = K * sigma_v
    principal_stresses = sorted([sigma_v, sigma_h], reverse=True)
    direction = "Vertical" if sigma_v >= sigma_h else "Horizontal"
    return principal_stresses + [direction]

def mohr_coulomb_envelope(sigma_n, c, phi):
    return c + sigma_n * np.tan(np.radians(phi))

def fit_mohr_coulomb(sigma1_exp, sigma3_exp, method='linear', threshold=1.0):
    sigma_n = (sigma1_exp + sigma3_exp) / 2
    tau = (sigma1_exp - sigma3_exp) / 2
    X = sigma_n.reshape(-1, 1)
    y = tau

    models = {
        'linear': LinearRegression(),
        'ransac': RANSACRegressor(LinearRegression(), residual_threshold=threshold),
        'theilsen': TheilSenRegressor()
    }
    
    model = models[method].fit(X, y)
    
    if hasattr(model, 'estimator_'):  # For RANSAC
        slope = model.estimator_.coef_[0]
        intercept = model.estimator_.intercept_
    else:
        slope = model.coef_[0]
        intercept = model.intercept_
    
    phi = np.degrees(np.arctan(slope))
    cohesion = intercept
    return cohesion, phi, model, sigma_n, tau

# --- Interactive Elements ---
st.sidebar.header("🛠️ Laboratory Setup")
h = st.sidebar.number_input("Tunnel Depth (m)", 10.0, 2000.0, 180.0, help="Depth below surface")
K = st.sidebar.slider("Horizontal Stress Ratio (K)", 0.1, 5.0, 2.0, 0.1)
unit_weight = st.sidebar.selectbox("Rock Unit Weight (kN/m³)", [22.0, 25.0, 27.0, 30.0], index=2)

# --- Data Input Enhancements ---
st.sidebar.header("📊 Experiment Data")
input_method = st.sidebar.radio("Data Input Method", ["Manual Entry", "File Upload", "Sample Data"])

if input_method == "Manual Entry":
    default_data = "0,5\n2,10\n4,16\n6,21"
    manual_data = st.sidebar.text_area("Enter σ₃ and σ₁ pairs (MPa):", value=default_data,
                                      help="Enter confining pressure (σ₃) and peak stress (σ₁) pairs")
    data_lines = manual_data.strip().split("\n")
elif input_method == "File Upload":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv", "xlsx"],
                                            help="File should contain 'sigma3' and 'sigma1' columns")
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        if {"sigma3", "sigma1"}.issubset(data.columns):
            data_lines = [f"{row.sigma3},{row.sigma1}" for _, row in data.iterrows()]
        else:
            st.error("❌ Required columns: 'sigma3' and 'sigma1'")
            st.stop()
else:  # Sample Data
    sample_options = {
        "Granite": [(0, 55), (5, 85), (10, 115)],
        "Sandstone": [(0, 12), (2, 20), (5, 35)],
        "Clay": [(0, 5), (1, 8), (2, 11)]
    }
    selected_sample = st.sidebar.selectbox("Choose Sample Rock", list(sample_options.keys()))
    data_lines = [f"{p[0]},{p[1]}" for p in sample_options[selected_sample]]

# Process data with validation
sigma3_list, sigma1_list = [], []
try:
    for line in data_lines:
        if line.strip():
            parts = line.split(',')
            if len(parts) == 2:
                σ3 = float(parts[0].strip())
                σ1 = float(parts[1].strip())
                if σ1 <= σ3:
                    st.error(f"⚠️ σ₁ must be greater than σ₃ in row: {line}")
                    st.stop()
                sigma3_list.append(σ3)
                sigma1_list.append(σ1)
except ValueError:
    st.error("❌ Invalid number format detected")
    st.stop()

sigma3_values = np.array(sigma3_list)
sigma1_values = np.array(sigma1_list)

# --- Analysis Parameters ---
st.sidebar.header("🔬 Analysis Settings")
fit_method = st.sidebar.selectbox("Regression Method", 
                                 ["linear", "ransac", "theilsen"],
                                 index=0,
                                 help="RANSAC is robust to outliers, Theil-Sen for small datasets")
thresh = st.sidebar.slider("Outlier Threshold (RANSAC only)", 0.1, 5.0, 1.0, 0.1)

# --- Enhanced Calculations ---
sigma_1, sigma_3, direction = calculate_insitu_stresses(h, K, unit_weight)[:3]
cohesion, friction_angle, model, sigma_n, tau = fit_mohr_coulomb(
    sigma1_values, sigma3_values, method=fit_method, threshold=thresh
)

# --- Interactive Visualization ---
st.header("🔍 Interactive Analysis Dashboard")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Principal Stress Space", "Mohr Circles", "Parameter Sensitivity"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    # MC Envelope
    mc_sig3 = np.linspace(0, max(sigma3_values)*1.2, 100)
    mc_sig1 = (2*cohesion*np.cos(np.radians(friction_angle)))/(1-np.sin(np.radians(friction_angle))) + \
              (1+np.sin(np.radians(friction_angle)))/(1-np.sin(np.radians(friction_angle))) * mc_sig3
    
    ax1.plot(mc_sig3, mc_sig1, 'g--', lw=2, alpha=0.7, 
            label='Mohr-Coulomb Failure Envelope')
    ax1.scatter(sigma3_values, sigma1_values, c='navy', s=100, 
               edgecolors='w', label='Lab Experiments')
    ax1.scatter(sigma_3, sigma_1, c='red', s=200, marker='*',
               label=f'In-situ Stress\n({direction} Dominant)')
    
    ax1.set_title("Principal Stress Space")
    ax1.set_xlabel(r'Confining Stress $\sigma_3$ (MPa)')
    ax1.set_ylabel(r'Peak Strength $\sigma_1$ (MPa)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Mohr Circles
    colors = plt.cm.plasma(np.linspace(0, 1, len(sigma3_values)))
    for idx, (σ3, σ1) in enumerate(zip(sigma3_values, sigma1_values)):
        center = (σ1 + σ3)/2
        radius = (σ1 - σ3)/2
        ax2.add_patch(Arc((center, 0), 2*radius, 2*radius, 
                         theta1=0, theta2=180, color=colors[idx], 
                         alpha=0.6, lw=2))
        ax2.text(center, radius*0.7, f'Test {idx+1}', 
                ha='center', color=colors[idx], fontsize=8)
    
    # Failure envelope
    x_fit = np.linspace(0, max(sigma_n)*1.2, 100)
    y_fit = mohr_coulomb_envelope(x_fit, cohesion, friction_angle)
    
    ax2.plot(x_fit, y_fit, 'k--', lw=2, 
            label=f'τ = {cohesion:.2f} + σ·tan({friction_angle:.1f}°)')
    ax2.scatter(sigma_n, tau, c='darkorange', s=80, 
               edgecolor='k', label='Peak Stress States')
    
    ax2.set_title("Mohr Circle Representation")
    ax2.set_xlabel(r'Normal Stress $\sigma_n$ (MPa)')
    ax2.set_ylabel(r'Shear Stress $\tau$ (MPa)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)

with tab3:
    st.subheader("Parameter Sensitivity Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        phi_sensitivity = st.slider("Friction Angle Variation (±°)", 0, 20, 5)
    with col2:
        c_sensitivity = st.slider("Cohesion Variation (±MPa)", 0.0, 5.0, 1.0)
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    # Base case
    ax3.plot(x_fit, y_fit, 'k-', lw=2, label='Best Fit')
    
    # Phi variations
    ax3.plot(x_fit, mohr_coulomb_envelope(x_fit, cohesion, friction_angle+phi_sensitivity),
            'r--', label=f'φ + {phi_sensitivity}°')
    ax3.plot(x_fit, mohr_coulomb_envelope(x_fit, cohesion, friction_angle-phi_sensitivity),
            'r:', label=f'φ - {phi_sensitivity}°')
    
    # Cohesion variations
    ax3.plot(x_fit, mohr_coulomb_envelope(x_fit, cohesion+c_sensitivity, friction_angle),
            'b--', label=f'c + {c_sensitivity} MPa')
    ax3.plot(x_fit, mohr_coulomb_envelope(x_fit, cohesion-c_sensitivity, friction_angle),
            'b:', label=f'c - {c_sensitivity} MPa')
    
    ax3.set_title("Parameter Sensitivity Study")
    ax3.legend()
    st.pyplot(fig3)

# --- Educational Content ---
with st.expander("📚 Mohr-Coulomb Theory Primer"):
    st.markdown("""
    **Fundamental Equations**
    
    - Failure criterion in shear-normal space:
    """)
    st.latex(r"\tau = c + \sigma_n \tan\phi")
    
    st.markdown("- Transformation to principal stresses:")
    st.latex(r"\sigma_1 = \frac{2c \cos\phi}{1 - \sin\phi} + \frac{1 + \sin\phi}{1 - \sin\phi} \sigma_3")
    
    st.markdown("""
    **Key Concepts**
    
    - **Cohesion (c):** Shear strength at zero normal stress
    - **Friction Angle (φ):** Rate of strength increase with normal stress
    - **Mohr Circle:** Graphical representation of stress states
    """)

# --- Data Export ---
st.sidebar.header("📤 Export Results")
if st.sidebar.button("Download Parameters Report"):
    report = f"""
    Mohr-Coulomb Analysis Report
    ---------------------------
    - In-situ Conditions:
      Depth: {h} m
      K0: {K}
      Unit Weight: {unit_weight} kN/m³
      σ1: {sigma_1:.2f} MPa ({direction})
      σ3: {sigma_3:.2f} MPa
    
    - Laboratory Results:
      Cohesion (c): {cohesion:.2f} MPa
      Friction Angle (φ): {friction_angle:.2f}°
      Regression Method: {fit_method.upper()}
    """
    st.sidebar.download_button("Download Report", report, file_name="mohr_coulomb_report.txt")

# --- Error Handling & Validation ---
if len(sigma3_values) < 2:
    st.warning("⚠️ At least 2 data points required for regression analysis")
    st.stop()

if any(sigma1_values - sigma3_values <= 0):
    st.error("❌ All σ₁ values must be greater than σ₃")
    st.stop()
