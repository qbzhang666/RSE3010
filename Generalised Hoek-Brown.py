import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares

st.set_page_config(page_title="Generalised Hoek-Brown Hybrid", layout="wide")

# --- Functions ---
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

def calculate_hoek_martin_cutoff(mi):
    return 1 / (8.62 + 0.7 * mi)
    
def calculate_deformation_modulus(GSI, D):
    Em = 100000 * ((1 - D/2) / (1 + np.exp((75 + 15 * D - GSI) / 11)))
    return Em
    
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
st.sidebar.header("Rock Mass and Disturbance Factor D")
GSI = st.sidebar.slider("Geological Strength Index (GSI)", 10, 100, 45)
D = st.sidebar.slider("Disturbance Factor (D)", 0.0, 1.0, 1.0, step=0.1)

st.sidebar.markdown("### Rock Type Selection mi")
category = st.sidebar.selectbox("Rock Category", list(rock_type_dict.keys()))
rock = st.sidebar.selectbox("Rock Type", list(rock_type_dict[category].keys()))
mi = rock_type_dict[category][rock]
st.sidebar.write(f"**Selected mi value:** {mi}")

st.sidebar.markdown("### Manual Input of Experimental Data")
sigci = st.sidebar.number_input("UCS of Intact Rock (σci) [MPa]", 5.0, 250.0, 25.0)
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

# --- Computation ---
mb, s, a = calculate_hb_parameters(GSI, mi, D)
sigma3_dense = np.linspace(0, max(sigma3_values)*1.2, 200)
df = hoek_brown(sigci, mb, s, a, sigma3_dense)
cohesion, phi_deg = fit_mohr_coulomb_tangent(sigma1_values, sigma3_values)
Em = calculate_deformation_modulus(GSI, D)

x_fit = np.linspace(0, max((sigma1_values + sigma3_values)/2), 100)
y_fit = cohesion + np.tan(np.radians(phi_deg)) * x_fit

mc_sig3 = np.linspace(0, max(sigma3_values), 100)
mc_sig1 = ((2 * cohesion * np.cos(np.radians(phi_deg))) / (1 - np.sin(np.radians(phi_deg))) +
           ((1 + np.sin(np.radians(phi_deg))) / (1 - np.sin(np.radians(phi_deg)))) * mc_sig3)


# --- Output ---
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

# --- Tensile Cut-off ---
tensile_ratio = calculate_hoek_martin_cutoff(mi)
st.subheader("Tensile Cut-off (Hoek-Martin 2014)")
st.markdown(f"""
- $\sigma_c / |\sigma_t| = 8.62 + 0.7m_i$ → $\sigma_t = \sigma_c / ({8.62 + 0.7*mi:.2f})$  
- **Tensile Cut-off Ratio**: {tensile_ratio:.3f}  
""")

st.subheader("Deformation Modulus (Hoek & Diederichs 2006)")
st.markdown(f"""
- **Deformation Modulus** $(E_m)$: {Em:.2f} MPa  
""")

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Hoek-Brown & Mohr-Coulomb Envelopes", fontsize=16)

ax1.plot(df.sig3, df.sig1, 'b-', lw=2,
         label=r'Hoek-Brown: $\sigma_1 = \sigma_3 + \sigma_{ci}(m_b \frac{\sigma_3}{\sigma_{ci}} + s)^a$')
ax1.plot(mc_sig3, mc_sig1, 'g--', lw=2,
         label=r'Mohr-Coulomb: $\sigma_1 = \frac{2c \cos\phi}{1 - \sin\phi} + \frac{1 + \sin\phi}{1 - \sin\phi} \cdot \sigma_3$')
ax1.scatter(sigma3_values, sigma1_values, c='black', label='Experimental Data', zorder=10)
ax1.set_xlabel(r'$\sigma_3$ [MPa]')
ax1.set_ylabel(r'$\sigma_1$ [MPa]')
ax1.grid(True)
ax1.legend(loc="upper left", fontsize=9)

ax2.plot(df['sign'], df['tau'], 'r-', lw=2,
         label=r'Hoek-Brown: $\tau = \frac{(\sigma_1 - \sigma_3) \sqrt{d\sigma_1/d\sigma_3}}{d\sigma_1/d\sigma_3 + 1}$')
ax2.plot(x_fit, y_fit, 'k--', lw=2,
         label=fr'Mohr-Coulomb: $\tau = c + \sigma_n \tan\phi$ (c = {cohesion:.2f} MPa, φ = {phi_deg:.1f}°)')

for σ3, σ1 in zip(sigma3_values, sigma1_values):
    center = (σ1 + σ3) / 2
    radius = (σ1 - σ3) / 2
    arc = Arc((center, 0), 2 * radius, 2 * radius, theta1=0, theta2=180, color='grey', alpha=0.4)
    ax2.add_patch(arc)

x_max = max((sigma3_values + sigma1_values) / 2 + (sigma1_values - sigma3_values) / 2) * 1.1
y_max = max((sigma1_values - sigma3_values) / 2) * 1.1
lim = max(x_max, y_max)

ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)
ax2.set_aspect('equal')
ax2.set_xlabel(r'$\sigma_n$ [MPa]')
ax2.set_ylabel(r'$\tau$ [MPa]')
ax2.grid(True)
ax2.legend(loc="upper left", fontsize=9)

st.pyplot(fig)


with st.expander("\U0001F4D8 Show All Equations Used"):
    st.markdown("#### Hoek-Brown and Mohr-Coulomb Strength Criteria")
    st.latex(r"\sigma_1 = \sigma_3 + \sigma_{ci} \left( m_b \frac{\sigma_3}{\sigma_{ci}} + s \right)^a")
    st.latex(r"\sigma_1 = \frac{2c \cos \phi}{1 - \sin \phi} + \frac{1 + \sin \phi}{1 - \sin \phi} \cdot \sigma_3")
    st.latex(r"\tau = \frac{(\sigma_1 - \sigma_3) \sqrt{\frac{d\sigma_1}{d\sigma_3}}}{\frac{d\sigma_1}{d\sigma_3} + 1}")
    st.latex(r"\tau = c + \sigma_n \tan \phi")

    st.markdown("#### Hoek-Brown Parameter Equations (Hoek et al., 2002)")
    st.latex(r"m_b = m_i \cdot \exp\left(\frac{{\text{GSI} - 100}}{{28 - 14D}}\right)")
    st.latex(r"s = \exp\left(\frac{{\text{GSI} - 100}}{{9 - 3D}}\right)")
    st.latex(r"a = 0.5 + \frac{1}{6} \left( \exp\left(-\frac{\text{GSI}}{15}\right) - \exp\left(-\frac{20}{3} \right) \right)")

    st.markdown("#### Hoek-Martin 2014 Tensile Cut-off")
    st.latex(r"\sigma_c / |\sigma_t| = 8.62 + 0.7m_i")

with st.expander("\U0001F4D8 Suggested $m_i$ Values for Rock Types (Hoek & Marinos, 2000)", expanded=False):
    st.image("mi_reference.png", caption="Suggested $m_i$ values for various rock types", use_container_width=True)

with st.expander("View Failure Envelope Data"):
    st.dataframe(df.reset_index(drop=True))
