# Streamlit App: CCM Analysis with Corrected Hoek LDP
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CCM Analysis Tool", layout="wide")
st.title("Convergence-Confinement Method Analysis")

# ========================
# 1. Input Parameters
# ========================
with st.sidebar:
    st.header("1. Tunnel Parameters")
    r0 = st.number_input("Tunnel radius [m]", 1.0, 10.0, 5.0)
    diameter = 2 * r0
    
    st.header("2. Rock Mass Parameters")
    p0 = st.number_input("In-situ stress [MPa]", 1.0, 50.0, 10.0)
    E = st.number_input("Young's modulus [MPa]", 500.0, 1e5, 3e4)
    nu = st.slider("Poisson's ratio [-]", 0.1, 0.49, 0.3)
    
    st.header("3. Failure Criterion")
    criterion = st.selectbox("Select criterion", [
        "Mohr-Coulomb",
        "Hoek-Brown"
    ])
    
    if "Mohr" in criterion:
        c = st.number_input("Cohesion [MPa]", 0.1, 10.0, 1.5)
        phi = np.radians(st.number_input("Friction angle [°]", 5.0, 60.0, 30.0))
    else:
        sigma_ci = st.number_input("σ_ci [MPa]", 1.0, 100.0, 30.0)
        m_b = st.number_input("m_b [-]", 0.1, 35.0, 15.0)
        s = st.number_input("s [-]", 0.0, 1.0, 0.1)
        a = st.number_input("a [-]", 0.3, 1.0, 0.5)
        
    st.header("4. LDP Parameters")
    ldp_model = st.selectbox("LDP model", ["Vlachopoulos", "Panet", "Hoek"])
    
    ldp_params = {}
    if ldp_model == "Panet":
        ldp_params['alpha'] = st.slider("α [-]", 0.6, 0.95, 0.85)
    elif ldp_model == "Vlachopoulos":
        ldp_params['R_star'] = st.number_input("R* [-]", 1.0, 5.0, 2.5)
        
    st.header("5. Support System")
    k_supp = st.number_input("Support stiffness [MPa/m]", 100, 5000, 650)
    p_max = st.number_input("Max support pressure [MPa]", 0.1, 10.0, 3.0)
    install_criteria = st.selectbox("Installation criteria", [
        "Distance from face", 
        "Displacement threshold",
        "Convergence %"
    ])
    
    if install_criteria == "Distance from face":
        x_install = st.slider("Installation distance x/r₀", 0.0, 5.0, 1.0)
    elif install_criteria == "Displacement threshold":
        st.session_state.u_install = st.number_input("Installation displacement [mm]", 1.0, 500.0, 30.0)/1000
    elif install_criteria == "Convergence %":
        conv_pct = st.slider("Convergence [%]", 0.1, 10.0, 1.0)

# ========================
# 2. GRC Calculations
# ========================
def calculate_GRC():
    p = np.linspace(p0, 0.1, 1000)
    u = np.zeros_like(p)
    
    if "Mohr" in criterion:
        sin_phi = np.sin(phi)
        k = (1 + sin_phi)/(1 - sin_phi)
        sigma_cm = (2*c*np.cos(phi))/(1 - sin_phi)
        p_cr = (2*p0 - sigma_cm)/(1 + k)
        exponent = (k - 1)/2
    else:
        sigma_cm = (sigma_ci/(m_b*s)) * (m_b*p0/sigma_ci + s)**a - sigma_ci/m_b
        p_cr = p0 - sigma_cm
        exponent = 0.65
    
    G = E/(2*(1 + nu))
    u_elastic = (p0 - p_cr)*r0/(2*G)
    
    elastic_mask = p >= p_cr
    u[elastic_mask] = (p0 - p[elastic_mask])*r0/(2*G)
    u[~elastic_mask] = u_elastic * (p_cr/p[~elastic_mask])**exponent
    
    return p, u, p_cr

p_grc, u_grc, p_cr = calculate_GRC()

# ========================
# 3. Corrected LDP Calculations
# ========================
def calculate_LDP():
    x = np.linspace(-2, 6, 400)  # Standardized range for all models
    
    if ldp_model == "Vlachopoulos":
        R_star = ldp_params['R_star']
        y = np.where(x <= 0,
                    (1/3) * np.exp(2*x - 0.15*R_star),
                    1 - (1 - (1/3)*np.exp(-0.15*R_star)) * np.exp(-3*x/R_star))
    elif ldp_model == "Panet":
        alpha = ldp_params['alpha']
        y = np.where(x <= 0, 
                    1 - alpha*np.exp(1.5*x),
                    np.exp(-1.5*x))
    elif ldp_model == "Hoek":
        # Corrected Hoek implementation (no R* dependency)
        y = np.where(x <= 0,
                    0.25*np.exp(2.5*x),
                    1 - 0.75*np.exp(-0.5*x))
    
    return x, y*u_grc.max()

ldp_x, ldp_u = calculate_LDP()

# ========================
# 4. Support Calculations
# ========================
def calculate_SCC():
    if install_criteria == "Distance from face":
        u_install = np.interp(x_install, ldp_x, ldp_u)
    elif install_criteria == "Displacement threshold":
        u_install = st.session_state.u_install
    elif install_criteria == "Convergence %":
        u_install = (conv_pct/100) * diameter

    u_scc = np.linspace(0, u_grc.max(), 1000)
    scc = np.clip(k_supp * (u_scc - u_install), 0, p_max)
    return u_scc, scc, u_install

u_scc, scc_p, u_install = calculate_SCC()

# ========================
# 5. Plotting with Physical Distance
# ========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# GRC + SCC Plot
ax1.plot(u_grc*1000, p_grc, label='GRC', lw=2)
ax1.plot(u_scc*1000, scc_p, '--', label='SCC', lw=2)
ax1.set_xlabel("Radial Displacement [mm]", fontsize=12)
ax1.set_ylabel("Radial Pressure [MPa]", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# LDP Plot with Physical Distance
physical_distance = ldp_x * r0  # Convert normalized distance to meters
ax2.plot(physical_distance, ldp_u*1000, label=f'{ldp_model} LDP', lw=2)

if install_criteria == "Distance from face":
    install_pos = x_install * r0  # Convert to physical distance
    ax2.axvline(install_pos, color='r', ls='--', 
               label=f'Installation @ {install_pos:.1f}m')

ax2.set_xlabel("Distance from Tunnel Face [m]", fontsize=12)
ax2.set_ylabel("Radial Displacement [mm]", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

st.pyplot(fig)

# ========================
# 6. Safety Analysis
# ========================
st.subheader("Safety Analysis")
diff = scc_p - np.interp(u_scc, u_grc, p_grc)
crossings = np.where(np.diff(np.sign(diff)))[0]

if len(crossings) > 0:
    idx = crossings[0]
    u_int = np.interp(0, [diff[idx], diff[idx+1]], [u_scc[idx], u_scc[idx+1]])
    p_eq = np.interp(u_int, u_grc, p_grc)
    fos = p_max / p_eq
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Critical Pressure (p_cr)", f"{p_cr:.2f} MPa")
    with cols[1]:
        st.metric("Equilibrium Pressure (p_eq)", f"{p_eq:.2f} MPa")
    with cols[2]:
        st.metric("Factor of Safety (FoS)", f"{fos:.2f}")
    
    st.success(f"Intersection at {u_int*1000:.1f} mm displacement")
else:
    st.error("No intersection - support system inadequate!")

# Model Documentation
with st.expander("LDP Model Equations"):
    st.markdown("""
    **Hoek et al. (2002) LDP:**
    \[
    u^{*(X^*)} = 
    \begin{cases} 
    0.25 e^{2.5X^*} & X^* \leq 0 \\ 
    1 - 0.75 e^{-0.5X^*} & X^* > 0 
    \end{cases}
    \]
    where \( X^* = x/R_0 \) is the normalized distance from the tunnel face.
    """)
