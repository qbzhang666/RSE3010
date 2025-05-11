# Streamlit App: Advanced CCM Analysis with Corrected GRC/LDP
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
    
    if ldp_model == "Panet":
        alpha = st.slider("α [-]", 0.6, 0.95, 0.85)
    elif ldp_model == "Vlachopoulos":
        R_star = st.number_input("R* [-]", 1.0, 5.0, 2.5)
        
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
        u_install = st.number_input("Installation displacement [mm]", 1.0, 100.0, 30.0)/1000
    elif install_criteria == "Convergence %":
        conv_pct = st.slider("Convergence [%]", 0.1, 10.0, 1.0)
        u_install = (conv_pct/100) * diameter

# ========================
# 2. GRC Calculations (Corrected Pressure Array)
# ========================
def calculate_GRC():
    p = np.linspace(p0, 0.1, 500)  # Corrected: descending pressure values
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
    
    elastic_mask = p >= p_cr
    G = E/(2*(1 + nu))
    u_elastic = (p0 - p_cr)*r0/(2*G)
    
    # Elastic zone calculation
    u[elastic_mask] = (p0 - p[elastic_mask])*r0/(2*G)
    
    # Plastic zone calculation
    u[~elastic_mask] = u_elastic * (p_cr/p[~elastic_mask])**exponent
    
    return p, u, p_cr

p_grc, u_grc, p_cr = calculate_GRC()

# ========================
# 3. Corrected LDP Calculations
# ========================
def calculate_LDP():
    x = np.linspace(-3, 10, 500)
    
    if ldp_model == "Vlachopoulos":
        y = np.where(x <= 0,
                    (1/3) * np.exp(2*x - 0.15*R_star),
                    1 - (1 - (1/3)*np.exp(-0.15*R_star)) * np.exp(-3*x/R_star))
    elif ldp_model == "Panet":
        y = np.where(x <= 0, 
                    1 - alpha*np.exp(1.5*x),
                    np.exp(-1.5*x))
    elif ldp_model == "Hoek":
        y = np.where(x <= 0,
                    0.25*np.exp(2.5*x),
                    1 - 0.75*np.exp(-0.5*x))
    
    return x, y*u_grc.max()

ldp_x, ldp_u = calculate_LDP()

# ========================
# 4. Support System Calculations
# ========================
def calculate_SCC():
    # Determine installation displacement
    if install_criteria == "Distance from face":
        u_install = np.interp(x_install, ldp_x, ldp_u)
    elif install_criteria == "Convergence %":
        u_install = (conv_pct/100) * diameter
    
    # Calculate SCC values
    scc = np.zeros_like(u_grc)
    mask = u_grc >= u_install
    scc[mask] = np.minimum(k_supp*(u_grc[mask] - u_install), p_max)
    
    return scc, u_install

scc_p, u_install = calculate_SCC()

# ========================
# 5. Plotting
# ========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# GRC + SCC Plot
ax1.plot(u_grc*1000, p_grc, label='GRC', lw=2)
ax1.plot(u_grc*1000, scc_p, '--', label='SCC', lw=2)
ax1.set_xlabel("Radial Displacement [mm]", fontsize=12)
ax1.set_ylabel("Radial Pressure [MPa]", fontsize=12)
ax1.set_xlim(left=0)
ax1.grid(True, alpha=0.3)
ax1.legend()

# LDP Plot
ax2.plot(ldp_x, ldp_u*1000, label=f'{ldp_model} LDP', lw=2)
if install_criteria == "Distance from face":
    ax2.axvline(x_install, color='r', ls='--', 
               label=f'Support Installation (x/r₀ = {x_install:.1f})')
ax2.set_xlabel("Normalized Distance x/r₀", fontsize=12)
ax2.set_ylabel("Radial Displacement [mm]", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

st.pyplot(fig)

# ========================
# 6. Safety Analysis
# ========================
st.subheader("Safety Analysis")
diff = scc_p - p_grc
crossings = np.where(np.diff(np.sign(diff)))[0]

if len(crossings) > 0:
    idx = crossings[0]
    try:
        u_int = np.interp(0, 
                         [diff[idx], diff[idx+1]], 
                         [u_grc[idx], u_grc[idx+1]])
        p_eq = np.interp(u_int, u_grc, p_grc)
        fos = p_max / p_eq
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("Critical Pressure (p_cr)", f"{p_cr:.2f} MPa")
        with cols[1]:
            st.metric("Equilibrium Pressure (p_eq)", f"{p_eq:.2f} MPa")
        with cols[2]:
            st.metric("Factor of Safety (FoS)", f"{fos:.2f}")
            
        st.success(f"""Support system adequate!
                   Intersection at:
                   - Displacement: {u_int*1000:.1f} mm
                   - Pressure: {p_eq:.2f} MPa""")
    
    except IndexError:
        st.error("Intersection detection error - check input parameters")
else:
    st.error("No intersection detected - support system inadequate!")

# Model Validation Section
with st.expander("Theory References"):
    st.markdown("""
    **Vlachopoulos & Diederichs (2009) LDP Equations:**
    \[
    u^{*(X^*)} = 
    \begin{cases} 
    \frac{1}{3}e^{2X^* -0.15R^*} & X^* \leq 0 \\ 
    1 - \left[1 - \frac{1}{3}e^{-0.15R^*}\right]e^{-3X^*/R^*} & X^* > 0 
    \end{cases}
    \]
    
    **GRC Formulation:**
    - Elastic zone: $u = \\frac{(p_0 - p)r_0}{2G}$
    - Plastic zone: $u = u_{elastic} \\left(\\frac{p_{cr}}{p}\\right)^{\\frac{k-1}{2}}$
    """)
