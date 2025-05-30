# Streamlit App: CCM Analysis Tool (MC + Hoek-Brown)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CCM Analysis Tool", layout="wide")
st.title("Convergence-Confinement Method Analysis")

GRAVITY = 9.81  # m/s²

# -------------------------------
# 1. Tunnel & Rock Parameters
# -------------------------------
with st.sidebar:
    st.header("1. Tunnel & Rock Parameters")
    r0 = st.number_input("Tunnel Radius [m]", 1.0, 10.0, 5.0, step=0.1, format="%.1f")
    diameter = 2 * r0
    tunnel_depth = st.number_input("Tunnel Depth [m]", 10, 5000, 500, step=10, format="%d")
    density = st.number_input("Rock Density [kg/m³]", 1500, 3500, 2650, step=50, format="%d")
    p0 = (tunnel_depth * density * GRAVITY) / 1e6  # MPa
    st.metric("In-situ Stress p₀ [MPa]", f"{p0:.2f}")
    E = st.number_input("Young's Modulus E [MPa]", 500, 100000, 10000, step=500, format="%d")
    nu = st.number_input("Poisson's Ratio ν", min_value=0.10, max_value=0.49, value=0.30, step=0.01, format="%.2f")

# -------------------------------
# 2. GRC Calculation
# -------------------------------
    st.subheader("2. GRC Calculation")
    criterion = st.selectbox("Select Failure Criterion", ["Mohr-Coulomb", "Hoek-Brown"])

    if criterion == "Mohr-Coulomb":
        c = st.number_input("Cohesion c [MPa]", 0.1, 10.0, 1.5)
        phi_deg = st.number_input("Friction Angle φ [°]", 5.0, 60.0, 30.0)
        phi_rad = np.radians(phi_deg)
    else:
        sigma_ci = st.number_input("Uniaxial compressive strength σ_ci [MPa]", 1, 100, 30, step=1, format="%d")
        GSI = st.slider("Geological Strength Index (GSI)", 10, 100, 25)
        mi = st.number_input("Intact rock constant (mᵢ)", 5, 35, 15, step=1, format="%d")
        D = st.slider("Disturbance Factor (D)", 0.0, 1.0, 0.0)

        # Hoek-Brown calculations
        mb = mi * np.exp((GSI - 100) / (28 - 14 * D))
        s_val = np.exp((GSI - 100) / (9 - 3 * D))
        a_val = 0.5 + (1 / 6) * (np.exp(-GSI / 15) - np.exp(-20 / 3))

        st.markdown(f"**Calculated Parameters:**  \n"
                    f"m_b = {mb:.2f}  \n"
                    f"s = {s_val:.4f}  \n"
                    f"a = {a_val:.3f}")

def calculate_GRC():
    p = np.linspace(p0, 0.1, 500)
    u = np.zeros_like(p)
    G = E / (2 * (1 + nu))  # Shear modulus

    if criterion == "Mohr-Coulomb":
        sin_phi = np.sin(phi_rad)
        k = (1 + sin_phi) / (1 - sin_phi)
        sigma_cm = (2 * c * np.cos(phi_rad)) / (1 - sin_phi)
        p_cr = (2 * p0 - sigma_cm) / (1 + k)
        u_elastic = (p0 - p_cr) * r0 / (2 * G)

        for i, pi in enumerate(p):
            if pi >= p_cr:
                u[i] = (p0 - pi) * r0 / (2 * G)
            else:
                exponent = (k - 1) / 2
                u[i] = u_elastic * (p_cr / pi) ** exponent

    else:  # Hoek-Brown
        # Calculate strength parameters
        sigma_cm = (sigma_ci / 2) * ((mb + 4 * s_val) ** a_val - mb ** a_val)
        p_cr = p0 * (1 - (sigma_cm / (2 * p0)))  # Transition pressure
        k_HB = (2 * (1 - nu) * (mb + 4 * s_val) ** a_val) / (1 + nu)
        R_pl = r0 * ((2 * p0 / sigma_cm) + 1) ** (1 / k_HB)
        u_elastic = (p0 - p_cr) * r0 / (2 * G)

        for i, pi in enumerate(p):
            if pi >= p_cr:
                u[i] = (p0 - pi) * r0 / (2 * G)  # Same elastic stage as MC
            else:
                u[i] = u_elastic * (R_pl / r0) ** k_HB * (p_cr / pi) ** k_HB  # Plastic HB

    return p, u, p_cr


p_grc, u_grc, p_cr = calculate_GRC()

# -------------------------------
# 3. LDP & Support Criteria
# -------------------------------
with st.sidebar:
    st.header("3. LDP Curve & Support Criteria")
    ldp_model = st.selectbox("LDP Model", ["Vlachopoulos and Diederichs (2009)", "Hoek et al. (2002)", "Panet (1995)"])
    alpha = st.slider("Alpha α", 0.6, 0.95, 0.85)
    R_star = st.slider("Plastic Radius R*", 1.0, 5.0, 2.5)

    install_criteria = st.selectbox("Installation Criteria", [
        "Distance from Tunnel Face",
        "When Radial Displacement = uₛ₀",
        "Convergence %"
    ])
    if install_criteria == "Distance from Tunnel Face":
        x_install = st.slider("Support Distance x/r₀", 0.0, 10.0, 1.5)
    elif install_criteria == "When Radial Displacement = uₛ₀":
        u_install_mm = st.number_input("Target Displacement uₛ₀ [mm]", 1.0, 500.0, 30.0)
        u_install = u_install_mm / 1000
    elif install_criteria == "Convergence %":
        conv_pct = st.slider("Convergence [%]", 0.01, 1.0, 0.05)
        u_install = (conv_pct / 100) * diameter

ldp_x = np.linspace(-5, 10, 500)
def ldp_profile(x_star, model, alpha, R_star):
    if model == "Panet (1995)":
        return np.where(
            x_star <= 0,
            (1 - alpha) * np.exp(1.5 * x_star),
            1 - alpha * np.exp(-1.5 * x_star)
        )
    elif model == "Hoek et al. (2002)":
        return np.where(x_star <= 0, 0.25 * np.exp(2.5 * x_star), 1 - 0.75 * np.exp(-0.5 * x_star))
    elif model == "Vlachopoulos and Diederichs (2009)":
        return np.where(
            x_star <= 0,
            (1/3) * np.exp(2 * x_star - 0.15 * x_star / alpha),
            1 - (1 - (1/3) * np.exp(-0.15 * x_star)) * np.exp(-3 * x_star / R_star)
        )


ldp_y = ldp_profile(ldp_x, ldp_model, alpha, R_star)
u_max = 0.06  # 60 mm max displacement
u_ldp = ldp_y * u_max
if install_criteria == "Distance from Tunnel Face":
    u_install = np.interp(x_install, ldp_x, u_ldp)

# -------------------------------
# 4. SCC Curve
# -------------------------------
with st.sidebar:
    st.header("4. Support System & Threshold")
    k_supp = st.number_input("Support Stiffness k [MPa/m]", 100, 5000, 650, step=50, format="%d")
    p_max = st.number_input("Max Support Pressure pₛₘ [MPa]", 0.1, 10.0, 3.1, step=0.1, format="%.1f")
    disp_thresh = st.slider("Display Threshold [mm]", 10, 100, 40)

u_scc = np.linspace(0, np.max(u_grc) * 1.2, 500)
scc_vals = np.where(u_scc >= u_install, np.minimum(k_supp * (u_scc - u_install), p_max), 0)
scc_on_grc = np.where(u_grc >= u_install, np.minimum(k_supp * (u_grc - u_install), p_max), 0)

# -------------------------------
# 5. Intersection & FoS
# -------------------------------
def find_intersection(u_vals, p1, p2):
    for i in range(1, len(u_vals)):
        if (p1[i] - p2[i]) * (p1[i-1] - p2[i-1]) < 0:
            u_eq = np.interp(0, [p1[i-1]-p2[i-1], p1[i]-p2[i]], [u_vals[i-1], u_vals[i]])
            p_eq = np.interp(u_eq, u_vals, p1)
            return u_eq, p_eq
    return None, None

u_eq, p_eq = find_intersection(u_grc, p_grc, scc_on_grc)
p_scc_at_u_eq = np.interp(u_eq, u_scc, scc_vals) if u_eq is not None else None

# Determine FoS only if SCC hasn't reached max pressure
FoS = None
if u_eq is not None and p_eq is not None and p_eq > 0:
    if p_scc_at_u_eq is not None and not np.isclose(p_scc_at_u_eq, p_max, atol=1e-3):
        FoS = p_max / p_eq

# -------------------------------
# 6. Plotting
# -------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# GRC + SCC
ax1.plot(u_grc * 1000, p_grc, label="GRC", lw=2)
ax1.plot(u_scc * 1000, scc_vals, '--', color='orange', label="SCC", lw=2)
ax1.axvline(disp_thresh, linestyle=':', color='green', label=f"Threshold = {disp_thresh} mm")

# Mark point where SCC reaches max pressure
if u_eq and p_scc_at_u_eq and np.isclose(p_scc_at_u_eq, p_max, atol=1e-3):
    ax1.plot(u_eq * 1000, p_scc_at_u_eq, 'ro', label="Max Support Reached")

ax1.set_xlabel("Radial Displacement [mm]")
ax1.set_ylabel("Radial Pressure [MPa]")
ax1.set_title("GRC + SCC Interaction")
ax1.set_xlim(0, 100)
ax1.grid(True, color='lightgrey', alpha=0.4)

if u_eq and p_eq and FoS is not None:
    ax1.legend(title=f"FoS = pₛₘ / p_eq = {p_max:.2f} / {p_eq:.2f} = {FoS:.2f}")
else:
    ax1.legend()

# LDP
ax2.plot(ldp_x * r0, u_ldp * 1000, label=f"{ldp_model} LDP", lw=2)

# Vertical line for support installation
if install_criteria == "Distance from Tunnel Face":
    ax2.axvline(x_install * r0, color='red', linestyle='--', linewidth=2, label=f'Support @ x/r₀ = {x_install}')

# Solid line at tunnel face (x = 0)
ax2.axvline(0, color='black', linestyle='-', linewidth=1.5, label='Tunnel Face (x = 0)')

# Formatting
ax2.set_xlabel("Distance to Tunnel Face [m]")
ax2.set_ylabel("Radial Displacement [mm]")
ax2.set_title("LDP Longitudinal Deformation Profile")
ax2.set_xlim(20, -10)  # reversed x-axis range
ax2.grid(True, color='lightgrey', alpha=0.4)
ax2.legend()

st.pyplot(fig)


# -------------------------------
# 7. Results
# -------------------------------
st.markdown("### Safety Summary")
st.write(f"- Critical Pressure $p_{{cr}}$: **{p_cr:.2f} MPa**")
st.write(f"- Installation Displacement $u_{{install}}$: **{u_install*1000:.2f} mm**")

if u_eq and p_eq:
    if np.isclose(p_scc_at_u_eq, p_max, atol=1e-3):
        st.warning("⚠️ Intersection occurs after support reaches max pressure. FoS is not valid – support system may be insufficient.")
    elif FoS is not None:
        st.success(f"✅ GRC and SCC intersect at displacement = {u_eq*1000:.2f} mm, pressure = {p_eq:.2f} MPa → FoS = {FoS:.2f}")
else:
    st.warning("⚠️ No intersection found – support system insufficient.")

# Optional Geotechnical Summary
with st.expander("📘 Geotechnical Parameters & Equations"):
    st.markdown("**1. In-situ Stress**")
    st.latex(r"p_0 = \frac{\rho \cdot g \cdot z}{10^6} \quad [\text{MPa}]")

    st.markdown("**2. Mohr-Coulomb Parameters**")
    st.latex(r"\sigma_{cm} = \frac{2c \cdot \cos(\phi)}{1 - \sin(\phi)}")
    st.latex(r"k = \frac{1 + \sin(\phi)}{1 - \sin(\phi)}")
    st.latex(r"p_{cr} = \frac{2p_0 - \sigma_{cm}}{1 + k}")
    st.latex(r"u_{\text{elastic}} = \frac{(p_0 - p_{cr}) \cdot r_0}{2G}")
    st.latex(r"G = \frac{E}{2(1 + \nu)}")

    st.markdown("**3. Hoek-Brown Parameters**")
    st.latex(r"m_b = m_i \cdot e^{\frac{GSI - 100}{28 - 14D}}")
    st.latex(r"s = e^{\frac{GSI - 100}{9 - 3D}}")
    st.latex(r"a = 0.5 + \frac{1}{6} \left( e^{-GSI/15} - e^{-20/3} \right)")

    st.markdown("**4. Hoek-Brown Strength & Transition**")
    st.latex(r"\sigma_{cm} = \frac{\sigma_{ci}}{2} \left[ (m_b + 4s)^a - m_b^a \right]")
    st.latex(r"p_{cr} = p_0 \cdot \left( 1 - \frac{\sigma_{cm}}{2p_0} \right)")
    st.latex(r"k_{\text{HB}} = \frac{2(1 - \nu)(m_b + 4s)^a}{1 + \nu}")
    st.latex(r"R_{pl} = r_0 \cdot \left( \frac{2p_0}{\sigma_{cm}} + 1 \right)^{1 / k_{\text{HB}}}")

    st.markdown("**5. Longitudinal Deformation Profiles (LDP)**")

    st.markdown("_Panet (1995):_")
    st.latex(r"u^*(X^*) = \begin{cases} (1 - \alpha) e^{1.5X^*}, & X^* \leq 0 \\ 1 - \alpha e^{-1.5X^*}, & X^* > 0 \end{cases}")

    st.markdown("_Hoek et al. (2002):_")
    st.latex(r"u^*(X^*) = \begin{cases} 0.25 e^{2.5X^*}, & X^* \leq 0 \\ 1 - 0.75 e^{-0.5X^*}, & X^* > 0 \end{cases}")

    st.markdown("_Vlachopoulos & Diederichs (2009):_")
    st.latex(r"u^*(X^*) = \begin{cases} \frac{1}{3} e^{2X^* - 0.15R^*}, & X^* \leq 0 \\ 1 - \left[1 - \frac{1}{3} e^{-0.15R^*} \right] e^{-3X^*/R^*}, & X^* > 0 \end{cases}")

    st.markdown("**6. Support Pressure (SCC)**")
    st.latex(r"p_{\text{supp}}(u) = \begin{cases} 0, & u < u_{\text{install}} \\ \min(k \cdot (u - u_{\text{install}}), p_{\text{max}}), & u \geq u_{\text{install}} \end{cases}")

    st.markdown("**7. Factor of Safety**")
    st.latex(r"\text{FoS} = \frac{p_{\text{max}}}{p_{\text{eq}}}")
