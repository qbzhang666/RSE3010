
import streamlit as st
import math
import pandas as pd

st.set_page_config(page_title="Week 12 Planar LEM", layout="wide")

st.title("RSE3010 Week 12 — Planar Limit Equilibrium Method")
st.caption("Standalone teaching app: quantitative stability — What will fail?")

st.markdown("""
This app uses a simple **planar wedge** geometry for a daylighting discontinuity.
It is intended for teaching and assignment checking, not final design.

Assumption: unit thickness into the page, triangular wedge between the slope face and the sliding plane.
""")

with st.sidebar:
    st.header("Geometry")
    H = st.number_input("Slope height H (m)", min_value=1.0, value=20.0, step=1.0)
    psi = st.number_input("Slope angle ψf (deg)", min_value=1.0, max_value=89.0, value=60.0, step=1.0)
    beta = st.number_input("Sliding plane dip βj (deg)", min_value=1.0, max_value=89.0, value=40.0, step=1.0)
    st.header("Material")
    gamma = st.number_input("Unit weight γ (kN/m³)", min_value=1.0, value=22.0, step=0.5)
    c = st.number_input("Cohesion c (kPa)", min_value=0.0, value=0.0, step=1.0)
    phi_dry = st.number_input("Dry friction angle φdry (deg)", min_value=0.0, max_value=89.0, value=45.0, step=1.0)
    phi_wet = st.number_input("Wet friction angle φwet (deg)", min_value=0.0, max_value=89.0, value=28.0, step=1.0)
    ru = st.slider("Wet pore-pressure ratio ru", min_value=0.0, max_value=0.8, value=0.2, step=0.05)
    support = st.number_input("Additional support force along sliding plane (kN/m)", min_value=0.0, value=0.0, step=10.0)

def planar_wedge_fos(H, psi_deg, beta_deg, gamma, c_kpa, phi_deg, ru=0.0, support=0.0):
    psi = math.radians(psi_deg)
    beta = math.radians(beta_deg)
    phi = math.radians(phi_deg)

    if beta_deg >= psi_deg:
        return {
            "valid": False,
            "reason": "Sliding plane does not daylight because βj ≥ ψf.",
        }

    x_face = H / math.tan(psi)
    x_plane = H / math.tan(beta)
    area = 0.5 * H * (x_plane - x_face)
    W = gamma * area
    L = H / math.sin(beta)
    N = W * math.cos(beta)
    U = ru * N
    driving = W * math.sin(beta)
    resisting = c_kpa * L + (N - U) * math.tan(phi) + support
    fos = resisting / driving if driving > 0 else float("inf")
    return {
        "valid": True, "Area (m²/m)": area, "Plane length L (m)": L, "Weight W (kN/m)": W,
        "Normal N (kN/m)": N, "Pore force U (kN/m)": U, "Driving Wsinβ (kN/m)": driving,
        "Resisting (kN/m)": resisting, "FoS": fos
    }

dry = planar_wedge_fos(H, psi, beta, gamma, c, phi_dry, ru=0.0, support=support)
wet = planar_wedge_fos(H, psi, beta, gamma, c, phi_wet, ru=ru, support=support)

if not dry["valid"]:
    st.warning(dry["reason"])
else:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dry condition")
        st.metric("FoS dry", f"{dry['FoS']:.2f}")
        st.dataframe(pd.DataFrame([dry]).T.rename(columns={0:"Value"}), use_container_width=True)
    with col2:
        st.subheader("Wet condition")
        st.metric("FoS wet", f"{wet['FoS']:.2f}")
        st.dataframe(pd.DataFrame([wet]).T.rename(columns={0:"Value"}), use_container_width=True)

    st.subheader("Stability interpretation")
    summary = pd.DataFrame([
        ["Dry", phi_dry, 0.0, dry["FoS"], "Stable" if dry["FoS"] >= 1.5 else ("Marginal" if dry["FoS"] >= 1.0 else "Unstable")],
        ["Wet", phi_wet, ru, wet["FoS"], "Stable" if wet["FoS"] >= 1.5 else ("Marginal" if wet["FoS"] >= 1.0 else "Unstable")],
    ], columns=["Condition", "Friction angle φ (deg)", "ru", "FoS", "Interpretation"])
    st.dataframe(summary, use_container_width=True)

    st.markdown("""
### Equation used
\[
FoS = \\frac{cL + (W\\cos\\beta - U)\\tan\\phi + T}{W\\sin\\beta}
\]

where \(T\) is the added support force along the sliding direction.

### Teaching notes
- If \(\\beta_j \\geq \\psi_f\), the plane does not daylight.
- Dry/wet comparison should normally reduce FoS because friction decreases and pore pressure may increase.
- Use this app after SMR and kinematic analysis have identified the controlling plane.
""")
