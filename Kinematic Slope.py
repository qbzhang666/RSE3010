import streamlit as st
import math
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Week 12 Planar LEM", layout="wide")

st.title("RSE3010 Week 12 — Planar Limit Equilibrium Method")
st.caption("Standalone teaching app: multiple failure modes — Planar, Wedge, Circular")

st.markdown("""
This app covers **multiple failure modes** for slope stability:
- **Planar wedge** geometry
- **Wedge failure**
- **Circular sliding failure**

Each mode uses simple geometry with **unit thickness** into the page.
""")

# -------------------------------------------------------------------
# Sidebar inputs
# -------------------------------------------------------------------

with st.sidebar:
    st.header("Geometry")

    H = st.number_input(
        "Slope height H (m)",
        min_value=1.0,
        value=20.0,
        step=1.0
    )

    psi = st.number_input(
        "Slope angle ψf (deg)",
        min_value=1.0,
        max_value=89.0,
        value=60.0,
        step=1.0
    )

    beta = st.number_input(
        "Sliding plane dip βj (deg)",
        min_value=1.0,
        max_value=89.0,
        value=40.0,
        step=1.0
    )

    failure_mode = st.selectbox(
        "Select failure mode",
        ["Planar", "Wedge", "Circular"]
    )

    st.header("Material")

    gamma = st.number_input(
        "Unit weight γ (kN/m³)",
        min_value=1.0,
        value=22.0,
        step=0.5
    )

    c = st.number_input(
        "Cohesion c (kPa)",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

    phi_dry = st.number_input(
        "Dry friction angle φdry (deg)",
        min_value=0.0,
        max_value=89.0,
        value=45.0,
        step=1.0
    )

    phi_wet = st.number_input(
        "Wet friction angle φwet (deg)",
        min_value=0.0,
        max_value=89.0,
        value=28.0,
        step=1.0
    )

    ru = st.slider(
        "Wet pore-pressure ratio ru",
        min_value=0.0,
        max_value=0.8,
        value=0.2,
        step=0.05
    )

    support_force = st.number_input(
        "Rockbolt / Cablebolt force T (kN)",
        min_value=0.0,
        value=0.0,
        step=10.0
    )

    bolt_angle = st.number_input(
        "Rockbolt / Cablebolt installation angle (degrees)",
        min_value=0.0,
        max_value=89.0,
        value=30.0,
        step=1.0
    )

    bolt_length = st.number_input(
        "Rockbolt / Cablebolt length (m)",
        min_value=0.0,
        value=2.0,
        step=0.5
    )

# -------------------------------------------------------------------
# LEM calculation function with rockbolt support
# -------------------------------------------------------------------

def planar_wedge_fos(
    H,
    psi_deg,
    beta_deg,
    gamma,
    c_kpa,
    phi_deg,
    ru=0.0,
    support=0.0,
    bolt_angle_deg=0.0,
    bolt_length=0.0
):
    """
    Simple planar wedge FoS calculation, now with rockbolt support.
    """

    psi_rad = math.radians(psi_deg)
    beta_rad = math.radians(beta_deg)
    phi_rad = math.radians(phi_deg)
    bolt_angle_rad = math.radians(bolt_angle_deg)

    if beta_deg >= psi_deg:
        return {
            "valid": False,
            "reason": "Sliding plane does not daylight because βj ≥ ψf."
        }

    # Horizontal distances to crest
    x_face = H / math.tan(psi_rad)
    x_plane = H / math.tan(beta_rad)

    # Wedge area per metre thickness
    area = 0.5 * H * (x_plane - x_face)

    # Weight
    W = gamma * area

    # Sliding plane length
    L = H / math.sin(beta_rad)

    # Normal force
    N = W * math.cos(beta_rad)

    # Pore pressure force using simplified ru ratio
    U = ru * N

    # Rockbolt support calculation (force applied perpendicular to the sliding plane)
    bolt_support = (support_force * math.cos(bolt_angle_rad)) * bolt_length

    # Driving force
    driving = W * math.sin(beta_rad)

    # Resisting force
    resisting = c_kpa * L + (N - U) * math.tan(phi_rad) + bolt_support

    # Factor of Safety (FoS)
    fos = resisting / driving if driving > 0 else float("inf")

    return {
        "valid": True,
        "Area (m²/m)": area,
        "Plane length L (m)": L,
        "Weight W (kN/m)": W,
        "Normal N (kN/m)": N,
        "Pore force U (kN/m)": U,
        "Rockbolt Support T (kN)": bolt_support,
        "Driving Wsinβ (kN/m)": driving,
        "Resisting (kN/m)": resisting,
        "FoS": round(fos, 2)
    }

# -------------------------------------------------------------------
# Run dry and wet calculations based on selected failure mode
# -------------------------------------------------------------------

if failure_mode == "Planar":
    dry = planar_wedge_fos(
        H=H,
        psi_deg=psi,
        beta_deg=beta,
        gamma=gamma,
        c_kpa=c,
        phi_deg=phi_dry,
        ru=0.0,
        support=support_force,
        bolt_angle_deg=bolt_angle,
        bolt_length=bolt_length
    )
    wet = planar_wedge_fos(
        H=H,
        psi_deg=psi,
        beta_deg=beta,
        gamma=gamma,
        c_kpa=c,
        phi_deg=phi_wet,
        ru=ru,
        support=support_force,
        bolt_angle_deg=bolt_angle,
        bolt_length=bolt_length
    )

# Additional failure modes: Wedge, Circular can be added here following similar approach

# -------------------------------------------------------------------
# Display outputs
# -------------------------------------------------------------------

if not dry["valid"]:
    st.warning(dry["reason"])

else:
    # ---------------------------------------------------------------
    # LEM schematic with rockbolt
    # ---------------------------------------------------------------

    st.subheader("LEM schematic with Rockbolt support")

    fig, ax = plt.subplots(figsize=(7, 5))

    # Geometry points
    x_toe = 0.0
    y_toe = 0.0
    y_crest = H

    x_crest_slope = H / math.tan(math.radians(psi))
    x_crest_plane = H / math.tan(math.radians(beta))

    # Plot slope face
    ax.plot(
        [x_toe, x_crest_slope],
        [y_toe, y_crest],
        linewidth=2.5,
        label="Slope face"
    )

    # Plot sliding plane
    ax.plot(
        [x_toe, x_crest_plane],
        [y_toe, y_crest],
        linewidth=2.5,
        linestyle="--",
        label="Sliding plane"
    )

    # Plot ground surface
    ax.plot(
        [x_crest_slope, x_crest_plane],
        [y_crest, y_crest],
        linewidth=2.0,
        label="Ground surface"
    )

    # Fill wedge
    ax.fill(
        [x_toe, x_crest_slope, x_crest_plane],
        [y_toe, y_crest, y_crest],
        alpha=0.25
    )

    # Wedge centroid approximation
    xw = (x_toe + x_crest_slope + x_crest_plane) / 3
    yw = (y_toe + y_crest + y_crest) / 3 + 1.5

    # Weight arrow
    ax.annotate(
        "W",
        xy=(xw, yw - 2.5),
        xytext=(xw, yw),
        arrowprops=dict(arrowstyle="->", linewidth=2),
        ha="center"
    )

    # Point on sliding plane
    xm = 0.55 * x_crest_plane
    ym = 0.55 * y_crest

    beta_rad = math.radians(beta)

    # Normal force arrow
    ax.annotate(
        "N",
        xy=(
            xm - 1.6 * math.sin(beta_rad),
            ym + 1.6 * math.cos(beta_rad)
        ),
        xytext=(xm, ym),
        arrowprops=dict(arrowstyle="->", linewidth=2),
        ha="center"
    )

    # Driving force arrow
    ax.annotate(
        "Driving\n$W\\sin\\beta$",
        xy=(
            xm + 2.0 * math.cos(beta_rad),
            ym + 2.0 * math.sin(beta_rad)
        ),
        xytext=(xm + 0.2, ym + 0.2),
        arrowprops=dict(arrowstyle="->", linewidth=2),
        ha="left"
    )

    # Resisting force arrow
    ax.annotate(
        "Resisting\n$cL+(N-U)\\tan\\phi+T$",
        xy=(
            xm - 2.0 * math.cos(beta_rad),
            ym - 2.0 * math.sin(beta_rad)
        ),
        xytext=(xm - 0.5, ym - 1.0),
        arrowprops=dict(arrowstyle="->", linewidth=2),
        ha="right"
    )

    # Pore-pressure force arrow
    if ru > 0:
        ax.annotate(
            "U",
            xy=(
                xm + 1.1 * math.sin(beta_rad),
                ym - 1.1 * math.cos(beta_rad)
            ),
            xytext=(
                xm + 2.0 * math.sin(beta_rad),
                ym - 2.0 * math.cos(beta_rad)
            ),
            arrowprops=dict(arrowstyle="->", linewidth=1.8),
            ha="center"
        )

    # Angle labels
    ax.text(
        x_toe + 1.0,
        0.4,
        f"β = {beta:.1f}°",
        fontsize=11
    )

    ax.text(
        x_toe + 0.8,
        2.0,
        f"ψ = {psi:.1f}°",
        fontsize=11
    )

    ax.text(
        (x_crest_slope + x_crest_plane) / 2,
        y_crest + 0.8,
        "Potential sliding wedge",
        ha="center"
    )

    ax.set_aspect("equal")
    ax.set_xlim(-1, max(x_crest_plane, x_crest_slope) + 4)
    ax.set_ylim(-1, H + 4)

    ax.set_xlabel("Horizontal distance (m)")
    ax.set_ylabel("Elevation (m)")
    ax.legend(loc="upper right")

    st.pyplot(fig)

    st.caption(
        "Schematic of the planar LEM geometry used in this teaching app: "
        "slope face, sliding plane, wedge weight W, normal force N, pore-pressure force U, "
        "and resisting/driving components."
    )

    # ---------------------------------------------------------------
    # Numerical outputs
    # ---------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dry condition")
        st.metric("FoS dry", f"{dry['FoS']:.2f}")
        st.dataframe(
            pd.DataFrame([dry]).T.rename(columns={0: "Value"}),
            use_container_width=True
        )

    with col2:
        st.subheader("Wet condition")
        st.metric("FoS wet", f"{wet['FoS']:.2f}")
        st.dataframe(
            pd.DataFrame([wet]).T.rename(columns={0: "Value"}),
            use_container_width=True
        )

    # ---------------------------------------------------------------
    # Stability interpretation table
    # ---------------------------------------------------------------

    st.subheader("Stability interpretation")

    summary = pd.DataFrame(
        [
            [
                "Dry",
                phi_dry,
                0.0,
                dry["FoS"],
                "Stable" if dry["FoS"] >= 1.5 else (
                    "Marginal" if dry["FoS"] >= 1.0 else "Unstable"
                )
            ],
            [
                "Wet",
                phi_wet,
                ru,
                wet["FoS"],
                "Stable" if wet["FoS"] >= 1.5 else (
                    "Marginal" if wet["FoS"] >= 1.0 else "Unstable"
                )
            ]
        ],
        columns=[
            "Condition",
            "Friction angle φ (deg)",
            "ru",
            "FoS",
            "Interpretation"
        ]
    )

    st.dataframe(summary, use_container_width=True)

    # ---------------------------------------------------------------
    # Equation and teaching notes
    # ---------------------------------------------------------------

    st.markdown("""
### Equation used

\\[
FoS =
\\frac{
cL + (W\\cos\\beta - U)\\tan\\phi + T
}{
W\\sin\\beta
}
\\]

where:

- \\(c\\) = cohesion  
- \\(L\\) = sliding plane length  
- \\(W\\) = wedge weight  
- \\(\\beta\\) = sliding plane dip  
- \\(U\\) = pore-pressure force  
- \\(\\phi\\) = friction angle  
- \\(T\\) = additional support force along the sliding direction  

### Teaching notes

- If \\(\\beta_j \\geq \\psi_f\\), the plane does not daylight.
- Dry/wet comparison should normally reduce FoS because friction decreases and pore pressure may increase.
- The schematic helps students connect the geometry to the equilibrium equation.
- Use this app after SMR and kinematic analysis have identified the controlling plane.
""")
