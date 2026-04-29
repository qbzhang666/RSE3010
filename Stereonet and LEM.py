"""
RSE3010 Week 12 — Planar Limit Equilibrium Method
Hoek & Bray planar failure with:
- vertical tension crack
- water in tension crack
- uplift on sliding plane
- optional rockbolt / cable anchor support
"""

import streamlit as st
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

st.set_page_config(page_title="RSE3010 Week 12 LEM", layout="wide")

st.title("RSE3010 Week 12 — Planar LEM Solution")
st.caption("Hoek & Bray planar limit equilibrium method with tension crack, water and optional support")

st.markdown("""
This app calculates the **Factor of Safety (FoS)** for planar failure using the
**Hoek & Bray planar limit equilibrium method**.

Default Assignment 3 values:

- Slope Side A: 60° / 170°
- Governing planar joint: J1 = 40° / 160°
- Slope height: H = 15 m
- Unit weight: γ = 20 kN/m³
- Cohesion: c′ = 20 kPa
- Dry friction angle: φdry = 45°
- Wet friction angle: φwet = 28°
""")

# =========================================================
# Sidebar inputs
# =========================================================
with st.sidebar:
    st.header("Slope geometry")

    H = st.number_input(
        "Slope height H (m)",
        min_value=1.0,
        value=15.0,
        step=1.0,
    )

    psi_f = st.number_input(
        "Slope face angle ψf (degrees)",
        min_value=10.0,
        max_value=89.0,
        value=60.0,
        step=1.0,
    )

    psi_p = st.number_input(
        "Sliding plane dip ψp (degrees)",
        min_value=1.0,
        max_value=89.0,
        value=40.0,
        step=1.0,
    )

    st.header("Tension crack and water")

    use_auto_z = st.checkbox(
        "Estimate tension crack depth automatically",
        value=True,
        help="Uses z/H = 1 - sqrt(cotψf tanψp). You may untick this and enter z manually.",
    )

    def auto_tension_crack_depth(H, psi_f, psi_p):
        psi_f_rad = math.radians(psi_f)
        psi_p_rad = math.radians(psi_p)

        if psi_p >= psi_f:
            return 0.0

        ratio = 1.0 - math.sqrt((1.0 / math.tan(psi_f_rad)) * math.tan(psi_p_rad))
        ratio = max(0.0, min(ratio, 0.95))
        return ratio * H

    z_auto = auto_tension_crack_depth(H, psi_f, psi_p)

    if use_auto_z:
        z = z_auto
        st.info(f"Estimated tension crack depth z = {z:.2f} m")
    else:
        z = st.number_input(
            "Tension crack depth z (m)",
            min_value=0.0,
            max_value=max(H - 0.01, 0.01),
            value=min(0.30 * H, H - 0.01),
            step=0.1,
        )

    z_w = st.number_input(
        "Water depth in tension crack zw (m)",
        min_value=0.0,
        max_value=max(z, 0.0),
        value=min(z, z),
        step=0.1,
    )

    st.header("Material properties")

    gamma = st.number_input(
        "Rock unit weight γ (kN/m³)",
        min_value=1.0,
        value=20.0,
        step=0.5,
    )

    gamma_w = st.number_input(
        "Water unit weight γw (kN/m³)",
        min_value=1.0,
        value=9.81,
        step=0.01,
    )

    c = st.number_input(
        "Cohesion c′ (kPa)",
        min_value=0.0,
        value=20.0,
        step=1.0,
    )

    phi_dry = st.number_input(
        "Dry friction angle φdry (degrees)",
        min_value=0.0,
        max_value=89.0,
        value=45.0,
        step=1.0,
    )

    phi_wet = st.number_input(
        "Wet friction angle φwet (degrees)",
        min_value=0.0,
        max_value=89.0,
        value=28.0,
        step=1.0,
    )

    st.header("Support")

    T = st.number_input(
        "Anchor / bolt support T (kN/m run)",
        min_value=0.0,
        value=0.0,
        step=10.0,
        help="Total working support force per metre run of slope. Do not multiply this by bolt length.",
    )

    psi_T = st.number_input(
        "Anchor inclination ψT (degrees below horizontal)",
        min_value=-30.0,
        max_value=60.0,
        value=15.0,
        step=1.0,
    )

    st.header("Design target")

    fos_target = st.number_input(
        "Target FoS",
        min_value=1.0,
        value=1.30,
        step=0.05,
    )


# =========================================================
# Hoek & Bray planar LEM function
# =========================================================
def planar_fos(H, psi_f, psi_p, z, z_w, c, phi, gamma, gamma_w, T=0.0, psi_T=15.0):
    if psi_p >= psi_f:
        return {
            "valid": False,
            "reason": "Invalid geometry: sliding plane does not daylight because ψp ≥ ψf.",
        }

    if z >= H:
        return {
            "valid": False,
            "reason": "Invalid geometry: tension crack depth z must be smaller than slope height H.",
        }

    if z_w > z:
        return {
            "valid": False,
            "reason": "Invalid water condition: zw cannot exceed z.",
        }

    psi_f_rad = math.radians(psi_f)
    psi_p_rad = math.radians(psi_p)
    phi_rad = math.radians(phi)
    psi_T_rad = math.radians(psi_T)

    A = (H - z) / math.sin(psi_p_rad)

    W = 0.5 * gamma * H**2 * (
        (1.0 - (z / H) ** 2) / math.tan(psi_p_rad)
        - 1.0 / math.tan(psi_f_rad)
    )

    V = 0.5 * gamma_w * z_w**2
    U = 0.5 * gamma_w * z_w * A

    T_along = T * math.cos(psi_p_rad + psi_T_rad)
    T_normal = T * math.sin(psi_p_rad + psi_T_rad)

    N_eff_raw = W * math.cos(psi_p_rad) - U - V * math.sin(psi_p_rad) + T_normal
    N_eff = max(0.0, N_eff_raw)

    Driving = W * math.sin(psi_p_rad) + V * math.cos(psi_p_rad) - T_along
    Resisting = c * A + N_eff * math.tan(phi_rad)

    if Driving <= 0:
        FoS = float("inf")
    else:
        FoS = Resisting / Driving

    return {
        "valid": True,
        "A (m²/m)": A,
        "W (kN/m)": W,
        "V (kN/m)": V,
        "U (kN/m)": U,
        "T along plane (kN/m)": T_along,
        "T normal (kN/m)": T_normal,
        "N_eff_raw (kN/m)": N_eff_raw,
        "N_eff used (kN/m)": N_eff,
        "Driving (kN/m)": Driving,
        "Resisting (kN/m)": Resisting,
        "FoS": FoS,
    }


dry = planar_fos(
    H=H,
    psi_f=psi_f,
    psi_p=psi_p,
    z=z,
    z_w=0.0,
    c=c,
    phi=phi_dry,
    gamma=gamma,
    gamma_w=gamma_w,
    T=T,
    psi_T=psi_T,
)

wet = planar_fos(
    H=H,
    psi_f=psi_f,
    psi_p=psi_p,
    z=z,
    z_w=z_w,
    c=c,
    phi=phi_wet,
    gamma=gamma,
    gamma_w=gamma_w,
    T=T,
    psi_T=psi_T,
)

if not dry["valid"]:
    st.error(dry["reason"])
    st.stop()

if not wet["valid"]:
    st.error(wet["reason"])
    st.stop()


def status(fos):
    if fos == float("inf"):
        return "Locked / excessive support"
    if fos < 1.0:
        return "Unstable"
    if fos < fos_target:
        return "Marginal / support required"
    return "Stable"


# =========================================================
# Results
# =========================================================
st.subheader("1. Key Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("FoS dry", f"{dry['FoS']:.2f}", status(dry["FoS"]))

with col2:
    st.metric("FoS wet", f"{wet['FoS']:.2f}", status(wet["FoS"]))

with col3:
    st.metric("Tension crack depth z", f"{z:.2f} m")

with col4:
    st.metric("Water depth zw", f"{z_w:.2f} m")


summary = pd.DataFrame(
    [
        ["Dry", phi_dry, 0.0, dry["FoS"], status(dry["FoS"])],
        ["Wet", phi_wet, z_w, wet["FoS"], status(wet["FoS"])],
    ],
    columns=["Condition", "Friction angle φ (°)", "Water depth zw (m)", "FoS", "Interpretation"],
)

st.dataframe(
    summary.style.format(
        {
            "Friction angle φ (°)": "{:.0f}",
            "Water depth zw (m)": "{:.2f}",
            "FoS": "{:.2f}",
        }
    ),
    use_container_width=True,
)

st.subheader("2. Calculation Details")

calc_df = pd.DataFrame(
    {
        "Dry": {k: v for k, v in dry.items() if k != "valid"},
        "Wet": {k: v for k, v in wet.items() if k != "valid"},
    }
)

st.dataframe(calc_df.style.format("{:.2f}"), use_container_width=True)


# =========================================================
# Equations
# =========================================================
st.subheader("3. Equations Used")

st.latex(r"""
A = \frac{H-z}{\sin\psi_p}
""")

st.latex(r"""
W =
\frac{1}{2}\gamma H^2
\left[
\left(1-\left(\frac{z}{H}\right)^2\right)\cot\psi_p
-\cot\psi_f
\right]
""")

st.latex(r"""
V = \frac{1}{2}\gamma_w z_w^2
\qquad
U = \frac{1}{2}\gamma_w z_w A
""")

st.latex(r"""
N' =
W\cos\psi_p
-
U
-
V\sin\psi_p
+
T\sin(\psi_p+\psi_T)
""")

st.latex(r"""
Driving =
W\sin\psi_p
+
V\cos\psi_p
-
T\cos(\psi_p+\psi_T)
""")

st.latex(r"""
Resisting =
cA + N'\tan\phi
""")

st.latex(r"""
FoS =
\frac{cA + N'\tan\phi}
{W\sin\psi_p + V\cos\psi_p - T\cos(\psi_p+\psi_T)}
""")


# =========================================================
# Schematic
# =========================================================
st.subheader("4. Corrected Schematic")

psi_f_rad = math.radians(psi_f)
psi_p_rad = math.radians(psi_p)

x_toe = 0.0
y_toe = 0.0

x_crest = H / math.tan(psi_f_rad)
y_crest = H

x_tc = (H - z) / math.tan(psi_p_rad)
y_tc_base = H - z
y_tc_top = H

fig, ax = plt.subplots(figsize=(10, 7))

wedge_points = [
    (x_toe, y_toe),
    (x_crest, y_crest),
    (x_tc, y_tc_top),
    (x_tc, y_tc_base),
]

wedge = Polygon(wedge_points, closed=True, alpha=0.22, edgecolor="none")
ax.add_patch(wedge)

ax.plot([x_toe, x_crest], [y_toe, y_crest], linewidth=3, label="Slope face")
ax.plot([x_toe, x_tc], [y_toe, y_tc_base], linestyle="--", linewidth=3, label="Sliding plane")
ax.plot([x_tc, x_tc], [y_tc_base, y_tc_top], linestyle=":", linewidth=3, label="Tension crack")
ax.plot([x_crest, x_tc], [y_crest, y_tc_top], linewidth=2, label="Upper ground")

if z_w > 0:
    water_top = y_tc_base + z_w
    ax.fill_between([x_tc - 0.20, x_tc + 0.20], y_tc_base, water_top, alpha=0.50)
    ax.text(x_tc + 0.35, water_top, f"zw = {z_w:.2f} m", fontsize=10)

arrow = dict(arrowstyle="-|>", mutation_scale=18, linewidth=2)

cx = sum(p[0] for p in wedge_points) / len(wedge_points)
cy = sum(p[1] for p in wedge_points) / len(wedge_points)

ax.add_patch(FancyArrowPatch((cx, cy + 0.15 * H), (cx, cy - 0.15 * H), **arrow))
ax.text(cx + 0.2, cy, "W", fontsize=12)

plane_dx = x_tc - x_toe
plane_dy = y_tc_base - y_toe
plane_len = math.hypot(plane_dx, plane_dy)

tx = plane_dx / plane_len
ty = plane_dy / plane_len

nx = -ty
ny = tx

xm = 0.55 * x_tc
ym = (xm / x_tc) * y_tc_base

ax.add_patch(FancyArrowPatch((xm, ym), (xm + 0.18 * H * nx, ym + 0.18 * H * ny), **arrow))
ax.text(xm + 0.20 * H * nx, ym + 0.20 * H * ny, "N'", fontsize=12)

if z_w > 0:
    xu = 0.25 * x_tc
    yu = (xu / x_tc) * y_tc_base
    ax.add_patch(FancyArrowPatch((xu, yu), (xu + 0.12 * H * nx, yu + 0.12 * H * ny), **arrow))
    ax.text(xu + 0.14 * H * nx, yu + 0.14 * H * ny, "U", fontsize=12)

D_start_x = 0.75 * x_tc + 0.7 * nx
D_start_y = (0.75 * x_tc / x_tc) * y_tc_base + 0.7 * ny
ax.add_patch(
    FancyArrowPatch(
        (D_start_x, D_start_y),
        (D_start_x - 0.18 * H * tx, D_start_y - 0.18 * H * ty),
        **arrow,
    )
)
ax.text(D_start_x + 0.5, D_start_y + 0.5, "Driving", fontsize=10)

R_start_x = 0.40 * x_tc - 0.7 * nx
R_start_y = (0.40 * x_tc / x_tc) * y_tc_base - 0.7 * ny
ax.add_patch(
    FancyArrowPatch(
        (R_start_x, R_start_y),
        (R_start_x + 0.18 * H * tx, R_start_y + 0.18 * H * ty),
        **arrow,
    )
)
ax.text(R_start_x - 0.5, R_start_y - 1.0, "Resisting", fontsize=10)

if z_w > 0:
    yv = y_tc_base + z_w / 2
    ax.add_patch(FancyArrowPatch((x_tc + 0.12 * H, yv), (x_tc, yv), **arrow))
    ax.text(x_tc + 0.13 * H, yv, "V", fontsize=12)

if T > 0:
    sx = 0.45 * x_crest
    sy = 0.45 * y_crest

    psi_T_rad = math.radians(psi_T)
    Td_x = math.cos(psi_T_rad)
    Td_y = -math.sin(psi_T_rad)

    ax.add_patch(
        FancyArrowPatch(
            (sx, sy),
            (sx + 0.18 * H * Td_x, sy + 0.18 * H * Td_y),
            **arrow,
        )
    )
    ax.text(sx + 0.20 * H * Td_x, sy + 0.20 * H * Td_y, "T", fontsize=12)

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Horizontal distance (m)")
ax.set_ylabel("Elevation (m)")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title("Planar LEM schematic with corrected force directions")

st.pyplot(fig)


# =========================================================
# Sensitivity analysis
# =========================================================
st.subheader("5. Sensitivity Analysis")

tab1, tab2, tab3 = st.tabs(["FoS vs water depth", "FoS vs slope angle", "FoS vs support force"])

with tab1:
    zws = np.linspace(0, z, 25)
    fos_dry = [
        planar_fos(H, psi_f, psi_p, z, zw, c, phi_dry, gamma, gamma_w, T, psi_T)["FoS"]
        for zw in zws
    ]
    fos_wet = [
        planar_fos(H, psi_f, psi_p, z, zw, c, phi_wet, gamma, gamma_w, T, psi_T)["FoS"]
        for zw in zws
    ]

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(zws, fos_dry, marker="o", label=f"Dry friction φ = {phi_dry:.0f}°")
    ax1.plot(zws, fos_wet, marker="s", label=f"Wet friction φ = {phi_wet:.0f}°")
    ax1.axhline(fos_target, linestyle="--", label=f"Target FoS = {fos_target:.2f}")
    ax1.axhline(1.0, linestyle=":", label="FoS = 1.0")
    ax1.set_xlabel("Water depth in tension crack zw (m)")
    ax1.set_ylabel("FoS")
    ax1.set_title("FoS sensitivity to water depth")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)

with tab2:
    slope_angles = np.arange(max(psi_p + 1, 41), 81, 2)
    fos_values = [
        planar_fos(H, angle, psi_p, z, z_w, c, phi_wet, gamma, gamma_w, T, psi_T)["FoS"]
        for angle in slope_angles
    ]

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(slope_angles, fos_values, marker="o")
    ax2.axhline(fos_target, linestyle="--", label=f"Target FoS = {fos_target:.2f}")
    ax2.axhline(1.0, linestyle=":", label="FoS = 1.0")
    ax2.set_xlabel("Slope face angle ψf (degrees)")
    ax2.set_ylabel("Wet FoS")
    ax2.set_title("FoS sensitivity to slope angle")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)

with tab3:
    support_values = np.linspace(0, 300, 31)
    fos_values = [
        planar_fos(H, psi_f, psi_p, z, z_w, c, phi_wet, gamma, gamma_w, support, psi_T)["FoS"]
        for support in support_values
    ]

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(support_values, fos_values, marker="o")
    ax3.axhline(fos_target, linestyle="--", label=f"Target FoS = {fos_target:.2f}")
    ax3.axhline(1.0, linestyle=":", label="FoS = 1.0")
    ax3.set_xlabel("Support force T (kN/m run)")
    ax3.set_ylabel("Wet FoS")
    ax3.set_title("FoS sensitivity to support force")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    st.pyplot(fig3)


# =========================================================
# Required support
# =========================================================
st.subheader("6. Required Support Force")

def required_support_for_target(H, psi_f, psi_p, z, z_w, c, phi, gamma, gamma_w, psi_T, fos_target):
    base = planar_fos(H, psi_f, psi_p, z, z_w, c, phi, gamma, gamma_w, 0.0, psi_T)

    psi_p_rad = math.radians(psi_p)
    psi_T_rad = math.radians(psi_T)
    phi_rad = math.radians(phi)

    A = base["A (m²/m)"]
    W = base["W (kN/m)"]
    V = base["V (kN/m)"]
    U = base["U (kN/m)"]

    resisting_no_T = c * A + (
        W * math.cos(psi_p_rad)
        - U
        - V * math.sin(psi_p_rad)
    ) * math.tan(phi_rad)

    driving_no_T = W * math.sin(psi_p_rad) + V * math.cos(psi_p_rad)

    support_coeff = (
        math.sin(psi_p_rad + psi_T_rad) * math.tan(phi_rad)
        + fos_target * math.cos(psi_p_rad + psi_T_rad)
    )

    T_req = (fos_target * driving_no_T - resisting_no_T) / support_coeff

    return max(0.0, T_req)


T_req = required_support_for_target(
    H, psi_f, psi_p, z, z_w, c, phi_wet, gamma, gamma_w, psi_T, fos_target
)

st.metric("Required support force for wet target FoS", f"{T_req:.1f} kN/m run")

bolt_capacity = st.number_input(
    "Assumed working load per bolt (kN)",
    min_value=50.0,
    value=250.0,
    step=50.0,
)

horizontal_spacing = st.number_input(
    "Horizontal bolt spacing sh (m)",
    min_value=0.5,
    value=2.0,
    step=0.5,
)

rows_required = math.ceil(T_req * horizontal_spacing / bolt_capacity)

st.latex(rf"""
n_{{rows}} =
\left\lceil
\frac{{T_{{req}}s_h}}{{T_{{bolt}}}}
\right\rceil
=
\left\lceil
\frac{{{T_req:.1f}\times {horizontal_spacing:.1f}}}{{{bolt_capacity:.0f}}}
\right\rceil
=
{rows_required}
""")


# =========================================================
# Teaching notes
# =========================================================
st.subheader("7. Teaching Notes")

st.markdown("""
**Common mistakes to avoid**

1. Do not use \( r_u \) in the planar Hoek & Bray closed-form calculation.  
   Water is included through \( V \) and \( U \), calculated directly from \( z_w \).

2. Do not multiply anchor force by bolt length.  
   Bolt length is a design check to confirm anchorage beyond the failure surface.

3. The anchor contributes in two ways:
   - It reduces driving force through \( T\\cos(\\psi_p+\\psi_T) \)
   - It increases normal force through \( T\\sin(\\psi_p+\\psi_T) \)

4. If both planar and wedge mechanisms are possible, this app performs the required planar LEM hand-calculation component.  
   Wedge failure should be discussed using 3D wedge software or additional wedge checks.

5. Always check that \( \\psi_p < \\psi_f \).  
   If \( \\psi_p \\geq \\psi_f \), the sliding plane does not daylight.
""")
