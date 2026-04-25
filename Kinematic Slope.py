"""
RSE3010 Week 12 — Planar Limit Equilibrium Method (CORRECTED)

Hoek & Bray (1981) closed-form planar failure with:
  - vertical tension crack at depth z behind crest
  - water in tension crack to depth z_w
  - water seepage along sliding plane (uplift U)
  - optional rockbolt / cable anchor with inclination ψ_T

Validation: H=60m, ψ_f=60°, ψ_p=35°, z=14m, z_w=5m, c=25 kPa, φ=37°
            → FoS = 1.094 (textbook 1.09).
"""

import streamlit as st
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

st.set_page_config(page_title="Planar LEM (Hoek & Bray)", layout="wide")

st.title("RSE3010 Week 12 — Planar Limit Equilibrium Method")
st.caption("Hoek & Bray (1981) closed-form, with tension crack, water, and anchor")

st.markdown("""
This app computes the factor of safety of a planar slope failure using the
**Hoek & Bray (1981)** closed-form equations. The wedge slides on a single
discontinuity that daylights on the slope face. A vertical tension crack
may exist behind the crest; water can fill the crack and seep along the
base. An optional rockbolt or cable anchor adds a stabilising force.
""")

# =====================================================================
# Sidebar inputs
# =====================================================================
with st.sidebar:
    st.header("Slope geometry")
    H = st.number_input("Slope height H (m)",
                        min_value=1.0, value=60.0, step=1.0)
    psi_f = st.number_input("Slope face angle ψ_f (deg)",
                            min_value=10.0, max_value=89.0,
                            value=60.0, step=1.0)
    psi_p = st.number_input("Sliding plane dip ψ_p (deg)",
                            min_value=1.0, max_value=89.0,
                            value=35.0, step=1.0)

    st.header("Tension crack")
    z = st.number_input("Tension crack depth z (m)",
                        min_value=0.0, value=14.0, step=1.0,
                        help="Vertical depth from upper ground surface. "
                             "Set to 0 for a block without TC.")
    z_w = st.number_input("Water depth in TC, z_w (m)",
                          min_value=0.0, value=5.0, step=0.5,
                          help="Water level in the tension crack, "
                               "measured upward from the base of the TC. "
                               "Cannot exceed z.")

    st.header("Material properties")
    gamma = st.number_input("Rock unit weight γ (kN/m³)",
                            min_value=10.0, value=26.0, step=0.5)
    gamma_w = st.number_input("Water unit weight γ_w (kN/m³)",
                              min_value=9.0, value=9.81, step=0.01)
    c = st.number_input("Cohesion c (kPa)",
                        min_value=0.0, value=25.0, step=1.0)
    phi_dry = st.number_input("Dry friction angle φ_dry (deg)",
                              min_value=0.0, max_value=89.0,
                              value=37.0, step=1.0)
    phi_wet = st.number_input("Wet friction angle φ_wet (deg)",
                              min_value=0.0, max_value=89.0,
                              value=24.0, step=1.0)

    st.header("Support (per metre run)")
    T = st.number_input("Anchor tension T (kN/m)",
                        min_value=0.0, value=0.0, step=50.0,
                        help="Total anchor force per metre length of pit "
                             "wall (sum of all bolts in 1 m horizontal strip).")
    psi_T = st.number_input("Anchor inclination ψ_T (deg, +ve = below horizontal)",
                            min_value=-30.0, max_value=60.0,
                            value=15.0, step=1.0,
                            help="Positive: drilled downward (drainage-friendly). "
                                 "Negative: drilled upward (mechanical optimum).")

# =====================================================================
# Hoek & Bray planar FoS
# =====================================================================
def planar_fos(H, psi_f, psi_p, z, z_w, c, phi, gamma, gamma_w, T, psi_T):
    """
    Hoek & Bray (1981) planar failure with vertical TC and horizontal upper slope.

    Sign convention:
      - ψ_T > 0  →  anchor drilled below horizontal (typical, drainage-friendly)
      - ψ_T < 0  →  anchor drilled above horizontal
      - The angle between the anchor line and the sliding plane is (ψ_p + ψ_T)

    Returns dict with all intermediate values.
    """
    if psi_p >= psi_f:
        return {"valid": False,
                "reason": "Sliding plane does not daylight (ψ_p ≥ ψ_f)."}
    if z >= H:
        return {"valid": False,
                "reason": "Tension crack depth must be less than slope height."}
    if z_w > z:
        return {"valid": False,
                "reason": "Water depth z_w cannot exceed TC depth z."}

    psi_f_r = math.radians(psi_f)
    psi_p_r = math.radians(psi_p)
    psi_T_r = math.radians(psi_T)
    phi_r   = math.radians(phi)

    # Base area (per metre run, out of plane)
    A = (H - z) / math.sin(psi_p_r)

    # Block weight (Hoek & Bray, horizontal upper slope, vertical TC)
    W = 0.5 * gamma * H**2 * (
            (1 - (z/H)**2) / math.tan(psi_p_r)
            - 1.0 / math.tan(psi_f_r)
        )

    # Water forces
    V = 0.5 * gamma_w * z_w**2          # horizontal force in TC
    U = 0.5 * gamma_w * z_w * A         # uplift on base

    # Anchor contributions
    T_along_plane = T * math.cos(psi_p_r + psi_T_r)   # resists sliding
    T_normal      = T * math.sin(psi_p_r + psi_T_r)   # adds to N

    # Effective normal force on the plane
    N_eff = (W * math.cos(psi_p_r)
             - U
             - V * math.sin(psi_p_r)
             + T_normal)
    N_eff_clamped = max(0.0, N_eff)

    # Driving and resisting (along the plane)
    Driving   = W * math.sin(psi_p_r) + V * math.cos(psi_p_r) - T_along_plane
    Resisting = c * A + N_eff_clamped * math.tan(phi_r)

    if Driving <= 0:
        FoS = float("inf")
        reason_inf = "Anchor exceeds gravity + water driving forces; block locked."
    else:
        FoS = Resisting / Driving
        reason_inf = None

    return {
        "valid": True,
        "A (m²/m)":              A,
        "W (kN/m)":              W,
        "V (kN/m)":              V,
        "U (kN/m)":              U,
        "T·cos(ψ_p+ψ_T) (kN/m)": T_along_plane,
        "T·sin(ψ_p+ψ_T) (kN/m)": T_normal,
        "N' (kN/m)":             N_eff_clamped,
        "Driving (kN/m)":        Driving,
        "Resisting (kN/m)":      Resisting,
        "FoS":                   FoS,
        "reason_inf":            reason_inf,
    }

# =====================================================================
# Compute dry and wet
# =====================================================================
dry = planar_fos(H, psi_f, psi_p, z, 0,    c, phi_dry, gamma, gamma_w, T, psi_T)
wet = planar_fos(H, psi_f, psi_p, z, z_w,  c, phi_wet, gamma, gamma_w, T, psi_T)

if not dry["valid"]:
    st.error(dry["reason"])
    st.stop()

# =====================================================================
# Schematic — corrected geometry and force directions
# =====================================================================
st.subheader("Schematic of planar failure with force components")

psi_f_r = math.radians(psi_f)
psi_p_r = math.radians(psi_p)

# Geometry: toe at origin, slope face goes up-right
x_toe, y_toe = 0.0, 0.0
x_crest = H / math.tan(psi_f_r)        # toe-to-crest horizontal
y_crest = H
# Sliding plane meets upper ground at x_plane (flatter than face)
x_plane = (H - z) / math.tan(psi_p_r)  # toe-to-base-of-TC horizontal
# Tension crack at top of sliding plane, vertical
x_TC = x_plane
y_TC_top = H
y_TC_base = H - z

# Wedge polygon: toe → crest → along upper ground to TC → down TC → back to toe along plane
wedge_vertices = [
    (x_toe,   y_toe),
    (x_crest, y_crest),
    (x_TC,    y_TC_top),
    (x_TC,    y_TC_base),
]

fig, ax = plt.subplots(figsize=(10, 7))

# Plot wedge fill
wedge = Polygon(wedge_vertices, alpha=0.18, facecolor="#E69F00",
                edgecolor="none")
ax.add_patch(wedge)

# Plot rock mass behind the slip surface (faint)
xmax = max(x_TC, x_crest) + 8
ymin = -3
behind = Polygon([
    (x_toe, y_toe), (x_TC, y_TC_base), (x_TC, y_TC_top),
    (xmax, y_TC_top), (xmax, ymin), (x_toe, ymin)
], alpha=0.08, facecolor="#888888", edgecolor="none")
ax.add_patch(behind)

# Slope face
ax.plot([x_toe, x_crest], [y_toe, y_crest],
        linewidth=3, color="#444444", label="Slope face")
# Upper ground from crest to TC
ax.plot([x_crest, x_TC], [y_crest, y_TC_top],
        linewidth=2.5, color="#444444")
# Tension crack
ax.plot([x_TC, x_TC], [y_TC_top, y_TC_base],
        linewidth=2.5, color="#0072B2", linestyle=":",
        label=f"Tension crack (z={z:.0f} m)")
# Sliding plane
ax.plot([x_toe, x_TC], [y_toe, y_TC_base],
        linewidth=3, color="#D55E00", linestyle="--",
        label=f"Sliding plane (ψ_p={psi_p:.0f}°)")

# Water in TC if present
if z_w > 0:
    water_top = y_TC_base + z_w
    ax.fill_between([x_TC - 0.3, x_TC + 0.3], y_TC_base, water_top,
                    color="#56B4E9", alpha=0.7)
    ax.plot([x_TC - 1.5, x_TC + 1.5], [water_top, water_top],
            color="#0072B2", linewidth=1, linestyle="-")
    ax.text(x_TC + 1.7, water_top, f"z_w={z_w:.1f} m",
            fontsize=9, color="#0072B2", va="center")

# --------- Force arrows (CORRECTED directions) ---------

# Centroid of wedge for placing W arrow
cx = sum(v[0] for v in wedge_vertices) / 4
cy = sum(v[1] for v in wedge_vertices) / 4

arrow_kw = dict(arrowstyle="-|>", linewidth=2.2, mutation_scale=18)

# Weight W: vertical down from centroid
W_len = 0.18 * H
ax.add_patch(FancyArrowPatch((cx, cy + W_len/2), (cx, cy - W_len/2),
                             color="#000000", **arrow_kw))
ax.text(cx + 0.3, cy + 0.05*H, "W",
        fontsize=14, fontweight="bold", color="#000000")

# Midpoint of sliding plane (where to anchor N, U, driving)
xm = 0.55 * x_TC
ym = (xm / x_TC) * y_TC_base   # on the sliding plane line

# Outward unit normal to sliding plane (out of rock, into wedge: up-and-to-the-LEFT)
# Plane direction (toe → TC): (cos ψ_p, sin ψ_p) approximately, but in 2D:
#   plane vector = (x_TC - 0, y_TC_base - 0), normalize
plane_dx = x_TC - x_toe
plane_dy = y_TC_base - y_toe
plane_len = math.hypot(plane_dx, plane_dy)
tx, ty = plane_dx/plane_len, plane_dy/plane_len
# Normal pointing upward (out of rock into wedge): (-ty, +tx) = (-sin ψ_p, +cos ψ_p)
nx, ny = -ty, tx

# Effective normal N' (out of rock into wedge): from base point along (nx, ny)
N_len = 0.15 * H
ax.add_patch(FancyArrowPatch((xm, ym), (xm + N_len*nx, ym + N_len*ny),
                             color="#0072B2", **arrow_kw))
ax.text(xm + (N_len + 0.5)*nx, ym + (N_len + 0.5)*ny, "N'",
        fontsize=13, fontweight="bold", color="#0072B2")

# Uplift U: same direction as N (out of rock into wedge from base)
# Place at toe-end of the plane to keep clear of N and Resisting
if z_w > 0 and wet["U (kN/m)"] > 1:
    xu = 0.18 * x_TC
    yu = (xu / x_TC) * y_TC_base
    U_len = 0.10 * H
    ax.add_patch(FancyArrowPatch((xu, yu), (xu + U_len*nx, yu + U_len*ny),
                                 color="#56B4E9", **arrow_kw))
    ax.text(xu + (U_len + 0.4)*nx - 0.6, yu + (U_len + 0.4)*ny, "U",
            fontsize=12, fontweight="bold", color="#0072B2")

# Driving force = W sin ψ_p along plane TOWARD THE TOE (down-left along plane)
# Place near the upper end of the plane, label OUTSIDE the wedge
D_len = 0.18 * H
xd_start = 0.78 * x_TC
yd_start = (xd_start / x_TC) * y_TC_base
# Offset start point slightly above the plane line so arrow sits in the wedge
xd_start += 1.0 * nx
yd_start += 1.0 * ny
ax.add_patch(FancyArrowPatch(
    (xd_start, yd_start),
    (xd_start - D_len*tx, yd_start - D_len*ty),
    color="#D55E00", **arrow_kw))
# Label OUTSIDE the wedge, to the upper-right of the upper plane end
ax.text(x_TC + 1.0, y_TC_base + 6,
        "Driving\n$W\\sin\\psi_p + V\\cos\\psi_p$",
        fontsize=10, color="#D55E00", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#D55E00", alpha=0.85))

# Resisting force, along plane AWAY FROM TOE (up-right along plane)
# Place near middle of plane, label BELOW the slope (outside the wedge)
R_len = 0.18 * H
xr_start = 0.42 * x_TC
yr_start = (xr_start / x_TC) * y_TC_base
# Offset slightly below the plane line
xr_start -= 1.0 * nx
yr_start -= 1.0 * ny
ax.add_patch(FancyArrowPatch(
    (xr_start, yr_start),
    (xr_start + R_len*tx, yr_start + R_len*ty),
    color="#009E73", **arrow_kw))
# Label below the toe area
ax.text(x_TC * 0.55, -0.18*H,
        "Resisting\n$cA + N'\\tan\\phi$",
        fontsize=10, color="#009E73", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#009E73", alpha=0.85))

# Tension-crack water force V: horizontal, into the wedge from the tension crack wall
# Points toward the slope face (toward -x), applied at mid-water
if z_w > 0 and wet["V (kN/m)"] > 1:
    V_len = 0.10 * H
    yv = y_TC_base + z_w/2
    xv = x_TC
    ax.add_patch(FancyArrowPatch((xv + V_len, yv), (xv, yv),
                                 color="#56B4E9", **arrow_kw))
    ax.text(xv + V_len + 0.5, yv, "V",
            fontsize=12, fontweight="bold", color="#0072B2", va="center")

# Anchor T: drawn from a point on slope face into the rock (positive ψ_T = below horizontal)
if T > 0:
    # Place on slope face at ~40% up to stay clear of W and N labels
    sx = 0.40 * x_crest
    sy = 0.40 * y_crest
    psi_T_r_local = math.radians(psi_T)
    # Anchor direction: drilled INTO the rock, below horizontal by ψ_T
    # Direction vector: (cos ψ_T, -sin ψ_T) for positive ψ_T below horizontal
    Td_x =  math.cos(psi_T_r_local)
    Td_y = -math.sin(psi_T_r_local)
    T_len_arrow = 0.18 * H
    ax.add_patch(FancyArrowPatch(
        (sx, sy), (sx + T_len_arrow*Td_x, sy + T_len_arrow*Td_y),
        color="#CC79A7", **arrow_kw))
    # Label well outside the wedge, in upper-right
    ax.text(x_crest * 1.15, y_crest * 0.55,
            f"T = {T:.0f} kN/m\n($\\psi_T$ = {psi_T:.0f}°)",
            fontsize=10, color="#CC79A7", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CC79A7", alpha=0.85))

# Angle annotations
arc_r = min(2.5, x_crest * 0.3)
theta = np.linspace(0, psi_f_r, 30)
ax.plot(arc_r*np.cos(theta), arc_r*np.sin(theta), color="#444444", linewidth=1)
ax.text(arc_r*1.3*math.cos(psi_f_r/2), arc_r*1.3*math.sin(psi_f_r/2),
        f"ψ_f={psi_f:.0f}°", fontsize=10, color="#444444")

theta2 = np.linspace(0, psi_p_r, 30)
arc_r2 = arc_r * 1.7
ax.plot(arc_r2*np.cos(theta2), arc_r2*np.sin(theta2),
        color="#D55E00", linewidth=1)
ax.text(arc_r2*1.15*math.cos(psi_p_r/2), arc_r2*1.15*math.sin(psi_p_r/2) - 0.4,
        f"ψ_p={psi_p:.0f}°", fontsize=10, color="#D55E00")

ax.set_aspect("equal")
ax.set_xlim(-3, xmax)
ax.set_ylim(ymin, H + 8)
ax.set_xlabel("Horizontal distance (m)")
ax.set_ylabel("Elevation (m)")
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_title("Planar failure schematic — Hoek & Bray (1981)")

st.pyplot(fig)

with st.expander("Explanation of the force arrows"):
    st.markdown("""
- **W** — wedge self-weight, vertical down, applied at centroid.
- **N'** — effective normal force on the sliding plane.  Points
  **outward from the rock mass into the wedge** (perpendicular to
  the plane, up-and-to-the-left for a slope dipping right).
- **U** — uplift due to water pressure on the base of the wedge.
  Same direction as N' (acts to reduce effective normal stress).
  Only present when z_w > 0.
- **V** — horizontal water force in the tension crack.  Points
  toward the slope face (drives sliding).
- **Driving** — net force component along the plane, toward the toe.
  Combines W sin ψ_p and V cos ψ_p, with anchor reduction T cos(ψ_p+ψ_T).
- **Resisting** — total shear strength along the plane, opposing driving.
  Cohesion contribution cA plus friction (N')tan φ.
- **T** — anchor tension applied to the slope face, drilled into the
  rock at angle ψ_T below horizontal.  Contributes to **both** the
  resisting force (via T cos(ψ_p+ψ_T)) and the effective normal
  stress (via T sin(ψ_p+ψ_T)) — increasing friction mobilisation.
""")

# =====================================================================
# Equations
# =====================================================================
st.subheader("Equations (Hoek & Bray, 1981, horizontal upper slope)")

st.latex(r"""
A = \frac{H - z}{\sin\psi_p}
\qquad\text{(base area per metre run)}
""")

st.latex(r"""
W = \tfrac{1}{2}\,\gamma\,H^{2}\!\left[
        \left(1 - \left(\tfrac{z}{H}\right)^{\!2}\right)\cot\psi_p
        - \cot\psi_f
    \right]
""")

st.latex(r"""
V = \tfrac{1}{2}\,\gamma_w\,z_w^{\,2}
\qquad
U = \tfrac{1}{2}\,\gamma_w\,z_w\,A
""")

st.latex(r"""
N' = W\cos\psi_p \;-\; U \;-\; V\sin\psi_p \;+\; T\sin(\psi_p + \psi_T)
""")

st.latex(r"""
\text{Driving} = W\sin\psi_p \;+\; V\cos\psi_p \;-\; T\cos(\psi_p + \psi_T)
""")

st.latex(r"""
\text{Resisting} = c\,A \;+\; N'\tan\phi
""")

st.latex(r"""
\boxed{\;\mathrm{FoS} = \frac{c\,A + N'\tan\phi}
                            {W\sin\psi_p + V\cos\psi_p - T\cos(\psi_p + \psi_T)}\;}
""")

st.markdown("""
where (sign convention as in the schematic):

- $H$ = slope height; $\\psi_f$ = slope face angle; $\\psi_p$ = sliding plane dip;
- $z$ = vertical depth of tension crack; $z_w$ = water depth in TC;
- $c$, $\\phi$ = cohesion and friction angle on the sliding plane;
- $\\gamma$, $\\gamma_w$ = rock and water unit weights;
- $T$ = anchor tension per metre run; $\\psi_T$ = anchor angle below horizontal
  (positive = drilled downward, drainage-friendly).
""")

# =====================================================================
# Results — dry and wet
# =====================================================================
st.subheader("Results")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Dry condition** (z_w = 0, φ = φ_dry)")
    st.metric("FoS (dry)", f"{dry['FoS']:.3f}")
    df_dry = pd.DataFrame({"Value": dry}).drop(["valid", "reason_inf"]).T
    st.dataframe(pd.DataFrame.from_dict(
        {k: v for k, v in dry.items() if k not in ("valid", "reason_inf")},
        orient="index", columns=["Value"]).style.format("{:.3f}"),
        use_container_width=True)

with col2:
    st.markdown(f"**Wet condition** (z_w = {z_w:.1f} m, φ = φ_wet)")
    st.metric("FoS (wet)", f"{wet['FoS']:.3f}")
    st.dataframe(pd.DataFrame.from_dict(
        {k: v for k, v in wet.items() if k not in ("valid", "reason_inf")},
        orient="index", columns=["Value"]).style.format("{:.3f}"),
        use_container_width=True)

# Stability interpretation
st.subheader("Stability interpretation")

def interp(F):
    if F == float("inf"): return "Locked (anchor exceeds driving)"
    if F < 1.0:  return "Unstable"
    if F < 1.2:  return "Marginal — design FoS not met"
    if F < 1.5:  return "Stable for short-term / low consequence"
    return "Stable for permanent / high consequence"

summary = pd.DataFrame([
    ["Dry", phi_dry, 0.0,   dry["FoS"], interp(dry["FoS"])],
    ["Wet", phi_wet, z_w,   wet["FoS"], interp(wet["FoS"])],
], columns=["Condition", "φ (°)", "z_w (m)", "FoS", "Interpretation"])
st.dataframe(summary.style.format({"FoS": "{:.3f}",
                                   "φ (°)": "{:.0f}",
                                   "z_w (m)": "{:.1f}"}),
             use_container_width=True)

# =====================================================================
# Sensitivity plots
# =====================================================================
st.subheader("Sensitivity plots")

tab1, tab2, tab3 = st.tabs(["FoS vs z_w", "FoS vs ψ_f", "FoS vs T"])

with tab1:
    zws = np.linspace(0, z, 25)
    F_dry_zw = [planar_fos(H, psi_f, psi_p, z, zw_, c, phi_dry,
                           gamma, gamma_w, T, psi_T)["FoS"] for zw_ in zws]
    F_wet_zw = [planar_fos(H, psi_f, psi_p, z, zw_, c, phi_wet,
                           gamma, gamma_w, T, psi_T)["FoS"] for zw_ in zws]
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(zws, F_dry_zw, label=f"φ = {phi_dry:.0f}° (dry friction)", linewidth=2)
    ax2.plot(zws, F_wet_zw, label=f"φ = {phi_wet:.0f}° (wet friction)", linewidth=2)
    ax2.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="FoS = 1.0")
    ax2.axhline(1.3, color="orange", linestyle=":", alpha=0.6, label="FoS = 1.3 (target)")
    ax2.set_xlabel("Tension crack water depth z_w (m)")
    ax2.set_ylabel("Factor of Safety")
    ax2.set_title(f"FoS vs water depth (TC depth z = {z:.0f} m)")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

with tab2:
    psifs = np.linspace(max(psi_p+1, 30), 85, 30)
    F_psif = []
    for pf in psifs:
        r = planar_fos(H, pf, psi_p, z, z_w, c, phi_wet, gamma, gamma_w, T, psi_T)
        F_psif.append(r["FoS"] if r["valid"] else float("nan"))
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(psifs, F_psif, linewidth=2, color="#D55E00")
    ax3.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="FoS = 1.0")
    ax3.axhline(1.3, color="orange", linestyle=":", alpha=0.6, label="FoS = 1.3 target")
    ax3.set_xlabel("Slope face angle ψ_f (deg)")
    ax3.set_ylabel("FoS (wet)")
    ax3.set_title("Effect of flattening the slope")
    ax3.legend(); ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

with tab3:
    Ts = np.linspace(0, max(T, 1) * 4, 30)
    F_T = [planar_fos(H, psi_f, psi_p, z, z_w, c, phi_wet,
                      gamma, gamma_w, T_, psi_T)["FoS"] for T_ in Ts]
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.plot(Ts, F_T, linewidth=2, color="#CC79A7")
    ax4.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="FoS = 1.0")
    ax4.axhline(1.3, color="orange", linestyle=":", alpha=0.6, label="FoS = 1.3 target")
    ax4.set_xlabel("Anchor tension T (kN/m)")
    ax4.set_ylabel("FoS (wet)")
    ax4.set_title(f"Effect of anchor tension at ψ_T = {psi_T:.0f}°")
    ax4.legend(); ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)

# =====================================================================
# Teaching notes
# =====================================================================
st.subheader("Teaching notes")
st.markdown("""
**Validation.** With $H = 60$ m, $\\psi_f = 60°$, $\\psi_p = 35°$, $z = 14$ m,
$z_w = 5$ m, $c = 25$ kPa, $\\phi = 37°$, $\\gamma = 26$ kN/m³, the formula gives
$\\mathrm{FoS} \\approx 1.094$, matching the Hoek & Bray (1981) textbook
example to three decimals.

**Common errors to avoid.**

1. **Don't use $r_u$ for planar failure.**  Skempton's pore-pressure ratio
   $r_u = u / (\\gamma h)$ is a slice-method parameter for circular Bishop
   or Janbu analysis.  For planar failure, water enters the equation via
   the tension-crack force $V$ and the base uplift $U$, both calculated
   from $z_w$ and $\\gamma_w$ directly.

2. **Don't multiply anchor force by bolt length.**  Bolt working load
   (kN) is a force, not a stress.  Length is used only to verify the
   bolt extends past the failure surface — it's a design check, not a
   force multiplier.

3. **Anchors contribute to $N'$ as well as resisting force.**  An
   anchor at angle $\\psi_T$ below horizontal increases the effective
   normal stress by $T \\sin(\\psi_p + \\psi_T)$, which mobilises
   additional friction.  Many simplified treatments forget the second
   contribution.

4. **Mind the anchor-angle convention.**  Hoek & Bray's optimum
   anchor angle is $\\psi_T = \\phi - \\psi_p$ (anchor inclined upward
   above horizontal when $\\phi < \\psi_p$).  This minimises the
   required anchor force.  In practice, anchors are often drilled
   at $\\psi_T = +10°$ to $+20°$ below horizontal for drainage, which
   is sub-optimal but operationally simpler.

**Use this app after** SMR (Week 10) and kinematic analysis (Week 11)
have identified the controlling failure plane.
""")
