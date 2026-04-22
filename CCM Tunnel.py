"""
RSE3010 — Convergence-Confinement Method (CCM) Interactive App
================================================================
Streamlit application for Mine Geotechnical Engineering, Monash University.

Reproduces the classic CCM four-panel figure (LDP, GRC, SCC, coupling)
and the Snowy 2.0 worked example with full FoS calculation.

Run with:    streamlit run ccm_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="CCM — RSE3010",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .stMetric { background-color: #f8f9fa; padding: 0.75rem; border-radius: 0.5rem;
                border-left: 3px solid #1F4E78; }
    .stMetric label { font-size: 0.85rem !important; color: #555 !important; }
    h1 { color: #1F4E78; border-bottom: 2px solid #1F4E78; padding-bottom: 0.3rem; }
    h2 { color: #2E5984; margin-top: 1.5rem; }
    h3 { color: #4A7BA7; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CORE CCM CALCULATIONS
# =============================================================================

def hoek_brown_params(GSI: float, mi: float, D: float) -> dict:
    """Compute Hoek-Brown rock mass parameters (m_b, s, a_HB)."""
    mb = mi * np.exp((GSI - 100) / (28 - 14 * D))
    s = np.exp((GSI - 100) / (9 - 3 * D))
    a_HB = 0.5 + (1/6) * (np.exp(-GSI/15) - np.exp(-20/3))
    return {"mb": mb, "s": s, "a_HB": a_HB}


def rock_mass_strength(sigma_ci: float, GSI: float, mi: float, D: float) -> dict:
    """Rock mass UCS from Hoek-Brown parameters."""
    hb = hoek_brown_params(GSI, mi, D)
    sigma_cm = sigma_ci * hb["s"] ** hb["a_HB"]
    return {**hb, "sigma_cm": sigma_cm}


def rock_mass_modulus(E_i: float, GSI: float, D: float) -> float:
    """Hoek-Diederichs (2006) rock mass deformation modulus."""
    return E_i * (0.02 + (1 - D/2) / (1 + np.exp((60 + 15*D - GSI) / 11)))


def mc_parameters(phi_deg: float, c: float) -> dict:
    """Mohr-Coulomb derived quantities."""
    phi = np.radians(phi_deg)
    k = (1 + np.sin(phi)) / (1 - np.sin(phi))
    sigma_cm_MC = 2 * c * np.cos(phi) / (1 - np.sin(phi))
    return {"k": k, "sigma_cm_MC": sigma_cm_MC}


def critical_pressure(p0: float, sigma_cm_MC: float, k: float) -> float:
    """Critical internal pressure below which yielding initiates."""
    return (2 * p0 - sigma_cm_MC) / (1 + k)


def plastic_radius(a: float, p0: float, pi: float, k: float, sigma_cm_MC: float) -> float:
    """Plastic zone radius at internal pressure p_i."""
    num = 2 * (p0 * (k - 1) + sigma_cm_MC)
    den = (1 + k) * (pi * (k - 1) + sigma_cm_MC)
    return a * (num / den) ** (1 / (k - 1))


def grc_displacement(pi: float, p0: float, pcr: float, a: float, nu: float,
                     Em: float, k: float, sigma_cm_MC: float) -> float:
    """Ground Reaction Curve: radial displacement at pressure p_i (mm)."""
    if pi >= pcr:
        # Elastic branch
        return (1 + nu) * (p0 - pi) * a / Em * 1000
    # Plastic branch
    ur_pcr = (1 + nu) * (p0 - pcr) * a / Em * 1000
    Rp = plastic_radius(a, p0, pi, k, sigma_cm_MC)
    return ur_pcr * (Rp / a) ** 2


def ldp_vlachopoulos(Xstar: float, Rstar: float) -> float:
    """Vlachopoulos & Diederichs (2009) LDP. Returns u/u_max."""
    if Xstar <= 0:
        return (1/3) * np.exp(2 * Xstar - 0.15 * Rstar)
    return 1 - (1 - (1/3) * np.exp(-0.15 * Rstar)) * np.exp(-3 * Xstar / Rstar)


def ldp_panet(Xstar: float, alpha: float = 0.75) -> float:
    """Panet (1995) elastic LDP. Returns u/u_max."""
    if Xstar <= 0:
        return (1 - alpha) * np.exp(1.5 * Xstar)
    return 1 - alpha * np.exp(-1.5 * Xstar)


def scc_properties(sigma_ci_lining: float, Ec: float, nu_c: float,
                   t: float, a: float) -> dict:
    """Concrete lining Support Characteristic Curve properties."""
    psm = sigma_ci_lining * t / a
    ks = Ec * t / ((1 - nu_c**2) * a**2)
    usm = psm / ks * 1000  # mm
    return {"psm": psm, "ks": ks, "usm": usm}


def solve_fos(p0, pcr, a, nu, Em, k, sigma_cm_MC, us0, ks, psm):
    """Solve GRC-SCC intersection for p_eq and FoS. Returns dict."""
    def diff(pi):
        u_grc = grc_displacement(pi, p0, pcr, a, nu, Em, k, sigma_cm_MC)
        u_scc = us0 + pi / ks * 1000  # mm
        return u_grc - u_scc

    # Check saturation: if at u = us0 + usm, GRC is still above, SCC saturates
    u_scc_max = us0 + psm / ks * 1000
    # Find pressure on GRC at u_scc_max (invert numerically)
    try:
        # If GRC(pi=psm) displacement > u_scc_max, the SCC reaches psm before intersecting
        u_at_psm = grc_displacement(psm, p0, pcr, a, nu, Em, k, sigma_cm_MC)
        if u_at_psm > u_scc_max:
            # SCC saturates before intersection
            return {"p_eq": psm, "u_eq": u_scc_max, "FoS": 1.0, "saturated": True}

        # Otherwise solve for intersection in (small positive, psm)
        p_eq = brentq(diff, 0.001, psm * 0.999, xtol=1e-6)
        u_eq = us0 + p_eq / ks * 1000
        return {"p_eq": p_eq, "u_eq": u_eq, "FoS": psm / p_eq, "saturated": False}
    except Exception as e:
        return {"p_eq": None, "u_eq": None, "FoS": None, "saturated": False, "error": str(e)}


def full_ccm_analysis(params: dict) -> dict:
    """Run full CCM workflow and return all intermediate results."""
    p = params
    p0 = p["gamma"] * p["z"] / 1000

    # Hoek-Brown & rock mass
    rm = rock_mass_strength(p["sigma_ci"], p["GSI"], p["mi"], p["D"])
    Em = rock_mass_modulus(p["E_i"], p["GSI"], p["D"])

    # Mohr-Coulomb
    mc = mc_parameters(p["phi_deg"], p["c"])
    pcr = critical_pressure(p0, mc["sigma_cm_MC"], mc["k"])
    is_plastic = p0 > pcr

    # Tunnel response (unsupported)
    if is_plastic:
        Rp_unsup = plastic_radius(p["a"], p0, 0, mc["k"], mc["sigma_cm_MC"])
    else:
        Rp_unsup = p["a"]
    Rstar = Rp_unsup / p["a"]

    # u_max
    ur_pcr = (1 + p["nu"]) * (p0 - pcr) * p["a"] / Em * 1000 if is_plastic else 0
    u_max = (1 + p["nu"]) * p0 * p["a"] / Em * 1000 if not is_plastic \
            else ur_pcr * Rstar**2

    # LDP
    Xstar = p["L"] / p["a"]
    u_star = ldp_vlachopoulos(Xstar, Rstar)
    us0 = u_star * u_max

    # Support
    scc = scc_properties(p["sigma_ci_lining"], p["Ec"], p["nu_c"], p["t"], p["a"])

    # FoS
    fos_result = solve_fos(p0, pcr, p["a"], p["nu"], Em, mc["k"], mc["sigma_cm_MC"],
                           us0, scc["ks"], scc["psm"])

    return {
        "p0": p0, "Em": Em, "pcr": pcr, "is_plastic": is_plastic,
        "Rp_unsup": Rp_unsup, "Rstar": Rstar, "ur_pcr": ur_pcr, "u_max": u_max,
        "Xstar": Xstar, "u_star": u_star, "us0": us0,
        **rm, **mc, **scc, **fos_result,
    }


# =============================================================================
# PLOTS
# =============================================================================

PALETTE = {
    "grc": "#1F4E78",
    "scc": "#C00000",
    "ldp": "#2E7D32",
    "face": "#888",
    "ref": "#aaa",
    "highlight": "#FF6B35",
}


def plot_four_panel_ccm(result: dict, params: dict, Xstar_cross: float = -1.2):
    """Classic four-panel figure: LDP (top-left), coupling line (top-right),
    tunnel schematic (bottom-left), GRC (bottom-right). Cross-section marker
    at Xstar_cross highlights the coupling."""

    u_max = result["u_max"]
    Rstar = result["Rstar"]
    p0 = result["p0"]

    # Evaluate LDP on a range
    # Note: convention here: X > 0 behind face, X < 0 ahead
    # The figure shows X negative going right (ahead). We'll use normalised
    # distance d = X/D where D = 2a (tunnel diameter), following the figure.
    d_range = np.linspace(-3, 5, 200)  # -3 behind to 5 ahead (figure convention)
    # Convert figure convention to V&D: their X* = X/a where X is distance behind
    # In the figure, larger "normalised distance" is ahead of face (NOT behind)
    # So V&D X* = -d * 2 (since D = 2a, and we flip the sign)
    u_norm = np.array([ldp_vlachopoulos(-d * 2, Rstar) * 100 for d in d_range])

    # Coupling at the cross-section
    u_cs = ldp_vlachopoulos(-Xstar_cross * 2, Rstar) * 100  # % of u_max

    # GRC (plastic case)
    pi_range = np.linspace(0, p0, 200)
    u_grc = np.array([grc_displacement(pi, p0, result["pcr"], params["a"],
                                       params["nu"], result["Em"], result["k"],
                                       result["sigma_cm_MC"]) for pi in pi_range])
    u_grc_norm = u_grc / u_max * 100
    pi_norm = pi_range / p0 * 100
    # GRC elastic line (for reference)
    u_grc_elastic = (1 + params["nu"]) * (p0 - pi_range) * params["a"] / result["Em"] * 1000
    u_grc_elastic_norm = u_grc_elastic / u_max * 100

    # Pressure at cross-section from GRC at u_cs
    from scipy.interpolate import interp1d
    grc_interp = interp1d(u_grc_norm, pi_norm, bounds_error=False, fill_value="extrapolate")
    p_cs = float(grc_interp(u_cs))

    # --- Build subplot figure ---
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Longitudinal Displacement Profile (LDP)",
                        "u — u coupling (45° line)",
                        "Tunnel schematic",
                        "Ground Reaction Curve (GRC)"),
        vertical_spacing=0.12, horizontal_spacing=0.1,
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
    )

    # --- Panel 1: LDP (top-left) ---
    fig.add_trace(go.Scatter(x=d_range, y=u_norm, mode="lines",
                             line=dict(color=PALETTE["ldp"], width=3),
                             name="LDP", showlegend=False), row=1, col=1)
    # Face marker (vertical line at d=0)
    fig.add_vline(x=0, line=dict(color=PALETTE["face"], width=1.5, dash="solid"),
                  row=1, col=1)
    # Cross-section dashed line
    fig.add_vline(x=Xstar_cross, line=dict(color=PALETTE["highlight"], width=1.5, dash="dash"),
                  row=1, col=1)
    fig.add_hline(y=u_cs, line=dict(color=PALETTE["highlight"], width=1.5, dash="dash"),
                  row=1, col=1)
    # Highlight point
    fig.add_trace(go.Scatter(x=[Xstar_cross], y=[u_cs], mode="markers",
                             marker=dict(size=12, color=PALETTE["highlight"],
                                         line=dict(color="white", width=2)),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_annotation(x=Xstar_cross, y=u_cs + 8, text=f"Section A-A<br>u = {u_cs:.0f}% u_max",
                       showarrow=False, font=dict(size=10, color=PALETTE["highlight"]),
                       row=1, col=1)

    # --- Panel 2: 45° coupling line (top-right) ---
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                             line=dict(color="#555", width=2),
                             showlegend=False), row=1, col=2)
    # Coupling arrows: from LDP y-value horizontally to 45° line, then down to GRC
    fig.add_hline(y=u_cs, line=dict(color=PALETTE["highlight"], width=1.5, dash="dash"),
                  row=1, col=2)
    fig.add_vline(x=u_cs, line=dict(color=PALETTE["highlight"], width=1.5, dash="dash"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=[u_cs], y=[u_cs], mode="markers",
                             marker=dict(size=12, color=PALETTE["highlight"],
                                         line=dict(color="white", width=2)),
                             showlegend=False, hoverinfo="skip"), row=1, col=2)

    # --- Panel 3: Tunnel schematic (bottom-left) ---
    # Draw tunnel rectangle: face at d=0, tunnel extends to d < 0 (behind face)
    # (No, in the figure, tunnel extends d > 0 being the excavated portion behind face,
    #  and d < 0 is ahead of face. We'll match figure orientation.)
    # Rock above and below
    fig.add_shape(type="rect", x0=-3, y0=0.6, x1=5, y1=1.0,
                  fillcolor="#d4d4d4", line=dict(width=0), row=2, col=1)
    fig.add_shape(type="rect", x0=-3, y0=0, x1=5, y1=0.4,
                  fillcolor="#d4d4d4", line=dict(width=0), row=2, col=1)
    # Tunnel void (to the right of face, i.e., d > 0... wait figure shows excavated
    # portion on the left (d > 0 region in our range) — actually the arrow shows
    # "Tunnel advance" to the right, excavated behind it. Let's match that:
    # excavated portion: d from 0 to 5 (left side, behind face)
    # Actually from figure: face is at d = 0, excavated region is to the right of face
    # (where d is positive in the figure's left-going axis). Given the axis runs
    # high to low left-to-right, d_positive is on the LEFT of the plot.
    # I'll mirror the schematic so face is at d=0 with tunnel behind (d>0 on left).
    fig.add_shape(type="rect", x0=0, y0=0.4, x1=5, y1=0.6,
                  fillcolor="white", line=dict(color="black", width=2), row=2, col=1)
    # Face
    fig.add_shape(type="line", x0=0, y0=0.4, x1=0, y1=0.6,
                  line=dict(color="black", width=3), row=2, col=1)
    # Arrow showing advance
    fig.add_annotation(x=-1, y=0.83, ax=-2, ay=0.83, xref="x3", yref="y3",
                       axref="x3", ayref="y3", text="", showarrow=True,
                       arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor="black")
    fig.add_annotation(x=-1.5, y=0.9, text="Tunnel advance", showarrow=False,
                       font=dict(size=11), row=2, col=1)
    # Cross-section A-A line
    fig.add_shape(type="line", x0=Xstar_cross, y0=0.25, x1=Xstar_cross, y1=0.75,
                  line=dict(color=PALETTE["highlight"], width=2, dash="dash"),
                  row=2, col=1)
    fig.add_annotation(x=Xstar_cross, y=0.85, text=f"A–A<br>({Xstar_cross}D)",
                       showarrow=False, font=dict(size=10, color=PALETTE["highlight"]),
                       row=2, col=1)

    # --- Panel 4: GRC (bottom-right) ---
    fig.add_trace(go.Scatter(x=u_grc_norm, y=pi_norm, mode="lines",
                             line=dict(color=PALETTE["grc"], width=3),
                             name="GRC", showlegend=False), row=2, col=2)
    # Elastic branch (dashed) for reference
    mask = u_grc_elastic_norm <= max(u_grc_norm) * 1.1
    fig.add_trace(go.Scatter(x=u_grc_elastic_norm[mask], y=pi_norm[mask],
                             mode="lines",
                             line=dict(color=PALETTE["grc"], width=1.5, dash="dash"),
                             showlegend=False, hoverinfo="skip"), row=2, col=2)
    # Cross-section markers
    fig.add_hline(y=p_cs, line=dict(color=PALETTE["highlight"], width=1.5, dash="dash"),
                  row=2, col=2)
    fig.add_vline(x=u_cs, line=dict(color=PALETTE["highlight"], width=1.5, dash="dash"),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=[u_cs], y=[p_cs], mode="markers",
                             marker=dict(size=12, color=PALETTE["highlight"],
                                         line=dict(color="white", width=2)),
                             showlegend=False, hoverinfo="skip"), row=2, col=2)
    fig.add_annotation(x=u_cs + 3, y=p_cs + 5,
                       text=f"p/p₀ = {p_cs:.0f}%",
                       showarrow=False, font=dict(size=10, color=PALETTE["highlight"]),
                       row=2, col=2)

    # --- Axis labels ---
    fig.update_xaxes(title_text="Normalised distance X/D (positive = ahead of face)",
                     autorange="reversed", row=1, col=1, range=[5, -3])
    fig.update_yaxes(title_text="u / u_max  (%)", row=1, col=1, range=[0, 105])

    fig.update_xaxes(title_text="u / u_max  (%)", row=1, col=2, range=[0, 105])
    fig.update_yaxes(title_text="u / u_max  (%)", row=1, col=2, range=[0, 105])

    fig.update_xaxes(title_text="Normalised distance X/D",
                     autorange="reversed", row=2, col=1, range=[5, -3],
                     showgrid=False, zeroline=False)
    fig.update_yaxes(range=[0, 1], row=2, col=1, showgrid=False, zeroline=False,
                     showticklabels=False)

    fig.update_xaxes(title_text="u / u_max  (%)", row=2, col=2, range=[0, 105])
    fig.update_yaxes(title_text="p_i / p₀  (%)", row=2, col=2, range=[0, 105])

    fig.update_layout(
        height=800, showlegend=False,
        margin=dict(l=60, r=40, t=60, b=60),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(family="Georgia, serif", size=11),
    )
    for axis in fig.layout:
        if axis.startswith(("xaxis", "yaxis")):
            fig.layout[axis].update(gridcolor="#e5e5e5", linecolor="#333",
                                    mirror=True, showline=True)
    return fig


def plot_grc_scc(result, params):
    """GRC + SCC intersection plot for the Snowy 2.0 example."""
    p0 = result["p0"]
    u_max = result["u_max"]
    us0 = result["us0"]
    usm = result["usm"]
    psm = result["psm"]
    ks = result["ks"]

    # GRC points
    pi_range = np.linspace(0, p0, 200)
    u_grc = np.array([grc_displacement(pi, p0, result["pcr"], params["a"],
                                       params["nu"], result["Em"], result["k"],
                                       result["sigma_cm_MC"]) for pi in pi_range])

    # SCC: three segments
    u_scc = [us0, us0 + usm, max(u_max, us0 + usm) + 20]
    p_scc = [0, psm, psm]

    fig = go.Figure()

    # GRC
    fig.add_trace(go.Scatter(x=u_grc, y=pi_range, mode="lines",
                             line=dict(color=PALETTE["grc"], width=3),
                             name="GRC (ground reaction)"))
    # p_cr horizontal reference
    fig.add_hline(y=result["pcr"],
                  line=dict(color=PALETTE["grc"], width=1, dash="dot"),
                  annotation_text=f"p_cr = {result['pcr']:.2f} MPa",
                  annotation_position="right")

    # SCC
    fig.add_trace(go.Scatter(x=u_scc, y=p_scc, mode="lines+markers",
                             line=dict(color=PALETTE["scc"], width=3),
                             marker=dict(size=8),
                             name="SCC (support characteristic)"))

    # Intersection marker
    if result["p_eq"] is not None:
        fig.add_trace(go.Scatter(
            x=[result["u_eq"]], y=[result["p_eq"]], mode="markers",
            marker=dict(size=16, color=PALETTE["highlight"], symbol="star",
                        line=dict(color="white", width=2)),
            name=f"Equilibrium (FoS = {result['FoS']:.2f})"))
        fig.add_annotation(
            x=result["u_eq"], y=result["p_eq"],
            text=f"<b>p_eq = {result['p_eq']:.2f} MPa<br>u_eq = {result['u_eq']:.1f} mm</b>",
            showarrow=True, arrowhead=2, ax=40, ay=-40,
            font=dict(size=11, color=PALETTE["highlight"]),
            bgcolor="white", bordercolor=PALETTE["highlight"], borderwidth=1)

    # us0 marker
    fig.add_vline(x=us0, line=dict(color=PALETTE["scc"], width=1, dash="dot"),
                  annotation_text=f"u_s0 = {us0:.1f} mm", annotation_position="top")

    fig.update_layout(
        title=dict(text="<b>GRC–SCC Intersection</b>", font=dict(size=16)),
        xaxis_title="Radial displacement, u_r (mm)",
        yaxis_title="Internal pressure, p_i (MPa)",
        height=500,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#ddd", borderwidth=1),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(family="Georgia, serif", size=12),
    )
    fig.update_xaxes(gridcolor="#e5e5e5", linecolor="#333", mirror=True, showline=True,
                     range=[0, max(u_max, us0 + usm) * 1.15])
    fig.update_yaxes(gridcolor="#e5e5e5", linecolor="#333", mirror=True, showline=True,
                     range=[0, p0 * 1.05])
    return fig


def plot_envelope(result, params):
    """H-B vs MC envelope comparison."""
    sigma_ci = params["sigma_ci"]
    mb, s, a_HB = result["mb"], result["s"], result["a_HB"]
    phi_deg, c = params["phi_deg"], params["c"]

    sig3_max = params["sigma_3_max"]
    sig3 = np.linspace(0, sig3_max, 100)
    sig1_HB = sig3 + sigma_ci * (mb * sig3 / sigma_ci + s) ** a_HB
    phi = np.radians(phi_deg)
    k = (1 + np.sin(phi)) / (1 - np.sin(phi))
    sig1_MC = 2 * c * np.cos(phi) / (1 - np.sin(phi)) + k * sig3

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sig3, y=sig1_HB, mode="lines",
                             line=dict(color=PALETTE["grc"], width=3),
                             name="Hoek-Brown envelope"))
    fig.add_trace(go.Scatter(x=sig3, y=sig1_MC, mode="lines",
                             line=dict(color=PALETTE["scc"], width=2.5, dash="dash"),
                             name=f"MC fit (φ={phi_deg:.1f}°, c={c:.2f} MPa)"))
    fig.update_layout(
        title=dict(text="<b>Failure Envelopes</b>", font=dict(size=16)),
        xaxis_title="σ₃ (MPa)", yaxis_title="σ₁ (MPa)",
        height=400,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98,
                    bgcolor="rgba(255,255,255,0.9)"),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(family="Georgia, serif", size=11),
    )
    fig.update_xaxes(gridcolor="#e5e5e5", linecolor="#333", mirror=True, showline=True)
    fig.update_yaxes(gridcolor="#e5e5e5", linecolor="#333", mirror=True, showline=True)
    return fig


# =============================================================================
# MAIN APP
# =============================================================================

st.title("⛰️ Convergence-Confinement Method (CCM)")
st.markdown(
    "**RSE3010 — Mine Geotechnical Engineering | Monash University** \n"
    "*Interactive tool for ground-support interaction analysis in circular tunnels.*"
)

# ---- Sidebar: Inputs ----
st.sidebar.header("📐 Project Inputs")

with st.sidebar.expander("**Rock Mass**", expanded=True):
    sigma_ci = st.number_input("σ_ci — intact UCS (MPa)", 1.0, 500.0, 72.0, 1.0)
    E_i = st.number_input("E_i — intact modulus (GPa)", 0.1, 100.0, 26.0, 0.5) * 1000
    mi = st.number_input("m_i (Hoek-Brown)", 1.0, 40.0, 7.0, 0.5,
                         help="7 = siltstone, 17 = sandstone")
    GSI = st.slider("GSI (Geological Strength Index)", 10, 100, 60, 5)
    D = st.slider("Disturbance factor D", 0.0, 1.0, 0.0, 0.1)
    nu = st.number_input("Poisson's ratio ν", 0.0, 0.5, 0.22, 0.01)

with st.sidebar.expander("**Tunnel Geometry & Stress**", expanded=True):
    gamma = st.number_input("γ — unit weight (kN/m³)", 15.0, 35.0, 26.0, 0.5)
    z = st.number_input("z — depth (m)", 10.0, 2000.0, 650.0, 10.0)
    a = st.number_input("a — tunnel radius (m)", 0.5, 15.0, 4.75, 0.25)
    L = st.number_input("L — support distance from face (m)", 0.0, 30.0, 3.0, 0.5)

with st.sidebar.expander("**MC Parameters (from RocData fit)**", expanded=False):
    st.caption("Fit the H-B envelope in RocData and paste φ, c here. "
               "See 'Envelopes' tab for the visual fit check.")
    phi_deg = st.number_input("φ — friction angle (°)", 5.0, 60.0, 31.5, 0.5)
    c = st.number_input("c — cohesion (MPa)", 0.0, 20.0, 2.85, 0.1)
    sigma_3_max = st.number_input("σ₃,max for envelope plot (MPa)",
                                  1.0, 100.0, 17.0, 1.0,
                                  help="Upper bound of σ₃ for the H-B→MC fit range.")

with st.sidebar.expander("**Concrete Lining**", expanded=True):
    sigma_ci_lining = st.number_input("σ_ci,lining (MPa)", 20.0, 200.0, 60.0, 5.0)
    Ec = st.number_input("E_c (GPa)", 10.0, 80.0, 40.0, 1.0) * 1000
    nu_c = st.number_input("ν_c", 0.0, 0.5, 0.25, 0.01)
    t = st.number_input("t — lining thickness (m)", 0.1, 2.0, 0.5, 0.05)

# Compile all parameters
params = dict(sigma_ci=sigma_ci, E_i=E_i, mi=mi, GSI=GSI, D=D, nu=nu,
              gamma=gamma, z=z, a=a, L=L, phi_deg=phi_deg, c=c,
              sigma_3_max=sigma_3_max,
              sigma_ci_lining=sigma_ci_lining, Ec=Ec, nu_c=nu_c, t=t)

# Run analysis
result = full_ccm_analysis(params)

# ---- Main area: tabs ----
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Results Summary", "📈 GRC–SCC", "🎨 Classic CCM Figure",
    "🧭 Envelopes", "📋 Workflow",
])

# --- Tab 1: Results ---
with tab1:
    st.header("Calculated Results")

    # FoS verdict at top
    if result["FoS"] is not None:
        fos = result["FoS"]
        if fos >= 2.0:
            verdict_color = "#2E7D32"; verdict = "✓ Adequate"
        elif fos >= 1.5:
            verdict_color = "#F57C00"; verdict = "⚠ Marginal"
        else:
            verdict_color = "#C62828"; verdict = "✗ Inadequate"
        st.markdown(
            f"""<div style='background:{verdict_color}15; border-left:4px solid {verdict_color};
                padding:1rem 1.25rem; border-radius:4px; margin-bottom:1rem;'>
            <h3 style='color:{verdict_color}; margin:0;'>
            FoS = {fos:.2f} &nbsp; — &nbsp; {verdict}
            </h3>
            <p style='margin:0.25rem 0 0 0; color:#333; font-size:0.95rem;'>
            p_eq = {result['p_eq']:.2f} MPa, u_eq = {result['u_eq']:.1f} mm,
            p_sm = {result['psm']:.2f} MPa
            </p></div>""",
            unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Rock Mass")
        st.metric("p₀ (in-situ stress)", f"{result['p0']:.2f} MPa")
        st.metric("m_b", f"{result['mb']:.3f}")
        st.metric("s", f"{result['s']:.5f}")
        st.metric("a_HB", f"{result['a_HB']:.3f}")
        st.metric("σ_cm (HB)", f"{result['sigma_cm']:.2f} MPa")
        st.metric("E_m (rock mass)", f"{result['Em']/1000:.2f} GPa")

    with col2:
        st.subheader("Elastoplastic Response")
        st.metric("σ_cm,MC", f"{result['sigma_cm_MC']:.2f} MPa")
        st.metric("k = (1+sinφ)/(1-sinφ)", f"{result['k']:.2f}")
        st.metric("p_cr (critical)", f"{result['pcr']:.2f} MPa")
        st.metric("Response", "Plastic" if result["is_plastic"] else "Elastic")
        st.metric("R_p (unsupported)", f"{result['Rp_unsup']:.2f} m")
        st.metric("R_p / a", f"{result['Rstar']:.2f}")

    with col3:
        st.subheader("Convergence & Support")
        st.metric("u_max (unsupported)", f"{result['u_max']:.1f} mm")
        st.metric("X* = L/a", f"{result['Xstar']:.3f}")
        st.metric("u* (V&D LDP)", f"{result['u_star']:.3f}")
        st.metric("u_s0 (at installation)", f"{result['us0']:.1f} mm")
        st.metric("p_sm (lining capacity)", f"{result['psm']:.2f} MPa")
        st.metric("k_s (lining stiffness)", f"{result['ks']:.0f} MPa/m")
        st.metric("u_sm (elastic capacity)", f"{result['usm']:.2f} mm")

# --- Tab 2: GRC-SCC ---
with tab2:
    st.header("GRC–SCC Intersection Analysis")
    st.plotly_chart(plot_grc_scc(result, params), width="stretch")

    if result["saturated"]:
        st.error(
            "⚠️ **Support saturated.** The SCC reached p_sm before intersecting the GRC. "
            "The lining is at its capacity and the FoS is effectively 1.0. "
            "Consider yielding support or reducing the stand-off distance L."
        )
    elif result["FoS"] is not None:
        st.info(
            f"**Equilibrium state:** the GRC and the SCC intersect at "
            f"p_eq = {result['p_eq']:.3f} MPa and u_eq = {result['u_eq']:.2f} mm. "
            f"The post-installation displacement is {result['u_eq'] - result['us0']:.2f} mm, "
            f"well within the lining's elastic capacity of {result['usm']:.2f} mm. "
            f"FoS = p_sm / p_eq = {result['FoS']:.2f}."
        )

# --- Tab 3: Classic four-panel ---
with tab3:
    st.header("Classic CCM Four-Panel Figure")
    st.markdown(
        "This is the canonical visualisation coupling LDP and GRC through the "
        "cross-section position. Move the slider to reposition the reference "
        "cross-section A–A and see how u_r and p_i change along the tunnel."
    )
    Xstar_cs = st.slider(
        "Cross-section position X/D (negative = ahead of face, positive = behind)",
        min_value=-3.0, max_value=5.0, value=-1.2, step=0.1)
    st.plotly_chart(plot_four_panel_ccm(result, params, Xstar_cs),
                    width="stretch")

# --- Tab 4: Envelopes ---
with tab4:
    st.header("H-B vs MC Envelope Comparison")
    st.markdown(
        "Use this plot to check the quality of the H-B → MC fit. Adjust φ and c "
        "in the sidebar until the MC line is a good tangent fit to the H-B curve "
        "over the stress range of interest (typically 0 to p₀ for deep tunnels)."
    )
    st.plotly_chart(plot_envelope(result, params), width="stretch")

# --- Tab 5: Workflow ---
with tab5:
    st.header("Calculation Workflow")
    st.markdown("""
The CCM analysis in this app follows nine steps. All formulas are visible in 
the source code (`ccm_app.py`) and reproducible in a spreadsheet.

### Step 1 — In-situ stress
$$p_0 = \\gamma z$$

### Step 2 — Hoek-Brown parameters
$$m_b = m_i \\, e^{(GSI-100)/(28-14D)}, \\quad
  s = e^{(GSI-100)/(9-3D)}, \\quad
  a_{HB} = \\tfrac{1}{2} + \\tfrac{1}{6}\\big(e^{-GSI/15} - e^{-20/3}\\big)$$

### Step 3 — Rock mass strength & modulus
$$\\sigma_{cm} = \\sigma_{ci} \\cdot s^{a_{HB}}, \\quad
  E_m = E_i \\left[0.02 + \\frac{1-D/2}{1+e^{(60+15D-GSI)/11}}\\right]$$

### Step 4 — Mohr-Coulomb equivalent (from RocData)
$$k = \\frac{1+\\sin\\phi}{1-\\sin\\phi}, \\quad 
  \\sigma_{cm,MC} = \\frac{2c\\cos\\phi}{1-\\sin\\phi}$$

### Step 5 — Critical pressure
$$p_{cr} = \\frac{2p_0 - \\sigma_{cm,MC}}{1+k}$$

### Step 6 — Plastic zone radius (unsupported)
$$R_p = a \\left[\\frac{2(p_0(k-1)+\\sigma_{cm,MC})}{(1+k)\\,\\sigma_{cm,MC}}\\right]^{1/(k-1)}$$

### Step 7 — Maximum convergence
$$u_r(p_{cr}) = \\frac{(1+\\nu)(p_0-p_{cr})\\,a}{E_m}, \\quad
  u_{\\max} \\approx u_r(p_{cr}) \\left(\\frac{R_p}{a}\\right)^2$$

### Step 8 — LDP at installation (Vlachopoulos & Diederichs 2009, behind face)
$$u^*(X^*) = 1 - \\left[1 - \\tfrac{1}{3}e^{-0.15 R^*}\\right] e^{-3X^*/R^*}, \\quad
  u_{s0} = u^* \\cdot u_{\\max}$$

### Step 9 — Support Characteristic Curve
$$p_{sm} = \\frac{\\sigma_{ci,lining} \\cdot t}{a}, \\quad
  k_s = \\frac{E_c \\cdot t}{(1-\\nu_c^2)\\,a^2}, \\quad
  u_{sm} = \\frac{p_{sm}}{k_s}$$

### Step 10 — FoS via GRC–SCC intersection
Solve $u_{GRC}(p_{eq}) = u_{s0} + p_{eq}/k_s$ for $p_{eq}$ (numerical root-find), 
then $\\mathrm{FoS} = p_{sm} / p_{eq}$.
""")

    st.subheader("Current analysis — numerical values")
    df = pd.DataFrame({
        "Quantity": ["p_0", "m_b", "s", "a_HB", "σ_cm", "E_m", "k", "σ_cm,MC",
                     "p_cr", "R_p", "R_p/a", "u_r(p_cr)", "u_max", "X*",
                     "u*", "u_s0", "p_sm", "k_s", "u_sm",
                     "p_eq", "u_eq", "FoS"],
        "Value": [f"{result['p0']:.2f} MPa", f"{result['mb']:.3f}",
                  f"{result['s']:.5f}", f"{result['a_HB']:.3f}",
                  f"{result['sigma_cm']:.2f} MPa",
                  f"{result['Em']/1000:.2f} GPa",
                  f"{result['k']:.2f}",
                  f"{result['sigma_cm_MC']:.2f} MPa",
                  f"{result['pcr']:.2f} MPa",
                  f"{result['Rp_unsup']:.2f} m", f"{result['Rstar']:.2f}",
                  f"{result['ur_pcr']:.2f} mm", f"{result['u_max']:.2f} mm",
                  f"{result['Xstar']:.3f}", f"{result['u_star']:.3f}",
                  f"{result['us0']:.2f} mm", f"{result['psm']:.2f} MPa",
                  f"{result['ks']:.0f} MPa/m", f"{result['usm']:.2f} mm",
                  f"{result['p_eq']:.3f} MPa" if result["p_eq"] else "—",
                  f"{result['u_eq']:.2f} mm" if result["u_eq"] else "—",
                  f"{result['FoS']:.2f}" if result["FoS"] else "—"],
    })
    st.dataframe(df, width="stretch", hide_index=True)

# Footer
st.markdown("---")
st.caption(
    "RSE3010 Mine Geotechnical Engineering · Semester 1, 2026 · "
    "Monash University · CCM framework based on Hoek (1998), Panet (1995), "
    "Vlachopoulos & Diederichs (2009)."
)
