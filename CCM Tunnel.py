"""
RSE3010 — Convergence-Confinement Method (CCM) Interactive App
================================================================
RocSupport-style interactive tool for Mine Geotechnical Engineering,
Monash University.

Features (mirrors RocSupport workflow):
- Multiple GRC solution methods: Duncan-Fama (MC), Carranza-Torres (HB, 2004),
  Vrakas & Anagnostou (MC, 2014)
- Multiple LDP methods: Panet (1995), Vlachopoulos & Diederichs (2009),
  Hoek (2002)
- Support library: Steel Sets, Rockbolts, Shotcrete, Custom (combined support)
- Tunnel Section View with plastic zone visualisation
- H-B to MC fit visualisation (Hoek 2002)
- Automatic FoS via GRC-SCC intersection

Run with:    streamlit run ccm_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
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
    section[data-testid="stSidebar"] { background-color: #f7f9fc; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CORE CALCULATIONS — HOEK-BROWN & ROCK MASS
# =============================================================================

def hoek_brown_params(GSI, mi, D):
    mb = mi * np.exp((GSI - 100) / (28 - 14 * D))
    s = np.exp((GSI - 100) / (9 - 3 * D))
    a_HB = 0.5 + (1/6) * (np.exp(-GSI/15) - np.exp(-20/3))
    return {"mb": mb, "s": s, "a_HB": a_HB}


def rock_mass_modulus(E_i, GSI, D):
    return E_i * (0.02 + (1 - D/2) / (1 + np.exp((60 + 15*D - GSI) / 11)))


def hoek_to_mc_fit(sigma_ci, mb, s, a_HB, sig3_max):
    """Hoek 2002 closed-form equivalent Mohr-Coulomb parameters."""
    sig3n = sig3_max / sigma_ci
    base = s + mb * sig3n
    pwr = base ** (a_HB - 1)
    num_phi = 6 * a_HB * mb * pwr
    den_phi = 2 * (1 + a_HB) * (2 + a_HB) + 6 * a_HB * mb * pwr
    phi = np.degrees(np.arcsin(num_phi / den_phi))
    num_c = sigma_ci * ((1 + 2*a_HB)*s + (1 - a_HB)*mb*sig3n) * pwr
    den_c = (1 + a_HB) * (2 + a_HB) * np.sqrt(
        1 + 6 * a_HB * mb * pwr / ((1 + a_HB) * (2 + a_HB)))
    c = num_c / den_c
    return phi, c


# =============================================================================
# GRC METHODS
# =============================================================================

def grc_duncan_fama(pi, p0, a, nu, Em, phi_deg, c):
    """Duncan-Fama (MC) GRC — the RocSupport default method. Returns u_r in mm."""
    phi = np.radians(phi_deg)
    k = (1 + np.sin(phi)) / (1 - np.sin(phi))
    sigma_cm = 2 * c * np.cos(phi) / (1 - np.sin(phi))
    pcr = (2 * p0 - sigma_cm) / (1 + k)

    if pi >= pcr:
        ur = (1 + nu) * (p0 - pi) * a / Em
    else:
        ur_pcr = (1 + nu) * (p0 - pcr) * a / Em
        Rp = a * ((2 * (p0*(k-1) + sigma_cm)) /
                  ((1+k) * (pi*(k-1) + sigma_cm))) ** (1/(k-1))
        ur = ur_pcr * (Rp / a) ** 2
    return ur * 1000


def grc_carranza_torres_hb(pi, p0, a, nu, Em, sigma_ci, mb, s, a_HB):
    """Carranza-Torres (2004) simplified HB GRC.

    Uses a curve-matched approach: computes pcr and Rp from the generalised
    Hoek-Brown criterion directly without requiring an equivalent MC fit.
    For a_HB = 0.5 this recovers the original Carranza-Torres 2004 form;
    for a_HB != 0.5 it uses a numerical root-find for pcr.

    Simplifications from full CT 2004:
    - No residual strength (peak = residual)
    - Non-dilatant (associated flow with ψ = 0)
    - Small-strain
    """
    from scipy.optimize import brentq as _brentq

    # Hoek-Brown sigma_1 as function of sigma_3
    def sig1_HB(sig3):
        return sig3 + sigma_ci * (mb * sig3 / sigma_ci + s) ** a_HB

    # Critical pressure: the value of pi at which the wall stress
    # sigma_theta (= 2p0 - pi) equals sigma_1 from H-B at sigma_3 = pi
    # i.e. 2p0 - pi = pi + sigma_ci * (mb*pi/sigma_ci + s)^a
    def pcr_residual(p):
        return (2 * p0 - p) - sig1_HB(p)

    try:
        pcr = _brentq(pcr_residual, 0, p0, xtol=1e-8)
    except ValueError:
        # No yielding — fully elastic
        pcr = -1

    if pi >= pcr:
        # Elastic branch (Kirsch)
        ur = (1 + nu) * (p0 - pi) * a / Em
    else:
        # Plastic branch. Use Carranza-Torres 2004 plastic-radius formula
        # for generalised H-B (Eq. 42 in the paper, simplified, a = 0.5 form
        # as an approximation for general a).
        # Plastic zone radius:
        # Rp/a = exp[ integral from pi to pcr of (1/sigma_cm) dp ]
        # For the generalised HB with no dilation:
        # Using the approximate form from CT 2004:
        sigma_cm_0 = sig1_HB(0)  # Uniaxial HB strength = sigma_ci * s^a
        # Tangent slope of HB envelope at sig_3 = pi:
        if mb * pi / sigma_ci + s > 0:
            k_tang = 1 + a_HB * mb * (mb * pi / sigma_ci + s) ** (a_HB - 1)
        else:
            k_tang = 1 + mb * a_HB  # limiting case
        # Use the tangent-MC approximation at the yielding boundary:
        sigma_cm_tang = sig1_HB(pi) - k_tang * pi

        # Plastic radius (Duncan-Fama-like with tangent k):
        Rp = a * ((2 * (p0 * (k_tang - 1) + sigma_cm_tang)) /
                  ((1 + k_tang) * (pi * (k_tang - 1) + sigma_cm_tang))) ** (1 / (k_tang - 1))

        ur_pcr = (1 + nu) * (p0 - pcr) * a / Em
        ur = ur_pcr * (Rp / a) ** 2
    return ur * 1000


def grc_vrakas_anagnostou(pi, p0, a, nu, Em, phi_deg, c, psi_deg=0.0):
    """V&A (2014) — large-strain MC with simplified dilation correction."""
    phi = np.radians(phi_deg)
    psi = np.radians(psi_deg)
    k = (1 + np.sin(phi)) / (1 - np.sin(phi))
    k_psi = (1 + np.sin(psi)) / (1 - np.sin(psi))
    sigma_cm = 2 * c * np.cos(phi) / (1 - np.sin(phi))
    pcr = (2 * p0 - sigma_cm) / (1 + k)

    if pi >= pcr:
        ur = (1 + nu) * (p0 - pi) * a / Em
    else:
        ur_pcr = (1 + nu) * (p0 - pcr) * a / Em
        Rp = a * ((2 * (p0*(k-1) + sigma_cm)) /
                  ((1+k) * (pi*(k-1) + sigma_cm))) ** (1/(k-1))
        ur_ss = ur_pcr * (Rp / a) ** 2
        strain = ur_ss / a
        correction = 1 + 0.5 * strain * k_psi
        ur = ur_ss * correction
    return ur * 1000


def grc_displacement(method, pi, p0, a, nu, Em, **kwargs):
    if method == "Duncan-Fama (MC)":
        return grc_duncan_fama(pi, p0, a, nu, Em, kwargs["phi_deg"], kwargs["c"])
    elif method == "Carranza-Torres (HB, 2004)":
        return grc_carranza_torres_hb(pi, p0, a, nu, Em, kwargs["sigma_ci"],
                                      kwargs["mb"], kwargs["s"], kwargs["a_HB"])
    elif method == "Vrakas & Anagnostou (MC, 2014)":
        return grc_vrakas_anagnostou(pi, p0, a, nu, Em, kwargs["phi_deg"],
                                     kwargs["c"], kwargs.get("psi_deg", 0.0))
    raise ValueError(method)


def critical_pressure_mc(p0, phi_deg, c):
    phi = np.radians(phi_deg)
    k = (1 + np.sin(phi)) / (1 - np.sin(phi))
    sigma_cm = 2 * c * np.cos(phi) / (1 - np.sin(phi))
    return max(0, (2 * p0 - sigma_cm) / (1 + k)), k, sigma_cm


def plastic_radius_mc(a, p0, pi, phi_deg, c):
    pcr, k, sigma_cm = critical_pressure_mc(p0, phi_deg, c)
    if pi >= pcr:
        return a
    num = 2 * (p0 * (k - 1) + sigma_cm)
    den = (1 + k) * (pi * (k - 1) + sigma_cm)
    return a * (num / den) ** (1 / (k - 1))


# =============================================================================
# LDP METHODS
# =============================================================================

def ldp_panet(Xstar, alpha=0.75):
    if Xstar <= 0:
        return (1 - alpha) * np.exp(1.5 * Xstar)
    return 1 - alpha * np.exp(-1.5 * Xstar)


def ldp_vlachopoulos(Xstar, Rstar):
    if Xstar <= 0:
        return (1/3) * np.exp(2 * Xstar - 0.15 * Rstar)
    return 1 - (1 - (1/3) * np.exp(-0.15 * Rstar)) * np.exp(-3 * Xstar / Rstar)


def ldp_hoek(Xstar):
    """Hoek (2002) empirical — uses X/D; here Xstar = X/a so X/D = Xstar/2."""
    Xd = Xstar / 2
    return (1 + np.exp(-Xd / 1.1)) ** (-1.7)


def ldp(method, Xstar, Rstar=1.0, alpha=0.75):
    if method == "Panet (1995)":
        return ldp_panet(Xstar, alpha)
    elif method == "Vlachopoulos & Diederichs (2009)":
        return ldp_vlachopoulos(Xstar, Rstar)
    elif method == "Hoek (2002)":
        return ldp_hoek(Xstar)
    raise ValueError(method)


# =============================================================================
# SUPPORT SYSTEMS
# =============================================================================

def support_concrete(sigma_ci, Ec, nu_c, t, a):
    psm = sigma_ci * t / a
    ks = Ec * t / ((1 - nu_c**2) * a**2)
    usm = psm / ks * 1000
    return {"psm": psm, "ks": ks, "usm": usm}


def support_rockbolt(rb_type, pattern_spacing, a):
    props = {
        "20 mm rebar":    {"T": 184, "strain": 0.0024},
        "25 mm rebar":    {"T": 287, "strain": 0.0024},
        "34 mm rebar":    {"T": 500, "strain": 0.0024},
        "Swellex Mn12":   {"T": 110, "strain": 0.010},
        "Split Set SS39": {"T": 100, "strain": 0.015},
    }
    p = props.get(rb_type, props["34 mm rebar"])
    psm = p["T"] / 1000.0 / (pattern_spacing ** 2)
    usm = p["strain"] * a * 1000
    ks = psm / (usm / 1000) if usm > 0 else 0
    return {"psm": psm, "ks": ks, "usm": usm}


def support_steel_set(set_type, spacing, a):
    props = {
        "W6x20": {"A": 38.0e-4, "sigma_y": 248},
        "W8x31": {"A": 58.9e-4, "sigma_y": 248},
        "TH-29": {"A": 37.0e-4, "sigma_y": 345},
        "TH-36": {"A": 45.7e-4, "sigma_y": 345},
    }
    p = props.get(set_type, props["W8x31"])
    psm = p["sigma_y"] * p["A"] / (spacing * a)
    E_steel = 200_000
    ks = E_steel * p["A"] / (spacing * a**2)
    usm = psm / ks * 1000
    return {"psm": psm, "ks": ks, "usm": usm}


# =============================================================================
# FoS SOLVER
# =============================================================================

def solve_fos(grc_func, psm, ks, us0, p0_max):
    def diff(pi):
        return grc_func(pi) - (us0 + pi / ks * 1000)

    u_scc_max = us0 + psm / ks * 1000
    u_grc_at_psm = grc_func(psm)
    if u_grc_at_psm > u_scc_max:
        return {"p_eq": psm, "u_eq": u_scc_max, "FoS": 1.0, "saturated": True}

    try:
        p_eq = brentq(diff, 0.0001, psm * 0.9999, xtol=1e-6)
        u_eq = us0 + p_eq / ks * 1000
        return {"p_eq": p_eq, "u_eq": u_eq, "FoS": psm / p_eq, "saturated": False}
    except Exception:
        return {"p_eq": None, "u_eq": None, "FoS": None, "saturated": False}


# =============================================================================
# FULL ANALYSIS
# =============================================================================

def run_analysis(params):
    p = params
    p0 = p["gamma"] * p["z"] / 1000

    hb = hoek_brown_params(p["GSI"], p["mi"], p["D"])
    Em = rock_mass_modulus(p["E_i"], p["GSI"], p["D"])
    sigma_cm = p["sigma_ci"] * hb["s"] ** hb["a_HB"]

    if p["mc_source"] == "Fit automatically (Hoek 2002)":
        phi_deg, c = hoek_to_mc_fit(p["sigma_ci"], hb["mb"], hb["s"], hb["a_HB"],
                                    p["sig3_max"])
    else:
        phi_deg, c = p["phi_deg"], p["c"]

    pcr, k, sigma_cm_MC = critical_pressure_mc(p0, phi_deg, c)
    is_plastic = p0 > pcr
    Rp_unsup = plastic_radius_mc(p["a"], p0, 0, phi_deg, c) if is_plastic else p["a"]
    Rstar = Rp_unsup / p["a"]

    grc_kwargs = {"phi_deg": phi_deg, "c": c, "psi_deg": p.get("psi_deg", 0.0),
                  "sigma_ci": p["sigma_ci"], **hb}

    def grc_fn(pi):
        return grc_displacement(p["grc_method"], pi, p0, p["a"], p["nu"], Em,
                                **grc_kwargs)

    u_max = grc_fn(0.0)
    ur_pcr = grc_fn(pcr) if is_plastic else (1 + p["nu"]) * (p0 - pcr) * p["a"] / Em * 1000

    Xstar = p["L"] / p["a"]
    u_star = ldp(p["ldp_method"], Xstar, Rstar, p.get("alpha_panet", 0.75))
    us0 = u_star * u_max

    supports = p["supports"]
    if supports:
        psm_total = sum(s["psm"] for s in supports)
        ks_total = sum(s["ks"] for s in supports)
        usm_total = max(s["usm"] for s in supports)
    else:
        psm_total, ks_total, usm_total = 0.01, 1, 0

    if supports:
        fos_result = solve_fos(grc_fn, psm_total, ks_total, us0, p0)
    else:
        fos_result = {"p_eq": None, "u_eq": None, "FoS": None, "saturated": False}

    if fos_result["p_eq"] is not None and is_plastic:
        Rp_sup = plastic_radius_mc(p["a"], p0, fos_result["p_eq"], phi_deg, c)
    else:
        Rp_sup = Rp_unsup

    return {
        "p0": p0, "Em": Em, "phi_deg": phi_deg, "c": c, "k": k,
        "sigma_cm_MC": sigma_cm_MC, "sigma_cm_HB": sigma_cm,
        "pcr": pcr, "is_plastic": is_plastic,
        "Rp_unsup": Rp_unsup, "Rp_sup": Rp_sup, "Rstar": Rstar,
        "ur_pcr": ur_pcr, "u_max": u_max,
        "Xstar": Xstar, "u_star": u_star, "us0": us0,
        "psm_total": psm_total, "ks_total": ks_total, "usm_total": usm_total,
        "grc_fn": grc_fn, **hb, **fos_result,
    }


# =============================================================================
# PLOTS
# =============================================================================

PALETTE = {
    "grc": "#1F4E78", "scc": "#C00000", "ldp": "#2E7D32",
    "highlight": "#FF6B35", "elastic_zone": "#e8d99a",
    "plastic_zone": "#f4c2a8", "tunnel": "#ffffff", "support": "#C00000",
}


def plot_grc_scc(result, params):
    p0 = result["p0"]
    pi_range = np.linspace(0, p0, 250)
    u_grc = np.array([result["grc_fn"](pi) for pi in pi_range])
    u_max_plot = max(result["u_max"], result["us0"] + result["usm_total"]) * 1.15

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_grc, y=pi_range, mode="lines",
                             line=dict(color=PALETTE["grc"], width=3),
                             name="Ground Reaction Curve (GRC)"))

    if result["is_plastic"]:
        fig.add_hline(y=result["pcr"],
                      line=dict(color=PALETTE["grc"], width=1, dash="dot"),
                      annotation_text=f"p_cr = {result['pcr']:.2f} MPa",
                      annotation_position="right",
                      annotation_font_color=PALETTE["grc"])

    if params["supports"] and result["psm_total"] > 0.02:
        us0 = result["us0"]; usm = result["usm_total"]; psm = result["psm_total"]
        u_scc = [us0, us0 + usm, max(u_max_plot, us0 + usm) * 1.1]
        p_scc = [0, psm, psm]
        fig.add_trace(go.Scatter(x=u_scc, y=p_scc, mode="lines+markers",
                                 line=dict(color=PALETTE["scc"], width=3, dash="dash"),
                                 marker=dict(size=7),
                                 name="Support Reaction Curve (SCC)"))
        fig.add_trace(go.Scatter(x=[us0], y=[0], mode="markers",
                                 marker=dict(size=12, color="gold",
                                             line=dict(color=PALETTE["scc"], width=2)),
                                 showlegend=False))
        fig.add_annotation(x=us0 + usm, y=psm, text=f"p_sm = {psm:.2f} MPa",
                           showarrow=False, xshift=60, yshift=10,
                           font=dict(size=10, color=PALETTE["scc"]))

    if result.get("p_eq") is not None:
        fos = result["FoS"]
        verdict = "Safe" if fos >= 1.5 else "Marginal" if fos >= 1 else "Unsafe"
        color = "#2E7D32" if fos >= 1.5 else "#F57C00" if fos >= 1 else "#C62828"
        fig.add_trace(go.Scatter(
            x=[result["u_eq"]], y=[result["p_eq"]], mode="markers",
            marker=dict(size=14, color=color, symbol="circle",
                        line=dict(color="white", width=2)),
            name=f"Intersection: p={result['p_eq']:.2f} MPa, FoS={fos:.2f} ({verdict})"))
        fig.add_annotation(x=result["u_eq"], y=result["p_eq"],
                           text=f"<b>p_eq = {result['p_eq']:.2f} MPa</b>",
                           showarrow=True, arrowhead=2, ax=60, ay=-40,
                           font=dict(size=11, color=color),
                           bgcolor="white", bordercolor=color, borderwidth=1)

    fig.update_layout(
        title=dict(text=f"<b>Ground Reaction & Support Reaction</b> — {params['grc_method']}",
                   font=dict(size=15)),
        xaxis_title="Tunnel wall displacement u_i (mm)",
        yaxis_title="Support pressure p_i (MPa)",
        height=520, hovermode="closest",
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98,
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=10)),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(family="Georgia, serif", size=12),
    )
    fig.update_xaxes(gridcolor="#e5e5e5", linecolor="#333", mirror=True,
                     showline=True, range=[0, u_max_plot])
    fig.update_yaxes(gridcolor="#e5e5e5", linecolor="#333", mirror=True,
                     showline=True, range=[0, p0 * 1.05])
    return fig


def plot_tunnel_section(result, params):
    a = params["a"]
    Rp_unsup = result["Rp_unsup"]
    Rp_sup = result["Rp_sup"]

    fig = go.Figure()
    theta = np.linspace(0, 2*np.pi, 100)

    if Rp_unsup > a:
        fig.add_trace(go.Scatter(
            x=Rp_unsup * np.cos(theta), y=Rp_unsup * np.sin(theta),
            mode="lines", line=dict(color="#aaa", dash="dot", width=1.5),
            name=f"Unsupported plastic zone: {Rp_unsup:.2f} m",
            fill="toself", fillcolor=PALETTE["elastic_zone"], opacity=0.4))

    if params["supports"] and Rp_sup > a and Rp_sup < Rp_unsup:
        fig.add_trace(go.Scatter(
            x=Rp_sup * np.cos(theta), y=Rp_sup * np.sin(theta),
            mode="lines", line=dict(color="#d4827b", width=2),
            name=f"Supported plastic zone: {Rp_sup:.2f} m",
            fill="toself", fillcolor=PALETTE["plastic_zone"], opacity=0.6))
    elif Rp_unsup > a and not params["supports"]:
        fig.add_trace(go.Scatter(
            x=Rp_unsup * np.cos(theta), y=Rp_unsup * np.sin(theta),
            mode="lines", line=dict(color="#d4827b", width=2),
            name=f"Plastic zone: {Rp_unsup:.2f} m",
            fill="toself", fillcolor=PALETTE["plastic_zone"], opacity=0.6))

    fig.add_trace(go.Scatter(
        x=a * np.cos(theta), y=a * np.sin(theta),
        mode="lines", line=dict(color="black", width=2),
        name=f"Tunnel (r = {a} m)",
        fill="toself", fillcolor=PALETTE["tunnel"]))

    if params["supports"]:
        for s in params["supports"]:
            if "t" in s:
                t = s["t"]
                fig.add_trace(go.Scatter(
                    x=(a - t) * np.cos(theta), y=(a - t) * np.sin(theta),
                    mode="lines", line=dict(color=PALETTE["support"], width=2),
                    name=f"Lining ({t*1000:.0f} mm)"))
                break

    if result.get("p_eq") is not None:
        n_arrows = 12
        arrow_theta = np.linspace(0, 2*np.pi, n_arrows, endpoint=False)
        for th in arrow_theta:
            fig.add_annotation(
                x=a * 0.95 * np.cos(th), y=a * 0.95 * np.sin(th),
                ax=a * 0.75 * np.cos(th), ay=a * 0.75 * np.sin(th),
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=2,
                arrowcolor=PALETTE["support"])
        fig.add_annotation(x=0, y=0, text=f"<b>p_eq = {result['p_eq']:.2f} MPa</b>",
                           showarrow=False, font=dict(size=13, color=PALETTE["support"]))

    max_r = max(Rp_unsup, a) * 1.3
    fig.update_layout(
        title=dict(text="<b>Tunnel Section View</b>", font=dict(size=15)),
        xaxis=dict(title="x (m)", range=[-max_r, max_r], gridcolor="#e5e5e5",
                   showline=True, linecolor="#333", scaleanchor="y",
                   scaleratio=1, zeroline=False),
        yaxis=dict(title="y (m)", range=[-max_r, max_r], gridcolor="#e5e5e5",
                   showline=True, linecolor="#333", zeroline=False),
        height=500, paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(family="Georgia, serif", size=11),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#ccc", borderwidth=1, font=dict(size=10)),
    )
    return fig


def plot_ldp(result, params):
    Rstar = result["Rstar"]
    Xstar_range = np.linspace(-3, 10, 300)

    fig = go.Figure()
    methods = ["Panet (1995)", "Vlachopoulos & Diederichs (2009)", "Hoek (2002)"]
    colors = ["#888", "#888", "#888"]
    widths = [1.5, 1.5, 1.5]
    for i, m in enumerate(methods):
        if m == params["ldp_method"]:
            colors[i] = PALETTE["ldp"]; widths[i] = 3.5

    for m, col, w in zip(methods, colors, widths):
        u_vals = np.array([ldp(m, x, Rstar, params.get("alpha_panet", 0.75)) * 100
                           for x in Xstar_range])
        fig.add_trace(go.Scatter(x=Xstar_range, y=u_vals, mode="lines",
                                 line=dict(color=col, width=w), name=m))

    fig.add_vline(x=0, line=dict(color="black", width=2),
                  annotation_text="Tunnel face", annotation_position="top")
    fig.add_vline(x=result["Xstar"],
                  line=dict(color=PALETTE["highlight"], width=2, dash="dash"),
                  annotation_text=f"Support (L={params['L']}m)",
                  annotation_position="top")
    fig.add_trace(go.Scatter(
        x=[result["Xstar"]], y=[result["u_star"] * 100], mode="markers",
        marker=dict(size=14, color=PALETTE["highlight"], symbol="star",
                    line=dict(color="white", width=2)),
        name=f"u_s0 point: u*={result['u_star']:.2f}"))

    fig.update_layout(
        title=dict(text="<b>Longitudinal Deformation Profiles</b>", font=dict(size=15)),
        xaxis_title="Normalised distance X/a  (positive = behind face)",
        yaxis_title="u / u_max  (%)",
        height=450, hovermode="x",
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=10)),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(family="Georgia, serif", size=11),
    )
    fig.update_xaxes(gridcolor="#e5e5e5", linecolor="#333", mirror=True, showline=True)
    fig.update_yaxes(gridcolor="#e5e5e5", linecolor="#333", mirror=True, showline=True,
                     range=[0, 110])
    return fig


def plot_envelope(result, params):
    sigma_ci = params["sigma_ci"]
    mb, s, a_HB = result["mb"], result["s"], result["a_HB"]
    phi_deg, c = result["phi_deg"], result["c"]

    sig3_max = params.get("sig3_max", 17)
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
        title=dict(text="<b>H-B vs MC Failure Envelopes</b>", font=dict(size=15)),
        xaxis_title="σ₃ (MPa)", yaxis_title="σ₁ (MPa)",
        height=420,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98,
                    bgcolor="rgba(255,255,255,0.9)"),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(family="Georgia, serif", size=11),
    )
    fig.update_xaxes(gridcolor="#e5e5e5", linecolor="#333", mirror=True, showline=True)
    fig.update_yaxes(gridcolor="#e5e5e5", linecolor="#333", mirror=True, showline=True)
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("⚙️ Project Settings")

with st.sidebar.expander("**Solution method**", expanded=True):
    grc_method = st.selectbox(
        "GRC (Ground Reaction Curve) method",
        ["Duncan-Fama (MC)",
         "Carranza-Torres (HB, 2004)",
         "Vrakas & Anagnostou (MC, 2014)"],
        index=0,
        help="Duncan-Fama = default (Mohr-Coulomb, small-strain). "
             "Carranza-Torres = generalised Hoek-Brown (small-strain). "
             "Vrakas & Anagnostou = large-strain MC with dilation.")

    ldp_method = st.selectbox(
        "LDP (Longitudinal Deformation Profile)",
        ["Vlachopoulos & Diederichs (2009)",
         "Panet (1995)",
         "Hoek (2002)"],
        index=0,
        help="V&D (2009) = elastoplastic, depends on R_p. "
             "Panet (1995) = elastic, exponential. "
             "Hoek (2002) = empirical, diameter-normalised.")

    if ldp_method == "Panet (1995)":
        alpha_panet = st.slider("Panet α", 0.5, 0.95, 0.75, 0.05)
    else:
        alpha_panet = 0.75

with st.sidebar.expander("**Tunnel & stress**", expanded=True):
    a = st.number_input("Tunnel radius, a (m)", 0.5, 15.0, 4.75, 0.25)
    gamma = st.number_input("Unit weight, γ (kN/m³)", 15.0, 35.0, 26.0, 0.5)
    z = st.number_input("Depth, z (m)", 10.0, 2000.0, 650.0, 10.0)
    st.caption(f"→ in-situ stress p₀ = γ·z = {gamma * z / 1000:.2f} MPa")
    L = st.number_input("Support distance from face, L (m)", 0.0, 30.0, 3.0, 0.5)

with st.sidebar.expander("**Rock mass (Hoek-Brown)**", expanded=True):
    sigma_ci = st.number_input("σ_ci — intact UCS (MPa)", 1.0, 500.0, 72.0, 1.0)
    E_i = st.number_input("E_i — intact modulus (GPa)", 0.1, 100.0, 26.0, 0.5) * 1000
    mi = st.number_input("m_i (Hoek-Brown)", 1.0, 40.0, 7.0, 0.5,
                         help="7 ≈ siltstone, 17 ≈ sandstone, 10 ≈ limestone")
    GSI = st.slider("GSI (Geological Strength Index)", 10, 100, 60, 5)
    D = st.slider("Disturbance factor D", 0.0, 1.0, 0.0, 0.1)
    nu = st.number_input("Poisson's ratio ν", 0.0, 0.5, 0.22, 0.01)

with st.sidebar.expander("**Mohr-Coulomb parameters**", expanded=True):
    mc_source = st.radio(
        "Source of φ and c",
        ["Fit automatically (Hoek 2002)", "Enter manually"],
        help="Hoek 2002 closed-form fit over 0 ≤ σ₃ ≤ σ_3,max.")
    if mc_source == "Fit automatically (Hoek 2002)":
        sig3_max_default = gamma * z / 1000
        sig3_max = st.number_input("σ_3,max for MC fit (MPa)",
                                   0.5, 50.0, float(sig3_max_default), 0.5,
                                   help="Upper bound of fit range.")
        _hb = hoek_brown_params(GSI, mi, D)
        _phi, _c = hoek_to_mc_fit(sigma_ci, _hb["mb"], _hb["s"], _hb["a_HB"], sig3_max)
        st.caption(f"→ Fitted: φ = {_phi:.2f}°, c = {_c:.3f} MPa")
        phi_deg, c = _phi, _c
    else:
        phi_deg = st.number_input("φ — friction angle (°)", 5.0, 60.0, 31.5, 0.5)
        c = st.number_input("c — cohesion (MPa)", 0.0, 20.0, 2.85, 0.1)
        sig3_max = gamma * z / 1000

    if grc_method == "Vrakas & Anagnostou (MC, 2014)":
        psi_deg = st.slider("Dilation angle ψ (°)", 0.0, 40.0, 0.0, 1.0)
    else:
        psi_deg = 0.0

st.sidebar.header("🔩 Support System")
supports = []

with st.sidebar.expander("**Shotcrete / Concrete lining**", expanded=True):
    use_concrete = st.checkbox("Add concrete lining", value=True)
    if use_concrete:
        sigma_ci_lining = st.number_input("σ_ci,lining (MPa)", 20.0, 200.0, 60.0, 5.0,
                                          key="sc_ci")
        Ec = st.number_input("E_c (GPa)", 10.0, 80.0, 40.0, 1.0, key="sc_Ec") * 1000
        nu_c = st.number_input("ν_c", 0.0, 0.5, 0.25, 0.01, key="sc_nuc")
        t = st.number_input("Thickness t (m)", 0.05, 2.0, 0.5, 0.05, key="sc_t")
        c_props = support_concrete(sigma_ci_lining, Ec, nu_c, t, a)
        c_props["name"] = "Concrete lining"; c_props["t"] = t
        supports.append(c_props)
        st.caption(f"→ p_sm = {c_props['psm']:.2f} MPa, "
                   f"k_s = {c_props['ks']:.0f} MPa/m, "
                   f"u_sm = {c_props['usm']:.2f} mm")

with st.sidebar.expander("**Rockbolts**", expanded=False):
    use_bolts = st.checkbox("Add rockbolts", value=False)
    if use_bolts:
        rb_type = st.selectbox("Bolt type",
                               ["20 mm rebar", "25 mm rebar", "34 mm rebar",
                                "Swellex Mn12", "Split Set SS39"])
        pattern = st.number_input("Pattern spacing s × s (m)", 0.5, 3.0, 1.0, 0.1)
        b_props = support_rockbolt(rb_type, pattern, a)
        b_props["name"] = f"Rockbolts ({rb_type})"
        supports.append(b_props)
        st.caption(f"→ p_sm = {b_props['psm']:.3f} MPa, "
                   f"u_sm = {b_props['usm']:.1f} mm")

with st.sidebar.expander("**Steel sets**", expanded=False):
    use_sets = st.checkbox("Add steel sets", value=False)
    if use_sets:
        set_type = st.selectbox("Set type", ["W6x20", "W8x31", "TH-29", "TH-36"])
        ss_spacing = st.number_input("Set spacing (m)", 0.5, 3.0, 1.0, 0.1)
        ss_props = support_steel_set(set_type, ss_spacing, a)
        ss_props["name"] = f"Steel sets ({set_type})"
        supports.append(ss_props)
        st.caption(f"→ p_sm = {ss_props['psm']:.2f} MPa, "
                   f"u_sm = {ss_props['usm']:.1f} mm")

with st.sidebar.expander("**Custom support**", expanded=False):
    use_custom = st.checkbox("Add custom support", value=False)
    if use_custom:
        cust_psm = st.number_input("p_sm (MPa)", 0.1, 50.0, 1.0, 0.1)
        cust_ks = st.number_input("k_s (MPa/m)", 1.0, 5000.0, 500.0, 10.0)
        cust_usm = cust_psm / cust_ks * 1000
        supports.append({"psm": cust_psm, "ks": cust_ks, "usm": cust_usm,
                         "name": "Custom"})
        st.caption(f"→ u_sm = {cust_usm:.2f} mm (computed)")

params = dict(
    sigma_ci=sigma_ci, E_i=E_i, mi=mi, GSI=GSI, D=D, nu=nu,
    gamma=gamma, z=z, a=a, L=L,
    phi_deg=phi_deg, c=c, sig3_max=sig3_max, psi_deg=psi_deg,
    mc_source=mc_source,
    grc_method=grc_method, ldp_method=ldp_method, alpha_panet=alpha_panet,
    supports=supports,
)
result = run_analysis(params)


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("⛰️ Convergence-Confinement Method (CCM)")
st.markdown(
    "**RSE3010 — Mine Geotechnical Engineering | Monash University** \n"
    "*Interactive ground–support interaction tool, modelled on RocSupport.*"
)

if result["FoS"] is not None:
    fos = result["FoS"]
    if fos >= 2.0: color, verdict = "#2E7D32", "✓ Safe"
    elif fos >= 1.5: color, verdict = "#F57C00", "⚠ Marginal"
    else: color, verdict = "#C62828", "✗ Unsafe"
    support_list = ", ".join(s["name"] for s in supports) if supports else "None"
    st.markdown(
        f"""<div style='background:{color}15; border-left:5px solid {color};
            padding:1.1rem 1.25rem; border-radius:4px; margin-bottom:1rem;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
        <div>
            <h2 style='color:{color}; margin:0; font-size:1.4rem;'>
            FoS = {fos:.2f} — {verdict}
            </h2>
            <p style='margin:0.3rem 0 0 0; color:#333; font-size:0.95rem;'>
            p_eq = {result['p_eq']:.3f} MPa · u_eq = {result['u_eq']:.2f} mm ·
            p_sm = {result['psm_total']:.2f} MPa · Support: {support_list}
            </p>
        </div>
        <div style='text-align:right; color:#555; font-size:0.85rem;'>
            GRC: {grc_method}<br>LDP: {ldp_method}
        </div>
        </div>
        </div>""",
        unsafe_allow_html=True)
else:
    st.warning("No support specified — add at least one support system in the sidebar "
               "to compute FoS.")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Summary", "📈 GRC / SCC", "⭕ Tunnel Section", "📉 LDP", "🧭 Envelopes",
    "📋 Workflow",
])

with tab1:
    st.header("Analysis Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Rock Mass")
        st.metric("p₀ (in-situ stress)", f"{result['p0']:.2f} MPa")
        st.metric("m_b", f"{result['mb']:.3f}")
        st.metric("s", f"{result['s']:.5f}")
        st.metric("a_HB (HB exponent)", f"{result['a_HB']:.3f}")
        st.metric("σ_cm (HB uniaxial)", f"{result['sigma_cm_HB']:.2f} MPa")
        st.metric("E_m (rock mass)", f"{result['Em']/1000:.2f} GPa")
    with col2:
        st.subheader("Mohr-Coulomb & Response")
        st.metric("φ (friction angle)", f"{result['phi_deg']:.2f}°")
        st.metric("c (cohesion)", f"{result['c']:.3f} MPa")
        st.metric("σ_cm,MC", f"{result['sigma_cm_MC']:.2f} MPa")
        st.metric("p_cr (critical pressure)", f"{result['pcr']:.2f} MPa")
        st.metric("Response", "Plastic" if result['is_plastic'] else "Elastic")
        st.metric("R_p (unsupported)", f"{result['Rp_unsup']:.2f} m")
    with col3:
        st.subheader("Convergence & Support")
        st.metric("u_max (unsupported)", f"{result['u_max']:.2f} mm")
        st.metric("u_s0 (at installation)", f"{result['us0']:.2f} mm")
        st.metric("R_p (supported)", f"{result['Rp_sup']:.2f} m")
        st.metric("Combined p_sm", f"{result['psm_total']:.2f} MPa")
        st.metric("Combined k_s", f"{result['ks_total']:.0f} MPa/m")
        st.metric("u_sm (max elastic)", f"{result['usm_total']:.2f} mm")

    if supports:
        st.subheader("Support breakdown")
        df_support = pd.DataFrame([
            {"Support": s["name"],
             "p_sm (MPa)": round(s["psm"], 3),
             "k_s (MPa/m)": round(s["ks"], 1),
             "u_sm (mm)": round(s["usm"], 2)}
            for s in supports
        ])
        st.dataframe(df_support, hide_index=True, width="stretch")

with tab2:
    st.plotly_chart(plot_grc_scc(result, params), width="stretch")
    if result.get("saturated"):
        st.error(
            "⚠️ **Support saturated.** The SCC reached its maximum pressure p_sm "
            "before intersecting the GRC — the lining is at capacity and the "
            "FoS is effectively 1.0. Consider a stiffer support, yielding "
            "elements, or reducing the stand-off distance L.")

with tab3:
    st.plotly_chart(plot_tunnel_section(result, params), width="stretch")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tunnel radius", f"{params['a']:.2f} m")
    col2.metric("R_p unsupported", f"{result['Rp_unsup']:.2f} m")
    col3.metric("R_p supported", f"{result['Rp_sup']:.2f} m")
    col4.metric("Unsupported convergence",
                f"{result['u_max']/params['a']/1000 * 100:.2f} %")

with tab4:
    st.plotly_chart(plot_ldp(result, params), width="stretch")
    st.info(
        f"**Current installation position:** L = {params['L']} m "
        f"(X* = L/a = {result['Xstar']:.3f}). "
        f"The {params['ldp_method']} method gives u* = {result['u_star']:.3f}, "
        f"so u_s0 = {result['us0']:.2f} mm of the {result['u_max']:.1f} mm "
        f"total unsupported convergence has already occurred at the installation point.")

with tab5:
    st.plotly_chart(plot_envelope(result, params), width="stretch")
    if params["mc_source"] == "Fit automatically (Hoek 2002)":
        st.success(
            f"MC parameters fitted automatically via Hoek (2002) closed form "
            f"over 0 ≤ σ₃ ≤ {params['sig3_max']:.1f} MPa: "
            f"**φ = {result['phi_deg']:.2f}°**, **c = {result['c']:.3f} MPa**.")

with tab6:
    st.header("Calculation Workflow")

    st.markdown("### Step 1 — In-situ stress")
    st.latex(r"p_0 = \gamma \cdot z")

    st.markdown("### Step 2 — Hoek-Brown parameters")
    st.latex(r"m_b = m_i \, \exp\!\left(\frac{GSI - 100}{28 - 14 D}\right)")
    st.latex(r"s = \exp\!\left(\frac{GSI - 100}{9 - 3 D}\right)")
    st.latex(r"a_{HB} = \tfrac{1}{2} + \tfrac{1}{6}\left(e^{-GSI/15} - e^{-20/3}\right)")

    st.markdown("### Step 3 — Rock mass strength & modulus")
    st.latex(r"\sigma_{cm} = \sigma_{ci} \cdot s^{a_{HB}}")
    st.latex(r"E_m = E_i \left[0.02 + \frac{1 - D/2}{1 + \exp\!\big((60 + 15 D - GSI)/11\big)}\right]")

    st.markdown("### Step 4 — Equivalent Mohr-Coulomb (Hoek 2002 closed form)")
    st.latex(r"\phi = \arcsin\!\left[\frac{6\,a_{HB}\, m_b \,(s + m_b \sigma_{3n})^{a_{HB} - 1}}"
             r"{2(1+a_{HB})(2+a_{HB}) + 6\,a_{HB}\, m_b \,(s + m_b \sigma_{3n})^{a_{HB}-1}}\right]")
    st.latex(r"c = \frac{\sigma_{ci}\,[(1+2 a_{HB})s + (1-a_{HB}) m_b \sigma_{3n}]\,(s+m_b \sigma_{3n})^{a_{HB}-1}}"
             r"{(1+a_{HB})(2+a_{HB})\sqrt{1 + 6 a_{HB} m_b (s+m_b \sigma_{3n})^{a_{HB}-1}/[(1+a_{HB})(2+a_{HB})]}}")
    st.caption(r"where $\sigma_{3n} = \sigma_{3,\max}/\sigma_{ci}$. Derived: "
               r"$k=(1+\sin\phi)/(1-\sin\phi)$, $\sigma_{cm,MC}=2c\cos\phi/(1-\sin\phi)$.")

    st.markdown("### Step 5 — Critical pressure")
    st.latex(r"p_{cr} = \frac{2 p_0 - \sigma_{cm,MC}}{1 + k}")

    st.markdown("### Step 6 — Plastic zone radius (unsupported, $p_i=0$)")
    st.latex(r"R_p = a \left[\frac{2\left(p_0(k-1) + \sigma_{cm,MC}\right)}"
             r"{(1+k)\,\sigma_{cm,MC}}\right]^{1/(k-1)}")

    st.markdown("### Step 7 — Maximum convergence")
    st.latex(r"u_r(p_{cr}) = \frac{(1+\nu)(p_0-p_{cr})\,a}{E_m}")
    st.latex(r"u_{\max} \approx u_r(p_{cr}) \left(\frac{R_p}{a}\right)^2")

    st.markdown("### Step 8 — LDP at installation distance L")
    st.markdown("**Vlachopoulos & Diederichs (2009), behind face $X^*>0$:**")
    st.latex(r"u^*(X^*) = 1 - \left[1 - \tfrac{1}{3}\,e^{-0.15 R^*}\right] "
             r"e^{-3 X^*/R^*}")
    st.markdown(r"**Panet (1995):** $u^*(X^*) = 1 - \alpha\, e^{-1.5 X^*}$, "
                r"$\alpha = 0.75$.")
    st.markdown(r"**Hoek (2002):** $u^*(X^*) = (1 + e^{-X^*/2.2})^{-1.7}$ "
                r"(with $X^*$ normalised by diameter).")
    st.latex(r"u_{s0} = u^* \cdot u_{\max}")

    st.markdown("### Step 9 — Support Characteristic Curve")
    st.markdown("**Concrete / shotcrete ring:**")
    st.latex(r"p_{sm} = \frac{\sigma_{ci,\text{lining}} \cdot t}{a}, \quad "
             r"k_s = \frac{E_c \cdot t}{(1 - \nu_c^2)\,a^2}, \quad "
             r"u_{sm} = p_{sm}/k_s")
    st.markdown(r"**Combined support** (concrete + bolts + sets): "
                r"$p_{sm}^{\text{total}} = \sum p_{sm,i}$, "
                r"$k_s^{\text{total}} = \sum k_{s,i}$, "
                r"$u_{sm}^{\text{total}} = \max(u_{sm,i})$.")

    st.markdown("### Step 10 — FoS via GRC–SCC intersection")
    st.markdown(
        r"Solve $u_{GRC}(p_{eq}) = u_{s0} + p_{eq}/k_s$ numerically (root-find). "
        r"Then $\mathrm{FoS} = p_{sm}/p_{eq}$.")

    st.markdown("---")
    st.subheader("Current numerical results")
    rows = [
        ("p₀ (in-situ stress, MPa)", result["p0"]),
        ("m_b", result["mb"]), ("s", result["s"]), ("a_HB", result["a_HB"]),
        ("σ_cm HB (MPa)", result["sigma_cm_HB"]),
        ("E_m (MPa)", result["Em"]),
        ("φ (°)", result["phi_deg"]), ("c (MPa)", result["c"]),
        ("k", result["k"]), ("σ_cm,MC (MPa)", result["sigma_cm_MC"]),
        ("p_cr (MPa)", result["pcr"]),
        ("R_p unsupported (m)", result["Rp_unsup"]),
        ("R_p supported (m)", result["Rp_sup"]),
        ("R*=R_p/a", result["Rstar"]),
        ("u_max (mm)", result["u_max"]),
        ("X*=L/a", result["Xstar"]),
        ("u* LDP", result["u_star"]),
        ("u_s0 (mm)", result["us0"]),
        ("Combined p_sm (MPa)", result["psm_total"]),
        ("Combined k_s (MPa/m)", result["ks_total"]),
        ("Combined u_sm (mm)", result["usm_total"]),
        ("p_eq (MPa)", result["p_eq"]),
        ("u_eq (mm)", result["u_eq"]),
        ("FoS", result["FoS"]),
    ]
    df = pd.DataFrame({
        "Quantity": [r[0] for r in rows],
        "Value": [f"{r[1]:.4g}" if isinstance(r[1], (int, float)) else "—"
                  for r in rows],
    })
    st.dataframe(df, hide_index=True, width="stretch")

st.markdown("---")
st.caption(
    "RSE3010 Mine Geotechnical Engineering · Semester 1, 2026 · Monash University · "
    "GRC: Duncan-Fama, Carranza-Torres (2004), Vrakas & Anagnostou (2014) · "
    "LDP: Panet (1995), Vlachopoulos & Diederichs (2009), Hoek (2002) · "
    "MC fit: Hoek et al. (2002)"
)
