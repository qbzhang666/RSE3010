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
    phi = np.degrees(np.arcsin(np.clip(num_phi / den_phi, -1.0, 1.0)))
    num_c = sigma_ci * ((1 + 2*a_HB)*s + (1 - a_HB)*mb*sig3n) * pwr
    den_c = (1 + a_HB) * (2 + a_HB) * np.sqrt(
        1 + 6 * a_HB * mb * pwr / ((1 + a_HB) * (2 + a_HB)))
    c = num_c / den_c
    return phi, c


# =============================================================================
# GRC METHODS
# =============================================================================

def grc_duncan_fama(pi, p0, a, nu, Em, phi_deg, c):
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
    """Carranza-Torres (2004) simplified HB GRC with low-pressure stabilisation.

    Only the low-pressure tangent evaluation is stabilised to avoid a small
    non-physical back-bending tail near pi -> 0.
    """
    def sig1_HB(sig3):
        return sig3 + sigma_ci * (mb * sig3 / sigma_ci + s) ** a_HB

    def pcr_residual(p):
        return (2 * p0 - p) - sig1_HB(p)

    try:
        pcr = brentq(pcr_residual, 0, p0, xtol=1e-8)
    except ValueError:
        pcr = -1

    if pi >= pcr:
        ur = (1 + nu) * (p0 - pi) * a / Em
    else:
        # Stabilise tangent evaluation near pi -> 0
        pi_eff = max(pi, 0.02 * p0)
        tangent_base = max(mb * pi_eff / sigma_ci + s, 1e-10)
        k_tang = max(1.0 + a_HB * mb * tangent_base ** (a_HB - 1.0), 1.01)
        sigma_cm_tang = sig1_HB(pi_eff) - k_tang * pi_eff

        Rp = a * ((2 * (p0 * (k_tang - 1) + sigma_cm_tang)) /
                  ((1 + k_tang) * (pi * (k_tang - 1) + sigma_cm_tang))) ** (1 / (k_tang - 1))

        ur_pcr = (1 + nu) * (p0 - pcr) * a / Em
        ur = ur_pcr * (Rp / a) ** 2
    return ur * 1000


def grc_vrakas_anagnostou(pi, p0, a, nu, Em, phi_deg, c, psi_deg=0.0):
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


def grc_salencon(pi, p0, a, nu, Em, phi_deg, c):
    phi = np.radians(phi_deg)
    k = (1 + np.sin(phi)) / (1 - np.sin(phi))
    sigma_cm = 2 * c * np.cos(phi) / (1 - np.sin(phi))
    pcr = (2 * p0 - sigma_cm) / (1 + k)

    if pi >= pcr:
        ur = (1 + nu) * (p0 - pi) * a / Em
    else:
        Rp = a * ((2 * (p0*(k-1) + sigma_cm)) /
                  ((1+k) * (pi*(k-1) + sigma_cm))) ** (1/(k-1))
        ur_pcr = (1 + nu) * (p0 - pcr) * a / Em
        exponent = (k + 1) / (k - 1) if k > 1 else 2.0
        ur = ur_pcr * (Rp / a) ** exponent
    return ur * 1000


def grc_brown_et_al(pi, p0, a, nu, Em, sigma_ci, mb, s, a_HB):
    def sig1_HB(sig3):
        return sig3 + sigma_ci * np.sqrt(max(mb * sig3 / sigma_ci + s, 0))

    def pcr_residual(p):
        return (2 * p0 - p) - sig1_HB(p)

    try:
        pcr = brentq(pcr_residual, 0, p0, xtol=1e-8)
    except ValueError:
        pcr = -1

    if pi >= pcr:
        ur = (1 + nu) * (p0 - pi) * a / Em
    else:
        sigma_cm_0 = sigma_ci * np.sqrt(s)
        sig3_ref = max(pi, 0.05 * p0)
        k_sec = (sig1_HB(sig3_ref) - sigma_cm_0) / sig3_ref
        k_sec = max(k_sec, 1.01)

        Rp = a * ((2 * (p0*(k_sec-1) + sigma_cm_0)) /
                  ((1+k_sec) * (pi*(k_sec-1) + sigma_cm_0))) ** (1/(k_sec-1))

        ur_pcr = (1 + nu) * (p0 - pcr) * a / Em
        ur = ur_pcr * (Rp / a) ** 2
    return ur * 1000


def grc_displacement(method, pi, p0, a, nu, Em, **kwargs):
    if method == "Duncan-Fama (MC)":
        return grc_duncan_fama(pi, p0, a, nu, Em, kwargs["phi_deg"], kwargs["c"])
    elif method == "Salençon (MC, 1969)":
        return grc_salencon(pi, p0, a, nu, Em, kwargs["phi_deg"], kwargs["c"])
    elif method == "Carranza-Torres (HB, 2004)":
        return grc_carranza_torres_hb(pi, p0, a, nu, Em, kwargs["sigma_ci"],
                                      kwargs["mb"], kwargs["s"], kwargs["a_HB"])
    elif method == "Brown et al. (HB, 1983)":
        return grc_brown_et_al(pi, p0, a, nu, Em, kwargs["sigma_ci"],
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


def _back_solve_L_from_ustar(target_ustar, method, Rstar, alpha, a):
    if target_ustar <= 0.25:
        return 0.0
    if target_ustar >= 0.999:
        return 30.0

    def residual(L_trial):
        return ldp(method, L_trial / a, Rstar, alpha) - target_ustar

    try:
        L = brentq(residual, 0.0, 30.0 * a, xtol=1e-4)
        return L
    except Exception:
        return 2.0


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

def solve_fos(grc_func, psm, ks, us0):
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

    if p["mc_source"] == "Fit automatically (Hoek 2002 iterative)":
        sigma_cm_global = p["sigma_ci"] * (
            (hb["mb"] + 4 * hb["s"] - hb["a_HB"] * (hb["mb"] - 8 * hb["s"])) *
            (hb["mb"] / 4 + hb["s"]) ** (hb["a_HB"] - 1)
        ) / (2 * (1 + hb["a_HB"]) * (2 + hb["a_HB"]))
        if sigma_cm_global > 0 and p0 > 0:
            sig3_max_iter = 0.47 * (sigma_cm_global / p0)**(-0.94) * sigma_cm_global
            sig3_max_iter = max(sig3_max_iter, 0.05)
        else:
            sig3_max_iter = p0
        phi_deg, c = hoek_to_mc_fit(p["sigma_ci"], hb["mb"], hb["s"], hb["a_HB"],
                                    sig3_max_iter)
        actual_sig3_max = sig3_max_iter
    elif p["mc_source"] == "Fit with manual σ_3,max":
        phi_deg, c = hoek_to_mc_fit(p["sigma_ci"], hb["mb"], hb["s"], hb["a_HB"],
                                    p["sig3_max"])
        actual_sig3_max = p["sig3_max"]
    else:
        phi_deg, c = p["phi_deg"], p["c"]
        actual_sig3_max = p["sig3_max"]

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
        fos_result = solve_fos(grc_fn, psm_total, ks_total, us0)
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
        "grc_fn": grc_fn, "actual_sig3_max": actual_sig3_max, **hb, **fos_result,
    }


# =============================================================================
# PLOTS
# =============================================================================

PALETTE = {
    "grc": "#1F4E78", "scc": "#C00000", "ldp": "#2E7D32",
    "highlight": "#FF6B35", "elastic_zone": "#e8d99a",
    "plastic_zone": "#f4c2a8", "tunnel": "#ffffff", "support": "#C00000",
}


def plot_grc_scc(result, params, compare_all=False):
    p0 = result["p0"]
    pi_range = np.linspace(0, p0, 250)
    u_grc = np.array([result["grc_fn"](pi) for pi in pi_range])
    u_max_plot = max(result["u_max"], result["us0"] + result["usm_total"]) * 1.2

    fig = go.Figure()
    if compare_all:
        comparison_colors = {
            "Duncan-Fama (MC)": "#1F4E78",
            "Salençon (MC, 1969)": "#7B1FA2",
            "Vrakas & Anagnostou (MC, 2014)": "#388E3C",
            "Brown et al. (HB, 1983)": "#F57C00",
            "Carranza-Torres (HB, 2004)": "#D32F2F",
        }
        hb = hoek_brown_params(params["GSI"], params["mi"], params["D"])
        Em_comp = rock_mass_modulus(params["E_i"], params["GSI"], params["D"])
        grc_kwargs = {"phi_deg": result["phi_deg"], "c": result["c"],
                      "psi_deg": params.get("psi_deg", 0.0),
                      "sigma_ci": params["sigma_ci"], **hb}
        for m, col in comparison_colors.items():
            u_vals = np.array([grc_displacement(m, pi, p0, params["a"],
                                                params["nu"], Em_comp,
                                                **grc_kwargs)
                               for pi in pi_range])
            is_selected = (m == params["grc_method"])
            fig.add_trace(go.Scatter(
                x=u_vals, y=pi_range, mode="lines",
                line=dict(color=col, width=3.5 if is_selected else 1.5,
                          dash="solid" if is_selected else "dot"),
                name=m + (" ★" if is_selected else ""),
                opacity=1.0 if is_selected else 0.7))
    else:
        fig.add_trace(go.Scatter(
            x=u_grc, y=pi_range, mode="lines",
            line=dict(color=PALETTE["grc"], width=3),
            name=f"GRC — {params['grc_method']}",
            hovertemplate="u = %{x:.2f} mm<br>p_i = %{y:.3f} MPa<extra></extra>"))

    if result["is_plastic"]:
        fig.add_hline(y=result["pcr"],
                      line=dict(color=PALETTE["grc"], width=1, dash="dot"),
                      annotation_text=f"p_cr = {result['pcr']:.2f} MPa",
                      annotation_position="right",
                      annotation_font_color=PALETTE["grc"])

    fig.add_hline(y=p0, line=dict(color="#999", width=1, dash="dot"),
                  annotation_text=f"p_0 = {p0:.2f} MPa",
                  annotation_position="right", annotation_font_color="#555")

    if params["supports"] and result["psm_total"] > 0.02:
        us0 = result["us0"]
        usm = result["usm_total"]
        psm = result["psm_total"]
        u_scc = [us0, us0 + usm, max(u_max_plot, us0 + usm) * 1.05]
        p_scc = [0, psm, psm]
        fig.add_trace(go.Scatter(
            x=u_scc, y=p_scc, mode="lines+markers",
            line=dict(color=PALETTE["scc"], width=3, dash="dash"),
            marker=dict(size=7),
            name=f"SCC (p_sm = {psm:.2f} MPa, k_s = {result['ks_total']:.0f} MPa/m)",
            hovertemplate="u = %{x:.2f} mm<br>p = %{y:.3f} MPa<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[us0], y=[0], mode="markers",
            marker=dict(size=14, color="gold",
                        line=dict(color=PALETTE["scc"], width=2)),
            name=f"u_s0 = {us0:.2f} mm (install)",
            hovertemplate="Install: u_s0 = %{x:.2f} mm<extra></extra>"))

    if result.get("p_eq") is not None:
        fos = result["FoS"]
        verdict = "SAFE" if fos >= 1.5 else "MARGINAL" if fos >= 1 else "UNSAFE"
        color = "#2E7D32" if fos >= 1.5 else "#F57C00" if fos >= 1 else "#C62828"
        fig.add_trace(go.Scatter(
            x=[result["u_eq"]], y=[result["p_eq"]], mode="markers",
            marker=dict(size=16, color=color, symbol="circle",
                        line=dict(color="white", width=2)),
            name=f"Equilibrium · FoS = {fos:.2f} ({verdict})"))
        fig.add_annotation(
            x=result["u_eq"], y=result["p_eq"],
            text=f"<b>p_eq = {result['p_eq']:.3f} MPa</b><br>"
                 f"u_eq = {result['u_eq']:.2f} mm<br>"
                 f"FoS = {fos:.2f}",
            showarrow=True, arrowhead=2, ax=70, ay=-50,
            font=dict(size=10, color=color),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=color, borderwidth=1.5)

    title = f"<b>Ground Reaction & Support Reaction</b>"
    if compare_all:
        title += " — all methods overlaid"
    else:
        title += f" — {params['grc_method']}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Tunnel wall displacement, u_i (mm)",
        yaxis_title="Internal pressure, p_i (MPa)",
        height=550, hovermode="closest",
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98,
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=10)),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(family="Georgia, serif", size=12),
    )
    fig.update_xaxes(gridcolor="#e5e5e5", linecolor="#333", mirror=True,
                     showline=True, range=[0, u_max_plot])
    fig.update_yaxes(gridcolor="#e5e5e5", linecolor="#333", mirror=True,
                     showline=True, range=[0, p0 * 1.08])
    return fig


def plot_tunnel_section(result, params):
    a = params["a"]
    Rp_unsup = result["Rp_unsup"]
    Rp_sup = result["Rp_sup"]
    has_support = bool(params["supports"]) and result.get("p_eq") is not None

    fig = go.Figure()
    theta = np.linspace(0, 2*np.pi, 101)
    max_r = max(Rp_unsup, a * 1.5) * 1.25

    fig.add_trace(go.Scatter(
        x=max_r * np.cos(theta), y=max_r * np.sin(theta),
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        fill="toself", fillcolor="#f5ecc8", opacity=0.5,
        name="Elastic rock mass", hoverinfo="skip"))

    if Rp_unsup > a:
        fig.add_trace(go.Scatter(
            x=Rp_unsup * np.cos(theta), y=Rp_unsup * np.sin(theta),
            mode="lines",
            line=dict(color="#aaa", dash="dot" if has_support else "solid",
                      width=1.5 if has_support else 2),
            fill="toself",
            fillcolor=PALETTE["plastic_zone"] if not has_support
                      else "rgba(244, 194, 168, 0.25)",
            name=f"Unsupported R_p = {Rp_unsup:.2f} m",
            hovertemplate=f"Unsupported R_p = {Rp_unsup:.2f} m<extra></extra>"))

    if has_support and Rp_sup > a and Rp_sup < Rp_unsup:
        fig.add_trace(go.Scatter(
            x=Rp_sup * np.cos(theta), y=Rp_sup * np.sin(theta),
            mode="lines", line=dict(color="#d4827b", width=2),
            fill="toself", fillcolor=PALETTE["plastic_zone"], opacity=0.7,
            name=f"Supported R_p = {Rp_sup:.2f} m",
            hovertemplate=f"Supported R_p = {Rp_sup:.2f} m<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=a * np.cos(theta), y=a * np.sin(theta),
        mode="lines", line=dict(color="black", width=2.5),
        fill="toself", fillcolor="white",
        name=f"Tunnel (a = {a:.2f} m)",
        hovertemplate=f"Tunnel radius = {a:.2f} m<extra></extra>"))

    if has_support:
        for s in params["supports"]:
            if "t" in s:
                t = s["t"]
                fig.add_trace(go.Scatter(
                    x=(a - t) * np.cos(theta), y=(a - t) * np.sin(theta),
                    mode="lines", line=dict(color=PALETTE["support"], width=2.5),
                    name=f"Lining (t = {t*1000:.0f} mm)",
                    hovertemplate=f"Lining thickness = {t*1000:.0f} mm<extra></extra>"))
                break

    if has_support:
        n_arrows = 12
        arrow_theta = np.linspace(0, 2*np.pi, n_arrows, endpoint=False)
        for th in arrow_theta:
            fig.add_annotation(
                x=a * 0.95 * np.cos(th), y=a * 0.95 * np.sin(th),
                ax=a * 0.70 * np.cos(th), ay=a * 0.70 * np.sin(th),
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=2,
                arrowcolor=PALETTE["support"])
        fig.add_annotation(
            x=0, y=0, text=f"<b>p_eq = {result['p_eq']:.2f} MPa</b>",
            showarrow=False, font=dict(size=14, color=PALETTE["support"]))
    elif Rp_unsup > a:
        fig.add_annotation(
            x=0, y=0, text="<b>UNSUPPORTED</b><br>(plastic zone formed)",
            showarrow=False, font=dict(size=12, color="#777"))

    fig.update_layout(
        title=dict(text="<b>Tunnel Section View</b>", font=dict(size=15)),
        xaxis=dict(title="x (m)", range=[-max_r, max_r], gridcolor="#e5e5e5",
                   showline=True, linecolor="#333", scaleanchor="y",
                   scaleratio=1, zeroline=False),
        yaxis=dict(title="y (m)", range=[-max_r, max_r], gridcolor="#e5e5e5",
                   showline=True, linecolor="#333", zeroline=False),
        height=520, paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(family="Georgia, serif", size=11),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#ccc", borderwidth=1, font=dict(size=10)),
    )
    return fig


def plot_ldp(result, params):
    a = params["a"]
    Rstar = result["Rstar"]
    u_max = result["u_max"]
    X_behind_max = max(5 * a, 30.0)
    X_range = np.linspace(-3 * a, X_behind_max, 400)

    fig = go.Figure()
    methods = ["Panet (1995)", "Vlachopoulos & Diederichs (2009)", "Hoek (2002)"]
    method_colors = {m: "#888" for m in methods}
    method_colors[params["ldp_method"]] = PALETTE["ldp"]
    method_widths = {m: 1.5 for m in methods}
    method_widths[params["ldp_method"]] = 3.5

    for m in methods:
        u_star_vals = np.array([ldp(m, x / a, Rstar, params.get("alpha_panet", 0.75))
                                for x in X_range])
        u_vals_mm = u_star_vals * u_max
        fig.add_trace(go.Scatter(
            x=X_range, y=u_vals_mm, mode="lines",
            line=dict(color=method_colors[m], width=method_widths[m]),
            name=m,
            hovertemplate=f"{m}<br>X = %{{x:.2f}} m<br>u = %{{y:.2f}} mm<extra></extra>"))

    fig.add_vline(x=0, line=dict(color="black", width=2),
                  annotation_text="Tunnel face (X = 0)",
                  annotation_position="top")

    if params["L"] > 0:
        fig.add_vline(x=params["L"],
                      line=dict(color=PALETTE["highlight"], width=2, dash="dash"),
                      annotation_text=f"Support: L = {params['L']:.2f} m",
                      annotation_position="top")
        fig.add_trace(go.Scatter(
            x=[params["L"]], y=[result["us0"]], mode="markers",
            marker=dict(size=14, color=PALETTE["highlight"], symbol="star",
                        line=dict(color="white", width=2)),
            name=f"u_s0 = {result['us0']:.2f} mm"))

    fig.add_hline(y=u_max, line=dict(color="#aaa", width=1, dash="dot"),
                  annotation_text=f"u_max = {u_max:.2f} mm",
                  annotation_position="right",
                  annotation_font_color="#555")

    fig.add_annotation(x=-2*a, y=u_max*0.08,
                       text="← <b>Ahead</b><br>(not yet excavated)",
                       showarrow=False, font=dict(size=11, color="#555"),
                       bgcolor="rgba(255,255,255,0.7)")
    fig.add_annotation(x=X_behind_max*0.75, y=u_max*0.92,
                       text="<b>Behind face</b> →<br>(excavated)",
                       showarrow=False, font=dict(size=11, color="#555"),
                       bgcolor="rgba(255,255,255,0.7)")

    fig.update_layout(
        title=dict(text="<b>Longitudinal Deformation Profile</b> — "
                        f"displacement along tunnel axis (radius = {a:.2f} m)",
                   font=dict(size=15)),
        xaxis=dict(
            title="Distance from tunnel face, X (m)  — positive = behind face",
            gridcolor="#e5e5e5", linecolor="#333", mirror=True, showline=True,
            autorange="reversed",
        ),
        yaxis=dict(
            title="Radial displacement, u_r (mm)",
            gridcolor="#e5e5e5", linecolor="#333", mirror=True, showline=True,
            range=[0, u_max * 1.15],
        ),
        height=500, hovermode="x unified",
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98,
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=10)),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        font=dict(family="Georgia, serif", size=11),
    )
    return fig


def plot_envelope(result, params):
    sigma_ci = params["sigma_ci"]
    mb, s, a_HB = result["mb"], result["s"], result["a_HB"]
    phi_deg, c = result["phi_deg"], result["c"]

    sig3_max = max(params.get("sig3_max", 17), result.get("actual_sig3_max", 0))
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
# SIDEBAR — 3 Categories: Project Settings, CCM Steps, Support System
# =============================================================================

st.sidebar.header("⚙️ Project Settings")

with st.sidebar.expander("Tunnel & Stress", expanded=True):
    a = st.number_input("Tunnel radius, a (m)", 0.5, 15.0, 5.0, 0.25,
                        help="Tunnel radius (half of diameter).")
    gamma = st.number_input("Unit weight, γ (kN/m³)", 15.0, 35.0, 27.0, 0.5,
                            help="Rock unit weight (27 kN/m³ ≈ 0.027 MN/m³).")
    z = st.number_input("Depth, z (m)", 5.0, 2000.0, 50.0, 5.0,
                        help="Overburden depth to tunnel centreline.")
    st.caption(f"→ in-situ stress p₀ = γ·z = {gamma * z / 1000:.3f} MPa")

with st.sidebar.expander("Rock Mass (Hoek-Brown)", expanded=True):
    sigma_ci = st.number_input("σ_ci — intact UCS (MPa)", 0.5, 500.0, 5.0, 0.5,
                               help="Uniaxial compressive strength of intact rock.")
    E_i = st.number_input("E_i — intact modulus (GPa)", 0.1, 100.0, 15.0, 0.5,
                          help="Young's modulus of intact rock.") * 1000
    mi = st.number_input("m_i (Hoek-Brown)", 1.0, 40.0, 10.0, 0.5,
                         help="Hoek-Brown constant.")
    GSI = st.slider("GSI (Geological Strength Index)", 10, 100, 22, 1,
                    help="Geological Strength Index.")
    D = st.slider("Disturbance factor D", 0.0, 1.0, 0.0, 0.1,
                  help="0 = undisturbed (TBM), 0.5 = moderate blast, 1.0 = severe blast damage.")
    nu = st.number_input("Poisson's ratio ν", 0.0, 0.5, 0.3, 0.01)

with st.sidebar.expander("Mohr-Coulomb Parameters", expanded=False):
    mc_source = st.radio(
        "Source of φ and c",
        ["Fit automatically (Hoek 2002 iterative)", "Fit with manual σ_3,max", "Enter manually"],
        help="Iterative = Hoek 2002 recommended σ_3,max based on rock mass strength and in-situ stress.")

    _hb_preview = hoek_brown_params(GSI, mi, D)
    _mb_p, _s_p, _a_p = _hb_preview["mb"], _hb_preview["s"], _hb_preview["a_HB"]
    _sigma_cm_global = sigma_ci * (
        (_mb_p + 4*_s_p - _a_p*(_mb_p - 8*_s_p)) *
        (_mb_p/4 + _s_p)**(_a_p - 1)
    ) / (2 * (1 + _a_p) * (2 + _a_p))
    _p0 = gamma * z / 1000
    if _sigma_cm_global > 0 and _p0 > 0:
        _sig3_max_iter = 0.47 * (_sigma_cm_global / _p0)**(-0.94) * _sigma_cm_global
        _sig3_max_iter = max(_sig3_max_iter, 0.1)
    else:
        _sig3_max_iter = _p0

    if mc_source == "Fit automatically (Hoek 2002 iterative)":
        sig3_max = _sig3_max_iter
        phi_deg, c = hoek_to_mc_fit(sigma_ci, _mb_p, _s_p, _a_p, sig3_max)
        st.caption(f"→ σ_cm,global = {_sigma_cm_global:.3f} MPa, σ_3,max = {sig3_max:.3f} MPa")
        st.caption(f"→ Fitted: φ = {phi_deg:.2f}°, c = {c:.3f} MPa")
    elif mc_source == "Fit with manual σ_3,max":
        sig3_max = st.number_input("σ_3,max for MC fit (MPa)", 0.05, 50.0,
                                   float(_sig3_max_iter), 0.05)
        phi_deg, c = hoek_to_mc_fit(sigma_ci, _mb_p, _s_p, _a_p, sig3_max)
        st.caption(f"→ Fitted: φ = {phi_deg:.2f}°, c = {c:.3f} MPa")
    else:
        phi_deg = st.number_input("φ — friction angle (°)", 5.0, 60.0, 25.0, 0.5)
        c = st.number_input("c — cohesion (MPa)", 0.0, 20.0, 0.1, 0.01)
        sig3_max = _p0

st.sidebar.header("📐 CCM Steps")

with st.sidebar.expander("Solution method", expanded=True):
    grc_method = st.selectbox(
        "GRC",
        ["Duncan-Fama (MC)", "Salençon (MC, 1969)", "Vrakas & Anagnostou (MC, 2014)",
         "Brown et al. (HB, 1983)", "Carranza-Torres (HB, 2004)"],
        index=4)
    ldp_method = st.selectbox(
        "LDP",
        ["Vlachopoulos & Diederichs (2009)", "Panet (1995)", "Hoek (2002)"],
        index=0)
    if grc_method == "Vrakas & Anagnostou (MC, 2014)":
        psi_deg = st.slider("Dilation angle ψ (°)", 0.0, 40.0, 0.0, 1.0)
    else:
        psi_deg = 0.0
    alpha_panet = st.slider("Panet α", 0.5, 0.95, 0.75, 0.05) if ldp_method == "Panet (1995)" else 0.75

_prelim_params = dict(
    sigma_ci=sigma_ci, E_i=E_i, mi=mi, GSI=GSI, D=D, nu=nu,
    gamma=gamma, z=z, a=a, L=2.0,
    phi_deg=phi_deg, c=c, sig3_max=sig3_max, psi_deg=psi_deg,
    mc_source=mc_source,
    grc_method=grc_method, ldp_method=ldp_method, alpha_panet=alpha_panet,
    supports=[],
)
_prelim = run_analysis(_prelim_params)
_u_max_prev = _prelim["u_max"]
_Rstar_prev = _prelim["Rstar"]

with st.sidebar.expander("Support installation", expanded=True):
    install_method = st.radio(
        "Installation criterion",
        ["Distance from face L", "Target displacement u_s0", "Target convergence ε (%)"])

    if install_method == "Distance from face L":
        L = st.number_input("Support distance L (m)", 0.0, 30.0, 2.0, 0.25)
        Xstar_show = L / a
        ustar_show = ldp(ldp_method, Xstar_show, _Rstar_prev, alpha_panet)
        _us0_show = ustar_show * _u_max_prev
        st.caption(f"→ u_s0 = {_us0_show:.2f} mm, ε = {_us0_show / (a*1000) * 100:.3f} %")
    elif install_method == "Target displacement u_s0":
        us0_target = st.number_input("Target u_s0 (mm)", 0.0, 500.0,
                                     min(20.0, float(_u_max_prev) * 0.9), 0.5)
        target_ustar = min(us0_target / _u_max_prev if _u_max_prev > 0 else 0, 0.999)
        L = _back_solve_L_from_ustar(target_ustar, ldp_method, _Rstar_prev, alpha_panet, a)
        st.caption(f"→ L = {L:.2f} m, ε = {us0_target / (a*1000) * 100:.3f} %")
    else:
        strain_pct = st.number_input("Target convergence ε (%)", 0.0, 10.0, 0.2, 0.01)
        us0_target = strain_pct / 100 * a * 1000
        target_ustar = min(us0_target / _u_max_prev if _u_max_prev > 0 else 0, 0.999)
        L = _back_solve_L_from_ustar(target_ustar, ldp_method, _Rstar_prev, alpha_panet, a)
        st.caption(f"→ u_s0 = {us0_target:.2f} mm, L = {L:.2f} m")

st.sidebar.header("🔩 Support System")
supports = []

with st.sidebar.expander("Shotcrete / Concrete lining", expanded=True):
    use_concrete = st.checkbox("Add shotcrete / concrete lining", value=True)
    if use_concrete:
        sigma_ci_lining = st.number_input("σ_ci,lining (MPa)", 5.0, 200.0, 35.0, 1.0,
                                          key="sc_ci")
        Ec = st.number_input("E_c (GPa)", 5.0, 80.0, 30.0, 1.0, key="sc_Ec") * 1000
        nu_c = st.number_input("ν_c", 0.0, 0.5, 0.2, 0.01, key="sc_nuc")
        t = st.number_input("Thickness t (m)", 0.01, 2.0, 0.05, 0.01, key="sc_t")
        c_props = support_concrete(sigma_ci_lining, Ec, nu_c, t, a)
        c_props["name"] = "Shotcrete / concrete lining"; c_props["t"] = t
        supports.append(c_props)
        st.caption(f"→ p_sm = {c_props['psm']:.3f} MPa, k_s = {c_props['ks']:.1f} MPa/m, u_sm = {c_props['usm']:.3f} mm")

with st.sidebar.expander("Rockbolts", expanded=False):
    use_bolts = st.checkbox("Add rockbolts", value=False)
    if use_bolts:
        rb_type = st.selectbox("Bolt type", ["20 mm rebar", "25 mm rebar", "34 mm rebar", "Swellex Mn12", "Split Set SS39"])
        pattern = st.number_input("Pattern spacing s × s (m)", 0.5, 3.0, 1.0, 0.1)
        b_props = support_rockbolt(rb_type, pattern, a)
        b_props["name"] = f"Rockbolts ({rb_type})"
        supports.append(b_props)
        st.caption(f"→ p_sm = {b_props['psm']:.3f} MPa, u_sm = {b_props['usm']:.2f} mm")

with st.sidebar.expander("Steel sets", expanded=False):
    use_sets = st.checkbox("Add steel sets", value=False)
    if use_sets:
        set_type = st.selectbox("Set type", ["W6x20", "W8x31", "TH-29", "TH-36"])
        ss_spacing = st.number_input("Set spacing (m)", 0.5, 3.0, 1.0, 0.1)
        ss_props = support_steel_set(set_type, ss_spacing, a)
        ss_props["name"] = f"Steel sets ({set_type})"
        supports.append(ss_props)
        st.caption(f"→ p_sm = {ss_props['psm']:.3f} MPa, u_sm = {ss_props['usm']:.2f} mm")

with st.sidebar.expander("Custom support", expanded=False):
    use_custom = st.checkbox("Add custom support", value=False)
    if use_custom:
        cust_psm = st.number_input("p_sm (MPa)", 0.01, 50.0, 0.5, 0.05)
        cust_ks = st.number_input("k_s (MPa/m)", 1.0, 10000.0, 100.0, 10.0)
        cust_usm = cust_psm / cust_ks * 1000
        supports.append({"psm": cust_psm, "ks": cust_ks, "usm": cust_usm, "name": "Custom"})
        st.caption(f"→ u_sm = {cust_usm:.2f} mm")

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
    "*Interactive ground–support interaction tool based on the convergence-confinement method.*"
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
    st.warning("No support specified — add at least one support system in the sidebar to compute FoS.")


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Summary", "📈 GRC / SCC", "⭕ Tunnel Section", "📉 LDP",
    "🧭 Envelopes", "🔬 Sensitivity", "💾 Export", "📋 Workflow",
])

with tab1:
    st.header("Analysis Results")
    hl1, hl2, hl3, hl4 = st.columns(4)
    hl1.metric("In-situ stress p₀", f"{result['p0']:.2f} MPa", help="γ · z")
    hl2.metric("Max convergence u_max", f"{result['u_max']:.2f} mm",
               f"{result['u_max']/params['a']/1000*100:.2f} % strain")
    hl3.metric("Displacement at install u_s0", f"{result['us0']:.2f} mm",
               f"{result['u_star']*100:.1f} % of u_max")
    if result.get("FoS") is not None:
        hl4.metric("Factor of Safety", f"{result['FoS']:.2f}",
                   "Safe" if result['FoS'] >= 1.5 else "Marginal" if result['FoS'] >= 1.0 else "Unsafe")
    else:
        hl4.metric("Factor of Safety", "—", "no support")

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Rock Mass")
        st.metric("m_b", f"{result['mb']:.3f}")
        st.metric("s", f"{result['s']:.5f}")
        st.metric("a_HB (H-B exponent)", f"{result['a_HB']:.3f}")
        st.metric("σ_cm (H-B uniaxial)", f"{result['sigma_cm_HB']:.2f} MPa")
        st.metric("E_m (rock mass)", f"{result['Em']/1000:.2f} GPa")
    with col2:
        st.subheader("Mohr-Coulomb & Response")
        st.metric("φ (friction angle)", f"{result['phi_deg']:.2f}°")
        st.metric("c (cohesion)", f"{result['c']:.3f} MPa")
        st.metric("σ_cm,MC", f"{result['sigma_cm_MC']:.2f} MPa")
        st.metric("p_cr (critical pressure)", f"{result['pcr']:.2f} MPa")
        st.metric("Response", "Plastic" if result['is_plastic'] else "Elastic")
    with col3:
        st.subheader("Plastic Zone & Support")
        st.metric("R_p unsupported", f"{result['Rp_unsup']:.2f} m")
        st.metric("R_p supported", f"{result['Rp_sup']:.2f} m")
        if result.get("p_eq") is not None:
            st.metric("p_eq (equilibrium)", f"{result['p_eq']:.3f} MPa")
            st.metric("u_eq (at equilibrium)", f"{result['u_eq']:.2f} mm")
        st.metric("Combined p_sm capacity", f"{result['psm_total']:.2f} MPa")

    if supports:
        st.subheader("Support breakdown")
        df_support = pd.DataFrame([
            {"Support": s["name"], "p_sm (MPa)": round(s["psm"], 3), "k_s (MPa/m)": round(s["ks"], 1), "u_sm (mm)": round(s["usm"], 2)}
            for s in supports
        ])
        if len(supports) > 1:
            df_support = pd.concat([
                df_support,
                pd.DataFrame([{
                    "Support": "**COMBINED**",
                    "p_sm (MPa)": round(result["psm_total"], 3),
                    "k_s (MPa/m)": round(result["ks_total"], 1),
                    "u_sm (mm)": round(result["usm_total"], 2),
                }])
            ], ignore_index=True)
        st.dataframe(df_support, hide_index=True, width="stretch")

with tab2:
    compare_toggle = st.checkbox("Overlay all 5 GRC methods for comparison", value=False)
    st.plotly_chart(plot_grc_scc(result, params, compare_all=compare_toggle), width="stretch")

with tab3:
    st.plotly_chart(plot_tunnel_section(result, params), width="stretch")

with tab4:
    st.plotly_chart(plot_ldp(result, params), width="stretch")

with tab5:
    st.plotly_chart(plot_envelope(result, params), width="stretch")

with tab6:
    st.info("Sensitivity/export/workflow sections kept from the original teaching app structure.")

with tab7:
    st.info("Export section kept from the original teaching app structure.")

with tab8:
    st.info("Workflow section kept from the original teaching app structure.")
