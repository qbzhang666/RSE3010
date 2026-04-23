import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import brentq

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="CCM — RSE3010 (Theme-safe v2)",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-safe CSS: avoid forcing dark/light backgrounds.
# Use subtle borders and spacing only, so Streamlit theme controls the colours.
st.markdown(
    """
<style>
    .main > div { padding-top: 1rem; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    h1, h2, h3 { letter-spacing: -0.01em; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 0.75rem;
        padding: 0.35rem 0.45rem;
    }
    div[data-testid="stExpander"] {
        border-radius: 0.75rem;
        overflow: hidden;
    }
    .small-note {
        opacity: 0.8;
        font-size: 0.92rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("RSE3010 — Convergence–Confinement Method (CCM)")


# =============================================================================
# CORE CALCULATIONS
# =============================================================================

def hoek_brown_params(GSI, mi, D):
    mb = mi * np.exp((GSI - 100.0) / (28.0 - 14.0 * D))
    s = np.exp((GSI - 100.0) / (9.0 - 3.0 * D))
    a_HB = 0.5 + (1.0 / 6.0) * (np.exp(-GSI / 15.0) - np.exp(-20.0 / 3.0))
    return {"mb": mb, "s": s, "a_HB": a_HB}


def rock_mass_modulus(E_i, GSI, D):
    return E_i * (0.02 + (1.0 - D / 2.0) / (1.0 + np.exp((60.0 + 15.0 * D - GSI) / 11.0)))


def hoek_to_mc_fit(sigma_ci, mb, s, a_HB, sig3_max):
    sig3n = sig3_max / sigma_ci
    base = s + mb * sig3n
    pwr = base ** (a_HB - 1.0)
    num_phi = 6.0 * a_HB * mb * pwr
    den_phi = 2.0 * (1.0 + a_HB) * (2.0 + a_HB) + 6.0 * a_HB * mb * pwr
    phi = np.degrees(np.arcsin(np.clip(num_phi / den_phi, -1.0, 1.0)))
    num_c = sigma_ci * ((1.0 + 2.0 * a_HB) * s + (1.0 - a_HB) * mb * sig3n) * pwr
    den_c = (1.0 + a_HB) * (2.0 + a_HB) * np.sqrt(
        1.0 + 6.0 * a_HB * mb * pwr / ((1.0 + a_HB) * (2.0 + a_HB))
    )
    c = num_c / den_c
    return float(phi), float(c)


def critical_pressure_mc(p0, phi_deg, c):
    phi = np.radians(phi_deg)
    k = (1.0 + np.sin(phi)) / (1.0 - np.sin(phi))
    sigma_cm = 2.0 * c * np.cos(phi) / (1.0 - np.sin(phi))
    pcr = max(0.0, (2.0 * p0 - sigma_cm) / (1.0 + k))
    return pcr, k, sigma_cm


def plastic_radius_mc(a, p0, pi, phi_deg, c):
    pcr, k, sigma_cm = critical_pressure_mc(p0, phi_deg, c)
    if pi >= pcr:
        return a
    num = 2.0 * (p0 * (k - 1.0) + sigma_cm)
    den = (1.0 + k) * (pi * (k - 1.0) + sigma_cm)
    if den <= 0 or k <= 1.0:
        return a
    return a * (num / den) ** (1.0 / (k - 1.0))


def grc_duncan_fama(pi, p0, a, nu, Em, phi_deg, c):
    phi = np.radians(phi_deg)
    k = (1.0 + np.sin(phi)) / (1.0 - np.sin(phi))
    sigma_cm = 2.0 * c * np.cos(phi) / (1.0 - np.sin(phi))
    pcr = (2.0 * p0 - sigma_cm) / (1.0 + k)
    if pi >= pcr:
        ur = (1.0 + nu) * (p0 - pi) * a / Em
    else:
        ur_pcr = (1.0 + nu) * (p0 - pcr) * a / Em
        Rp = a * ((2.0 * (p0 * (k - 1.0) + sigma_cm)) / ((1.0 + k) * (pi * (k - 1.0) + sigma_cm))) ** (1.0 / (k - 1.0))
        ur = ur_pcr * (Rp / a) ** 2
    return ur * 1000.0


def grc_salencon(pi, p0, a, nu, Em, phi_deg, c):
    phi = np.radians(phi_deg)
    k = (1.0 + np.sin(phi)) / (1.0 - np.sin(phi))
    sigma_cm = 2.0 * c * np.cos(phi) / (1.0 - np.sin(phi))
    pcr = (2.0 * p0 - sigma_cm) / (1.0 + k)
    if pi >= pcr:
        ur = (1.0 + nu) * (p0 - pi) * a / Em
    else:
        Rp = a * ((2.0 * (p0 * (k - 1.0) + sigma_cm)) / ((1.0 + k) * (pi * (k - 1.0) + sigma_cm))) ** (1.0 / (k - 1.0))
        ur_pcr = (1.0 + nu) * (p0 - pcr) * a / Em
        exponent = (k + 1.0) / (k - 1.0) if k > 1.0 else 2.0
        ur = ur_pcr * (Rp / a) ** exponent
    return ur * 1000.0


def grc_vrakas_anagnostou(pi, p0, a, nu, Em, phi_deg, c, psi_deg=0.0):
    phi = np.radians(phi_deg)
    psi = np.radians(psi_deg)
    k = (1.0 + np.sin(phi)) / (1.0 - np.sin(phi))
    k_psi = (1.0 + np.sin(psi)) / (1.0 - np.sin(psi)) if psi_deg > 0 else 1.0
    sigma_cm = 2.0 * c * np.cos(phi) / (1.0 - np.sin(phi))
    pcr = (2.0 * p0 - sigma_cm) / (1.0 + k)
    if pi >= pcr:
        ur = (1.0 + nu) * (p0 - pi) * a / Em
    else:
        ur_pcr = (1.0 + nu) * (p0 - pcr) * a / Em
        Rp = a * ((2.0 * (p0 * (k - 1.0) + sigma_cm)) / ((1.0 + k) * (pi * (k - 1.0) + sigma_cm))) ** (1.0 / (k - 1.0))
        ur_ss = ur_pcr * (Rp / a) ** 2
        strain = ur_ss / a
        ur = ur_ss * (1.0 + 0.5 * strain * k_psi)
    return ur * 1000.0


def grc_brown_et_al(pi, p0, a, nu, Em, sigma_ci, mb, s, a_HB):
    def sig1_hb(sig3):
        return sig3 + sigma_ci * np.sqrt(max(mb * sig3 / sigma_ci + s, 0.0))
    def pcr_residual(p):
        return (2.0 * p0 - p) - sig1_hb(p)
    try:
        pcr = brentq(pcr_residual, 0.0, p0, xtol=1e-8)
    except ValueError:
        pcr = -1.0
    if pi >= pcr:
        ur = (1.0 + nu) * (p0 - pi) * a / Em
    else:
        sigma_cm_0 = sigma_ci * np.sqrt(max(s, 0.0))
        sig3_ref = max(pi, 0.05 * p0, 1e-6)
        k_sec = max((sig1_hb(sig3_ref) - sigma_cm_0) / sig3_ref, 1.01)
        Rp = a * ((2.0 * (p0 * (k_sec - 1.0) + sigma_cm_0)) / ((1.0 + k_sec) * (pi * (k_sec - 1.0) + sigma_cm_0))) ** (1.0 / (k_sec - 1.0))
        ur_pcr = (1.0 + nu) * (p0 - pcr) * a / Em
        ur = ur_pcr * (Rp / a) ** 2
    return ur * 1000.0


def grc_carranza_torres_hb(pi, p0, a, nu, Em, sigma_ci, mb, s, a_HB):
    def sig1_hb(sig3):
        return sig3 + sigma_ci * max(mb * sig3 / sigma_ci + s, 0.0) ** a_HB
    def pcr_residual(p):
        return (2.0 * p0 - p) - sig1_hb(p)
    try:
        pcr = brentq(pcr_residual, 0.0, p0, xtol=1e-8)
    except ValueError:
        pcr = -1.0
    if pi >= pcr:
        ur = (1.0 + nu) * (p0 - pi) * a / Em
    else:
        tangent_base = max(mb * pi / sigma_ci + s, 1e-10)
        k_tang = max(1.0 + a_HB * mb * tangent_base ** (a_HB - 1.0), 1.01)
        sigma_cm_tang = sig1_hb(pi) - k_tang * pi
        Rp = a * ((2.0 * (p0 * (k_tang - 1.0) + sigma_cm_tang)) / ((1.0 + k_tang) * (pi * (k_tang - 1.0) + sigma_cm_tang))) ** (1.0 / (k_tang - 1.0))
        ur_pcr = (1.0 + nu) * (p0 - pcr) * a / Em
        ur = ur_pcr * (Rp / a) ** 2
    return ur * 1000.0


def grc_displacement(method, pi, p0, a, nu, Em, **kwargs):
    if method == "Duncan-Fama (MC)":
        return grc_duncan_fama(pi, p0, a, nu, Em, kwargs["phi_deg"], kwargs["c"])
    if method == "Salençon (MC, 1969)":
        return grc_salencon(pi, p0, a, nu, Em, kwargs["phi_deg"], kwargs["c"])
    if method == "Vrakas & Anagnostou (MC, 2014)":
        return grc_vrakas_anagnostou(pi, p0, a, nu, Em, kwargs["phi_deg"], kwargs["c"], kwargs.get("psi_deg", 0.0))
    if method == "Brown et al. (HB, 1983)":
        return grc_brown_et_al(pi, p0, a, nu, Em, kwargs["sigma_ci"], kwargs["mb"], kwargs["s"], kwargs["a_HB"])
    if method == "Carranza-Torres (HB, 2004)":
        return grc_carranza_torres_hb(pi, p0, a, nu, Em, kwargs["sigma_ci"], kwargs["mb"], kwargs["s"], kwargs["a_HB"])
    raise ValueError(f"Unknown GRC method: {method}")


# =============================================================================
# LDP
# =============================================================================

def ldp_panet(Xstar, alpha=0.75):
    if Xstar <= 0:
        return (1.0 - alpha) * np.exp(1.5 * Xstar)
    return 1.0 - alpha * np.exp(-1.5 * Xstar)


def ldp_vlachopoulos(Xstar, Rstar):
    Rstar = max(Rstar, 1.0)
    if Xstar <= 0:
        return (1.0 / 3.0) * np.exp(2.0 * Xstar - 0.15 * Rstar)
    return 1.0 - (1.0 - (1.0 / 3.0) * np.exp(-0.15 * Rstar)) * np.exp(-3.0 * Xstar / Rstar)


def ldp_hoek(Xstar):
    Xd = Xstar / 2.0
    return (1.0 + np.exp(-Xd / 1.1)) ** (-1.7)


def ldp(method, Xstar, Rstar=1.0, alpha=0.75):
    if method == "Panet (1995)":
        return ldp_panet(Xstar, alpha)
    if method == "Vlachopoulos & Diederichs (2009)":
        return ldp_vlachopoulos(Xstar, Rstar)
    if method == "Hoek (2002)":
        return ldp_hoek(Xstar)
    raise ValueError(f"Unknown LDP method: {method}")


def back_solve_L_from_ustar(target_ustar, method, Rstar, alpha, a, L_min=-2.0, L_max=30.0):
    target_ustar = float(np.clip(target_ustar, 1e-6, 0.999))
    def residual(L_trial):
        return ldp(method, L_trial / a, Rstar, alpha) - target_ustar
    lo = L_min * a
    hi = L_max * a
    xs = np.linspace(lo, hi, 400)
    vals = np.array([residual(x) for x in xs])
    for i in range(len(xs) - 1):
        if vals[i] == 0:
            return xs[i]
        if vals[i] * vals[i + 1] < 0:
            try:
                return brentq(residual, xs[i], xs[i + 1], xtol=1e-5)
            except Exception:
                break
    return 3.0


# =============================================================================
# SUPPORTS
# =============================================================================

def support_concrete(sigma_ci, Ec, nu_c, t, a):
    psm = sigma_ci * t / a
    ks = Ec * t / ((1.0 - nu_c ** 2) * a ** 2)
    usm = psm / ks * 1000.0 if ks > 0 else 0.0
    return {"type": "Concrete lining", "psm": psm, "ks": ks, "usm": usm, "t": t}


def support_rockbolt(rb_type, pattern_spacing, a):
    props = {
        "20 mm rebar": {"T": 184.0, "strain": 0.0024},
        "25 mm rebar": {"T": 287.0, "strain": 0.0024},
        "34 mm rebar": {"T": 500.0, "strain": 0.0024},
        "Swellex Mn12": {"T": 110.0, "strain": 0.0100},
        "Split Set SS39": {"T": 100.0, "strain": 0.0150},
    }
    p = props.get(rb_type, props["34 mm rebar"])
    psm = p["T"] / 1000.0 / (pattern_spacing ** 2)
    usm = p["strain"] * a * 1000.0
    ks = psm / (usm / 1000.0) if usm > 0 else 0.0
    return {"type": f"Rockbolt ({rb_type})", "psm": psm, "ks": ks, "usm": usm}


def support_steel_set(set_type, spacing, a):
    props = {
        "W6x20": {"A": 38.0e-4, "sigma_y": 248.0},
        "W8x31": {"A": 58.9e-4, "sigma_y": 248.0},
        "TH-29": {"A": 37.0e-4, "sigma_y": 345.0},
        "TH-36": {"A": 45.7e-4, "sigma_y": 345.0},
    }
    p = props.get(set_type, props["W8x31"])
    psm = p["sigma_y"] * p["A"] / (spacing * a)
    E_steel = 200000.0
    ks = E_steel * p["A"] / (spacing * a ** 2)
    usm = psm / ks * 1000.0 if ks > 0 else 0.0
    return {"type": f"Steel set ({set_type})", "psm": psm, "ks": ks, "usm": usm}


def support_custom(psm, ks):
    usm = psm / ks * 1000.0 if ks > 0 else 0.0
    return {"type": "Custom", "psm": psm, "ks": ks, "usm": usm}


def support_response_at_u(u_mm, us0_mm, supports):
    if not supports:
        return 0.0
    du_m = max(0.0, (u_mm - us0_mm) / 1000.0)
    p_total = 0.0
    for s in supports:
        p_total += min(s["ks"] * du_m, s["psm"])
    return p_total


def support_capacity_total(supports):
    return sum(s["psm"] for s in supports) if supports else 0.0


# =============================================================================
# EQUILIBRIUM SOLVER
# =============================================================================

def build_grc_u_of_p(grc_method, p0, a, nu, Em, grc_kwargs):
    def grc_u(pi):
        return grc_displacement(grc_method, pi, p0, a, nu, Em, **grc_kwargs)
    return grc_u


def build_physical_grc_curve(grc_fn_raw, p0, n=800):
    """
    Build a monotonic GRC for plotting and equilibrium solving.

    The raw approximate H-B implementations can occasionally produce a small
    non-monotonic tail near low confinement. Physically, tunnel wall displacement
    should increase monotonically as internal pressure decreases.
    """
    p_desc = np.linspace(p0, 0.0, n)
    u_raw_desc = np.array([grc_fn_raw(p) for p in p_desc], dtype=float)
    u_phys_desc = np.maximum.accumulate(u_raw_desc)
    was_corrected = bool(np.any(np.diff(u_raw_desc) < -1e-9))
    return p_desc, u_raw_desc, u_phys_desc, was_corrected


def make_monotonic_grc_interpolator(grc_fn_raw, p0, n=800):
    p_desc, u_raw_desc, u_phys_desc, was_corrected = build_physical_grc_curve(grc_fn_raw, p0, n=n)
    p_asc = p_desc[::-1]
    u_asc = u_phys_desc[::-1]

    def grc_u(p):
        p_clip = float(np.clip(p, 0.0, p0))
        return float(np.interp(p_clip, p_asc, u_asc))

    return grc_u, p_desc, u_raw_desc, u_phys_desc, was_corrected


def solve_equilibrium(grc_u, supports, us0_mm, p0, n_grid=600):
    if not supports:
        return {"p_eq": None, "u_eq": None, "FoS": None, "saturated": False, "valid": False}
    psm_total = support_capacity_total(supports)
    p_grid = np.linspace(0.0, min(p0, psm_total), n_grid)
    u_grid = np.array([grc_u(p) for p in p_grid])
    p_scc_grid = np.array([support_response_at_u(u, us0_mm, supports) for u in u_grid])
    residual = p_grid - p_scc_grid
    if np.all(residual < 0):
        return {"p_eq": psm_total, "u_eq": float(np.max(u_grid)), "FoS": 1.0, "saturated": True, "valid": True}
    sign_change_idx = None
    for i in range(len(residual) - 1):
        if residual[i] == 0 or residual[i] * residual[i + 1] < 0:
            sign_change_idx = i
            break
    if sign_change_idx is None:
        j = int(np.argmin(np.abs(residual)))
        p_eq = p_grid[j]
        u_eq = u_grid[j]
        fos = psm_total / p_eq if p_eq > 1e-9 else None
        return {"p_eq": float(p_eq), "u_eq": float(u_eq), "FoS": float(fos) if fos is not None else None, "saturated": False, "valid": True}
    p_lo = p_grid[sign_change_idx]
    p_hi = p_grid[min(sign_change_idx + 1, len(p_grid) - 1)]
    def residual_p(p):
        u = grc_u(p)
        return p - support_response_at_u(u, us0_mm, supports)
    try:
        p_eq = brentq(residual_p, p_lo, p_hi, xtol=1e-7, maxiter=200)
        u_eq = grc_u(p_eq)
        fos = psm_total / p_eq if p_eq > 1e-9 else None
        return {"p_eq": float(p_eq), "u_eq": float(u_eq), "FoS": float(fos) if fos is not None else None, "saturated": False, "valid": True}
    except Exception:
        j = int(np.argmin(np.abs(residual)))
        p_eq = p_grid[j]
        u_eq = u_grid[j]
        fos = psm_total / p_eq if p_eq > 1e-9 else None
        return {"p_eq": float(p_eq), "u_eq": float(u_eq), "FoS": float(fos) if fos is not None else None, "saturated": False, "valid": True}


# =============================================================================
# ANALYSIS
# =============================================================================

def run_analysis(params):
    p = params
    p0 = p["gamma"] * p["z"] / 1000.0
    hb = hoek_brown_params(p["GSI"], p["mi"], p["D"])
    Em = rock_mass_modulus(p["E_i"], p["GSI"], p["D"])
    sigma_cm_HB = p["sigma_ci"] * hb["s"] ** hb["a_HB"]

    if p["mc_source"] == "Fit automatically (Hoek 2002)":
        phi_deg, c = hoek_to_mc_fit(p["sigma_ci"], hb["mb"], hb["s"], hb["a_HB"], p["sig3_max"])
    else:
        phi_deg, c = p["phi_deg"], p["c"]

    pcr, k, sigma_cm_MC = critical_pressure_mc(p0, phi_deg, c)
    is_plastic = p0 > pcr
    Rp_unsup = plastic_radius_mc(p["a"], p0, 0.0, phi_deg, c) if is_plastic else p["a"]
    Rstar = Rp_unsup / p["a"]

    grc_kwargs = {
        "phi_deg": phi_deg,
        "c": c,
        "psi_deg": p.get("psi_deg", 0.0),
        "sigma_ci": p["sigma_ci"],
        **hb,
    }
    grc_fn_raw = build_grc_u_of_p(p["grc_method"], p0, p["a"], p["nu"], Em, grc_kwargs)
    grc_fn, p_desc_grc, u_raw_desc_grc, u_phys_desc_grc, grc_was_corrected = make_monotonic_grc_interpolator(grc_fn_raw, p0)
    u_max = grc_fn(0.0)

    Xstar = p["L"] / p["a"]
    u_star = ldp(p["ldp_method"], Xstar, Rstar, p.get("alpha_panet", 0.75))
    us0 = u_star * u_max

    supports = p["supports"]
    psm_total = support_capacity_total(supports)
    if supports:
        fos_result = solve_equilibrium(grc_fn, supports, us0, p0)
    else:
        fos_result = {"p_eq": None, "u_eq": None, "FoS": None, "saturated": False, "valid": False}

    if fos_result["p_eq"] is not None and is_plastic:
        Rp_sup = plastic_radius_mc(p["a"], p0, fos_result["p_eq"], phi_deg, c)
    else:
        Rp_sup = Rp_unsup

    warnings = []
    if Rp_unsup / p["a"] > 3.0:
        warnings.append("Large plastic zone (R_p/a > 3). Check assumptions and consider severe squeezing behaviour.")
    if fos_result.get("FoS") is not None and fos_result["FoS"] < 1.0:
        warnings.append("FoS < 1.0. Current support is inadequate.")
    if p["grc_method"] in ["Brown et al. (HB, 1983)", "Carranza-Torres (HB, 2004)"]:
        warnings.append("Selected H-B GRC method is an approximate teaching implementation.")
    if grc_was_corrected:
        warnings.append("Raw GRC from the selected method showed a small non-monotonic segment; a monotonic physical envelope has been enforced for plotting and equilibrium calculations.")

    return {
        "p0": p0,
        "Em": Em,
        "phi_deg": phi_deg,
        "c": c,
        "k": k,
        "sigma_cm_MC": sigma_cm_MC,
        "sigma_cm_HB": sigma_cm_HB,
        "pcr": pcr,
        "is_plastic": is_plastic,
        "Rp_unsup": Rp_unsup,
        "Rp_sup": Rp_sup,
        "Rstar": Rstar,
        "u_max": u_max,
        "Xstar": Xstar,
        "u_star": u_star,
        "us0": us0,
        "psm_total": psm_total,
        "grc_fn": grc_fn,
        "grc_fn_raw": grc_fn_raw,
        "grc_p_desc": p_desc_grc,
        "grc_u_raw_desc": u_raw_desc_grc,
        "grc_u_phys_desc": u_phys_desc_grc,
        "grc_was_corrected": grc_was_corrected,
        "supports": supports,
        "warnings": warnings,
        **hb,
        **fos_result,
    }


# =============================================================================
# PLOTS (theme-safe)
# =============================================================================

PLOT_COLORS = {
    "grc": "#2E86DE",
    "scc": "#E74C3C",
    "ldp": "#27AE60",
    "highlight": "#8E44AD",
    "neutral": "#7F8C8D",
}


def plot_grc_scc(result, params):
    p0 = result["p0"]
    p_desc = result.get("grc_p_desc", np.linspace(p0, 0.0, 300))
    u_desc = result.get("grc_u_phys_desc", np.array([result["grc_fn"](p) for p in p_desc]))

    max_u = max(result["u_max"], result["us0"] + 50.0)
    if result.get("u_eq") is not None:
        max_u = max(max_u, result["u_eq"] * 1.2)
    u_plot = np.linspace(0.0, max_u * 1.15, 400)
    p_scc = np.array([support_response_at_u(u, result["us0"], result["supports"]) for u in u_plot]) if params["supports"] else np.zeros_like(u_plot)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_desc, y=p_desc, mode="lines", line=dict(color=PLOT_COLORS["grc"], width=3), name="GRC"))
    if params["supports"]:
        fig.add_trace(go.Scatter(x=u_plot, y=p_scc, mode="lines", line=dict(color=PLOT_COLORS["scc"], width=3, dash="dash"), name="SCC"))
        fig.add_trace(go.Scatter(x=[result["us0"]], y=[0.0], mode="markers", marker=dict(size=10, color=PLOT_COLORS["highlight"]), name="u_s0"))
    if result.get("p_eq") is not None:
        fig.add_trace(go.Scatter(x=[result["u_eq"]], y=[result["p_eq"]], mode="markers", marker=dict(size=12, color="black"), name=f"Intersection (FoS={result['FoS']:.2f})"))
    if result["is_plastic"]:
        fig.add_hline(y=result["pcr"], line=dict(color=PLOT_COLORS["neutral"], dash="dot"), annotation_text=f"p_cr = {result['pcr']:.2f} MPa")

    fig.update_layout(
        title="Ground Reaction & Support Reaction",
        xaxis_title="Tunnel wall displacement uᵢ (mm)",
        yaxis_title="Support pressure pᵢ (MPa)",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(borderwidth=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    return fig


def plot_ldp(result, params):
    Rstar = result["Rstar"]
    Xstar_range = np.linspace(-5.0, 10.0, 400)
    fig = go.Figure()
    methods = ["Panet (1995)", "Vlachopoulos & Diederichs (2009)", "Hoek (2002)"]
    for m in methods:
        width = 3.2 if m == params["ldp_method"] else 1.5
        color = PLOT_COLORS["ldp"] if m == params["ldp_method"] else PLOT_COLORS["neutral"]
        u_vals = np.array([ldp(m, x, Rstar, params.get("alpha_panet", 0.75)) * 100.0 for x in Xstar_range])
        fig.add_trace(go.Scatter(x=Xstar_range, y=u_vals, mode="lines", line=dict(color=color, width=width), name=m))
    fig.add_vline(x=0.0, line=dict(color=PLOT_COLORS["neutral"], width=2), annotation_text="Tunnel face")
    fig.add_vline(x=result["Xstar"], line=dict(color=PLOT_COLORS["highlight"], width=2, dash="dash"), annotation_text=f"L = {params['L']:.2f} m")
    fig.add_trace(go.Scatter(x=[result["Xstar"]], y=[result["u_star"] * 100.0], mode="markers", marker=dict(size=11, color=PLOT_COLORS["highlight"]), name="u* point"))
    fig.update_layout(
        title="Longitudinal Deformation Profile",
        xaxis_title="Normalised distance X* = X/a",
        yaxis_title="u / u_max (%)",
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)", autorange="reversed")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)", range=[0, 110])
    return fig


def plot_tunnel_section(result, params):
    a = params["a"]
    Rp_unsup = result["Rp_unsup"]
    Rp_sup = result["Rp_sup"]
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    fig = go.Figure()
    if Rp_unsup > a:
        fig.add_trace(go.Scatter(x=Rp_unsup * np.cos(theta), y=Rp_unsup * np.sin(theta), mode="lines", line=dict(color=PLOT_COLORS["neutral"], dash="dot"), name=f"Unsupported R_p = {Rp_unsup:.2f} m"))
    if params["supports"] and Rp_sup > a:
        fig.add_trace(go.Scatter(x=Rp_sup * np.cos(theta), y=Rp_sup * np.sin(theta), mode="lines", line=dict(color=PLOT_COLORS["scc"], width=2), name=f"Supported R_p = {Rp_sup:.2f} m"))
    fig.add_trace(go.Scatter(x=a * np.cos(theta), y=a * np.sin(theta), mode="lines", line=dict(color=PLOT_COLORS["grc"], width=3), name=f"Tunnel radius = {a:.2f} m"))
    fig.update_layout(
        title="Tunnel Section View",
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(scaleanchor="y", scaleratio=1, title="x (m)"),
        yaxis=dict(title="y (m)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    return fig


def plot_envelope(result, params):
    sigma_ci = params["sigma_ci"]
    mb, s, a_HB = result["mb"], result["s"], result["a_HB"]
    phi_deg, c = result["phi_deg"], result["c"]
    sig3_max = params.get("sig3_max", max(5.0, result["p0"]))
    sig3 = np.linspace(0.0, sig3_max, 100)
    sig1_hb = sig3 + sigma_ci * (mb * sig3 / sigma_ci + s) ** a_HB
    phi = np.radians(phi_deg)
    k = (1.0 + np.sin(phi)) / (1.0 - np.sin(phi))
    sig1_mc = 2.0 * c * np.cos(phi) / (1.0 - np.sin(phi)) + k * sig3
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sig3, y=sig1_hb, mode="lines", line=dict(color=PLOT_COLORS["grc"], width=3), name="Hoek-Brown"))
    fig.add_trace(go.Scatter(x=sig3, y=sig1_mc, mode="lines", line=dict(color=PLOT_COLORS["scc"], width=2, dash="dash"), name=f"MC fit (φ={phi_deg:.1f}°, c={c:.2f})"))
    fig.update_layout(
        title="H-B vs MC Failure Envelopes",
        xaxis_title="σ₃ (MPa)",
        yaxis_title="σ₁ (MPa)",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("Project Settings")

with st.sidebar.expander("Solution method", expanded=True):
    grc_options = ["Duncan-Fama (MC)", "Salençon (MC, 1969)", "Vrakas & Anagnostou (MC, 2014)", "Brown et al. (HB, 1983)", "Carranza-Torres (HB, 2004)"]
    grc_method = st.selectbox("GRC", grc_options, index=0)
    ldp_method = st.selectbox("LDP", ["Vlachopoulos & Diederichs (2009)", "Panet (1995)", "Hoek (2002)"])
    alpha_panet = st.slider("Panet α", 0.5, 0.95, 0.75, 0.05) if ldp_method == "Panet (1995)" else 0.75

with st.sidebar.expander("Project & stress", expanded=True):
    a = st.number_input("Tunnel radius a (m)", 0.5, 20.0, 4.75, 0.25)
    gamma = st.number_input("Unit weight γ (kN/m³)", 15.0, 35.0, 26.0, 0.5)
    z = st.number_input("Depth z (m)", 5.0, 2000.0, 650.0, 10.0)
    st.caption(f"p₀ = γ·z = {gamma * z / 1000.0:.2f} MPa")

with st.sidebar.expander("Rock mass (Hoek-Brown)", expanded=True):
    sigma_ci = st.number_input("σ_ci (MPa)", 1.0, 500.0, 72.0, 1.0)
    E_i = st.number_input("E_i (GPa)", 0.1, 100.0, 26.0, 0.5) * 1000.0
    mi = st.number_input("m_i", 1.0, 40.0, 17.0, 0.5)
    GSI = st.slider("GSI", 10, 100, 60, 5)
    D = st.slider("Disturbance factor D", 0.0, 1.0, 0.0, 0.1)
    nu = st.number_input("Poisson's ratio ν", 0.0, 0.49, 0.22, 0.01)

with st.sidebar.expander("MC parameters", expanded=True):
    mc_source = st.radio("Source", ["Fit automatically (Hoek 2002)", "Enter manually"])
    if mc_source == "Fit automatically (Hoek 2002)":
        sig3_max = st.number_input("σ₃,max for MC fit (MPa)", 0.1, 100.0, float(gamma * z / 1000.0), 0.5)
        _hb = hoek_brown_params(GSI, mi, D)
        _phi, _c = hoek_to_mc_fit(sigma_ci, _hb["mb"], _hb["s"], _hb["a_HB"], sig3_max)
        st.caption(f"φ = {_phi:.2f}°, c = {_c:.3f} MPa")
        phi_deg, c = _phi, _c
    else:
        phi_deg = st.number_input("φ (deg)", 5.0, 60.0, 31.5, 0.5)
        c = st.number_input("c (MPa)", 0.0, 30.0, 2.85, 0.1)
        sig3_max = gamma * z / 1000.0
    psi_deg = st.slider("Dilation angle ψ (deg)", 0.0, 40.0, 0.0, 1.0) if grc_method == "Vrakas & Anagnostou (MC, 2014)" else 0.0

_prelim_params = dict(
    sigma_ci=sigma_ci, E_i=E_i, mi=mi, GSI=GSI, D=D, nu=nu,
    gamma=gamma, z=z, a=a, L=3.0,
    phi_deg=phi_deg, c=c, sig3_max=sig3_max, psi_deg=psi_deg,
    mc_source=mc_source, grc_method=grc_method, ldp_method=ldp_method,
    alpha_panet=alpha_panet, supports=[]
)
_prelim = run_analysis(_prelim_params)
_u_max_prev = _prelim["u_max"]
_Rstar_prev = _prelim["Rstar"]

with st.sidebar.expander("Installation timing", expanded=True):
    install_method = st.radio("Criterion", ["Distance from face L", "Target displacement u_s0", "Target convergence ε (%)"])
    if install_method == "Distance from face L":
        L = st.number_input("L (m)", -2.0 * a, 30.0 * a, 3.0, 0.5)
        us0_show = ldp(ldp_method, L / a, _Rstar_prev, alpha_panet) * _u_max_prev
        st.caption(f"u_s0 = {us0_show:.2f} mm")
    elif install_method == "Target displacement u_s0":
        us0_target = st.number_input("Target u_s0 (mm)", 0.0, 500.0, min(20.0, float(_u_max_prev)), 1.0)
        L = back_solve_L_from_ustar(us0_target / max(_u_max_prev, 1e-9), ldp_method, _Rstar_prev, alpha_panet, a)
        st.caption(f"Back-solved L = {L:.2f} m")
    else:
        eps_target = st.number_input("Target ε (%)", 0.0, 10.0, 0.3, 0.05)
        us0_target = eps_target / 100.0 * a * 1000.0
        L = back_solve_L_from_ustar(us0_target / max(_u_max_prev, 1e-9), ldp_method, _Rstar_prev, alpha_panet, a)
        st.caption(f"Target u_s0 = {us0_target:.2f} mm, back-solved L = {L:.2f} m")

with st.sidebar.expander("Support system", expanded=True):
    supports = []
    include_lining = st.checkbox("Concrete lining", value=True)
    if include_lining:
        c1, c2 = st.columns(2)
        with c1:
            lining_sigma = st.number_input("Lining σ_ci (MPa)", 1.0, 150.0, 55.0, 1.0)
            lining_E = st.number_input("Lining E (GPa)", 1.0, 80.0, 32.0, 1.0) * 1000.0
        with c2:
            lining_nu = st.number_input("Lining ν", 0.0, 0.49, 0.22, 0.01)
            lining_t = st.number_input("Lining thickness t (m)", 0.01, 2.0, 0.50, 0.01)
        supports.append(support_concrete(lining_sigma, lining_E, lining_nu, lining_t, a))
    include_bolts = st.checkbox("Rockbolts", value=False)
    if include_bolts:
        bolt_type = st.selectbox("Bolt type", ["20 mm rebar", "25 mm rebar", "34 mm rebar", "Swellex Mn12", "Split Set SS39"])
        bolt_spacing = st.number_input("Bolt spacing (m)", 0.4, 5.0, 1.5, 0.1)
        supports.append(support_rockbolt(bolt_type, bolt_spacing, a))
    include_steel = st.checkbox("Steel sets", value=False)
    if include_steel:
        set_type = st.selectbox("Steel set type", ["W6x20", "W8x31", "TH-29", "TH-36"])
        set_spacing = st.number_input("Steel set spacing (m)", 0.4, 5.0, 1.0, 0.1)
        supports.append(support_steel_set(set_type, set_spacing, a))
    include_custom = st.checkbox("Custom support", value=False)
    if include_custom:
        custom_psm = st.number_input("Custom p_sm (MPa)", 0.01, 200.0, 1.0, 0.1)
        custom_ks = st.number_input("Custom k_s (MPa/m)", 0.1, 1e6, 1000.0, 100.0)
        supports.append(support_custom(custom_psm, custom_ks))
    st.caption(f"Total p_sm,total = {support_capacity_total(supports):.2f} MPa" if supports else "No support selected")

# =============================================================================
# RUN
# =============================================================================

params = dict(
    sigma_ci=sigma_ci, E_i=E_i, mi=mi, GSI=GSI, D=D, nu=nu,
    gamma=gamma, z=z, a=a, L=L, phi_deg=phi_deg, c=c,
    sig3_max=sig3_max, psi_deg=psi_deg, mc_source=mc_source,
    grc_method=grc_method, ldp_method=ldp_method,
    alpha_panet=alpha_panet, supports=supports,
)
result = run_analysis(params)

# =============================================================================
# LAYOUT
# =============================================================================

if result["warnings"]:
    for w in result["warnings"]:
        st.warning(w)

with st.expander("Calculation workflow", expanded=False):
    st.markdown(
        """
1. Compute in-situ stress **p₀ = γz**  
2. Compute Hoek–Brown parameters **m_b, s, a**  
3. Compute rock mass modulus **E_m** and equivalent **MC parameters**  
4. Compute **critical pressure p_cr** and unsupported **R_p**  
5. Compute unsupported maximum displacement **u_max**  
6. Use the **LDP** to obtain **u_s0**  
7. Build the **combined SCC** from selected supports  
8. Solve the **GRC–SCC intersection** for **p_eq, u_eq, FoS**
"""
    )

m1, m2, m3, m4 = st.columns(4)
m1.metric("p₀ (MPa)", f"{result['p0']:.2f}")
m2.metric("p_cr (MPa)", f"{result['pcr']:.2f}")
m3.metric("R_p unsupported (m)", f"{result['Rp_unsup']:.2f}")
m4.metric("u_max (mm)", f"{result['u_max']:.2f}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("u_s0 (mm)", f"{result['us0']:.2f}")
m6.metric("p_sm,total (MPa)", f"{result['psm_total']:.2f}")
m7.metric("p_eq (MPa)", "—" if result['p_eq'] is None else f"{result['p_eq']:.2f}")
m8.metric("FoS", "—" if result['FoS'] is None else f"{result['FoS']:.2f}")

with st.expander("Summary table", expanded=True):
    summary = {
        "GRC method": params["grc_method"],
        "LDP method": params["ldp_method"],
        "Depth z (m)": params["z"],
        "Radius a (m)": params["a"],
        "p0 (MPa)": result["p0"],
        "Em (MPa)": result["Em"],
        "phi (deg)": result["phi_deg"],
        "c (MPa)": result["c"],
        "pcr (MPa)": result["pcr"],
        "Rp_unsup (m)": result["Rp_unsup"],
        "Rp_sup (m)": result["Rp_sup"],
        "u_max (mm)": result["u_max"],
        "L (m)": params["L"],
        "X*": result["Xstar"],
        "u*": result["u_star"],
        "us0 (mm)": result["us0"],
        "psm_total (MPa)": result["psm_total"],
        "peq (MPa)": result["p_eq"],
        "ueq (mm)": result["u_eq"],
        "FoS": result["FoS"],
        "Rstar": result["Rstar"],
        "is_plastic": result["is_plastic"],
    }
    st.table({"Parameter": list(summary.keys()), "Value": [f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v) for v in summary.values()]})

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(plot_grc_scc(result, params), use_container_width=True)
    st.plotly_chart(plot_ldp(result, params), use_container_width=True)
with c2:
    st.plotly_chart(plot_tunnel_section(result, params), use_container_width=True)
    st.plotly_chart(plot_envelope(result, params), use_container_width=True)

with st.expander("Support components", expanded=False):
    if supports:
        rows = []
        for i, s in enumerate(supports, start=1):
            rows.append({
                "#": i,
                "Type": s["type"],
                "p_sm (MPa)": f"{s['psm']:.3f}",
                "k_s (MPa/m)": f"{s['ks']:.3f}",
                "u_sm (mm)": f"{s['usm']:.3f}",
            })
        st.table(rows)
    else:
        st.info("No support elements selected.")


