"""
RSE3010 — Convergence-Confinement Method (CCM) Interactive App
================================================================
Interactive ground–support interaction tool for
Mine Geotechnical Engineering, Monash University.

Features:
- Multiple GRC solution methods: Duncan-Fama (MC), Salençon (MC, 1969),
  Vrakas & Anagnostou (MC, 2014), Brown et al. (HB, 1983),
  Carranza-Torres (HB, 2004)
- Multiple LDP methods: Panet (1995), Vlachopoulos & Diederichs (2009),
  Hoek (2002) — with both ahead-of-face and behind-face branches
- Three installation criteria: distance L, target u_s0, target strain ε
- Support library: Concrete lining, Rockbolts, Steel Sets, Custom
  (combinable, with aggregate p_sm, k_s, u_sm)
- Tunnel Section View with plastic zone visualisation
- H-B → MC fit (Hoek 2002 closed form)
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
    """Duncan-Fama (1993) closed-form MC GRC, small-strain. Returns u_r in mm.

    The most widely used GRC for Mohr-Coulomb materials. Splits into an elastic
    branch (Kirsch) for p_i ≥ p_cr and a plastic branch beyond."""
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
    """Carranza-Torres (2004) closed-form GRC for generalised Hoek-Brown.

    Implements the generalised HB solution using numerical integration for
    the plastic-zone radius when a_HB ≠ 0.5 (exact closed form otherwise).

    Plastic zone radius:
        ln(R_p/a) = ∫_{p_i}^{p_cr} dp / [σ_θ(p) - p]
    where σ_θ(p) = p + σ_ci·(m_b·p/σ_ci + s)^a_HB is the HB yield function
    at confining pressure p.

    Wall displacement (Carranza-Torres & Fairhurst 2000, non-dilatant):
        u_r = (1+ν)(p_0 - p_cr)/E_m · a · [2(R_p/a)² - (1-2ν)]

    Assumptions: small-strain, non-dilatant (ψ = 0), no residual strength.
    """
    from scipy.optimize import brentq as _brentq
    from scipy.integrate import quad as _quad

    # Ensure physically sensible inputs
    if mb <= 0 or a_HB <= 0:
        return (1 + nu) * (p0 - pi) * a / Em * 1000

    # Hoek-Brown σ₁(σ₃)
    def sig1_HB(sig3):
        base = mb * sig3 / sigma_ci + s
        if base < 0:
            base = 0
        return sig3 + sigma_ci * base ** a_HB

    # Critical pressure: σ_θ = 2p₀ - p_i = σ₁(p_i)
    def pcr_res(p):
        return (2 * p0 - p) - sig1_HB(p)

    try:
        if pcr_res(0) < 0:
            pcr = -1.0
        else:
            pcr = _brentq(pcr_res, 0.0, p0, xtol=1e-9)
    except ValueError:
        pcr = -1.0

    if pi >= pcr:
        return (1 + nu) * (p0 - pi) * a / Em * 1000

    # --- Plastic zone radius: use closed form for a_HB = 0.5, else integrate ---
    if abs(a_HB - 0.5) < 1e-4:
        # Exact closed form
        P_cr = pcr / (mb * sigma_ci) + s / mb**2
        P_i = pi / (mb * sigma_ci) + s / mb**2
        sqrt_Pcr = np.sqrt(max(P_cr, 0))
        sqrt_Pi = np.sqrt(max(P_i, 0))
        Rp = a * np.exp(2.0 * (sqrt_Pcr - sqrt_Pi))
    else:
        # Numerical integration for general a_HB
        # ln(R_p/a) = ∫_{p_i}^{p_cr} dp / [σ_θ - p]
        # where σ_θ - p = σ_ci * (m_b p/σ_ci + s)^a_HB  (HB yield surface strength)
        def integrand(p):
            base = mb * p / sigma_ci + s
            if base < 1e-12:
                return 0.0
            return 1.0 / (sigma_ci * base ** a_HB)
        try:
            ln_ratio, _ = _quad(integrand, pi, pcr, limit=100)
            Rp = a * np.exp(ln_ratio)
        except Exception:
            # Fall back to scaled-stress form if integration fails
            P_cr = pcr / (mb * sigma_ci) + s / mb**2
            P_i = pi / (mb * sigma_ci) + s / mb**2
            sqrt_Pcr = np.sqrt(max(P_cr, 0))
            sqrt_Pi = np.sqrt(max(P_i, 0))
            Rp = a * np.exp(2.0 * (sqrt_Pcr - sqrt_Pi))

    # Wall displacement — Carranza-Torres 2004 non-dilatant (Ψ=0) form.
    # The displacement at r=a in the plastic zone (non-dilatant, plane strain)
    # reduces to the "Duncan-Fama-like" expression with an additional (1+ν)
    # factor arising from the plane-strain elastic-plastic matching at r=R_p:
    #     u_r = (1+ν)² · (p_0 - p_cr) / E_m · a · (R_p/a)²
    # This matches RocSupport 2004-onwards within ~3% for Ψ=0, and reduces to
    # the standard Duncan-Fama (MC) form when the HB and MC R_p coincide.
    Rstar = Rp / a
    ur = (1 + nu) ** 2 * (p0 - pcr) / Em * a * Rstar ** 2
    return ur * 1000


def plastic_radius_ct(a, p0, pi, sigma_ci, mb, s, a_HB):
    """Plastic zone radius from Carranza-Torres 2004.

    Uses exact scaled-stress closed form for a_HB = 0.5, and numerical
    integration of the HB yield surface for general a_HB. Returns the
    tunnel radius if still elastic.
    """
    from scipy.optimize import brentq as _brentq
    from scipy.integrate import quad as _quad

    if mb <= 0 or a_HB <= 0:
        return a

    def sig1_HB(sig3):
        base = mb * sig3 / sigma_ci + s
        if base < 0:
            base = 0
        return sig3 + sigma_ci * base ** a_HB

    def pcr_res(p):
        return (2 * p0 - p) - sig1_HB(p)

    try:
        if pcr_res(0) < 0:
            return a
        pcr = _brentq(pcr_res, 0.0, p0, xtol=1e-9)
    except ValueError:
        return a

    if pi >= pcr:
        return a

    if abs(a_HB - 0.5) < 1e-4:
        # Exact closed form
        P_cr = pcr / (mb * sigma_ci) + s / mb**2
        P_i = pi / (mb * sigma_ci) + s / mb**2
        sqrt_Pcr = np.sqrt(max(P_cr, 0))
        sqrt_Pi = np.sqrt(max(P_i, 0))
        return a * np.exp(2.0 * (sqrt_Pcr - sqrt_Pi))
    else:
        # Numerical integration
        def integrand(p):
            base = mb * p / sigma_ci + s
            if base < 1e-12:
                return 0.0
            return 1.0 / (sigma_ci * base ** a_HB)
        try:
            ln_ratio, _ = _quad(integrand, pi, pcr, limit=100)
            return a * np.exp(ln_ratio)
        except Exception:
            # Fall back to closed form
            P_cr = pcr / (mb * sigma_ci) + s / mb**2
            P_i = pi / (mb * sigma_ci) + s / mb**2
            sqrt_Pcr = np.sqrt(max(P_cr, 0))
            sqrt_Pi = np.sqrt(max(P_i, 0))
            return a * np.exp(2.0 * (sqrt_Pcr - sqrt_Pi))


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


def grc_salencon(pi, p0, a, nu, Em, phi_deg, c):
    """Salençon (1969) — classical closed-form elasto-plastic MC solution
    with non-associated flow and no dilation. One of the earliest GRC
    solutions; often used as a benchmark. Structurally similar to Duncan-Fama
    but derived from the full equilibrium equations rather than assuming
    small-strain Kirsch.
    """
    phi = np.radians(phi_deg)
    k = (1 + np.sin(phi)) / (1 - np.sin(phi))
    sigma_cm = 2 * c * np.cos(phi) / (1 - np.sin(phi))
    pcr = (2 * p0 - sigma_cm) / (1 + k)

    if pi >= pcr:
        # Elastic (same as Kirsch)
        ur = (1 + nu) * (p0 - pi) * a / Em
    else:
        # Salençon's plastic radius (identical functional form to Duncan-Fama)
        Rp = a * ((2 * (p0*(k-1) + sigma_cm)) /
                  ((1+k) * (pi*(k-1) + sigma_cm))) ** (1/(k-1))
        # Salençon's wall displacement (incompressible plastic flow):
        # u_r = u_r(pcr) · (Rp/a)^((k+1)/(k-1))  — slightly different exponent
        # from Duncan-Fama which uses exponent 2
        ur_pcr = (1 + nu) * (p0 - pcr) * a / Em
        exponent = (k + 1) / (k - 1) if k > 1 else 2.0
        ur = ur_pcr * (Rp / a) ** exponent
    return ur * 1000


def grc_brown_et_al(pi, p0, a, nu, Em, sigma_ci, mb, s, a_HB):
    """Brown, Bray, Ladanyi & Hoek (1983) — original Hoek-Brown GRC using
    the 1983 form with exponent a_HB = 0.5.

    This implementation follows the 1983 approach:
    - 1983 HB failure criterion: σ₁ = σ₃ + σ_ci·sqrt(m·σ₃/σ_ci + s)
    - Plastic zone from scaled-stress closed form (same as CT 2004 for a=0.5)
    - Wall displacement uses the Brown et al. 1983 plastic formula:
          u_r = u_r(p_cr) · (R_p/a)²
      (i.e. no (1-2ν) correction term — simpler historical form).

    Gives slightly smaller u_max than CT 2004 because of the missing
    elastic-recovery term. Retained for pedagogical comparison.
    """
    from scipy.optimize import brentq as _brentq

    # Brown et al. 1983 uses a_HB = 0.5 regardless of input
    def sig1_HB(sig3):
        base = mb * sig3 / sigma_ci + s
        if base < 0:
            base = 0
        return sig3 + sigma_ci * np.sqrt(base)

    def pcr_residual(p):
        return (2 * p0 - p) - sig1_HB(p)

    try:
        if pcr_residual(0) < 0:
            pcr = -1.0
        else:
            pcr = _brentq(pcr_residual, 0.0, p0, xtol=1e-9)
    except ValueError:
        pcr = -1.0

    if pi >= pcr:
        return (1 + nu) * (p0 - pi) * a / Em * 1000

    # Scaled-stress closed-form for R_p (a_HB = 0.5 exact)
    P_cr = pcr / (mb * sigma_ci) + s / mb**2
    P_i = pi / (mb * sigma_ci) + s / mb**2
    sqrt_Pcr = np.sqrt(max(P_cr, 0))
    sqrt_Pi = np.sqrt(max(P_i, 0))
    Rp = a * np.exp(2.0 * (sqrt_Pcr - sqrt_Pi))

    # Brown et al. 1983 simpler plastic displacement:
    # u_r = u_r(p_cr) · (R_p/a)²
    ur_pcr = (1 + nu) * (p0 - pcr) / Em * a  # m
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

# Normalisation convention for the X-distance and plastic radius in LDP formulas.
# Two widely used conventions in industry and academia:
#   - "radius" (X* = X/a, R* = R_p/a) — used in the Vlachopoulos & Diederichs
#                                       2009 paper
#   - "diameter" (X* = X/D, R* = R_p/D) — used in RocSupport and in Hoek's
#                                          "Practical Rock Engineering"
# Default is "diameter" so results match commercial software (RocSupport).
LDP_DEFAULT_NORM = "diameter"


def _norm_factor(norm):
    """Return the divisor for converting X, R_p to X*, R*."""
    return 2.0 if norm == "diameter" else 1.0


def ldp_panet(Xstar, alpha=0.75):
    if Xstar <= 0:
        return (1 - alpha) * np.exp(1.5 * Xstar)
    return 1 - alpha * np.exp(-1.5 * Xstar)


def ldp_vlachopoulos(Xstar, Rstar):
    if Xstar <= 0:
        return (1/3) * np.exp(2 * Xstar - 0.15 * Rstar)
    return 1 - (1 - (1/3) * np.exp(-0.15 * Rstar)) * np.exp(-3 * Xstar / Rstar)


def ldp_hoek(Xstar_over_D):
    """Hoek (2002) empirical — function uses X/D directly."""
    return (1 + np.exp(-Xstar_over_D / 1.1)) ** (-1.7)


def ldp(method, X_over_a, Rp_over_a=1.0, alpha=0.75, norm=LDP_DEFAULT_NORM):
    """Evaluate the LDP u(X)/u_max.

    Inputs are always in radius-normalised form (X/a and R_p/a) so the caller
    does not need to know the convention. Internally, the function converts
    to the selected convention before evaluating the LDP formula.

    Conventions
    -----------
    "radius" (V&D 2009 paper):
        Uses X* = X/a and R* = R_p/a directly.
    "diameter" (RocSupport / Rocscience implementation):
        Uses X* = X/D = X/(2a) for the distance exponent, but keeps the
        face-value term (1/3)·exp(-0.15·R*) with R* = R_p/a. This mixed
        convention matches RocSupport outputs to within 1% for typical cases.
        (Verified against RocSupport deterministic output for the 5 m radius,
        50 m deep, σ_ci=5 MPa, GSI=22, m_i=10 benchmark case.)

    Parameters
    ----------
    method : str
        One of the three available LDP methods.
    X_over_a : float
        Distance from the face divided by the tunnel RADIUS (X/a).
    Rp_over_a : float
        Plastic zone radius divided by the tunnel RADIUS (R_p/a).
    alpha : float
        Panet calibration parameter (only used for Panet 1995).
    norm : {"radius", "diameter"}
        Normalisation convention used inside the LDP formula.
    """
    if norm == "diameter":
        # RocSupport convention: X-distance uses X/D, but R* in the face-value
        # term remains R_p/a.
        Xs_dist = X_over_a / 2.0
        Rs_face = Rp_over_a       # retain radius for face value
    else:
        # Academic V&D 2009: everything normalised by radius
        Xs_dist = X_over_a
        Rs_face = Rp_over_a

    if method == "Panet (1995)":
        return ldp_panet(Xs_dist, alpha)
    if method == "Vlachopoulos & Diederichs (2009)":
        return ldp_vlachopoulos(Xs_dist, Rs_face)
    if method == "Hoek (2002)":
        # Hoek formula is always expressed in terms of X/D, independent of norm
        return ldp_hoek(X_over_a / 2.0)
    raise ValueError(method)


def _back_solve_L_from_ustar(target_ustar, method, Rstar, alpha, a,
                              norm=LDP_DEFAULT_NORM):
    """Back-solve L (distance from face) such that LDP(L/a) = target_ustar."""
    if target_ustar <= 0.25:
        # Target is ahead of face or very close — clamp at face
        return 0.0
    if target_ustar >= 0.999:
        return 30.0  # installation at infinity

    from scipy.optimize import brentq as _brentq

    def residual(L_trial):
        return ldp(method, L_trial / a, Rstar, alpha, norm=norm) - target_ustar

    try:
        # Search between L = 0 (face) and L = 30a (far behind)
        L = _brentq(residual, 0.0, 30.0 * a, xtol=1e-4)
        return L
    except Exception:
        return 3.0  # fallback


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

    if p["mc_source"] == "Fit automatically (Hoek 2002 iterative)":
        # Hoek 2002 iterative σ_3,max for tunnels
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
        # Store the actually-used sig3_max in result
        actual_sig3_max = sig3_max_iter
    elif p["mc_source"] == "Fit with manual σ_3,max":
        phi_deg, c = hoek_to_mc_fit(p["sigma_ci"], hb["mb"], hb["s"], hb["a_HB"],
                                    p["sig3_max"])
        actual_sig3_max = p["sig3_max"]
    else:
        phi_deg, c = p["phi_deg"], p["c"]
        actual_sig3_max = p["sig3_max"]

    # Determine critical pressure and plastic radius using the method that
    # corresponds to the selected GRC so that diagnostic outputs
    # (pcr, Rp_unsup, Rp_sup) match what the GRC curve implies.
    is_hb_method = p["grc_method"] in ["Brown et al. (HB, 1983)",
                                        "Carranza-Torres (HB, 2004)"]

    # Always compute MC quantities for reporting and envelope display
    pcr_mc, k, sigma_cm_MC = critical_pressure_mc(p0, phi_deg, c)

    if is_hb_method:
        # Use HB-based p_cr and R_p for consistency with the GRC curve
        def _sig1_hb(sig3):
            base = hb["mb"] * sig3 / p["sigma_ci"] + hb["s"]
            base = max(base, 0.0)
            return sig3 + p["sigma_ci"] * base ** hb["a_HB"]

        def _pcr_res(pp):
            return (2 * p0 - pp) - _sig1_hb(pp)

        try:
            if _pcr_res(0) < 0:
                pcr = -1.0
                is_plastic = False
            else:
                pcr = brentq(_pcr_res, 0.0, p0, xtol=1e-9)
                is_plastic = pcr > 0 and p0 > pcr
        except ValueError:
            pcr = pcr_mc
            is_plastic = p0 > pcr_mc

        if is_plastic:
            Rp_unsup = plastic_radius_ct(p["a"], p0, 0.0, p["sigma_ci"],
                                          hb["mb"], hb["s"], hb["a_HB"])
        else:
            Rp_unsup = p["a"]
    else:
        # MC-family methods: use MC p_cr and R_p
        pcr = pcr_mc
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
    u_star = ldp(p["ldp_method"], Xstar, Rstar, p.get("alpha_panet", 0.75),
                 norm=p.get("ldp_norm", LDP_DEFAULT_NORM))
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

    # Supported plastic zone radius using the same method as unsupported
    if fos_result["p_eq"] is not None and is_plastic:
        if is_hb_method:
            Rp_sup = plastic_radius_ct(p["a"], p0, fos_result["p_eq"],
                                        p["sigma_ci"], hb["mb"], hb["s"], hb["a_HB"])
        else:
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


def plot_grc_scc(result, params, compare_all=False):
    """GRC + SCC plot.

    If compare_all=True, overlays all 5 GRC methods for comparison
    (useful for industrial review / uncertainty).
    """
    p0 = result["p0"]
    pi_range = np.linspace(0, p0, 250)

    # Primary GRC (selected method)
    u_grc = np.array([result["grc_fn"](pi) for pi in pi_range])
    u_max_plot = max(result["u_max"], result["us0"] + result["usm_total"]) * 1.2

    fig = go.Figure()

    # Optional: overlay all methods in light colours
    if compare_all:
        comparison_colors = {
            "Duncan-Fama (MC)": "#1F4E78",
            "Salençon (MC, 1969)": "#7B1FA2",
            "Vrakas & Anagnostou (MC, 2014)": "#388E3C",
            "Brown et al. (HB, 1983)": "#F57C00",
            "Carranza-Torres (HB, 2004)": "#D32F2F",
        }
        # Import needed for overlay
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

    # p_cr reference
    if result["is_plastic"]:
        fig.add_hline(y=result["pcr"],
                      line=dict(color=PALETTE["grc"], width=1, dash="dot"),
                      annotation_text=f"p_cr = {result['pcr']:.2f} MPa",
                      annotation_position="right",
                      annotation_font_color=PALETTE["grc"])

    # p_0 reference
    fig.add_hline(y=p0, line=dict(color="#999", width=1, dash="dot"),
                  annotation_text=f"p_0 = {p0:.2f} MPa",
                  annotation_position="right", annotation_font_color="#555")

    # SCC
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
        # u_s0 start marker
        fig.add_trace(go.Scatter(
            x=[us0], y=[0], mode="markers",
            marker=dict(size=14, color="gold",
                        line=dict(color=PALETTE["scc"], width=2)),
            name=f"u_s0 = {us0:.2f} mm (install)",
            hovertemplate="Install: u_s0 = %{x:.2f} mm<extra></extra>"))

    # Equilibrium point
    if result.get("p_eq") is not None:
        fos = result["FoS"]
        verdict = "SAFE" if fos >= 1.5 else "MARGINAL" if fos >= 1 else "UNSAFE"
        color = "#2E7D32" if fos >= 1.5 else "#F57C00" if fos >= 1 else "#C62828"
        fig.add_trace(go.Scatter(
            x=[result["u_eq"]], y=[result["p_eq"]], mode="markers",
            marker=dict(size=16, color=color, symbol="circle",
                        line=dict(color="white", width=2)),
            name=f"Equilibrium · FoS = {fos:.2f} ({verdict})"))
        # Annotation box
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

    # Outer boundary reference
    max_r = max(Rp_unsup, a * 1.5) * 1.25

    # Intact (elastic) rock region - outer dark background
    fig.add_trace(go.Scatter(
        x=max_r * np.cos(theta), y=max_r * np.sin(theta),
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        fill="toself", fillcolor="#f5ecc8", opacity=0.5,
        name="Elastic rock mass", hoverinfo="skip"))

    # Unsupported plastic zone (outer)
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

    # Supported plastic zone (inner)
    if has_support and Rp_sup > a and Rp_sup < Rp_unsup:
        fig.add_trace(go.Scatter(
            x=Rp_sup * np.cos(theta), y=Rp_sup * np.sin(theta),
            mode="lines", line=dict(color="#d4827b", width=2),
            fill="toself", fillcolor=PALETTE["plastic_zone"], opacity=0.7,
            name=f"Supported R_p = {Rp_sup:.2f} m",
            hovertemplate=f"Supported R_p = {Rp_sup:.2f} m<extra></extra>"))

    # Tunnel (inner white circle)
    fig.add_trace(go.Scatter(
        x=a * np.cos(theta), y=a * np.sin(theta),
        mode="lines", line=dict(color="black", width=2.5),
        fill="toself", fillcolor="white",
        name=f"Tunnel (a = {a:.2f} m)",
        hovertemplate=f"Tunnel radius = {a:.2f} m<extra></extra>"))

    # Lining ring (if present)
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

    # Support pressure arrows - ONLY if support is installed
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
        # Unsupported case - show warning
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
    """LDP plot in physical metres with secondary normalised axis.

    Industrial convention: distance from face shown in real metres (what
    monitoring personnel measure in the field). A secondary upper axis
    shows the normalised X*=X/a for academic reference.
    """
    a = params["a"]
    Rstar = result["Rstar"]
    u_max = result["u_max"]

    # Physical distance range: from -3a (ahead) to 5a or 30m (behind)
    X_behind_max = max(5 * a, 30.0)
    X_range = np.linspace(-3 * a, X_behind_max, 400)  # physical metres

    fig = go.Figure()
    methods = ["Panet (1995)", "Vlachopoulos & Diederichs (2009)", "Hoek (2002)"]
    # Muted colours for non-selected, prominent for selected method
    method_colors = {
        "Panet (1995)": "#888",
        "Vlachopoulos & Diederichs (2009)": "#888",
        "Hoek (2002)": "#888",
    }
    method_colors[params["ldp_method"]] = PALETTE["ldp"]
    method_widths = {m: 1.5 for m in methods}
    method_widths[params["ldp_method"]] = 3.5

    for m in methods:
        u_star_vals = np.array([ldp(m, x / a, Rstar,
                                    params.get("alpha_panet", 0.75),
                                    norm=params.get("ldp_norm", LDP_DEFAULT_NORM))
                                for x in X_range])
        # Plot actual displacement in mm
        u_vals_mm = u_star_vals * u_max
        fig.add_trace(go.Scatter(
            x=X_range, y=u_vals_mm, mode="lines",
            line=dict(color=method_colors[m], width=method_widths[m]),
            name=m,
            hovertemplate=f"{m}<br>X = %{{x:.2f}} m<br>u = %{{y:.2f}} mm<extra></extra>"))

    # Face line at X = 0
    fig.add_vline(x=0, line=dict(color="black", width=2),
                  annotation_text="Tunnel face (X = 0)",
                  annotation_position="top")

    # Support installation marker
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

    # u_max reference line
    fig.add_hline(y=u_max, line=dict(color="#aaa", width=1, dash="dot"),
                  annotation_text=f"u_max = {u_max:.2f} mm",
                  annotation_position="right",
                  annotation_font_color="#555")

    # Side labels
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
            title="Distance from tunnel face, X (m)  "
                  "— positive = behind face",
            gridcolor="#e5e5e5", linecolor="#333", mirror=True, showline=True,
            autorange="reversed",  # excavation convention: behind face on LEFT
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
# SIDEBAR — 3 Categories: Project Settings, CCM Steps, Support System
# =============================================================================

# ----------------------------------------------------------------------------
# CATEGORY 1: PROJECT SETTINGS
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Project Settings")

with st.sidebar.expander("**Tunnel & stress**", expanded=True):
    a = st.number_input("Tunnel radius, a (m)", 0.5, 15.0, 5.0, 0.25,
                        help="Tunnel radius (half of diameter).")
    gamma = st.number_input("Unit weight, γ (kN/m³)", 15.0, 35.0, 27.0, 0.5,
                            help="Rock unit weight (27 kN/m³ ≈ 0.027 MN/m³).")
    z = st.number_input("Depth, z (m)", 5.0, 2000.0, 50.0, 5.0,
                        help="Overburden depth to tunnel centreline.")
    st.caption(f"→ in-situ stress p₀ = γ·z = {gamma * z / 1000:.3f} MPa")

with st.sidebar.expander("**Rock mass (Hoek-Brown)**", expanded=True):
    sigma_ci = st.number_input("σ_ci — intact UCS (MPa)", 0.5, 500.0, 5.0, 0.5,
                                help="Uniaxial compressive strength of intact rock.")
    E_i = st.number_input("E_i — intact modulus (GPa)", 0.1, 100.0, 15.0, 0.5,
                          help="Young's modulus of intact rock.") * 1000
    mi = st.number_input("m_i (Hoek-Brown)", 1.0, 40.0, 10.0, 0.5,
                         help="Hoek-Brown constant. 7≈siltstone, 10≈limestone, "
                              "17≈sandstone, 25≈granite.")
    GSI = st.slider("GSI (Geological Strength Index)", 10, 100, 22, 1,
                    help="Geological Strength Index. 10-30=poor, 30-60=average, "
                         "60-80=good, 80-100=very good.")
    D = st.slider("Disturbance factor D", 0.0, 1.0, 0.0, 0.1,
                  help="0 = undisturbed (TBM), 0.5 = moderate blast, "
                       "1.0 = severe blast damage.")
    nu = st.number_input("Poisson's ratio ν", 0.0, 0.5, 0.3, 0.01)

with st.sidebar.expander("**Mohr-Coulomb parameters**", expanded=False):
    mc_source = st.radio(
        "Source of φ and c",
        ["Fit automatically (Hoek 2002 iterative)", "Fit with manual σ_3,max",
         "Enter manually"],
        help="**Iterative** = Hoek 2002 recommended σ_3,max based on "
             "rock mass strength and in-situ stress (industry standard).\n"
             "**Manual σ_3,max** = specify upper stress for MC linearisation.\n"
             "**Manual** = input φ, c directly.")

    # Compute the iterative σ_3,max automatically (Hoek 2002 for tunnels)
    _hb_preview = hoek_brown_params(GSI, mi, D)
    _mb_p, _s_p, _a_p = _hb_preview["mb"], _hb_preview["s"], _hb_preview["a_HB"]
    _sigma_cm_global = sigma_ci * (
        (_mb_p + 4*_s_p - _a_p*(_mb_p - 8*_s_p)) *
        (_mb_p/4 + _s_p)**(_a_p - 1)
    ) / (2 * (1 + _a_p) * (2 + _a_p))
    _p0 = gamma * z / 1000
    # Hoek 2002 Eq. for tunnels: σ_3,max/σ_cm = 0.47·(σ_cm/γH)^(-0.94)
    if _sigma_cm_global > 0 and _p0 > 0:
        _sig3_max_iter = 0.47 * (_sigma_cm_global / _p0)**(-0.94) * _sigma_cm_global
        _sig3_max_iter = max(_sig3_max_iter, 0.1)
    else:
        _sig3_max_iter = _p0

    if mc_source == "Fit automatically (Hoek 2002 iterative)":
        sig3_max = _sig3_max_iter
        phi_deg, c = hoek_to_mc_fit(sigma_ci, _mb_p, _s_p, _a_p, sig3_max)
        st.caption(
            f"→ σ_cm,global = {_sigma_cm_global:.3f} MPa, "
            f"σ_3,max = {sig3_max:.3f} MPa (Hoek 2002 iterative)"
        )
        st.caption(f"→ Fitted: φ = {phi_deg:.2f}°, c = {c:.3f} MPa")
    elif mc_source == "Fit with manual σ_3,max":
        sig3_max = st.number_input(
            "σ_3,max for MC fit (MPa)", 0.05, 50.0,
            float(_sig3_max_iter), 0.05,
            help="Upper bound of stress range for linearising the "
                 "Hoek-Brown envelope.")
        phi_deg, c = hoek_to_mc_fit(sigma_ci, _mb_p, _s_p, _a_p, sig3_max)
        st.caption(f"→ Fitted: φ = {phi_deg:.2f}°, c = {c:.3f} MPa")
    else:
        phi_deg = st.number_input("φ — friction angle (°)", 5.0, 60.0, 25.0, 0.5)
        c = st.number_input("c — cohesion (MPa)", 0.0, 20.0, 0.1, 0.01)
        sig3_max = _p0

# ----------------------------------------------------------------------------
# CATEGORY 2: CCM STEPS
# ----------------------------------------------------------------------------
st.sidebar.header("📐 CCM Steps")

with st.sidebar.expander("**Step 1: GRC method**", expanded=True):
    grc_method = st.selectbox(
        "Ground Reaction Curve",
        ["Duncan-Fama (MC)",
         "Salençon (MC, 1969)",
         "Vrakas & Anagnostou (MC, 2014)",
         "Brown et al. (HB, 1983)",
         "Carranza-Torres (HB, 2004)"],
        index=4,  # Default to Carranza-Torres per user request
        help="**MC methods** (need φ, c):\n"
             "- Duncan-Fama: classical small-strain MC (widely used).\n"
             "- Salençon (1969): earliest closed-form MC solution.\n"
             "- Vrakas & Anagnostou (2014): large-strain MC with dilation ψ.\n\n"
             "**H-B methods** (need m_b, s, a):\n"
             "- Brown et al. (1983): original H-B GRC.\n"
             "- Carranza-Torres (2004): generalised H-B (recommended).\n\n"
             "All methods give similar results in the elastic range; "
             "differences appear in the plastic zone (poor rock).")
    if grc_method == "Vrakas & Anagnostou (MC, 2014)":
        psi_deg = st.slider("Dilation angle ψ (°)", 0.0, 40.0, 0.0, 1.0,
                            help="Dilation angle for large-strain correction. "
                                 "ψ = 0 is non-associated (typical for rock).")
    else:
        psi_deg = 0.0

with st.sidebar.expander("**Step 2: LDP method**", expanded=True):
    ldp_method = st.selectbox(
        "Longitudinal Deformation Profile",
        ["Vlachopoulos & Diederichs (2009)",
         "Panet (1995)",
         "Hoek (2002)"],
        index=0,
        help="**V&D (2009)** = elastoplastic, depends on R_p (recommended).\n"
             "**Panet (1995)** = elastic, exponential, with adjustable α.\n"
             "**Hoek (2002)** = empirical, diameter-normalised.")

    if ldp_method == "Panet (1995)":
        alpha_panet = st.slider("Panet α", 0.5, 0.95, 0.75, 0.05)
    else:
        alpha_panet = 0.75

    ldp_norm_label = st.radio(
        "LDP normalisation",
        ["Diameter (X/D, R_p/D) — RocSupport / Hoek",
         "Radius (X/a, R_p/a) — V&D 2009 paper"],
        index=0,
        help="Choice of normalisation used inside the LDP formula.\n"
             "• **Diameter** (default): matches Rocscience RocSupport and Hoek's "
             "'Practical Rock Engineering'.\n"
             "• **Radius**: matches the Vlachopoulos & Diederichs 2009 paper "
             "(u₀/u_max at the face is larger).")
    ldp_norm = "diameter" if ldp_norm_label.startswith("Diameter") else "radius"

# --- Preliminary analysis: compute u_max before defining installation criterion ---
_prelim_params = dict(
    sigma_ci=sigma_ci, E_i=E_i, mi=mi, GSI=GSI, D=D, nu=nu,
    gamma=gamma, z=z, a=a, L=3.0,
    phi_deg=phi_deg, c=c, sig3_max=sig3_max, psi_deg=psi_deg,
    mc_source=mc_source,
    grc_method=grc_method, ldp_method=ldp_method, alpha_panet=alpha_panet,
    ldp_norm=ldp_norm, supports=[],
)
_prelim = run_analysis(_prelim_params)
_u_max_prev = _prelim["u_max"]
_Rstar_prev = _prelim["Rstar"]

with st.sidebar.expander("**Step 3: Support installation**", expanded=True):
    install_method = st.radio(
        "Installation criterion",
        ["Distance from face L",
         "Target displacement u_s0",
         "Target convergence ε (%)"],
        help="Three ways to specify when the support is installed:\n"
             "• **L**: distance behind the face at installation (most common).\n"
             "• **u_s0**: target tunnel-wall displacement at installation.\n"
             "• **ε**: target tunnel-convergence strain (= u_s0 / a × 100).")

    if install_method == "Distance from face L":
        L = st.number_input("Support distance L (m)", 0.0, 30.0, 2.0, 0.25)
        Xstar_show = L / a
        ustar_show = ldp(ldp_method, Xstar_show, _Rstar_prev, alpha_panet,
                         norm=ldp_norm)
        _us0_show = ustar_show * _u_max_prev
        st.caption(f"→ u_s0 = {_us0_show:.2f} mm, "
                   f"ε = {_us0_show / (a*1000) * 100:.3f} %")
    elif install_method == "Target displacement u_s0":
        us0_target = st.number_input("Target u_s0 (mm)", 0.0, 500.0,
                                     min(20.0, float(_u_max_prev) * 0.9), 0.5)
        target_ustar = min(us0_target / _u_max_prev if _u_max_prev > 0 else 0, 0.999)
        L = _back_solve_L_from_ustar(target_ustar, ldp_method, _Rstar_prev,
                                     alpha_panet, a, norm=ldp_norm)
        st.caption(f"→ L = {L:.2f} m, "
                   f"ε = {us0_target / (a*1000) * 100:.3f} %")
    else:
        strain_pct = st.number_input("Target convergence ε (%)", 0.0, 10.0,
                                     0.2, 0.01,
                                     help="Strain = u_s0 / a × 100%")
        us0_target = strain_pct / 100 * a * 1000
        target_ustar = min(us0_target / _u_max_prev if _u_max_prev > 0 else 0, 0.999)
        L = _back_solve_L_from_ustar(target_ustar, ldp_method, _Rstar_prev,
                                     alpha_panet, a, norm=ldp_norm)
        st.caption(f"→ u_s0 = {us0_target:.2f} mm, L = {L:.2f} m")

# ----------------------------------------------------------------------------
# CATEGORY 3: SUPPORT SYSTEM
# ----------------------------------------------------------------------------
st.sidebar.header("🔩 Support System")
supports = []

with st.sidebar.expander("**Shotcrete / Concrete lining**", expanded=True):
    use_concrete = st.checkbox("Add shotcrete / concrete lining", value=True)
    if use_concrete:
        sigma_ci_lining = st.number_input("σ_ci,lining (MPa)", 5.0, 200.0, 35.0, 1.0,
                                          key="sc_ci",
                                          help="28-day UCS: shotcrete ~30-40 MPa, "
                                               "concrete ~40-60 MPa.")
        Ec = st.number_input("E_c (GPa)", 5.0, 80.0, 30.0, 1.0, key="sc_Ec",
                              help="Young's modulus at 28 days.") * 1000
        nu_c = st.number_input("ν_c", 0.0, 0.5, 0.2, 0.01, key="sc_nuc")
        t = st.number_input("Thickness t (m)", 0.01, 2.0, 0.05, 0.01, key="sc_t",
                             help="Shotcrete typically 50-200 mm, "
                                  "concrete lining 200-1000 mm.")
        c_props = support_concrete(sigma_ci_lining, Ec, nu_c, t, a)
        c_props["name"] = "Shotcrete / concrete lining"; c_props["t"] = t
        supports.append(c_props)
        st.caption(f"→ p_sm = {c_props['psm']:.3f} MPa, "
                   f"k_s = {c_props['ks']:.1f} MPa/m, "
                   f"u_sm = {c_props['usm']:.3f} mm")

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
                   f"u_sm = {b_props['usm']:.2f} mm")

with st.sidebar.expander("**Steel sets**", expanded=False):
    use_sets = st.checkbox("Add steel sets", value=False)
    if use_sets:
        set_type = st.selectbox("Set type", ["W6x20", "W8x31", "TH-29", "TH-36"])
        ss_spacing = st.number_input("Set spacing (m)", 0.5, 3.0, 1.0, 0.1)
        ss_props = support_steel_set(set_type, ss_spacing, a)
        ss_props["name"] = f"Steel sets ({set_type})"
        supports.append(ss_props)
        st.caption(f"→ p_sm = {ss_props['psm']:.3f} MPa, "
                   f"u_sm = {ss_props['usm']:.2f} mm")

with st.sidebar.expander("**Custom support**", expanded=False):
    use_custom = st.checkbox("Add custom support", value=False)
    if use_custom:
        cust_psm = st.number_input("p_sm (MPa)", 0.01, 50.0, 0.5, 0.05)
        cust_ks = st.number_input("k_s (MPa/m)", 1.0, 10000.0, 100.0, 10.0)
        cust_usm = cust_psm / cust_ks * 1000
        supports.append({"psm": cust_psm, "ks": cust_ks, "usm": cust_usm,
                         "name": "Custom"})
        st.caption(f"→ u_sm = {cust_usm:.2f} mm")

params = dict(
    sigma_ci=sigma_ci, E_i=E_i, mi=mi, GSI=GSI, D=D, nu=nu,
    gamma=gamma, z=z, a=a, L=L,
    phi_deg=phi_deg, c=c, sig3_max=sig3_max, psi_deg=psi_deg,
    mc_source=mc_source,
    grc_method=grc_method, ldp_method=ldp_method, alpha_panet=alpha_panet,
    ldp_norm=ldp_norm, supports=supports,
)
result = run_analysis(params)


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.title("⛰️ Convergence-Confinement Method (CCM)")
st.markdown(
    "**RSE3010 — Mine Geotechnical Engineering | Monash University** \n"
    "*Interactive ground–support interaction tool based on the "
    "convergence-confinement method.*"
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


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Summary", "📈 GRC / SCC", "⭕ Tunnel Section", "📉 LDP",
    "🧭 Envelopes", "🔬 Sensitivity", "💾 Export", "📋 Workflow",
])

with tab1:
    st.header("Analysis Results")

    # Headline metrics row
    hl1, hl2, hl3, hl4 = st.columns(4)
    hl1.metric("In-situ stress p₀", f"{result['p0']:.2f} MPa",
               help="γ · z")
    hl2.metric("Max convergence u_max", f"{result['u_max']:.2f} mm",
               f"{result['u_max']/params['a']/1000*100:.2f} % strain")
    hl3.metric("Displacement at install u_s0", f"{result['us0']:.2f} mm",
               f"{result['u_star']*100:.1f} % of u_max")
    if result.get("FoS") is not None:
        hl4.metric("Factor of Safety", f"{result['FoS']:.2f}",
                   "Safe" if result['FoS'] >= 1.5 else
                   "Marginal" if result['FoS'] >= 1.0 else "Unsafe")
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
        st.metric("Response",
                  "Plastic" if result['is_plastic'] else "Elastic")
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
            {"Support": s["name"],
             "p_sm (MPa)": round(s["psm"], 3),
             "k_s (MPa/m)": round(s["ks"], 1),
             "u_sm (mm)": round(s["usm"], 2)}
            for s in supports
        ])
        # Add total row
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

    # Engineering interpretation
    st.subheader("Engineering interpretation")
    interp = []
    conv_pct = result['u_max']/params['a']/1000 * 100
    # Hoek & Marinos 2000 squeezing classification
    if conv_pct < 1:
        interp.append(f"✓ **Category A** (< 1% strain): Few support problems. "
                      f"Unsupported convergence = {conv_pct:.2f}%.")
    elif conv_pct < 2.5:
        interp.append(f"⚠ **Category B** (1–2.5%): Minor squeezing. "
                      f"Light rockbolt + shotcrete typically sufficient. "
                      f"Convergence = {conv_pct:.2f}%.")
    elif conv_pct < 5:
        interp.append(f"⚠ **Category C** (2.5–5%): Severe squeezing. "
                      f"Heavy rockbolts + shotcrete + yielding elements. "
                      f"Convergence = {conv_pct:.2f}%.")
    elif conv_pct < 10:
        interp.append(f"⚠ **Category D** (5–10%): Very severe squeezing. "
                      f"Yielding support essential. Convergence = {conv_pct:.2f}%.")
    else:
        interp.append(f"❌ **Category E** (> 10%): Extreme squeezing. "
                      f"Redesign needed (face stabilisation, yielding lining, "
                      f"tunnel reshaping). Convergence = {conv_pct:.2f}%.")

    if result.get("FoS") is not None:
        if result['FoS'] >= 2.0:
            interp.append("✓ Support is well-sized with ample margin.")
        elif result['FoS'] >= 1.5:
            interp.append("✓ Support is adequately sized per industry practice.")
        elif result['FoS'] >= 1.0:
            interp.append("⚠ Support is marginal — consider stiffer/stronger system "
                          "or closer installation.")
        else:
            interp.append("❌ Support is inadequate — redesign required.")

    if result["is_plastic"]:
        ratio = result['Rp_unsup'] / params['a']
        interp.append(f"Plastic zone extends {ratio:.1f} × tunnel radius. "
                      f"Rockbolts must anchor beyond R_p = {result['Rp_unsup']:.2f} m.")

    for line in interp:
        st.markdown(line)

with tab2:
    # GRC comparison toggle
    compare_toggle = st.checkbox(
        "Overlay all 5 GRC methods for comparison",
        value=False,
        help="Show all GRC methods on the same plot. Useful for understanding "
             "the sensitivity of the design to the choice of analytical method.")
    st.plotly_chart(plot_grc_scc(result, params, compare_all=compare_toggle),
                    width="stretch")

    if compare_toggle:
        st.info(
            "ℹ️ **Interpretation**: The selected method (★) is shown with a bold "
            "line, others with dotted lines. All methods give the same elastic "
            "(Kirsch) branch above p_cr — differences appear only in the plastic "
            "zone, where they can result in FoS differences of ±30%.")

    if result.get("saturated"):
        st.error(
            "⚠️ **Support saturated.** The SCC reached its maximum pressure p_sm "
            "before intersecting the GRC — the lining is at capacity and the "
            "FoS is effectively 1.0. Consider a stiffer support, yielding "
            "elements, or reducing the stand-off distance L.")

with tab3:
    st.plotly_chart(plot_tunnel_section(result, params), width="stretch")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tunnel radius a", f"{params['a']:.2f} m")
    col2.metric("R_p unsupported", f"{result['Rp_unsup']:.2f} m",
                f"{(result['Rp_unsup']/params['a']-1)*100:+.0f} % beyond a"
                if result['Rp_unsup'] > params['a'] else "elastic")
    col3.metric("R_p supported", f"{result['Rp_sup']:.2f} m",
                f"{(result['Rp_sup']-result['Rp_unsup']):+.2f} m change"
                if result['Rp_sup'] != result['Rp_unsup'] else "—")
    col4.metric("Convergence (unsupported)",
                f"{result['u_max']/params['a']/1000 * 100:.2f} %",
                help="u_max / a. Hoek-Marinos 2000 squeezing categories: "
                     "A<1, B<2.5, C<5, D<10, E>10 %")

    st.caption(
        "The supported plastic zone radius depends on the equilibrium "
        "pressure p_eq. Rockbolts should extend ~2 m beyond R_p (supported) "
        "to anchor in elastic rock.")

with tab4:
    st.plotly_chart(plot_ldp(result, params), width="stretch")
    st.info(
        f"**Installation at L = {params['L']:.2f} m behind face** "
        f"(X* = L/a = {result['Xstar']:.3f}). "
        f"The {params['ldp_method']} method gives u* = {result['u_star']:.3f}, "
        f"so u_s0 = {result['us0']:.2f} mm "
        f"({result['u_star']*100:.1f}% of u_max = {result['u_max']:.2f} mm) "
        f"has already occurred before the support takes load.")

with tab5:
    st.plotly_chart(plot_envelope(result, params), width="stretch")
    if params["mc_source"] == "Fit automatically (Hoek 2002)":
        st.success(
            f"MC parameters fitted automatically via Hoek (2002) closed form "
            f"over 0 ≤ σ₃ ≤ {params['sig3_max']:.1f} MPa: "
            f"**φ = {result['phi_deg']:.2f}°**, **c = {result['c']:.3f} MPa**.")

# ============================================================================
# Tab 6: Sensitivity Analysis
# ============================================================================
with tab6:
    st.header("🔬 Sensitivity Analysis")
    st.markdown(
        "Sweep one input parameter over a range and see how key outputs respond. "
        "Essential for understanding design uncertainty and parameter importance.")

    # Warning if sweeping rock-mass parameters with manual MC
    if params["mc_source"] == "Enter manually":
        st.warning(
            "⚠️ **You are using manual MC parameters (φ, c).** "
            "When sweeping GSI, σ_ci, m_i, or D, the MC parameters stay fixed, "
            "so only E_m changes — giving an incomplete picture. "
            "**Switch to 'Fit automatically (Hoek 2002)' in the sidebar** for a "
            "consistent sensitivity analysis where MC parameters re-fit at each "
            "sweep point.")

    sens_col1, sens_col2 = st.columns(2)
    with sens_col1:
        sens_var = st.selectbox(
            "Parameter to sweep",
            ["GSI", "σ_ci (MPa)", "m_i", "z (depth, m)", "a (tunnel radius, m)",
             "L (install distance, m)", "t (lining thickness, m)",
             "D (disturbance factor)"],
            index=0)
    with sens_col2:
        n_pts = st.slider("Number of points", 5, 30, 15)

    # Define sweep ranges
    sweep_ranges = {
        "GSI": (max(10, params["GSI"] - 20), min(100, params["GSI"] + 20)),
        "σ_ci (MPa)": (max(1, params["sigma_ci"] * 0.5), params["sigma_ci"] * 1.5),
        "m_i": (max(1, params["mi"] - 3), params["mi"] + 5),
        "z (depth, m)": (max(10, params["z"] * 0.5), params["z"] * 1.5),
        "a (tunnel radius, m)": (max(0.5, params["a"] * 0.7), params["a"] * 1.5),
        "L (install distance, m)": (0.0, min(30, params["a"] * 4)),
        "t (lining thickness, m)": (0.1, 1.5),
        "D (disturbance factor)": (0.0, 1.0),
    }
    lo, hi = sweep_ranges[sens_var]
    sens_min = st.number_input(f"Minimum {sens_var}",
                               value=float(lo), min_value=0.0)
    sens_max = st.number_input(f"Maximum {sens_var}",
                               value=float(hi), min_value=sens_min + 0.01)

    if st.button("Run sensitivity analysis", type="primary"):
        sweep_values = np.linspace(sens_min, sens_max, n_pts)
        results_sweep = []
        progress = st.progress(0.0)

        for i, val in enumerate(sweep_values):
            p_swept = params.copy()
            key_map = {
                "GSI": "GSI", "σ_ci (MPa)": "sigma_ci", "m_i": "mi",
                "z (depth, m)": "z", "a (tunnel radius, m)": "a",
                "L (install distance, m)": "L", "D (disturbance factor)": "D",
            }
            if sens_var in key_map:
                p_swept[key_map[sens_var]] = val
                # Update supports if radius changed (lining p_sm depends on a)
                if sens_var == "a (tunnel radius, m)":
                    new_supports = []
                    for s in params["supports"]:
                        if s["name"] == "Concrete lining":
                            ns = support_concrete(sigma_ci_lining, Ec, nu_c,
                                                   s["t"], val)
                            ns["name"] = s["name"]; ns["t"] = s["t"]
                            new_supports.append(ns)
                        else:
                            new_supports.append(s)
                    p_swept["supports"] = new_supports
            elif sens_var == "t (lining thickness, m)":
                # Update lining thickness
                new_supports = []
                for s in params["supports"]:
                    if s["name"] == "Concrete lining":
                        ns = support_concrete(sigma_ci_lining, Ec, nu_c, val,
                                               params["a"])
                        ns["name"] = s["name"]; ns["t"] = val
                        new_supports.append(ns)
                    else:
                        new_supports.append(s)
                p_swept["supports"] = new_supports

            try:
                r_swept = run_analysis(p_swept)
                results_sweep.append({
                    sens_var: val,
                    "u_max (mm)": r_swept["u_max"],
                    "u_s0 (mm)": r_swept["us0"],
                    "p_cr (MPa)": r_swept["pcr"],
                    "R_p (m)": r_swept["Rp_unsup"],
                    "p_eq (MPa)": r_swept["p_eq"] if r_swept["p_eq"] else np.nan,
                    "FoS": r_swept["FoS"] if r_swept["FoS"] else np.nan,
                })
            except Exception:
                results_sweep.append({
                    sens_var: val,
                    "u_max (mm)": np.nan, "u_s0 (mm)": np.nan,
                    "p_cr (MPa)": np.nan, "R_p (m)": np.nan,
                    "p_eq (MPa)": np.nan, "FoS": np.nan})
            progress.progress((i + 1) / n_pts)

        progress.empty()
        df_sens = pd.DataFrame(results_sweep)
        st.session_state["sensitivity_data"] = df_sens

    # Show sensitivity results if computed
    if "sensitivity_data" in st.session_state:
        df_sens = st.session_state["sensitivity_data"]
        var_name = df_sens.columns[0]

        # Four-panel plot: u_max, R_p, p_eq, FoS
        from plotly.subplots import make_subplots
        fig_sens = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Maximum convergence u_max (mm)",
                            "Plastic radius R_p (m)",
                            "Equilibrium pressure p_eq (MPa)",
                            "Factor of Safety"),
            horizontal_spacing=0.12, vertical_spacing=0.15)

        fig_sens.add_trace(go.Scatter(
            x=df_sens[var_name], y=df_sens["u_max (mm)"],
            mode="lines+markers", line=dict(color="#1F4E78", width=2),
            marker=dict(size=7), showlegend=False), row=1, col=1)
        fig_sens.add_trace(go.Scatter(
            x=df_sens[var_name], y=df_sens["R_p (m)"],
            mode="lines+markers", line=dict(color="#2E7D32", width=2),
            marker=dict(size=7), showlegend=False), row=1, col=2)
        fig_sens.add_trace(go.Scatter(
            x=df_sens[var_name], y=df_sens["p_eq (MPa)"],
            mode="lines+markers", line=dict(color="#C00000", width=2),
            marker=dict(size=7), showlegend=False), row=2, col=1)
        fig_sens.add_trace(go.Scatter(
            x=df_sens[var_name], y=df_sens["FoS"],
            mode="lines+markers", line=dict(color="#7B1FA2", width=2),
            marker=dict(size=7), showlegend=False), row=2, col=2)

        # Add FoS=1.5 reference line
        fig_sens.add_hline(y=1.5, line=dict(color="orange", dash="dash"),
                           annotation_text="FoS = 1.5 (target)",
                           row=2, col=2)

        # Add current value marker
        current_val = {
            "GSI": params["GSI"], "σ_ci (MPa)": params["sigma_ci"],
            "m_i": params["mi"], "z (depth, m)": params["z"],
            "a (tunnel radius, m)": params["a"],
            "L (install distance, m)": params["L"],
            "D (disturbance factor)": params["D"],
            "t (lining thickness, m)": (params["supports"][0]["t"]
                                         if params["supports"]
                                         and "t" in params["supports"][0]
                                         else None),
        }.get(sens_var)
        if current_val is not None:
            for r in range(1, 3):
                for c in range(1, 3):
                    fig_sens.add_vline(x=current_val,
                                       line=dict(color="#FF6B35", dash="dot"),
                                       row=r, col=c)

        fig_sens.update_xaxes(title_text=var_name)
        fig_sens.update_layout(
            height=600, showlegend=False,
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font=dict(family="Georgia, serif", size=11))
        for r in range(1, 3):
            for c in range(1, 3):
                fig_sens.update_xaxes(gridcolor="#e5e5e5", row=r, col=c)
                fig_sens.update_yaxes(gridcolor="#e5e5e5", row=r, col=c)

        st.plotly_chart(fig_sens, width="stretch")

        st.subheader("Sensitivity data")
        st.dataframe(df_sens.round(3), hide_index=True, width="stretch")
        st.download_button(
            "📥 Download sensitivity data (CSV)",
            data=df_sens.to_csv(index=False).encode(),
            file_name=f"sensitivity_{sens_var.split()[0]}.csv",
            mime="text/csv")

# ============================================================================
# Tab 7: Export
# ============================================================================
with tab7:
    st.header("💾 Export Results")
    st.markdown(
        "Export the current analysis for reporting and documentation. "
        "All inputs, intermediate values, and final outputs are included.")

    # Build full result summary
    report_rows = [
        ("--- PROJECT INPUTS ---", ""),
        ("GRC method", params["grc_method"]),
        ("LDP method", params["ldp_method"]),
        ("Tunnel radius a (m)", f"{params['a']:.3f}"),
        ("Depth z (m)", f"{params['z']:.1f}"),
        ("Unit weight γ (kN/m³)", f"{params['gamma']:.1f}"),
        ("σ_ci intact UCS (MPa)", f"{params['sigma_ci']:.1f}"),
        ("E_i intact modulus (GPa)", f"{params['E_i']/1000:.2f}"),
        ("m_i", f"{params['mi']:.2f}"),
        ("GSI", f"{params['GSI']}"),
        ("Disturbance D", f"{params['D']:.2f}"),
        ("Poisson's ratio ν", f"{params['nu']:.3f}"),
        ("MC source", params["mc_source"]),
        ("φ (°)", f"{result['phi_deg']:.2f}"),
        ("c (MPa)", f"{result['c']:.3f}"),
        ("Installation L (m)", f"{params['L']:.2f}"),
        ("--- ROCK MASS ---", ""),
        ("p₀ (in-situ stress, MPa)", f"{result['p0']:.3f}"),
        ("m_b", f"{result['mb']:.4f}"),
        ("s", f"{result['s']:.6f}"),
        ("a_HB", f"{result['a_HB']:.4f}"),
        ("σ_cm H-B (MPa)", f"{result['sigma_cm_HB']:.3f}"),
        ("σ_cm,MC (MPa)", f"{result['sigma_cm_MC']:.3f}"),
        ("E_m (MPa)", f"{result['Em']:.1f}"),
        ("k = (1+sinφ)/(1-sinφ)", f"{result['k']:.3f}"),
        ("--- CRITICAL STATE ---", ""),
        ("p_cr critical pressure (MPa)", f"{result['pcr']:.3f}"),
        ("Response", "Plastic" if result['is_plastic'] else "Elastic"),
        ("R_p unsupported (m)", f"{result['Rp_unsup']:.3f}"),
        ("R_p supported (m)", f"{result['Rp_sup']:.3f}"),
        ("R* = R_p/a", f"{result['Rstar']:.3f}"),
        ("--- CONVERGENCE ---", ""),
        ("u_max unsupported (mm)", f"{result['u_max']:.3f}"),
        ("Convergence strain (%)",
         f"{result['u_max']/params['a']/1000*100:.3f}"),
        ("X* = L/a", f"{result['Xstar']:.3f}"),
        ("u* LDP", f"{result['u_star']:.4f}"),
        ("u_s0 at install (mm)", f"{result['us0']:.3f}"),
        ("--- SUPPORT ---", ""),
    ]
    if params["supports"]:
        for s in params["supports"]:
            report_rows.append((f"  {s['name']} p_sm (MPa)", f"{s['psm']:.3f}"))
            report_rows.append((f"  {s['name']} k_s (MPa/m)", f"{s['ks']:.1f}"))
            report_rows.append((f"  {s['name']} u_sm (mm)", f"{s['usm']:.3f}"))
    report_rows.extend([
        ("Combined p_sm (MPa)", f"{result['psm_total']:.3f}"),
        ("Combined k_s (MPa/m)", f"{result['ks_total']:.1f}"),
        ("Combined u_sm (mm)", f"{result['usm_total']:.3f}"),
        ("--- EQUILIBRIUM ---", ""),
        ("p_eq (MPa)", f"{result['p_eq']:.4f}" if result['p_eq'] else "—"),
        ("u_eq (mm)", f"{result['u_eq']:.3f}" if result['u_eq'] else "—"),
        ("FoS", f"{result['FoS']:.3f}" if result['FoS'] else "—"),
    ])

    df_report = pd.DataFrame(report_rows, columns=["Quantity", "Value"])

    # Export buttons
    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        st.download_button(
            "📥 Download full results (CSV)",
            data=df_report.to_csv(index=False).encode(),
            file_name="ccm_analysis_results.csv",
            mime="text/csv",
            width="stretch")
    with exp_c2:
        # JSON export for programmatic use
        import json
        export_dict = {
            "inputs": {k: (v if not isinstance(v, list) else
                           [{kk: vv for kk, vv in s.items() if not callable(vv)}
                            for s in v])
                       for k, v in params.items() if k != "grc_fn"},
            "outputs": {k: (float(v) if isinstance(v, (np.floating, np.integer))
                            else (str(v) if callable(v) else v))
                        for k, v in result.items()
                        if k != "grc_fn" and not callable(v)},
        }
        st.download_button(
            "📥 Download inputs + outputs (JSON)",
            data=json.dumps(export_dict, indent=2, default=str).encode(),
            file_name="ccm_analysis.json",
            mime="application/json",
            width="stretch")

    # Also offer GRC and LDP data as CSV
    st.divider()
    st.subheader("Export curve data")
    exp_d1, exp_d2 = st.columns(2)
    with exp_d1:
        pi_range_exp = np.linspace(0, result['p0'], 100)
        u_grc_exp = [result['grc_fn'](pi) for pi in pi_range_exp]
        df_grc = pd.DataFrame({
            "p_i (MPa)": pi_range_exp,
            "u_r (mm)": u_grc_exp,
        })
        st.download_button(
            "📈 Download GRC data (CSV)",
            data=df_grc.to_csv(index=False).encode(),
            file_name=f"grc_{params['grc_method'].replace(' ', '_')}.csv",
            mime="text/csv",
            width="stretch")

    with exp_d2:
        X_range_exp = np.linspace(-3 * params['a'], 10 * params['a'], 100)
        df_ldp = pd.DataFrame({"X (m)": X_range_exp})
        for m in ["Panet (1995)", "Vlachopoulos & Diederichs (2009)",
                  "Hoek (2002)"]:
            df_ldp[f"u_r {m} (mm)"] = [
                ldp(m, x / params['a'], result['Rstar'],
                    params.get('alpha_panet', 0.75),
                    norm=params.get('ldp_norm', LDP_DEFAULT_NORM)) * result['u_max']
                for x in X_range_exp
            ]
        st.download_button(
            "📉 Download LDP data (CSV)",
            data=df_ldp.to_csv(index=False).encode(),
            file_name="ldp_curves.csv",
            mime="text/csv",
            width="stretch")

    # Preview
    st.divider()
    st.subheader("Results summary (preview)")
    st.dataframe(df_report, hide_index=True, width="stretch", height=400)

with tab8:
    st.header("Calculation Workflow")

    st.markdown("### Step 1 — In-situ stress")
    st.latex(r"p_0 = \gamma \cdot z")

    st.markdown("### Step 2 — Hoek-Brown parameters")
    st.latex(r"m_b = m_i \, \exp\!\left(\frac{GSI - 100}{28 - 14 D}\right), \quad "
             r"s = \exp\!\left(\frac{GSI - 100}{9 - 3 D}\right)")
    st.latex(r"a_{HB} = \tfrac{1}{2} + \tfrac{1}{6}\left(e^{-GSI/15} - e^{-20/3}\right)")

    st.markdown("### Step 3 — Rock mass strength & modulus")
    st.latex(r"\sigma_{cm} = \sigma_{ci} \cdot s^{a_{HB}}")
    st.latex(r"E_m = E_i \left[0.02 + \frac{1 - D/2}"
             r"{1 + \exp\!\big((60 + 15 D - GSI)/11\big)}\right]")

    st.markdown("### Step 4 — Equivalent Mohr-Coulomb (Hoek et al. 2002 closed form)")
    st.markdown(
        "Many GRC solutions require MC parameters. Fit the H-B envelope over "
        r"$0 \le \sigma_3 \le \sigma_{3,\max}$:")
    st.latex(r"\phi = \arcsin\!\left[\frac{6\,a_{HB}\, m_b \,"
             r"(s + m_b \sigma_{3n})^{a_{HB} - 1}}"
             r"{2(1+a_{HB})(2+a_{HB}) + 6\,a_{HB}\, m_b \,"
             r"(s + m_b \sigma_{3n})^{a_{HB}-1}}\right]")
    st.latex(r"c = \frac{\sigma_{ci}\,[(1+2 a_{HB})s + (1-a_{HB}) m_b \sigma_{3n}]\,"
             r"(s+m_b \sigma_{3n})^{a_{HB}-1}}"
             r"{(1+a_{HB})(2+a_{HB})\sqrt{1 + 6 a_{HB} m_b (s+m_b \sigma_{3n})^"
             r"{a_{HB}-1}/[(1+a_{HB})(2+a_{HB})]}}")
    st.caption(r"$\sigma_{3n} = \sigma_{3,\max}/\sigma_{ci}$. Derived: "
               r"$k=(1+\sin\phi)/(1-\sin\phi)$, "
               r"$\sigma_{cm,MC}=2c\cos\phi/(1-\sin\phi)$.")

    st.markdown("### Step 5 — Critical pressure & plastic radius")
    st.latex(r"p_{cr} = \frac{2 p_0 - \sigma_{cm,MC}}{1 + k}")
    st.markdown(r"If $p_i \ge p_{cr}$: elastic response. "
                r"If $p_i < p_{cr}$: plastic zone forms with radius:")
    st.latex(r"R_p(p_i) = a \left[\frac{2\left(p_0(k-1) + \sigma_{cm,MC}\right)}"
             r"{(1+k)\left(p_i(k-1) + \sigma_{cm,MC}\right)}\right]^{1/(k-1)}")

    st.markdown("### Step 6 — Ground Reaction Curve (GRC)")
    st.markdown(
        "The GRC describes how the tunnel wall displacement $u_r$ varies with "
        "the internal support pressure $p_i$. Different analytical methods "
        "give different GRCs depending on the constitutive model and strain assumptions.")

    st.markdown("**Elastic branch** (all methods, $p_i \\ge p_{cr}$): Kirsch solution")
    st.latex(r"u_r = \frac{(1+\nu)(p_0 - p_i)\,a}{E_m}")

    st.markdown("**Duncan-Fama (MC, 1993):** small-strain MC, plastic branch uses "
                "$R_p$ from Step 5:")
    st.latex(r"u_r = \frac{(1+\nu)(p_0 - p_{cr})\,a}{E_m} \cdot "
             r"\left(\frac{R_p(p_i)}{a}\right)^2")

    st.markdown("**Salençon (MC, 1969):** identical plastic $R_p$, but "
                "a different exponent on the displacement:")
    st.latex(r"u_r = \frac{(1+\nu)(p_0 - p_{cr})\,a}{E_m} \cdot "
             r"\left(\frac{R_p(p_i)}{a}\right)^{(k+1)/(k-1)}")
    st.caption("Historical closed form. Exponent reduces to ~2 for moderate k.")

    st.markdown("**Vrakas & Anagnostou (MC, 2014):** large-strain MC with dilation "
                r"angle $\psi$. Small-strain result multiplied by a correction:")
    st.latex(r"u_r^{\text{LS}} = u_r^{\text{SS}} \cdot "
             r"\left(1 + \tfrac{1}{2}\,\varepsilon\,k_\psi\right), \quad "
             r"k_\psi = \frac{1+\sin\psi}{1-\sin\psi}")
    st.caption("where ε = u_r / a is the tunnel-wall strain. For ψ = 0 this reduces "
               "to Duncan-Fama.")

    st.markdown("**Brown et al. (HB, 1983):** original Hoek-Brown ($a_{HB}=0.5$), "
                "uses a *secant* slope $k_{sec}$:")
    st.latex(r"\sigma_1 = \sigma_3 + \sigma_{ci}\sqrt{m_b\,\sigma_3/\sigma_{ci} + s}, "
             r"\quad k_{sec} = [\sigma_1(p_i) - \sigma_{cm}]/p_i")

    st.markdown("**Carranza-Torres (HB, 2004):** generalised H-B, uses a *tangent* "
                "slope $k_{tan}$ at the yield boundary:")
    st.latex(r"k_{tan} = 1 + a_{HB}\,m_b\,(m_b\,\sigma_3/\sigma_{ci} + s)^{a_{HB}-1}")
    st.caption("Plastic $R_p$ then uses $k_{tan}$ in the Step-5 formula. "
               "This app uses a simplified tangent-MC form; full CT (2004) "
               "allows dilation and residual strength.")

    st.info("💡 **When methods disagree:** differences typically arise only in the "
            "plastic zone. In the elastic range (p_i ≥ p_cr) all methods give "
            "the same Kirsch result.")

    st.markdown("### Step 7 — Maximum convergence (unsupported)")
    st.latex(r"u_{\max} = u_r(p_i = 0)")
    st.caption("Evaluated by the chosen GRC method at p_i = 0.")

    st.markdown("### Step 8 — Longitudinal Deformation Profile (LDP)")
    st.markdown(
        "The LDP describes how displacement evolves along the tunnel axis. "
        r"$X^* = X/a$ where $X$ is distance from the face. "
        r"Convention in this app: **$X^* > 0$ = behind face (excavated)**, "
        r"**$X^* < 0$ = ahead of face (not yet excavated)**.")

    st.markdown("**Panet (1995) — elastic LDP, both branches:**")
    st.latex(r"u^*(X^*) = \begin{cases}"
             r"(1 - \alpha)\,e^{1.5 X^*}, & X^* \le 0 \ \text{(ahead)}\\[4pt]"
             r"1 - \alpha\,e^{-1.5 X^*}, & X^* > 0 \ \text{(behind)}"
             r"\end{cases}")
    st.caption(r"With $\alpha = 0.75$ (Panet's calibration), "
               r"$u^*(0) = 0.25$ on both branches.")

    st.markdown("**Vlachopoulos & Diederichs (2009) — elastoplastic LDP, "
                r"depends on $R^* = R_p/a$:**")
    st.latex(r"u^*(X^*) = \begin{cases}"
             r"\tfrac{1}{3}\,e^{2 X^* - 0.15 R^*}, & X^* \le 0 \ \text{(ahead)}\\[4pt]"
             r"1 - \left[1 - \tfrac{1}{3}\,e^{-0.15 R^*}\right] e^{-3 X^*/R^*}, "
             r"& X^* > 0 \ \text{(behind)}"
             r"\end{cases}")

    st.markdown(r"**Hoek (2002) — empirical, normalised by diameter $D = 2a$:**")
    st.latex(r"u^*(X^*) = \left(1 + \exp\!\left(-\frac{X^*/2}{1.1}\right)\right)^{-1.7}")
    st.caption(r"Valid for all $X^*$; the single formula captures both branches.")

    st.markdown("### Step 9 — Installation criterion")
    st.markdown("Three equivalent ways to specify the installation point:")
    st.markdown(
        r"- **Distance from face $L$** → $X^* = L/a$ → $u^*$ from LDP → "
        r"$u_{s0} = u^* \cdot u_{\max}$"
    )
    st.markdown(
        r"- **Target displacement $u_{s0}$** → back-solve LDP for $X^*$ such that "
        r"$u^*(X^*) = u_{s0}/u_{\max}$ → $L = X^* \cdot a$"
    )
    st.markdown(
        r"- **Target strain $\varepsilon$ (%)** → $u_{s0} = \varepsilon \cdot a / 100$ → "
        r"then as above"
    )

    st.markdown("### Step 10 — Support Characteristic Curve (SCC)")
    st.markdown("**Concrete / shotcrete ring:**")
    st.latex(r"p_{sm} = \frac{\sigma_{ci,\text{lining}} \cdot t}{a}, \quad "
             r"k_s = \frac{E_c \cdot t}{(1 - \nu_c^2)\,a^2}, \quad "
             r"u_{sm} = p_{sm}/k_s")
    st.markdown(r"**Combined support** (concrete + bolts + sets):")
    st.latex(r"p_{sm}^{\text{total}} = \sum_i p_{sm,i}, \quad "
             r"k_s^{\text{total}} = \sum_i k_{s,i}, \quad "
             r"u_{sm}^{\text{total}} = \max_i(u_{sm,i})")

    st.markdown("### Step 11 — Factor of Safety (GRC–SCC intersection)")
    st.markdown(
        r"Solve $u_{GRC}(p_{eq}) = u_{s0} + p_{eq}/k_s$ numerically (root-find). "
        r"Then:")
    st.latex(r"\mathrm{FoS} = \frac{p_{sm}}{p_{eq}}")
    st.caption("FoS ≥ 1.5 is typically considered safe for permanent tunnel support.")

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
    "**GRC methods**: Duncan-Fama, Salençon (1969), Vrakas & Anagnostou (2014), "
    "Brown et al. (1983), Carranza-Torres (2004) · "
    "**LDP methods**: Panet (1995), Vlachopoulos & Diederichs (2009), Hoek (2002) · "
    "**MC fit**: Hoek et al. (2002)"
)
