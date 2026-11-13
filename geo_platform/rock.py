"""Strength and stiffness of intact rock and of the rock mass.

The expressions are those printed in *Underground Design and Construction*,
Chapter 11 (intact rock: Mohr--Coulomb, Hoek--Brown, tension, conversion to an
equivalent Mohr--Coulomb line) and Chapter 12 (generalised Hoek--Brown, rock
mass strength and modulus). Stresses are in MPa, compression positive; moduli
in GPa. Nothing here imports Streamlit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# In-situ stress (Chapter 11, two-stress description with one ratio k)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InSitu:
    sigma_v: float
    sigma_h: float
    sigma_1: float          # sigma_1^IS
    sigma_3: float          # sigma_3^IS
    sigma3_fit_max: float   # upper limit of the fitting range


def insitu_stresses(depth_m: float, k: float, gamma_kn_m3: float = 27.0) -> InSitu:
    """sigma_v = gamma z; sigma_h,k = k sigma_v; ranked; fitting limit = max."""
    sv = gamma_kn_m3 * depth_m / 1000.0
    sh = k * sv
    return InSitu(sv, sh, max(sv, sh), min(sv, sh), max(sv, sh))


# ---------------------------------------------------------------------------
# Mohr--Coulomb (Chapter 11)
# ---------------------------------------------------------------------------


def mc_k_phi(phi_deg: float) -> float:
    s = math.sin(math.radians(phi_deg))
    return (1 + s) / (1 - s)


def mc_ucs(c: float, phi_deg: float) -> float:
    """sigma_c = 2c cos(phi) / (1 - sin(phi))."""
    phi = math.radians(phi_deg)
    return 2 * c * math.cos(phi) / (1 - math.sin(phi))


def mc_tension(c: float, phi_deg: float) -> float:
    """sigma_t = -2c cos(phi) / (1 + sin(phi)) (negative: tension)."""
    phi = math.radians(phi_deg)
    return -2 * c * math.cos(phi) / (1 + math.sin(phi))


def mc_sigma1(sigma3, c: float, phi_deg: float):
    return mc_ucs(c, phi_deg) + mc_k_phi(phi_deg) * np.asarray(sigma3, dtype=float)


# ---------------------------------------------------------------------------
# Hoek--Brown, intact and generalised (Chapters 11 and 12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HBConstants:
    mb: float
    s: float
    a: float


def hb_constants(gsi: float, mi: float, d: float = 0.0) -> HBConstants:
    """Generalised Hoek--Brown constants from GSI, m_i and the disturbance factor D."""
    mb = mi * math.exp((gsi - 100) / (28 - 14 * d))
    s = math.exp((gsi - 100) / (9 - 3 * d))
    a = 0.5 + (math.exp(-gsi / 15) - math.exp(-20 / 3)) / 6
    return HBConstants(mb, s, a)


def hb_sigma1(sigma3, sigma_ci: float, mb: float, s: float = 1.0, a: float = 0.5):
    """sigma_1 at failure; intact rock with m_b = m_i, s = 1, a = 0.5."""
    s3 = np.asarray(sigma3, dtype=float)
    base = np.clip(mb * s3 / sigma_ci + s, 0.0, None)
    return s3 + sigma_ci * base ** a


def hb_intact_tension(sigma_ci: float, mi: float) -> float:
    """sigma_t = (sigma_ci / 2)(m_i - sqrt(m_i^2 + 4)), intact criterion."""
    return sigma_ci / 2 * (mi - math.sqrt(mi * mi + 4))


def hoek_martin_tension(sigma_ci: float, mi: float) -> float:
    """sigma_t = -sigma_ci / (8.62 + 0.7 m_i) (Hoek and Martin 2014)."""
    return -sigma_ci / (8.62 + 0.7 * mi)


def rock_mass_ucs(sigma_ci: float, hb: HBConstants) -> float:
    """sigma_cm = sigma_ci s^a."""
    return sigma_ci * hb.s ** hb.a


def rock_mass_tension(sigma_ci: float, hb: HBConstants) -> float:
    """sigma_tm = -s sigma_ci / m_b."""
    return -hb.s * sigma_ci / hb.mb


def rock_mass_tension_hm(sigma_ci: float, hb: HBConstants, mi: float) -> float:
    """Hoek--Martin cut-off applied to the rock mass: -sigma_cm / (8.62 + 0.7 m_i)."""
    return -rock_mass_ucs(sigma_ci, hb) / (8.62 + 0.7 * mi)


def global_strength(sigma_ci: float, hb: HBConstants) -> float:
    """Global rock mass strength (Hoek, Carranza-Torres and Corkum 2002)."""
    mb, s, a = hb.mb, hb.s, hb.a
    return sigma_ci * (mb + 4 * s - a * (mb - 8 * s)) * (mb / 4 + s) ** (a - 1) / (2 * (1 + a) * (2 + a))


# ---------------------------------------------------------------------------
# Rock mass modulus (Chapter 12): five relations, in GPa
# ---------------------------------------------------------------------------


def modulus_hd_generalised(ei_gpa: float, gsi: float, d: float) -> float:
    """Hoek and Diederichs (2006), generalised, from the intact modulus."""
    return ei_gpa * (0.02 + (1 - d / 2) / (1 + math.exp((60 + 15 * d - gsi) / 11)))


def modulus_hd_simplified(gsi: float, d: float) -> float:
    """Hoek and Diederichs (2006), simplified: E_m = 100 (1 - D/2)/(1 + e^((75+25D-GSI)/11)) GPa."""
    return 100 * (1 - d / 2) / (1 + math.exp((75 + 25 * d - gsi) / 11))


def modulus_hoek_2002(sigma_ci: float, gsi: float, d: float) -> float:
    """Hoek, Carranza-Torres and Corkum (2002); the root term applies for sigma_ci <= 100 MPa."""
    factor = math.sqrt(sigma_ci / 100) if sigma_ci <= 100 else 1.0
    return (1 - d / 2) * factor * 10 ** ((gsi - 10) / 40)


def modulus_bieniawski(rmr: float) -> float:
    """E_m = 2 RMR - 100 GPa, valid for RMR > 50."""
    return 2 * rmr - 100


def modulus_serafim_pereira(rmr: float) -> float:
    """E_m = 10^((RMR - 10)/40) GPa."""
    return 10 ** ((rmr - 10) / 40)


# ---------------------------------------------------------------------------
# Equivalent Mohr--Coulomb line over a stated range (Chapters 11 and 12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCFit:
    phi_deg: float
    c: float
    r2: float
    intercept: float     # A (method 1) or a_fit (method 2)
    slope: float         # B (method 1) or b_fit (method 2)


def fit_mohr_circles(sigma3, sigma1) -> MCFit:
    """Method 1: regress R on C, R = A + B C; phi = asin B, c = A / cos phi."""
    s3, s1 = np.asarray(sigma3, float), np.asarray(sigma1, float)
    centre, radius = (s1 + s3) / 2, (s1 - s3) / 2
    slope, intercept = np.polyfit(centre, radius, 1)
    phi = math.asin(min(max(slope, -1.0), 1.0))
    r2 = float(np.corrcoef(centre, radius)[0, 1] ** 2) if len(s3) > 2 else 1.0
    return MCFit(math.degrees(phi), intercept / math.cos(phi), r2, intercept, slope)


def fit_principal_line(sigma3, sigma1) -> MCFit:
    """Method 2: sigma_1 = a + b sigma_3; sin phi = (b-1)/(b+1), c = a(1 - sin phi)/(2 cos phi)."""
    s3, s1 = np.asarray(sigma3, float), np.asarray(sigma1, float)
    slope, intercept = np.polyfit(s3, s1, 1)
    sphi = (slope - 1) / (slope + 1)
    phi = math.asin(min(max(sphi, -1.0), 1.0))
    r2 = float(np.corrcoef(s3, s1)[0, 1] ** 2) if len(s3) > 2 else 1.0
    return MCFit(math.degrees(phi), intercept * (1 - sphi) / (2 * math.cos(phi)), r2, intercept, slope)


def hb_points(sigma_ci: float, mb: float, s: float, a: float, s3_max: float, n: int = 6):
    """n evenly spaced sigma_3 values over 0..s3_max and the Hoek--Brown sigma_1."""
    s3 = np.linspace(0.0, s3_max, n)
    return s3, hb_sigma1(s3, sigma_ci, mb, s, a)


def hoek_2002_mc(sigma_ci: float, hb: HBConstants, sig3_max: float) -> tuple[float, float]:
    """Closed-form equivalent phi (deg) and c (MPa) of Hoek et al. (2002) over 0..sig3_max."""
    mb, s, a = hb.mb, hb.s, hb.a
    s3n = sig3_max / sigma_ci
    pw = (s + mb * s3n) ** (a - 1)
    sphi = 6 * a * mb * pw / (2 * (1 + a) * (2 + a) + 6 * a * mb * pw)
    c = (sigma_ci * ((1 + 2 * a) * s + (1 - a) * mb * s3n) * pw
         / ((1 + a) * (2 + a) * math.sqrt(1 + 6 * a * mb * pw / ((1 + a) * (2 + a)))))
    return math.degrees(math.asin(sphi)), c


def tunnel_sig3_max(sigma_ci: float, hb: HBConstants, p0: float) -> float:
    """Hoek et al. (2002) upper limit of confinement for a tunnel, 0.47 scm' (scm'/p0)^-0.94."""
    scg = global_strength(sigma_ci, hb)
    return 0.47 * scg * (scg / p0) ** -0.94 if scg > 0 and p0 > 0 else p0


# ---------------------------------------------------------------------------
# GSI from a rating (Chapter 10)
# ---------------------------------------------------------------------------


def gsi_from_rmr89_dry(rmr_dry: float) -> float:
    """GSI = RMR'89 - 5, with RMR'89 the dry rating without orientation adjustment."""
    return rmr_dry - 5


def gsi_from_q_prime(q_prime: float) -> float:
    """GSI = 9 ln Q' + 44."""
    return 9 * math.log(q_prime) + 44


# ---------------------------------------------------------------------------
# Typical m_i (Chapter 11 and Appendix G, Table of m_i)
# ---------------------------------------------------------------------------

MI_TYPICAL = {
    "Granite": 32, "Granodiorite": 29, "Diorite": 25, "Gabbro": 27, "Basalt": 25,
    "Rhyolite": 25, "Andesite": 25, "Gneiss": 28, "Schist": 12, "Phyllite": 7,
    "Amphibolite": 26, "Sandstone": 17, "Siltstone": 7, "Claystone": 4,
    "Greywacke": 18, "Limestone, crystalline": 12, "Limestone, micritic": 9,
    "Dolomite": 9, "Marble": 9, "Quartzite": 20, "Slate": 7, "Chalk": 7,
}
"""Table of m_i in Appendix G of the book."""
