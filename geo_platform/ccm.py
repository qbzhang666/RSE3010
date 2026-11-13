"""Convergence--confinement method (Chapter 16 and Appendix F of the book).

Ground reaction curve
    Circular tunnel of radius a in a hydrostatic field p0, Mohr--Coulomb
    ground. p_cr = (2 p0 - sigma_cm,MC)/(1 + k_phi); elastic branch
    u_r = (1+nu)(p0 - p_i) a / E_m. Two plastic branches are offered:

    * Salencon (zero dilation, plastic flow at constant volume):
      u_r = u_r(p_cr) (R_p/a)^2 -- the form derived in the book;
    * Duncan Fama (1993), as published:
      u_r = a(1+nu)/E_m [2(1-nu)(p0 - p_cr)(R_p/a)^2 - (1-2nu)(p0 - p_i)].
      The two coincide at nu = 0.5.

Longitudinal deformation profile (X* = X/a, R* = R_p/a, radius-normalised)
    * Vlachopoulos and Diederichs (2009), as published;
    * Panet (1995): u* = 0.25 + 0.75[1 - (0.75/(0.75 + X*))^2] behind the face;
    * the exponential variant (1 - alpha) e^(1.5 X*) / 1 - alpha e^(-1.5 X*).

Support characteristic curve
    Thin concrete or shotcrete ring, rock bolts (Hoek's simplified form) and
    steel sets (Appendix F). Elements installed together add their stiffness,
    and the combined capacity is reached when the first element reaches its own
    limit (Chapter 16): k_s = sum k_i, u_sm = min u_sm,i, p_sm = k_s u_sm.
    Summing the individual capacities gives a pressure that is never available.

Units: MPa, m for geometry, mm for displacement, GPa for moduli on input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

GRC_METHODS = ("Salençon (zero dilation; the book's form)", "Duncan Fama (1993)")
LDP_METHODS = ("Vlachopoulos and Diederichs (2009)", "Panet (1995)", "Panet, exponential variant")


# ---------------------------------------------------------------------------
# Ground reaction curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ground:
    p0: float          # MPa
    a: float           # m
    nu: float
    em_gpa: float
    phi_deg: float
    c: float           # MPa

    @property
    def k_phi(self) -> float:
        s = math.sin(math.radians(self.phi_deg))
        return (1 + s) / (1 - s)

    @property
    def scm_mc(self) -> float:
        phi = math.radians(self.phi_deg)
        return 2 * self.c * math.cos(phi) / (1 - math.sin(phi))

    @property
    def p_cr(self) -> float:
        return max(0.0, (2 * self.p0 - self.scm_mc) / (1 + self.k_phi))

    @property
    def elastic(self) -> bool:
        return self.p_cr <= 0.0


def plastic_radius(g: Ground, pi: float) -> float:
    """R_p at support pressure p_i; a when the ground stays elastic; inf if unbounded."""
    if pi >= g.p_cr:
        return g.a
    k = g.k_phi
    den = (1 + k) * (pi * (k - 1) + g.scm_mc)
    if den <= 0:
        return math.inf
    return g.a * (2 * (g.p0 * (k - 1) + g.scm_mc) / den) ** (1 / (k - 1))


def grc_displacement(g: Ground, pi: float, method: str = GRC_METHODS[0]) -> float:
    """Wall displacement u_r (mm) at support pressure p_i."""
    em = g.em_gpa * 1000.0
    if pi >= g.p_cr:
        return (1 + g.nu) * (g.p0 - pi) * g.a / em * 1000.0
    rp = plastic_radius(g, pi)
    if not math.isfinite(rp):
        return math.inf
    if method == GRC_METHODS[1]:
        u = (g.a * (1 + g.nu) / em) * (
            2 * (1 - g.nu) * (g.p0 - g.p_cr) * (rp / g.a) ** 2 - (1 - 2 * g.nu) * (g.p0 - pi))
    else:
        u = (1 + g.nu) * (g.p0 - g.p_cr) * g.a / em * (rp / g.a) ** 2
    return u * 1000.0


def grc_curve(g: Ground, method: str = GRC_METHODS[0], n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """(p_i, u_r) from p0 to 0, finer near p_cr."""
    p = np.unique(np.concatenate([np.linspace(0.0, g.p0, n), np.linspace(0.0, max(g.p_cr, 1e-9), n // 2)]))[::-1]
    u = np.array([grc_displacement(g, float(x), method) for x in p])
    return p, u


# ---------------------------------------------------------------------------
# Longitudinal deformation profile
# ---------------------------------------------------------------------------


def ldp(x_over_a: float, rp_over_a: float = 1.0, method: str = LDP_METHODS[0], alpha: float = 0.75) -> float:
    """u(X)/u_max at X/a from the face, positive behind it."""
    x = float(x_over_a)
    if method == LDP_METHODS[0]:
        r = max(float(rp_over_a), 1.0)
        u0 = math.exp(-0.15 * r) / 3
        if x <= 0:
            return u0 * math.exp(x)
        return 1 - (1 - u0) * math.exp(-3 * x / (2 * r))
    if method == LDP_METHODS[1]:
        if x <= 0:
            return 0.25 * math.exp(1.5 * x)   # ahead of the face: the exponential branch (Panet gives none)
        return 0.25 + 0.75 * (1 - (0.75 / (0.75 + x)) ** 2)
    if x <= 0:
        return (1 - alpha) * math.exp(1.5 * x)
    return 1 - alpha * math.exp(-1.5 * x)


def distance_for_ustar(target: float, rp_over_a: float, method: str, a: float, alpha: float = 0.75) -> float:
    """Distance behind the face (m) at which u* reaches ``target``."""
    if target <= ldp(0.0, rp_over_a, method, alpha):
        return 0.0
    if target >= ldp(60.0, rp_over_a, method, alpha):
        return math.inf
    return brentq(lambda x: ldp(x / a, rp_over_a, method, alpha) - target, 0.0, 60.0 * a, xtol=1e-6)


# ---------------------------------------------------------------------------
# Support characteristic curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Support:
    name: str
    psm: float     # MPa
    ks: float      # MPa/m
    usm: float     # mm


def support_ring(a: float, t: float, fcc: float, ec_gpa: float, nu_c: float, name: str = "Concrete ring") -> Support:
    """Thin ring: p_sm = sigma_cc t / a, k_s = E_c t / ((1 - nu_c^2) a^2)."""
    psm = fcc * t / a
    ks = ec_gpa * 1000.0 * t / ((1 - nu_c ** 2) * a ** 2)
    return Support(name, psm, ks, psm / ks * 1000.0)


ROCKBOLTS = {  # tensile capacity (kN), strain to capacity
    "20 mm grouted rebar": (184.0, 0.0024),
    "25 mm grouted rebar": (287.0, 0.0024),
    "34 mm grouted rebar": (500.0, 0.0024),
    "Expandable friction bolt": (110.0, 0.010),
    "Split friction bolt (39 mm)": (100.0, 0.015),
}


def support_bolts(a: float, kind: str, spacing: float) -> Support:
    """Pattern bolting at a square spacing (m): p_sm = T / s^2, u_sm = strain x a (Hoek's simplification)."""
    tension, strain = ROCKBOLTS[kind]
    psm = tension / 1000.0 / spacing ** 2
    usm = strain * a * 1000.0
    return Support(f"Bolts, {kind}, {spacing:g} m", psm, psm / (usm / 1000.0), usm)


STEEL_SETS = {  # area (m^2), yield stress (MPa)
    "Wide-flange 150 mm (38 cm2)": (38.0e-4, 248.0),
    "Wide-flange 200 mm (59 cm2)": (58.9e-4, 248.0),
    "Yielding arch 29 kg/m (37 cm2)": (37.0e-4, 345.0),
    "Yielding arch 36 kg/m (46 cm2)": (45.7e-4, 345.0),
}


def support_steel_sets(a: float, kind: str, spacing: float, es_gpa: float = 200.0) -> Support:
    """Sets at longitudinal spacing s: k_s = E_s A_s/(s a^2), p_sm = A_s sigma_ys/(s a) (Appendix F)."""
    area, fy = STEEL_SETS[kind]
    psm = area * fy / (spacing * a)
    ks = es_gpa * 1000.0 * area / (spacing * a ** 2)
    return Support(f"Steel sets, {kind}, {spacing:g} m", psm, ks, psm / ks * 1000.0)


def combine(supports: list[Support]) -> Support | None:
    """Elements acting together: stiffness adds; capacity is reached at the first element's limit."""
    if not supports:
        return None
    ks = sum(s.ks for s in supports)
    usm = min(s.usm for s in supports)
    return Support(" + ".join(s.name for s in supports), ks * usm / 1000.0, ks, usm)


# ---------------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------------


@dataclass
class Equilibrium:
    p_eq: float | None
    u_eq: float | None
    fos: float | None
    yielded: bool           # the support reached its capacity before equilibrium
    note: str = ""


def equilibrium(g: Ground, sup: Support, us0: float, method: str = GRC_METHODS[0]) -> Equilibrium:
    """Intersection of the ground reaction curve with the support line from (u_s0, 0)."""
    def gap(p: float) -> float:   # ground demand minus support supply, as a displacement
        return grc_displacement(g, p, method) - (us0 + p / sup.ks * 1000.0)

    if gap(0.0) <= 0:
        return Equilibrium(0.0, grc_displacement(g, 0.0, method), math.inf, False,
                           "The ground comes to rest before the support is loaded.")
    if gap(sup.psm) > 0:
        # Support yields; with a perfectly plastic support the ground finds rest on the plateau
        # where its demand equals p_sm, if it reaches that pressure at all.
        u_plateau = grc_displacement(g, sup.psm, method)
        if math.isfinite(u_plateau):
            return Equilibrium(sup.psm, u_plateau, 1.0, True,
                               "The support reaches its capacity before equilibrium and deforms "
                               "plastically; the factor of safety is 1.0 by definition.")
        return Equilibrium(None, None, None, True, "No equilibrium: the support yields and the ground keeps moving.")
    p_eq = brentq(gap, 0.0, sup.psm, xtol=1e-9)
    return Equilibrium(p_eq, us0 + p_eq / sup.ks * 1000.0, sup.psm / p_eq, False)


@dataclass
class CCMResult:
    ground: Ground
    method: str
    ldp_method: str
    u_max: float
    u_pcr: float
    rp0: float
    install_distance: float
    ustar: float
    us0: float
    support: Support | None
    eq: Equilibrium | None
    rp_eq: float = field(default=float("nan"))


def analyse(g: Ground, supports: list[Support], *, method: str = GRC_METHODS[0],
            ldp_method: str = LDP_METHODS[0], alpha: float = 0.75,
            install: str = "distance", value: float = 3.0) -> CCMResult:
    """Run the method. ``install`` is "distance" (m behind the face), "us0" (mm) or "strain" (u_s0/a, %)."""
    u_max = grc_displacement(g, 0.0, method)
    rp0 = plastic_radius(g, 0.0)
    rstar = rp0 / g.a if math.isfinite(rp0) else 20.0
    u_pcr = grc_displacement(g, g.p_cr, method) if not g.elastic else 0.0
    if install == "us0":
        us0 = float(value)
        dist = distance_for_ustar(us0 / u_max, rstar, ldp_method, g.a, alpha) if u_max > 0 else 0.0
    elif install == "strain":
        us0 = float(value) / 100.0 * g.a * 1000.0
        dist = distance_for_ustar(us0 / u_max, rstar, ldp_method, g.a, alpha) if u_max > 0 else 0.0
    else:
        dist = float(value)
        us0 = ldp(dist / g.a, rstar, ldp_method, alpha) * u_max
    ustar = us0 / u_max if u_max > 0 else float("nan")
    sup = combine(supports)
    eq = equilibrium(g, sup, us0, method) if sup is not None and math.isfinite(u_max) else None
    rp_eq = plastic_radius(g, eq.p_eq) if eq is not None and eq.p_eq is not None else float("nan")
    return CCMResult(g, method, ldp_method, u_max, u_pcr, rp0, dist, ustar, us0, sup, eq, rp_eq)
