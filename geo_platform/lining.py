"""Segmental lining checks (Chapter 19 and Appendix F of the book).

1. A segment under a transient load case as a curved member.

   The segment is a circular arc of radius R (to the centroid) subtending an
   angle theta, symmetric about its crown, carried on a pin at one end and a
   roller at the other, so the support reactions are vertical and the member
   is statically determinate. The load is the self-weight plus an optional
   uniform vertical load, both per metre of arc. At the section at angle phi
   from the crown (x = R sin phi):

       M(phi) = V_A R (sin phi + sin theta/2)
                - q R^2 [sin phi (phi + theta/2) + cos phi - cos theta/2]
       F_v    = V_A - q R (phi + theta/2)          (vertical force, left part)
       N      = -F_v sin phi                       (compression positive)
       V      =  F_v cos phi

   with V_A = q R theta / 2 and M positive when the intrados is in tension.
   For a semicircle the crown moment is q R^2 (pi/2 - 1); as R grows at a
   fixed chord the expressions tend to the straight beam, q L^2 / 8. The
   crown deflection follows by virtual work from bending alone.

   The earlier course app took the arc length as the span of a straight,
   simply supported beam and so computed no axial force; that is replaced.

2. A ring in the ground: the compressibility and flexibility ratios of
   Appendix F, C = E a (1 - nu_l^2) / (E_l A_l (1 - nu^2)) and
   F = E a^3 (1 - nu_l^2) / (E_l I_l (1 - nu^2)), per metre of tunnel, and the
   hoop thrust of a thin ring under a uniform radial pressure, N = p a.

Units: kN, m and kPa on input unless stated; stresses reported in MPa.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid


@dataclass(frozen=True)
class Segment:
    radius: float        # to the centroid (m)
    angle_deg: float     # subtended angle
    width: float         # ring length b (m)
    thickness: float     # h (m)
    unit_weight: float   # concrete (kN/m3)
    extra_load: float    # additional vertical load per metre of arc (kN/m)
    e_gpa: float

    @property
    def theta(self) -> float:
        return math.radians(self.angle_deg)

    @property
    def q(self) -> float:
        return self.unit_weight * self.width * self.thickness + self.extra_load

    @property
    def area(self) -> float:
        return self.width * self.thickness

    @property
    def inertia(self) -> float:
        return self.width * self.thickness ** 3 / 12.0

    @property
    def modulus(self) -> float:
        return self.width * self.thickness ** 2 / 6.0

    @property
    def chord(self) -> float:
        return 2 * self.radius * math.sin(self.theta / 2)

    @property
    def rise(self) -> float:
        return self.radius * (1 - math.cos(self.theta / 2))


def segment_actions(seg: Segment, n: int = 721) -> dict[str, np.ndarray]:
    """Axial force N (kN, compression +), shear V (kN), moment M (kNm) along the arc."""
    r, th, q = seg.radius, seg.theta, seg.q
    phi = np.linspace(-th / 2, th / 2, n)
    va = q * r * th / 2
    m = va * r * (np.sin(phi) + math.sin(th / 2)) - q * r ** 2 * (
        np.sin(phi) * (phi + th / 2) + np.cos(phi) - math.cos(th / 2))
    fv = va - q * r * (phi + th / 2)
    n_force = -fv * np.sin(phi)
    v = fv * np.cos(phi)
    s = r * (phi + th / 2)
    return {"phi": phi, "s": s, "x": r * np.sin(phi), "y": r * np.cos(phi), "N": n_force, "V": v, "M": m,
            "reaction": np.array([va])}


def segment_stresses(seg: Segment, act: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fibre stresses (MPa, compression +) at the intrados and extrados."""
    axial = act["N"] / seg.area / 1000.0
    bend = act["M"] / seg.modulus / 1000.0
    return {"intrados": axial - bend, "extrados": axial + bend}


def crown_deflection_mm(seg: Segment, act: dict[str, np.ndarray]) -> float:
    """Vertical deflection of the crown by virtual work (bending only), downward positive."""
    r, th = seg.radius, seg.theta
    phi = act["phi"]
    unit = 0.5 * r * (np.sin(phi) + math.sin(th / 2)) - np.where(phi > 0, r * np.sin(phi), 0.0)
    ei = seg.e_gpa * 1e6 * seg.inertia            # kN m^2
    return float(trapezoid(act["M"] * unit / ei, r * phi) * 1000.0)


@dataclass(frozen=True)
class Ring:
    radius: float          # a (m)
    thickness: float       # t (m)
    e_lining_gpa: float
    nu_lining: float
    e_ground_mpa: float
    nu_ground: float
    joint_factor: float = 1.0   # effective I_eff / I for joints (Muir Wood's reduction), 1 = no joints


def ring_ratios(ring: Ring) -> dict[str, float]:
    """Compressibility C and flexibility F per metre of tunnel (Appendix F)."""
    area = ring.thickness * 1.0
    inertia = ring.joint_factor * ring.thickness ** 3 / 12.0
    el = ring.e_lining_gpa * 1000.0
    num = ring.e_ground_mpa * (1 - ring.nu_lining ** 2)
    den = el * (1 - ring.nu_ground ** 2)
    return {
        "C": num * ring.radius / (den * area),
        "F": num * ring.radius ** 3 / (den * inertia),
    }


def hoop_thrust(pressure_kpa: float, radius: float) -> float:
    """Thin ring under a uniform radial pressure: N = p a (kN per metre of tunnel)."""
    return pressure_kpa * radius
