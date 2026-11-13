"""Ground movement above a tunnel (Chapters 17 and 26, Appendix H of the book).

Transverse trough (Gaussian):  S(y) = S_max exp(-y^2 / 2 i^2), i = K z0,
    V_s = (V_L / 100) pi D^2 / 4 with V_L in per cent, S_max = V_s / (i sqrt(2 pi)).
Longitudinal development (cumulative normal, Attewell and Woodman 1982):
    S(x, y) = S(y) Phi((x_f - x) / i), x_f the face position; half the final
    settlement at the face, the whole of it well behind. (A Gaussian
    longitudinal profile, which returns to zero behind the face, is wrong.)
Horizontal movement towards the axis (O'Reilly and New 1982):
    u_h = y S / z0, and the horizontal strain, tension positive,
    eps_h = (S / z0)(y^2 / i^2 - 1): compressive over the centre, tensile
    beyond the point of inflection.
Twin tunnels are superposed.

Lengths in m, settlement in mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Bore:
    offset: float      # horizontal position of the axis (m)
    z0: float          # depth to axis (m)
    diameter: float    # excavated diameter (m)
    vl_percent: float  # volume loss (%)
    k: float           # trough width parameter K

    @property
    def i(self) -> float:
        return self.k * self.z0

    @property
    def vs(self) -> float:
        return self.vl_percent / 100.0 * math.pi * self.diameter ** 2 / 4.0

    @property
    def smax_mm(self) -> float:
        return self.vs / (self.i * math.sqrt(2 * math.pi)) * 1000.0


def transverse(bores: list[Bore], y) -> np.ndarray:
    """Final settlement (mm) at offsets y, superposed over the bores."""
    y = np.asarray(y, float)
    total = np.zeros_like(y)
    for b in bores:
        total += b.smax_mm * np.exp(-((y - b.offset) ** 2) / (2 * b.i ** 2))
    return total


def horizontal(bores: list[Bore], y) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal displacement (mm, towards each axis) and strain (microstrain, tension +)."""
    y = np.asarray(y, float)
    disp = np.zeros_like(y)
    strain = np.zeros_like(y)
    for b in bores:
        dy = y - b.offset
        s = b.smax_mm * np.exp(-(dy ** 2) / (2 * b.i ** 2))
        disp += -dy * s / b.z0                     # towards the axis
        strain += (s / 1000.0 / b.z0) * (dy ** 2 / b.i ** 2 - 1) * 1e6
    return disp, strain


def slope(bores: list[Bore], y) -> np.ndarray:
    """Ground slope dS/dy (rad, dimensionless)."""
    y = np.asarray(y, float)
    out = np.zeros_like(y)
    for b in bores:
        dy = y - b.offset
        s = b.smax_mm / 1000.0 * np.exp(-(dy ** 2) / (2 * b.i ** 2))
        out += -dy / b.i ** 2 * s
    return out


def longitudinal_factor(x, face: float, i: float):
    """Fraction of the final settlement developed at x when the face is at ``face``."""
    return norm.cdf((face - np.asarray(x, float)) / i)


def plan_field(bores: list[Bore], faces: list[float], x, y) -> np.ndarray:
    """Settlement (mm) on an x-y grid; each bore has its own face position (x increasing in the drive direction)."""
    xx, yy = np.meshgrid(np.asarray(x, float), np.asarray(y, float))
    total = np.zeros_like(xx)
    for b, f in zip(bores, faces):
        total += b.smax_mm * np.exp(-((yy - b.offset) ** 2) / (2 * b.i ** 2)) * norm.cdf((f - xx) / b.i)
    return total


def screen_building(bores: list[Bore], y_left: float, y_right: float,
                    s_limit_mm: float = 10.0, slope_limit: float = 1 / 500) -> dict:
    """First-stage screen over a building footprint (Chapter 26, Appendix H).

    A building is excluded only where both the settlement is below ``s_limit_mm``
    and the slope is below ``slope_limit``.
    """
    y = np.linspace(min(y_left, y_right), max(y_left, y_right), 201)
    s = transverse(bores, y)
    sl = np.abs(slope(bores, y))
    _, eps = horizontal(bores, y)
    smax, slmax = float(s.max()), float(sl.max())
    return {
        "max_settlement_mm": smax,
        "max_slope": slmax,
        "max_tensile_strain_microstrain": float(max(eps.max(), 0.0)),
        "excluded": smax < s_limit_mm and slmax < slope_limit,
    }


K_TYPICAL = {  # indicative trough width parameter (Appendix G)
    "Soft clay": 0.6,
    "Stiff clay": 0.5,
    "Granular soil": 0.35,
    "Weak rock": 0.3,
}
