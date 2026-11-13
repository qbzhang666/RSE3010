"""Rock-slope calculations: orientations, stereonets, SMR, kinematics and limit equilibrium.

Pure Python and numpy (scipy only for root finding); Streamlit is never
imported here, so every function can be unit tested.  The pages that use this
module are in :mod:`geo_platform.slope_pages`.

Conventions
-----------
* Planes are given as dip / dip direction (degrees); lines as trend / plunge.
* Vectors are (east, north, up).  ``plane_normal`` is the *upward* unit normal;
  a line vector points *down* (lower hemisphere).
* Stereonets are lower-hemisphere, unit primitive radius, x east and y north,
  equal-angle (Wulff) or equal-area (Schmidt).
* Forces are per metre run (2D) or totals (3D wedge): kN/m or kN; stresses kPa.

Sources and the corrections made to the course apps
---------------------------------------------------
The course apps ported here (``SMR Slope.py``, ``smr_streamlit.py``,
``Kinematic Slope.py``, ``Stereonet and LEM.py``, ``LEM Slope.py``) were treated
as claims to verify.  The standard sources are Romana (1985, 1993) for SMR,
Markland (1972), Hocking (1976) and Goodman and Bray (1976) for kinematic
screening, Hoek and Bray (1981) / Wyllie and Mah (2004) for planar and wedge
sliding, and Bishop (1955) / Fellenius (1936) for circular sliding.

1. SMR, F3 for planar and wedge sliding (``smr_streamlit.py``): the
   "unfavourable" class was bounded at C >= -5 deg, so C between -5 and -10
   deg received -60.  Romana's table bounds it at -10 deg (0 to -10 deg gives
   -50; below -10 deg gives -60), as ``SMR Slope.py`` had it.  Corrected.
2. SMR, F1 (``smr_streamlit.py``): F1 = (1 - sin A)^2 for A <= 30 deg but 0.15
   above, a jump from 0.25 to 0.15 at A = 30 deg, and a continuous F1 mixed
   with a tabulated F2.  Replaced by a single choice for both factors: Romana's
   tabulated values (default), or Romana's continuous fits F1 = (1 - sin A)^2
   and F2 = tan^2 B, each bounded to the table's range 0.15 to 1.00 (the bound
   is ours: the fits leave the table range outside about A = 38 deg and
   B = 21 deg and 45 deg).
3. SMR classes (``SMR Slope.py``): thresholds ``>= 81, >= 61, ...`` put a
   non-integer SMR of 80.5 in class II.  Romana's integer ranges 81-100,
   61-80, ... are read here as SMR > 80, > 60, > 40, > 20.  The class V failure
   description "large wedges / circular" is replaced by Romana's "big planar or
   soil-like".
4. Kinematic screening: ``Kinematic Slope.py`` contains no kinematic screening
   (it is a second copy of the planar limit-equilibrium app), so the screening
   here is written from the sources: planar sliding (Markland), wedge sliding
   (Markland, with Hocking's single-plane refinement) and flexural toppling
   (Goodman and Bray).  The daylight test compares the plunge with the
   *apparent* dip of the face in the sliding direction, which is what the
   stereonet crescent shows.
5. Planar sliding (``Kinematic Slope.py``, ``Stereonet and LEM.py``): the weight
   W = 1/2 gamma H^2 [(1 - (z/H)^2) cot psi_p - cot psi_f] holds only when the
   tension crack is behind the crest, i.e. z <= H (1 - cot psi_f tan psi_p).
   The apps used it for any z.  For a crack in the face Hoek and Bray give
   W = 1/2 gamma H^2 (1 - z/H)^2 cot psi_p (cot psi_p tan psi_f - 1); both are
   used here (they agree at the transition), and z_w is limited to the open
   length of the crack.  The U, V, anchor and factor-of-safety expressions of
   the apps were checked and are correct (active anchor, Hoek and Bray form);
   a passive (untensioned) anchor option and the optimum anchor angle
   tan(psi_p + psi_T) = tan(phi) / F were added.
6. Planar sliding in ``LEM Slope.py``: water entered as U = r_u N, which is not
   a planar-sliding water model (r_u is a slice-method parameter, as the other
   course app itself notes), and there was no tension crack.  Planar sliding is
   therefore done once, in the Hoek and Bray form above, not on the wedge and
   circular page.
7. Wedge sliding (``LEM Slope.py``): the "wedge" was the 2D planar triangle
   between the face and one plane, with no second plane, so it was not a wedge
   analysis.  Replaced by the Hoek and Bray wedge (the comprehensive solution
   written in vector form): the tetrahedron is built from the two planes, the
   face, the upper surface and the height H of the line of intersection; the
   normal reactions follow from equilibrium normal to that line; water uses
   Hoek and Bray's fully saturated distribution (zero on the outcrop edges,
   peak 1/2 gamma_w H at mid-height of the intersection, so the mean pressure
   on each face is gamma_w H / 6).  It reproduces the wedge factor
   K = sin(beta) / sin(xi / 2) for a symmetric frictional wedge.
8. Circular sliding (``LEM Slope.py``): the sliding mass was the area of a full
   circle pi R^2, cohesion multiplied that area instead of the slip-surface
   length, and the driving force used the dip of a plane.  Replaced by
   Bishop's simplified method (and the ordinary method of slices for
   comparison) with r_u, and a search over toe circles.  The support force T
   of that app is not carried to the wedge and circular calculations.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Orientation geometry
# ---------------------------------------------------------------------------


def _rad(deg: float) -> float:
    return math.radians(float(deg))


def angular_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two azimuths, 0 to 180 deg."""
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def plane_normal(dip: float, dip_dir: float) -> np.ndarray:
    """Upward unit normal (east, north, up) of a plane."""
    d, a = _rad(dip), _rad(dip_dir)
    return np.array([math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d)])


def dip_vector(dip: float, dip_dir: float) -> np.ndarray:
    """Unit vector pointing down the dip of a plane."""
    d, a = _rad(dip), _rad(dip_dir)
    return np.array([math.cos(d) * math.sin(a), math.cos(d) * math.cos(a), -math.sin(d)])


def strike_vector(dip_dir: float) -> np.ndarray:
    """Horizontal unit vector along strike (right-hand rule, dip direction - 90)."""
    a = _rad(dip_dir - 90.0)
    return np.array([math.sin(a), math.cos(a), 0.0])


def line_vector(trend: float, plunge: float) -> np.ndarray:
    """Downward unit vector of a line given by trend / plunge."""
    t, p = _rad(trend), _rad(plunge)
    return np.array([math.cos(p) * math.sin(t), math.cos(p) * math.cos(t), -math.sin(p)])


def trend_plunge(vector: Sequence[float]) -> tuple[float, float]:
    """Trend and plunge (deg) of a vector, taken in the lower hemisphere."""
    v = np.asarray(vector, dtype=float)
    v = v / np.linalg.norm(v)
    if v[2] > 0:
        v = -v
    trend = math.degrees(math.atan2(v[0], v[1])) % 360.0
    plunge = math.degrees(math.asin(min(1.0, max(-1.0, -v[2]))))
    return trend, plunge


def intersection_line(dip1: float, dd1: float, dip2: float, dd2: float) -> tuple[float, float] | None:
    """Trend / plunge of the line common to two planes; ``None`` if parallel."""
    v = np.cross(plane_normal(dip1, dd1), plane_normal(dip2, dd2))
    if np.linalg.norm(v) < 1e-9:
        return None
    return trend_plunge(v)


def apparent_dip(dip: float, dip_dir: float, direction: float) -> float:
    """Apparent dip (deg) of a plane in a vertical section of given azimuth.

    tan(apparent) = tan(dip) cos(direction - dip direction); negative when the
    plane rises in that direction.
    """
    d = _rad(dip)
    delta = _rad(direction - dip_dir)
    return math.degrees(math.atan2(math.sin(d) * math.cos(delta), math.cos(d)))


def pole(dip: float, dip_dir: float) -> tuple[float, float]:
    """Trend / plunge of the lower-hemisphere pole (normal) of a plane."""
    return (float(dip_dir) + 180.0) % 360.0, 90.0 - float(dip)


# ---------------------------------------------------------------------------
# Stereographic projection (lower hemisphere, unit primitive)
# ---------------------------------------------------------------------------

EQUAL_ANGLE = "equal-angle"
EQUAL_AREA = "equal-area"


def projection_radius(plunge, net: str = EQUAL_ANGLE):
    """Distance from the centre of the net of a line with the given plunge."""
    theta = np.radians(90.0 - np.asarray(plunge, dtype=float))
    if net == EQUAL_AREA:
        return np.sqrt(2.0) * np.sin(theta / 2.0)
    return np.tan(theta / 2.0)


def project(trend, plunge, net: str = EQUAL_ANGLE):
    """Net coordinates (x east, y north) of lines; arrays are accepted."""
    r = projection_radius(plunge, net)
    t = np.radians(np.asarray(trend, dtype=float))
    return r * np.sin(t), r * np.cos(t)


def unproject(x, y, net: str = EQUAL_ANGLE):
    """Trend and plunge (deg) of the lines at net coordinates ``x, y``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.hypot(x, y)
    trend = np.degrees(np.arctan2(x, y)) % 360.0
    if net == EQUAL_AREA:
        theta = 2.0 * np.arcsin(np.clip(r / np.sqrt(2.0), 0.0, 1.0))
    else:
        theta = 2.0 * np.arctan(r)
    return trend, 90.0 - np.degrees(theta)


def great_circle(dip: float, dip_dir: float, net: str = EQUAL_ANGLE, n: int = 181):
    """Net coordinates of the great circle of a plane."""
    s = strike_vector(dip_dir)
    d = dip_vector(dip, dip_dir)
    t = np.linspace(0.0, math.pi, n)
    v = np.outer(np.cos(t), s) + np.outer(np.sin(t), d)
    v[:, 2] = np.minimum(v[:, 2], 0.0)
    trend = np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360.0
    plunge = np.degrees(np.arcsin(np.clip(-v[:, 2], 0.0, 1.0)))
    return project(trend, plunge, net)


def plunge_circle(plunge: float, net: str = EQUAL_ANGLE, n: int = 361):
    """Small circle of all lines with one plunge (e.g. the friction circle)."""
    t = np.linspace(0.0, 360.0, n)
    return project(t, np.full_like(t, float(plunge)), net)


# ---------------------------------------------------------------------------
# Slope Mass Rating (Romana 1985, 1993; wedge extension by Anbalagan 1992)
# ---------------------------------------------------------------------------

F4_METHODS: dict[str, int] = {
    "Natural slope (+15)": 15,
    "Pre-splitting (+10)": 10,
    "Smooth blasting (+8)": 8,
    "Normal blasting or mechanical excavation (0)": 0,
    "Deficient blasting (-8)": -8,
}

SMR_CLASSES = (
    # lower bound (exclusive), class, description, stability, failures, support
    (80.0, "I", "Very good", "Completely stable", "None", "None"),
    (60.0, "II", "Good", "Stable", "Some blocks", "Occasional"),
    (40.0, "III", "Fair", "Partially stable", "Some joints or many wedges", "Systematic"),
    (20.0, "IV", "Poor", "Unstable", "Planar or big wedges", "Important / corrective"),
    (-math.inf, "V", "Very poor", "Completely unstable", "Big planar or soil-like", "Re-excavation"),
)


def smr_f1(A: float, continuous: bool = False) -> float:
    """F1 from the parallelism A between the slope face and the joint or line (deg)."""
    a = abs(float(A))
    if continuous:
        return float(min(1.0, max(0.15, (1.0 - math.sin(_rad(min(a, 90.0)))) ** 2)))
    if a > 30.0:
        return 0.15
    if a > 20.0:
        return 0.40
    if a > 10.0:
        return 0.70
    if a > 5.0:
        return 0.85
    return 1.00


def smr_f2(B: float | None, mode: str, continuous: bool = False) -> float:
    """F2 from the joint dip (planar) or intersection plunge (wedge); 1 for toppling."""
    if mode.lower() == "toppling" or B is None:
        return 1.0
    b = abs(float(B))
    if continuous:
        return float(min(1.0, max(0.15, math.tan(_rad(min(b, 89.999))) ** 2)))
    if b < 20.0:
        return 0.15
    if b <= 30.0:
        return 0.40
    if b <= 35.0:
        return 0.70
    if b <= 45.0:
        return 0.85
    return 1.00


def smr_f3(C: float, mode: str) -> float:
    """F3 from the relation C between joint (or line) dip and slope dip (deg)."""
    c = float(C)
    m = mode.lower()
    if m in ("planar", "wedge"):
        if c > 10.0:
            return 0.0
        if c > 1e-6:
            return -6.0
        if c >= -1e-6:
            return -25.0
        if c >= -10.0:
            return -50.0
        return -60.0
    if m == "toppling":
        if c < 110.0:
            return 0.0
        if c <= 120.0:
            return -6.0
        return -25.0
    raise ValueError("mode must be planar, wedge or toppling")


def smr_class(smr: float) -> dict[str, str]:
    """Romana's class, description, stability, failures and support for an SMR."""
    for bound, cls, quality, stability, failures, support in SMR_CLASSES:
        if smr > bound:
            return {"Class": cls, "Quality": quality, "Stability": stability,
                    "Failures": failures, "Support": support}
    raise AssertionError("unreachable")


def smr_case(mode: str, A: float, B: float | None, C: float, rmr: float, f4: float,
             continuous: bool = False) -> dict:
    """One SMR evaluation: SMR = RMRb + F1 F2 F3 + F4."""
    f1 = smr_f1(A, continuous)
    f2 = smr_f2(B, mode, continuous)
    f3 = smr_f3(C, mode)
    product = f1 * f2 * f3
    smr = float(rmr) + product + float(f4)
    out = {"A (deg)": A, "B (deg)": B, "C (deg)": C, "F1": f1, "F2": f2, "F3": f3,
           "F1 F2 F3": product, "F4": float(f4), "RMRb": float(rmr), "SMR": smr}
    out.update(smr_class(smr))
    return out


def smr_planar(joint_dip: float, joint_dd: float, face_dip: float, face_dd: float,
               rmr: float, f4: float, continuous: bool = False) -> dict:
    """Planar sliding: A = |aj - as|, B = bj, C = bj - bs."""
    return smr_case("planar", angular_difference(joint_dd, face_dd), abs(joint_dip),
                    joint_dip - face_dip, rmr, f4, continuous)


def smr_toppling(joint_dip: float, joint_dd: float, face_dip: float, face_dd: float,
                 rmr: float, f4: float, continuous: bool = False) -> dict:
    """Toppling: A = |aj - as - 180|, F2 = 1, C = bj + bs."""
    return smr_case("toppling", angular_difference(joint_dd, face_dd + 180.0), None,
                    joint_dip + face_dip, rmr, f4, continuous)


def smr_wedge(trend: float, plunge: float, face_dip: float, face_dd: float,
              rmr: float, f4: float, continuous: bool = False) -> dict:
    """Wedge sliding on a line of intersection: A = |ai - as|, B = bi, C = bi - bs."""
    return smr_case("wedge", angular_difference(trend, face_dd), abs(plunge),
                    plunge - face_dip, rmr, f4, continuous)


def smr_table(joints: Iterable[tuple[str, float, float]], faces: Iterable[tuple[str, float, float]],
              rmr: float, f4: float, continuous: bool = False) -> list[dict]:
    """SMR rows for every face: planar and toppling per joint, wedge per joint pair.

    ``joints`` and ``faces`` are (name, dip, dip direction) tuples.
    """
    joints = list(joints)
    rows: list[dict] = []
    for face_name, f_dip, f_dd in faces:
        for name, dip, dd in joints:
            row = {"Face": face_name, "Mode": "Planar", "Case": name}
            row.update(smr_planar(dip, dd, f_dip, f_dd, rmr, f4, continuous))
            rows.append(row)
        for (n1, d1, a1), (n2, d2, a2) in combinations(joints, 2):
            line = intersection_line(d1, a1, d2, a2)
            if line is None:
                continue
            trend, plunge = line
            row = {"Face": face_name, "Mode": "Wedge",
                   "Case": f"{n1}-{n2} ({plunge:.0f}/{trend:03.0f})"}
            row.update(smr_wedge(trend, plunge, f_dip, f_dd, rmr, f4, continuous))
            rows.append(row)
        for name, dip, dd in joints:
            row = {"Face": face_name, "Mode": "Toppling", "Case": name}
            row.update(smr_toppling(dip, dd, f_dip, f_dd, rmr, f4, continuous))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Kinematic screening (Markland 1972; Hocking 1976; Goodman and Bray 1976)
# ---------------------------------------------------------------------------


def planar_kinematics(dip: float, dip_dir: float, face_dip: float, face_dd: float,
                      phi: float, lateral_limit: float = 20.0) -> dict:
    """Planar sliding test: phi < dip < apparent face dip, within the lateral limit."""
    delta = angular_difference(dip_dir, face_dd)
    face_app = apparent_dip(face_dip, face_dd, dip_dir)
    daylights = delta < 90.0 and dip < face_app
    steeper = dip > phi
    aligned = delta <= lateral_limit
    return {"Dip (deg)": dip, "Dip direction (deg)": dip_dir,
            "Face apparent dip (deg)": face_app, "Offset from face (deg)": delta,
            "Daylights": daylights, "Steeper than phi": steeper, "Within lateral limit": aligned,
            "Possible": bool(daylights and steeper and aligned)}


def _between(azimuth: float, start: float, end: float) -> bool:
    """True if ``azimuth`` lies on the shorter arc from ``start`` to ``end``."""
    span = angular_difference(start, end)
    return angular_difference(azimuth, start) + angular_difference(azimuth, end) <= span + 1e-9


def wedge_kinematics(plane1: tuple[float, float], plane2: tuple[float, float],
                     face_dip: float, face_dd: float, phi: float) -> dict:
    """Wedge sliding test on the line of intersection of two planes (dip, dip direction).

    Possible when the line daylights in the face (plunge below the face's
    apparent dip in its trend) and plunges more steeply than phi.  Hocking's
    refinement: if the dip direction of one plane lies between the trend of the
    line and the face dip direction, that plane slides alone (planar sliding).
    """
    line = intersection_line(plane1[0], plane1[1], plane2[0], plane2[1])
    if line is None:
        return {"Trend (deg)": math.nan, "Plunge (deg)": math.nan, "Face apparent dip (deg)": math.nan,
                "Daylights": False, "Steeper than phi": False, "Possible": False,
                "Sliding on": "parallel planes"}
    trend, plunge = line
    delta = angular_difference(trend, face_dd)
    face_app = apparent_dip(face_dip, face_dd, trend)
    daylights = delta < 90.0 and plunge < face_app
    steeper = plunge > phi
    possible = bool(daylights and steeper)
    sliding_on = "both planes"
    if possible:
        if _between(plane1[1], trend, face_dd) and plane1[0] > phi:
            sliding_on = "plane 1 alone"
        elif _between(plane2[1], trend, face_dd) and plane2[0] > phi:
            sliding_on = "plane 2 alone"
    return {"Trend (deg)": trend, "Plunge (deg)": plunge, "Face apparent dip (deg)": face_app,
            "Daylights": daylights, "Steeper than phi": steeper, "Possible": possible,
            "Sliding on": sliding_on if possible else "-"}


def toppling_kinematics(dip: float, dip_dir: float, face_dip: float, face_dd: float,
                        phi: float, lateral_limit: float = 10.0) -> dict:
    """Flexural toppling test (Goodman and Bray): dip >= 90 - face dip + phi, dipping into the face."""
    delta = angular_difference(dip_dir, face_dd + 180.0)
    limit = 90.0 - face_dip + phi
    steep = dip >= limit
    aligned = delta <= lateral_limit
    return {"Dip (deg)": dip, "Dip direction (deg)": dip_dir, "Offset from opposite of face (deg)": delta,
            "Required dip (deg)": limit, "Steep enough": steep, "Within lateral limit": aligned,
            "Possible": bool(steep and aligned)}


# ---------------------------------------------------------------------------
# Planar sliding (Hoek and Bray 1981; Wyllie and Mah 2004)
# ---------------------------------------------------------------------------


def critical_tension_crack(H: float, psi_f: float, psi_p: float) -> tuple[float, float]:
    """Critical (dry) tension crack depth z_c and its distance b behind the crest.

    z_c / H = 1 - sqrt(cot psi_f tan psi_p);  b / H = sqrt(cot psi_f cot psi_p) - cot psi_f.
    """
    if psi_p >= psi_f:
        return math.nan, math.nan
    cot_f = 1.0 / math.tan(_rad(psi_f))
    tan_p = math.tan(_rad(psi_p))
    z = H * (1.0 - math.sqrt(cot_f * tan_p))
    b = H * (math.sqrt(cot_f / tan_p) - cot_f)
    return z, b


def crack_behind_crest_limit(H: float, psi_f: float, psi_p: float) -> float:
    """Largest z for which the tension crack lies behind the crest: H (1 - cot psi_f tan psi_p)."""
    return H * (1.0 - math.tan(_rad(psi_p)) / math.tan(_rad(psi_f)))


def planar_geometry(H: float, psi_f: float, psi_p: float, z: float) -> dict:
    """Block geometry for planar sliding with a vertical tension crack of depth z.

    ``z`` is the depth of the crack base below the crest level, as in Hoek and
    Bray, for a crack either behind the crest or in the face.  Returns the
    block area (m^2/m), base length A (m/m), crack position and open length.
    """
    tf, tp = math.tan(_rad(psi_f)), math.tan(_rad(psi_p))
    cot_f, cot_p = 1.0 / tf, 1.0 / tp
    x_crack = (H - z) * cot_p                  # horizontal distance toe -> crack
    in_face = x_crack < H * cot_f - 1e-12
    if in_face:
        area = 0.5 * H ** 2 * (1.0 - z / H) ** 2 * cot_p * (cot_p * tf - 1.0)
        y_top = x_crack * tf
    else:
        area = 0.5 * H ** 2 * ((1.0 - (z / H) ** 2) * cot_p - cot_f)
        y_top = H
    return {"area": area, "A": (H - z) / math.sin(_rad(psi_p)), "x_crack": x_crack,
            "y_crack_base": H - z, "y_crack_top": y_top, "crack_length": y_top - (H - z),
            "crack_in_face": in_face, "x_crest": H * cot_f}


def planar_fos(H: float, psi_f: float, psi_p: float, z: float, z_w: float, c: float, phi: float,
               gamma: float, gamma_w: float = 9.81, T: float = 0.0, psi_T: float = 0.0,
               anchor: str = "active") -> dict:
    """Factor of safety of planar sliding, Hoek and Bray form, per metre run.

    FoS = [cA + (W cos psi_p - U - V sin psi_p + T sin(psi_p + psi_T)) tan phi]
          / [W sin psi_p + V cos psi_p - T cos(psi_p + psi_T)]               (active)
    For a passive anchor T cos(psi_p + psi_T) is added to the numerator instead.
    psi_T is the anchor inclination below horizontal.  The effective normal
    force is not allowed to go below zero.
    """
    if psi_p >= psi_f:
        return {"valid": False, "reason": "The sliding plane does not daylight (psi_p >= psi_f)."}
    if psi_p <= 0.0:
        return {"valid": False, "reason": "The sliding plane must dip out of the face (psi_p > 0)."}
    if not 0.0 <= z < H:
        return {"valid": False, "reason": "The tension crack depth must satisfy 0 <= z < H."}
    geo = planar_geometry(H, psi_f, psi_p, z)
    if z_w < 0.0 or z_w > geo["crack_length"] + 1e-9:
        return {"valid": False,
                "reason": f"The water depth z_w must lie between 0 and the open crack length "
                          f"{geo['crack_length']:.2f} m."}
    p, f, t = _rad(psi_p), _rad(phi), _rad(psi_p + psi_T)
    A = geo["A"]
    W = gamma * geo["area"]
    V = 0.5 * gamma_w * z_w ** 2
    U = 0.5 * gamma_w * z_w * A
    N0 = W * math.cos(p) - U - V * math.sin(p)
    D0 = W * math.sin(p) + V * math.cos(p)
    T_shear = T * math.cos(t)
    T_normal = T * math.sin(t)
    N = N0 + T_normal
    N_used = max(0.0, N)
    if anchor == "passive":
        driving = D0
        resisting = c * A + N_used * math.tan(f) + T_shear
    else:
        driving = D0 - T_shear
        resisting = c * A + N_used * math.tan(f)
    fos = resisting / driving if driving > 0 else math.inf
    return {"valid": True, "A": A, "W": W, "V": V, "U": U, "T_shear": T_shear, "T_normal": T_normal,
            "N": N, "N_used": N_used, "driving": driving, "resisting": resisting, "FoS": fos,
            "crack_in_face": geo["crack_in_face"], "geometry": geo}


def planar_required_anchor(H: float, psi_f: float, psi_p: float, z: float, z_w: float, c: float,
                           phi: float, gamma: float, gamma_w: float, psi_T: float, target: float,
                           anchor: str = "active") -> float:
    """Anchor force per metre run that raises the planar FoS to ``target``.

    Active:  T = (F D0 - cA - N0 tan phi) / (F cos(psi_p + psi_T) + tan phi sin(psi_p + psi_T))
    Passive: T = (F D0 - cA - N0 tan phi) / (cos(psi_p + psi_T) + tan phi sin(psi_p + psi_T))
    Returns 0 if the slope already meets the target and ``inf`` if no anchor at
    this inclination can.  Solved numerically if the effective normal force is
    negative without the anchor (the closed form then does not apply).
    """
    base = planar_fos(H, psi_f, psi_p, z, z_w, c, phi, gamma, gamma_w, 0.0, psi_T, anchor)
    if not base["valid"]:
        return math.nan
    if base["FoS"] >= target:
        return 0.0
    p, f, t = _rad(psi_p), _rad(phi), _rad(psi_p + psi_T)
    N0 = base["W"] * math.cos(p) - base["U"] - base["V"] * math.sin(p)
    D0 = base["W"] * math.sin(p) + base["V"] * math.cos(p)
    numerator = target * D0 - c * base["A"] - N0 * math.tan(f)
    shear_factor = target if anchor != "passive" else 1.0
    denominator = shear_factor * math.cos(t) + math.tan(f) * math.sin(t)
    if denominator > 1e-12:
        T = max(0.0, numerator / denominator)
        if N0 + T * math.sin(t) >= 0.0:
            return T

    def gap(T_try: float) -> float:
        r = planar_fos(H, psi_f, psi_p, z, z_w, c, phi, gamma, gamma_w, T_try, psi_T, anchor)
        return min(r["FoS"], 1e6) - target

    upper = max(D0, 1.0)
    for _ in range(40):
        if gap(upper) >= 0:
            break
        upper *= 2.0
    else:
        return math.inf
    from scipy.optimize import brentq
    return float(brentq(gap, 0.0, upper, xtol=1e-6 * upper))


def optimum_anchor_angle(phi: float, psi_p: float, target: float = 1.0, anchor: str = "active") -> float:
    """Anchor inclination below horizontal psi_T that minimises the anchor force.

    Active: tan(psi_p + psi_T) = tan(phi) / F (psi_T = phi - psi_p at F = 1).
    Passive: psi_p + psi_T = phi.
    """
    tan_f = math.tan(_rad(phi))
    theta = math.degrees(math.atan(tan_f / (target if anchor != "passive" else 1.0)))
    return theta - float(psi_p)


def bolt_rows(T_required: float, bolt_load: float, spacing: float) -> int:
    """Rows of bolts per metre run: ceil(T s_h / T_bolt)."""
    if not math.isfinite(T_required) or T_required <= 0:
        return 0
    return int(math.ceil(T_required * spacing / bolt_load - 1e-12))


# ---------------------------------------------------------------------------
# Wedge sliding (Hoek and Bray 1981, comprehensive solution in vector form)
# ---------------------------------------------------------------------------


def wedge_geometry(plane_a: tuple[float, float], plane_b: tuple[float, float],
                   face: tuple[float, float], upper: tuple[float, float], H: float) -> dict:
    """Tetrahedral wedge from two planes, the face, the upper surface and height H.

    Planes are (dip, dip direction).  ``H`` is the vertical height of the line of
    intersection between its outcrops on the face (point O, at the origin) and
    on the upper surface (point T), as in Hoek and Bray.  Returns the vertices,
    volume, face areas, line of intersection and validity.
    """
    nA, nB = plane_normal(*plane_a), plane_normal(*plane_b)
    nF, nS = plane_normal(*face), plane_normal(*upper)
    line = np.cross(nA, nB)
    if np.linalg.norm(line) < 1e-9:
        return {"valid": False, "reason": "Planes A and B are parallel."}
    trend, plunge = trend_plunge(line)
    d5 = line_vector(trend, plunge)
    out = {"trend": trend, "plunge": plunge, "valid": False}
    face_app = apparent_dip(face[0], face[1], trend)
    upper_app = apparent_dip(upper[0], upper[1], trend)
    out.update(face_apparent=face_app, upper_apparent=upper_app)
    if angular_difference(trend, face[1]) >= 90.0 or plunge >= face_app:
        out["reason"] = "The line of intersection does not daylight in the slope face."
        return out
    if plunge <= upper_app:
        out["reason"] = "The line of intersection does not outcrop on the upper slope surface."
        return out
    if plunge <= 0.0:
        out["reason"] = "The line of intersection is horizontal."
        return out
    O = np.zeros(3)
    T = -(H / math.sin(_rad(plunge))) * d5
    rhs = float(nS @ T)
    try:
        P1 = np.linalg.solve(np.vstack([nA, nF, nS]), np.array([0.0, 0.0, rhs]))
        P2 = np.linalg.solve(np.vstack([nB, nF, nS]), np.array([0.0, 0.0, rhs]))
    except np.linalg.LinAlgError:
        out["reason"] = "A plane is parallel to the face or the upper surface."
        return out
    volume = abs(float(np.linalg.det(np.vstack([T, P1, P2])))) / 6.0
    area_a = 0.5 * float(np.linalg.norm(np.cross(T, P1)))
    area_b = 0.5 * float(np.linalg.norm(np.cross(T, P2)))
    if volume <= 0.0 or min(P1[2], P2[2]) < -1e-9:
        out["reason"] = "The planes do not form a wedge with this face and upper surface."
        return out
    # Normals pointing into the wedge (the side holding the vertex off each plane).
    nA_in = nA if float(nA @ P2) > 0 else -nA
    nB_in = nB if float(nB @ P1) > 0 else -nB
    out.update(valid=True, reason="", O=O, T=T, P1=P1, P2=P2, volume=volume,
               area_a=area_a, area_b=area_b, d5=d5, nA=nA_in, nB=nB_in,
               theta_nanb=math.degrees(math.acos(max(-1.0, min(1.0, float(nA @ nB))))))
    return out


def wedge_fos(plane_a: tuple[float, float], plane_b: tuple[float, float],
              face: tuple[float, float], upper: tuple[float, float], H: float,
              c_a: float, phi_a: float, c_b: float, phi_b: float,
              gamma: float, gamma_w: float = 9.81, saturated: bool = False) -> dict:
    """Factor of safety of a tetrahedral wedge (Hoek and Bray), dry or fully saturated.

    Total normal reactions N_A, N_B balance the weight normal to the line of
    intersection; with water, U_A = gamma_w H / 6 * area_A (and likewise for B)
    is subtracted.  FoS = (c_A A_A + c_B A_B + N'_A tan phi_A + N'_B tan phi_B)
    / (W sin psi_i).  If one normal reaction is negative, the wedge leaves that
    plane and slides down the dip of the other plane alone.
    """
    geo = wedge_geometry(plane_a, plane_b, face, upper, H)
    if not geo["valid"]:
        return {"valid": False, "reason": geo["reason"], "geometry": geo}
    W = gamma * geo["volume"]
    k = np.array([0.0, 0.0, 1.0])
    nA, nB = geo["nA"], geo["nB"]
    cos_ab = float(nA @ nB)
    M = np.array([[1.0, cos_ab], [cos_ab, 1.0]])
    NA, NB = np.linalg.solve(M, W * np.array([float(k @ nA), float(k @ nB)]))
    mean_pressure = gamma_w * H / 6.0 if saturated else 0.0
    UA, UB = mean_pressure * geo["area_a"], mean_pressure * geo["area_b"]
    tan_a, tan_b = math.tan(_rad(phi_a)), math.tan(_rad(phi_b))
    plunge = geo["plunge"]
    result = {"valid": True, "geometry": geo, "W": W, "NA": NA, "NB": NB, "UA": UA, "UB": UB,
              "area_a": geo["area_a"], "area_b": geo["area_b"], "plunge": plunge, "trend": geo["trend"]}
    if NA >= 0.0 and NB >= 0.0:
        NA_eff, NB_eff = max(0.0, NA - UA), max(0.0, NB - UB)
        driving = W * math.sin(_rad(plunge))
        resisting = c_a * geo["area_a"] + c_b * geo["area_b"] + NA_eff * tan_a + NB_eff * tan_b
        mode = "Sliding on both planes along the line of intersection"
    elif NA < 0.0 <= NB or (NA < 0.0 and NB < 0.0 and NB >= NA):
        dip_b = plane_b[0]
        NB_eff = max(0.0, W * math.cos(_rad(dip_b)) - UB)
        NA_eff = 0.0
        driving = W * math.sin(_rad(dip_b))
        resisting = c_b * geo["area_b"] + NB_eff * tan_b
        mode = "Contact lost on plane A: sliding on plane B alone"
    else:
        dip_a = plane_a[0]
        NA_eff = max(0.0, W * math.cos(_rad(dip_a)) - UA)
        NB_eff = 0.0
        driving = W * math.sin(_rad(dip_a))
        resisting = c_a * geo["area_a"] + NA_eff * tan_a
        mode = "Contact lost on plane B: sliding on plane A alone"
    result.update(NA_eff=NA_eff, NB_eff=NB_eff, driving=driving, resisting=resisting, mode=mode,
                  FoS=resisting / driving if driving > 0 else math.inf)
    return result


def wedge_factor(plane_a: tuple[float, float], plane_b: tuple[float, float]) -> dict:
    """Wedge factor K = sin(beta) / sin(xi / 2) for a cohesionless dry wedge.

    xi is the included angle of the wedge and beta the inclination of its
    bisector, both measured in the plane normal to the line of intersection;
    FoS = K tan(phi) / tan(psi_i) for equal phi on both planes.
    """
    nA, nB = plane_normal(*plane_a), plane_normal(*plane_b)
    line = np.cross(nA, nB)
    trend, plunge = trend_plunge(line)
    d5 = line_vector(trend, plunge)
    xi = 180.0 - math.degrees(math.acos(max(-1.0, min(1.0, float(nA @ nB)))))
    bisector = nA + nB
    bisector = bisector / np.linalg.norm(bisector)
    k = np.array([0.0, 0.0, 1.0])
    k_perp = k - float(k @ d5) * d5
    k_perp = k_perp / np.linalg.norm(k_perp)
    # The bisector of the vee is parallel to nA + nB; beta is its angle from the
    # horizontal of the section, i.e. 90 deg less its angle from the section's vertical.
    beta = 90.0 - math.degrees(math.acos(max(-1.0, min(1.0, float(bisector @ k_perp)))))
    K =math.sin(_rad(beta)) / math.sin(_rad(xi / 2.0))
    return {"xi": xi, "beta": beta, "K": K, "plunge": plunge, "trend": trend}


# ---------------------------------------------------------------------------
# Circular sliding: ordinary method of slices and Bishop's simplified method
# ---------------------------------------------------------------------------


def ground_level(x, H: float, psi_f: float):
    """Slope profile: level ground y = 0 left of the toe (x < 0), face to the crest, level at H."""
    x = np.asarray(x, dtype=float)
    if psi_f >= 89.999:
        return np.where(x > 0.0, H, 0.0)
    return np.clip(x * math.tan(_rad(psi_f)), 0.0, H)


def circle_slices(H: float, psi_f: float, xc: float, yc: float, R: float, n: int = 60,
                  gamma: float = 20.0, ru: float = 0.0, x_min: float | None = 0.0) -> dict | None:
    """Slices of the mass above a circular slip surface (centre xc, yc; radius R).

    The mass lies between the lower arc of the circle and the ground, for
    x >= ``x_min`` (0 = the toe, so a circle through the toe exits there and the
    rest of the circle in front of the toe is not part of the slip surface;
    ``None`` lets the circle cut the ground in front of the toe).  Returns
    ``None`` if the circle does not cut the slope or would need its upper arc.
    Slice widths are equal; base angles are taken at the slice midpoint and
    base lengths are exact arc lengths.
    """
    x_lo, x_hi = xc - R, xc + R
    if x_min is not None:
        x_lo = max(x_lo, float(x_min))
    if x_hi <= x_lo:
        return None
    xs = np.linspace(x_lo, x_hi, 2001)

    def depth(x):
        x = np.asarray(x, dtype=float)
        arc = yc - np.sqrt(np.clip(R ** 2 - (x - xc) ** 2, 0.0, None))
        return ground_level(x, H, psi_f) - arc

    f = depth(xs)
    tol = 1e-9 * max(H, 1.0)
    inside = np.where(f > tol)[0]
    if inside.size < 2:
        return None
    from scipy.optimize import brentq

    def refine(lo: float, hi: float) -> float:
        f_lo, f_hi = float(depth(lo)), float(depth(hi))
        if f_lo * f_hi < 0.0:
            return float(brentq(depth, lo, hi))
        return lo if abs(f_lo) <= abs(f_hi) else hi

    i0, i1 = inside[0], inside[-1]
    x_entry = xs[0] if i0 == 0 else refine(xs[i0 - 1], xs[i0])
    x_exit = xs[-1] if i1 == xs.size - 1 else refine(xs[i1], xs[i1 + 1])
    if np.any(f[i0:i1 + 1] < -1e-6 * max(H, 1.0)):
        return None  # the mass is not one piece
    if float(np.max(ground_level(np.array([x_entry, x_exit]), H, psi_f))) > yc + 1e-6 * H:
        return None  # the upper arc would bound the mass
    edges = np.linspace(x_entry, x_exit, n + 1)
    xm = 0.5 * (edges[:-1] + edges[1:])
    b = np.diff(edges)
    h = np.maximum(depth(xm), 0.0)
    s_edges = np.clip((edges - xc) / R, -1.0, 1.0)
    a_edges = np.arcsin(s_edges)
    alpha = np.arcsin(np.clip((xm - xc) / R, -1.0, 1.0))
    length = R * np.diff(a_edges)
    W = gamma * b * h
    u = ru * gamma * h
    return {"x": xm, "b": b, "h": h, "alpha": alpha, "l": length, "W": W, "u": u,
            "x_entry": float(x_entry), "x_exit": float(x_exit), "xc": xc, "yc": yc, "R": R,
            "area": float(np.sum(b * h))}


def fellenius_fos(slices: dict, c: float, phi: float) -> float:
    """Ordinary method of slices: F = sum[c l + (W cos a - u l) tan phi] / sum W sin a.

    As in the textbook method, a negative effective normal force on a steep
    slice is kept (not set to zero); the method is conservative with high r_u.
    """
    a = slices["alpha"]
    driving = float(np.sum(slices["W"] * np.sin(a)))
    if driving <= 0:
        return math.inf
    normal = slices["W"] * np.cos(a) - slices["u"] * slices["l"]
    return float(np.sum(c * slices["l"] + normal * math.tan(_rad(phi)))) / driving


def bishop_fos(slices: dict, c: float, phi: float, tol: float = 1e-7, max_iter: int = 200) -> float:
    """Bishop's simplified method: F = sum[(c b + (W - u b) tan phi) / m_a] / sum W sin a,
    m_a = cos a + sin a tan phi / F, solved by iteration.

    The cohesion term uses b = l cos a with the exact arc length l, which is the
    same in the limit of thin slices and stays accurate on the steep last slice.
    """
    a = slices["alpha"]
    driving = float(np.sum(slices["W"] * np.sin(a)))
    if driving <= 0:
        return math.inf
    tan_phi = math.tan(_rad(phi))
    numer = (c * slices["l"] * np.cos(a)
             + np.maximum(slices["W"] - slices["u"] * slices["b"], 0.0) * tan_phi)
    F = max(fellenius_fos(slices, c, phi), 0.1)
    if not math.isfinite(F):
        F = 1.0
    for _ in range(max_iter):
        m_alpha = np.cos(a) + np.sin(a) * tan_phi / F
        m_alpha = np.maximum(m_alpha, 1e-3)
        F_new = float(np.sum(numer / m_alpha)) / driving
        if abs(F_new - F) < tol * max(1.0, F):
            return F_new
        F = F_new
    return F


def toe_circle_search(H: float, psi_f: float, c: float, phi: float, gamma: float, ru: float = 0.0,
                      n_slices: int = 40, grid: int = 25, method: str = "bishop",
                      min_area_ratio: float = 0.005) -> dict:
    """Critical circle through the toe over a grid of centres, refined once locally.

    Centres span -H <= xc <= crest + 1.5 H and 0.5 H <= yc <= 3 H; a circle
    through the toe has R = sqrt(xc^2 + yc^2) and the slip surface runs from
    the toe to where the circle meets the ground.  Masses smaller than
    ``min_area_ratio`` H^2 are skipped.
    """
    solve = bishop_fos if method == "bishop" else fellenius_fos
    x_crest = 0.0 if psi_f >= 89.999 else H / math.tan(_rad(psi_f))

    def scan(xcs, ycs):
        best = (math.inf, None, None)
        table = np.full((ycs.size, xcs.size), np.nan)
        for j, yc in enumerate(ycs):
            for i, xc in enumerate(xcs):
                R = math.hypot(xc, yc)
                s = circle_slices(H, psi_f, xc, yc, R, n_slices, gamma, ru)
                if s is None or s["area"] < min_area_ratio * H ** 2:
                    continue
                F = solve(s, c, phi)
                table[j, i] = F
                if F < best[0]:
                    best = (F, xc, yc)
        return best, table

    xcs = np.linspace(-H, x_crest + 1.5 * H, grid)
    ycs = np.linspace(0.5 * H, 3.0 * H, grid)
    (F, xb, yb), table = scan(xcs, ycs)
    if xb is None:
        return {"valid": False, "reason": "No circle through the toe cuts the slope."}
    dx, dy = xcs[1] - xcs[0], ycs[1] - ycs[0]
    fine_x = np.linspace(xb - dx, xb + dx, 11)
    fine_y = np.linspace(max(0.05 * H, yb - dy), yb + dy, 11)
    (F2, xb2, yb2), _ = scan(fine_x, fine_y)
    if xb2 is not None and F2 < F:
        F, xb, yb = F2, xb2, yb2
    return {"valid": True, "FoS": F, "xc": xb, "yc": yb, "R": math.hypot(xb, yb),
            "grid_x": xcs, "grid_y": ycs, "grid_fos": table, "method": method}


__all__ = [name for name in dir() if not name.startswith("_")]
