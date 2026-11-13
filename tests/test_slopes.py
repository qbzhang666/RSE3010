"""Tests of geo_platform.slopes against hand calculations, and smoke tests of the slope pages.

Every numeric test states the hand calculation it checks.
"""

from __future__ import annotations

import math

import pytest

from geo_platform import slopes

# ---------------------------------------------------------------------------
# Orientations and projection
# ---------------------------------------------------------------------------


def test_intersection_of_symmetric_planes():
    # Planes 50/135 and 50/225 are mirror images about the north-south vertical
    # plane, so their line of intersection trends 180.  Its plunge follows from
    # the apparent dip of either plane at 45 deg from its dip direction:
    # tan(psi_i) = tan 50 cos 45 = 1.19175 x 0.70711 = 0.84270 -> 40.12 deg.
    trend, plunge = slopes.intersection_line(50, 135, 50, 225)
    assert trend == pytest.approx(180.0, abs=1e-9)
    assert plunge == pytest.approx(math.degrees(math.atan(math.tan(math.radians(50)) * math.cos(math.radians(45)))),
                                   abs=1e-9)
    assert plunge == pytest.approx(40.12, abs=0.01)


def test_apparent_dip_and_angular_difference():
    # tan(apparent) = tan 60 cos 30 = 1.73205 x 0.86603 = 1.5 -> 56.31 deg.
    assert slopes.apparent_dip(60, 170, 200) == pytest.approx(math.degrees(math.atan(1.5)), abs=1e-9)
    # 350 and 10 are 20 deg apart across north.
    assert slopes.angular_difference(350, 10) == pytest.approx(20.0)


def test_projection_radii():
    # Equal-angle: r = tan((90 - p)/2); a 45 deg plunge -> tan 22.5 = 0.41421.
    # Equal-area: r = sqrt(2) sin((90 - p)/2); 45 deg -> 1.41421 x 0.38268 = 0.54120.
    x, y = slopes.project(90.0, 45.0, slopes.EQUAL_ANGLE)
    assert (float(x), float(y)) == pytest.approx((0.41421, 0.0), abs=1e-5)
    x, y = slopes.project(90.0, 45.0, slopes.EQUAL_AREA)
    assert float(x) == pytest.approx(0.54120, abs=1e-5)
    trend, plunge = slopes.unproject(0.0, -0.41421356, slopes.EQUAL_ANGLE)
    assert (float(trend), float(plunge)) == pytest.approx((180.0, 45.0), abs=1e-6)


# ---------------------------------------------------------------------------
# SMR (Romana)
# ---------------------------------------------------------------------------


def test_smr_planar_course_joint():
    # Joint 40/160 on face 60/170, RMRb 50, deficient blasting F4 = -8.
    # A = |160 - 170| = 10 -> F1 = 0.85 (the 10-5 deg band; 0.70 needs A > 10);
    # B = 40 -> F2 = 0.85 (35-45); C = 40 - 60 = -20 < -10 -> F3 = -60.
    # SMR = 50 + 0.85 x 0.85 x (-60) - 8 = 50 - 43.35 - 8 = -1.35 -> class V.
    r = slopes.smr_planar(40, 160, 60, 170, 50, -8)
    assert (r["F1"], r["F2"], r["F3"]) == (0.85, 0.85, -60.0)
    assert r["SMR"] == pytest.approx(-1.35)
    assert r["Class"] == "V"


def test_smr_toppling_and_class_boundaries():
    # Joint 80/340 on face 60/170: A = |340 - 170 - 180| = 10 -> F1 = 0.85;
    # F2 = 1; C = 80 + 60 = 140 > 120 -> F3 = -25.
    # SMR = 50 + 0.85 x 1 x (-25) + 0 = 28.75 -> class IV.
    r = slopes.smr_toppling(80, 340, 60, 170, 50, 0)
    assert r["SMR"] == pytest.approx(28.75)
    assert r["Class"] == "IV"
    # Romana's ranges 61-80 / 81-100 read as > 80 for class I.
    assert slopes.smr_class(80.5)["Class"] == "I"
    assert slopes.smr_class(80.0)["Class"] == "II"


def test_smr_f3_unfavourable_band_is_ten_degrees():
    # Romana: C from 0 to -10 deg is "unfavourable" (-50); below -10 deg -60.
    # (One course app used -5 deg as the bound.)
    assert slopes.smr_f3(-7.0, "planar") == -50.0
    assert slopes.smr_f3(-10.0, "planar") == -50.0
    assert slopes.smr_f3(-10.5, "planar") == -60.0
    assert slopes.smr_f3(0.0, "wedge") == -25.0
    assert slopes.smr_f3(5.0, "planar") == -6.0
    assert slopes.smr_f3(115.0, "toppling") == -6.0


def test_smr_continuous_factors():
    # F1 = (1 - sin 20)^2 = (1 - 0.34202)^2 = 0.43294; F2 = tan^2 30 = 0.33333;
    # far from the face (A = 60) the fit gives 0.018, bounded up to 0.15.
    assert slopes.smr_f1(20, continuous=True) == pytest.approx(0.43294, abs=1e-5)
    assert slopes.smr_f2(30, "planar", continuous=True) == pytest.approx(1 / 3, abs=1e-6)
    assert slopes.smr_f1(60, continuous=True) == 0.15
    assert slopes.smr_f2(60, "planar", continuous=True) == 1.0


def test_smr_table_counts_modes():
    # Three joints and one face: 3 planar + 3 wedge pairs + 3 toppling rows.
    rows = slopes.smr_table([("J1", 40, 160), ("J2", 55, 110), ("J3", 80, 340)], [("S1", 60, 170)], 50, -8)
    modes = [r["Mode"] for r in rows]
    assert modes.count("Planar") == 3 and modes.count("Wedge") == 3 and modes.count("Toppling") == 3


# ---------------------------------------------------------------------------
# Kinematic screening
# ---------------------------------------------------------------------------


def test_planar_and_toppling_kinematics():
    # J1 40/160 on face 60/170, phi 30: 30 < 40 < apparent face dip in 160
    # (tan^-1(tan 60 cos 10) = 59.6) and |160 - 170| = 10 <= 20 -> possible.
    r = slopes.planar_kinematics(40, 160, 60, 170, 30, 20)
    assert r["Possible"]
    assert r["Face apparent dip (deg)"] == pytest.approx(math.degrees(math.atan(math.tan(math.radians(60))
                                                                                * math.cos(math.radians(10)))))
    # Same joint with phi 45: not steeper than phi.
    assert not slopes.planar_kinematics(40, 160, 60, 170, 45, 20)["Possible"]
    # J3 80/340 dips into the face (offset |340 - 350| = 10 <= 10); Goodman:
    # 80 >= 90 - 60 + 30 = 60 -> flexural toppling possible.
    t = slopes.toppling_kinematics(80, 340, 60, 170, 30, 10)
    assert t["Required dip (deg)"] == pytest.approx(60.0)
    assert t["Possible"]


def test_wedge_kinematics():
    # Planes 50/135 and 50/225 meet in a line 40.1/180 that daylights in a
    # 70/180 face; with phi 35 it slides on both planes, with phi 45 it does not.
    r = slopes.wedge_kinematics((50, 135), (50, 225), 70, 180, 35)
    assert r["Possible"] and r["Sliding on"] == "both planes"
    assert not slopes.wedge_kinematics((50, 135), (50, 225), 70, 180, 45)["Possible"]


# ---------------------------------------------------------------------------
# Planar sliding (Hoek and Bray)
# ---------------------------------------------------------------------------

PLANAR = dict(H=60.0, psi_f=60.0, psi_p=35.0, z=14.0, z_w=5.0, c=25.0, phi=37.0, gamma=26.0, gamma_w=9.81)


def test_planar_fos_hand_calculation():
    # A = (60 - 14) / sin 35 = 46 / 0.573576 = 80.198 m
    # W = 0.5 x 26 x 60^2 [(1 - (14/60)^2) cot 35 - cot 60]
    #   = 46 800 [0.945556 x 1.428148 - 0.577350] = 46 800 x 0.773043 = 36 178 kN/m
    # V = 0.5 x 9.81 x 5^2 = 122.63;  U = 0.5 x 9.81 x 5 x 80.198 = 1 966.9
    # N' = 36 178 x 0.819152 - 1 966.9 - 122.63 x 0.573576 = 27 598
    # FoS = (25 x 80.198 + 27 598 x tan 37) / (36 178 x 0.573576 + 122.63 x 0.819152)
    #     = (2 005.0 + 20 797) / (20 751 + 100.5) = 22 802 / 20 852 = 1.0935
    r = slopes.planar_fos(**PLANAR)
    assert r["A"] == pytest.approx(80.198, abs=1e-3)
    assert r["W"] == pytest.approx(36178, rel=1e-4)
    assert r["U"] == pytest.approx(1966.9, rel=1e-4)
    assert r["FoS"] == pytest.approx(1.0935, abs=5e-4)
    assert not r["crack_in_face"]


def test_planar_required_anchor_hand_calculation():
    # Target 1.3 with an active anchor at psi_T = 15, so psi_p + psi_T = 50:
    # T = (1.3 x 20 852 - 2 005.0 - 27 598 x 0.753554)
    #     / (1.3 cos 50 + tan 37 sin 50)
    #   = (27 107 - 22 802) / (0.835624 + 0.577256) = 4 305 / 1.41288 = 3 047 kN/m
    T = slopes.planar_required_anchor(PLANAR["H"], PLANAR["psi_f"], PLANAR["psi_p"], PLANAR["z"], PLANAR["z_w"],
                                      PLANAR["c"], PLANAR["phi"], PLANAR["gamma"], PLANAR["gamma_w"], 15.0, 1.3)
    assert T == pytest.approx(3047, rel=2e-3)
    r = slopes.planar_fos(**PLANAR, T=T, psi_T=15.0)
    assert r["FoS"] == pytest.approx(1.3, abs=1e-6)
    # Optimum (active, F = 1): psi_T = phi - psi_p = 37 - 35 = 2 deg.
    assert slopes.optimum_anchor_angle(37, 35, 1.0) == pytest.approx(2.0)
    # Rows: ceil(3 047 x 2 / 250) = ceil(24.4) = 25.
    assert slopes.bolt_rows(T, 250.0, 2.0) == 25


def test_tension_crack_in_face_weight():
    # H 10, psi_f 60, psi_p 30: the crack is behind the crest only if
    # z <= 10 (1 - cot 60 tan 30) = 10 (1 - 1/3) = 6.667 m.  With z = 8 the crack
    # is in the face: its base is at (2 cot 30, 2) = (3.464, 2) and its top on
    # the face at y = 3.464 tan 60 = 6, so the block is the triangle (0, 0),
    # (3.464, 6), (3.464, 2) of area 0.5 x 3.464 x 4 = 6.928 m^2/m, which is
    # Hoek and Bray's 0.5 H^2 (1 - z/H)^2 cot psi_p (cot psi_p tan psi_f - 1).
    g = slopes.planar_geometry(10, 60, 30, 8)
    assert g["crack_in_face"]
    assert g["area"] == pytest.approx(0.5 * 3.4641016 * 4.0, rel=1e-6)
    assert g["crack_length"] == pytest.approx(4.0)
    assert slopes.crack_behind_crest_limit(10, 60, 30) == pytest.approx(20 / 3)
    # The two weight expressions agree at the transition depth.
    z_t = slopes.crack_behind_crest_limit(10, 60, 30)
    a1 = slopes.planar_geometry(10, 60, 30, z_t - 1e-9)["area"]
    a2 = slopes.planar_geometry(10, 60, 30, z_t + 1e-9)["area"]
    assert a1 == pytest.approx(a2, rel=1e-6)


def test_critical_tension_crack():
    # H 15, psi_f 60, psi_p 40: z/H = 1 - sqrt(cot 60 tan 40) = 1 - sqrt(0.57735 x 0.83910)
    # = 1 - 0.69603 = 0.30397 -> z = 4.560 m;  b/H = sqrt(cot 60 cot 40) - cot 60
    # = sqrt(0.57735 x 1.19175) - 0.57735 = 0.82949 - 0.57735 = 0.25214 -> b = 3.782 m.
    z, b = slopes.critical_tension_crack(15, 60, 40)
    assert z == pytest.approx(4.560, abs=2e-3)
    assert b == pytest.approx(3.782, abs=2e-3)


# ---------------------------------------------------------------------------
# Wedge sliding
# ---------------------------------------------------------------------------


def test_symmetric_wedge_against_wedge_factor():
    # Planes 50/135 and 50/225, face 70/180, horizontal top, c = 0, phi = 35, dry.
    # psi_i = 40.12 deg (tan psi_i = 0.84270).  The upward normals make
    # cos(theta) = cos^2 50 = 0.413176 -> theta = 65.595, so the included angle
    # xi = 180 - 65.595 = 114.405 and the bisector is vertical (beta = 90):
    # K = sin 90 / sin 57.2027 = 1 / 0.84058 = 1.18964;
    # FoS = K tan 35 / tan psi_i = 1.18964 x 0.700208 / 0.842701 = 0.98848.
    r = slopes.wedge_fos((50, 135), (50, 225), (70, 180), (0, 180), 20.0, 0, 35, 0, 35, 25.0)
    assert r["mode"].startswith("Sliding on both")
    assert r["FoS"] == pytest.approx(0.98848, abs=2e-4)
    k = slopes.wedge_factor((50, 135), (50, 225))
    assert k["K"] == pytest.approx(1.18964, abs=2e-4)
    assert k["beta"] == pytest.approx(90.0, abs=1e-6)


def test_hoek_bray_wedge_example():
    # Hoek and Bray's worked wedge: A 45/105, B 70/235, face 65/185, upper
    # 12/195, H 40 m, c_A 24 kPa, phi_A 20, c_B 48 kPa, phi_B 30, gamma 25,
    # gamma_w 9.81.  With the angles read from the stereonet (psi_5 = 31.2,
    # theta_na.nb = 101, theta_24 = 65, theta_45 = 25, theta_2.na = 50,
    # theta_13 = 62, theta_35 = 31, theta_1.nb = 60) the comprehensive equations give
    #   X = sin 65 / (sin 25 cos 50) = 0.90631 / (0.42262 x 0.64279) = 3.3363
    #   Y = sin 62 / (sin 31 cos 60) = 0.88295 / (0.51504 x 0.5)    = 3.4287
    #   A = (cos 45 - cos 70 cos 101) / (sin 31.2 sin^2 101) = 0.77236 / 0.49914 = 1.5474
    #   B = (cos 70 - cos 45 cos 101) / (sin 31.2 sin^2 101) = 0.47693 / 0.49914 = 0.9555
    #   F_dry = 3/(25 x 40) (24 X + 48 Y) + A tan 20 + B tan 30
    #         = 0.003 x 244.65 + 0.56321 + 0.55165 = 1.849
    #   F_sat = 0.73395 + (A - 0.1962 X) tan 20 + (B - 0.1962 Y) tan 30
    #         = 0.73395 + 0.32499 + 0.16326 = 1.222
    # The rounded stereonet angles carry about 1 % uncertainty.
    args = ((45, 105), (70, 235), (65, 185), (12, 195), 40.0, 24, 20, 48, 30, 25.0, 9.81)
    dry = slopes.wedge_fos(*args, saturated=False)
    wet = slopes.wedge_fos(*args, saturated=True)
    assert dry["plunge"] == pytest.approx(31.2, abs=0.1)
    assert dry["geometry"]["theta_nanb"] == pytest.approx(101.0, abs=0.5)
    assert dry["FoS"] == pytest.approx(1.849, rel=0.01)
    assert wet["FoS"] == pytest.approx(1.222, rel=0.015)


def test_wedge_rejects_non_daylighting_line():
    # Planes dipping into the face: their intersection plunges away from a 60/170 face.
    r = slopes.wedge_fos((50, 315), (50, 45), (60, 170), (0, 170), 20.0, 0, 30, 0, 30, 25.0)
    assert not r["valid"]


# ---------------------------------------------------------------------------
# Circular sliding
# ---------------------------------------------------------------------------


def _quarter_circle(ru=0.0, n=400):
    # Vertical face of height H = 10, circle centred at the crest (0, 10) with
    # R = 10 through the toe: the sliding mass is a quarter disc, the slip
    # surface leaves the ground at (10, 10), and slice height h = sqrt(R^2 - x^2).
    return slopes.circle_slices(10.0, 90.0, 0.0, 10.0, 10.0, n, 20.0, ru)


def test_circle_phi_zero_moment_hand_calculation():
    # phi = 0: F = c R (arc length) / (W x lever arm).  Quarter disc: arc = pi R / 2,
    # W = gamma pi R^2 / 4 at x = 4R / (3 pi), so W x = gamma R^3 / 3 and
    # F = c R (pi R / 2) / (gamma R^3 / 3) = 3 pi c / (2 gamma R) = 3 pi 20 / (2 x 20 x 10) = 0.47124.
    s = _quarter_circle()
    assert s["area"] == pytest.approx(math.pi * 100 / 4, rel=2e-3)
    assert slopes.fellenius_fos(s, 20.0, 0.0) == pytest.approx(0.47124, rel=1e-3)
    assert slopes.bishop_fos(s, 20.0, 0.0) == pytest.approx(0.47124, rel=1e-3)


def test_circle_ordinary_method_hand_calculation():
    # c = 0: with sin(a) = x / R, cos(a) = h / R and h = sqrt(R^2 - x^2),
    #   sum W cos a -> gamma int (R^2 - x^2)/R dx = 2 gamma R^2 / 3
    #   sum W sin a -> gamma int x sqrt(R^2 - x^2)/R dx = gamma R^2 / 3
    #   sum u l     -> r_u gamma int h (R / h) dx = r_u gamma R^2
    # so F = tan(phi) (2/3 - r_u) / (1/3) = tan(phi) (2 - 3 r_u).
    # phi 30, r_u 0: 2 tan 30 = 1.15470; r_u 0.2: 1.4 tan 30 = 0.80829.
    assert slopes.fellenius_fos(_quarter_circle(0.0), 0.0, 30.0) == pytest.approx(1.15470, rel=2e-3)
    assert slopes.fellenius_fos(_quarter_circle(0.2), 0.0, 30.0) == pytest.approx(0.80829, rel=2e-3)


def test_bishop_not_below_ordinary_method():
    # Bishop's method is usually above the ordinary method of slices for phi > 0.
    s = _quarter_circle(0.2, 60)
    assert slopes.bishop_fos(s, 10.0, 30.0) >= slopes.fellenius_fos(s, 10.0, 30.0)


@pytest.mark.parametrize("beta, m", [(60.0, 0.191), (90.0, 0.261)])
def test_toe_circle_search_matches_taylor(beta, m):
    # Taylor (1937) stability numbers for phi = 0 and critical toe circles:
    # m = c / (F gamma H) = 0.191 at 60 deg and 0.261 at 90 deg, so with c = 20,
    # gamma = 20, H = 10: F = 20 / (m x 200) = 0.5236 and 0.3831.
    res = slopes.toe_circle_search(10.0, beta, 20.0, 0.0, 20.0)
    assert res["valid"]
    assert res["FoS"] == pytest.approx(20.0 / (m * 200.0), rel=0.015)


# ---------------------------------------------------------------------------
# Smoke tests of the pages
# ---------------------------------------------------------------------------


def _smr_app():
    from geo_platform.slope_pages import render_smr
    render_smr({})


def _kinematic_app():
    from geo_platform.slope_pages import render_kinematic
    render_kinematic({})


def _lem_app():
    from geo_platform.slope_pages import render_slope_lem
    render_slope_lem({})


@pytest.mark.parametrize("app", [_smr_app, _kinematic_app, _lem_app])
def test_page_runs(app):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(app, default_timeout=120)
    at.run()
    assert not at.exception


def test_lem_page_circular_mode_runs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_lem_app, default_timeout=120)
    at.run()
    at.radio(key="lem_mode").set_value("Circular sliding").run()
    assert not at.exception
