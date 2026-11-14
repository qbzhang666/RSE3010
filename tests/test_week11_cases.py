"""Kinematic screening against the worked cases of the RSE3010 Week 11 lecture
(Stereonet and Kinematic, 2026). Independent of tests/test_slopes.py.

The intersections are checked against the exact geometry: two planes of equal
dip intersect on the bisector of their dip directions, with
tan(plunge) = tan(dip) cos(half the angle between the dip directions). The
slides first gave 58/114 (slides 67-71) and 42/150 (slide 64), read from the
stereonet; they were corrected to 58/110 and 41/150 in November 2026.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geo_platform import slopes  # noqa: E402


@pytest.mark.parametrize("phi,dip,dd,planar,topple", [
    # slide 54-56: slope 55/220
    (35, 40, 225, True, False),     # joint 1 daylights and dips steeper than phi
    (35, 65, 225, False, False),    # joint 2 too steep to daylight
    (35, 80, 45, False, True),      # joint 3 dips into the face: toppling
    (50, 40, 225, False, False),    # dry friction 50 deg: no planar failure
])
def test_slope_55_220(phi, dip, dd, planar, topple):
    assert slopes.planar_kinematics(dip, dd, 55, 220, phi, 20)["Possible"] is planar
    assert slopes.toppling_kinematics(dip, dd, 55, 220, phi, 10)["Possible"] is topple


@pytest.mark.parametrize("dip,dd,planar,topple", [
    # slides 58-59: slope 60/090, phi 35
    (40, 90, True, False), (80, 90, False, False), (80, 270, False, True),
])
def test_slope_60_090(dip, dd, planar, topple):
    assert slopes.planar_kinematics(dip, dd, 60, 90, 35, 20)["Possible"] is planar
    assert slopes.toppling_kinematics(dip, dd, 60, 90, 35, 10)["Possible"] is topple


@pytest.mark.parametrize("b,plunge,trend,possible,single", [
    # slides 62-71: slope 70/135, phi 35, joint A 60/090
    ((60, 210), 40.9, 150.0, True, False),    # wedge on both planes
    ((40, 210), 29.0, 161.0, False, False),   # intersection flatter than phi
    ((60, 130), 58.4, 110.0, True, True),     # joint B alone: planar on B (Hocking)
])
def test_wedges_on_slope_70_135(b, plunge, trend, possible, single):
    w = slopes.wedge_kinematics((60, 90), b, 70, 135, 35)
    assert w["Plunge (deg)"] == pytest.approx(plunge, abs=0.5)
    assert w["Trend (deg)"] == pytest.approx(trend, abs=1.0)
    assert w["Possible"] is possible
    assert ("alone" in w["Sliding on"]) is single


def test_equal_dip_intersection_closed_form():
    w = slopes.wedge_kinematics((60, 90), (60, 130), 70, 135, 35)
    assert w["Plunge (deg)"] == pytest.approx(math.degrees(math.atan(math.tan(math.radians(60)) * math.cos(math.radians(20)))), abs=1e-6)
    assert w["Trend (deg)"] == pytest.approx(110.0, abs=1e-6)
