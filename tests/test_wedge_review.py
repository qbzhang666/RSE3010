"""Regression tests for the saturated-wedge finding of the independent review
(November 2026). Reference values from an independent vector solution following
the Rocscience SWedge theory manual (tetrahedral wedges), with Hoek and Bray's
water distribution (mean pressure gamma_w H / 6 on each plane).

    python -m pytest tests/test_wedge_review.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geo_platform import slopes  # noqa: E402

HOEK_BRAY = ((45, 105), (70, 235), (65, 185), (12, 195), 40.0, 24, 20, 48, 30)


@pytest.mark.parametrize("args,fos,mode", [
    # water drives N'_B negative: the wedge slides on A alone (the earlier code gave 1.737 on both planes)
    (((30, 110), (70, 190), (65, 180), (10, 180), 30.0, 20, 30, 20, 30, 26), 0.187, "plane A alone"),
    # the water on B, which has lost contact, still acts (the earlier code ignored it and gave 0.870)
    (((30, 120), (70, 190), (65, 180), (10, 180), 30.0, 20, 30, 20, 30, 26), 0.155, "plane A alone"),
    # the worked geometry with a light rock (gamma 15, allowed by the page; earlier 1.387)
    ((*HOEK_BRAY, 15), 0.567, "plane A alone"),
    # the water forces exceed the weight on both planes: the wedge lifts (earlier 2.999)
    (((61, 146), (72, 207), (65, 180), (10, 180), 20.0, 10, 30, 10, 30, 25), 0.0, "lift"),
    # the worked example itself is unchanged: both planes, 1.214 saturated
    ((*HOEK_BRAY, 25), 1.214, "both planes"),
])
def test_saturated_wedge_modes(args, fos, mode):
    r = slopes.wedge_fos(*args, gamma_w=9.81, saturated=True)
    assert r["FoS"] == pytest.approx(fos, abs=1e-3)
    assert mode in r["mode"]


def test_dry_worked_example_unchanged():
    r = slopes.wedge_fos(*HOEK_BRAY, 25, saturated=False)
    assert r["FoS"] == pytest.approx(1.845, abs=1e-3)
    assert "both planes" in r["mode"]
