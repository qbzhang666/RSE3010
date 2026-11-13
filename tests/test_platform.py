"""Tests of the platform: calculations against the book's worked values, and
every page drawn without an exception.

    python -m pytest tests -q
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from geo_platform import ccm, ground, lining, rock, settlement  # noqa: E402

# ---------------------------------------------------------------------------
# Chapter 11: intact rock
# ---------------------------------------------------------------------------


def test_marble_example():
    # sigma_1 = 20 + 125 (9 x 20/125 + 1)^0.5 = 215 MPa; sigma_t = 62.5 (9 - 9.22) = -13.7 MPa
    assert float(rock.hb_sigma1(20, 125, 9)) == pytest.approx(215.3, abs=0.1)
    assert rock.hb_intact_tension(125, 9) == pytest.approx(-13.72, abs=0.01)
    assert rock.hoek_martin_tension(125, 9) == pytest.approx(-8.38, abs=0.01)


def test_sandstone_intact_fits():
    ins = rock.insitu_stresses(450, 1.8, 27)
    assert (ins.sigma_v, ins.sigma_h) == pytest.approx((12.15, 21.87))
    s3, s1 = rock.hb_points(110, 17, 1.0, 0.5, ins.sigma3_fit_max, 6)
    assert s1[1] == pytest.approx(146.78, abs=0.01)
    m1, m2 = rock.fit_mohr_circles(s3, s1), rock.fit_principal_line(s3, s1)
    assert (m1.phi_deg, m1.c) == pytest.approx((47.1, 22.76), abs=0.05)
    assert (m2.phi_deg, m2.c) == pytest.approx((47.0, 22.93), abs=0.05)


# ---------------------------------------------------------------------------
# Chapter 12: rock mass
# ---------------------------------------------------------------------------


def test_sandstone_rock_mass():
    hb = rock.hb_constants(63, 17, 0.0)
    assert (hb.mb, hb.s, hb.a) == pytest.approx((4.535, 0.01639, 0.502), abs=5e-4)
    assert rock.rock_mass_ucs(110, hb) == pytest.approx(13.95, abs=0.01)
    s3, s1 = rock.hb_points(110, hb.mb, hb.s, hb.a, 21.87, 6)
    m1, m2 = rock.fit_mohr_circles(s3, s1), rock.fit_principal_line(s3, s1)
    assert (m1.phi_deg, m1.c) == pytest.approx((42.3, 5.2), abs=0.05)
    assert (m2.phi_deg, m2.c) == pytest.approx((41.7, 5.6), abs=0.05)
    moduli = (rock.modulus_hd_generalised(30, 63, 0), rock.modulus_hd_simplified(63, 0),
              rock.modulus_hoek_2002(110, 63, 0), rock.modulus_bieniawski(60), rock.modulus_serafim_pereira(60))
    assert moduli == pytest.approx((17.6, 25.1, 21.1, 20.0, 17.8), abs=0.05)


# ---------------------------------------------------------------------------
# Chapter 16: convergence-confinement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("em,phi,c,umax,us0,peq,fos", [
    (13.52, 31.5, 2.85, 9.97, 6.2, 1.47, 4.3),
    (2.12, 23.8, 1.72, 126.9, 62.0, 2.27, 2.8),
])
def test_deep_tunnel_worked_example(em, phi, c, umax, us0, peq, fos):
    g = ccm.Ground(16.9, 4.75, 0.22, em, phi, c)
    res = ccm.analyse(g, [ccm.support_ring(4.75, 0.5, 60, 40, 0.25)])
    assert res.u_max == pytest.approx(umax, abs=0.05)
    assert res.us0 == pytest.approx(us0, abs=0.05)
    assert res.eq.p_eq == pytest.approx(peq, abs=0.005)
    assert res.eq.fos == pytest.approx(fos, abs=0.05)


def test_ring_support_properties():
    ring = ccm.support_ring(4.75, 0.5, 60, 40, 0.25)
    assert (ring.psm, ring.ks, ring.usm) == pytest.approx((6.32, 945.5, 6.68), abs=0.05)


def test_motorway_case_simplified_expressions():
    c = ground.motorway_cohesion()
    g = ccm.Ground(1.62, 6.0, 0.3, 0.353, 25.64, c)
    assert g.scm_mc == pytest.approx(0.47, abs=1e-6)
    assert ccm.plastic_radius(g, 0.0) == pytest.approx(13.77, abs=0.01)
    assert ccm.grc_displacement(g, 0.0) == pytest.approx(97.1, abs=0.1)
    # Panet profile 3 m behind the face of a 6 m radius tunnel: u* = 0.73, the 86 mm of the course example
    assert ccm.ldp(0.5, 1.0, ccm.LDP_METHODS[1]) == pytest.approx(0.73, abs=1e-6)


def test_duncan_fama_equals_salencon_at_half():
    g = ccm.Ground(16.9, 4.75, 0.5, 2.12, 23.8, 1.72)
    assert ccm.grc_displacement(g, 0.0, ccm.GRC_METHODS[1]) == pytest.approx(
        ccm.grc_displacement(g, 0.0, ccm.GRC_METHODS[0]), rel=1e-12)


def test_combined_support_capacity_is_not_the_sum():
    a = 4.75
    ring = ccm.support_ring(a, 0.1, 30, 25, 0.2)
    bolts = ccm.support_bolts(a, "25 mm grouted rebar", 1.5)
    both = ccm.combine([ring, bolts])
    assert both.ks == pytest.approx(ring.ks + bolts.ks)
    assert both.usm == pytest.approx(min(ring.usm, bolts.usm))
    assert both.psm < ring.psm + bolts.psm


# ---------------------------------------------------------------------------
# Chapter 17: ground movement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("z0,k,smax", [(5, 0.5, 45.1), (20, 0.25, 22.6), (20, 0.5, 11.3)])
def test_trough_examples(z0, k, smax):
    assert settlement.Bore(0, z0, 6, 1.0, k).smax_mm == pytest.approx(smax, abs=0.05)


def test_longitudinal_profile_is_cumulative():
    i = 10.0
    assert float(settlement.longitudinal_factor(0.0, 0.0, i)) == pytest.approx(0.5)
    assert float(settlement.longitudinal_factor(-40.0, 0.0, i)) > 0.999   # well behind the face
    assert float(settlement.longitudinal_factor(40.0, 0.0, i)) < 0.001    # well ahead of it


def test_horizontal_strain_sign():
    b = settlement.Bore(0, 20, 6, 1.0, 0.5)
    _, eps = settlement.horizontal([b], np.array([0.0, 2 * b.i]))
    assert eps[0] < 0 < eps[1]      # compressive over the centre, tensile beyond i


# ---------------------------------------------------------------------------
# Chapter 19: segment as a curved member
# ---------------------------------------------------------------------------


def test_semicircle_crown_moment():
    seg = lining.Segment(3.0, 180, 1.0, 0.3, 0.0, 10.0, 35)
    act = lining.segment_actions(seg)
    mid = len(act["M"]) // 2
    assert act["M"][mid] == pytest.approx(10 * 9 * (math.pi / 2 - 1), rel=1e-9)
    assert act["M"][0] == pytest.approx(0, abs=1e-9) and act["M"][-1] == pytest.approx(0, abs=1e-9)


def test_flat_arc_tends_to_straight_beam():
    span, q = 5.0, 10.0
    radius = 2000.0
    seg = lining.Segment(radius, math.degrees(2 * math.asin(span / 2 / radius)), 1.0, 0.3, 0.0, q, 35)
    act = lining.segment_actions(seg)
    mid = len(act["M"]) // 2
    assert act["M"][mid] == pytest.approx(q * span ** 2 / 8, rel=1e-4)
    ei = 35e6 * 0.3 ** 3 / 12
    assert lining.crown_deflection_mm(seg, act) == pytest.approx(5 * q * span ** 4 / (384 * ei) * 1000, rel=1e-3)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

PAGES = ["overview", "ground", "intact", "rockmass", "ccm", "settlement", "lining", "vibration", "equations"]
EXPECTED_WARNINGS = ("Cover above", "support reaches", "Screening model")


@pytest.mark.parametrize("page", PAGES)
def test_page_draws(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, "rse3010_geo.py"), default_timeout=60)
    at.session_state["page"] = page
    at.run()
    assert not at.exception, at.exception
    assert all(any(s in w.value for s in EXPECTED_WARNINGS) for w in at.warning)


@pytest.mark.parametrize("label", list(ground.PRESETS))
def test_every_preset_runs_through_the_ccm_page(label):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, "rse3010_geo.py"), default_timeout=60)
    at.run()
    ground.load_preset(at.session_state, label)
    at.session_state["page"] = "ccm"
    at.run()
    assert not at.exception, at.exception


def test_default_page_reproduces_the_book():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, "rse3010_geo.py"), default_timeout=60)
    at.session_state["page"] = "ccm"
    at.run()
    values = {m.label: m.value for m in at.metric}
    assert values["Factor of safety p_sm/p_eq"] == "2.78"
    assert values["u_max"] == "126.9 mm"
