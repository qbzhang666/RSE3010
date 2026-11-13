"""Regression tests for the three findings of the independent review of the
vibration model (November 2026), each against a value worked out separately.

    python -m pytest tests/test_vibration_review.py -q
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geo_platform import vibration as vib  # noqa: E402


def _receiver(**kw):
    base = vib.DEFAULT_RECEIVERS[0]
    return replace(base, **kw)


def test_long_footprint_finds_the_peak_settlement_and_slope():
    """A 150 m footprint across a single bore: the largest settlement is S_max at the
    axis and the steepest slope S_max e^-1/2 / i at the inflection points, whatever
    the sampling. Eleven points used to give 8.94 mm and 1 in 2920."""
    m = replace(vib.VibrationInputs(), twin_tunnel_spacing_m=0.0, second_bore_lag_days=0.0)
    i, smax = m.trough_width_m, m.smax_per_bore_mm
    rec = _receiver(receiver_id="T", chainage_m=0.0, lateral_offset_m=7.5, footprint_length_m=150.0)
    _, summary = vib.run_model(m, [rec])
    two_bores = 2 * smax                     # both bores on the centreline, spacing 0
    assert summary["final_settlement_mm"].iloc[0] == pytest.approx(two_bores, rel=2e-3)
    slope_exact = two_bores * math.exp(-0.5) / i / 1000.0
    assert summary["max_slope"].iloc[0] == pytest.approx(slope_exact, rel=1e-2)


def test_vdv_takes_the_worst_window_not_the_calendar_day():
    """VDV must not depend on where the passage falls within a calendar day: moving
    the launch by a fraction of a day changes the calendar-day value by about 6 per
    cent, but the worst window by much less. With a 24 h period and one bore the
    worst-window value is at least the calendar-day value."""
    m = vib.VibrationInputs()
    rec = [r for r in vib.DEFAULT_RECEIVERS if r.receiver_id == "R04"]
    peaks = []
    for shift in (0.0, 0.25, 0.5):
        mm = replace(m, start_chainage_m=m.start_chainage_m - shift * m.advance_rate_m_per_day)
        _, summary = vib.run_model(mm, rec)
        peaks.append(float(summary["peak_vdv_m_s175"].iloc[0]))
    assert max(peaks) / min(peaks) - 1 < 0.02
    assert peaks[0] > 0.4385 * 0.99         # the reviewer's worst 16 h window (calendar day gave 0.4005)


def test_band_velocity_check_uses_both_bores():
    """Band velocity of two equal bores is sqrt(2) times one; the overall RMS bound must
    be formed the same way, so a receiver midway between simultaneous bores raises no
    warning when each bore alone satisfies the bound."""
    m = replace(vib.VibrationInputs(), spec_ref_um_s=400.0, spec_geom_exponent=1.2,
                spec_damping_per_m=0.012, second_bore_lag_days=0.0)
    rec = _receiver(receiver_id="C", chainage_m=300.0, lateral_offset_m=0.0, footprint_length_m=0.0)
    detail, _ = vib.run_model(m, [rec])
    assert not bool(detail["spec_above_overall_rms"].any())
