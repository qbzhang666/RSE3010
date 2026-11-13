"""Checks of geo_platform.vibration against hand calculations, and a page smoke test.

Hand calculations (worked independently of the module, by calculator):

Defaults: D = 7 m, z0 = 26 m, K = 0.5, V_L = 0.5 per cent.
    i = 0.5 x 26 = 13 m;  A = pi 7^2/4 = 38.4845 m^2;  V_s = 0.005 x 38.4845 = 0.192423 m^3/m
    S_max = 0.192423 / (13 x 2.506628) = 5.90504e-3 m = 5.90504 mm

Longitudinal (one bore, receiver over the axis, launch 100 m behind it):
    face 10 m short:  S = S_max Phi(-10/13) = 5.90504 x 0.220878 = 1.30429 mm
    face at receiver: S = 0.5 S_max = 2.95252 mm;  far behind: S = S_max

Footprint L = 26 m = 2i over one bore:
    along the drive, face centred: range = S_max [Phi(1) - Phi(-1)] = 0.682689 S_max = 4.03131 mm
    across, after the face: range = S_max (1 - e^-0.5) = 2.32345 mm
    steepest slope, at y = i: S_max e^-0.5 / i = 0.275507 mm/m = 2.75507e-4

Vibration at d = z0 = 26 m (receiver over bore 1, face beneath it), d_ref = 10 m:
    PPV = 0.65 x 2.6^-1.2 x exp(-0.012 x 16) = 0.65 x 0.317702 x 0.825307 = 0.170436 mm/s
    A = 1.2, C = sqrt 2: v_rms = 0.170436 x 1.2 / 1.414214 = 0.144620 mm/s
    a = 2 pi 25 x 1.44620e-4 = 0.0227168 m/s^2;  t = 16 x 3600 x 0.5 = 28 800 s, t^1/4 = 13.0277
    VDV = 1.4 x 0.0227168 x 13.0277 = 0.414308 m/s^1.75
    L = 52 + 20 log10(0.170436 x 1.2 / 0.65) + 6 + 10 log10(0.5) = 52 - 10.0434 + 6 - 3.0103 = 44.9463 dBA
    V = 35 x 2.6^-1.15 x exp(-0.015 x 16) x 1.2 = 11.0103 um/s

Two bores, s = 15 m, receiver on the centreline above both faces, A = 1:
    d = sqrt(7.5^2 + 26^2) = 27.0601 m;  PPV_j = 0.160401 mm/s;  PPV = max = 0.160401 (not the sum)
    VDV = 1.4 (2 a^4 t)^1/4 = 2^1/4 x 0.324930 = 0.386409 m/s^1.75
    L_j = 42.8356 dBA;  L = L_j + 10 log10 2 = 45.8459 dBA;  V = sqrt 2 x 8.62487 = 12.1974 um/s
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geo_platform import vibration as vib  # noqa: E402

REL = 1e-4


def _receiver(**kw) -> vib.Receiver:
    base = dict(receiver_id="T1", receiver_name="Test", zone="Test", chainage_m=0.0, lateral_offset_m=0.0,
                footprint_length_m=0.0, occupancy="Teaching and office", structure_class="Ordinary",
                equipment_class="None", floor_amp=1.0)
    base.update(kw)
    return vib.Receiver(**base)


def _row(detail: pd.DataFrame, day: int, rid: str = "T1") -> pd.Series:
    return detail[(detail["day"] == day) & (detail["receiver_id"] == rid)].iloc[0]


# ---------------------------------------------------------------- settlement

def test_smax_per_bore():
    m = vib.VibrationInputs()
    assert m.trough_width_m == pytest.approx(13.0)
    assert m.smax_per_bore_mm == pytest.approx(5.90504, rel=REL)


def test_longitudinal_profile_is_cumulative_normal():
    m = vib.VibrationInputs(start_chainage_m=0.0, advance_rate_m_per_day=10.0, second_bore_lag_days=1000.0,
                            twin_tunnel_spacing_m=0.0, model_days=30, substeps_per_day=10)
    detail, _ = vib.run_model(m, [_receiver(chainage_m=100.0)])
    assert _row(detail, 9)["settlement_point_mm"] == pytest.approx(1.30429, rel=REL)
    assert _row(detail, 10)["settlement_point_mm"] == pytest.approx(2.95252, rel=REL)
    assert _row(detail, 30)["settlement_point_mm"] == pytest.approx(5.90504, rel=REL)
    # settlement is permanent: it never recovers behind the face (the source script's defect)
    series = detail["settlement_point_mm"].to_numpy()
    assert np.all(np.diff(series) >= -1e-12)


def test_unlaunched_bore_gives_no_settlement_or_vibration():
    m = vib.VibrationInputs(start_chainage_m=0.0, advance_rate_m_per_day=10.0, second_bore_lag_days=5.0,
                            model_days=8, substeps_per_day=4)
    detail, _ = vib.run_model(m, [_receiver(chainage_m=0.0)])
    early = detail[detail["day"] <= 5]
    late = detail[detail["day"] >= 6]
    assert np.all(early["bore2_face_m"] == 0.0)
    assert np.all(early["ppv_2_mm_s"] == 0.0)
    assert np.all(late["ppv_2_mm_s"] > 0.0)


def test_footprint_range_and_slope():
    m = vib.VibrationInputs(start_chainage_m=0.0, advance_rate_m_per_day=10.0, second_bore_lag_days=1000.0,
                            twin_tunnel_spacing_m=0.0, model_days=30, substeps_per_day=10)
    detail, summary = vib.run_model(m, [_receiver(chainage_m=100.5, footprint_length_m=26.0)])
    # face centred on the building at a sub-step: the along-drive range governs
    assert summary.loc[0, "max_diff_settlement_mm"] == pytest.approx(4.03131, rel=REL)
    # long after the face: only the transverse trough remains
    assert _row(detail, 30)["diff_settlement_mm"] == pytest.approx(2.32345, rel=REL)
    assert summary.loc[0, "max_slope"] == pytest.approx(2.75507e-4, rel=REL)
    assert summary.loc[0, "final_settlement_mm"] == pytest.approx(5.90504, rel=REL)


# ----------------------------------------------------------------- vibration

def _stationary_single(**rkw):
    m = vib.VibrationInputs(start_chainage_m=0.0, advance_rate_m_per_day=0.0, second_bore_lag_days=1000.0,
                            model_days=2, substeps_per_day=4)
    r = _receiver(chainage_m=0.0, lateral_offset_m=7.5, floor_amp=1.2, equipment_class="L1", **rkw)
    return vib.run_model(m, [r])


def test_ppv_attenuation_law():
    detail, _ = _stationary_single()
    assert _row(detail, 1)["ppv_mm_s"] == pytest.approx(0.170436, rel=REL)
    assert vib.attenuation(0.65, 26.0, 10.0, 1.2, 0.012) == pytest.approx(0.170436, rel=REL)


def test_vdv_estimated_over_period():
    detail, _ = _stationary_single()
    assert _row(detail, 1)["vdv_m_s175"] == pytest.approx(0.414308, rel=REL)
    assert vib.evdv(0.0227168, 28800.0) == pytest.approx(0.414308, rel=REL)


def test_ground_borne_noise_from_velocity():
    detail, _ = _stationary_single()
    assert _row(detail, 1)["gbn_dba"] == pytest.approx(44.9463, abs=1e-3)


def test_spectral_velocity():
    detail, _ = _stationary_single()
    assert _row(detail, 1)["spec_um_s"] == pytest.approx(11.0103, rel=REL)
    assert _row(detail, 1)["spec_criterion_um_s"] == pytest.approx(25.0)


def test_two_bores_combine_by_their_own_rules():
    m = vib.VibrationInputs(start_chainage_m=0.0, advance_rate_m_per_day=0.0, second_bore_lag_days=0.0,
                            model_days=1, substeps_per_day=2)
    detail, _ = vib.run_model(m, [_receiver(chainage_m=0.0, lateral_offset_m=0.0)])
    row = _row(detail, 1)
    assert row["ppv_mm_s"] == pytest.approx(0.160401, rel=REL)       # larger, not summed
    assert row["vdv_m_s175"] == pytest.approx(0.386409, rel=REL)     # fourth-power sum
    assert row["gbn_dba"] == pytest.approx(45.8459, abs=1e-3)        # energetic sum
    assert row["spec_um_s"] == pytest.approx(12.1974, rel=REL)       # root sum of squares


def test_noise_margin_is_a_level_difference():
    assert vib.gbn_ratio(46.0, 40.0) == pytest.approx(10 ** 0.3, rel=1e-9)
    assert vib.gbn_ratio(44.5, 45.0) == pytest.approx(0.944061, rel=REL)   # 0.5 dB under: close
    assert vib.gbn_ratio(43.0, 45.0) == pytest.approx(0.794328, rel=REL)   # 2 dB under: within
    # the source script's rule 43 > 0.9 x 45 would have flagged the 43 dBA case


def test_criteria_by_class():
    m = vib.VibrationInputs()
    unocc = _receiver(occupancy="Unoccupied")
    assert math.isnan(unocc.vdv_criterion(m)) and math.isnan(unocc.gbn_criterion(m))
    assert math.isnan(unocc.settlement_criterion(m))          # point receiver
    res = _receiver(occupancy="Residential", structure_class="Heritage", footprint_length_m=10.0)
    assert res.vdv_criterion(m) == 0.40 and res.gbn_criterion(m) == 40.0
    assert res.ppv_criterion(m) == 3.0
    assert res.slope_criterion(m) == pytest.approx(1 / 500)
    assert math.isnan(res.spectral_criterion(m))


# ------------------------------------------------------------- whole model

def test_default_run_shape_and_sense():
    m = vib.VibrationInputs()
    detail, summary = vib.run_model(m, vib.DEFAULT_RECEIVERS)
    assert len(detail) == m.model_days * len(vib.DEFAULT_RECEIVERS)
    assert list(summary["receiver_id"]) == [r.receiver_id for r in vib.DEFAULT_RECEIVERS]
    for _, grp in detail.groupby("receiver_id"):
        assert np.all(np.diff(grp["settlement_mm"].to_numpy()) >= -1e-9)
    # twin bores, 15 m apart: the most a point can settle is 2 S_max exp(-7.5^2/2i^2) = 9.998 mm
    assert summary["final_settlement_mm"].max() <= 2 * 5.90504 * math.exp(-7.5 ** 2 / 338) + 1e-3
    assert set(summary["status"]) <= {"Exceeds", "Close to criterion", "Within", "No criterion"}
    assert not detail["spec_above_overall_rms"].any()
    text = " ".join(r.receiver_name for r in vib.DEFAULT_RECEIVERS)
    assert "Heritage" in text and "Student residence" in text


def test_receiver_frame_round_trip():
    df = vib.receivers_to_frame(vib.DEFAULT_RECEIVERS)
    back = vib.receivers_from_frame(df)
    assert back == vib.DEFAULT_RECEIVERS
    df.loc[len(df)] = [None] * len(df.columns)       # an empty editor row is skipped
    assert len(vib.receivers_from_frame(df)) == len(vib.DEFAULT_RECEIVERS)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        vib.run_model(vib.VibrationInputs(tunnel_axis_depth_m=0.0), vib.DEFAULT_RECEIVERS)
    with pytest.raises(ValueError):
        vib.run_model(vib.VibrationInputs(), [])


# ----------------------------------------------------------------- the page

def test_page_runs_in_apptest():
    from streamlit.testing.v1 import AppTest

    def app():
        from geo_platform.vibration_page import render_vibration

        render_vibration({})

    at = AppTest.from_function(app, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("TBM passage" in t.value for t in at.title)
    # change an input and the receiver in focus; the page must rerun cleanly
    at.number_input(key="vib_second_bore_lag_days").set_value(10.0)
    at.selectbox(key="vib_focus").set_value("R10")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
