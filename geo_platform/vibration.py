"""TBM passage beneath buildings: settlement and vibration screening model.

A tunnel boring machine (or two, the second following at a lag) advances at a
constant rate beneath a set of receivers. Day by day the model predicts, at
each receiver, ground settlement, differential settlement and slope across the
footprint, peak particle velocity (PPV) at the foundation, the vibration dose
value (VDV) on the receiving floor, ground-borne noise in the receiving room
and the one-third-octave velocity used for vibration-sensitive equipment. It is
a screening model: every attenuation constant is a placeholder to be replaced
by a site regression from monitored data.

Ground movement (Chapters 17 and 26 of the book)
    i = K z0;  V_s = (V_L/100) pi D^2/4;  S_max = V_s / (i sqrt(2 pi))  per bore.
    Settlement of bore j (axis at lateral position y_j, launched at x_0, face
    at x_f,j(t)) after Attewell and Woodman (1982):

        S_j = S_max exp(-(y - y_j)^2 / 2i^2) [Phi((x_f,j - x)/i) - Phi((x_0 - x)/i)]

    with Phi the standard normal cumulative distribution. Far from the launch
    the second term vanishes and the form is the familiar cumulative normal:
    half the final settlement at the face and the full value well behind it.
    The two bores are superposed; the second bore's face is
    x_0 + v max(0, t - t_lag).

Vibration (Chapter 22)
    Slant distance from face to receiver at ground level,
    d = sqrt(dx^2 + dy^2 + z0^2).
    PPV_j = PPV_ref (d/d_ref)^(-n) exp(-beta (d - d_ref))   (foundation, mm/s);
    PPV = max_j PPV_j (the peaks of two machines do not coincide).
    Floor RMS velocity v_rms,j = A_floor PPV_j / C (C the crest factor);
    weighted RMS acceleration a_j = W 2 pi f v_rms,j (m/s^2);
    estimated VDV over the assessment period T_p (BS 6472-1),
    VDV = 1.4 [max_window mean(sum_j a_j^4) 3600 T_p delta]^(1/4)   (m/s^1.75),
    the mean taken over the positions in a window of length T_p and the worst
    window ending in each day kept; delta the active (boring) fraction of time.
    Ground-borne noise, an equivalent continuous A-weighted level over the
    period, tracks the velocity of the radiating room surfaces dB for dB:
    L_j = L_ref + 20 log10(A_floor PPV_j / PPV_ref) + G_room + 10 log10(delta),
    combined energetically. One-third-octave RMS velocity at the dominant band,
    V_j = V_ref (d/d_ref)^(-n_s) exp(-beta_s (d - d_ref)) A_floor (um/s),
    combined as the root sum of squares, against equipment classes L1 to L3.

Corrections relative to the command-line script ``TBM_Vibration.py``
    1. Longitudinal profile. The script used a Gaussian in (x_f - x), so the
       settlement recovered to zero once the face had passed. Settlement is
       permanent: the cumulative normal above replaces it.
    2. Longitudinal spread. The script carried a separate sigma = 35 m with no
       source for it; Attewell and Woodman use the same i = K z0 along and
       across the drive, and so does this model.
    3. Launch. A bore that has not been launched is excavated nowhere: it now
       gives no settlement (the Phi(x_0) term) and no vibration. The script let
       the second bore vibrate from the launch chainage during its lag.
    4. Differential settlement and slope. The script took the total settlement
       times min(1, L/2i), which is the whole settlement for any building wider
       than 2i, and divided it by L to give an "angular distortion" that was in
       fact a mean slope. Both are now computed from the settlement field
       across the footprint, a line of length L laid along and across the
       drive: the largest settlement, the larger range of settlement and the
       steepest slope are reported, and they are screened against 10 mm and
       1 in 500 (the first stage of Chapter 26). The settlement at the
       reference point is kept in the detail table.
    5. Path length. The height of the receiving floor was added to the ground
       path, which handles the building twice (floor amplification already
       does). Distance is now face to ground level; PPV, a structural quantity
       checked at the foundation, carries no amplification; VDV, noise and the
       equipment spectrum, all floor quantities, do. VDV previously had none.
    6. VDV. The estimated VDV is 1.4 a_rms t^(1/4); the factor 1.4 was missing.
       The criteria (0.4 and 0.8 m/s^1.75) are 16-hour daytime values, but the
       dose was accumulated over 15 minutes, which understates it by
       (64)^(1/4) = 2.8. The two bores now add as fourth powers rather than the
       larger being taken, and a frequency-weighting factor is exposed.
    7. Ground-borne noise. The script attenuated noise with 20 log10(d) and
       0.035 dB/m, a law of its own, while the vibration that radiates it fell
       with n = 1.2 and beta = 0.012 /m (0.104 dB/m). Noise now follows the
       velocity law, so the two cannot disagree.
    8. Decibel ratios. "Close to criterion" was L > 0.9 L_crit, a ratio of
       logarithmic levels with no meaning (0.9 x 45 dBA = 40.5 dBA). The margin
       is now the level difference, expressed as the amplitude ratio
       10^((L - L_crit)/20) in the governing ratio.
    9. Sampling. Daily peaks are taken over sub-steps of the day's advance, so
       a 28.6 m daily step cannot step over the closest approach.
    10. Human-response criteria (VDV, noise) are no longer applied to
        unoccupied receivers such as a services corridor or a ground monitor.

Units: m, days, mm (settlement), mm/s (PPV), um/s (spectral), m/s^1.75 (VDV),
dBA (noise). Pure numpy and pandas; no Streamlit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd
from scipy.special import ndtr

OCCUPANCIES = ("Residential", "Teaching and office", "Unoccupied")
STRUCTURE_CLASSES = ("Ordinary", "Heritage")
EQUIPMENT_CLASSES = ("None", "L1", "L2", "L3")

EVDV_FACTOR = 1.4          # BS 6472-1 estimated VDV, eVDV = 1.4 a_rms t^(1/4)
FOOTPRINT_POINTS = 11      # least number of samples along each footprint line; more if L > 2.5 i

QUANTITIES = ("settlement", "slope", "ppv", "vdv", "gbn", "spec")
QUANTITY_LABELS = {
    "settlement": "Settlement",
    "slope": "Slope",
    "ppv": "PPV",
    "vdv": "VDV",
    "gbn": "Ground-borne noise",
    "spec": "Spectral velocity",
}


@dataclass
class VibrationInputs:
    # Tunnel and drive
    start_chainage_m: float = -150.0
    advance_rate_m_per_day: float = 28.57
    active_duty_factor: float = 0.50
    tunnel_axis_depth_m: float = 26.0
    twin_tunnel_spacing_m: float = 15.0
    tunnel_diameter_m: float = 7.0
    second_bore_lag_days: float = 0.0
    model_days: int = 36
    substeps_per_day: int = 48
    # Ground
    trough_width_K: float = 0.50
    volume_loss_pct: float = 0.5
    # Vibration and noise attenuation
    ppv_ref_mm_s: float = 0.65
    ref_distance_m: float = 10.0
    ppv_geom_exponent: float = 1.20
    ppv_damping_per_m: float = 0.012
    dominant_frequency_hz: float = 25.0
    crest_factor: float = math.sqrt(2.0)
    frequency_weighting: float = 1.0
    vdv_period_h: float = 16.0
    gbn_ref_dba: float = 52.0
    room_gain_db: float = 6.0
    spec_ref_um_s: float = 35.0
    spec_geom_exponent: float = 1.15
    spec_damping_per_m: float = 0.015
    # Criteria
    settlement_limit_mm: float = 10.0
    slope_limit_one_in: float = 500.0
    ppv_ordinary_mm_s: float = 5.0
    ppv_heritage_mm_s: float = 3.0
    gbn_teaching_dba: float = 45.0
    gbn_residential_dba: float = 40.0
    vdv_teaching: float = 0.80
    vdv_residential: float = 0.40
    spec_L1_um_s: float = 25.0
    spec_L2_um_s: float = 12.5
    spec_L3_um_s: float = 6.25
    near_fraction: float = 0.90

    @property
    def trough_width_m(self) -> float:
        """i = K z0, used across and along the drive."""
        return self.trough_width_K * self.tunnel_axis_depth_m

    @property
    def tunnel_area_m2(self) -> float:
        return math.pi * self.tunnel_diameter_m ** 2 / 4.0

    @property
    def volume_loss_m3_per_m(self) -> float:
        return self.volume_loss_pct / 100.0 * self.tunnel_area_m2

    @property
    def smax_per_bore_mm(self) -> float:
        return self.volume_loss_m3_per_m / (self.trough_width_m * math.sqrt(2.0 * math.pi)) * 1000.0

    @property
    def vdv_active_seconds(self) -> float:
        """Boring time within the VDV assessment period (s)."""
        return self.vdv_period_h * 3600.0 * self.active_duty_factor

    def validate(self) -> None:
        positive = (
            "active_duty_factor", "tunnel_axis_depth_m", "tunnel_diameter_m", "model_days",
            "substeps_per_day", "trough_width_K", "ppv_ref_mm_s", "ref_distance_m",
            "dominant_frequency_hz", "crest_factor", "frequency_weighting", "vdv_period_h",
            "spec_ref_um_s", "settlement_limit_mm", "slope_limit_one_in", "ppv_ordinary_mm_s",
            "ppv_heritage_mm_s", "vdv_teaching", "vdv_residential", "spec_L1_um_s",
            "spec_L2_um_s", "spec_L3_um_s", "near_fraction",
        )
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive.")
        non_negative = (
            "advance_rate_m_per_day", "twin_tunnel_spacing_m", "second_bore_lag_days",
            "volume_loss_pct", "ppv_geom_exponent", "ppv_damping_per_m",
            "spec_geom_exponent", "spec_damping_per_m",
        )
        for name in non_negative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative.")
        if self.active_duty_factor > 1:
            raise ValueError("active_duty_factor cannot exceed 1.")
        if self.near_fraction > 1:
            raise ValueError("near_fraction cannot exceed 1.")


@dataclass
class Receiver:
    receiver_id: str
    receiver_name: str
    zone: str
    chainage_m: float
    lateral_offset_m: float      # from the centreline between the bores
    footprint_length_m: float    # 0 for a point (monitor)
    occupancy: str               # see OCCUPANCIES
    structure_class: str         # see STRUCTURE_CLASSES
    equipment_class: str         # see EQUIPMENT_CLASSES
    floor_amp: float             # foundation-to-receiving-floor velocity ratio

    def ppv_criterion(self, m: VibrationInputs) -> float:
        return m.ppv_heritage_mm_s if self.structure_class == "Heritage" else m.ppv_ordinary_mm_s

    def gbn_criterion(self, m: VibrationInputs) -> float:
        return {"Residential": m.gbn_residential_dba,
                "Teaching and office": m.gbn_teaching_dba}.get(self.occupancy, math.nan)

    def vdv_criterion(self, m: VibrationInputs) -> float:
        return {"Residential": m.vdv_residential,
                "Teaching and office": m.vdv_teaching}.get(self.occupancy, math.nan)

    def spectral_criterion(self, m: VibrationInputs) -> float:
        return {"L1": m.spec_L1_um_s, "L2": m.spec_L2_um_s,
                "L3": m.spec_L3_um_s}.get(self.equipment_class, math.nan)

    def settlement_criterion(self, m: VibrationInputs) -> float:
        return m.settlement_limit_mm if self.footprint_length_m > 0 else math.nan

    def slope_criterion(self, m: VibrationInputs) -> float:
        return 1.0 / m.slope_limit_one_in if self.footprint_length_m > 0 else math.nan


DEFAULT_RECEIVERS: list[Receiver] = [
    Receiver("R01", "Teaching block A", "Teaching precinct", 120, 45, 60, "Teaching and office", "Ordinary", "None", 1.10),
    Receiver("R02", "Student residence", "Residential", 180, 90, 40, "Residential", "Ordinary", "None", 1.20),
    Receiver("R03", "Engineering laboratory (L1 equipment)", "Research precinct", 250, 28, 50, "Teaching and office", "Ordinary", "L1", 1.25),
    Receiver("R04", "Imaging laboratory (VC-L3 equipment)", "Research precinct", 310, 18, 35, "Teaching and office", "Ordinary", "L3", 1.40),
    Receiver("R05", "Biomedical laboratory (L2 equipment)", "Research precinct", 355, 24, 45, "Teaching and office", "Ordinary", "L2", 1.35),
    Receiver("R06", "Buried services corridor", "Infrastructure", 420, 12, 20, "Unoccupied", "Ordinary", "None", 1.00),
    Receiver("R07", "Technology building (L1 equipment)", "Technology precinct", 500, 60, 55, "Teaching and office", "Ordinary", "L1", 1.20),
    Receiver("R08", "Health research building (L2 equipment)", "Health precinct", 580, 72, 50, "Teaching and office", "Ordinary", "L2", 1.25),
    Receiver("R09", "Teaching block B", "Teaching precinct", 650, 35, 65, "Teaching and office", "Ordinary", "None", 1.10),
    Receiver("R10", "Heritage building", "Heritage precinct", 690, 22, 25, "Teaching and office", "Heritage", "None", 1.15),
    Receiver("R11", "Ground surface monitor", "Monitoring", 740, 0, 0, "Unoccupied", "Ordinary", "None", 1.00),
]

RECEIVER_COLUMNS = [f.name for f in fields(Receiver)]


def receivers_to_frame(receivers: list[Receiver]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in receivers], columns=RECEIVER_COLUMNS)


def receivers_from_frame(df: pd.DataFrame) -> list[Receiver]:
    """Receivers from an edited table; rows without an id or chainage are skipped."""
    out: list[Receiver] = []
    for _, row in df.iterrows():
        rid = row.get("receiver_id")
        chain = row.get("chainage_m")
        if rid is None or (isinstance(rid, float) and math.isnan(rid)) or str(rid).strip() == "":
            continue
        if chain is None or pd.isna(chain):
            continue

        def num(key: str, default: float) -> float:
            value = row.get(key)
            return default if value is None or pd.isna(value) else float(value)

        def text(key: str, default: str, allowed: tuple[str, ...] | None = None) -> str:
            value = row.get(key)
            value = default if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
            return value if allowed is None or value in allowed else default

        out.append(Receiver(
            receiver_id=str(rid).strip(),
            receiver_name=text("receiver_name", str(rid)),
            zone=text("zone", ""),
            chainage_m=float(chain),
            lateral_offset_m=num("lateral_offset_m", 0.0),
            footprint_length_m=max(num("footprint_length_m", 0.0), 0.0),
            occupancy=text("occupancy", "Teaching and office", OCCUPANCIES),
            structure_class=text("structure_class", "Ordinary", STRUCTURE_CLASSES),
            equipment_class=text("equipment_class", "None", EQUIPMENT_CLASSES),
            floor_amp=max(num("floor_amp", 1.0), 0.0),
        ))
    return out


# --------------------------------------------------------------------------
# Field functions (vectorised; broadcastable arrays)
# --------------------------------------------------------------------------

def _pdf(u):
    return np.exp(-0.5 * np.asarray(u) ** 2) / math.sqrt(2.0 * math.pi)


def bore_lateral_positions(m: VibrationInputs) -> tuple[float, float]:
    """Bore 1 at +s/2, bore 2 at -s/2 from the centreline."""
    return 0.5 * m.twin_tunnel_spacing_m, -0.5 * m.twin_tunnel_spacing_m


def face_chainage(m: VibrationInputs, t, bore: int):
    """Face chainage of bore 1 or 2 at time t (days); the launch chainage before launch."""
    lag = 0.0 if bore == 1 else m.second_bore_lag_days
    return m.start_chainage_m + m.advance_rate_m_per_day * np.maximum(0.0, np.asarray(t, dtype=float) - lag)


def bore_active(m: VibrationInputs, t, bore: int):
    """True once the bore has been launched (t beyond its lag)."""
    lag = 0.0 if bore == 1 else m.second_bore_lag_days
    return np.asarray(t, dtype=float) > lag


def settlement_bore(m: VibrationInputs, x, y, face, y_bore):
    """Settlement (mm) and its gradients (mm/m) from one bore, Attewell and Woodman (1982)."""
    i = m.trough_width_m
    dy = np.asarray(y, dtype=float) - y_bore
    transverse = np.exp(-dy ** 2 / (2.0 * i ** 2))
    u_face = (np.asarray(face, dtype=float) - x) / i
    u_launch = (m.start_chainage_m - np.asarray(x, dtype=float)) / i
    longitudinal = ndtr(u_face) - ndtr(u_launch)
    s = m.smax_per_bore_mm * transverse * longitudinal
    ds_dx = m.smax_per_bore_mm * transverse * (_pdf(u_launch) - _pdf(u_face)) / i
    ds_dy = -s * dy / i ** 2
    return s, ds_dx, ds_dy


def attenuation(ref_value: float, d, d_ref: float, n: float, beta: float):
    """value(d) = ref (d/d_ref)^-n exp(-beta (d - d_ref)): geometric spreading and material damping."""
    d = np.asarray(d, dtype=float)
    return ref_value * (d / d_ref) ** (-n) * np.exp(-beta * (d - d_ref))


def evdv(a_rms, active_seconds: float):
    """Estimated VDV (m/s^1.75) of a steady weighted RMS acceleration over the active seconds."""
    return EVDV_FACTOR * np.asarray(a_rms, dtype=float) * active_seconds ** 0.25


def gbn_ratio(level_dba, criterion_dba):
    """Noise margin as an amplitude ratio, 10^((L - L_crit)/20)."""
    return 10.0 ** ((np.asarray(level_dba, dtype=float) - criterion_dba) / 20.0)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

def _receiver_arrays(m: VibrationInputs, receivers: list[Receiver]) -> dict[str, np.ndarray]:
    return {
        "x": np.array([r.chainage_m for r in receivers], dtype=float),
        "y": np.array([r.lateral_offset_m for r in receivers], dtype=float),
        "L": np.array([r.footprint_length_m for r in receivers], dtype=float),
        "amp": np.array([r.floor_amp for r in receivers], dtype=float),
        "crit_settlement": np.array([r.settlement_criterion(m) for r in receivers], dtype=float),
        "crit_slope": np.array([r.slope_criterion(m) for r in receivers], dtype=float),
        "crit_ppv": np.array([r.ppv_criterion(m) for r in receivers], dtype=float),
        "crit_vdv": np.array([r.vdv_criterion(m) for r in receivers], dtype=float),
        "crit_gbn": np.array([r.gbn_criterion(m) for r in receivers], dtype=float),
        "crit_spec": np.array([r.spectral_criterion(m) for r in receivers], dtype=float),
    }


def _settlement_day(m: VibrationInputs, ra: dict, t_sub: np.ndarray, t_end: float):
    """Settlement at the reference point and the footprint's largest settlement at the
    end of the day; the day's largest settlement range and slope across the footprint.

    The footprint is a line of length L centred on the receiver, laid once along and
    once across the drive (the orientation is not an input), and the larger of the two
    is kept. A point receiver (L = 0) has no range; its slope is the ground slope.
    """
    yb = bore_lateral_positions(m)
    s_point = 0.0
    for bore in (1, 2):
        s, _, _ = settlement_bore(m, ra["x"], ra["y"], face_chainage(m, t_end, bore), yb[bore - 1])
        s_point = s_point + s
    times = np.append(t_sub, t_end)
    # sample spacing no larger than i/4, so the largest settlement and slope are not missed
    n_pts = max(FOOTPRINT_POINTS, int(math.ceil(4.0 * float(ra["L"].max(initial=0.0)) / m.trough_width_m)) + 1)
    n_pts += 1 - n_pts % 2                                                              # odd: keep the centre
    u = np.linspace(-0.5, 0.5, n_pts)[None, :] * ra["L"][:, None]                      # (R, P)
    # and the exact positions of the peak settlement (bore axes) and the steepest slope
    # (inflection points y_j +/- i) where they fall within the footprint
    half = 0.5 * ra["L"][:, None]
    exact = np.stack([yb[j] - ra["y"] + off for j in (0, 1)
                      for off in (-m.trough_width_m, 0.0, m.trough_width_m)], axis=1)      # (R, 6)
    u = np.concatenate([u, np.clip(exact, -half, half)], axis=1)
    x0 = ra["x"][:, None]
    y0 = ra["y"][:, None]
    n_rec = len(ra["x"])
    best_end = np.zeros(n_rec)
    best_range = np.zeros(n_rec)
    best_slope = np.zeros(n_rec)
    for orientation in ("along", "across"):
        xs = (x0 + u) if orientation == "along" else np.broadcast_to(x0, u.shape)
        ys = np.broadcast_to(y0, u.shape) if orientation == "along" else (y0 + u)
        total = 0.0
        grad = 0.0
        for bore in (1, 2):
            face = face_chainage(m, times, bore)[:, None, None]                    # (T, 1, 1)
            s, gx, gy = settlement_bore(m, xs[None], ys[None], face, yb[bore - 1])  # (T, R, P)
            total = total + s
            grad = grad + (gx if orientation == "along" else gy)
        best_end = np.maximum(best_end, total[-1].max(axis=1))
        best_range = np.maximum(best_range, (total.max(axis=2) - total.min(axis=2)).max(axis=0))
        best_slope = np.maximum(best_slope, np.abs(grad).max(axis=2).max(axis=0) / 1000.0)
    return s_point, best_end, best_range, best_slope


def _vibration_day(m: VibrationInputs, ra: dict, t_sub: np.ndarray):
    """Daily peak PPV, VDV over the period, peak noise level and peak spectral velocity."""
    yb = bore_lateral_positions(m)
    z0 = m.tunnel_axis_depth_m
    ppv_b, spec_b, a4_b, level_b = [], [], [], []
    for bore in (1, 2):
        face = face_chainage(m, t_sub, bore)[:, None]                      # (n, 1)
        active = bore_active(m, t_sub, bore)[:, None]                      # (n, 1)
        d = np.sqrt((face - ra["x"][None, :]) ** 2 + (ra["y"][None, :] - yb[bore - 1]) ** 2 + z0 ** 2)
        ppv = np.where(active, attenuation(m.ppv_ref_mm_s, d, m.ref_distance_m,
                                           m.ppv_geom_exponent, m.ppv_damping_per_m), 0.0)
        floor_ppv = ppv * ra["amp"][None, :]
        a_rms = m.frequency_weighting * 2.0 * math.pi * m.dominant_frequency_hz * floor_ppv / m.crest_factor / 1000.0
        with np.errstate(divide="ignore"):
            level = np.where(
                active & (floor_ppv > 0),
                m.gbn_ref_dba + 20.0 * np.log10(np.where(floor_ppv > 0, floor_ppv, 1.0) / m.ppv_ref_mm_s)
                + m.room_gain_db + 10.0 * math.log10(m.active_duty_factor),
                -np.inf,
            )
        spec = np.where(active, attenuation(m.spec_ref_um_s, d, m.ref_distance_m,
                                            m.spec_geom_exponent, m.spec_damping_per_m), 0.0) * ra["amp"][None, :]
        ppv_b.append(ppv)
        spec_b.append(spec)
        a4_b.append(a_rms ** 4)
        level_b.append(level)
    ppv_tot = np.maximum(ppv_b[0], ppv_b[1])
    spec_tot = np.sqrt(spec_b[0] ** 2 + spec_b[1] ** 2)
    with np.errstate(divide="ignore"):
        energy = 10.0 ** (level_b[0] / 10.0) + 10.0 ** (level_b[1] / 10.0)
        level_tot = np.where(energy > 0, 10.0 * np.log10(np.where(energy > 0, energy, 1.0)), np.nan)
    a4 = a4_b[0] + a4_b[1]                                                 # (n, R); VDV is set in run_model
    with np.errstate(all="ignore"):
        level_peak = np.nanmax(np.where(np.isnan(level_tot), -np.inf, level_tot), axis=0)
    level_peak = np.where(np.isfinite(level_peak), level_peak, np.nan)
    # overall RMS velocity of both bores together, the bound on their combined band velocity (um/s)
    rms_overall = np.sqrt(ppv_b[0] ** 2 + ppv_b[1] ** 2) * ra["amp"][None, :] / m.crest_factor * 1000.0
    spec_excess = (spec_tot > rms_overall + 1e-9).any(axis=0)
    return {
        "ppv_1": ppv_b[0].max(axis=0), "ppv_2": ppv_b[1].max(axis=0),
        "ppv": ppv_tot.max(axis=0), "a4": a4, "gbn": level_peak,
        "spec": spec_tot.max(axis=0), "spec_above_rms": spec_excess,
    }


def _vdv_by_day(m: VibrationInputs, a4: np.ndarray, n_sub: int) -> np.ndarray:
    """Estimated VDV of the worst assessment window ending in each day, shape (days, R).

    ``a4`` holds the sum over the bores of a_rms^4 at every sub-step, shape (days n_sub, R).
    A window of length T_p covers round(n_sub T_p / 24) sub-steps; time before the
    first day contributes nothing. The mean of a^4 over the window is applied to the
    boring seconds of the period, 3600 T_p delta.
    """
    width = max(1, int(round(n_sub * m.vdv_period_h / 24.0)))
    padded = np.vstack([np.zeros((width, a4.shape[1])), a4])
    csum = np.vstack([np.zeros((1, a4.shape[1])), np.cumsum(padded, axis=0)])
    # window ending at sub-step k (0-based in a4) spans padded rows k+1 .. k+width
    k = np.arange(a4.shape[0])
    window_mean = (csum[k + width + 1] - csum[k + 1]) / width                  # (days n_sub, R)
    worst = window_mean.reshape(-1, n_sub, a4.shape[1]).max(axis=1)              # (days, R)
    return EVDV_FACTOR * (worst * m.vdv_active_seconds) ** 0.25


def _status(ratio: float, near: float) -> str:
    if not np.isfinite(ratio):
        return "No criterion"
    if ratio > 1.0:
        return "Exceeds"
    if ratio >= near:
        return "Close to criterion"
    return "Within"


def run_model(inputs: VibrationInputs, receivers: list[Receiver]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily detail (one row per receiver and day) and the per-receiver summary."""
    m = inputs
    m.validate()
    if not receivers:
        raise ValueError("At least one receiver is needed.")
    ra = _receiver_arrays(m, receivers)
    n_sub = int(m.substeps_per_day)
    days = np.arange(1, int(m.model_days) + 1)
    frames = []
    a4_days = []
    for day in days:
        t_sub = day - 1 + (np.arange(n_sub) + 0.5) / n_sub
        s_point, s_max, rng, slope = _settlement_day(m, ra, t_sub, float(day))
        vib = _vibration_day(m, ra, t_sub)
        a4_days.append(vib["a4"])
        frames.append(pd.DataFrame({
            "day": int(day),
            "receiver_id": [r.receiver_id for r in receivers],
            "receiver_name": [r.receiver_name for r in receivers],
            "zone": [r.zone for r in receivers],
            "bore1_face_m": float(face_chainage(m, day, 1)),
            "bore2_face_m": float(face_chainage(m, day, 2)),
            "settlement_point_mm": s_point,
            "settlement_mm": s_max,
            "diff_settlement_mm": rng,
            "slope": slope,
            "ppv_1_mm_s": vib["ppv_1"],
            "ppv_2_mm_s": vib["ppv_2"],
            "ppv_mm_s": vib["ppv"],
            "vdv_m_s175": np.nan,
            "gbn_dba": vib["gbn"],
            "spec_um_s": vib["spec"],
            "spec_above_overall_rms": vib["spec_above_rms"],
            "settlement_criterion_mm": ra["crit_settlement"],
            "slope_criterion": ra["crit_slope"],
            "ppv_criterion_mm_s": ra["crit_ppv"],
            "vdv_criterion_m_s175": ra["crit_vdv"],
            "gbn_criterion_dba": ra["crit_gbn"],
            "spec_criterion_um_s": ra["crit_spec"],
        }))
    detail = pd.concat(frames, ignore_index=True)
    detail["vdv_m_s175"] = _vdv_by_day(m, np.concatenate(a4_days, axis=0), n_sub).reshape(-1)

    ratios = _ratios(detail, prefix="")
    for q in QUANTITIES:
        detail[f"{q}_ratio"] = ratios[q]
        detail[f"{q}_exceed"] = (ratios[q] > 1.0).astype(int)
    detail["governing_ratio"] = ratios.max(axis=1, skipna=True)
    detail["status"] = [_status(v, m.near_fraction) for v in detail["governing_ratio"]]

    keys = ["receiver_id", "receiver_name", "zone"]
    summary = detail.groupby(keys, as_index=False, sort=False).agg(
        final_settlement_mm=("settlement_mm", "max"),
        max_diff_settlement_mm=("diff_settlement_mm", "max"),
        max_slope=("slope", "max"),
        peak_ppv_mm_s=("ppv_mm_s", "max"),
        peak_vdv_m_s175=("vdv_m_s175", "max"),
        peak_gbn_dba=("gbn_dba", "max"),
        peak_spec_um_s=("spec_um_s", "max"),
        settlement_criterion_mm=("settlement_criterion_mm", "first"),
        slope_criterion=("slope_criterion", "first"),
        ppv_criterion_mm_s=("ppv_criterion_mm_s", "first"),
        vdv_criterion_m_s175=("vdv_criterion_m_s175", "first"),
        gbn_criterion_dba=("gbn_criterion_dba", "first"),
        spec_criterion_um_s=("spec_criterion_um_s", "first"),
        days_settlement_exceed=("settlement_exceed", "sum"),
        days_slope_exceed=("slope_exceed", "sum"),
        days_ppv_exceed=("ppv_exceed", "sum"),
        days_vdv_exceed=("vdv_exceed", "sum"),
        days_gbn_exceed=("gbn_exceed", "sum"),
        days_spec_exceed=("spec_exceed", "sum"),
    )
    occ = {r.receiver_id: r for r in receivers}
    summary.insert(3, "occupancy", [occ[i].occupancy for i in summary["receiver_id"]])
    summary.insert(4, "structure_class", [occ[i].structure_class for i in summary["receiver_id"]])
    summary.insert(5, "equipment_class", [occ[i].equipment_class for i in summary["receiver_id"]])

    peak_ratios = _ratios(summary, prefix="peak")
    for q in QUANTITIES:
        summary[f"{q}_ratio"] = peak_ratios[q]
    gov = peak_ratios.max(axis=1, skipna=True)
    summary["governing_ratio"] = gov
    summary["governing_quantity"] = [
        QUANTITY_LABELS[row.idxmax()] if row.notna().any() else "--" for _, row in peak_ratios.iterrows()
    ]
    summary["status"] = [_status(v, m.near_fraction) for v in gov]
    actions = []
    for _, row in summary.iterrows():
        if row["status"] == "Exceeds":
            ground = row["settlement_ratio"] > 1 or row["slope_ratio"] > 1
            actions.append("Stage-two damage assessment; mitigation and monitoring plan" if ground
                           else "Refine the prediction; mitigation and monitoring plan")
        elif row["status"] == "Close to criterion":
            actions.append("Refine the inputs; close monitoring")
        elif row["status"] == "Within":
            actions.append("Routine baseline monitoring")
        else:
            actions.append("Record only")
    summary["recommended_action"] = actions
    return detail, summary


def _ratios(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Demand/criterion ratios; NaN where no criterion applies. Noise via 10^(dL/20)."""
    if prefix == "peak":
        cols = {"settlement": "final_settlement_mm", "slope": "max_slope", "ppv": "peak_ppv_mm_s",
                "vdv": "peak_vdv_m_s175", "gbn": "peak_gbn_dba", "spec": "peak_spec_um_s"}
    else:
        cols = {"settlement": "settlement_mm", "slope": "slope", "ppv": "ppv_mm_s",
                "vdv": "vdv_m_s175", "gbn": "gbn_dba", "spec": "spec_um_s"}
    crits = {"settlement": "settlement_criterion_mm", "slope": "slope_criterion",
             "ppv": "ppv_criterion_mm_s", "vdv": "vdv_criterion_m_s175",
             "gbn": "gbn_criterion_dba", "spec": "spec_criterion_um_s"}
    out = {}
    for q in QUANTITIES:
        value = df[cols[q]].to_numpy(dtype=float)
        crit = df[crits[q]].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[q] = gbn_ratio(value, crit) if q == "gbn" else value / crit
    return pd.DataFrame(out, index=df.index)


__all__ = [
    "VibrationInputs", "Receiver", "DEFAULT_RECEIVERS", "OCCUPANCIES", "STRUCTURE_CLASSES",
    "EQUIPMENT_CLASSES", "QUANTITIES", "QUANTITY_LABELS", "RECEIVER_COLUMNS", "run_model",
    "receivers_to_frame", "receivers_from_frame", "settlement_bore", "face_chainage",
    "bore_active", "attenuation", "evdv", "gbn_ratio", "bore_lateral_positions",
]
