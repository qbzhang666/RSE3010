
#!/usr/bin/env python3
"""
Monash Clayton Campus - SRL East induced construction prediction model
Worked, teaching-ready screening script.

What it does
------------
- Simulates an illustrative TBM pass beneath the eastern side of Monash Clayton campus
- Predicts settlement, differential settlement, PPV, VDV, ground-borne noise, and lab spectral RMS
- Produces detailed and summary CSV outputs plus a preview plot

Important
---------
This is a transparent screening model based on public project information and placeholder engineering
assumptions. Replace alignment geometry, geology, building data, and room-specific criteria with project-
specific values before using it for design or governance decisions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from math import exp, log10, pi, sqrt
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class ModelInputs:
    scenario_start_chainage_m: float = -150.0
    tbm_advance_rate_m_per_day: float = 28.57
    active_duty_factor: float = 0.50
    tunnel_axis_depth_m: float = 26.0
    twin_tunnel_spacing_m: float = 15.0
    tunnel_diameter_m: float = 7.0
    model_days: int = 36
    longitudinal_sigma_m: float = 35.0
    settlement_trough_coeff_K: float = 0.50
    volume_loss_fraction: float = 0.005
    second_bore_lag_days: float = 0.0
    ppv_ref_mm_s: float = 0.65
    ref_distance_m: float = 10.0
    ppv_geom_exponent: float = 1.20
    ppv_damping_beta: float = 0.012
    dominant_frequency_hz: float = 25.0
    gbn_ref_dba: float = 52.0
    gbn_damping_db_per_m: float = 0.035
    room_gain_db: float = 6.0
    spec_ref_um_s: float = 35.0
    spec_geom_exponent: float = 1.15
    spec_damping_beta: float = 0.015
    ppv_ordinary_mm_s: float = 5.0
    ppv_heritage_mm_s: float = 3.0
    gbn_academic_dba: float = 45.0
    gbn_residential_dba: float = 40.0
    vdv_academic: float = 0.80
    vdv_residential: float = 0.40
    spec_L1_um_s: float = 25.0
    spec_L2_um_s: float = 12.5
    spec_L3_um_s: float = 6.25

    @property
    def tunnel_area_m2(self) -> float:
        return pi * (self.tunnel_diameter_m / 2) ** 2

    @property
    def trough_width_m(self) -> float:
        return self.settlement_trough_coeff_K * self.tunnel_axis_depth_m

    @property
    def smax_per_bore_mm(self) -> float:
        return (self.volume_loss_fraction * self.tunnel_area_m2) / (sqrt(2 * pi) * self.trough_width_m) * 1000

    @property
    def active_seconds_15min(self) -> float:
        return 900 * self.active_duty_factor


@dataclass
class Receiver:
    receiver_id: str
    receiver_name: str
    zone: str
    chainage_m: float
    lateral_offset_m: float
    floor_height_m: float
    footprint_length_m: float
    occupancy: str
    structure_class: str
    equipment_class: str
    floor_amp: float

    def gbn_criterion(self, m: ModelInputs) -> float:
        return m.gbn_residential_dba if self.occupancy == "Residential" else m.gbn_academic_dba

    def ppv_criterion(self, m: ModelInputs) -> float:
        return m.ppv_heritage_mm_s if self.structure_class == "Heritage" else m.ppv_ordinary_mm_s

    def vdv_criterion(self, m: ModelInputs) -> float:
        return m.vdv_residential if self.occupancy == "Residential" else m.vdv_academic

    def spectral_criterion(self, m: ModelInputs) -> float:
        return {
            "L1": m.spec_L1_um_s,
            "L2": m.spec_L2_um_s,
            "L3": m.spec_L3_um_s,
        }.get(self.equipment_class, 0.0)


DEFAULT_RECEIVERS: List[Receiver] = [
    Receiver("R01","East Teaching Block","Academic Core",120,45,4,60,"Academic","Ordinary","None",1.10),
    Receiver("R02","Student Accommodation North","Residential",180,90,12,40,"Residential","Ordinary","None",1.20),
    Receiver("R03","Engineering Labs East","Sensitive Research",250,28,6,50,"Academic","Ordinary","L1",1.25),
    Receiver("R04","Microscopy Suite Proxy","Sensitive Research",310,18,8,35,"Academic","Ordinary","L3",1.40),
    Receiver("R05","Biomedical Imaging Proxy","Sensitive Research",355,24,6,45,"Academic","Ordinary","L2",1.35),
    Receiver("R06","Utilities Corridor","Infrastructure",420,12,0,20,"Academic","Ordinary","None",1.00),
    Receiver("R07","CSIRO Interface Proxy","Tech Precinct",500,60,6,55,"Academic","Ordinary","L1",1.20),
    Receiver("R08","Victorian Heart Interface Proxy","Health Interface",580,72,5,50,"Academic","Ordinary","L2",1.25),
    Receiver("R09","East Campus Teaching South","Academic Core",650,35,3,65,"Academic","Ordinary","None",1.10),
    Receiver("R10","Ground Surface Monitor","Monitoring",740,0,0,0,"Academic","Ordinary","None",1.00),
]


def build_time_steps(m: ModelInputs) -> pd.DataFrame:
    rows = []
    for day in range(m.model_days + 1):
        bore1 = m.scenario_start_chainage_m + day * m.tbm_advance_rate_m_per_day
        if day < m.second_bore_lag_days:
            bore2 = m.scenario_start_chainage_m
        else:
            bore2 = m.scenario_start_chainage_m + (day - m.second_bore_lag_days) * m.tbm_advance_rate_m_per_day
        rows.append({
            "day": day,
            "bore1_chainage_m": bore1,
            "bore2_chainage_m": bore2,
            "active_hours_day": 24 * m.active_duty_factor,
            "week_equivalent": day / 7,
        })
    return pd.DataFrame(rows)


def receiver_day_row(m: ModelInputs, ts_row: pd.Series, r: Receiver) -> dict:
    long1 = ts_row["bore1_chainage_m"] - r.chainage_m
    long2 = ts_row["bore2_chainage_m"] - r.chainage_m
    lat1 = abs(r.lateral_offset_m - m.twin_tunnel_spacing_m / 2)
    lat2 = abs(r.lateral_offset_m + m.twin_tunnel_spacing_m / 2)
    vert = m.tunnel_axis_depth_m + r.floor_height_m
    d1 = sqrt(long1**2 + lat1**2 + vert**2)
    d2 = sqrt(long2**2 + lat2**2 + vert**2)

    s1 = m.smax_per_bore_mm * exp(-(long1**2) / (2 * m.longitudinal_sigma_m**2)) * exp(-(lat1**2) / (2 * m.trough_width_m**2))
    s2 = m.smax_per_bore_mm * exp(-(long2**2) / (2 * m.longitudinal_sigma_m**2)) * exp(-(lat2**2) / (2 * m.trough_width_m**2))
    settlement = s1 + s2
    diff_settlement = settlement * min(1.0, (r.footprint_length_m / (2 * m.trough_width_m)) if r.footprint_length_m else 0.0)
    angular_distortion = diff_settlement / (r.footprint_length_m * 1000) if r.footprint_length_m else 0.0

    ppv1 = m.ppv_ref_mm_s * (d1 / m.ref_distance_m) ** (-m.ppv_geom_exponent) * exp(-m.ppv_damping_beta * (d1 - m.ref_distance_m))
    ppv2 = m.ppv_ref_mm_s * (d2 / m.ref_distance_m) ** (-m.ppv_geom_exponent) * exp(-m.ppv_damping_beta * (d2 - m.ref_distance_m))
    ppv = max(ppv1, ppv2)

    weighted_acc = ppv / 1000 * 2 * pi * m.dominant_frequency_hz / sqrt(2)
    vdv15 = (weighted_acc**4 * m.active_seconds_15min) ** 0.25

    gbn1 = m.gbn_ref_dba - 20 * log10(d1 / m.ref_distance_m) - m.gbn_damping_db_per_m * (d1 - m.ref_distance_m) + m.room_gain_db + 10 * log10(m.active_duty_factor)
    gbn2 = m.gbn_ref_dba - 20 * log10(d2 / m.ref_distance_m) - m.gbn_damping_db_per_m * (d2 - m.ref_distance_m) + m.room_gain_db + 10 * log10(m.active_duty_factor)
    gbn = 10 * log10(10 ** (gbn1 / 10) + 10 ** (gbn2 / 10))

    spec1 = m.spec_ref_um_s * (d1 / m.ref_distance_m) ** (-m.spec_geom_exponent) * exp(-m.spec_damping_beta * (d1 - m.ref_distance_m)) * r.floor_amp
    spec2 = m.spec_ref_um_s * (d2 / m.ref_distance_m) ** (-m.spec_geom_exponent) * exp(-m.spec_damping_beta * (d2 - m.ref_distance_m)) * r.floor_amp
    spec = sqrt(spec1**2 + spec2**2)

    gbn_crit = r.gbn_criterion(m)
    ppv_crit = r.ppv_criterion(m)
    vdv_crit = r.vdv_criterion(m)
    spec_crit = r.spectral_criterion(m)

    risk = "GREEN"
    if gbn > gbn_crit or ppv > ppv_crit or vdv15 > vdv_crit or (spec_crit > 0 and spec > spec_crit):
        risk = "RED"
    elif gbn > 0.9 * gbn_crit or ppv > 0.9 * ppv_crit or vdv15 > 0.9 * vdv_crit or (spec_crit > 0 and spec > 0.9 * spec_crit):
        risk = "AMBER"

    return {
        "day": int(ts_row["day"]),
        "receiver_id": r.receiver_id,
        "receiver_name": r.receiver_name,
        "zone": r.zone,
        "bore1_chainage_m": ts_row["bore1_chainage_m"],
        "bore2_chainage_m": ts_row["bore2_chainage_m"],
        "receiver_chainage_m": r.chainage_m,
        "longitudinal_1_m": long1,
        "longitudinal_2_m": long2,
        "lateral_1_m": lat1,
        "lateral_2_m": lat2,
        "vertical_sep_m": vert,
        "distance_1_m": d1,
        "distance_2_m": d2,
        "settlement_1_mm": s1,
        "settlement_2_mm": s2,
        "settlement_total_mm": settlement,
        "diff_settlement_mm": diff_settlement,
        "angular_distortion": angular_distortion,
        "ppv_1_mm_s": ppv1,
        "ppv_2_mm_s": ppv2,
        "ppv_total_mm_s": ppv,
        "weighted_acc_m_s2": weighted_acc,
        "vdv_15min": vdv15,
        "gbn_1_dba": gbn1,
        "gbn_2_dba": gbn2,
        "gbn_total_dba": gbn,
        "spec_1_um_s": spec1,
        "spec_2_um_s": spec2,
        "spec_total_um_s": spec,
        "gbn_criterion_dba": gbn_crit,
        "ppv_criterion_mm_s": ppv_crit,
        "vdv_criterion": vdv_crit,
        "spec_criterion_um_s": spec_crit,
        "gbn_exceed": int(gbn > gbn_crit),
        "ppv_exceed": int(ppv > ppv_crit),
        "vdv_exceed": int(vdv15 > vdv_crit),
        "spec_exceed": int(spec_crit > 0 and spec > spec_crit),
        "risk": risk,
    }


def run_model(m: ModelInputs, receivers: List[Receiver]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = build_time_steps(m)
    detail = []
    for _, ts_row in ts.iterrows():
        for receiver in receivers:
            detail.append(receiver_day_row(m, ts_row, receiver))
    detail_df = pd.DataFrame(detail)

    summary = (
        detail_df.groupby(["receiver_id", "receiver_name", "zone"], as_index=False)
        .agg(
            peak_settlement_mm=("settlement_total_mm", "max"),
            peak_diff_settlement_mm=("diff_settlement_mm", "max"),
            peak_ppv_mm_s=("ppv_total_mm_s", "max"),
            peak_vdv=("vdv_15min", "max"),
            peak_gbn_dba=("gbn_total_dba", "max"),
            peak_spec_um_s=("spec_total_um_s", "max"),
            gbn_criterion_dba=("gbn_criterion_dba", "max"),
            ppv_criterion_mm_s=("ppv_criterion_mm_s", "max"),
            vdv_criterion=("vdv_criterion", "max"),
            spec_criterion_um_s=("spec_criterion_um_s", "max"),
            days_gbn_exceed=("gbn_exceed", "sum"),
            days_ppv_exceed=("ppv_exceed", "sum"),
            days_vdv_exceed=("vdv_exceed", "sum"),
            days_spec_exceed=("spec_exceed", "sum"),
        )
    )
    ratios = []
    risk_flags = []
    actions = []
    for _, row in summary.iterrows():
        ratio = max(
            row["peak_gbn_dba"] / row["gbn_criterion_dba"],
            row["peak_ppv_mm_s"] / row["ppv_criterion_mm_s"],
            row["peak_vdv"] / row["vdv_criterion"],
            (row["peak_spec_um_s"] / row["spec_criterion_um_s"]) if row["spec_criterion_um_s"] else 0.0,
        )
        ratios.append(ratio)
        risk = "GREEN"
        if max(row["days_gbn_exceed"], row["days_ppv_exceed"], row["days_vdv_exceed"], row["days_spec_exceed"]) > 0:
            risk = "RED"
        elif ratio >= 0.90:
            risk = "AMBER"
        risk_flags.append(risk)
        actions.append(
            "Mitigation and monitoring plan required" if risk == "RED"
            else "Close monitoring / scenario refinement" if risk == "AMBER"
            else "Routine baseline monitoring"
        )
    summary["governing_ratio"] = ratios
    summary["overall_risk"] = risk_flags
    summary["recommended_action"] = actions
    return detail_df, summary


def preview_plot(detail_df: pd.DataFrame, summary_df: pd.DataFrame, output_png: Path) -> None:
    focus = "R04"
    r04 = detail_df[detail_df["receiver_id"] == focus].copy()

    plt.figure(figsize=(13, 8))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(r04["day"], r04["ppv_total_mm_s"])
    ax1.axhline(r04["ppv_criterion_mm_s"].iloc[0], linestyle="--")
    ax1.set_title("Microscopy suite proxy - PPV")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("mm/s")

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(r04["day"], r04["gbn_total_dba"])
    ax2.axhline(r04["gbn_criterion_dba"].iloc[0], linestyle="--")
    ax2.set_title("Microscopy suite proxy - GBN")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("dBA")

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(r04["day"], r04["spec_total_um_s"])
    if r04["spec_criterion_um_s"].iloc[0] > 0:
        ax3.axhline(r04["spec_criterion_um_s"].iloc[0], linestyle="--")
    ax3.set_title("Microscopy suite proxy - spectral RMS")
    ax3.set_xlabel("Day")
    ax3.set_ylabel("um/s")

    ax4 = plt.subplot(2, 2, 4)
    ranked = summary_df.sort_values("governing_ratio", ascending=False)
    ax4.bar(ranked["receiver_id"], ranked["governing_ratio"])
    ax4.axhline(1.0, linestyle="--")
    ax4.set_title("Governing ratio by receiver")
    ax4.set_xlabel("Receiver")
    ax4.set_ylabel("Peak / criterion")

    plt.tight_layout()
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results", help="Directory for CSVs and preview plot")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    m = ModelInputs()
    detail_df, summary_df = run_model(m, DEFAULT_RECEIVERS)

    detail_df.to_csv(out / "monash_srl_detail.csv", index=False)
    summary_df.to_csv(out / "monash_srl_summary.csv", index=False)
    preview_plot(detail_df, summary_df, out / "monash_srl_preview.png")

    print("Wrote:")
    print(out / "monash_srl_detail.csv")
    print(out / "monash_srl_summary.csv")
    print(out / "monash_srl_preview.png")


if __name__ == "__main__":
    main()
