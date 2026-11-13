"""Page: settlement and vibration as a TBM passes beneath buildings.

The calculations are in :mod:`geo_platform.vibration`; this module only reads
the inputs, runs the model and shows the results.
"""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pandas as pd
import streamlit as st

from geo_platform import report, style, ui
from geo_platform import vibration as vib

PAGE = "vib"

# (field, label, min, max, step, format, help)
_TUNNEL = [
    ("start_chainage_m", "Launch chainage, x0 (m)", -2000.0, 2000.0, 10.0, "%.1f", None),
    ("advance_rate_m_per_day", "Advance rate, v (m/day)", 0.0, 200.0, 1.0, "%.2f", None),
    ("active_duty_factor", "Boring fraction of time, delta (-)", 0.05, 1.0, 0.05, "%.2f",
     "Share of the time the cutterhead is turning; the rest is ring build and stoppages."),
    ("tunnel_axis_depth_m", "Axis depth, z0 (m)", 2.0, 100.0, 1.0, "%.1f", None),
    ("tunnel_diameter_m", "Excavated diameter, D (m)", 2.0, 20.0, 0.1, "%.2f", None),
    ("twin_tunnel_spacing_m", "Bore spacing, s (m)", 0.0, 100.0, 1.0, "%.1f",
     "Centre-to-centre; bore 1 at +s/2 and bore 2 at -s/2 from the centreline."),
    ("second_bore_lag_days", "Second bore lag, t_lag (days)", 0.0, 365.0, 1.0, "%.1f",
     "The second machine is launched this many days after the first."),
    ("model_days", "Days modelled (-)", 1, 730, 1, "%d", None),
    ("substeps_per_day", "Sub-steps per day (-)", 1, 240, 1, "%d",
     "Positions per day at which peaks are taken and the dose accumulated."),
]
_GROUND = [
    ("trough_width_K", "Trough width parameter, K (-)", 0.1, 1.0, 0.05, "%.2f",
     "i = K z0, used both across and along the drive."),
    ("volume_loss_pct", "Volume loss, V_L (%)", 0.0, 5.0, 0.1, "%.2f",
     "Per cent of the excavated area; 0.5 per cent enters as 0.005."),
]
_VIBRATION = [
    ("ppv_ref_mm_s", "Reference PPV, PPV_ref (mm/s)", 0.001, 100.0, 0.05, "%.3f", None),
    ("ref_distance_m", "Reference distance, d_ref (m)", 1.0, 100.0, 1.0, "%.1f", None),
    ("ppv_geom_exponent", "Geometric exponent, n (-)", 0.0, 3.0, 0.05, "%.2f",
     "0.5 for surface waves, 1.0 for body waves from a point source."),
    ("ppv_damping_per_m", "Material damping, beta (1/m)", 0.0, 0.2, 0.001, "%.4f",
     "beta = pi f eta / c, with eta the loss factor and c the wave speed."),
    ("dominant_frequency_hz", "Dominant frequency, f (Hz)", 1.0, 200.0, 1.0, "%.1f", None),
    ("crest_factor", "Crest factor, C = peak/RMS (-)", 1.0, 6.0, 0.1, "%.2f",
     "1.41 for a sinusoid; larger for a more impulsive signal. The estimated VDV holds only for "
     "steady vibration with a crest factor below about 6; above that, integrate a measured record."),
    ("frequency_weighting", "Frequency weighting at f, W (-)", 0.01, 1.0, 0.05, "%.2f",
     "Human-response weighting factor at the dominant frequency: W_b for vertical vibration "
     "(most sensitive at 4-12.5 Hz), W_d for horizontal (1-2 Hz). 1.0 is unweighted and conservative."),
    ("vdv_period_h", "VDV assessment period, T_p (h)", 1.0, 24.0, 1.0, "%.1f",
     "16 h for the daytime criteria, 8 h for night-time."),
    ("gbn_ref_dba", "Reference noise level, L_ref (dBA)", 0.0, 100.0, 1.0, "%.1f",
     "Level radiated when the room surfaces move at PPV_ref, continuous boring, no room gain."),
    ("room_gain_db", "Room gain, G_room (dB)", -10.0, 20.0, 0.5, "%.1f", None),
    ("spec_ref_um_s", "Reference band velocity, V_ref (um/s)", 0.01, 1000.0, 1.0, "%.2f",
     "One-third-octave RMS velocity in the dominant band at d_ref."),
    ("spec_geom_exponent", "Band geometric exponent, n_s (-)", 0.0, 3.0, 0.05, "%.2f", None),
    ("spec_damping_per_m", "Band material damping, beta_s (1/m)", 0.0, 0.2, 0.001, "%.4f", None),
]
_CRITERIA = [
    ("settlement_limit_mm", "Settlement screen (mm)", 0.1, 100.0, 1.0, "%.1f", None),
    ("slope_limit_one_in", "Slope screen, 1 in (-)", 10.0, 10000.0, 50.0, "%.0f", None),
    ("ppv_ordinary_mm_s", "PPV, ordinary structure (mm/s)", 0.1, 100.0, 0.5, "%.2f", None),
    ("ppv_heritage_mm_s", "PPV, heritage structure (mm/s)", 0.1, 100.0, 0.5, "%.2f", None),
    ("vdv_residential", "VDV, residential (m/s^1.75)", 0.01, 5.0, 0.05, "%.2f",
     "BS 6472-1, 16 h day: adverse comment has a low probability at 0.2-0.4, is possible at 0.4-0.8 "
     "and is probable at 0.8-1.6. The default 0.4 is the top of the low-probability band."),
    ("vdv_teaching", "VDV, teaching and office (m/s^1.75)", 0.01, 5.0, 0.05, "%.2f",
     "Offices take twice the residential values (0.4-0.8 low probability, 16 h day); the default 0.8 "
     "is the top of that band. Teaching space is treated as office here, a judgement."),
    ("gbn_residential_dba", "Noise, residential (dBA)", 10.0, 80.0, 1.0, "%.1f", None),
    ("gbn_teaching_dba", "Noise, teaching and office (dBA)", 10.0, 80.0, 1.0, "%.1f", None),
    ("spec_L1_um_s", "Equipment class L1 (um/s)", 0.1, 500.0, 1.0, "%.2f", None),
    ("spec_L2_um_s", "Equipment class L2 (um/s)", 0.1, 500.0, 1.0, "%.2f", None),
    ("spec_L3_um_s", "Equipment class L3 (um/s)", 0.1, 500.0, 0.5, "%.2f", None),
    ("near_fraction", "Close-to-criterion band from (-)", 0.5, 1.0, 0.05, "%.2f",
     "A ratio of demand to criterion at or above this is flagged as close to the criterion."),
]

_ASSUMPTIONS = [
    "Screening model with placeholder parameters. Every attenuation constant, reference level, "
    "floor amplification and criterion is illustrative and must be replaced by a site "
    "regression from monitored data and by the project's own criteria before any decision.",
    "Settlement: Gaussian transverse trough with i = K z0, and the cumulative normal along the "
    "drive (Attewell and Woodman 1982): half the final settlement at the face and the full value "
    "well behind it, with the same i in both directions. Greenfield, no building stiffness; "
    "the two bores superpose.",
    "The footprint is a line of length L centred on the receiver; as the orientation is not an "
    "input, it is laid both along and across the drive and the worse case kept. The 10 mm and "
    "1 in 500 screens are the first stage of the damage assessment of Chapter 26, not damage limits.",
    "Vibration source at the face, at axis depth; slant distance to ground level. "
    "PPV = PPV_ref (d/d_ref)^-n exp(-beta (d - d_ref)) at the foundation, with no floor "
    "amplification; the larger of the two bores governs.",
    "VDV, noise and the equipment spectrum are floor quantities and include the floor "
    "amplification. VDV is the estimated value 1.4 a_rms t^1/4 with "
    "a_rms = W 2 pi f A PPV / C, accumulated over the boring seconds of the assessment period "
    "with a^4 averaged over the positions in the worst window of length T_p ending in each day; "
    "bores add as fourth powers. The estimated VDV of "
    "BS 6472-1 applies to steady vibration with a crest factor below about 6; for impulsive or "
    "intermittent vibration it underestimates, and the VDV must be integrated from a measured "
    "acceleration record.",
    "VDV criteria (BS 6472-1, 16 h day): the defaults are the upper end of the band in which adverse "
    "comment has a low probability, 0.4 m/s^1.75 residential and 0.8 m/s^1.75 office (multiplying "
    "factor 2); teaching space is treated as office.",
    "Ground-borne noise is an equivalent continuous level over the period and tracks the floor "
    "velocity dB for dB, so it attenuates by the same law as the vibration. The margin to the "
    "criterion is a level difference, shown as the amplitude ratio 10^(dL/20).",
    "Equipment criteria are one-third-octave RMS velocities in the manner of the generic "
    "vibration criterion curves; the band velocity follows its own attenuation law and the two "
    "bores add as the root sum of squares.",
    "Human-response criteria do not apply to unoccupied receivers; settlement and slope screens "
    "do not apply to a point receiver (footprint 0).",
]


def _number_inputs(spec, defaults: vib.VibrationInputs, columns: int = 3) -> dict:
    values = {}
    cols = st.columns(columns)
    for k, (name, label, lo, hi, step, fmt, help_text) in enumerate(spec):
        default = getattr(defaults, name)
        with cols[k % columns]:
            if isinstance(lo, int) and isinstance(step, int):
                values[name] = int(st.number_input(label, min_value=lo, max_value=hi, value=int(default),
                                                   step=step, key=f"{PAGE}_{name}", help=help_text))
            else:
                values[name] = float(st.number_input(label, min_value=float(lo), max_value=float(hi),
                                                     value=float(default), step=float(step), format=fmt,
                                                     key=f"{PAGE}_{name}", help=help_text))
    return values


def _inputs() -> vib.VibrationInputs:
    defaults = vib.VibrationInputs()
    values: dict = {}
    with st.expander("Tunnel and drive", expanded=True):
        values.update(_number_inputs(_TUNNEL, defaults))
    with st.expander("Ground and settlement", expanded=False):
        values.update(_number_inputs(_GROUND, defaults, columns=2))
        st.caption("The longitudinal spread uses the same i = K z0 as the transverse trough; "
                   "the source script's separate longitudinal sigma has no basis and is not used.")
    with st.expander("Vibration and noise attenuation", expanded=False):
        values.update(_number_inputs(_VIBRATION, defaults))
    with st.expander("Criteria", expanded=False):
        values.update(_number_inputs(_CRITERIA, defaults))
    names = {f.name for f in fields(vib.VibrationInputs)}
    return vib.VibrationInputs(**{k: v for k, v in values.items() if k in names})


def _receiver_editor() -> list[vib.Receiver]:
    st.subheader("Receivers")
    st.caption("Chainage along the drive; lateral offset from the centreline between the bores; "
               "footprint 0 for a point. Floor amplification is the ratio of the receiving floor's "
               "velocity to the foundation's. Add or delete rows as needed.")
    base = vib.receivers_to_frame(vib.DEFAULT_RECEIVERS)
    edited = st.data_editor(
        base,
        key=f"{PAGE}_receivers",
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "receiver_id": st.column_config.TextColumn("Id", required=True),
            "receiver_name": st.column_config.TextColumn("Name"),
            "zone": st.column_config.TextColumn("Zone"),
            "chainage_m": st.column_config.NumberColumn("Chainage (m)", format="%.1f", required=True),
            "lateral_offset_m": st.column_config.NumberColumn("Offset (m)", format="%.1f"),
            "footprint_length_m": st.column_config.NumberColumn("Footprint, L (m)", min_value=0.0, format="%.1f"),
            "occupancy": st.column_config.SelectboxColumn("Occupancy", options=list(vib.OCCUPANCIES)),
            "structure_class": st.column_config.SelectboxColumn("Structure", options=list(vib.STRUCTURE_CLASSES)),
            "equipment_class": st.column_config.SelectboxColumn("Equipment", options=list(vib.EQUIPMENT_CLASSES)),
            "floor_amp": st.column_config.NumberColumn("Floor amplification, A (-)", min_value=0.0, format="%.2f"),
        },
    )
    return vib.receivers_from_frame(pd.DataFrame(edited))


def _summary_table(summary: pd.DataFrame) -> None:
    show = pd.DataFrame({
        "Id": summary["receiver_id"],
        "Name": summary["receiver_name"],
        "Settlement (mm)": summary["final_settlement_mm"],
        "Differential (mm)": summary["max_diff_settlement_mm"],
        "Slope, 1 in": 1.0 / summary["max_slope"].where(summary["max_slope"] > 0),
        "PPV (mm/s)": summary["peak_ppv_mm_s"],
        "VDV (m/s^1.75)": summary["peak_vdv_m_s175"],
        "Noise (dBA)": summary["peak_gbn_dba"],
        "Spectral (um/s)": summary["peak_spec_um_s"],
        "Governing ratio": summary["governing_ratio"],
        "Governs": summary["governing_quantity"],
        "Status": summary["status"],
        "Action": summary["recommended_action"],
    })
    value_to_ratio = {
        "Settlement (mm)": "settlement_ratio", "Slope, 1 in": "slope_ratio", "PPV (mm/s)": "ppv_ratio",
        "VDV (m/s^1.75)": "vdv_ratio", "Noise (dBA)": "gbn_ratio", "Spectral (um/s)": "spec_ratio",
        "Governing ratio": "governing_ratio",
    }
    exceed_css = f"background-color: {style.CARMINE_TINT}; color: {style.CARMINE_DARK}; font-weight: 600"
    near_css = f"color: {style.CARMINE}"
    within_css = f"color: {style.MALACHITE}"
    near = float(st.session_state.get(f"{PAGE}_near_fraction", vib.VibrationInputs().near_fraction))

    def css_for(ratio: float) -> str:
        if ratio is None or not np.isfinite(ratio):
            return f"color: {style.GREY}"
        if ratio > 1.0:
            return exceed_css
        if ratio >= near:
            return near_css
        return within_css

    def row_styles(row: pd.Series) -> list[str]:
        idx = row.name
        out = []
        for col in show.columns:
            if col in value_to_ratio:
                out.append(css_for(float(summary.loc[idx, value_to_ratio[col]])))
            elif col == "Status":
                out.append(css_for(float(summary.loc[idx, "governing_ratio"])))
            else:
                out.append("")
        return out

    styler = (
        show.style.apply(row_styles, axis=1)
        .format({
            "Settlement (mm)": "{:.2f}", "Differential (mm)": "{:.2f}",
            "Slope, 1 in": lambda v: "--" if not np.isfinite(v) else (">100,000" if v > 1e5 else f"{v:,.0f}"),
            "PPV (mm/s)": "{:.3f}", "VDV (m/s^1.75)": "{:.3f}", "Noise (dBA)": "{:.1f}",
            "Spectral (um/s)": "{:.2f}", "Governing ratio": "{:.2f}",
        }, na_rep="--")
    )
    st.dataframe(styler, width="stretch", hide_index=True, height=35 * (len(show) + 1) + 3)
    st.caption("Shaded carmine: the peak exceeds its criterion. Carmine text: close to it. "
               "Malachite: within. Grey: no criterion applies. The noise ratio is 10^((L - L_crit)/20).")


def _ratio_chart(summary: pd.DataFrame) -> None:
    import plotly.graph_objects as go

    ratio = summary["governing_ratio"].fillna(0.0)
    colours = [style.CARMINE if r > 1.0 else style.MALACHITE for r in ratio]
    fig = go.Figure(go.Bar(
        x=summary["receiver_id"], y=ratio, marker_color=colours,
        customdata=np.stack([summary["receiver_name"], summary["governing_quantity"]], axis=-1),
        hovertemplate="%{x} %{customdata[0]}<br>Ratio %{y:.2f} (%{customdata[1]})<extra></extra>",
    ))
    ref = style.reference_style("plotly")
    fig.add_hline(y=1.0, line=dict(color=ref["color"], dash=ref["dash"], width=1))
    fig.update_layout(**style.plotly_layout(
        height=320, showlegend=False,
        xaxis=dict(style.plotly_layout()["xaxis"], title=style.axis_label("Receiver")),
        yaxis=dict(style.plotly_layout()["yaxis"],
                   title=style.axis_label("Governing ratio, demand/criterion", None, "-")),
    ))
    ui.show_plotly(fig)


_HISTORY = [
    ("settlement_mm", "settlement_criterion_mm", "Settlement", "S", "mm"),
    ("slope", "slope_criterion", "Slope", "S'", "-"),
    ("ppv_mm_s", "ppv_criterion_mm_s", "Peak particle velocity", "PPV", "mm/s"),
    ("vdv_m_s175", "vdv_criterion_m_s175", "Vibration dose value", "VDV", "m/s<sup>1.75</sup>"),
    ("gbn_dba", "gbn_criterion_dba", "Ground-borne noise", "L<sub>Aeq</sub>", "dBA"),
    ("spec_um_s", "spec_criterion_um_s", "Band velocity", "V<sub>1/3</sub>", "&#956;m/s"),
]


def _history_chart(detail: pd.DataFrame, receiver_id: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = detail[detail["receiver_id"] == receiver_id]
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, vertical_spacing=0.08, horizontal_spacing=0.12)
    ref = style.reference_style("plotly")
    for k, (col, crit_col, name, sym, unit) in enumerate(_HISTORY):
        r, c = k // 2 + 1, k % 2 + 1
        y = rows[col].to_numpy(dtype=float)
        crit = float(rows[crit_col].iloc[0]) if len(rows) else math.nan
        fig.add_trace(go.Scatter(x=rows["day"], y=y, mode="lines", line=dict(color=style.MALACHITE, width=1.6),
                                 name=name, hovertemplate=f"Day %{{x}}<br>{name} %{{y:.4g}}<extra></extra>"),
                      row=r, col=c)
        if np.isfinite(crit):
            over = y > crit
            if over.any():
                fig.add_trace(go.Scatter(x=rows["day"][over], y=y[over], mode="markers",
                                         marker=dict(color=style.CARMINE, size=6), name=f"{name} exceeds",
                                         hovertemplate="Day %{x}: exceeds<extra></extra>"), row=r, col=c)
            fig.add_hline(y=crit, line=dict(color=ref["color"], dash=ref["dash"], width=1), row=r, col=c)
        fig.update_yaxes(title_text=style.axis_label(name, sym, unit), row=r, col=c)
        if col == "gbn_dba":
            # far from the face the level falls below audibility; show the useful range only
            top = np.nanmax(np.append(y, crit if np.isfinite(crit) else np.nan))
            if np.isfinite(top):
                fig.update_yaxes(range=[max(0.0, top - 40.0), top + 5.0], row=r, col=c)
    axis = style.plotly_layout()["xaxis"]
    fig.update_layout(**style.plotly_layout(height=760, showlegend=False))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**{k: v for k, v in axis.items()})
    for c in (1, 2):
        fig.update_xaxes(title_text=style.axis_label("Day", "t", "d"), row=3, col=c)
    ui.show_plotly(fig)
    st.caption("Malachite: the daily value (peak within the day; VDV over the assessment period; "
               "settlement at the end of the day). Carmine markers: days above the criterion. "
               "Grey dashed: the criterion, where one applies.")


def _profile_figure(m: vib.VibrationInputs) -> None:
    import matplotlib.pyplot as plt

    i = m.trough_width_m
    x = np.linspace(-4.0 * i, 4.0 * i, 401)          # face position past the point
    from scipy.special import ndtr

    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.plot(x / i, ndtr(x / i), color=style.MALACHITE, label="Cumulative normal (used)")
        ref = style.reference_style()
        ax.plot(x / i, np.exp(-0.5 * (x / 35.0) ** 2), color=ref["color"], linestyle=ref["linestyle"],
                label="Gaussian in the face distance, sigma = 35 m (source script)")
        ax.axvline(0.0, color=style.GREY_LIGHT, linewidth=0.8)
        ax.set_xlabel(style.axis_label("Face position past the point", r"(x_f - x)/i", "-", math=True))
        ax.set_ylabel(style.axis_label("Settlement ratio", r"S/S_{\max}", "-", math=True))
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
        col, _ = st.columns([3, 2])
        with col:
            ui.show_matplotlib(fig)


def _ratio_figure(summary: pd.DataFrame):
    import matplotlib.pyplot as plt

    ratio = summary["governing_ratio"].fillna(0.0).to_numpy(dtype=float)
    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        ax.bar(summary["receiver_id"].astype(str), ratio,
               color=[style.CARMINE if r > 1.0 else style.MALACHITE for r in ratio])
        ref = style.reference_style()
        ax.axhline(1.0, color=ref["color"], linestyle=ref["linestyle"], linewidth=1.0)
        ax.set_xlabel(style.axis_label("Receiver"))
        ax.set_ylabel(style.axis_label("Governing ratio, demand/criterion", None, "-"))
    return fig


def _split_label(label: str) -> tuple[str, str]:
    """'Advance rate, v (m/day)' -> ('Advance rate, v', 'm/day')."""
    if label.endswith(")") and " (" in label:
        head, unit = label.rsplit(" (", 1)
        return head, unit[:-1]
    return label, ""


VIBRATION_EQUATIONS = [
    r"S=\sum_j S_{max}\,e^{-(y-y_j)^2/2i^2}\left[\Phi\!\left(\frac{x_{f,j}-x}{i}\right)"
    r"-\Phi\!\left(\frac{x_0-x}{i}\right)\right],\qquad i=Kz_0",
    r"\mathrm{PPV}=\mathrm{PPV}_{ref}\left(\frac{d}{d_{ref}}\right)^{-n}e^{-\beta\,(d-d_{ref})}",
    r"\mathrm{VDV}=1.4\left(\overline{\textstyle\sum_j a_{rms,j}^4}\;t\right)^{1/4},\qquad "
    r"a_{rms,j}=\frac{W\,2\pi f\,A\,\mathrm{PPV}_j}{C},\qquad t=3600\,T_p\,\delta",
    r"L=L_{ref}+20\log_{10}\frac{A\,\mathrm{PPV}}{\mathrm{PPV}_{ref}}+G_{room}+10\log_{10}\delta",
]


def _record(m: vib.VibrationInputs, summary: pd.DataFrame, n_exceed: int, n_near: int, gov) -> None:
    n = report.num
    inputs = []
    for spec in (_TUNNEL, _GROUND):
        for name, label, *_ in spec:
            if name == "substeps_per_day":
                continue
            quantity, unit = _split_label(label)
            value = getattr(m, name)
            inputs.append((quantity, "", f"{value:d}" if isinstance(value, int) else n(value, 2), unit))
    checks = []
    for _, row in summary.iterrows():
        ratio = float(row["governing_ratio"]) if pd.notna(row["governing_ratio"]) else math.nan
        if math.isfinite(ratio):
            checks.append((f"{row['receiver_id']} {row['receiver_name']}: governed by {row['governing_quantity']}, "
                           f"demand / criterion {ratio:.2f}.", ratio <= 1.0))
    table = pd.DataFrame({
        "Id": summary["receiver_id"], "Name": summary["receiver_name"],
        "S (mm)": summary["final_settlement_mm"], "PPV (mm/s)": summary["peak_ppv_mm_s"],
        "VDV": summary["peak_vdv_m_s175"], "Noise (dBA)": summary["peak_gbn_dba"],
        "Band (μm/s)": summary["peak_spec_um_s"], "Ratio": summary["governing_ratio"],
        "Governs": summary["governing_quantity"],
    })
    report.record(st.session_state, report.Section(
        key="vibration", title="Construction vibration", chapters="Chapters 17, 22 and 26",
        purpose="Settlement, slope, peak particle velocity, vibration dose value, ground-borne noise and band "
                "velocity at each receiver as one or two tunnel boring machines pass, each against its "
                "criterion. A screening model with placeholder attenuation constants and criteria.",
        inputs=inputs, equations=VIBRATION_EQUATIONS,
        results=[("Maximum settlement per bore", "S_{max}", n(m.smax_per_bore_mm, 2), "mm"),
                 ("Receivers exceeding a criterion", "", f"{n_exceed} of {len(summary)}", ""),
                 ("Receivers close to a criterion", "", f"{n_near}", ""),
                 ("Governing receiver", "", f"{gov['receiver_id']}", ""),
                 ("Governing ratio, demand / criterion", "", n(gov["governing_ratio"], 2), "-")],
        checks=checks,
        tables=[("Peak values at each receiver (VDV in m/s^1.75); ratio is the governing demand / criterion.",
                 report.text_frame(table, 2))],
        figures=[("ratio", "Governing ratio of demand to criterion at each receiver.", lambda: _ratio_figure(summary))],
        assumptions=_ASSUMPTIONS,
        references=["Attewell PB, Woodman JP (1982) Predicting the dynamics of ground settlement and its derivatives "
                    "caused by tunnelling in soil. Ground Eng 15(8):13-22",
                    "BS 6472-1:2008 Guide to evaluation of human exposure to vibration in buildings. Part 1: "
                    "Vibration sources other than blasting. British Standards Institution, London"]))


def render_vibration(ctx: dict) -> None:
    ui.page_header(
        "TBM passage: settlement and vibration",
        "Chapters 17 and 26 (ground movement); Chapter 22 (vibration)",
        "Day by day, as one or two tunnel boring machines pass beneath a set of receivers: "
        "settlement, differential settlement and slope across each footprint, peak particle "
        "velocity at the foundation, vibration dose value, ground-borne noise and the band "
        "velocity that governs sensitive equipment, each against its criterion.",
    )
    st.warning("Screening model with placeholder parameters. The attenuation constants, reference "
               "levels, floor amplifications and criteria are illustrative; replace them with a site "
               "regression from monitored data and the project's own criteria before relying on a result.")
    ui.assumptions(_ASSUMPTIONS)

    m = _inputs()
    receivers = _receiver_editor()
    try:
        detail, summary = vib.run_model(m, receivers)
    except ValueError as exc:
        st.error(f"The model cannot run with these inputs: {exc}")
        return

    st.subheader("Results")
    n_exceed = int((summary["status"] == "Exceeds").sum())
    n_near = int((summary["status"] == "Close to criterion").sum())
    gov = summary.loc[summary["governing_ratio"].fillna(-1).idxmax()]
    ui.metrics([
        ("Maximum settlement per bore, S_max", ui.fmt(m.smax_per_bore_mm, 2, "mm"),
         "V_s/(i sqrt(2 pi)) with i = K z0, over the axis of one bore."),
        ("Receivers exceeding a criterion", f"{n_exceed} of {len(summary)}",
         f"{n_near} more are close to a criterion."),
        ("Governing receiver", f"{gov['receiver_id']}", f"{gov['receiver_name']}: {gov['governing_quantity']}"),
        ("Governing ratio", ui.fmt(gov["governing_ratio"], 2), "Peak demand over criterion."),
    ])
    ui.status_line(n_exceed == 0, "Every receiver is within its screening criteria.",
                   f"{n_exceed} receiver(s) exceed a screening criterion: see the shaded cells.")
    if bool(detail["spec_above_overall_rms"].any()):
        st.warning("At some receivers the one-third-octave band velocity exceeds the overall RMS velocity "
                   "implied by the PPV law and crest factor, which is not physically possible: the two "
                   "attenuation laws are inconsistent for these inputs.")

    _record(m, summary, n_exceed, n_near, gov)
    _summary_table(summary)
    _ratio_chart(summary)

    st.subheader("Time history at a receiver")
    ids = list(summary["receiver_id"])
    labels = {rid: f"{rid}  {name}" for rid, name in zip(summary["receiver_id"], summary["receiver_name"])}
    default_index = ids.index(gov["receiver_id"]) if gov["receiver_id"] in ids else 0
    focus = st.selectbox("Receiver", ids, index=default_index, format_func=lambda r: labels.get(r, r),
                         key=f"{PAGE}_focus")
    _history_chart(detail, focus)

    with st.expander("Longitudinal settlement profile", expanded=False):
        st.markdown(
            "Settlement along the drive follows the cumulative normal distribution "
            "(Attewell and Woodman 1982): "
            r"$S = S_{\max}\,e^{-(y-y_j)^2/2i^2}\left[\Phi\!\left((x_f - x)/i\right) - \Phi\!\left((x_0 - x)/i\right)\right]$ "
            "for each bore, with x_0 the launch chainage; about half the final value as the face passes and "
            "the full value well behind it. "
            "The source script used a Gaussian in the face distance, so settlement recovered to zero "
            "once the face had passed; settlement is permanent, and that form is wrong.")
        _profile_figure(m)

    st.subheader("Downloads")
    c1, c2 = st.columns(2)
    with c1:
        ui.download_table(summary, "tbm_vibration_summary.csv", "Download summary (CSV)", key=f"{PAGE}_dl_summary")
    with c2:
        ui.download_table(detail, "tbm_vibration_detail.csv", "Download daily detail (CSV)", key=f"{PAGE}_dl_detail")


__all__ = ["render_vibration"]
