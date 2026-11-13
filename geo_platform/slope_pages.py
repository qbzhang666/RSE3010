"""Rock-slope pages of the platform: SMR, kinematic screening with planar sliding,
and wedge and circular limit equilibrium.

Each ``render_*`` function draws one page; the calculations are in
:mod:`geo_platform.slopes`, where the corrections made to the course apps are
listed.  ``ctx`` (the shared ground model) is not used by these pages.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

from geo_platform import report, slopes, style, ui

DEFAULT_JOINTS = pd.DataFrame({
    "Joint": ["J1", "J2", "J3"],
    "Dip (deg)": [40.0, 55.0, 80.0],
    "Dip direction (deg)": [160.0, 110.0, 340.0],
})
DEFAULT_FACES = pd.DataFrame({"Face": ["S1"], "Dip (deg)": [60.0], "Dip direction (deg)": [170.0]})

_LINESTYLES = ("-", "--", ":", "-.")
_REF_ROMANA = 'Romana M (1985) New adjustment ratings for application of Bieniawski classification to slopes. In: Proceedings of the International Symposium on the Role of Rock Mechanics in Excavations for Mining and Civil Works, Zacatecas, Mexico. International Society for Rock Mechanics, pp 49-53'
_REF_HOEK_BRAY = 'Hoek E, Bray JW (1981) Rock slope engineering, 3rd edn. Institution of Mining and Metallurgy, London'
_REF_MARKLAND = 'Markland JT (1972) A useful technique for estimating the stability of rock slopes when the rigid wedge sliding type of failure is expected. Imperial College Rock Mechanics Research Report 19, London, 10 pp'
_REF_GOODMAN_BRAY = 'Goodman RE, Bray JW (1976) Toppling of rock slopes. In: Proceedings of the Specialty Conference on Rock Engineering for Foundations and Slopes, Boulder, Colorado, vol 2. ASCE, pp 201-234'
_REF_BISHOP = 'Bishop AW (1955) The use of the slip circle in the stability analysis of slopes. Géotechnique 5(1):7-17'
_REF_ANBALAGAN = ("Anbalagan R (1992) Landslide hazard evaluation and zonation mapping in mountainous terrain. "
                  "Eng Geol 32(4):269-277")
_REF_HOCKING = ("Hocking G (1976) A method for distinguishing between single and double plane sliding of "
                "tetrahedral wedges. Int J Rock Mech Min Sci Geomech Abstr 13(7):225-226")


def _plane_rows(planes, kind: str) -> list:
    return [(f"{kind} {name}: dip / dip direction", "", f"{dip:.0f} / {dd:03.0f}", "°") for name, dip, dd in planes]


_NETS = {"Equal-angle (Wulff)": slopes.EQUAL_ANGLE, "Equal-area (Schmidt)": slopes.EQUAL_AREA}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _plane_editor(default: pd.DataFrame, name_col: str, key: str) -> list[tuple[str, float, float]]:
    """Editable table of planes; returns clean (name, dip, dip direction) tuples."""
    edited = st.data_editor(
        default, key=key, num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            name_col: st.column_config.TextColumn(name_col),
            "Dip (deg)": st.column_config.NumberColumn("Dip (deg)", min_value=0.0, max_value=90.0,
                                                       step=1.0, format="%.0f"),
            "Dip direction (deg)": st.column_config.NumberColumn("Dip direction (deg)", min_value=0.0,
                                                                 max_value=360.0, step=1.0, format="%.0f"),
        },
    )
    planes: list[tuple[str, float, float]] = []
    for i, row in enumerate(pd.DataFrame(edited).itertuples(index=False), start=1):
        dip = pd.to_numeric(row[1], errors="coerce")
        dd = pd.to_numeric(row[2], errors="coerce")
        if pd.isna(dip) or pd.isna(dd):
            continue
        name = str(row[0]).strip() if row[0] is not None and str(row[0]).strip() not in ("", "nan", "None") \
            else f"{name_col[0]}{i}"
        planes.append((name, float(min(max(dip, 0.0), 90.0)), float(dd) % 360.0))
    return planes


def _net_axes(ax, title: str | None = None) -> None:
    """Primitive circle, ten-degree ticks and compass letters."""
    t = np.linspace(0.0, 2.0 * math.pi, 361)
    ax.plot(np.cos(t), np.sin(t), color=style.INK, lw=0.8)
    for az in range(0, 360, 10):
        a = math.radians(az)
        inner = 0.97 if az % 30 else 0.94
        ax.plot([inner * math.sin(a), math.sin(a)], [inner * math.cos(a), math.cos(a)],
                color=style.INK, lw=0.6)
    for text, (x, y) in {"N": (0, 1.08), "E": (1.08, 0), "S": (0, -1.08), "W": (-1.08, 0)}.items():
        ax.text(x, y, text, ha="center", va="center", fontsize=9, color=style.INK)
    ax.plot([0], [0], marker="+", color=style.GREY, ms=6)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)


def _fmt_deg(dip: float, dd: float) -> str:
    return f"{dip:.0f}/{dd:03.0f}"


def _tint_rows(df: pd.DataFrame, column: str, bad, good):
    """Row tints: carmine for ``bad`` values, malachite for ``good`` ones."""
    def colour(row):
        value = row[column]
        if value in bad:
            return [f"background-color: {style.CARMINE_TINT}"] * len(row)
        if value in good:
            return [f"background-color: {style.MALACHITE_TINT}"] * len(row)
        return [""] * len(row)
    return df.style.apply(colour, axis=1)


def _yes(value: bool) -> str:
    return "yes" if value else "no"


# ---------------------------------------------------------------------------
# Page 1: Slope Mass Rating
# ---------------------------------------------------------------------------


def render_smr(ctx: dict | None = None) -> None:
    """Slope Mass Rating (Romana) for planar, wedge and toppling modes, with a stereonet."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ui.page_header(
        "Slope Mass Rating",
        "Chapter 9 (structural analysis) and Chapter 10 (rock mass classification); RSE3010 Week 10",
        "Empirical screening of a rock slope: Romana's SMR adjusts the basic RMR for the orientation "
        "of each joint set (planar sliding, toppling) and each joint intersection (wedge sliding) "
        "relative to the face, and for the excavation method.",
    )
    notes = [
        "SMR = RMRb + F1 F2 F3 + F4 (Romana 1985); the wedge mode uses the line of intersection "
        "(trend, plunge) in place of the joint (Anbalagan 1992).",
        "Planar: A = |αj − αs|, B = βj, C = βj − βs.  Toppling: A = |αj − αs − 180°|, F2 = 1, "
        "C = βj + βs.  Wedge: A = |αi − αs|, B = βi, C = βi − βs.  α is a dip direction or trend, "
        "β a dip or plunge.",
        "F1 and F2 from Romana's table by default.  The continuous option uses F1 = (1 − sin A)² and "
        "F2 = tan² B bounded to the table's range 0.15 to 1.00; it is an approximation of the table.",
        "F3 'fair' (−25) applies only when C is exactly 0° for planar and wedge sliding, as in the table.",
        "RMRb is the basic RMR without the orientation adjustment for tunnels.",
        "SMR is a screening index: it says which mechanisms deserve a kinematic and limit-equilibrium "
        "check, not a factor of safety.",
    ]
    ui.assumptions(notes)

    c1, c2, c3, c4 = st.columns(4)
    rmr = c1.number_input("Basic rock mass rating, RMRb (-)", min_value=0.0, max_value=100.0, value=50.0,
                          step=1.0, key="smr_rmr")
    method = c2.selectbox("Excavation method (F4)", list(slopes.F4_METHODS), index=4, key="smr_f4")
    form = c3.selectbox("F1 and F2", ["Romana's table", "Continuous fit (approximate)"], key="smr_form")
    net_label = c4.selectbox("Stereonet", list(_NETS), key="smr_net")
    f4 = slopes.F4_METHODS[method]
    continuous = form.startswith("Continuous")
    net = _NETS[net_label]

    st.markdown("**Joint sets and slope faces** (dip / dip direction, degrees; rows can be added or removed)")
    cj, cf = st.columns([3, 2])
    with cj:
        joints = _plane_editor(DEFAULT_JOINTS, "Joint", "smr_joints")
    with cf:
        faces = _plane_editor(DEFAULT_FACES, "Face", "smr_faces")
    if not joints or not faces:
        st.warning("Enter at least one joint set and one slope face.")
        return

    rows = slopes.smr_table(joints, faces, rmr, f4, continuous)
    df = pd.DataFrame(rows)
    critical = df.loc[df["SMR"].idxmin()]
    n_bad = int((df["SMR"] <= 40.0).sum())
    ui.metrics([
        ("Lowest SMR", ui.fmt(critical["SMR"], 1)),
        ("Critical case", f"{critical['Mode']} {critical['Case']}"),
        ("Class", f"{critical['Class']} ({critical['Quality'].lower()})", critical["Stability"]),
        ("Cases in classes IV-V", f"{n_bad} of {len(df)}"),
    ])
    ui.status_line(critical["SMR"] > 40.0,
                   f"All cases are class III or better; lowest SMR {critical['SMR']:.1f}.",
                   f"{critical['Mode']} {critical['Case']} on face {critical['Face']} rates SMR "
                   f"{critical['SMR']:.1f}, class {critical['Class']}: {critical['Stability'].lower()}; "
                   f"support: {critical['Support'].lower()}.")

    n = report.num
    smr_table = df[["Face", "Mode", "Case", "F1", "F2", "F3", "F4", "SMR", "Class"]].copy()
    report.record(st.session_state, report.Section(
        key="smr", title="Slope mass rating", chapters="Chapters 9 and 10; RSE3010 Week 10",
        purpose=f"Romana's slope mass rating for planar, wedge and toppling modes on each slope face. Excavation "
                f"method: {method}. F1 and F2: {form}.",
        inputs=[("Basic rock mass rating", r"\mathrm{RMR}_b", n(rmr, 0), "-"),
                ("Adjustment for the excavation method", "F_4", f"{f4:g}", "-")]
               + _plane_rows(joints, "Joint") + _plane_rows(faces, "Face"),
        equations=[r"\mathrm{SMR}=\mathrm{RMR}_b+F_1F_2F_3+F_4"],
        results=[("Lowest slope mass rating", r"\mathrm{SMR}", n(critical["SMR"], 1), "-"),
                 ("Critical case", "", f"{critical['Mode']} {critical['Case']}", ""),
                 ("Class", "", f"{critical['Class']}", ""),
                 ("Cases in classes IV-V", "", f"{n_bad} of {len(df)}", "")],
        checks=[(f"Lowest SMR {critical['SMR']:.1f} on face {critical['Face']} ({critical['Mode'].lower()} "
                 f"{critical['Case']}) is above 40, class III or better.", bool(critical["SMR"] > 40.0))],
        tables=[("Slope mass rating of each case.", report.text_frame(smr_table, 2))],
        assumptions=notes, references=[_REF_ROMANA, _REF_ANBALAGAN]))

    modes = st.multiselect("Show modes", ["Planar", "Wedge", "Toppling"], default=["Planar", "Wedge", "Toppling"],
                           key="smr_modes")
    shown = df[df["Mode"].isin(modes)].copy()
    shown["B (deg)"] = shown["B (deg)"].map(lambda v: "-" if v is None or pd.isna(v) else f"{v:.1f}")
    columns = ["Face", "Mode", "Case", "A (deg)", "B (deg)", "C (deg)", "F1", "F2", "F3", "F1 F2 F3", "F4",
               "SMR", "Class", "Quality", "Stability", "Failures", "Support"]
    st.dataframe(
        _tint_rows(shown[columns], "Class", bad=("IV", "V"), good=("I", "II")).format(
            {"A (deg)": "{:.1f}", "C (deg)": "{:.1f}", "F1": "{:.2f}", "F2": "{:.2f}", "F3": "{:.0f}",
             "F1 F2 F3": "{:.2f}", "F4": "{:.0f}", "SMR": "{:.1f}"}),
        width="stretch", hide_index=True)
    st.caption("Rows tinted carmine are classes IV-V (unstable), malachite classes I-II (stable).")
    ui.download_table(df[columns], "smr_results.csv", key="smr_dl")

    # Stereonet
    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        _net_axes(ax)
        handles = []
        for i, (name, dip, dd) in enumerate(faces):
            x, y = slopes.great_circle(dip, dd, net)
            ls = _LINESTYLES[i % len(_LINESTYLES)]
            ax.plot(x, y, color=style.INK, lw=2.0, ls=ls)
            handles.append(Line2D([], [], color=style.INK, lw=2.0, ls=ls, label=f"Face {name} ({_fmt_deg(dip, dd)})"))
        worst = df.groupby("Case")["SMR"].min()
        for i, (name, dip, dd) in enumerate(joints):
            bad = float(min(worst.get(name, 100.0), 100.0)) <= 40.0
            colour = style.CARMINE if bad else style.MALACHITE
            ls = _LINESTYLES[i % len(_LINESTYLES)]
            x, y = slopes.great_circle(dip, dd, net)
            ax.plot(x, y, color=colour, lw=1.2, ls=ls)
            px, py = slopes.project(*slopes.pole(dip, dd), net)
            ax.plot(px, py, marker="s", color=colour, ms=5, ls="none")
            ax.annotate(name, (px, py), xytext=(4, 4), textcoords="offset points", fontsize=8, color=colour)
            handles.append(Line2D([], [], color=colour, lw=1.2, ls=ls, marker="s", ms=4,
                                  label=f"{name} ({_fmt_deg(dip, dd)}) and pole"))
        wedge_rows = df[(df["Mode"] == "Wedge") & (df["Face"] == faces[0][0])]
        for (n1, d1, a1), (n2, d2, a2) in combinations(joints, 2):
            line = slopes.intersection_line(d1, a1, d2, a2)
            if line is None:
                continue
            trend, plunge = line
            match = wedge_rows[wedge_rows["Case"].str.startswith(f"{n1}-{n2} ")]
            bad = (not match.empty) and float(match["SMR"].iloc[0]) <= 40.0
            colour = style.CARMINE if bad else style.MALACHITE
            ix, iy = slopes.project(trend, plunge, net)
            ax.plot(ix, iy, marker="^", color=colour, ms=6, ls="none")
            ax.annotate(f"{n1}-{n2}", (ix, iy), xytext=(4, -10), textcoords="offset points", fontsize=8,
                        color=colour)
        handles.append(Line2D([], [], color=style.GREY_DARK, marker="^", ls="none", ms=5,
                              label="Line of intersection"))
        handles.append(Line2D([], [], color=style.CARMINE, lw=3, label=r"Worst SMR $\leq 40$ (classes IV-V)"))
        handles.append(Line2D([], [], color=style.MALACHITE, lw=3, label=r"Worst SMR $> 40$"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
        ax.set_title(f"Lower hemisphere, {net_label.split(' ')[0].lower()} projection")
        ui.show_matplotlib(fig)

    lines = []
    for (n1, d1, a1), (n2, d2, a2) in combinations(joints, 2):
        line = slopes.intersection_line(d1, a1, d2, a2)
        if line is not None:
            lines.append({"Intersection": f"{n1}-{n2}", "Trend, αi (deg)": line[0], "Plunge, βi (deg)": line[1]})
    if lines:
        st.markdown("**Lines of intersection**")
        st.dataframe(pd.DataFrame(lines).style.format({"Trend, αi (deg)": "{:.1f}", "Plunge, βi (deg)": "{:.1f}"}),
                     width="stretch", hide_index=True)

    with st.expander("Romana's adjustment factors and classes"):
        st.markdown(
            "| Case | Very favourable | Favourable | Fair | Unfavourable | Very unfavourable |\n"
            "|---|---|---|---|---|---|\n"
            "| A (P, T, W) | > 30° | 30-20° | 20-10° | 10-5° | < 5° |\n"
            "| F1 | 0.15 | 0.40 | 0.70 | 0.85 | 1.00 |\n"
            "| B (P, W) | < 20° | 20-30° | 30-35° | 35-45° | > 45° |\n"
            "| F2 (P, W); F2 (T) | 0.15; 1 | 0.40; 1 | 0.70; 1 | 0.85; 1 | 1.00; 1 |\n"
            "| C (P, W) | > 10° | 10-0° | 0° | 0 to -10° | < -10° |\n"
            "| C (T) | < 110° | 110-120° | > 120° | - | - |\n"
            "| F3 | 0 | -6 | -25 | -50 | -60 |\n\n"
            "F4: natural slope +15, pre-splitting +10, smooth blasting +8, normal blasting or mechanical "
            "excavation 0, deficient blasting -8.\n\n"
            "| SMR | Class | Description | Stability | Failures | Support |\n"
            "|---|---|---|---|---|---|\n"
            "| 81-100 | I | Very good | Completely stable | None | None |\n"
            "| 61-80 | II | Good | Stable | Some blocks | Occasional |\n"
            "| 41-60 | III | Fair | Partially stable | Some joints or many wedges | Systematic |\n"
            "| 21-40 | IV | Poor | Unstable | Planar or big wedges | Important / corrective |\n"
            "| 0-20 | V | Very poor | Completely unstable | Big planar or soil-like | Re-excavation |"
        )
        st.markdown("Use α for a dip direction or trend and β for a dip or plunge; for wedges use the "
                    "trend and plunge of the line of intersection.")


# ---------------------------------------------------------------------------
# Page 2: kinematic screening and planar sliding
# ---------------------------------------------------------------------------


def _kinematic_stereonet(joints, face_dip, face_dd, phi, lat_planar, lat_topple, net, net_label,
                         planar_rows, wedge_rows, topple_rows, show=True):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        # Sliding crescent: plunge > phi and below the face's apparent dip, facing out of the face.
        g = np.linspace(-1.0, 1.0, 401)
        X, Y = np.meshgrid(g, g)
        inside = X ** 2 + Y ** 2 <= 1.0
        trend, plunge = slopes.unproject(X, Y, net)
        delta = np.radians(trend - face_dd)
        face_app = np.degrees(np.arctan2(math.sin(math.radians(face_dip)) * np.cos(delta),
                                         math.cos(math.radians(face_dip))))
        offset = np.abs((trend - face_dd + 180.0) % 360.0 - 180.0)
        sliding = inside & (plunge > phi) & (plunge < face_app) & (offset < 90.0)
        topple = inside & (offset <= lat_topple) & (plunge <= face_dip - phi)
        if topple.any():
            ax.contourf(X, Y, topple.astype(float), levels=[0.5, 1.5], colors=[style.tint(style.CARMINE, 0.94)])
            ax.contour(X, Y, topple.astype(float), levels=[0.5], colors=[style.CARMINE], linewidths=0.6,
                       linestyles=":")
        if sliding.any():
            ax.contourf(X, Y, sliding.astype(float), levels=[0.5, 1.5], colors=[style.CARMINE_TINT])
            ax.contour(X, Y, sliding.astype(float), levels=[0.5], colors=[style.CARMINE], linewidths=0.6)
        _net_axes(ax)
        fx, fy = slopes.plunge_circle(phi, net)
        ax.plot(fx, fy, color=style.GREY, ls="--", lw=1.0)
        r_phi = float(slopes.projection_radius(phi, net))
        for az in (face_dd - lat_planar, face_dd + lat_planar):
            a = math.radians(az)
            ax.plot([0.0, r_phi * math.sin(a)], [0.0, r_phi * math.cos(a)], color=style.GREY, ls="-.", lw=0.8)
        x, y = slopes.great_circle(face_dip, face_dd, net)
        ax.plot(x, y, color=style.INK, lw=2.0)
        handles = [Line2D([], [], color=style.INK, lw=2.0, label=f"Slope face ({_fmt_deg(face_dip, face_dd)})"),
                   Line2D([], [], color=style.GREY, ls="--", label=rf"Friction circle, $\phi = {phi:.0f}^\circ$"),
                   Line2D([], [], color=style.GREY, ls="-.", label=rf"Planar lateral limits $\pm{lat_planar:.0f}^\circ$"),
                   Patch(facecolor=style.CARMINE_TINT, edgecolor=style.CARMINE, lw=0.6,
                         label="Sliding zone (dip vectors, intersections)"),
                   Patch(facecolor=style.tint(style.CARMINE, 0.94), edgecolor=style.CARMINE, lw=0.6, ls=":",
                         label="Flexural toppling zone (poles)")]
        for i, (name, dip, dd) in enumerate(joints):
            ls = _LINESTYLES[i % len(_LINESTYLES)]
            gx, gy = slopes.great_circle(dip, dd, net)
            ax.plot(gx, gy, color=style.GREY_DARK, lw=1.0, ls=ls)
            planar_bad = planar_rows[i]["Possible"]
            dx, dy = slopes.project(dd, dip, net)
            ax.plot(dx, dy, marker="o", ls="none", ms=6,
                    color=style.CARMINE if planar_bad else style.MALACHITE)
            ax.annotate(name, (dx, dy), xytext=(4, 4), textcoords="offset points", fontsize=8)
            topple_bad = topple_rows[i]["Possible"]
            px, py = slopes.project(*slopes.pole(dip, dd), net)
            ax.plot(px, py, marker="s", ls="none", ms=5, mfc="none", mew=1.2,
                    color=style.CARMINE if topple_bad else style.MALACHITE)
            ax.annotate(name, (px, py), xytext=(4, -10), textcoords="offset points", fontsize=7,
                        color=style.GREY_DARK)
            handles.append(Line2D([], [], color=style.GREY_DARK, lw=1.0, ls=ls, label=f"{name} ({_fmt_deg(dip, dd)})"))
        for row in wedge_rows:
            if not math.isfinite(row["Trend (deg)"]):
                continue
            ix, iy = slopes.project(row["Trend (deg)"], row["Plunge (deg)"], net)
            ax.plot(ix, iy, marker="^", ls="none", ms=6,
                    color=style.CARMINE if row["Possible"] else style.MALACHITE)
            ax.annotate(row["Pair"], (ix, iy), xytext=(4, 4), textcoords="offset points", fontsize=7)
        handles += [
            Line2D([], [], marker="o", ls="none", color=style.GREY_DARK, label="Dip vector (planar)"),
            Line2D([], [], marker="s", ls="none", mfc="none", color=style.GREY_DARK, label="Pole (toppling)"),
            Line2D([], [], marker="^", ls="none", color=style.GREY_DARK, label="Intersection (wedge)"),
            Line2D([], [], marker="o", ls="none", color=style.CARMINE, label="Mechanism possible"),
            Line2D([], [], marker="o", ls="none", color=style.MALACHITE, label="Mechanism not possible"),
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
        ax.set_title(f"Lower hemisphere, {net_label.split(' ')[0].lower()} projection")
        if not show:
            return fig
        ui.show_matplotlib(fig)
    return None


def _planar_schematic(H, psi_f, psi_p, z, z_w, T, psi_T, fos, show=True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    geo = slopes.planar_geometry(H, psi_f, psi_p, z)
    xc, xk = geo["x_crest"], geo["x_crack"]
    yb, yt = geo["y_crack_base"], geo["y_crack_top"]
    if geo["crack_in_face"]:
        block = [(0.0, 0.0), (xk, yt), (xk, yb)]
    else:
        block = [(0.0, 0.0), (xc, H), (xk, H), (xk, yb)]
    x_end = max(xk, xc) + 0.35 * H
    tint = style.CARMINE_TINT if fos < 1.0 else style.MALACHITE_TINT
    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.add_patch(Polygon([(0.0, 0.0), (xc, H), (x_end, H), (x_end, -0.15 * H), (-0.15 * H, -0.15 * H),
                              (-0.15 * H, 0.0)], closed=True, facecolor=style.tint(style.GREY_LIGHT, 0.6),
                             edgecolor="none"))
        ax.add_patch(Polygon(block, closed=True, facecolor=tint, edgecolor="none"))
        ax.plot([-0.15 * H, 0.0, xc, x_end], [0.0, 0.0, H, H], color=style.INK, lw=1.6)
        ax.plot([0.0, xk], [0.0, yb], color=style.INK, lw=1.4, ls="--")
        ax.plot([xk, xk], [yb, yt], color=style.INK, lw=1.2, ls=":")
        if z_w > 0:
            w = 0.012 * x_end
            ax.add_patch(Polygon([(xk - w, yb), (xk + w, yb), (xk + w, yb + z_w), (xk - w, yb + z_w)],
                                 closed=True, facecolor=style.GREY, edgecolor="none"))
        cx = sum(p[0] for p in block) / len(block)
        cy = sum(p[1] for p in block) / len(block)
        arrow = dict(arrowstyle="-|>", lw=1.3, mutation_scale=12)
        L = 0.16 * H
        ax.annotate("", xy=(cx, cy - L), xytext=(cx, cy), arrowprops=dict(arrow, color=style.INK))
        ax.text(cx + 0.02 * H, cy - 0.5 * L, "$W$", fontsize=10)
        p = math.radians(psi_p)
        nx, ny = -math.sin(p), math.cos(p)
        xm, ym = 0.55 * xk, 0.55 * yb
        ax.annotate("", xy=(xm + L * nx, ym + L * ny), xytext=(xm, ym),
                    arrowprops=dict(arrow, color=style.GREY_DARK))
        ax.text(xm + 1.1 * L * nx, ym + 1.1 * L * ny, "$N'$", fontsize=10, color=style.GREY_DARK)
        if z_w > 0:
            xu, yu = 0.25 * xk, 0.25 * yb
            ax.annotate("", xy=(xu + 0.7 * L * nx, yu + 0.7 * L * ny), xytext=(xu, yu),
                        arrowprops=dict(arrow, color=style.GREY))
            ax.text(xu + 0.8 * L * nx - 0.04 * H, yu + 0.8 * L * ny, "$U$", fontsize=10, color=style.GREY_DARK)
            yv = yb + z_w / 3.0
            ax.annotate("", xy=(xk - 0.6 * L, yv), xytext=(xk, yv), arrowprops=dict(arrow, color=style.GREY))
            ax.text(xk - 0.75 * L, yv + 0.03 * H, "$V$", fontsize=10, color=style.GREY_DARK)
        if T > 0:
            t = math.radians(psi_T)
            sx, sy = 0.45 * xc, 0.45 * H
            ax.annotate("", xy=(sx + 1.3 * L * math.cos(t), sy - 1.3 * L * math.sin(t)), xytext=(sx, sy),
                        arrowprops=dict(arrow, color=style.MALACHITE))
            ax.text(sx - 0.12 * H, sy + 0.03 * H, "$T$", fontsize=10, color=style.MALACHITE)
        ax.text(0.04 * H, 0.015 * H, rf"$\psi_p = {psi_p:.0f}^\circ$", fontsize=9)
        ax.text(xc * 0.25 - 0.12 * H, 0.35 * H, rf"$\psi_f = {psi_f:.0f}^\circ$", fontsize=9)
        ax.text(xk + 0.02 * H, yb + 0.5 * (yt - yb), f"$z = {z:.1f}$ m", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-0.15 * H, x_end)
        ax.set_ylim(-0.15 * H, 1.15 * H)
        ax.set_xlabel(style.axis_label("Horizontal distance", "x", "m", math=True))
        ax.set_ylabel(style.axis_label("Elevation", "y", "m", math=True))
        if not show:
            return fig
        ui.show_matplotlib(fig)
    return None


def _fos_line_chart(x, series, x_label, target, title):
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, (name, values) in enumerate(series):
        s = style.series_style(i, "plotly")
        y = [v if math.isfinite(v) else None for v in values]
        fig.add_trace(go.Scatter(x=list(x), y=y, mode="lines", name=name,
                                 line=dict(color=s["color"], dash=s["dash"], width=2)))
    ref = style.reference_style("plotly")
    fig.add_hline(y=1.0, line=dict(color=ref["color"], dash="dot", width=1), annotation_text="FoS = 1")
    fig.add_hline(y=target, line=dict(color=ref["color"], dash="dash", width=1),
                  annotation_text=f"Target {target:.2f}")
    fig.update_layout(**style.plotly_layout(
        title=dict(text=title, font=dict(size=13)), height=360,
        xaxis=dict(style.plotly_layout()["xaxis"], title=x_label),
        yaxis=dict(style.plotly_layout()["yaxis"], title=style.axis_label("Factor of safety", "FoS", "-"))))
    ui.show_plotly(fig)


PLANAR_EQUATIONS = [
    r"A = \frac{H - z}{\sin\psi_p},\qquad V = \tfrac12\gamma_w z_w^2,\qquad U = \tfrac12\gamma_w z_w A",
    r"W = \tfrac12\gamma H^2\left[\left(1-\left(\tfrac{z}{H}\right)^2\right)\cot\psi_p - \cot\psi_f\right]"
    r"\quad\text{(crack behind the crest)}",
    r"W = \tfrac12\gamma H^2\left(1-\tfrac{z}{H}\right)^2\cot\psi_p\left(\cot\psi_p\tan\psi_f - 1\right)"
    r"\quad\text{(crack in the face)}",
    r"\mathrm{FoS} = \frac{cA + \left[W\cos\psi_p - U - V\sin\psi_p + T\sin(\psi_p+\psi_T)\right]\tan\phi}"
    r"{W\sin\psi_p + V\cos\psi_p - T\cos(\psi_p+\psi_T)}\quad\text{(active anchor)}",
    r"T_\mathrm{req} = \frac{F\,(W\sin\psi_p + V\cos\psi_p) - cA - (W\cos\psi_p - U - V\sin\psi_p)\tan\phi}"
    r"{F\cos(\psi_p+\psi_T) + \tan\phi\,\sin(\psi_p+\psi_T)}",
]


def render_kinematic(ctx: dict | None = None) -> None:
    """Stereonet kinematic screening and planar sliding limit equilibrium (Hoek and Bray)."""
    ui.page_header(
        "Kinematic screening and planar sliding",
        "Chapter 9 (structural analysis); RSE3010 Weeks 11-12",
        "Which mechanisms are kinematically possible on this face (planar sliding, wedge sliding, "
        "flexural toppling), and what is the factor of safety of a planar slide with a tension crack, "
        "water and anchors, and the anchor force for a target factor of safety?",
    )
    notes = [
        "Screening (Markland 1972; Hocking 1976; Goodman and Bray 1976): planar sliding if "
        "φ < ψj < apparent face dip and the dip direction is within the lateral limit of the face's; "
        "wedge sliding if the line of intersection daylights and plunges more steeply than φ; flexural "
        "toppling if the plane dips into the face within the lateral limit with ψj ≥ 90° − ψf + φ.",
        "Friction only (no cohesion) in the screening; a possible mechanism is a candidate for limit "
        "equilibrium, not a failure.",
        "Planar sliding (Hoek and Bray 1981): two-dimensional, per metre run, plane striking parallel to "
        "the face, horizontal upper surface, vertical tension crack (behind the crest or in the face), "
        "water in the crack to depth z_w with pressure falling linearly to zero at the toe.",
        "The effective normal force is not allowed below zero.  Active anchors reduce the driving "
        "force; passive anchors add T cos(ψp + ψT) to the resisting force.",
        "Dry case: z_w = 0 with φ_dry.  Wet case: the entered water level with φ_wet.",
    ]
    ui.assumptions(notes)
    sec = report.Section(
        key="kinematic", title="Kinematic screening and planar sliding", chapters="Chapter 9; RSE3010 Weeks 11-12",
        purpose="Which mechanisms are kinematically possible on the face, and the factor of safety of a planar "
                "slide with a tension crack, water and anchors.",
        assumptions=notes, references=[_REF_MARKLAND, _REF_HOCKING, _REF_GOODMAN_BRAY, _REF_HOEK_BRAY])

    st.subheader("1. Kinematic screening")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    face_dip = c1.number_input("Face dip, ψf (deg)", 1.0, 90.0, 60.0, 1.0, key="kin_face_dip")
    face_dd = c2.number_input("Face dip direction, αs (deg)", 0.0, 360.0, 170.0, 1.0, key="kin_face_dd")
    phi = c3.number_input("Friction angle, φ (deg)", 0.0, 80.0, 30.0, 1.0, key="kin_phi")
    lat_p = c4.number_input("Planar lateral limit (deg)", 0.0, 90.0, 20.0, 1.0, key="kin_lat_p")
    lat_t = c5.number_input("Toppling lateral limit (deg)", 0.0, 90.0, 10.0, 1.0, key="kin_lat_t")
    net_label = c6.selectbox("Stereonet", list(_NETS), key="kin_net")
    face_dd = face_dd % 360.0
    joints = _plane_editor(DEFAULT_JOINTS, "Joint", "kin_joints")

    planar_rows, topple_rows, wedge_rows = [], [], []
    for name, dip, dd in joints:
        planar_rows.append({"Joint": name, **slopes.planar_kinematics(dip, dd, face_dip, face_dd, phi, lat_p)})
        topple_rows.append({"Joint": name, **slopes.toppling_kinematics(dip, dd, face_dip, face_dd, phi, lat_t)})
    for (n1, d1, a1), (n2, d2, a2) in combinations(joints, 2):
        row = slopes.wedge_kinematics((d1, a1), (d2, a2), face_dip, face_dd, phi)
        row["Sliding on"] = row["Sliding on"].replace("plane 1", n1).replace("plane 2", n2)
        wedge_rows.append({"Pair": f"{n1}-{n2}", **row})

    ui.metrics([
        ("Planar sliding", f"{sum(r['Possible'] for r in planar_rows)} of {len(planar_rows)} joints"),
        ("Wedge sliding", f"{sum(r['Possible'] for r in wedge_rows)} of {len(wedge_rows)} pairs"),
        ("Flexural toppling", f"{sum(r['Possible'] for r in topple_rows)} of {len(topple_rows)} joints"),
        ("Friction circle", f"φ = {phi:.0f}°"),
    ])
    if joints:
        left, right = st.columns([3, 2])
        with left:
            _kinematic_stereonet(joints, face_dip, face_dd, phi, lat_p, lat_t, _NETS[net_label], net_label,
                                 planar_rows, wedge_rows, topple_rows)
        with right:
            def table(rows, columns, fmt):
                frame = pd.DataFrame(rows)
                if frame.empty:
                    st.caption("None.")
                    return frame
                frame = frame[columns].copy()
                for col in frame.columns:
                    if frame[col].dtype == bool:
                        frame[col] = frame[col].map(_yes)
                st.dataframe(_tint_rows(frame, "Possible", bad=("yes",), good=("no",)).format(fmt),
                             width="stretch", hide_index=True)
                return frame
            st.markdown("**Planar sliding**")
            tp = table(planar_rows, ["Joint", "Dip (deg)", "Dip direction (deg)", "Face apparent dip (deg)",
                                     "Offset from face (deg)", "Possible"],
                       {"Dip (deg)": "{:.0f}", "Dip direction (deg)": "{:.0f}", "Face apparent dip (deg)": "{:.1f}",
                        "Offset from face (deg)": "{:.1f}"})
            st.markdown("**Wedge sliding**")
            tw = table(wedge_rows, ["Pair", "Trend (deg)", "Plunge (deg)", "Face apparent dip (deg)", "Possible",
                                    "Sliding on"],
                       {"Trend (deg)": "{:.1f}", "Plunge (deg)": "{:.1f}", "Face apparent dip (deg)": "{:.1f}"})
            st.markdown("**Flexural toppling**")
            tt = table(topple_rows, ["Joint", "Dip (deg)", "Offset from opposite of face (deg)", "Required dip (deg)",
                                     "Possible"],
                       {"Dip (deg)": "{:.0f}", "Offset from opposite of face (deg)": "{:.1f}",
                        "Required dip (deg)": "{:.1f}"})
            st.caption("Carmine rows: mechanism kinematically possible; malachite rows: not possible. "
                       "'Sliding on' a single plane is Hocking's refinement of the wedge test.")
        export = pd.concat([tp.assign(Mode="Planar"), tw.assign(Mode="Wedge"), tt.assign(Mode="Toppling")],
                           ignore_index=True)
        ui.download_table(export, "kinematic_screening.csv", key="kin_dl_screen")
        n = report.num
        sec.inputs += [("Face: dip / dip direction", "", f"{face_dip:.0f} / {face_dd:03.0f}", "°"),
                       ("Friction angle for the screening", r"\phi", n(phi, 0), "°"),
                       ("Lateral limit, planar sliding", "", n(lat_p, 0), "°"),
                       ("Lateral limit, toppling", "", n(lat_t, 0), "°")] + _plane_rows(joints, "Joint")
        sec.results += [
            ("Planar sliding possible", "", f"{sum(r['Possible'] for r in planar_rows)} of {len(planar_rows)} joints", ""),
            ("Wedge sliding possible", "", f"{sum(r['Possible'] for r in wedge_rows)} of {len(wedge_rows)} pairs", ""),
            ("Flexural toppling possible", "", f"{sum(r['Possible'] for r in topple_rows)} of {len(topple_rows)} joints", "")]
        for caption, frame in (("Planar sliding screening.", tp), ("Wedge sliding screening.", tw),
                               ("Flexural toppling screening.", tt)):
            if len(frame):
                sec.tables.append((caption, report.text_frame(frame, 1)))
        sec.figures.append(("stereonet", "Kinematic screening on the stereonet (lower hemisphere).",
                            lambda: _kinematic_stereonet(joints, face_dip, face_dd, phi, lat_p, lat_t, _NETS[net_label],
                                                         net_label, planar_rows, wedge_rows, topple_rows, show=False)))
        report.record(st.session_state, sec)
    else:
        st.warning("Enter at least one joint set.")

    # -------------------------------------------------------------- planar LEM
    st.subheader("2. Planar sliding: limit equilibrium")
    options = ["Enter ψf and ψp"] + [f"{r['Joint']} ({_fmt_deg(r['Dip (deg)'], r['Dip direction (deg)'])})"
                                     for r in planar_rows]
    first_possible = next((i + 1 for i, r in enumerate(planar_rows) if r["Possible"]), 0)
    g1, g2, g3, g4 = st.columns(4)
    source = g1.selectbox("Sliding plane", options, index=first_possible, key="kin_plane")
    H = g2.number_input("Slope height, H (m)", 1.0, 500.0, 15.0, 1.0, key="kin_H")
    if source == options[0]:
        psi_f = g3.number_input("Face angle, ψf (deg)", 5.0, 90.0, float(face_dip), 1.0, key="kin_psi_f")
        psi_p = g4.number_input("Plane dip, ψp (deg)", 1.0, 89.0, 40.0, 1.0, key="kin_psi_p")
    else:
        idx = options.index(source) - 1
        psi_f, psi_p = float(face_dip), float(planar_rows[idx]["Dip (deg)"])
        g3.metric("Face angle, ψf", f"{psi_f:.0f}°")
        g4.metric("Plane dip, ψp", f"{psi_p:.0f}°")
    if psi_p >= psi_f or psi_p <= 0.0:
        st.error(f"The sliding plane (ψp = {psi_p:.0f}°) does not daylight in a face of ψf = {psi_f:.0f}°; "
                 "planar sliding is not possible on this plane.")
        return

    z_crit, b_crit = slopes.critical_tension_crack(H, psi_f, psi_p)
    m1, m2, m3, m4 = st.columns(4)
    auto_z = m1.checkbox("Critical (dry) crack depth", value=True, key="kin_auto_z",
                         help="z/H = 1 − √(cot ψf tan ψp), at b/H = √(cot ψf cot ψp) − cot ψf behind the crest.")
    if auto_z:
        z = min(z_crit, 0.95 * H)
        m2.metric("Tension crack depth, z", f"{z:.2f} m")
    else:
        z = m2.number_input("Tension crack depth, z (m)", 0.0, float(0.99 * H), float(min(z_crit, 0.95 * H)), 0.1,
                            key="kin_z", help="Depth of the crack base below the crest level.")
    fill = m3.slider("Water in the crack, z_w / open depth (-)", 0.0, 1.0, 1.0, 0.05, key="kin_fill")
    geo = slopes.planar_geometry(H, psi_f, psi_p, z)
    z_w = fill * geo["crack_length"]
    m4.metric("Water depth, z_w", f"{z_w:.2f} m")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    gamma = k1.number_input("Rock unit weight, γ (kN/m³)", 10.0, 35.0, 20.0, 0.5, key="kin_gamma")
    gamma_w = k2.number_input("Water unit weight, γw (kN/m³)", 9.0, 10.5, 9.81, 0.01, key="kin_gamma_w")
    c = k3.number_input("Cohesion, c (kPa)", 0.0, 1000.0, 20.0, 1.0, key="kin_c")
    phi_dry = k4.number_input("Dry friction, φ_dry (deg)", 0.0, 80.0, 45.0, 1.0, key="kin_phi_dry")
    phi_wet = k5.number_input("Wet friction, φ_wet (deg)", 0.0, 80.0, 28.0, 1.0, key="kin_phi_wet")
    target = k6.number_input("Target FoS (-)", 1.0, 3.0, 1.3, 0.05, key="kin_target")

    a1, a2, a3, a4, a5 = st.columns(5)
    T = a1.number_input("Anchor force, T (kN/m)", 0.0, 1e5, 0.0, 10.0, key="kin_T",
                        help="Total anchor force per metre run of slope; not multiplied by bolt length.")
    psi_T = a2.number_input("Anchor inclination below horizontal, ψT (deg)", -45.0, 75.0, 15.0, 1.0, key="kin_psi_T")
    anchor_label = a3.selectbox("Anchor action", ["Active (tensioned)", "Passive (untensioned)"], key="kin_anchor")
    bolt_load = a4.number_input("Working load per bolt (kN)", 10.0, 5000.0, 250.0, 10.0, key="kin_bolt")
    spacing = a5.number_input("Horizontal bolt spacing, s_h (m)", 0.25, 20.0, 2.0, 0.25, key="kin_spacing")
    anchor = "passive" if anchor_label.startswith("Passive") else "active"

    dry = slopes.planar_fos(H, psi_f, psi_p, z, 0.0, c, phi_dry, gamma, gamma_w, T, psi_T, anchor)
    wet = slopes.planar_fos(H, psi_f, psi_p, z, z_w, c, phi_wet, gamma, gamma_w, T, psi_T, anchor)
    if not (dry["valid"] and wet["valid"]):
        st.error(dry.get("reason") or wet.get("reason"))
        return
    T_req = slopes.planar_required_anchor(H, psi_f, psi_p, z, z_w, c, phi_wet, gamma, gamma_w, psi_T, target,
                                          anchor)
    psi_T_opt = slopes.optimum_anchor_angle(phi_wet, psi_p, target, anchor)
    T_req_opt = slopes.planar_required_anchor(H, psi_f, psi_p, z, z_w, c, phi_wet, gamma, gamma_w, psi_T_opt,
                                              target, anchor)
    rows_needed = slopes.bolt_rows(T_req, bolt_load, spacing)
    ui.metrics([
        ("FoS dry", ui.fmt(dry["FoS"], 2), f"z_w = 0, φ = {phi_dry:.0f}°"),
        ("FoS wet", ui.fmt(wet["FoS"], 2), f"z_w = {z_w:.2f} m, φ = {phi_wet:.0f}°"),
        (f"T for wet FoS {target:.2f}", ui.fmt(T_req, 0, "kN/m"), f"at ψT = {psi_T:.0f}°"),
        ("Bolt rows", f"{rows_needed}" if math.isfinite(T_req) else "--",
         "ceil(T s_h / bolt working load)"),
    ])
    ui.metrics([
        ("Optimum ψT", f"{psi_T_opt:.1f}°", "tan(ψp + ψT) = tan φ / F (active); ψp + ψT = φ (passive)"),
        ("T at the optimum ψT", ui.fmt(T_req_opt, 0, "kN/m")),
        ("Crack position", "in the face" if geo["crack_in_face"] else "behind the crest"),
        ("Critical crack, b behind crest", ui.fmt(b_crit, 2, "m")),
    ])
    ui.status_line(wet["FoS"] >= target,
                   f"The wet factor of safety {wet['FoS']:.2f} meets the target {target:.2f}.",
                   f"The wet factor of safety {wet['FoS']:.2f} is below the target {target:.2f}: "
                   f"an anchor force of about {T_req:.0f} kN/m at ψT = {psi_T:.0f}° is required.")
    if geo["crack_in_face"]:
        st.info("The tension crack lies in the slope face, so the weight uses Hoek and Bray's crack-in-face "
                "expression.")

    left, right = st.columns([3, 2])
    with left:
        _planar_schematic(H, psi_f, psi_p, z, z_w, T, psi_T, wet["FoS"])
    with right:
        names = [("A", "Base length A (m/m)"), ("W", "Weight W (kN/m)"), ("U", "Uplift U (kN/m)"),
                 ("V", "Crack water force V (kN/m)"), ("T_shear", "T cos(ψp + ψT) (kN/m)"),
                 ("T_normal", "T sin(ψp + ψT) (kN/m)"), ("N", "Effective normal N' (kN/m)"),
                 ("driving", "Driving force (kN/m)"), ("resisting", "Resisting force (kN/m)"), ("FoS", "FoS (-)")]
        detail = pd.DataFrame({"Quantity": [n for _, n in names],
                               "Dry": [dry[k] for k, _ in names], "Wet": [wet[k] for k, _ in names]})
        st.dataframe(detail.style.format({"Dry": "{:,.2f}", "Wet": "{:,.2f}"}), width="stretch", hide_index=True)
        ui.download_table(detail, "planar_sliding.csv", key="kin_dl_planar")

    with st.expander("Equations (Hoek and Bray, horizontal upper surface)"):
        for eq in PLANAR_EQUATIONS:
            st.latex(eq)
        st.markdown("The crack lies behind the crest when z ≤ H (1 − cot ψf tan ψp).  For a passive anchor "
                    "T cos(ψp + ψT) moves to the numerator and F leaves the first term of the denominator of "
                    "T_req.  Water is entered through U and V, not through a pore-pressure ratio r_u.")

    n = report.num
    sec.inputs += [("Slope height", "H", n(H, 1), "m"), ("Face angle", r"\psi_f", n(psi_f, 0), "°"),
                   ("Dip of the sliding plane", r"\psi_p", n(psi_p, 0), "°"),
                   ("Depth of the tension crack", "z", n(z, 2), "m"), ("Water depth in the crack", "z_w", n(z_w, 2), "m"),
                   ("Unit weight of the rock", r"\gamma", n(gamma, 1), "kN/m³"),
                   ("Unit weight of water", r"\gamma_w", n(gamma_w, 2), "kN/m³"), ("Cohesion", "c", n(c, 0), "kPa"),
                   ("Friction angle, dry", r"\phi_{dry}", n(phi_dry, 0), "°"),
                   ("Friction angle, wet", r"\phi_{wet}", n(phi_wet, 0), "°"),
                   ("Anchor force", "T", n(T, 0), "kN/m"), ("Anchor inclination below horizontal", r"\psi_T", n(psi_T, 0), "°"),
                   ("Anchor action", "", anchor, ""), ("Target factor of safety", "F", n(target, 2), "-")]
    sec.equations += PLANAR_EQUATIONS
    sec.results += [("Factor of safety, dry", "F", n(dry["FoS"], 2), "-"), ("Factor of safety, wet", "F", n(wet["FoS"], 2), "-"),
                    (f"Anchor force for a wet factor of safety of {target:.2f}", "T_{req}", n(T_req, 0), "kN/m"),
                    ("Bolt rows", "", f"{rows_needed}" if math.isfinite(T_req) else "--", ""),
                    ("Optimum anchor inclination", r"\psi_{T,opt}", n(psi_T_opt, 1), "°"),
                    ("Anchor force at the optimum inclination", "T", n(T_req_opt, 0), "kN/m"),
                    ("Position of the tension crack", "", "in the face" if geo["crack_in_face"] else "behind the crest", ""),
                    ("Critical crack, distance behind the crest", "b", n(b_crit, 2), "m")]
    sec.checks.append((f"Wet factor of safety {wet['FoS']:.2f} against the target {target:.2f}.",
                       bool(wet["FoS"] >= target)))
    sec.tables.append(("Forces of the planar slide per metre run, dry and wet.", report.text_frame(detail, 2)))
    sec.figures.append(("planar", "Planar slide with the tension crack, water and anchor (wet case).",
                        lambda: _planar_schematic(H, psi_f, psi_p, z, z_w, T, psi_T, wet["FoS"], show=False)))
    report.record(st.session_state, sec)

    st.markdown("**Sensitivity**")
    t1, t2, t3 = st.tabs(["Water in the crack", "Face angle", "Anchor force"])
    with t1:
        fills = np.linspace(0.0, 1.0, 21)
        zws = fills * geo["crack_length"]
        _fos_line_chart(zws, [
            (f"φ = {phi_dry:.0f}° (dry friction)", [slopes.planar_fos(H, psi_f, psi_p, z, w, c, phi_dry, gamma,
                                                                     gamma_w, T, psi_T, anchor)["FoS"] for w in zws]),
            (f"φ = {phi_wet:.0f}° (wet friction)", [slopes.planar_fos(H, psi_f, psi_p, z, w, c, phi_wet, gamma,
                                                                     gamma_w, T, psi_T, anchor)["FoS"] for w in zws]),
        ], style.axis_label("Water depth in the crack", "z<sub>w</sub>", "m"), target, "FoS against crack water depth")
    with t2:
        angles = np.linspace(psi_p + 1.0, 90.0, 40)
        values = []
        for a in angles:
            zz = min(slopes.critical_tension_crack(H, a, psi_p)[0], 0.95 * H) if auto_z else z
            g = slopes.planar_geometry(H, a, psi_p, zz)
            r = slopes.planar_fos(H, a, psi_p, zz, min(fill * g["crack_length"], g["crack_length"]), c, phi_wet,
                                  gamma, gamma_w, T, psi_T, anchor)
            values.append(r["FoS"] if r["valid"] else math.nan)
        _fos_line_chart(angles, [("Wet case", values)], style.axis_label("Face angle", "ψ<sub>f</sub>", "deg"),
                        target, "Effect of flattening the face")
    with t3:
        top = max(2.0 * (T_req if math.isfinite(T_req) else 0.0), 2.0 * T, 100.0)
        forces = np.linspace(0.0, top, 41)
        _fos_line_chart(forces, [
            ("Wet case", [slopes.planar_fos(H, psi_f, psi_p, z, z_w, c, phi_wet, gamma, gamma_w, f, psi_T,
                                            anchor)["FoS"] for f in forces]),
            ("Dry case", [slopes.planar_fos(H, psi_f, psi_p, z, 0.0, c, phi_dry, gamma, gamma_w, f, psi_T,
                                            anchor)["FoS"] for f in forces]),
        ], style.axis_label("Anchor force", "T", "kN/m"), target, f"Anchor force at ψT = {psi_T:.0f}°")


# ---------------------------------------------------------------------------
# Page 3: wedge and circular limit equilibrium
# ---------------------------------------------------------------------------


WEDGE_EQUATIONS = [
    r"N_A + N_B\cos\theta = W\cos\psi_a,\qquad N_A\cos\theta + N_B = W\cos\psi_b",
    r"\mathrm{FoS} = \frac{c_A A_A + c_B A_B + (N_A - U_A)\tan\phi_A + (N_B - U_B)\tan\phi_B}"
    r"{W\sin\psi_i},\qquad U = \frac{\gamma_w H}{6}\,A",
]
CIRCULAR_EQUATIONS = [
    r"\text{Bishop:}\quad F = \frac{\sum \left[c\,b + (W - u\,b)\tan\phi\right]/m_\alpha}{\sum W\sin\alpha},"
    r"\qquad m_\alpha = \cos\alpha + \frac{\sin\alpha\tan\phi}{F}",
    r"\text{Ordinary method:}\quad F = \frac{\sum \left[c\,l + (W\cos\alpha - u\,l)\tan\phi\right]}"
    r"{\sum W\sin\alpha},\qquad u = r_u\,\gamma\,h",
]


def _wedge_section(notes: list[str] | None = None) -> None:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from matplotlib.lines import Line2D

    st.markdown("Planes as dip / dip direction (deg).  Default inputs: the worked wedge example of Hoek and "
                "Bray (1981).  Plane A is conventionally the flatter plane.")
    r1 = st.columns(4)
    a_dip = r1[0].number_input("Plane A dip (deg)", 0.0, 90.0, 45.0, 1.0, key="lem_a_dip")
    a_dd = r1[1].number_input("Plane A dip direction (deg)", 0.0, 360.0, 105.0, 1.0, key="lem_a_dd")
    b_dip = r1[2].number_input("Plane B dip (deg)", 0.0, 90.0, 70.0, 1.0, key="lem_b_dip")
    b_dd = r1[3].number_input("Plane B dip direction (deg)", 0.0, 360.0, 235.0, 1.0, key="lem_b_dd")
    r2 = st.columns(5)
    f_dip = r2[0].number_input("Face dip (deg)", 1.0, 90.0, 65.0, 1.0, key="lem_f_dip")
    f_dd = r2[1].number_input("Face dip direction (deg)", 0.0, 360.0, 185.0, 1.0, key="lem_f_dd")
    u_dip = r2[2].number_input("Upper surface dip (deg)", 0.0, 60.0, 12.0, 1.0, key="lem_u_dip")
    u_dd = r2[3].number_input("Upper surface dip direction (deg)", 0.0, 360.0, 195.0, 1.0, key="lem_u_dd")
    H = r2[4].number_input("Wedge height, H (m)", 1.0, 500.0, 40.0, 1.0, key="lem_wH",
                           help="Vertical height of the line of intersection between its outcrops on the face "
                                "and on the upper surface.")
    r3 = st.columns(6)
    c_a = r3[0].number_input("Cohesion on A, c_A (kPa)", 0.0, 1000.0, 24.0, 1.0, key="lem_ca")
    phi_a = r3[1].number_input("Friction on A, φ_A (deg)", 0.0, 80.0, 20.0, 1.0, key="lem_phia")
    c_b = r3[2].number_input("Cohesion on B, c_B (kPa)", 0.0, 1000.0, 48.0, 1.0, key="lem_cb")
    phi_b = r3[3].number_input("Friction on B, φ_B (deg)", 0.0, 80.0, 30.0, 1.0, key="lem_phib")
    gamma = r3[4].number_input("Rock unit weight, γ (kN/m³)", 10.0, 35.0, 25.0, 0.5, key="lem_wgamma")
    gamma_w = r3[5].number_input("Water unit weight, γw (kN/m³)", 9.0, 10.5, 9.81, 0.01, key="lem_wgw")

    A, B = (a_dip, a_dd % 360.0), (b_dip, b_dd % 360.0)
    F, U = (f_dip, f_dd % 360.0), (u_dip, u_dd % 360.0)
    dry = slopes.wedge_fos(A, B, F, U, H, c_a, phi_a, c_b, phi_b, gamma, gamma_w, saturated=False)
    if not dry["valid"]:
        st.error(dry["reason"])
        return
    wet = slopes.wedge_fos(A, B, F, U, H, c_a, phi_a, c_b, phi_b, gamma, gamma_w, saturated=True)
    geo = dry["geometry"]
    ui.metrics([
        ("FoS dry", ui.fmt(dry["FoS"], 2)),
        ("FoS fully saturated", ui.fmt(wet["FoS"], 2), "Hoek and Bray water distribution"),
        ("Line of intersection", f"{geo['plunge']:.1f}/{geo['trend']:03.0f}", "plunge / trend"),
        ("Wedge weight, W", ui.fmt(dry["W"], 0, "kN"), f"volume {geo['volume']:,.0f} m³"),
    ])
    ui.status_line(wet["FoS"] >= 1.0,
                   f"{dry['mode']}; the saturated factor of safety is {wet['FoS']:.2f}.",
                   f"{dry['mode']}; the saturated factor of safety {wet['FoS']:.2f} is below 1.")
    if phi_a == phi_b and c_a == 0 and c_b == 0 and dry["mode"].startswith("Sliding on both"):
        k = slopes.wedge_factor(A, B)
        st.caption(f"Check: wedge factor K = sin β / sin(ξ/2) = {k['K']:.3f} (ξ = {k['xi']:.1f}°, "
                   f"β = {k['beta']:.1f}°), K tan φ / tan ψi = "
                   f"{k['K'] * math.tan(math.radians(phi_a)) / math.tan(math.radians(k['plunge'])):.3f}.")

    left, right = st.columns(2)
    with left:
        net = slopes.EQUAL_ANGLE
        with plt.rc_context(style.matplotlib_rcparams()):
            fig, ax = plt.subplots(figsize=(5.4, 5.4))
            _net_axes(ax)
            handles = []
            for (dip, dd), label, colour, ls, lw in (
                    (F, "Face", style.INK, "-", 2.0), (U, "Upper surface", style.GREY, "-.", 1.2),
                    (A, "Plane A", style.GREY_DARK, "-", 1.2), (B, "Plane B", style.GREY_DARK, "--", 1.2)):
                x, y = slopes.great_circle(dip, dd, net)
                ax.plot(x, y, color=colour, ls=ls, lw=lw)
                handles.append(Line2D([], [], color=colour, ls=ls, lw=lw, label=f"{label} ({_fmt_deg(dip, dd)})"))
            phi_max = max(phi_a, phi_b)
            fx, fy = slopes.plunge_circle(phi_max, net)
            ax.plot(fx, fy, color=style.GREY, ls=":", lw=1.0)
            handles.append(Line2D([], [], color=style.GREY, ls=":", label=rf"Friction circle, $\phi = {phi_max:.0f}^\circ$"))
            ix, iy = slopes.project(geo["trend"], geo["plunge"], net)
            colour = style.CARMINE if wet["FoS"] < 1.0 else style.MALACHITE
            ax.plot(ix, iy, marker="^", color=colour, ms=7, ls="none")
            handles.append(Line2D([], [], marker="^", color=colour, ls="none", label="Line of intersection"))
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8)
            ax.set_title("Lower hemisphere, equal-angle projection")
            ui.show_matplotlib(fig)
    with right:
        pts = np.vstack([geo["O"], geo["T"], geo["P1"], geo["P2"]])
        tint = style.CARMINE_TINT if wet["FoS"] < 1.0 else style.MALACHITE_TINT
        fig3 = go.Figure()
        fig3.add_trace(go.Mesh3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], i=[0, 0, 0, 1], j=[1, 1, 2, 2],
                                 k=[2, 3, 3, 3], color=tint, opacity=0.85, flatshading=True, name="Wedge",
                                 hoverinfo="skip"))
        edges = [(0, 1, "Line of intersection"), (0, 2, "Plane A on face"), (0, 3, "Plane B on face"),
                 (1, 2, "Plane A on upper surface"), (1, 3, "Plane B on upper surface"), (2, 3, "Crest")]
        for i, j, name in edges:
            main = i == 0 and j == 1
            fig3.add_trace(go.Scatter3d(
                x=pts[[i, j], 0], y=pts[[i, j], 1], z=pts[[i, j], 2], mode="lines", name=name,
                line=dict(color=style.CARMINE if main else style.INK, width=5 if main else 3),
                showlegend=main))
        layout = style.plotly_layout(height=430, margin=dict(l=0, r=0, t=30, b=0),
                                     title=dict(text="Wedge geometry (origin at the toe of the intersection)",
                                                font=dict(size=13)),
                                     scene=dict(xaxis_title="East (m)", yaxis_title="North (m)",
                                                zaxis_title="Elevation (m)", aspectmode="data"))
        fig3.update_layout(**layout)
        ui.show_plotly(fig3)

    names = [("W", "Weight W (kN)"), ("area_a", "Area of plane A (m²)"), ("area_b", "Area of plane B (m²)"),
             ("NA", "Total normal reaction N_A (kN)"), ("NB", "Total normal reaction N_B (kN)"),
             ("UA", "Water force U_A (kN)"), ("UB", "Water force U_B (kN)"),
             ("NA_eff", "Effective normal N'_A (kN)"), ("NB_eff", "Effective normal N'_B (kN)"),
             ("driving", "Driving force (kN)"), ("resisting", "Resisting force (kN)"), ("FoS", "FoS (-)")]
    detail = pd.DataFrame({"Quantity": [n for _, n in names], "Dry": [dry[k] for k, _ in names],
                           "Fully saturated": [wet[k] for k, _ in names]})
    st.dataframe(detail.style.format({"Dry": "{:,.2f}", "Fully saturated": "{:,.2f}"}), width="stretch",
                 hide_index=True)
    ui.download_table(detail, "wedge_sliding.csv", key="lem_dl_wedge")
    n = report.num
    report.record(st.session_state, report.Section(
        key="slope_lem", title="Wedge limit equilibrium", chapters="Chapter 9; RSE3010 Week 12",
        purpose=f"Factor of safety of a tetrahedral wedge on two discontinuities (Hoek and Bray), dry and fully "
                f"saturated. {dry['mode']}.",
        inputs=[("Plane A: dip / dip direction", "", f"{a_dip:.0f} / {a_dd:03.0f}", "°"),
                ("Plane B: dip / dip direction", "", f"{b_dip:.0f} / {b_dd:03.0f}", "°"),
                ("Face: dip / dip direction", "", f"{f_dip:.0f} / {f_dd:03.0f}", "°"),
                ("Upper surface: dip / dip direction", "", f"{u_dip:.0f} / {u_dd:03.0f}", "°"),
                ("Wedge height", "H", n(H, 1), "m"),
                ("Cohesion on A", "c_A", n(c_a, 0), "kPa"), ("Friction on A", r"\phi_A", n(phi_a, 0), "°"),
                ("Cohesion on B", "c_B", n(c_b, 0), "kPa"), ("Friction on B", r"\phi_B", n(phi_b, 0), "°"),
                ("Unit weight of the rock", r"\gamma", n(gamma, 1), "kN/m³"),
                ("Unit weight of water", r"\gamma_w", n(gamma_w, 2), "kN/m³")],
        equations=WEDGE_EQUATIONS,
        results=[("Factor of safety, dry", "F", n(dry["FoS"], 2), "-"),
                 ("Factor of safety, fully saturated", "F", n(wet["FoS"], 2), "-"),
                 ("Line of intersection: plunge / trend", "", f"{geo['plunge']:.1f} / {geo['trend']:03.0f}", "°"),
                 ("Weight of the wedge", "W", n(dry["W"], 0), "kN"),
                 ("Volume of the wedge", "V", n(geo["volume"], 0), "m³")],
        checks=[(f"Fully saturated factor of safety {wet['FoS']:.2f} against 1.0.", bool(wet["FoS"] >= 1.0))],
        tables=[("Forces on the wedge, dry and fully saturated.", report.text_frame(detail, 2))],
        assumptions=notes or [], references=[_REF_HOEK_BRAY]))
    with st.expander("Method"):
        for eq in WEDGE_EQUATIONS:
            st.latex(eq)
        st.markdown("θ is the angle between the normals of planes A and B, ψa and ψb their dips and ψi the "
                    "plunge of the line of intersection.  The wedge is the tetrahedron bounded by the two "
                    "planes, the face and the upper surface; this is Hoek and Bray's comprehensive solution "
                    "in vector form.  Fully saturated: water pressure zero on the outcrop edges and "
                    "½γw H at mid-height of the line of intersection.  The sliding mode is chosen from the "
                    "applied force, weight and water together: if it presses on one plane only and the sliding "
                    "direction on that plane leaves the other, the wedge slides on that plane alone, with "
                    "N' = −A·n and the in-plane part of A as the driving force; if it pulls away from both "
                    "planes, the water lifts the wedge (FoS = 0).")


def _circular_section(notes: list[str] | None = None) -> None:
    import matplotlib.pyplot as plt

    st.markdown("Circular sliding in heavily fractured or weak rock: Bishop's simplified method over a search "
                "of circles through the toe; the ordinary method of slices is shown on the same circle.")
    r1 = st.columns(4)
    H = r1[0].number_input("Slope height, H (m)", 1.0, 300.0, 20.0, 1.0, key="lem_cH")
    psi_f = r1[1].number_input("Slope angle, ψf (deg)", 5.0, 90.0, 60.0, 1.0, key="lem_cpsi")
    gamma = r1[2].number_input("Unit weight, γ (kN/m³)", 10.0, 35.0, 22.0, 0.5, key="lem_cgamma")
    c = r1[3].number_input("Cohesion, c (kPa)", 0.0, 1000.0, 20.0, 1.0, key="lem_cc")
    r2 = st.columns(4)
    phi_dry = r2[0].number_input("Dry friction, φ_dry (deg)", 0.0, 80.0, 45.0, 1.0, key="lem_cphid")
    phi_wet = r2[1].number_input("Wet friction, φ_wet (deg)", 0.0, 80.0, 28.0, 1.0, key="lem_cphiw")
    ru = r2[2].slider("Wet pore-pressure ratio, r_u (-)", 0.0, 0.8, 0.2, 0.05, key="lem_cru")
    n_slices = r2[3].slider("Slices (-)", 10, 80, 40, 5, key="lem_cn")

    dry = slopes.toe_circle_search(H, psi_f, c, phi_dry, gamma, 0.0, n_slices)
    wet = slopes.toe_circle_search(H, psi_f, c, phi_wet, gamma, ru, n_slices)
    if not (dry["valid"] and wet["valid"]):
        st.error("No circle through the toe cuts the slope for these inputs.")
        return
    s_dry = slopes.circle_slices(H, psi_f, dry["xc"], dry["yc"], dry["R"], n_slices, gamma, 0.0)
    s_wet = slopes.circle_slices(H, psi_f, wet["xc"], wet["yc"], wet["R"], n_slices, gamma, ru)
    fel_dry = slopes.fellenius_fos(s_dry, c, phi_dry)
    fel_wet = slopes.fellenius_fos(s_wet, c, phi_wet)
    ui.metrics([
        ("Bishop FoS, dry", ui.fmt(dry["FoS"], 2), f"φ = {phi_dry:.0f}°, r_u = 0"),
        ("Bishop FoS, wet", ui.fmt(wet["FoS"], 2), f"φ = {phi_wet:.0f}°, r_u = {ru:.2f}"),
        ("Ordinary method, dry", ui.fmt(fel_dry, 2), "same circle"),
        ("Ordinary method, wet", ui.fmt(fel_wet, 2), "same circle"),
    ])
    ui.status_line(wet["FoS"] >= 1.0,
                   f"The critical wet circle has a factor of safety of {wet['FoS']:.2f}.",
                   f"The critical wet circle has a factor of safety of {wet['FoS']:.2f}, below 1.")
    if c == 0:
        st.info("With c = 0 the critical surface tends to the shallowest one (the infinite-slope value "
                "tan φ / tan ψf); masses smaller than 0.5 % of H² are not searched.")

    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        x_crest = 0.0 if psi_f >= 89.999 else H / math.tan(math.radians(psi_f))
        x_right = max(x_crest + 1.6 * H, wet["xc"] + 0.2 * H, dry["xc"] + 0.2 * H,
                      s_wet["x_exit"] + 0.3 * H, s_dry["x_exit"] + 0.3 * H)
        x_left = min(-0.5 * H, wet["xc"] - 0.2 * H, dry["xc"] - 0.2 * H)
        xs = np.linspace(x_left, x_right, 400)
        ax.fill_between(xs, -0.4 * H, slopes.ground_level(xs, H, psi_f), color=style.tint(style.GREY_LIGHT, 0.6),
                        lw=0)
        ax.plot(xs, slopes.ground_level(xs, H, psi_f), color=style.INK, lw=1.6)
        for sl, res, colour, ls, label in ((s_dry, dry, style.GREY_DARK, "--", "Dry critical circle"),
                                           (s_wet, wet, style.CARMINE if wet["FoS"] < 1.0 else style.MALACHITE,
                                            "-", "Wet critical circle")):
            xx = np.linspace(sl["x_entry"], sl["x_exit"], 200)
            yy = res["yc"] - np.sqrt(np.clip(res["R"] ** 2 - (xx - res["xc"]) ** 2, 0.0, None))
            ax.plot(xx, yy, color=colour, ls=ls, lw=1.6, label=f"{label}, FoS = {res['FoS']:.2f}")
            ax.plot(res["xc"], res["yc"], marker="+", color=colour, ms=8, ls="none")
            for xe, ye in ((xx[0], yy[0]), (xx[-1], yy[-1])):
                ax.plot([res["xc"], xe], [res["yc"], ye], color=style.GREY_LIGHT, lw=0.6, ls=":")
        edges = np.linspace(s_wet["x_entry"], s_wet["x_exit"], n_slices + 1)
        for xe in edges[1:-1]:
            yb = wet["yc"] - math.sqrt(max(wet["R"] ** 2 - (xe - wet["xc"]) ** 2, 0.0))
            ax.plot([xe, xe], [yb, float(slopes.ground_level(xe, H, psi_f))], color=style.GREY, lw=0.4)
        ax.set_aspect("equal")
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(-0.4 * H, max(wet["yc"], dry["yc"]) + 0.15 * H)
        ax.set_xlabel(style.axis_label("Horizontal distance from the toe", "x", "m", math=True))
        ax.set_ylabel(style.axis_label("Elevation", "y", "m", math=True))
        ax.legend(loc="upper left")
        ui.show_matplotlib(fig)
    st.caption("Crosses mark the centres of the critical circles; thin vertical lines are the slices of the "
               "critical wet circle.")

    table = pd.DataFrame({
        "x (m)": s_wet["x"], "Width b (m)": s_wet["b"], "Height h (m)": s_wet["h"],
        "Base angle α (deg)": np.degrees(s_wet["alpha"]), "Base length l (m)": s_wet["l"],
        "Weight W (kN/m)": s_wet["W"], "Pore pressure u (kPa)": s_wet["u"]})
    with st.expander("Slices of the critical wet circle"):
        st.dataframe(table.style.format("{:.2f}"), width="stretch", hide_index=True)
    ui.download_table(table, "circular_slices.csv", key="lem_dl_circle")
    n = report.num
    report.record(st.session_state, report.Section(
        key="slope_lem", title="Circular limit equilibrium", chapters="Chapter 9; RSE3010 Week 12",
        purpose="Factor of safety of a circular slip through the toe by Bishop's simplified method over a search of "
                "circles, with the ordinary method of slices on the same circle.",
        inputs=[("Slope height", "H", n(H, 1), "m"), ("Slope angle", r"\psi_f", n(psi_f, 0), "°"),
                ("Unit weight", r"\gamma", n(gamma, 1), "kN/m³"), ("Cohesion", "c", n(c, 0), "kPa"),
                ("Friction angle, dry", r"\phi_{dry}", n(phi_dry, 0), "°"),
                ("Friction angle, wet", r"\phi_{wet}", n(phi_wet, 0), "°"),
                ("Pore-pressure ratio, wet", "r_u", n(ru, 2), "-"), ("Slices", "", f"{n_slices}", "-")],
        equations=CIRCULAR_EQUATIONS,
        results=[("Bishop factor of safety, dry", "F", n(dry["FoS"], 2), "-"),
                 ("Bishop factor of safety, wet", "F", n(wet["FoS"], 2), "-"),
                 ("Ordinary method, dry", "F", n(fel_dry, 2), "-"),
                 ("Ordinary method, wet", "F", n(fel_wet, 2), "-"),
                 ("Critical wet circle: centre", "", f"({wet['xc']:.1f}, {wet['yc']:.1f})", "m"),
                 ("Critical wet circle: radius", "R", n(wet["R"], 1), "m")],
        checks=[(f"Wet factor of safety of the critical circle {wet['FoS']:.2f} against 1.0.", bool(wet["FoS"] >= 1.0))],
        assumptions=notes or [], references=[_REF_BISHOP]))
    with st.expander("Method"):
        for eq in CIRCULAR_EQUATIONS:
            st.latex(eq)
        st.markdown(f"Centres searched on a grid from x = -H to the crest + 1.5H and y = 0.5H to 3H, refined "
                    f"once around the minimum; the critical wet circle has its centre at "
                    f"({wet['xc']:.1f}, {wet['yc']:.1f}) m and radius {wet['R']:.1f} m.")


def render_slope_lem(ctx: dict | None = None) -> None:
    """Wedge (Hoek and Bray) and circular (Bishop) limit equilibrium."""
    ui.page_header(
        "Wedge and circular limit equilibrium",
        "Chapter 9 (structural analysis); RSE3010 Week 12",
        "Factor of safety of a tetrahedral rock wedge sliding on two discontinuities, and of a circular "
        "slip through the toe of a slope in heavily fractured or weak rock.",
    )
    notes = [
        "Wedge: rigid tetrahedron bounded by planes A and B, the face and the upper surface; no tension "
        "crack and no external forces; sliding along the line of intersection unless a normal reaction "
        "becomes negative.",
        "Wedge water: dry, or fully saturated with Hoek and Bray's distribution (mean pressure γw H / 6 "
        "on each plane).  This is an idealisation; real water pressures may be lower or higher.",
        "Circular: two-dimensional, per metre run, homogeneous material with Mohr-Coulomb c and φ, pore "
        "pressure u = r_u γ h, no tension crack and no external load; only circles through the toe are "
        "searched (rock slopes, φ > 5°), so deep base circles in very weak material are not.",
        "Bishop's simplified method satisfies moment equilibrium but not horizontal force equilibrium; the "
        "ordinary method is usually conservative, markedly so with high r_u.",
        "Anchor forces are not included on this page; planar sliding with anchors is on the kinematic page.",
    ]
    ui.assumptions(notes)
    mode = st.radio("Failure mode", ["Wedge sliding", "Circular sliding"], horizontal=True, key="lem_mode")
    if mode == "Wedge sliding":
        _wedge_section(notes)
    else:
        _circular_section(notes)


__all__ = ["render_smr", "render_kinematic", "render_slope_lem"]
