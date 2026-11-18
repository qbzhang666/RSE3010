"""Figures for Chapters 10 to 12: classification, intact rock, rock mass parameters.

Every quantity is computed from the book's own equations and data (the four
rocks of Tables tab:classification-input-data and tab:intact_project_data, the
marble example of Chapter 11, the sandstone worked example of Chapter 12),
with the strength and modulus relations taken from the companion package
``geo_platform.rock`` so that a figure and the worked example it illustrates
cannot drift apart. Figures are drawn at printed width (117 mm text block) in
the house style of ``tools/qz_figure.py`` and written as PDF, 300 dpi PNG and
a CSV of the plotted data.

    python tools/figs/f4.py            # all figures
    python tools/figs/f4.py mohr       # figures whose name contains "mohr"
"""

from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
BOOK = os.path.dirname(TOOLS)
REPO = os.path.dirname(BOOK)
sys.path.insert(0, TOOLS)
sys.path.insert(0, REPO)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch, Polygon, Rectangle  # noqa: E402

import qz_figure as qz  # noqa: E402
from geo_platform import rock  # noqa: E402

FIG = os.path.join(BOOK, "figures")
W_FULL = 117 / 25.4
W_HALF = 0.62 * W_FULL

# The four rocks (Chapter 10 Table tab:classification-input-data, Chapter 11
# Table tab:intact_project_data): sigma_ci, m_i, depth, k, E_i, RQD, lab sigma_t.
ROCKS = {
    "Granite":   dict(sci=265.0, mi=32, z=780, k=3.2, ei=80.0, rqd=70, st_lab=-18.4),
    "Sandstone": dict(sci=110.0, mi=17, z=450, k=1.8, ei=30.0, rqd=68, st_lab=-6.7),
    "Mudstone":  dict(sci=27.0, mi=4, z=100, k=1.0, ei=7.6, rqd=38, st_lab=-3.1),
    "Limestone": dict(sci=85.0, mi=12, z=80, k=2.0, ei=20.0, rqd=52, st_lab=-4.5),
}
# Two hues only: the rocks are told apart by shade, line style and marker.
STYLE = {
    "Granite":   dict(color=qz.MAL_SHADES[0], ls="-", marker="o"),
    "Sandstone": dict(color=qz.MAL_SHADES[2], ls="--", marker="s"),
    "Mudstone":  dict(color=qz.CAR, ls=":", marker="^"),
    "Limestone": dict(color=qz.CAR_SHADES[3], ls="-.", marker="D"),
}


def _save(fig, chapter: str, stem: str, columns: dict) -> None:
    out = os.path.join(FIG, chapter)
    qz.save(fig, stem, out)
    keys = list(columns)
    n = max(len(np.atleast_1d(columns[k])) for k in keys)
    with open(os.path.join(out, stem + ".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for i in range(n):
            w.writerow([_cell(columns[k], i) for k in keys])
    plt.close(fig)
    print(f"wrote {chapter}/{stem}")


def _cell(col, i):
    col = np.atleast_1d(col)
    if i >= len(col):
        return ""
    v = col[i]
    return f"{v:.6g}" if isinstance(v, (int, float, np.floating, np.integer)) else str(v)


def _steps(edges, ratings, xmin, xmax):
    """Band rating as a step function. A value on a shared limit takes the band
    of which it is the lower limit (Section sec:rmr-system)."""
    xs = [xmin] + list(edges) + [xmax]
    ys = list(ratings) + [ratings[-1]]
    return np.array(xs, float), np.array(ys, float)


def _rate(x, edges, ratings):
    return ratings[int(np.searchsorted(edges, x, side="right"))]


def _schematic(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(top=False, right=False)


# ===========================================================================
# Chapter 10: classification
# ===========================================================================

R1_EDGES, R1_RAT = [1, 5, 25, 50, 100, 250], [0, 1, 2, 4, 7, 12, 15]
R2_EDGES, R2_RAT = [25, 50, 75, 90], [3, 8, 13, 17, 20]
R3_EDGES, R3_RAT = [0.06, 0.2, 0.6, 2.0], [5, 8, 10, 15, 20]
R5_EDGES, R5_RAT = [0.1, 0.2, 0.5], [10, 7, 4, 0]     # u/sigma_1 > 0; exactly 0 rates 15


def rmr_ratings():
    fig, axs = qz.figure(W_FULL, 0.78 * W_FULL, nrows=2, ncols=2)
    (a1, a2), (a3, a5) = axs
    cols = {}

    x, y = _steps(R1_EDGES, R1_RAT, 0.5, 500)
    a1.step(x, y, where="post", color=qz.INK, lw=1.1)
    a1.set_xscale("log")
    a1.set_xlim(0.5, 500)
    cols.update(R1_edges=x, R1=y)
    for name, d in ROCKS.items():
        st = STYLE[name]
        a1.plot(d["sci"], _rate(d["sci"], R1_EDGES, R1_RAT), st["marker"], color=st["color"],
                mfc="white", mew=1.1, ms=5.5, label=name, zorder=4)
    qz.label(a1, "Uniaxial strength, $\\sigma_{ci}$ (MPa)", "Rating, $R_1$ (-)")
    a1.set_ylim(-1, 17)
    a1.legend(loc="upper left", handletextpad=0.3, labelspacing=0.25)

    x, y = _steps(R2_EDGES, R2_RAT, 0, 100)
    a2.step(x, y, where="post", color=qz.INK, lw=1.1)
    a2.set_xlim(0, 100)
    a2.set_ylim(0, 23)
    cols.update(R2_edges=x, R2=y)
    for name, d in ROCKS.items():
        st = STYLE[name]
        a2.plot(d["rqd"], _rate(d["rqd"], R2_EDGES, R2_RAT), st["marker"], color=st["color"],
                mfc="white", mew=1.1, ms=5.5, zorder=4)
    qz.label(a2, "RQD (%)", "Rating, $R_2$ (-)")

    x, y = _steps(R3_EDGES, R3_RAT, 0.02, 5)
    a3.step(x, y, where="post", color=qz.INK, lw=1.1)
    a3.set_xscale("log")
    a3.set_xlim(0.02, 5)
    a3.set_ylim(0, 23)
    cols.update(R3_edges=x, R3=y)
    s_sand = 1 / ((110.4 - 68) / 3.68)           # Priest and Hudson, RQD 68 -> 0.087 m
    for name, s in (("Sandstone", s_sand), ("Limestone", 0.06)):
        st = STYLE[name]
        a3.plot(s, _rate(s, R3_EDGES, R3_RAT), st["marker"], color=st["color"],
                mfc="white", mew=1.1, ms=5.5, zorder=4)
    qz.label(a3, "Discontinuity spacing, $s$ (m)", "Rating, $R_3$ (-)")

    x, y = _steps(R5_EDGES, R5_RAT, 0, 0.7)
    a5.step(x, y, where="post", color=qz.INK, lw=1.1)
    a5.plot([0], [15], "o", color=qz.INK, ms=3.5, zorder=4)
    a5.set_xlim(-0.02, 0.7)
    a5.set_ylim(-1, 17)
    cols.update(R5_edges=x, R5=y)
    u_ratio = 0.69 / 4.32                          # limestone, Section sec:classification-design-data
    st = STYLE["Limestone"]
    a5.plot(u_ratio, _rate(u_ratio, R5_EDGES, R5_RAT), st["marker"], color=st["color"],
            mfc="white", mew=1.1, ms=5.5, zorder=4)
    qz.label(a5, "Pressure ratio, $u/\\sigma_1$ (-)", "Rating, $R_5$ (-)")
    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    _save(fig, "ch10", "rmr-ratings", cols)


def rqd_frequency():
    lam = np.linspace(0, 30, 301)
    exact = 100 * np.exp(-0.1 * lam) * (0.1 * lam + 1)
    linear = 110.4 - 3.68 * lam
    fig, ax = qz.figure(W_HALF, 0.82 * W_HALF)
    ax.axvspan(6, 16, color=qz.MAL_TINT, lw=0, zorder=0)
    ax.plot(lam, exact, color=qz.GREY, lw=1.1, label="Exponential")
    inside = (lam >= 6) & (lam <= 16)
    ax.plot(lam[inside], linear[inside], color=qz.MAL, lw=1.6, label="Linear")
    for part in (lam <= 6, (lam >= 16) & (linear >= 0)):
        ax.plot(lam[part], linear[part], color=qz.MAL, lw=1.0, ls="--")
    lam68 = (110.4 - 68) / 3.68
    ax.plot([lam68, lam68], [0, 68], color=qz.CAR, lw=0.7, ls=":")
    ax.plot([0, lam68], [68, 68], color=qz.CAR, lw=0.7, ls=":")
    ax.plot(lam68, 68, "o", color=qz.CAR, ms=4.5, zorder=4)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 115)
    qz.label(ax, "Discontinuity frequency, $\\lambda$ (m$^{-1}$)", "RQD (%)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "ch10", "rqd-frequency", {"lambda_per_m": lam, "RQD_exponential": exact,
                                         "RQD_linear": linear})


def q_quotient_ranges():
    # Extremes of each quotient from Tables tab:jn to tab:srf, RQD 10 to 100.
    rows = [
        ("$RQD/J_n$", 10 / 20, 100 / 0.5, qz.MAL_SHADES[0]),
        ("$J_r/J_a$", 0.5 / 20, 4 / 0.75, qz.MAL_SHADES[2]),
        ("$J_w/SRF$", 0.05 / 400, 1.0 / 0.5, qz.CAR),
    ]
    qmin = np.prod([r[1] for r in rows])
    qmax = np.prod([r[2] for r in rows])
    rows.append(("$Q$", qmin, qmax, qz.GREY))
    fig, ax = qz.figure(W_FULL, 0.36 * W_FULL)
    for lim in (0.01, 0.1, 1, 4, 10, 40, 100, 400):
        ax.axvline(lim, color=qz.GREY_LIGHT, lw=0.6, zorder=0)
    for i, (name, lo, hi, col) in enumerate(rows):
        yy = len(rows) - 1 - i
        ax.plot([lo, hi], [yy, yy], color=col, lw=5, solid_capstyle="butt")
        ax.text(hi * 1.6, yy, f"{math.log10(hi / lo):.1f}", va="center", ha="left", fontsize=8,
                color=qz.GREY)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.tick_params(axis="y", length=0, right=False)
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1e5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1, 1e2, 1e4])
    ax.minorticks_off()
    qz.label(ax, "Value of the quotient (-)")
    fig.tight_layout()
    _save(fig, "ch10", "q-quotient-ranges", {"quotient": [r[0] for r in rows],
                                             "min": [r[1] for r in rows],
                                             "max": [r[2] for r in rows],
                                             "decades": [math.log10(r[2] / r[1]) for r in rows]})


def gsi_correlations():
    fig, (a1, a2) = qz.figure(W_FULL, 0.45 * W_FULL, ncols=2, sharey=True)
    jc = np.linspace(0, 30, 61)
    x = np.logspace(math.log10(0.025), math.log10(5.4), 200)
    cols = {"Jcond89": jc, "Jr_over_Ja": x}
    shades = [qz.MAL_SHADES[3], qz.MAL_SHADES[2], qz.MAL_SHADES[1], qz.MAL_SHADES[0]]
    for rqd, col, ls in zip((25, 50, 75, 100), shades, (":", "-.", "--", "-")):
        g1 = 1.5 * jc + rqd / 2
        g2 = 52 * x / (1 + x) + rqd / 2
        a1.plot(jc, g1, color=col, ls=ls, lw=1.2, label=f"RQD = {rqd}")
        a2.plot(x, g2, color=col, ls=ls, lw=1.2)
        cols[f"GSI_a_RQD{rqd}"] = g1
        cols[f"GSI_b_RQD{rqd}"] = g2
    # Sandstone: R4 = 20 (slightly rough, highly weathered), RQD 68.
    a1.plot(20, 1.5 * 20 + 68 / 2, "s", color=qz.CAR, ms=5, zorder=4)
    a1.set_xlim(0, 30)
    a1.set_ylim(0, 100)
    a1.set_xticks([0, 10, 20, 30])
    qz.label(a1, "Condition rating, $J_{\\mathrm{cond89}}$ (-)", "GSI (-)")
    for rqd in (25, 50, 75, 100):
        a2.text(0.032, 52 * 0.032 / 1.032 + rqd / 2 + 1.5, f"{rqd}", fontsize=8, color=qz.GREY,
                ha="left", va="bottom")
    a2.set_xscale("log")
    a2.set_xlim(0.025, 5.4)
    a2.set_xticks([0.03, 0.1, 0.3, 1, 3])
    a2.set_xticklabels(["0.03", "0.1", "0.3", "1", "3"])
    a2.minorticks_off()
    qz.label(a2, "Joint quotient, $J_r/J_a$ (-)")
    fig.tight_layout(w_pad=1.0)
    _save(fig, "ch10", "gsi-correlations", cols)


def rmr_sandstone():
    # Sandstone of Table tab:classification-input-data: R1 from sigma_ci 110,
    # R2 from RQD 68, R3 from s = 0.087 m (Priest and Hudson), R4 slightly rough
    # and highly weathered, R5 wet. Basic RMR 60; dry RMR'89 68; GSI 63.
    r = {"$R_1$": 12, "$R_2$": 13, "$R_3$": 8, "$R_4$": 20}
    rows = (("Basic", 7), ("Dry", 15))
    fills = [qz.MAL_SHADES[0], qz.MAL_SHADES[1], qz.MAL_SHADES[2], qz.MAL_SHADES[3]]
    fig, ax = qz.figure(W_HALF, 0.5 * W_HALF)
    for yy, (name, r5) in zip((1, 0), rows):
        left = 0
        for (lab, v), col in zip(r.items(), fills):
            ax.barh(yy, v, left=left, height=0.55, color=col, edgecolor="white", lw=0.6)
            ax.text(left + v / 2, yy, lab, ha="center", va="center", fontsize=8,
                    color="white" if col != qz.MAL_SHADES[3] else qz.INK)
            left += v
        ax.barh(yy, r5, left=left, height=0.55, color=qz.CAR if r5 < 15 else qz.CAR_SHADES[3],
                edgecolor="white", lw=0.6)
        ax.text(left + r5 / 2, yy, "$R_5$", ha="center", va="center", fontsize=8, color="white")
        total = left + r5
        ax.text(total + 1.5, yy, f"{total}, GSI {total - 5}", va="center", ha="left", fontsize=8)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Basic RMR", "$\\mathrm{RMR}'_{89}$"])
    ax.tick_params(axis="y", length=0, right=False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 1.55)
    qz.label(ax, "Rating (-)")
    fig.tight_layout()
    _save(fig, "ch10", "rmr-sandstone", {"row": ["basic", "dry"], "R1": [12, 12], "R2": [13, 13],
                                         "R3": [8, 8], "R4": [20, 20], "R5": [7, 15],
                                         "RMR": [60, 68], "GSI_from_row": [55, 63]})


# ===========================================================================
# Chapter 11: stress and intact rock
# ===========================================================================

def insitu_stress_depth():
    fig, (a1, a2) = qz.figure(W_FULL, 0.52 * W_FULL, ncols=2)
    cols = {}
    for ax, names, zmax, smax in ((a1, ("Granite", "Sandstone"), 900, 100),
                                  (a2, ("Mudstone", "Limestone"), 120, 12)):
        z = np.linspace(0, zmax, 50)
        sv = 0.027 * z
        generic = ax is a2
        ax.plot(sv, z, color=qz.INK, lw=1.3, label="$\\sigma_v = \\gamma z$" if generic else None)
        ax.plot(0.33 * sv, z, color=qz.GREY, lw=1.0, ls=(0, (1, 1.5)),
                label="$k = 0.33$" if generic else None)
        for name in names:
            d, st = ROCKS[name], STYLE[name]
            ins = rock.insitu_stresses(d["z"], d["k"])
            if d["k"] != 1.0:
                ax.plot(d["k"] * sv, z, color=st["color"], ls=st["ls"], lw=1.0)
            ax.plot([ins.sigma_v, ins.sigma_h], [d["z"], d["z"]], color=st["color"], lw=0.8)
            ax.plot(ins.sigma_v, d["z"], st["marker"], color=st["color"], mfc="white", mew=1.1, ms=5)
            ax.plot(ins.sigma_h, d["z"], st["marker"], color=st["color"], ms=5,
                    label=f"{name}, $k = {d['k']:g}$")
            cols[f"{name}_z"] = [d["z"]]
            cols[f"{name}_sv"] = [ins.sigma_v]
            cols[f"{name}_sh"] = [ins.sigma_h]
        ax.set_xlim(0, smax)
        if smax == 12:
            ax.set_xticks([0, 3, 6, 9, 12])
        ax.set_ylim(zmax, 0)
        qz.label(ax, "Stress, $\\sigma$ (MPa)", "Depth, $z$ (m)")
        ax.legend(loc="upper right", handlelength=1.8, labelspacing=0.25)
    fig.tight_layout(w_pad=1.2)
    _save(fig, "ch11", "insitu-stress-depth", cols)


def post_peak_behaviour():
    e = np.linspace(0, 1, 400)
    ep = 0.3
    pre = np.where(e < ep, e / ep, 1.0)
    pre = 1 - (1 - np.clip(e / ep, 0, 1)) ** 2          # gently curved pre-peak branch
    plastic = np.where(e < ep, pre, 1.0)
    soft = np.where(e < ep, pre, 0.45 + 0.55 * np.exp(-(e - ep) / 0.18))
    brittle = np.where(e < ep, pre, np.where(e < ep + 0.02, 1 - (e - ep) / 0.02 * 0.75, 0.25))
    fig, ax = qz.figure(W_HALF, 0.72 * W_HALF)
    ax.plot(e, plastic, color=qz.MAL_SHADES[0], lw=1.4, label="Perfectly plastic")
    ax.plot(e, soft, color=qz.MAL_SHADES[2], lw=1.4, ls="--", label="Strain-softening")
    ax.plot(e, brittle, color=qz.CAR, lw=1.4, ls=":", label="Brittle")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.7)
    _schematic(ax)
    qz.label(ax, "Axial strain, $\\varepsilon_1$", "Deviatoric stress, $\\sigma_1 - \\sigma_3$")
    ax.legend(loc="upper right", handlelength=2.2)
    fig.tight_layout()
    _save(fig, "ch11", "post-peak-behaviour", {"strain": e, "plastic": plastic,
                                               "softening": soft, "brittle": brittle})


def mohr_coulomb_plane():
    phi = 30.0
    c = 10.0
    s3 = 10.0
    s1 = rock.mc_sigma1(s3, c, phi)
    C, R = (s1 + s3) / 2, (s1 - s3) / 2
    tphi = math.tan(math.radians(phi))
    fig = plt.figure(figsize=(W_FULL, 0.44 * W_FULL))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.6, 1.4], wspace=0.08)
    a1 = fig.add_subplot(gs[0])
    a2 = fig.add_subplot(gs[1])

    # Specimen with sigma_1 vertical, sigma_3 horizontal, plane at 45 - phi/2 to sigma_1.
    a1.set_aspect("equal")
    a1.axis("off")
    w, h = 1.0, 2.0
    a1.add_patch(Rectangle((-w / 2, -h / 2), w, h, fc=qz.MAL_TINT, ec=qz.INK, lw=0.8))
    beta = math.radians(45 - phi / 2)                   # angle from the sigma_1 direction
    dx = math.tan(beta) * h / 2
    a1.plot([-dx, dx], [-h / 2, h / 2], color=qz.CAR, lw=1.6)
    arr = dict(arrowstyle="-|>", mutation_scale=9, color=qz.INK, lw=1.0)
    for sgn in (1, -1):
        for xo in (-0.3, 0, 0.3):
            a1.add_patch(FancyArrowPatch((xo, sgn * (h / 2 + 0.45)), (xo, sgn * h / 2), **arr))
        for yo in (-0.5, 0.5):
            a1.add_patch(FancyArrowPatch((sgn * (w / 2 + 0.35), yo), (sgn * w / 2, yo), **arr))
    a1.text(0.38, h / 2 + 0.3, "$\\sigma_1$", fontsize=10, va="center")
    a1.text(w / 2 + 0.18, 0.0, "$\\sigma_3$", fontsize=10, ha="center", va="center")
    # angle between the plane and the sigma_3 direction, measured at the plane centre
    a1.plot([0, 0.42], [0, 0], color=qz.GREY, lw=0.6, ls="--")
    theta = 45 + phi / 2
    a1.add_patch(Arc((0, 0), 0.62, 0.62, theta1=0, theta2=theta, color=qz.INK, lw=0.7))
    a1.text(0.36, 0.2, "$\\theta$", fontsize=9)
    a1.set_xlim(-1.05, 1.05)
    a1.set_ylim(-h / 2 - 0.55, h / 2 + 0.55)

    sn = np.linspace(-c / tphi, s1 + 6, 100)
    a2.plot(sn, c + sn * tphi, color=qz.CAR, lw=1.3)
    t = np.linspace(0, math.pi, 200)
    a2.plot(C + R * np.cos(t), R * np.sin(t), color=qz.MAL, lw=1.4)
    a2.axhline(0, color=qz.INK, lw=0.6)
    a2.axvline(0, color=qz.INK, lw=0.6)
    two_theta = math.radians(90 + phi)
    tx, ty = C + R * math.cos(two_theta), R * math.sin(two_theta)
    a2.plot([C, tx], [0, ty], color=qz.GREY, lw=0.7)
    a2.plot(tx, ty, "o", color=qz.CAR, ms=4, zorder=4)
    # 2 theta measured from the sigma_1 end, anticlockwise, to the tangent point
    a2.add_patch(Arc((C, 0), 16, 16, theta1=0, theta2=90 + phi, color=qz.INK, lw=0.7))
    a2.text(C + 8.5 * math.cos(math.radians(60)), 8.5 * math.sin(math.radians(60)) + 1.0,
            "$2\\theta$", fontsize=9)
    a2.add_patch(Arc((-c / tphi, 0), 14, 14, theta1=0, theta2=phi, color=qz.INK, lw=0.7))
    a2.text(-c / tphi + 8.0, 1.5, "$\\phi$", fontsize=9)
    a2.annotate("", xy=(0, c), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", lw=0.6, color=qz.INK,
                                                              shrinkA=0, shrinkB=0))
    a2.text(-1.2, c / 2, "$c$", fontsize=9, ha="right", va="center")
    a2.text(s3, -1.5, "$\\sigma_3$", fontsize=9, ha="center", va="top")
    a2.text(s1, -1.5, "$\\sigma_1$", fontsize=9, ha="center", va="top")
    a2.set_aspect("equal")
    a2.set_xlim(-c / tphi - 2, s1 + 14)
    a2.set_ylim(-7, 45)
    for side in ("top", "right"):
        a2.spines[side].set_visible(False)
    a2.spines["left"].set_visible(False)
    a2.spines["bottom"].set_visible(False)
    _schematic(a2)
    a2.text(s1 + 13, -1.5, "$\\sigma_n$", fontsize=10, ha="right", va="top")
    a2.text(1.0, 44, "$\\tau$", fontsize=10, ha="left", va="top")
    _save(fig, "ch11", "mohr-coulomb-plane", {"c": [c], "phi_deg": [phi], "sigma3": [s3],
                                              "sigma1": [float(s1)], "theta_deg": [theta]})


def hb_intact_four_rocks():
    x = np.linspace(0, 0.15, 200)
    fig, ax = qz.figure(W_FULL, 0.5 * W_FULL)
    cols = {"sigma3_over_sci": x}
    for name, d in ROCKS.items():
        st = STYLE[name]
        y = rock.hb_sigma1(x * d["sci"], d["sci"], d["mi"]) / d["sci"]
        cols[name] = y
        ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.3)
        ins = rock.insitu_stresses(d["z"], d["k"])
        cap = float(rock.hb_sigma1(ins.sigma_3, d["sci"], d["mi"]))
        xs = ins.sigma_3 / d["sci"]
        ax.plot([xs, xs], [ins.sigma_1 / d["sci"], cap / d["sci"]], color=st["color"], lw=0.6)
        ax.plot(xs, cap / d["sci"], st["marker"], color=st["color"], ms=5, zorder=4)
        ax.plot(xs, ins.sigma_1 / d["sci"], st["marker"], color=st["color"], mfc="white", mew=1.1,
                ms=5, zorder=4)
        cols[f"{name}_point"] = [xs, ins.sigma_1 / d["sci"], cap / d["sci"]]
    handles = [Line2D([], [], color=STYLE[n]["color"], ls=STYLE[n]["ls"], marker=STYLE[n]["marker"],
                      ms=4.5, label=f"{n}, $m_i = {ROCKS[n]['mi']}$") for n in ROCKS]
    handles += [Line2D([], [], color=qz.GREY, ls="none", marker="o", ms=4.5, label="Capacity, $\\sigma_1^f$"),
                Line2D([], [], color=qz.GREY, ls="none", marker="o", mfc="white", ms=4.5,
                       label="Demand, $\\sigma_1^{IS}$")]
    ax.legend(handles=handles, loc="upper left", ncol=2, handlelength=2.4, columnspacing=1.2,
              labelspacing=0.3)
    ax.set_xlim(0, 0.15)
    ax.set_ylim(0, 3.5)
    qz.label(ax, "Confinement ratio, $\\sigma_3/\\sigma_{ci}$ (-)",
             "Strength ratio, $\\sigma_1/\\sigma_{ci}$ (-)")
    fig.tight_layout()
    _save(fig, "ch11", "hb-intact-four-rocks", cols)


def tensile_four_rocks():
    names = list(ROCKS)
    fig, ax = qz.figure(W_HALF, 0.68 * W_HALF)
    cols = {"rock": names, "HB": [], "HM": [], "lab": []}
    for i, name in enumerate(names):
        d = ROCKS[name]
        yy = len(names) - 1 - i
        hb = abs(rock.hb_intact_tension(d["sci"], d["mi"]))
        hm = abs(rock.hoek_martin_tension(d["sci"], d["mi"]))
        lab = abs(d["st_lab"])
        ax.plot([min(hb, hm, lab), max(hb, hm, lab)], [yy, yy], color=qz.GREY_LIGHT, lw=1.0, zorder=1)
        ax.plot(hb, yy, "o", color=qz.MAL, ms=5, zorder=3, label="Criterion" if i == 0 else None)
        ax.plot(hm, yy, "s", color=qz.MAL, mfc="white", mew=1.1, ms=5, zorder=3,
                label="Cut-off" if i == 0 else None)
        ax.plot(lab, yy, "D", color=qz.CAR, ms=5, zorder=3, label="Laboratory" if i == 0 else None)
        cols["HB"].append(-hb)
        cols["HM"].append(-hm)
        cols["lab"].append(-lab)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.tick_params(axis="y", length=0, right=False)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(0, 20)
    qz.label(ax, "Tensile strength, $|\\sigma_t|$ (MPa)")
    ax.legend(loc="center right", handletextpad=0.3, labelspacing=0.25)
    fig.tight_layout()
    _save(fig, "ch11", "tensile-four-rocks", cols)


def griffith_hb_marble():
    sci, mi = 125.0, 9.0
    st_hb = rock.hb_intact_tension(sci, mi)
    st_hm = rock.hoek_martin_tension(sci, mi)
    T = sci / 8                                     # Griffith: sigma_c = 8 |sigma_t|
    s3 = np.linspace(st_hb, 40, 400)
    hb = rock.hb_sigma1(s3, sci, mi)
    s3g = np.linspace(-T, 40, 400)
    gr = s3g + 4 * T + 4 * np.sqrt(T * (T + s3g))
    fig, ax = qz.figure(W_FULL, 0.58 * W_FULL)
    ax.axvline(0, color=qz.GREY_LIGHT, lw=0.6, zorder=0)
    ax.plot(s3, hb, color=qz.MAL, lw=1.5, label="Criterion, $m_i = 9$")
    ax.plot(s3g, gr, color=qz.GREY, lw=1.2, ls="--", label="Griffith, $\\sigma_c = 8|\\sigma_t|$")
    ax.plot([-T, -T], [0, 3 * T], color=qz.GREY, lw=1.2, ls="--")
    ax.plot(st_hb, 0, "o", color=qz.MAL, ms=5, zorder=4, clip_on=False)
    ax.plot(st_hm, 0, "s", color=qz.CAR, ms=5, zorder=4, clip_on=False, label="Cut-off")
    ax.plot(20, float(rock.hb_sigma1(20, sci, mi)), "o", color=qz.MAL, mfc="white", mew=1.1, ms=5, zorder=4)
    ax.set_xlim(-20, 40)
    ax.set_ylim(0, 300)
    qz.label(ax, "Minor principal stress, $\\sigma_3$ (MPa)", "Major principal stress, $\\sigma_1$ (MPa)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "ch11", "griffith-hb-marble", {"sigma3_hb": s3, "sigma1_hb": hb, "sigma3_griffith": s3g,
                                              "sigma1_griffith": gr, "st_hb": [st_hb], "st_hm": [st_hm],
                                              "T_griffith": [T]})


def mohr_fit_sandstone():
    sci, mi, s3max = 110.0, 17.0, 21.87
    p3, p1 = rock.hb_points(sci, mi, 1.0, 0.5, s3max, 6)
    m1 = rock.fit_mohr_circles(p3, p1)
    C, R = (p1 + p3) / 2, (p1 - p3) / 2
    fig = plt.figure(figsize=(W_FULL, 0.45 * W_FULL))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.32)
    a1 = fig.add_subplot(gs[0])
    a2 = fig.add_subplot(gs[1])
    t = np.linspace(0, math.pi, 200)
    shades = np.linspace(0.25, 1.0, len(C))
    for i, (cc, rr) in enumerate(zip(C, R)):
        a1.plot(cc + rr * np.cos(t), rr * np.sin(t), color=qz.MAL, alpha=float(shades[i]), lw=1.0)
    tphi = math.tan(math.radians(m1.phi_deg))
    sn = np.linspace(0, 270, 50)
    a1.plot(sn, m1.c + sn * tphi, color=qz.CAR, ls="--", lw=1.2)
    a1.set_aspect("equal")
    a1.set_xlim(0, 270)
    a1.set_ylim(0, 150)
    qz.label(a1, "Normal stress, $\\sigma_n$ (MPa)", "Shear stress, $\\tau$ (MPa)")
    a2.plot(C, R, "o", color=qz.MAL, ms=4.5)
    cc = np.linspace(40, 150, 20)
    a2.plot(cc, m1.intercept + m1.slope * cc, color=qz.CAR, ls="--", lw=1.2)
    a2.set_box_aspect(0.97)
    a2.set_xlim(40, 150)
    a2.set_ylim(40, 130)
    qz.label(a2, "Centre, $C$ (MPa)", "Radius, $R$ (MPa)")
    _save(fig, "ch11", "mohr-fit-sandstone", {"sigma3": p3, "sigma1": p1, "C": C, "R": R,
                                              "A": [m1.intercept], "B": [m1.slope],
                                              "phi_deg": [m1.phi_deg], "c": [m1.c]})


# ===========================================================================
# Chapter 12: rock mass parameters
# ===========================================================================

def hb_constants_gsi():
    gsi = np.linspace(10, 100, 181)
    fig, (a1, a2) = qz.figure(W_FULL, 0.47 * W_FULL, ncols=2)
    cols = {"GSI": gsi}
    for d, col in ((0.0, qz.MAL), (0.8, qz.CAR)):
        hb = [rock.hb_constants(g, 1.0, d) for g in gsi]
        mb = np.array([h.mb for h in hb])
        s = np.array([h.s for h in hb])
        a1.plot(gsi, mb, color=col, lw=1.3, label=f"$m_b/m_i$, $D = {d:g}$")
        a1.plot(gsi, s, color=col, lw=1.3, ls="--", label=f"$s$, $D = {d:g}$")
        cols[f"mb_over_mi_D{d:g}"] = mb
        cols[f"s_D{d:g}"] = s
    a1.set_yscale("log")
    a1.set_ylim(1e-7, 2)
    a1.set_yticks([1e-6, 1e-4, 1e-2, 1])
    a1.minorticks_off()
    a1.set_xlim(10, 100)
    qz.label(a1, "GSI (-)", "Rock mass constant (-)")
    a1.legend(loc="lower right", labelspacing=0.25, handlelength=2.0)
    for d, col, ls in ((0.0, qz.MAL_SHADES[0], "-"), (0.5, qz.MAL_SHADES[2], "--"),
                       (0.8, qz.CAR, "-."), (1.0, qz.CAR_SHADES[3], ":")):
        r = np.array([rock.rock_mass_ucs(1.0, rock.hb_constants(g, 1.0, d)) for g in gsi])
        a2.plot(gsi, r, color=col, ls=ls, lw=1.3, label=f"$D = {d:g}$")
        cols[f"scm_over_sci_D{d:g}"] = r
    a2.set_yscale("log")
    a2.set_ylim(1e-3, 1.5)
    a2.set_xlim(10, 100)
    qz.label(a2, "GSI (-)", "Strength ratio, $\\sigma_{cm}/\\sigma_{ci}$ (-)")
    a2.legend(loc="lower right", labelspacing=0.25)
    for ax in (a1, a2):
        ax.axvline(63, color=qz.GREY, lw=0.6, ls=":")
    fig.tight_layout(w_pad=1.2)
    _save(fig, "ch12", "hb-constants-gsi", cols)


def hb_disturbance_sandstone():
    sci, mi, gsi = 110.0, 17, 63
    ins = rock.insitu_stresses(450, 1.8)
    fig, (a1, a2) = qz.figure(W_FULL, 0.47 * W_FULL, ncols=2)
    cols = {}
    for d, col, ls, mk in ((0.0, qz.MAL, "-", "o"), (0.8, qz.CAR, "--", "s")):
        hb = rock.hb_constants(gsi, mi, d)
        st = rock.rock_mass_tension(sci, hb)
        sthm = rock.rock_mass_tension_hm(sci, hb, mi)
        s3 = np.concatenate([np.linspace(st, 0, 200), np.linspace(0, 25, 200)[1:]])
        s1 = rock.hb_sigma1(s3, sci, hb.mb, hb.s, hb.a)
        cap = float(rock.hb_sigma1(ins.sigma_3, sci, hb.mb, hb.s, hb.a))
        a1.plot(s3, s1, color=col, ls=ls, lw=1.4, label=f"$D = {d:g}$")
        a1.plot(ins.sigma_3, cap, mk, color=col, ms=5, zorder=4)
        a2.plot(s3, s1, color=col, ls=ls, lw=1.4)
        a2.plot(st, 0, mk, color=col, ms=5, zorder=4, clip_on=False)
        a2.plot(sthm, 0, mk, color=col, mfc="white", mew=1.1, ms=5, zorder=4, clip_on=False)
        cols[f"sigma3_D{d:g}"] = s3
        cols[f"sigma1_D{d:g}"] = s1
        cols[f"capacity_D{d:g}"] = [cap]
        cols[f"sigma_tm_D{d:g}"] = [st]
        cols[f"sigma_tm_HM_D{d:g}"] = [sthm]
    a1.plot([ins.sigma_3, ins.sigma_3], [ins.sigma_1, 100], color=qz.GREY_LIGHT, lw=0.7, zorder=0)
    a1.plot(ins.sigma_3, ins.sigma_1, "o", color=qz.INK, mfc="white", mew=1.1, ms=5, zorder=4,
            label="$\\sigma_1^{IS}$")
    a1.set_xlim(0, 25)
    a1.set_ylim(0, 160)
    qz.label(a1, "Minor stress, $\\sigma_3$ (MPa)", "Major stress, $\\sigma_1$ (MPa)")
    a1.legend(loc="upper left")
    a2.axvline(0, color=qz.GREY_LIGHT, lw=0.6, zorder=0)
    a2.set_xlim(-0.8, 0.8)
    a2.set_ylim(0, 25)
    qz.label(a2, "Minor stress, $\\sigma_3$ (MPa)", "Major stress, $\\sigma_1$ (MPa)")
    handles = [Line2D([], [], color=qz.GREY, ls="none", marker="o", ms=4.5, label="$\\sigma_{tm}$"),
               Line2D([], [], color=qz.GREY, ls="none", marker="o", mfc="white", ms=4.5,
                      label="$\\sigma_{tm,HM}$")]
    a2.legend(handles=handles, loc="upper left")
    cols["insitu"] = [ins.sigma_3, ins.sigma_1]
    fig.tight_layout(w_pad=1.2)
    _save(fig, "ch12", "hb-disturbance-sandstone", cols)


def modulus_estimates_sandstone():
    sci, ei, gsi, rmr = 110.0, 30.0, 63, 60
    rows = [
        ("Generalised, from $E_i$", lambda d: rock.modulus_hd_generalised(ei, gsi, d), True),
        ("Simplified", lambda d: rock.modulus_hd_simplified(gsi, d), True),
        ("From $\\sigma_{ci}$ and GSI", lambda d: rock.modulus_hoek_2002(sci, gsi, d), True),
        ("From RMR, linear", lambda d: rock.modulus_bieniawski(rmr), False),
        ("From RMR, power", lambda d: rock.modulus_serafim_pereira(rmr), False),
    ]
    fig, ax = qz.figure(W_FULL, 0.42 * W_FULL)
    cols = {"relation": [r[0] for r in rows], "D0": [], "D0.8": []}
    for i, (name, f, has_d) in enumerate(rows):
        yy = len(rows) - 1 - i
        e0 = f(0.0)
        ax.plot(e0, yy, "o", color=qz.MAL, ms=5, zorder=3, label="$D = 0$" if i == 0 else None)
        cols["D0"].append(e0)
        if has_d:
            e8 = f(0.8)
            ax.plot([e8, e0], [yy, yy], color=qz.GREY_LIGHT, lw=1.0, zorder=1)
            ax.plot(e8, yy, "s", color=qz.CAR, ms=5, zorder=3, label="$D = 0.8$" if i == 0 else None)
            cols["D0.8"].append(e8)
        else:
            cols["D0.8"].append("")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.tick_params(axis="y", length=0, right=False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 30)
    qz.label(ax, "Rock mass modulus, $E_m$ (GPa)")
    ax.legend(loc="lower right", handletextpad=0.3)
    fig.tight_layout()
    _save(fig, "ch12", "modulus-estimates-sandstone", cols)


def fitting_range_sandstone():
    sci, mi, gsi = 110.0, 17, 63
    smax = np.linspace(1, 40, 157)
    fig, (a1, a2) = qz.figure(W_FULL, 0.45 * W_FULL, ncols=2)
    cols = {"sigma3_fit_max": smax}
    cases = (("Intact", 1.0, mi, 1.0, 0.5, qz.GREY, ":"),)
    for d, col, ls in ((0.0, qz.MAL, "-"), (0.8, qz.CAR, "--")):
        hb = rock.hb_constants(gsi, mi, d)
        cases += ((f"Rock mass, $D = {d:g}$", None, hb.mb, hb.s, hb.a, col, ls),)
    for name, _, mb, s, a, col, ls in cases:
        fits = [rock.fit_principal_line(*rock.hb_points(sci, mb, s, a, m, 6)) for m in smax]
        phi = np.array([f.phi_deg for f in fits])
        c = np.array([f.c for f in fits])
        a1.plot(smax, phi, color=col, ls=ls, lw=1.3, label=name)
        a2.plot(smax, c, color=col, ls=ls, lw=1.3)
        key = name.split(",")[0].replace(" ", "_") + ("" if "D" not in name else name[-4:].replace(" ", ""))
        cols[f"phi_{key}"] = phi
        cols[f"c_{key}"] = c
        for m in (12.15, 21.87):
            f = rock.fit_principal_line(*rock.hb_points(sci, mb, s, a, m, 6))
            a1.plot(m, f.phi_deg, "o", color=col, ms=4, zorder=4)
            a2.plot(m, f.c, "o", color=col, ms=4, zorder=4)
    for ax in (a1, a2):
        for m in (12.15, 21.87):
            ax.axvline(m, color=qz.GREY_LIGHT, lw=0.6, zorder=0)
        ax.set_xlim(0, 40)
    a1.set_ylim(25, 70)
    qz.label(a1, "Fitting limit, $\\sigma_{3,\\mathrm{fit,max}}$ (MPa)", "Friction angle, $\\phi$ ($^\\circ$)")
    a1.legend(loc="upper right", labelspacing=0.25, handlelength=2.0)
    a2.set_yscale("log")
    a2.set_ylim(1, 50)
    a2.set_yticks([1, 2, 5, 10, 20, 50])
    a2.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    a2.minorticks_off()
    qz.label(a2, "Fitting limit, $\\sigma_{3,\\mathrm{fit,max}}$ (MPa)", "Cohesion, $c$ (MPa)")
    fig.tight_layout(w_pad=1.2)
    _save(fig, "ch12", "fitting-range-sandstone", cols)


FIGURES = [rmr_ratings, rqd_frequency, q_quotient_ranges, gsi_correlations, rmr_sandstone,
           insitu_stress_depth, post_peak_behaviour, mohr_coulomb_plane, hb_intact_four_rocks,
           tensile_four_rocks, griffith_hb_marble, mohr_fit_sandstone,
           hb_constants_gsi, hb_disturbance_sandstone, modulus_estimates_sandstone,
           fitting_range_sandstone]


if __name__ == "__main__":
    qz.use()
    pick = sys.argv[1:]
    for f in FIGURES:
        name = f.__name__.replace("_", "-")
        if not pick or any(p in name for p in pick):
            f()
