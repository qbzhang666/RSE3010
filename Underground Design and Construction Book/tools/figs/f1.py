"""Figures for Chapters 1 to 4 (Part I context and Chapter 4 engineering geology).

Data figures are computed from the book's own tables and equations, with the
companion ``geo_platform`` package where it holds the calculation; schematic
figures are drawn in the same house style with plausible, illustrative
geometry (each caption says so). Every figure is written as PDF and 300 dpi
PNG at its printed width, with the plotted or defining data as CSV beside it.

    python tools/figs/f1.py              # all figures
    python tools/figs/f1.py carbon       # figures whose name contains "carbon"
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
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,  # noqa: E402
                                Polygon, Rectangle)

import qz_figure as qz  # noqa: E402
from geo_platform import rock, settlement  # noqa: E402

FIG = os.path.join(BOOK, "figures")
W_FULL = 117 / 25.4
W_HALF = 0.62 * W_FULL
FS = 8          # schematic annotation size (pt)


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


def _box(ax, x, y, w, h, text, ec=qz.MAL, fc="white", fs=FS, lw=0.9, weight="normal", tc=qz.INK):
    """A rounded box centred at (x, y) in axes data coordinates."""
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.02",
                       ec=ec, fc=fc, lw=lw, mutation_aspect=1)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, weight=weight,
            linespacing=1.15)
    return p


def _arrow(ax, p0, p1, color=qz.INK, lw=0.9, ls="-", rad=0.0, ms=8):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                        ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def _blank(ax):
    ax.set_axis_off()


# ===========================================================================
# Chapter 1
# ===========================================================================

UNITS = [  # Table 1.1: unit, sigma_ci (MPa), m_i, E_i (GPa), nu, GSI, top, base (m)
    (1, 19.0, 13, 6.0, 0.20, 55, 30.0, 40.0),
    (2, 61.0, 3, 3.5, 0.20, 45, 15.0, 30.0),
    (3, 21.0, 3, 1.5, 0.25, 35, 5.0, 15.0),
    (4, 1.0, None, 0.4, 0.30, None, 0.0, 5.0),
]


def metro_units():
    """Profiles of strength, modulus and GSI with depth, Table 1.1."""
    fig, axes = qz.figure(W_FULL, 0.50 * W_FULL, ncols=3, sharey=True)
    a1, a2, a3 = axes
    cols = {"unit": [], "top_m": [], "base_m": [], "sigma_ci_MPa": [], "sigma_cm_MPa": [],
            "E_i_GPa": [], "E_rm_GPa": [], "GSI": []}
    for unit, sci, mi, ei, nu, gsi, top, base in UNITS:
        z = [top, base]
        a1.plot([sci, sci], z, color=qz.MAL, lw=1.6, solid_capstyle="butt")
        a2.plot([ei, ei], z, color=qz.MAL, lw=1.6, solid_capstyle="butt")
        scm = erm = np.nan
        if gsi is not None:
            hb = rock.hb_constants(gsi, mi, 0.0)
            scm = rock.rock_mass_ucs(sci, hb)
            erm = rock.modulus_hd_generalised(ei, gsi, 0.0)
            a1.plot([scm, scm], z, color=qz.CAR, lw=1.6, ls="--", solid_capstyle="butt")
            a2.plot([erm, erm], z, color=qz.CAR, lw=1.6, ls="--", solid_capstyle="butt")
            a3.plot([gsi, gsi], z, color=qz.MAL, lw=1.6, solid_capstyle="butt")
        for ax in axes:
            ax.axhline(top, color=qz.GREY_LIGHT, lw=0.5, zorder=0)
        a3.text(78, 0.5 * (top + base) + 1.5, f"Unit {unit}", ha="right", va="center", fontsize=FS,
                color=qz.GREY)
        for k, v in zip(cols, (unit, top, base, sci, scm, ei, erm, gsi if gsi else np.nan)):
            cols[k].append(v)
    a1.text(25, 10.0, "Intact", fontsize=FS, color=qz.MAL, va="center")
    a1.text(3.3, 22.5, "Rock\nmass", fontsize=FS, color=qz.CAR, va="center")
    a1.set_xscale("log")
    a1.set_xlim(0.2, 200)
    a2.set_xscale("log")
    a2.set_xlim(0.1, 20)
    a3.set_xlim(20, 80)
    a1.set_ylim(40, 0)
    qz.label(a1, "Strength, $\\sigma_c$ (MPa)", "Depth, $z$ (m)")
    qz.label(a2, "Modulus, $E$ (GPa)")
    qz.label(a3, "GSI (-)")
    fig.tight_layout(w_pad=1.0)
    _save(fig, "ch01", "chal-metro-units", cols)


def shallow_setting():
    """Section through a shallow urban tunnel with its settlement trough and drawdown."""
    fig = plt.figure(figsize=(W_FULL, 0.66 * W_FULL))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.0], hspace=0.08)
    a1 = fig.add_subplot(gs[0])
    a2 = fig.add_subplot(gs[1], sharex=a1)
    y = np.linspace(-90, 90, 721)
    bore = settlement.Bore(offset=0.0, z0=20.0, diameter=6.3, vl_percent=1.0, k=0.5)
    s_t = settlement.transverse([bore], y)
    s_dd = 6.0 * np.exp(-y ** 2 / (2 * 45.0 ** 2))          # schematic drawdown consolidation
    a1.plot(y, s_t, color=qz.CAR, lw=1.3, label="Tunnel trough")
    a1.plot(y, s_dd, color=qz.MAL, lw=1.3, ls="--", label="Drawdown")
    a1.set_ylim(15, 0)
    a1.set_xlim(-90, 90)
    a1.legend(loc="lower right", handlelength=2.0)
    qz.label(a1, None, "Settlement, $S$ (mm)")
    a1.tick_params(labelbottom=False)

    # section
    bands = [(0, 5, "#F2F2F2", "4"), (5, 15, "#E3E3E3", "3"), (15, 30, qz.MAL_TINT, "2"),
             (30, 40, "#CFE0DA", "1")]
    for top, base, fc, lab in bands:
        a2.add_patch(Rectangle((-90, -base), 180, base - top, fc=fc, ec="none", zorder=0))
        a2.text(87, {"4": -4.2, "1": -33.0}.get(lab, -(top + base) / 2), lab, ha="right", va="center", fontsize=FS, color=qz.GREY)
    a2.axhline(0, color=qz.INK, lw=0.8)
    for xb, wb, hb in ((-70, 14, 12), (-40, 16, 18), (28, 18, 22), (60, 12, 9)):
        a2.add_patch(Rectangle((xb, 0), wb, hb, fc=qz.GREY_LIGHT, ec=qz.GREY, lw=0.5))
        a2.add_patch(Rectangle((xb, -3), wb, 3, fc="white", ec=qz.GREY, lw=0.5, ls=":"))
    # water table with drawdown towards the drained tunnel
    wt = -2.5 - 9.0 * np.exp(-y ** 2 / (2 * 30.0 ** 2))
    a2.plot(y, np.full_like(y, -2.5), color=qz.MAL, lw=0.7, ls=":")
    a2.plot(y, wt, color=qz.MAL, lw=1.1, ls="--")
    a2.add_patch(Circle((0, -20), 3.15, fc="white", ec=qz.CAR, lw=1.3, zorder=3))
    a2.set_ylim(-36, 24)
    a2.tick_params(right=False)
    a2.set_aspect("equal", adjustable="box")
    a2.set_yticks([0, -10, -20, -30])
    a2.set_yticklabels(["0", "10", "20", "30"])
    qz.label(a2, "Offset from tunnel axis, $y$ (m)", "Depth, $z$ (m)")
    fig.align_ylabels([a1, a2])
    _save(fig, "ch01", "chal-shallow-setting",
          {"y_m": y, "S_tunnel_mm": s_t, "S_drawdown_schematic_mm": s_dd, "water_table_m": wt,
           "i_m": [bore.i], "Smax_mm": [bore.smax_mm]})


def deep_setting():
    """Long section of a deep pumped-storage waterway (schematic, Table 1.2 scale)."""
    fig, ax = qz.figure(W_FULL, 0.42 * W_FULL)
    x = np.linspace(0, 28, 561)
    surf = (1250 + 180 * np.sin(x / 2.3) * np.exp(-x / 30) + 60 * np.sin(x * 1.7)
            - 900 / (1 + np.exp(-(x - 24.5) / 1.0)))
    surf = np.minimum(surf, 1100 + 250 * x)            # valley of the upper reservoir at x = 0
    ax.fill_between(x, surf, -100, color="#EEEEEE", lw=0)
    ax.plot(x, surf, color=qz.INK, lw=0.7)
    # waterway: intake, inclined pressure shaft to station, tailrace to lower portal
    xs = 2.8
    zs_surf = float(np.interp(xs, x, surf))
    zs = zs_surf - 850
    ax.plot([0.05, 1.0, xs], [1095, 1060, zs], color=qz.MAL, lw=1.4)
    ax.plot([xs, 27.0], [zs, 360], color=qz.MAL, lw=1.4)
    ax.add_patch(Rectangle((xs - 0.35, zs - 25), 0.7, 70, fc=qz.MAL, ec="none", zorder=3))
    ax.annotate("", xy=(xs, zs_surf), xytext=(xs, zs + 50),
                arrowprops=dict(arrowstyle="<->", color=qz.GREY, lw=0.6))
    ax.text(xs + 0.25, zs + 420, "850 m", fontsize=FS, color=qz.GREY, va="center")
    # fault zone met 12 km into a single-entry drive from the lower portal
    xf = 27.0 - 12.0
    zf = float(np.interp(xf, [xs, 27.0], [zs, 360]))
    slope = 0.70 / 1700.0                                # km of offset per m of rise
    xt0, xt1 = xf + 0.25 + 0.0, xf + 0.75
    zt0 = float(np.interp(xt0 + slope * 1300, x, surf))
    zt1 = float(np.interp(xt1 + slope * 1300, x, surf))
    ax.add_patch(Polygon([[xf - 0.45, -100], [xf + 0.05, -100],
                          [xf + 0.05 + slope * (zt1 + 100), zt1],
                          [xf - 0.45 + slope * (zt0 + 100), zt0]],
                         fc=qz.CAR_TINT, ec=qz.CAR, lw=0.6, zorder=1))
    ax.plot([xf + 0.9, 27.0], [zf + 6, 366], color=qz.CAR, lw=2.2, solid_capstyle="butt")
    ax.text(xf + 0.95, zf + 130, "Drive", fontsize=FS, color=qz.CAR)
    ax.text(xf + 0.9, 900, "Fault zone", fontsize=FS, color=qz.CAR, ha="left")
    ax.annotate("", xy=(xf + 0.9, zf - 120), xytext=(27.0, zf - 120),
                arrowprops=dict(arrowstyle="<->", color=qz.GREY, lw=0.6))
    ax.text(0.5 * (xf + 27.9), zf - 190, "12 km", fontsize=FS, color=qz.GREY, ha="center", va="top")
    ax.set_xlim(0, 28)
    ax.set_ylim(-100, 1700)
    qz.label(ax, "Distance along waterway, $x$ (km)", "Elevation (m)")
    fig.tight_layout()
    _save(fig, "ch01", "chal-deep-setting",
          {"x_km": x, "surface_m": surf, "station_x_km": [xs], "station_z_m": [zs],
           "fault_x_km": [xf]})


def sampling_fraction():
    """Volume fraction of the ground sampled by vertical cored boreholes, f = d^2 / (D s)."""
    fig, ax = qz.figure(W_HALF, 0.78 * W_HALF)
    s = np.logspace(1, np.log10(500), 200)
    cols = {"spacing_m": s}
    for D, ls, col in ((6.0, "--", qz.MAL), (10.0, "-", qz.MAL), (14.0, ":", qz.CAR)):
        f = 0.1 ** 2 / (D * s)
        ax.plot(s, f, color=col, ls=ls, lw=1.3, label=f"$D$ = {D:g} m")
        cols[f"f_D{D:g}"] = f
    ax.plot([100], [1e-5], "o", mfc="white", mec=qz.INK, ms=5, mew=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 500)
    ax.set_ylim(1e-6, 2e-4)
    ax.legend(loc="upper right")
    qz.label(ax, "Borehole spacing, $s$ (m)", "Fraction sampled, $f$ (-)")
    fig.tight_layout()
    _save(fig, "ch01", "chal-sampling-fraction", cols)


def lifecycle_loop():
    """Need, geo-risk, model, design, construction and operation as one loop."""
    fig, ax = plt.subplots(figsize=(W_FULL, 0.50 * W_FULL))
    _blank(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.45, 4.6)
    bw, bh = 1.85, 0.95
    need = (1.1, 3.9)
    risk = (1.1, 1.1)
    model = (4.2, 2.5)
    design = (7.4, 3.9)
    constr = (8.9, 1.1)
    oper = (5.9, 1.1)
    _box(ax, *need, bw, bh, "Infrastructure\nneed")
    _box(ax, *risk, bw, bh, "Geo-risk\nground, water")
    _box(ax, *model, 2.2, 1.2, "Geo-integrated\nmodel", ec=qz.MAL, fc=qz.MAL_TINT, lw=1.2)
    _box(ax, *design, 2.3, 1.15, "Design\ncarbon, safety,\ndeformation, build")
    _box(ax, *constr, 1.75, bh, "Construction")
    _box(ax, *oper, 1.75, bh, "Operation")
    _arrow(ax, (need[0] + bw / 2, need[1] - 0.2), (model[0] - 1.1, model[1] + 0.35))
    _arrow(ax, (risk[0] + bw / 2, risk[1] + 0.2), (model[0] - 1.1, model[1] - 0.35))
    _arrow(ax, (model[0] + 1.1, model[1] + 0.4), (design[0] - 1.15, design[1] - 0.2), color=qz.MAL)
    _arrow(ax, (design[0] + 0.6, design[1] - 0.575), (constr[0], constr[1] + bh / 2), color=qz.MAL)
    _arrow(ax, (constr[0] - 0.875, constr[1]), (oper[0] + 0.875, oper[1]), color=qz.MAL)
    _arrow(ax, (constr[0] - 0.55, constr[1] + bh / 2), (model[0] + 0.85, model[1] - 0.6),
           color=qz.CAR, ls="--", rad=0.25)
    _arrow(ax, (oper[0] - 0.875, oper[1] + 0.1), (model[0] + 0.1, model[1] - 0.6),
           color=qz.CAR, ls="--", rad=-0.3)
    ax.text(6.55, 2.32, "Observations", fontsize=FS, color=qz.CAR, ha="center")
    ax.text(4.05, 1.0, "Condition\ndata", fontsize=FS, color=qz.CAR, ha="center", va="center")
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    _save(fig, "ch01", "chal-lifecycle-loop",
          {"node": ["need", "geo-risk", "model", "design", "construction", "operation"]})


def leverage():
    """Influence on outcome against knowledge of the ground and cost of change (schematic)."""
    fig, ax = qz.figure(W_HALF, 0.80 * W_HALF)
    t = np.linspace(0, 5, 300)
    infl = 1 / (1 + np.exp(2.2 * (t - 1.4)))
    know = 0.08 + 0.85 / (1 + np.exp(-1.6 * (t - 2.9)))
    cost = 0.05 + 0.9 / (1 + np.exp(-1.5 * (t - 3.2)))
    ax.plot(t, infl, color=qz.MAL, lw=1.4, label="Influence on outcome")
    ax.plot(t, know, color=qz.GREY, lw=1.2, ls="--", label="Knowledge of ground")
    ax.plot(t, cost, color=qz.CAR, lw=1.3, ls=":", label="Cost of change")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_xticklabels(["Feas.", "Ref.", "Detail", "Build", "Operate"])
    ax.tick_params(axis="x", which="both", top=False)
    ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.62, 1.0))
    qz.label(ax, "Project stage", "Relative level (-)")
    fig.tight_layout()
    _save(fig, "ch01", "chal-leverage", {"stage": t, "influence": infl, "knowledge": know,
                                          "cost_of_change": cost})


# ===========================================================================
# Chapter 2
# ===========================================================================

def dt_chainage():
    """One model indexed by chainage: unit, ground class, support class, monitoring, carbon."""
    fig = plt.figure(figsize=(W_FULL, 0.56 * W_FULL))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.08)
    a1 = fig.add_subplot(gs[0])
    a2 = fig.add_subplot(gs[1], sharex=a1)
    L = 1200
    unit_edges = [0, 260, 610, 680, 720, 1010, 1200]
    unit_id = [2, 3, 2, 9, 2, 1]                      # 9 = fault zone
    shade = {1: "#CFE0DA", 2: qz.MAL_TINT, 3: "#F2F2F2", 9: qz.CAR_TINT}
    gc_edges = [0, 240, 300, 590, 640, 760, 1000, 1200]
    gc = [2, 3, 3, 3, 4, 2, 1]
    gc = [2, 3, 2, 3, 4, 2, 1]
    sc_shade = {1: "#FFFFFF", 2: qz.MAL_TINT, 3: "#8FBCAC", 4: qz.MAL}
    rows = {"Unit": 3.5, "Ground class": 2.5, "Support": 1.5, "Monitoring": 0.5}
    for i in range(len(unit_id)):
        x0, x1 = unit_edges[i], unit_edges[i + 1]
        u = unit_id[i]
        a1.add_patch(Rectangle((x0, 3.1), x1 - x0, 0.8, fc=shade[u],
                               ec=qz.CAR if u == 9 else qz.GREY, lw=0.5))
    for i in range(len(gc)):
        x0, x1 = gc_edges[i], gc_edges[i + 1]
        a1.add_patch(Rectangle((x0, 2.1), x1 - x0, 0.8, fc=sc_shade[gc[i]], ec=qz.GREY, lw=0.5))
        a1.text(0.5 * (x0 + x1), 2.5, ["I", "II", "III", "IV"][gc[i] - 1], ha="center", va="center",
                fontsize=FS, color="white" if gc[i] == 4 else qz.INK)
    # support class follows ground class, extended 20 m either side of a change (transition)
    sc_edges = [0, 220, 320, 570, 620, 780, 980, 1200]
    for i in range(len(gc)):
        x0, x1 = sc_edges[i], sc_edges[i + 1]
        a1.add_patch(Rectangle((x0, 1.1), x1 - x0, 0.8, fc=sc_shade[gc[i]], ec=qz.GREY, lw=0.5))
        a1.text(0.5 * (x0 + x1), 1.5, f"S{gc[i]}", ha="center", va="center", fontsize=FS,
                color="white" if gc[i] == 4 else qz.INK)
    mon = np.concatenate([np.arange(25, 600, 100), np.arange(600, 800, 25), np.arange(825, 1200, 100)])
    a1.plot(mon, np.full_like(mon, 0.5, dtype=float), "v", ms=4, mfc=qz.CAR, mec=qz.CAR)
    a1.set_ylim(0, 4.0)
    a1.set_yticks(list(rows.values()))
    a1.set_yticklabels(list(rows.keys()), fontsize=8)
    a1.tick_params(axis="y", length=0)
    a1.tick_params(labelbottom=False)
    for side in ("top", "right", "left", "bottom"):
        a1.spines[side].set_visible(False)
    a1.tick_params(top=False, right=False, bottom=False)
    # lining carbon per metre, relative to the lightest class
    xc = np.linspace(0, L, 1201)
    rel = {1: 1.0, 2: 1.12, 3: 1.28, 4: 1.55}
    cls = np.array([gc[np.searchsorted(sc_edges, v, side="right") - 1] for v in np.minimum(xc, L - 1e-6)])
    carb = np.array([rel[c] for c in cls])
    a2.step(xc, carb, where="post", color=qz.MAL, lw=1.2)
    a2.fill_between(xc, 0, carb, step="post", color=qz.MAL_TINT, lw=0)
    a2.set_ylim(0.8, 1.7)
    a2.set_xlim(0, L)
    qz.label(a2, "Chainage, $x$ (m)", "Lining carbon (-)")
    pass
    fig.subplots_adjust(left=0.22, right=0.98, top=0.98, bottom=0.14)
    _save(fig, "ch02", "dt-chainage", {"chainage_m": xc, "support_class": cls,
                                        "relative_lining_carbon": carb,
                                        "monitoring_sections_m": mon})


def _gp(xo, yo, xs, ell, sig, noise=1e-6):
    k = lambda a, b: sig ** 2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell ** 2)
    K = k(xo, xo) + noise * np.eye(len(xo))
    Ks = k(xs, xo)
    mu0 = yo.mean()
    alpha = np.linalg.solve(K, yo - mu0)
    mean = mu0 + Ks @ alpha
    var = sig ** 2 - np.einsum("ij,ji->i", Ks, np.linalg.solve(K, Ks.T))
    return mean, np.sqrt(np.clip(var, 0, None))


def dt_interpolation():
    """Rockhead interpolated between boreholes 200 m apart, with its uncertainty."""
    fig, ax = qz.figure(W_FULL, 0.42 * W_FULL)
    xs = np.linspace(0, 800, 801)
    true = (14 + 3 * np.sin(xs / 95) + 2.0 * np.sin(xs / 37 + 1)
            + 9 * np.exp(-((xs - 300) / 28) ** 2) + 4 * np.exp(-((xs - 560) / 40) ** 2))
    xb = np.array([0, 200, 400, 600, 800], float)
    yb = np.interp(xb, xs, true)
    mean, sd = _gp(xb, yb, xs, ell=90.0, sig=5.0)
    ax.fill_between(xs, mean - 1.645 * sd, mean + 1.645 * sd, color=qz.MAL_TINT, lw=0,
                    label="90% band")
    ax.plot(xs, mean, color=qz.MAL, lw=1.3, label="Interpolated")
    ax.plot(xs, np.interp(xs, xb, yb), color=qz.GREY, lw=0.9, ls="--", label="Linear")
    ax.plot(xs, true, color=qz.CAR, lw=1.0, ls=":", label="Actual")
    for xi, yi in zip(xb, yb):
        ax.plot([xi, xi], [0, 30], color=qz.INK, lw=0.6)
        ax.plot([xi], [yi], "o", mfc="white", mec=qz.INK, ms=4, mew=0.9, zorder=4)
    ax.set_xlim(-25, 825)
    ax.set_ylim(32, 0)
    ax.legend(loc="lower left", ncol=4, handlelength=1.8, columnspacing=1.0)
    qz.label(ax, "Chainage, $x$ (m)", "Depth to rock, $z_r$ (m)")
    fig.tight_layout()
    _save(fig, "ch02", "dt-interpolation", {"x_m": xs, "actual_m": true, "gp_mean_m": mean,
                                             "gp_sd_m": sd, "bh_x_m": xb, "bh_z_m": yb})


def dt_update_loop():
    """Forward prediction and backward updating around the asset and ground model."""
    fig, ax = plt.subplots(figsize=(W_FULL, 0.47 * W_FULL))
    _blank(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    m = (1.5, 2.3)
    _box(ax, *m, 2.3, 1.4, "Asset and\nground model", fc=qz.MAL_TINT, lw=1.2)
    a = (5.0, 3.75)
    _box(ax, *a, 2.2, 0.85, "Analysis models")
    p = (8.5, 3.75)
    _box(ax, *p, 1.8, 0.85, "Prediction")
    o = (8.5, 0.85)
    _box(ax, *o, 1.8, 0.85, "Observation")
    at = (5.0, 0.85)
    _box(ax, *at, 2.2, 0.85, "Attribution\nto the works", ec=qz.CAR)
    _arrow(ax, (m[0] + 0.6, m[1] + 0.7), (a[0] - 1.1, a[1]), color=qz.MAL, rad=-0.25)
    _arrow(ax, (a[0] + 1.1, a[1]), (p[0] - 0.9, p[1]), color=qz.MAL)
    _arrow(ax, (p[0], p[1] - 0.425), (o[0], o[1] + 0.425), color=qz.GREY, ls=":")
    ax.text(p[0] + 0.12, 2.3, "Compare", fontsize=FS, color=qz.GREY, va="center")
    _arrow(ax, (o[0] - 0.9, o[1]), (at[0] + 1.1, at[1]), color=qz.CAR)
    _arrow(ax, (at[0] - 1.1, at[1]), (m[0] + 0.6, m[1] - 0.7), color=qz.CAR, rad=-0.25)
    ax.text(3.15, 3.95, "Forward", fontsize=FS, color=qz.MAL, ha="center")
    ax.text(3.15, 0.45, "Backward", fontsize=FS, color=qz.CAR, ha="center")
    ax.text(5.0, 2.95, "settlement, support,\nlining, sequence", fontsize=FS, color=qz.GREY,
            ha="center", va="top")
    ax.text(5.0, 1.62, "pre-construction\nfield removed", fontsize=FS, color=qz.GREY,
            ha="center", va="bottom")
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    _save(fig, "ch02", "dt-update-loop", {"node": ["model", "analysis", "prediction",
                                                   "observation", "attribution"]})


def dt_maturity():
    """Four levels of twin maturity as a staircase: each level needs the one below."""
    fig, ax = plt.subplots(figsize=(W_FULL, 0.50 * W_FULL))
    _blank(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.0)
    levels = [("Descriptive", "What is\nthe asset?", None),
              ("Reflective", "What is\nhappening now?", "+ monitoring data"),
              ("Predictive", "What will\nhappen next?", "+ forecasting models"),
              ("Prescriptive", "What should\nbe done?", "+ decision criteria")]
    fills = ["#FFFFFF", qz.MAL_TINT, "#8FBCAC", qz.MAL]
    w, h = 2.35, 1.3
    for i, ((name, q, add), fc) in enumerate(zip(levels, fills)):
        x0 = 0.15 + i * 2.45
        y0 = 0.15 + i * 0.95
        ax.add_patch(Rectangle((x0, 0.15), w, y0 - 0.15 + h, fc=fc, ec=qz.MAL, lw=0.9))
        tc = "white" if i == 3 else qz.INK
        ax.text(x0 + w / 2, y0 + h - 0.27, name, ha="center", va="center", fontsize=9, color=tc)
        ax.text(x0 + w / 2, y0 + h - 0.78, q, ha="center", va="center", fontsize=FS, color=tc)
        if add:
            ax.text(x0 + w / 2, y0 + h + 0.2, add, ha="center", va="bottom", fontsize=FS,
                    color=qz.CAR)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    _save(fig, "ch02", "dt-maturity", {"level": [lv[0] for lv in levels]})


# ===========================================================================
# Chapter 3
# ===========================================================================

# Element carbon per metre of a bored rail tunnel (internal diameter 6.3 m,
# 300 mm C50 segmental lining), kgCO2e/m, from the author's lecture material
# on the published A1-A5 baseline. Module in brackets.
CARBON = [
    ("Lining concrete", "A1-A3", 3568.0),
    ("Lining steel", "A1-A3", 1336.0),
    ("Invert backfill", "A1-A3", 1127.0),
    ("Annulus grout", "A1-A3", 841.8),
    ("Track and deck", "A1-A3", 78.4),
    ("Fit-out", "A1-A3", 3.4),
    ("Material delivery", "A4", 173.8),
    ("TBM electricity", "A5", 2510.0),
    ("Spoil removal", "A5", 962.2),
]
MOD_STYLE = {"A1-A3": dict(fc=qz.MAL, ec=qz.MAL, hatch=None),
             "A4": dict(fc=qz.GREY_LIGHT, ec=qz.GREY, hatch=None),
             "A5": dict(fc=qz.CAR_TINT, ec=qz.CAR, hatch="////")}
MOD_LABEL = {"A1-A3": "A1 to A3", "A4": "A4", "A5": "A5"}


def sus_carbon_elements():
    fig, ax = qz.figure(W_FULL, 0.50 * W_FULL)
    items = sorted(CARBON, key=lambda r: r[2])
    total = sum(r[2] for r in CARBON)
    yy = np.arange(len(items))
    seen = set()
    for yi, (name, mod, v) in zip(yy, items):
        st = MOD_STYLE[mod]
        lab = MOD_LABEL[mod] if mod not in seen else None
        seen.add(mod)
        ax.barh(yi, v / 1000, height=0.62, color=st["fc"], edgecolor=st["ec"], hatch=st["hatch"],
                lw=0.6, label=lab)
        ax.text(v / 1000 + 0.06, yi, f"{100 * v / total:.0f}%" if v / total >= 0.005 else "$<$1%",
                va="center", fontsize=FS, color=qz.GREY)
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in items], fontsize=8)
    ax.tick_params(axis="y", length=0, right=False)
    ax.set_xlim(0, 4.2)
    h, lb = ax.get_legend_handles_labels()
    order = [lb.index(k) for k in ("A1 to A3", "A4", "A5")]
    ax.legend([h[i] for i in order], [lb[i] for i in order], loc="lower right")
    qz.label(ax, "Carbon per metre, $GWP$ (tCO$_2$e/m)")
    fig.tight_layout()
    _save(fig, "ch03", "sus-carbon-elements",
          {"element": [r[0] for r in CARBON], "module": [r[1] for r in CARBON],
           "kgCO2e_per_m": [r[2] for r in CARBON]})


def sus_carbon_boundary():
    fig, ax = qz.figure(W_HALF, 0.78 * W_HALF)
    a13 = sum(v for _, m, v in CARBON if m == "A1-A3")
    a4 = sum(v for _, m, v in CARBON if m == "A4")
    a5 = sum(v for _, m, v in CARBON if m == "A5")
    bounds = [("A1 to A3", [a13]), ("A1 to A4", [a13, a4]), ("A1 to A5", [a13, a4, a5])]
    mods = ["A1-A3", "A4", "A5"]
    for xi, (name, parts) in enumerate(bounds):
        b = 0.0
        for mod, v in zip(mods, parts):
            st = MOD_STYLE[mod]
            ax.bar(xi, v / 1000, bottom=b, width=0.58, color=st["fc"], edgecolor=st["ec"],
                   hatch=st["hatch"], lw=0.6, label=MOD_LABEL[mod] if xi == 2 else None)
            b += v / 1000
        ax.text(xi, b + 0.2, f"{b:.1f}", ha="center", va="bottom", fontsize=FS)
    ax.set_xticks(range(3))
    ax.set_xticklabels([b[0] for b in bounds])
    ax.tick_params(axis="x", length=0, top=False)
    ax.set_ylim(0, 13.5)
    ax.set_xlim(-0.55, 2.55)
    ax.legend(loc="upper left")
    qz.label(ax, "Declared boundary", "Carbon, $GWP$ (tCO$_2$e/m)")
    fig.tight_layout()
    _save(fig, "ch03", "sus-carbon-boundary",
          {"boundary": ["A1-A3", "A1-A4", "A1-A5"],
           "kgCO2e_per_m": [a13, a13 + a4, a13 + a4 + a5]})


ECF_C50 = 598.0                 # kgCO2e/m3, C50 segment concrete (lecture material)
REBAR_KG_M3 = 135.0             # kg of reinforcement per m3 of concrete
ECF_STEEL = 12409.8 / 7850.0    # kgCO2e/kg (12409.8 kgCO2e per m3 of steel)


def sus_lining_carbon():
    fig, ax = qz.figure(W_HALF, 0.80 * W_HALF)
    di = np.linspace(5, 16, 200)
    per_m3 = ECF_C50 + REBAR_KG_M3 * ECF_STEEL
    cols = {"Di_m": di}
    for r, ls, col in ((18, "-", qz.CAR), (23, "--", qz.MAL), (25, ":", qz.MAL)):
        t = di / r
        area = math.pi / 4 * ((di + 2 * t) ** 2 - di ** 2)
        g = area * per_m3 / 1000
        ax.plot(di, g, color=col, ls=ls, lw=1.3, label=f"$D_i/t$ = {r}")
        cols[f"GWP_t_per_m_Di_t_{r}"] = g
    d0, t0 = 6.3, 0.3
    g0 = math.pi / 4 * ((d0 + 2 * t0) ** 2 - d0 ** 2) * per_m3 / 1000
    ax.plot([d0], [g0], "o", mfc="white", mec=qz.INK, ms=5, mew=1.0)
    ax.set_xlim(5, 16)
    ax.set_ylim(0, 40)
    ax.legend(loc="upper left")
    qz.label(ax, "Internal diameter, $D_i$ (m)", "Carbon, $GWP$ (tCO$_2$e/m)")
    fig.tight_layout()
    cols["case_Di_m"] = [d0]
    cols["case_GWP_t_per_m"] = [g0]
    cols["factor_kgCO2e_per_m3"] = [per_m3]
    _save(fig, "ch03", "sus-lining-carbon", cols)


def sus_support_timing():
    """Required support pressure against convergence before support: falling then rising."""
    fig, ax = qz.figure(W_HALF, 0.80 * W_HALF)
    u = np.linspace(0, 1, 400)
    p = 1.0 * np.exp(-6 * u) + 0.10 + 0.9 * np.clip(u - 0.45, 0, None) ** 2 / 0.3
    p = 0.9 * np.exp(-7 * u) + 0.12 + 1.2 * np.clip(u - 0.42, 0, None) ** 1.6
    i_min = int(np.argmin(p))
    ax.plot(u, p, color=qz.INK, lw=1.4)
    ax.fill_between(u[:i_min + 1], 0, p[:i_min + 1], color=qz.MAL_TINT, lw=0)
    ax.fill_between(u[i_min:], 0, p[i_min:], color=qz.CAR_TINT, lw=0)
    ax.axvline(u[i_min], color=qz.GREY, lw=0.6, ls=":")
    ax.text(0.5 * u[i_min], 0.06, "Delay lowers\ndemand", ha="center", va="bottom", fontsize=FS,
            color=qz.MAL)
    ax.text(0.5 * (u[i_min] + 1), 0.06, "Delay raises\ndemand", ha="center", va="bottom",
            fontsize=FS, color=qz.CAR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    qz.label(ax, "Convergence before support, $u_r$", "Required pressure, $p_i$")
    fig.tight_layout()
    _save(fig, "ch03", "sus-support-timing", {"u_rel": u, "p_rel": p})


def _q_curve(t, t0, residual, delay, dur, final):
    q = np.ones_like(t)
    q[t >= t0] = residual
    r0 = t0 + delay
    m = (t >= r0) & (t < r0 + dur)
    q[m] = residual + (final - residual) * (t[m] - r0) / dur
    q[t >= r0 + dur] = final
    return q


def sus_functionality():
    """Functionality after an event for two restoration options, and the index against window."""
    fig, (a1, a2) = qz.figure(W_FULL, 0.46 * W_FULL, ncols=2)
    t = np.linspace(-30, 400, 4301)
    opts = [("Partial, fast", 0.35, 10, 30, 0.85, qz.CAR, "--"),
            ("Full, slow", 0.35, 45, 120, 1.00, qz.MAL, "-")]
    cols = {"t_d": t}
    windows = np.linspace(20, 720, 701)
    cols_w = {"window_d": windows}
    for name, res, dl, du, fin, col, ls in opts:
        q = _q_curve(t, 0.0, res, dl, du, fin)
        a1.plot(t, q, color=col, ls=ls, lw=1.3, label=name)
        cols[name] = q
        tt = np.linspace(0, windows.max(), 72001)
        qq = _q_curve(tt, 0.0, res, dl, du, fin)
        cum = np.concatenate([[0], np.cumsum(0.5 * (qq[1:] + qq[:-1]) * np.diff(tt))])
        R = np.interp(windows, tt, cum) / windows
        a2.plot(windows, R, color=col, ls=ls, lw=1.3, label=name)
        cols_w[name] = R
    qs = _q_curve(t, 0.0, 0.35, 45, 120, 1.0)
    a1.fill_between(t, qs, 1.0, where=t >= 0, color=qz.MAL_TINT, lw=0)
    a1.axhline(1.0, color=qz.GREY, lw=0.6, ls=":")
    a1.set_xlim(-30, 400)
    a1.set_ylim(0, 1.15)
    a1.legend(loc="lower right")
    qz.label(a1, "Time after event, $t$ (d)", "Functionality, $Q$ (-)")
    # crossover
    d = cols_w["Full, slow"] - cols_w["Partial, fast"]
    k = np.where(np.diff(np.sign(d)) != 0)[0]
    tx = float(windows[k[0]]) if len(k) else float("nan")
    a2.axvline(tx, color=qz.GREY, lw=0.6, ls=":")
    a2.set_xlim(0, 720)
    a2.set_ylim(0.3, 1.0)
    a2.legend(loc="lower right")
    qz.label(a2, "Recovery window, $T$ (d)", "Resilience index, $R$ (-)")
    fig.tight_layout()
    cols.update(cols_w)
    cols["crossover_window_d"] = [tx]
    _save(fig, "ch03", "sus-functionality", cols)
    return tx


# ===========================================================================
# Chapter 4
# ===========================================================================

def geo_weathering():
    """Weathering profile with an irregular rockhead, corestones and a mixed face (schematic)."""
    fig, ax = qz.figure(W_FULL, 0.52 * W_FULL)
    L, bottom = 120.0, 36.0
    x = np.linspace(0, L, 961)
    xf = 82.0                                               # fault trace at tunnel level
    trough = np.exp(-((x - xf) / 7.0) ** 2) + 0.45 * np.exp(-((x - xf) / 18.0) ** 2)
    base = {6: 3.0, 5: 7.0, 4: 11.0, 3: 14.5, 2: 17.0}      # mean depth to the base of each grade
    deepen = {6: 3.0, 5: 7.0, 4: 13.0, 3: 15.0, 2: 15.0}    # deepening at the fault (m)
    wav = 1.0 * np.sin(x / 9) + 0.7 * np.sin(x / 5 + 2)
    depth_base = {g: d + wav * (0.25 + 0.12 * (6 - g)) + deepen[g] * trough
                  for g, d in base.items()}
    fills = {6: "#FFFFFF", 5: "#F2F2F2", 4: "#E3E3E3", 3: qz.MAL_TINT, 2: "#CFE0DA", 1: "#8FBCAC"}
    ax.fill_between(x, 0, bottom, color=fills[1], lw=0)
    for g in (2, 3, 4, 5, 6):
        ax.fill_between(x, 0, depth_base[g], color=fills[g], lw=0)
    for g in (2, 3, 5, 6):
        ax.plot(x, depth_base[g], color=qz.GREY, lw=0.4)
    ax.plot(x, depth_base[4], color=qz.INK, lw=1.1)        # rockhead: base of grade IV
    ax.axhline(0, color=qz.INK, lw=0.8)
    ax.plot([xf + 12, xf - 9], [-0.5, bottom], color=qz.CAR, lw=0.9, ls="--")
    # corestones in grades III to V, away from the fault
    rng = np.random.default_rng(11)
    for _ in range(60):
        xc = rng.uniform(3, L - 3)
        lo, hi = np.interp(xc, x, depth_base[5]), np.interp(xc, x, depth_base[3])
        if hi - lo < 3.0 or abs(xc - xf) < 9:
            continue
        zc = rng.uniform(lo + 1.0, hi - 0.8)
        ax.add_patch(plt.matplotlib.patches.Ellipse(
            (xc, zc), rng.uniform(1.4, 2.8), rng.uniform(0.8, 1.4), angle=rng.uniform(-20, 20),
            fc="#CFE0DA", ec=qz.GREY, lw=0.4))
    # boreholes 40 m apart, all of which miss the trough
    for xb in (20, 60, 100):
        ax.plot([xb, xb], [-0.8, 30], color=qz.INK, lw=1.4, solid_capstyle="butt")
    # tunnel, 6 m diameter, axis at 21 m
    zt, r = 21.0, 3.0
    ax.plot([0, L], [zt - r, zt - r], color=qz.CAR, lw=1.0)
    ax.plot([0, L], [zt + r, zt + r], color=qz.CAR, lw=1.0)
    rh = depth_base[4]
    xm = x[rh > zt - r]
    if xm.size:
        ax.add_patch(Rectangle((xm.min(), zt - r), xm.max() - xm.min(), 2 * r, fc="none",
                               ec=qz.CAR, hatch="////", lw=0.0))
    labels = {6: "VI", 5: "V", 4: "IV", 3: "III", 2: "II"}
    for g in (6, 5, 4, 3, 2):
        top = depth_base[g + 1][-1] if g < 6 else 0.0
        ax.text(L + 1.5, 0.5 * (top + depth_base[g][-1]), labels[g], fontsize=FS, va="center",
                ha="left", clip_on=False)
    ax.text(L + 1.5, 0.5 * (depth_base[2][-1] + bottom), "I", fontsize=FS, va="center", ha="left",
            clip_on=False)
    ax.set_xlim(0, L)
    ax.set_ylim(bottom, -1.5)
    ax.tick_params(right=False)
    qz.label(ax, "Distance, $x$ (m)", "Depth, $z$ (m)")
    fig.tight_layout()
    _save(fig, "ch04", "geo-weathering-profile",
          {"x_m": x, **{f"base_grade_{g}_m": depth_base[g] for g in base},
           "boreholes_x_m": [20, 60, 100], "tunnel_axis_m": [zt], "tunnel_radius_m": [r]})


def geo_joint_shear():
    """Patton bilinear and Barton JRC-JCS shear strength; the roughness term against stress."""
    fig, (a1, a2) = qz.figure(W_FULL, 0.46 * W_FULL, ncols=2)
    phib = 30.0
    jcs = 60.0
    sn = np.linspace(0.0, 4.0, 401)
    cols = {"sigma_n_MPa": sn}
    a1.plot(sn, sn * math.tan(math.radians(phib)), color=qz.GREY, lw=0.9, ls=":",
            label="$\\phi_b$ only")
    for jrc, ls, col in ((5, "--", qz.MAL), (10, "-", qz.MAL), (15, "-.", qz.CAR)):
        with np.errstate(divide="ignore"):
            ang = np.minimum(phib + jrc * np.log10(jcs / np.maximum(sn, 1e-9)), 70.0)
        tau = sn * np.tan(np.radians(ang))
        a1.plot(sn, tau, color=col, ls=ls, lw=1.3, label=f"JRC {jrc}")
        cols[f"tau_JRC{jrc}_MPa"] = tau
    # Patton bilinear, i = 15 degrees, asperities sheared at 1.5 MPa
    i_deg, snt = 15.0, 1.5
    taut = snt * math.tan(math.radians(phib + i_deg))
    tp = np.where(sn <= snt, sn * math.tan(math.radians(phib + i_deg)),
                  taut + (sn - snt) * math.tan(math.radians(phib)))
    a1.plot(sn, tp, color=qz.INK, lw=0.9, ls=(0, (1, 1.5)), label="Bilinear")
    cols["tau_bilinear_MPa"] = tp
    a1.set_xlim(0, 4)
    a1.set_ylim(0, 7)
    a1.legend(loc="upper left", handlelength=1.8)
    qz.label(a1, "Normal stress, $\\sigma_n'$ (MPa)", "Shear strength, $\\tau$ (MPa)")
    s2 = np.logspace(-2, 2, 400)
    cols2 = {"sigma_n_log_MPa": s2}
    for j, ls, col, name in ((60.0, "-", qz.MAL, "Fresh"), (15.0, "--", qz.CAR, "Weathered")):
        add = np.clip(10 * np.log10(j / s2), 0, None)
        a2.plot(s2, add, color=col, ls=ls, lw=1.3, label=f"JCS = {j:g} MPa")
        cols2[f"roughness_term_JCS{j:g}_deg"] = add
    a2.set_xscale("log")
    a2.set_xlim(0.01, 100)
    a2.set_ylim(0, 40)
    a2.legend(loc="upper right")
    qz.label(a2, "Normal stress, $\\sigma_n'$ (MPa)", "Roughness term, $i$ ($^\\circ$)")
    fig.tight_layout()
    cols.update(cols2)
    _save(fig, "ch04", "geo-joint-shear", cols)


def geo_fault_zone():
    """Fault core and damage zone crossed by a tunnel, with property profiles (schematic)."""
    fig = plt.figure(figsize=(W_FULL, 0.62 * W_FULL))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.08)
    a1 = fig.add_subplot(gs[0])
    a2 = fig.add_subplot(gs[1], sharex=a1)
    x = np.linspace(-60, 60, 1201)
    core, dmg = 3.0, 25.0
    # plan: host rock, damage zone with fracture traces, core
    a1.add_patch(Rectangle((-60, 0), 120, 20, fc="white", ec="none"))
    a1.add_patch(Rectangle((-dmg, 0), 2 * dmg, 20, fc=qz.MAL_TINT, ec="none"))
    a1.add_patch(Rectangle((-core, 0), 2 * core, 20, fc=qz.CAR_TINT, ec=qz.CAR, lw=0.6,
                           hatch="xxxx"))
    rng = np.random.default_rng(3)
    for _ in range(140):
        xc = rng.normal(0, dmg / 2.2)
        if abs(xc) < core or abs(xc) > 58:
            continue
        yc = rng.uniform(1, 19)
        L = rng.uniform(1.5, 4.0)
        th = math.radians(rng.choice([60, 120]) + rng.normal(0, 8))
        a1.plot([xc - L / 2 * math.cos(th), xc + L / 2 * math.cos(th)],
                [yc - L / 2 * math.sin(th), yc + L / 2 * math.sin(th)], color=qz.GREY, lw=0.5)
    a1.add_patch(Rectangle((-60, 8), 120, 4, fc="none", ec=qz.INK, lw=0.9))
    a1.set_ylim(0, 20)
    a1.set_yticks([])
    a1.tick_params(labelbottom=False, left=False, right=False)
    a1.text(0, 21, "Core", ha="center", va="bottom", fontsize=FS, color=qz.CAR)
    a1.text(-dmg, 21, "Damage zone", ha="left", va="bottom", fontsize=FS, color=qz.MAL)
    a1.text(dmg, 21, "Damage zone", ha="right", va="bottom", fontsize=FS, color=qz.MAL)
    a1.text(-45, 21, "Host rock", ha="center", va="bottom", fontsize=FS, color=qz.GREY)
    a1.text(45, 21, "Host rock", ha="center", va="bottom", fontsize=FS, color=qz.GREY)
    # profiles, normalised to the host rock
    dens = 1 + 7 * np.exp(-np.clip(np.abs(x) - core, 0, None) / 7.0)
    dens[np.abs(x) < core] = np.nan
    perm = 1 + 30 * np.exp(-np.clip(np.abs(x) - core, 0, None) / 6.0)
    perm[np.abs(x) < core] = 5.0
    strength = 1 - 0.6 * np.exp(-np.clip(np.abs(x) - core, 0, None) / 8.0)
    strength[np.abs(x) < core] = 0.12
    a2.plot(x, dens, color=qz.MAL, lw=1.3, label="Fracture density")
    a2.plot(x, perm, color=qz.CAR, lw=1.3, ls="--", label="Permeability")
    a2.plot(x, strength, color=qz.INK, lw=1.1, ls=":", label="Strength")
    a2.axvspan(-core, core, color=qz.CAR_TINT, lw=0)
    a2.set_yscale("log")
    a2.set_ylim(0.08, 80)
    a2.set_xlim(-60, 60)
    a2.legend(loc="upper left", handlelength=1.8)
    qz.label(a2, "Distance from fault core, $x$ (m)", "Relative to host (-)")
    fig.align_ylabels([a1, a2])
    fig.subplots_adjust(left=0.13, right=0.98, top=0.93, bottom=0.12)
    _save(fig, "ch04", "geo-fault-zone", {"x_m": x, "fracture_density_rel": dens,
                                           "permeability_rel": perm, "strength_rel": strength})


def geo_uncertainty():
    """How positional, interpretive and parametric uncertainty fall with observations (schematic)."""
    fig, ax = qz.figure(W_HALF, 0.80 * W_HALF)
    n = np.arange(0, 13)
    positional = 0.5 ** n
    interpretive = np.where(n < 3, 1.0, 0.08)
    sig = 1.0
    parametric = np.sqrt(sig ** 2 + sig ** 2 / np.maximum(n, 1)) / math.sqrt(2)
    parametric[0] = 1.0
    ax.plot(n, positional, "-o", color=qz.MAL, ms=3.5, lw=1.2, label="Positional")
    ax.step(n, interpretive, where="post", color=qz.CAR, lw=1.3, ls="--", label="Interpretive")
    ax.plot(n, parametric, "-s", color=qz.INK, ms=3.0, lw=1.1, label="Parametric")
    ax.axhline(1 / math.sqrt(2), color=qz.GREY, lw=0.6, ls=":")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42))
    qz.label(ax, "Observations, $n$ (-)", "Relative uncertainty (-)")
    fig.tight_layout()
    _save(fig, "ch04", "geo-uncertainty", {"n": n, "positional": positional,
                                            "interpretive": interpretive, "parametric": parametric})


ALL = [metro_units, shallow_setting, deep_setting, sampling_fraction, lifecycle_loop, leverage,
       dt_chainage, dt_interpolation, dt_update_loop, dt_maturity,
       sus_carbon_elements, sus_carbon_boundary, sus_lining_carbon, sus_support_timing,
       sus_functionality,
       geo_weathering, geo_joint_shear, geo_fault_zone, geo_uncertainty]


if __name__ == "__main__":
    qz.use()
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    for f in ALL:
        if key in f.__name__:
            out = f()
            if out is not None:
                print(f"  {f.__name__}: {out}")
