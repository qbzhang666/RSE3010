"""Figures for Chapters 24 to 28: openings, ground improvement, movement and
monitoring, assets, and delivery.

Data figures are computed from the equations and worked examples of the
chapters (and from ``geo_platform.settlement`` where it has the calculation);
schematics are drawn to plausible, internally consistent dimensions and are
labelled schematic in their captions. Every figure is drawn at its printed
width in the house style of ``tools/qz_figure.py`` and written as PDF and
300 dpi PNG, with the plotted data as CSV beside the data figures.

    python tools/figs/f9.py               # all figures
    python tools/figs/f9.py pillar trough # figures whose name contains a word
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
from matplotlib import patches as mp  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy.stats import norm  # noqa: E402

import qz_figure as qz  # noqa: E402
from geo_platform import settlement  # noqa: E402

FIG = os.path.join(BOOK, "figures")
W_FULL = 117 / 25.4          # SNmono text width
W_HALF = 0.62 * W_FULL
FS = 8                       # annotation size in schematics
CAR_MID = "#D9A9B4"          # a deeper carmine tint, between CAR_TINT and CAR
MAL_MID = "#B9D3CA"          # a deeper malachite tint


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save(fig, chapter: str, stem: str, columns: dict | None = None) -> None:
    out = os.path.join(FIG, chapter)
    qz.save(fig, stem, out)
    if columns:
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


def _canvas(w, xlim, ylim, ax=None):
    """An axis-free drawing area in data units, with the figure sized to keep 1:1 aspect."""
    if ax is None:
        h = w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
        fig, ax = plt.subplots(figsize=(w, h))
        fig.subplots_adjust(0, 0, 1, 1)
    else:
        fig = ax.figure
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _arrow(ax, p0, p1, color=qz.INK, lw=0.8, style="-|>", ms=6, z=5):
    ax.annotate("", xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=ms,
                                shrinkA=0, shrinkB=0))


def _dim(ax, p0, p1, text, off=(0, 0), color=qz.INK, ha="center", va="center", fs=FS):
    _arrow(ax, p0, p1, color=color, lw=0.6, style="<|-|>", ms=5)
    mx, my = (p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1]
    ax.text(mx, my, text, ha=ha, va=va, fontsize=fs, color=color)


def _lead(ax, text, xy, xytext, ha="left", va="center", color=qz.INK, fs=FS):
    ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va=va, fontsize=fs, color=color,
                arrowprops=dict(arrowstyle="-", color=qz.GREY, lw=0.5, shrinkA=1, shrinkB=0))


# ===========================================================================
# Chapter 24: shafts, caverns and multiple openings
# ===========================================================================

def shaft_stress():
    """Plan of a shaft: boundary tangential stress (Kirsch) and the spalling it locates."""
    k = 0.5                                       # sigma_h / sigma_H, sigma_H along x
    fig, ax = _canvas(W_HALF, (-3.6, 3.6), (-3.25, 3.25))
    th = np.linspace(0, 2 * np.pi, 721)
    s = (1 + k) - 2 * (1 - k) * np.cos(2 * th)    # sigma_theta / sigma_H
    a, c = 1.0, 0.5
    r = a + c * s
    ax.fill(r * np.cos(th), r * np.sin(th), color=qz.MAL_TINT, zorder=1, lw=0)
    ax.plot(r * np.cos(th), r * np.sin(th), color=qz.MAL, lw=1.1, zorder=2)
    # spalling notches where the stress peaks, perpendicular to sigma_H
    for sgn in (1, -1):
        phi = np.radians(np.linspace(-18, 18, 37)) + sgn * np.pi / 2
        depth = 0.32 * np.cos(np.linspace(-np.pi / 2, np.pi / 2, 37)) ** 0.7
        rr = a + depth
        xs = np.concatenate([rr * np.cos(phi), a * np.cos(phi[::-1])])
        ys = np.concatenate([rr * np.sin(phi), a * np.sin(phi[::-1])])
        ax.fill(xs, ys, color=qz.CAR, zorder=4, lw=0)
    ax.add_patch(mp.Circle((0, 0), a, facecolor="white", edgecolor=qz.INK, lw=1.0, zorder=3))
    # far-field stresses
    for yy in (-0.7, 0, 0.7):
        _arrow(ax, (-3.5, yy), (-2.55, yy), lw=1.0)
        _arrow(ax, (3.5, yy), (2.55, yy), lw=1.0)
    for xx in (-0.7, 0, 0.7):
        _arrow(ax, (xx, 3.2), (xx, 2.72), lw=1.0)
        _arrow(ax, (xx, -3.2), (xx, -2.72), lw=1.0)
    ax.text(-3.05, 1.05, r"$\sigma_H$", ha="center", va="bottom", fontsize=10)
    ax.text(1.0, 2.96, r"$\sigma_h$", ha="left", va="center", fontsize=10)
    ax.text(0.95, 2.05, r"$\sigma_\theta = 2.5\,\sigma_H$", ha="left", va="center", fontsize=FS)
    _lead(ax, r"$\sigma_\theta = 0.5\,\sigma_H$", (1.25, -0.04), (1.55, -1.75), ha="left", va="top")
    _save(fig, "ch24", "shaft-stress",
          {"theta_deg_from_sigmaH": np.degrees(th), "sigma_theta_over_sigmaH": s})


def shaft_base_stability():
    """Base heave and hydraulic uplift in a soft-ground shaft, side by side (schematic)."""
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 0.62 * W_FULL))
    fig.subplots_adjust(0.01, 0.0, 0.99, 0.92, wspace=0.06)
    xl, yl = (-8.5, 8.5), (-17, 2.2)
    for ax, title in zip(axes, ("Base heave", "Hydraulic uplift")):
        _canvas(None, xl, yl, ax=ax)
        ax.set_title(title, fontsize=10, pad=2)

    # --- base heave
    ax = axes[0]
    ax.plot([-8.5, -3.2], [0, 0], color=qz.INK, lw=0.8)
    ax.plot([3.2, 8.5], [0, 0], color=qz.INK, lw=0.8)
    for x0 in (-3.2, 2.8):
        ax.add_patch(mp.Rectangle((x0, -12.5), 0.4, 13.0, facecolor=qz.GREY_LIGHT, edgecolor=qz.INK, lw=0.6))
    ax.plot([-2.8, 2.8], [-9, -9], color=qz.INK, lw=0.8)
    t = np.radians(np.linspace(180, 360, 100))
    R = 3.9
    ax.plot(-3.0 + R * np.cos(t), -9 + R * np.sin(t), color=qz.CAR, lw=1.1, ls="--")
    _arrow(ax, (-5.6, -4.5), (-5.6, -7.6), color=qz.CAR, lw=1.1)
    _arrow(ax, (-0.3, -11.6), (-0.3, -9.3), color=qz.CAR, lw=1.1)
    ax.text(-0.3, -8.7, "Heave", ha="center", va="bottom", fontsize=FS, color=qz.CAR)
    for xx in (-7.6, -6.2, -4.8):
        _arrow(ax, (xx, 1.6), (xx, 0.15), lw=0.8)
    ax.text(-6.2, 1.75, r"$\gamma z$", ha="center", va="bottom", fontsize=FS)
    _dim(ax, (5.2, 0), (5.2, -9), r"$H$", off=(0.45, 0), ha="left")
    ax.text(0, -15.5, "Soft clay", ha="center", fontsize=FS, color=qz.GREY)

    # --- hydraulic uplift
    ax = axes[1]
    ax.add_patch(mp.Rectangle((-8.5, -13), 17, 7, facecolor=qz.GREY_LIGHT, edgecolor="none", alpha=0.45))
    ax.plot([-8.5, -3.2], [0, 0], color=qz.INK, lw=0.8)
    ax.plot([3.2, 8.5], [0, 0], color=qz.INK, lw=0.8)
    for x0 in (-3.2, 2.8):
        ax.add_patch(mp.Rectangle((x0, -10.5), 0.4, 11.0, facecolor=qz.GREY_LIGHT, edgecolor=qz.INK, lw=0.6))
    ax.add_patch(mp.Rectangle((-2.8, -9), 5.6, 9, facecolor="white", edgecolor="none"))
    ax.plot([-2.8, 2.8], [-9, -9], color=qz.INK, lw=0.8)
    ax.plot([-8.5, 8.5], [-13, -13], color=qz.INK, lw=0.6)
    ax.plot([-8.5, 8.5], [-1.5, -1.5], color=qz.CAR, lw=0.8, ls=(0, (4, 2)))
    ax.plot([6.6, 7.0, 7.4, 6.6], [-0.9, -1.5, -0.9, -0.9], color=qz.CAR, lw=0.7)
    for xx in (-1.8, 0, 1.8):
        _arrow(ax, (xx, -15.6), (xx, -13.2), color=qz.CAR, lw=1.1)
    ax.text(2.4, -14.5, "Uplift", ha="left", va="center", fontsize=FS, color=qz.CAR)
    _arrow(ax, (0.9, -10.0), (0.9, -12.0), color=qz.MAL, lw=1.1)
    ax.text(1.2, -10.9, r"$W$", ha="left", va="center", fontsize=FS, color=qz.MAL)
    _dim(ax, (-1.4, -9), (-1.4, -13), r"$t$", off=(-0.35, 0), ha="right")
    _dim(ax, (5.4, -13), (5.4, -1.5), r"$h_w$", off=(0.35, 0), ha="left")
    ax.text(-6.0, -7.4, "Aquitard", ha="center", fontsize=FS, color=qz.GREY)
    ax.text(-6.0, -15.5, "Aquifer", ha="center", fontsize=FS, color=qz.GREY)
    _save(fig, "ch24", "shaft-base-stability")


def portal_section():
    """Longitudinal section through a portal (schematic)."""
    fig, ax = _canvas(W_FULL, (-12, 44), (-2.5, 27))
    x = np.linspace(-12, 44, 561)
    gs, g0 = 0.5, 12.0                            # natural ground y = g0 + gs x
    tf = 3.73                                     # portal face at 75 degrees from (0, 0)
    xf_top = g0 / (tf - gs)
    nat = g0 + gs * x
    # natural ground before the cut (dotted) and the ground after it
    ax.plot(x[x < xf_top], nat[x < xf_top], color=qz.GREY, lw=0.6, ls=":")
    gx = np.concatenate([[-12, 0, xf_top], x[x > xf_top]])
    gy = np.concatenate([[0, 0, tf * xf_top], nat[x > xf_top]])
    # weathered zone: a band 8 m deep below the natural surface, behind the face
    xw = x[x > 0]
    top = np.minimum(g0 + gs * xw, tf * xw)
    ax.fill_between(xw, top, np.minimum(g0 + gs * xw - 8, top), color=qz.GREY_LIGHT, alpha=0.55, lw=0)
    ax.plot(gx, gy, color=qz.INK, lw=0.9)
    # tunnel: excavated length and the profile ahead
    H, xe = 7.0, 12.0
    ax.fill_between([0, xe], [0, 0], [H, H], color="white", zorder=2)
    ax.plot([H / tf, xe], [H, H], color=qz.INK, lw=0.9, zorder=3)
    ax.plot([0, xe], [0, 0], color=qz.INK, lw=0.9, zorder=3)
    ax.plot([xe, xe], [0, H], color=qz.INK, lw=0.9, zorder=3)
    ax.plot([xe, 44], [H, H], color=qz.GREY, lw=0.6, ls="--")
    ax.plot([xe, 44], [0, 0], color=qz.GREY, lw=0.6, ls="--")
    # umbrella arch: two overlapping stages above the crown
    for xs in ((H + 0.3) / tf, 9.5):
        xx = np.array([xs, xs + 15])
        ax.plot(xx, H + 0.3 + 0.07 * (xx - xs), color=qz.MAL, lw=1.3, zorder=3)
    # soil nails on the portal face above the umbrella
    for yy in (10.2, 12.2):
        x0 = yy / tf
        ang = math.radians(12)
        ax.plot([x0, x0 + 5.5 * math.cos(ang)], [yy, yy - 5.5 * math.sin(ang)], color=qz.CAR, lw=1.1, zorder=3)
    # cover above the crown
    xc = 16.0
    _dim(ax, (xc, H + 0.3 + 0.07 * (xc - 9.5) + 0.3), (xc, g0 + gs * xc), "")
    ax.text(xc + 0.6, 0.5 * (H + 1.0 + g0 + gs * xc), "Low cover", ha="left", va="center", fontsize=FS)
    _lead(ax, "Umbrella arch", (22.0, H + 0.3 + 0.07 * 12.5), (27.0, 4.0))
    _lead(ax, "Soil nails", (5.0, 11.4), (0.5, 19.0), ha="right")
    _lead(ax, "Weathered zone", (34.0, g0 + gs * 34 - 4), (38.0, 21.5), ha="center", va="top")
    ax.text(6.3, 3.5, "Short rounds", ha="center", va="center", fontsize=FS, zorder=4)
    ax.text(-6, 0.6, "Portal cut", ha="center", va="bottom", fontsize=FS, color=qz.GREY)
    _save(fig, "ch24", "portal-section")


def cavern_sequence():
    """Cavern cross-section: top heading, benches and the support installed from each (schematic)."""
    B, Hw, rise = 24.0, 36.0, 6.0
    fig, ax = _canvas(W_HALF, (-27, 27), (-5, 59))
    # outline: walls and an elliptical arch
    t = np.linspace(0, np.pi, 181)
    ax_x = (B / 2) * np.cos(t)
    ax_y = Hw + rise * np.sin(t)
    ox = np.concatenate([[B / 2], ax_x, [-B / 2, B / 2]])
    oy = np.concatenate([[0], ax_y, [0, 0]])
    ax.fill(np.concatenate([[B / 2], ax_x, [-B / 2]]), np.concatenate([[0], ax_y, [0]]),
            color="white", zorder=1)
    # top heading tint
    th_mask = ax_y >= 28
    ax.fill(np.concatenate([[B / 2], ax_x, [-B / 2]]), np.concatenate([[28], ax_y, [28]]),
            color=qz.MAL_TINT, zorder=1, lw=0)
    ax.plot(ox, oy, color=qz.INK, lw=1.0, zorder=3)
    levels = [28, 21, 14, 7]
    for lv in levels:
        ax.plot([-B / 2, B / 2], [lv, lv], color=qz.GREY, lw=0.6, ls="--", zorder=2)
    for n, yc in zip(range(1, 6), (34, 24.5, 17.5, 10.5, 3.5)):
        ax.text(0, yc, str(n), ha="center", va="center", fontsize=10)
    # crown support: bolts (short) and cables (long), normal to the arch
    for i, tt in enumerate(np.linspace(0.12 * np.pi, 0.88 * np.pi, 9)):
        px, py = (B / 2) * np.cos(tt), Hw + rise * np.sin(tt)
        nx, ny = (rise) * np.cos(tt), (B / 2) * np.sin(tt)        # outward normal of the ellipse
        nn = math.hypot(nx, ny)
        nx, ny = nx / nn, ny / nn
        L, col = (14.0, qz.CAR) if i % 2 == 0 else (5.0, qz.MAL)
        ax.plot([px, px + L * nx], [py, py + L * ny], color=col, lw=0.9, zorder=2)
    # wall support: cables and bolts horizontal
    for j, yy in enumerate((4, 8, 11.5, 15, 18.5, 22, 25.5, 30, 33.5)):
        L, col = (14.0, qz.CAR) if j % 2 == 0 else (5.0, qz.MAL)
        for sgn in (1, -1):
            x0 = sgn * B / 2
            ax.plot([x0, x0 + sgn * L], [yy, yy], color=col, lw=0.9, zorder=2)
    _lead(ax, "Cables", (9.96, 50.4), (13.0, 56.0))
    tt = 0.12 * np.pi + 7 * (0.76 * np.pi / 8)
    bx, by = (B / 2) * math.cos(tt), Hw + rise * math.sin(tt)
    _lead(ax, "Bolts", (bx - 2.5, by + 2.5), (-17.0, 55.0), ha="right")
    _dim(ax, (-B / 2, -1.5), (B / 2, -1.5), "", off=(0, 0))
    ax.text(0, -2.2, r"$B$", ha="center", va="top", fontsize=FS)
    _save(fig, "ch24", "cavern-sequence")


def zone_of_influence():
    """Disturbance of a hydrostatic field about a circular opening, against distance from it."""
    ra = np.linspace(1, 9, 400)
    d = (ra - 1) / 2                              # clear distance in diameters
    st, sr = 1 + ra ** -2, 1 - ra ** -2
    d5 = (math.sqrt(20) - 1) / 2
    fig, ax = qz.figure(W_HALF, 0.8 * W_HALF)
    ax.axhspan(0.95, 1.05, color=qz.GREY_LIGHT, alpha=0.5, lw=0)
    ax.plot(d, st, color=qz.MAL, lw=1.3, label=r"Tangential, $\sigma_\theta$")
    ax.plot(d, sr, color=qz.CAR, lw=1.3, ls="--", label=r"Radial, $\sigma_r$")
    ax.axvline(d5, color=qz.GREY, lw=0.7, ls=":")
    ax.text(d5 + 0.08, 0.35, f"{d5:.2f}", ha="left", va="center", fontsize=FS)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2.1)
    qz.label(ax, "Distance from boundary, $d/D$ (-)", r"Stress ratio, $\sigma/p_0$ (-)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    print(f"  zone of influence: 5 per cent at d/D = {d5:.3f}")
    _save(fig, "ch24", "zone-of-influence", {"d_over_D": d, "sigma_theta_over_p0": st, "sigma_r_over_p0": sr})


def pillar_stress():
    """Tributary area: a regular layout in plan, and pillar stress against extraction ratio."""
    fig = plt.figure(figsize=(W_FULL, 0.46 * W_FULL))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.25], wspace=0.22, left=0.0, right=0.98,
                          bottom=0.2, top=0.97)
    a0 = fig.add_subplot(gs[0])
    w, b = 10.0, 6.0
    p = w + b
    _canvas(None, (-p * 1.8, p * 1.8), (-p * 1.8, p * 1.8), ax=a0)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            a0.add_patch(mp.Rectangle((i * p - w / 2, j * p - w / 2), w, w, facecolor=qz.GREY_LIGHT,
                                      edgecolor=qz.INK, lw=0.6))
    a0.add_patch(mp.Rectangle((-p / 2, -p / 2), p, p, facecolor="none", edgecolor=qz.MAL, lw=1.1, ls="--"))
    yd = p + w / 2 + 2.0
    _dim(a0, (-w / 2, -yd), (w / 2, -yd), "", off=(0, 0))
    a0.text(0, -yd - 1.0, r"$w$", ha="center", va="top", fontsize=FS)
    _dim(a0, (w / 2, yd), (p - w / 2, yd), "", off=(0, 0))
    a0.text(p / 2, yd + 1.0, r"$b$", ha="center", va="bottom", fontsize=FS)
    e_plan = 1 - (w / p) ** 2

    ax = fig.add_subplot(gs[1])
    e = np.linspace(0, 0.9, 300)
    r = 1 / (1 - e)
    ax.plot(e, r, color=qz.MAL, lw=1.4)
    for ee in (0.5, 0.8):
        rr = 1 / (1 - ee)
        ax.plot([ee, ee, 0], [0, rr, rr], color=qz.GREY, lw=0.6, ls=":")
        ax.plot([ee], [rr], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9)
    ax.plot([e_plan], [1 / (1 - e_plan)], "s", mfc=qz.MAL, mec=qz.MAL, ms=4)
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 10.5)
    qz.label(ax, "Extraction ratio, $e$ (-)", r"Pillar stress, $\sigma_p/\gamma z$ (-)")
    print(f"  pillar plan: e = {e_plan:.3f}, ratio {1 / (1 - e_plan):.2f}")
    _save(fig, "ch24", "pillar-stress", {"e": e, "sigma_p_over_gamma_z": r})


# ===========================================================================
# Chapter 25: ground improvement, pre-support and hazards
# ===========================================================================

def groutability():
    """Groutability ratio N = D15 soil / D85 grout, with the limits 24 and 11."""
    fig, ax = qz.figure(W_HALF, 0.85 * W_HALF)
    d15 = np.logspace(-2, 1, 200)                       # mm
    g24, g11 = 1000 * d15 / 24, 1000 * d15 / 11        # D85 in micrometres
    ax.fill_between(d15, 0.1, g24, color=qz.MAL_TINT, lw=0)
    ax.fill_between(d15, g11, 1e5, color=qz.CAR_TINT, lw=0)
    ax.plot(d15, g24, color=qz.MAL, lw=1.2)
    ax.plot(d15, g11, color=qz.CAR, lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 10)
    ax.set_ylim(1, 1000)
    ax.text(1.6, 6, r"$N > 24$", ha="center", va="center", fontsize=FS + 1, color=qz.MAL)
    ax.text(0.04, 150, r"$N < 11$", ha="center", va="center", fontsize=FS + 1, color=qz.CAR)
    qz.label(ax, "Soil size, $D_{15}$ (mm)", r"Grout size, $D_{85}$ ($\mu$m)")
    fig.tight_layout()
    _save(fig, "ch25", "groutability", {"D15_mm": d15, "D85_um_N24": g24, "D85_um_N11": g11})


def jet_grouting():
    """Single, double and triple fluid jet grouting (schematic)."""
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 0.62 * W_FULL))
    fig.subplots_adjust(0.0, 0.0, 1.0, 0.93, wspace=0.02)
    radii = (0.75, 1.15, 1.5)
    tops = (0.35, 0.45, 0.9)
    ym = -4.0
    for ax, title, R, tp in zip(axes, ("Single fluid", "Double fluid", "Triple fluid"), radii, tops):
        _canvas(None, (-2.1, 2.1), (-8.6, 1.2), ax=ax)
        ax.set_title(title, fontsize=10, pad=2)
        ax.plot([-2.1, 2.1], [0, 0], color=qz.INK, lw=0.8)
        # column formed up to the current jetting level
        ax.add_patch(mp.Rectangle((-R, -8.0), 2 * R, 8.0 + ym + tp, facecolor=qz.MAL_TINT, edgecolor=qz.MAL,
                                  lw=0.9))
        # borehole above (spoil annulus) and rod
        ax.add_patch(mp.Rectangle((-0.28, ym + tp), 0.56, -ym - tp, facecolor="white", edgecolor=qz.GREY,
                                  lw=0.5))
        ax.add_patch(mp.Rectangle((-0.1, ym - 0.2), 0.2, 1.4 - ym, facecolor=qz.GREY, edgecolor="none"))
        _arrow(ax, (0.19, ym + 1.0), (0.19, -0.6), color=qz.GREY, lw=0.6, ms=4)
        _arrow(ax, (-0.19, ym + 1.0), (-0.19, -0.6), color=qz.GREY, lw=0.6, ms=4)
        # rotation and withdrawal
        _arrow(ax, (0.55, 0.5), (0.55, 1.15), color=qz.INK, lw=0.7, ms=4)
    # jets
    ax = axes[0]
    for s in (1, -1):
        _arrow(ax, (s * 0.12, ym), (s * 0.95, ym), color=qz.MAL, lw=1.4, ms=6)
    ax = axes[1]
    for s in (1, -1):
        _arrow(ax, (s * 0.12, ym), (s * 1.4, ym), color=qz.MAL, lw=1.4, ms=6)
        ax.plot([s * 0.12, s * 1.4], [ym + 0.15, ym + 0.32], color=qz.GREY, lw=0.6, ls=":")
        ax.plot([s * 0.12, s * 1.4], [ym - 0.15, ym - 0.32], color=qz.GREY, lw=0.6, ls=":")
    ax = axes[2]
    for s in (1, -1):
        _arrow(ax, (s * 0.12, ym + 0.45), (s * 1.75, ym + 0.45), color=qz.CAR, lw=1.4, ms=6)
        ax.plot([s * 0.12, s * 1.75], [ym + 0.6, ym + 0.77], color=qz.GREY, lw=0.6, ls=":")
        ax.plot([s * 0.12, s * 1.75], [ym + 0.3, ym + 0.13], color=qz.GREY, lw=0.6, ls=":")
        _arrow(ax, (s * 0.12, ym - 0.45), (s * 1.0, ym - 0.45), color=qz.MAL, lw=1.4, ms=6)
    _lead(axes[0], "Grout", (0.6, ym), (1.05, ym - 1.4), color=qz.MAL)
    _lead(axes[1], "Air", (1.2, ym + 0.29), (1.0, ym + 1.6), ha="left", va="bottom", color=qz.GREY)
    _lead(axes[2], "Water", (1.1, ym + 0.45), (0.8, ym + 2.0), ha="left", va="bottom", color=qz.CAR)
    _lead(axes[2], "Grout", (-0.7, ym - 0.45), (-0.75, ym - 1.6), ha="center", va="top", color=qz.MAL)
    axes[0].text(0.45, -1.8, "Spoil", ha="left", va="center", fontsize=FS, color=qz.GREY)
    _save(fig, "ch25", "jet-grouting")


def compensation_grouting():
    """Compensation grouting between a tunnel and a building (schematic)."""
    fig, ax = _canvas(W_FULL, (-30, 32), (-23, 12))
    x = np.linspace(-30, 32, 400)
    ax.plot(x, np.zeros_like(x), color=qz.MAL, lw=1.1, zorder=3)
    trough = -2.6 * np.exp(-((x - 8) ** 2) / (2 * 8.0 ** 2))
    ax.plot(x, trough, color=qz.CAR, lw=1.0, ls="--", zorder=3)
    ax.add_patch(mp.Rectangle((0, 0), 17, 9, facecolor=qz.GREY_LIGHT, edgecolor=qz.INK, lw=0.6, zorder=2))
    # tunnel and ground loss
    tc, tr = (8.0, -18.0), 3.0
    ax.add_patch(mp.Circle(tc, tr, facecolor="white", edgecolor=qz.INK, lw=0.9, zorder=3))
    for ang in np.radians([30, 90, 150]):
        p0 = (tc[0] + (tr + 1.6) * math.cos(ang), tc[1] + (tr + 1.6) * math.sin(ang))
        p1 = (tc[0] + (tr + 0.3) * math.cos(ang), tc[1] + (tr + 0.3) * math.sin(ang))
        _arrow(ax, p0, p1, color=qz.CAR, lw=0.9, ms=5)
    # shaft and sleeve pipes
    ax.add_patch(mp.Rectangle((-24, -11), 4, 11, facecolor="white", edgecolor=qz.INK, lw=0.8, zorder=3))
    for yend in (-5.5, -7.0, -8.5):
        xe = 22
        ax.plot([-20, xe], [-7.0, yend], color=qz.MAL, lw=1.0, zorder=2)
        for xp in np.arange(-2, xe, 3.0):
            yp = -7.0 + (yend + 7.0) * (xp + 20) / (xe + 20)
            ax.plot([xp], [yp], marker="o", ms=1.8, color=qz.MAL)
    for xp in (2, 8, 14):
        _arrow(ax, (xp, -5.3), (xp, -3.6), color=qz.MAL, lw=0.8, ms=4)
    _lead(ax, "Sleeve pipes", (-12, -7.0), (-14, -15.0), ha="center", va="top")
    ax.text(-22, 0.5, "Shaft", ha="center", va="bottom", fontsize=FS)
    _lead(ax, "Greenfield", (20, -1.0), (24, -4.5), ha="left", color=qz.CAR)
    _lead(ax, "Compensated", (26, 0.0), (24.5, 3.2), ha="left", color=qz.MAL)
    _save(fig, "ch25", "compensation-grouting")


def _freeze_ring(ax, radii, centres):
    for (cx, cy), r in zip(centres, radii):
        ax.add_patch(mp.Circle((cx, cy), r, facecolor="none", edgecolor=qz.MAL, lw=1.4, zorder=2))
    for (cx, cy), r in zip(centres, radii):
        ax.add_patch(mp.Circle((cx, cy), r - 0.04, facecolor=qz.MAL_TINT, edgecolor="none", zorder=3))


def freeze_wall():
    """Freeze pipes on a ring: columns, closure, and the gap seepage leaves (schematic)."""
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 0.39 * W_FULL))
    fig.subplots_adjust(0.0, 0.0, 1.0, 0.88, wspace=0.03)
    n, Rr = 16, 3.0
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pipes = [(Rr * math.cos(p), Rr * math.sin(p)) for p in phi]
    for ax, title in zip(axes, ("Columns", "Closed wall", "Seepage")):
        _canvas(None, (-5.2, 5.2), (-4.6, 4.6), ax=ax)
        ax.set_title(title, fontsize=10, pad=2)
        ax.add_patch(mp.Circle((0, 0), 1.8, facecolor="none", edgecolor=qz.GREY, lw=0.7, ls="--", zorder=1))
    _freeze_ring(axes[0], [0.45] * n, pipes)
    _freeze_ring(axes[1], [1.05] * n, pipes)
    rad = [max(0.32, 1.05 - 0.75 * max(0.0, -math.cos(p)) ** 0.8) for p in phi]
    cen = [(x + 0.25, y) for x, y in pipes]
    _freeze_ring(axes[2], rad, cen)
    for ax in axes:
        for x, y in pipes:
            ax.plot([x], [y], "o", ms=2.2, color=qz.INK, zorder=4)
    for yy in (-1.6, 0, 1.6):
        _arrow(axes[2], (-5.1, yy), (-4.15, yy), color=qz.CAR, lw=1.0, ms=5)
    axes[2].text(-4.6, 2.2, r"$v$", ha="center", va="bottom", fontsize=FS + 1, color=qz.CAR)
    _save(fig, "ch25", "freeze-wall")


def presupport():
    """Umbrella arch in overlapping stages and face dowels in the advance core (schematic)."""
    fig, ax = _canvas(W_FULL, (-21, 31), (-2.5, 15))
    H = 9.0
    ax.add_patch(mp.Rectangle((-21, 0), 21, H, facecolor="white", edgecolor="none"))
    ax.plot([-21, 0], [H, H], color=qz.INK, lw=1.0)
    ax.plot([-21, 0], [0, 0], color=qz.INK, lw=1.0)
    ax.plot([0, 0], [0, H], color=qz.INK, lw=1.2)
    ax.plot([0, 31], [H, H], color=qz.GREY, lw=0.6, ls="--")
    ax.plot([0, 31], [0, 0], color=qz.GREY, lw=0.6, ls="--")
    # colours as in the render umbrella_arch: both stages malachite, dowels pale malachite
    stages = ((-14.0, qz.MAL), (-3.0, qz.MAL))
    for xs, col in stages:
        xx = np.array([xs, xs + 15.0])
        ax.plot(xx, H + 0.15 + 0.1 * (xx - xs), color=col, lw=1.3)
    for yy in (1.5, 3.0, 4.5, 6.0, 7.5):
        ax.plot([0, 15], [yy, yy], color=qz.MAL_SHADES[3], lw=1.1)
    _dim(ax, (-3.0, H + 3.2), (1.0, H + 3.2), "", off=(0, 0))
    ax.text(-1.0, H + 3.5, "Overlap", ha="center", va="bottom", fontsize=FS)
    ax.plot([-3.0, -3.0], [H + 0.2, H + 3.2], color=qz.GREY, lw=0.4, ls=":")
    ax.plot([1.0, 1.0], [H + 1.6, H + 3.2], color=qz.GREY, lw=0.4, ls=":")
    _lead(ax, "Umbrella arch", (8.0, H + 0.15 + 1.1 + 0.05), (14.5, H + 4.4))
    _lead(ax, "Face dowels", (14.0, 6.0), (18.5, 7.6))
    ax.text(0.3, -0.6, "Face", ha="left", va="top", fontsize=FS)
    ax.text(-10.5, H / 2, "Supported tunnel", ha="center", va="center", fontsize=FS, color=qz.GREY)
    _save(fig, "ch25", "presupport")


# ===========================================================================
# Chapter 26: ground movement, monitoring and the observational method
# ===========================================================================

def subsurface_trough():
    """Troughs at depth above a tunnel in clay (Mair et al. 1993), constant volume."""
    z0, D, vl = 20.0, 6.0, 1.0
    vs = vl / 100 * math.pi * D ** 2 / 4
    y = np.linspace(-35, 35, 351)
    fig, ax = qz.figure(W_FULL, 0.42 * W_FULL)
    cols = {"y_m": y}
    for z, col, ls in ((0.0, qz.MAL_SHADES[3], "-"), (8.0, qz.MAL, "--"), (14.0, qz.MAL_SHADES[0], "-.")):
        i = 0.175 * z0 + 0.325 * (z0 - z)
        smax = vs / (i * math.sqrt(2 * math.pi)) * 1000
        s = smax * np.exp(-y ** 2 / (2 * i ** 2))
        ax.plot(y, s, color=col, ls=ls, lw=1.4, label=f"$z = {z:g}$ m, $i = {i:.1f}$ m")
        cols[f"S_mm_z{z:g}"] = s
        print(f"  subsurface: z {z:g} m, i {i:.2f} m, S_max {smax:.1f} mm")
    ax.set_xlim(-35, 35)
    ax.set_ylim(23, -1)
    qz.label(ax, "Offset from the axis, $y$ (m)", "Settlement, $S$ (mm)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "ch26", "subsurface-trough", cols)


def _deflection(y, s, a, b):
    m = (y >= a) & (y <= b)
    ya, sa = y[m], s[m]
    chord = sa[0] + (sa[-1] - sa[0]) * (ya - ya[0]) / (ya[-1] - ya[0])
    dev = sa - chord
    k = int(np.argmax(np.abs(dev)))
    return ya, chord, ya[k], sa[k], chord[k], float(abs(dev[k]))


def deflection_ratio():
    """Sagging and hogging lengths of a building on the greenfield trough, and their deflections."""
    bore = settlement.Bore(0.0, 20.0, 6.0, 1.0, 0.5)
    y = np.linspace(-20, 34, 1081)
    s = settlement.transverse([bore], y)
    y0, y1, yi = -6.0, 24.0, bore.i
    fig, ax = qz.figure(W_FULL, 0.47 * W_FULL)
    ax.plot(y, s, color=qz.GREY, lw=0.8)
    m = (y >= y0) & (y <= y1)
    ax.plot(y[m], s[m], color=qz.INK, lw=1.5)
    out = {}
    for (a, b), col, name in (((y0, yi), qz.MAL, "sag"), ((yi, y1), qz.CAR, "hog")):
        ya, chord, yk, sk, ck, dlt = _deflection(y, s, a, b)
        ax.plot(ya, chord, color=col, lw=1.0, ls="--")
        _arrow(ax, (yk, ck), (yk, sk), color=col, lw=0.9, style="<|-|>", ms=5)
        L = b - a
        out[name] = (dlt, L)
        if name == "sag":
            ax.text(yk - 0.5, 0.5 * (sk + ck) + 0.1, r"$\Delta_{\mathrm{sag}}$", ha="right", va="center",
                    fontsize=FS + 1, color=col)
        else:
            ax.annotate(r"$\Delta_{\mathrm{hog}}$", xy=(yk + 0.2, 0.5 * (sk + ck)), xytext=(yk + 2.5, -0.5),
                        ha="left", va="center", fontsize=FS + 1, color=col,
                        arrowprops=dict(arrowstyle="-", color=qz.GREY, lw=0.5, shrinkA=1, shrinkB=0))
        ax.add_patch(mp.Rectangle((a, -3.2), L, 1.6, facecolor=qz.MAL_TINT if name == "sag" else qz.CAR_TINT,
                                  edgecolor=col, lw=0.7))
        ax.text(a + L / 2, -2.4, "Sagging" if name == "sag" else "Hogging", ha="center", va="center",
                fontsize=FS, color=col)
    ax.axvline(yi, color=qz.GREY, lw=0.6, ls=":")
    ax.text(yi + 0.4, 13.2, r"$y = i$", ha="left", va="bottom", fontsize=FS)
    ax.set_xlim(-20, 34)
    ax.set_ylim(14, -4)
    qz.label(ax, "Offset from the axis, $y$ (m)", "Settlement, $S$ (mm)")
    fig.tight_layout()
    for k, (dlt, L) in out.items():
        print(f"  deflection {k}: Delta {dlt:.2f} mm over L {L:.1f} m, Delta/L {dlt / L / 1000:.2e}")
    _save(fig, "ch26", "deflection-ratio", {"y_m": y, "S_mm": s})


def damage_interaction():
    """Damage category against deflection ratio and horizontal strain, hogging, L/H = 1, E/G = 2.6.

    Deep-beam relations of Burland and Wroth (1974), neutral axis at the bottom
    fibre in hogging, with horizontal strain combined as Boscardin and Cording
    (1989) proposed: eps_br = eps_b + eps_h; eps_dr = 0.35 eps_h + sqrt((0.65 eps_h)^2 + eps_d^2).
    """
    nu, LH = 0.3, 1.0
    EG = 2 * (1 + nu)
    cb = LH / 12 * (1 + 6 * EG / LH ** 2)          # Delta/L per unit bending strain
    cd = 1 + LH ** 2 / (6 * EG)                    # Delta/L per unit diagonal strain
    eh = np.linspace(0, 3.2, 321)                  # 10^-3
    dl = np.linspace(0, 3.4, 341)
    EH, DL = np.meshgrid(eh, dl)
    eb, ed = DL / cb, DL / cd
    ebr = eb + EH
    edr = EH * (1 - nu) / 2 + np.sqrt((EH * (1 + nu) / 2) ** 2 + ed ** 2)
    et = np.maximum(ebr, edr)
    lv = [0, 0.5, 0.75, 1.5, 3.0, 99]
    shades = [qz.SEQ_CAR(v) for v in (0.04, 0.2, 0.38, 0.58, 0.8)]
    fig, ax = qz.figure(W_HALF, 0.92 * W_HALF)
    ax.contourf(EH, DL, et, levels=lv, colors=shades)
    ax.contour(EH, DL, et, levels=lv[1:-1], colors=[qz.INK], linewidths=0.6)
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 3.4)
    for x, y, t, c in ((0.07, 1.05, "Slight", qz.INK), (0.5, 2.15, "Moderate", "white"),
                       (2.35, 2.9, "Severe", "white")):
        ax.text(x, y, t, fontsize=FS - 1, color=c, ha="left", va="bottom")
    # the two smallest zones are narrower than their names: label them by leader
    for t, xy, xyt in (("Very slight", (0.3, 0.45), (1.45, 0.85)), ("Negligible", (0.15, 0.2), (1.75, 0.3))):
        ax.annotate(t, xy=xy, xytext=xyt, ha="left", va="center", fontsize=FS - 1, color="white",
                    arrowprops=dict(arrowstyle="-", color=qz.INK, lw=0.5, shrinkA=1, shrinkB=0))
        ax.plot(*xy, "o", ms=1.8, color=qz.INK)
    qz.label(ax, r"Horizontal strain, $\varepsilon_h$ ($10^{-3}$)", r"Deflection ratio, $\Delta/L$ ($10^{-3}$)")
    fig.tight_layout()
    # boundaries for the CSV
    cols = {"eps_h_1e-3": eh}
    for el in lv[1:-1]:
        b_bend = cb * np.clip(el - eh, 0, None)
        disc = (el - 0.35 * eh) ** 2 - (0.65 * eh) ** 2
        b_shear = cd * np.sqrt(np.clip(disc, 0, None))
        cols[f"DL_limit_eps{el:g}e-3"] = np.minimum(b_bend, b_shear)
    print(f"  damage interaction: c_b {cb:.3f}, c_d {cd:.3f}")
    _save(fig, "ch26", "damage-interaction", cols)


def void_migration():
    """Height of void migration before bulking arrests it, h = H_v/(B - 1)."""
    B = np.linspace(1.05, 1.6, 300)
    fig, ax = qz.figure(W_HALF, 0.8 * W_HALF)
    cols = {"B": B}
    for hv, col, ls in ((1, qz.MAL_SHADES[3], ":"), (2, qz.MAL, "--"), (3, qz.MAL_SHADES[0], "-")):
        h = hv / (B - 1)
        ax.plot(B, h, color=col, ls=ls, lw=1.3, label=f"$H_v = {hv}$ m")
        cols[f"h_m_Hv{hv}"] = h
    for bb in (1.15, 1.30):
        hh = 3 / (bb - 1)
        ax.plot([bb], [hh], "o", mfc="white", mec=qz.CAR, ms=4.5, mew=1.0)
        ax.plot([bb, bb], [0, hh], color=qz.GREY, lw=0.5, ls=":")
    ax.set_xlim(1.05, 1.6)
    ax.set_ylim(0, 60)
    qz.label(ax, "Bulking factor, $B$ (-)", "Migration height, $h$ (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "ch26", "void-migration", cols)


def instrument_section():
    """A monitoring section across a tunnel beneath a building (schematic)."""
    fig, ax = _canvas(W_FULL, (-30, 30), (-27, 12))
    ax.plot([-30, 30], [0, 0], color=qz.INK, lw=0.9)
    ax.add_patch(mp.Rectangle((9, 0), 13, 9, facecolor=qz.GREY_LIGHT, edgecolor=qz.INK, lw=0.6))
    tc, tr = (0.0, -18.0), 3.5
    ax.add_patch(mp.Circle(tc, tr, facecolor="white", edgecolor=qz.INK, lw=1.0, zorder=3))
    # convergence chords between five points on the lining
    pts = [(tc[0] + tr * math.cos(a), tc[1] + tr * math.sin(a)) for a in np.radians([90, 18, 162, -30, 210])]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if (i, j) in ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4)):
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color=qz.MAL, lw=0.7, zorder=4)
    for p in pts:
        ax.plot(*p, "o", ms=2.5, color=qz.MAL, zorder=5)
    # levelling points on the surface
    for xx in (-24, -18, -12, -6, 0, 5):
        ax.plot([xx], [0.6], marker="v", ms=4, color=qz.MAL, zorder=4)
    # prisms on the facade and a tiltmeter on the side wall
    for yy in (3, 7):
        ax.plot([9], [yy], marker="D", ms=3.5, color=qz.CAR, zorder=4)
    ax.add_patch(mp.Rectangle((21.5, 4.0), 1.2, 1.2, facecolor=qz.CAR, edgecolor="none", zorder=4))
    # extensometer above the crown
    ax.plot([-1.5, -1.5], [0, -13.2], color=qz.CAR, lw=1.0)
    for yy in (-4, -8, -11, -13.2):
        ax.plot([-2.0, -1.0], [yy, yy], color=qz.CAR, lw=1.2)
    # inclinometer beside the tunnel
    ax.plot([-8.5, -8.5], [0, -26], color=qz.MAL, lw=1.0, ls=(0, (3, 1.5)))
    # piezometers at two horizons
    ax.plot([-15, -15], [0, -21], color=qz.GREY, lw=0.8)
    for yy in (-10, -21):
        ax.plot([-15], [yy], "o", ms=4, color=qz.INK, zorder=4)
    _lead(ax, "Levelling", (-18, 0.8), (-22, 6.0), ha="center", va="bottom")
    _lead(ax, "Prisms", (9, 7), (4.0, 10.0), ha="right")
    _lead(ax, "Tiltmeter", (22.7, 4.6), (26.5, 10.0), ha="center", va="bottom")
    _lead(ax, "Extensometer", (-1.5, -6.0), (4.5, -6.0))
    _lead(ax, "Inclinometer", (-8.5, -23.0), (-4.0, -25.0))
    _lead(ax, "Piezometers", (-15, -10), (-24.0, -14.0), ha="center", va="top")
    _lead(ax, "Convergence", (3.0, -16.0), (8.0, -13.0))
    _save(fig, "ch26", "instrument-section")


def trigger_levels():
    """Trigger levels against the predicted development, and decelerating against steady movement."""
    fig, (a1, a2) = qz.figure(W_FULL, 0.45 * W_FULL, ncols=2)
    x = np.linspace(-3, 4, 300)
    smax, amber, red, limit = 11.3, 13.0, 16.0, 20.0
    pred = smax * norm.cdf(x)
    a1.axhspan(amber, red, color=qz.CAR_TINT, lw=0)
    a1.axhspan(red, limit, color=CAR_MID, lw=0)
    a1.axhline(limit, color=qz.CAR, lw=1.1)
    a1.plot(x, pred, color=qz.MAL, lw=1.4)
    for yv, t in ((amber, "Amber"), (red, "Red"), (limit, "Limit")):
        a1.text(-2.85, yv + 0.3, t, fontsize=FS, va="bottom", ha="left")
    a1.text(1.7, smax - 1.6, "Prediction", fontsize=FS, va="top", ha="left", color=qz.MAL)
    a1.set_xlim(-3, 4)
    a1.set_ylim(0, 23)
    qz.label(a1, "Distance behind face, $X/i$ (-)", "Settlement, $S$ (mm)")

    t = np.linspace(0, 60, 300)
    dec = 25 * (1 - np.exp(-t / 10))
    std = 0.2 * t
    a2.plot(t, dec, color=qz.MAL, lw=1.4, label="Decelerating")
    a2.plot(t, std, color=qz.CAR, lw=1.4, ls="--", label="Steady rate")
    a2.set_xlim(0, 60)
    a2.set_ylim(0, 30)
    qz.label(a2, "Time, $t$ (days)", "Convergence, $u$ (mm)")
    a2.legend(loc="lower right", bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout()
    _save(fig, "ch26", "trigger-levels", {"X_over_i": x, "S_pred_mm": pred, "t_days": t,
                                          "u_decel_mm": dec, "u_steady_mm": std})


# ===========================================================================
# Chapter 27: operation and asset management
# ===========================================================================

def _box(ax, xy, w, h, text, edge=qz.MAL, fill="white", fs=FS, lw=0.9, color=qz.INK):
    ax.add_patch(mp.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.0,rounding_size=0.12",
                                   facecolor=fill, edgecolor=edge, lw=lw, zorder=3))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4, color=color)


def handover_record():
    """What handover must transfer, and the operator's record it starts (schematic workflow)."""
    fig, ax = _canvas(W_FULL, (0, 19.6), (0.2, 7.4))
    items = ["Ground model", "As-built record", "Residual risks", "Monitoring baseline", "Maintenance assumptions"]
    for k, t in enumerate(items):
        y = 6.2 - k * 1.3
        _box(ax, (0.1, y), 5.8, 0.9, t)
        _arrow(ax, (5.9, y + 0.45), (7.4, 3.75 + (2 - k) * 0.35), color=qz.GREY, lw=0.7, ms=5)
    _box(ax, (7.4, 2.6), 2.9, 2.3, "Handover\npackage", edge=qz.INK, fill=qz.MAL_TINT, fs=9)
    _box(ax, (7.4, 0.4), 2.9, 1.2, "Named\ncustodian", edge=qz.CAR, fill=qz.CAR_TINT, fs=FS)
    ax.plot([8.85, 8.85], [1.6, 2.6], color=qz.CAR, lw=0.9)
    _arrow(ax, (10.3, 3.75), (11.55, 3.75), lw=0.9)
    # operator's cycle
    cx, cy, R = 15.55, 3.8, 2.4
    nodes = {"Record": 180, "Inspection": 90, "Forecast": 0, "Intervention": 270}
    pos = {}
    for t, ang in nodes.items():
        px, py = cx + R * math.cos(math.radians(ang)), cy + R * 1.15 * math.sin(math.radians(ang))
        pos[t] = (px, py)
        _box(ax, (px - 1.55, py - 0.4), 3.1, 0.8, t, edge=qz.MAL)
    for a, b in (("Record", "Inspection"), ("Inspection", "Forecast"), ("Forecast", "Intervention"),
                 ("Intervention", "Record")):
        pa, pb = pos[a], pos[b]
        ax.annotate("", xy=(pb[0] - 0.55 * np.sign(pb[0] - pa[0]) * 0 + (pa[0] - pb[0]) * 0.35,
                            pb[1] + (pa[1] - pb[1]) * 0.42),
                    xytext=(pa[0] + (pb[0] - pa[0]) * 0.35, pa[1] + (pb[1] - pa[1]) * 0.42),
                    arrowprops=dict(arrowstyle="-|>", color=qz.MAL, lw=0.9, mutation_scale=6,
                                    connectionstyle="arc3,rad=-0.25", shrinkA=0, shrinkB=0))
    _save(fig, "ch27", "handover-record")


def deterioration_inspection():
    """Condition under planned maintenance and under deferral, with the inspection rhythm (illustrative)."""
    t = np.linspace(0, 60, 1201)
    d = lambda tau: 0.6 * (tau / 40.0) ** 2       # noqa: E731  loss of condition since last restoration
    deferred = 1 - d(t)
    maintained = np.empty_like(t)
    for k, tt in enumerate(t):
        n = int(tt // 12)
        maintained[k] = 1 - 0.025 * n - d(tt - 12 * n)
    fig, ax = qz.figure(W_FULL, 0.45 * W_FULL)
    ax.axhline(0.5, color=qz.GREY, lw=0.7, ls=":")
    ax.text(59, 0.52, "Intervention level", ha="right", va="bottom", fontsize=FS)
    ax.plot(t[:-1], maintained[:-1], color=qz.MAL, lw=1.4, label="Planned maintenance")
    keep = deferred >= 0.2
    ax.plot(t[keep], deferred[keep], color=qz.CAR, lw=1.4, ls="--", label="Deferred")
    for tg in np.arange(2, 59, 2):
        ax.plot([tg, tg], [0, 0.035], color=qz.GREY, lw=0.6)
    for tp in np.arange(6, 59, 6):
        ax.plot([tp], [0.06], marker="v", ms=4, color=qz.INK)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.05)
    qz.label(ax, "Time in service, $t$ (years)", "Condition index, $C$ (-)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.1))
    fig.tight_layout()
    _save(fig, "ch27", "deterioration-inspection", {"t_years": t, "C_maintained": maintained,
                                                    "C_deferred": deferred})


def relining_clearance():
    """Internal area lost to relining against its thickness, for three bore radii."""
    tmm = np.linspace(0, 450, 300)
    fig, ax = qz.figure(W_HALF, 0.82 * W_HALF)
    ax.axvspan(50, 150, color=qz.MAL_TINT, lw=0)
    ax.axvspan(200, 400, color=qz.CAR_TINT, lw=0)
    cols = {"t_mm": tmm}
    for R, col, ls in ((2.5, qz.MAL_SHADES[0], "-"), (3.5, qz.MAL, "--"), (5.0, qz.MAL_SHADES[3], ":")):
        loss = (1 - (1 - tmm / 1000 / R) ** 2) * 100
        ax.plot(tmm, loss, color=col, ls=ls, lw=1.4, label=f"$R = {R:g}$ m")
        cols[f"area_lost_pct_R{R:g}"] = loss
    ax.text(100, 33.5, "Sprayed", ha="center", va="top", fontsize=FS)
    ax.text(300, 33.5, "Segmental", ha="center", va="top", fontsize=FS)
    ax.set_xlim(0, 450)
    ax.set_ylim(0, 35)
    qz.label(ax, "Relining thickness, $t$ (mm)", r"Area lost, $\Delta A/A$ (%)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    fig.tight_layout()
    _save(fig, "ch27", "relining-clearance", cols)


def discounting():
    """Discount factor over a century, and two illustrative intervention options against the rate."""
    fig, (a1, a2) = qz.figure(W_FULL, 0.45 * W_FULL, ncols=2)
    t = np.linspace(0, 100, 401)
    cols = {"t_years": t}
    for r, col, ls in ((0.02, qz.MAL_SHADES[0], "-"), (0.04, qz.MAL, "--"), (0.07, qz.CAR, ":")):
        df = (1 + r) ** -t
        a1.plot(t, df, color=col, ls=ls, lw=1.4, label=f"$r = {r * 100:g}$%")
        cols[f"DF_r{r * 100:g}"] = df
    a1.set_xlim(0, 100)
    a1.set_ylim(0, 1.02)
    qz.label(a1, "Time ahead, $t$ (years)", "Factor, $(1+r)^{-t}$ (-)")
    a1.legend(loc="upper right")

    r = np.linspace(0, 0.08, 161)
    rep = sum((1 + r) ** -(15.0 * k) for k in range(7))
    rel = 2.5 + 0.5 * (1 + r) ** -50.0
    a2.plot(r * 100, rep, color=qz.CAR, lw=1.4, ls="--", label="Repeated repair")
    a2.plot(r * 100, rel, color=qz.MAL, lw=1.4, label="Relining")
    k = int(np.argmin(np.abs(rep - rel)))
    a2.axvline(r[k] * 100, color=qz.GREY, lw=0.6, ls=":")
    a2.set_xlim(0, 8)
    a2.set_ylim(0, 7.5)
    qz.label(a2, "Discount rate, $r$ (%)", "Present cost (-)")
    a2.legend(loc="upper right")
    fig.tight_layout()
    print(f"  discounting: crossover at r = {r[k] * 100:.2f} per cent")
    cols.update({"r_pct": r * 100, "PV_repair": rep, "PV_reline": rel})
    _save(fig, "ch27", "discounting", cols)


# ===========================================================================
# Chapter 28: risk, contracts and project delivery
# ===========================================================================

def settlement_volume_loss():
    """The 20 mm settlement condition as a limit on volume loss, with the slope check."""
    vl = np.linspace(0, 3, 301)
    bores = [settlement.Bore(0.0, 20.0, 6.0, v, 0.5) for v in vl]
    smax = np.array([b.smax_mm for b in bores])
    i = 10.0
    slope = smax * math.exp(-0.5) / i              # 10^-3, from mm over m
    per = settlement.Bore(0.0, 20.0, 6.0, 1.0, 0.5).smax_mm
    vlim = 20.0 / per
    fig, (a1, a2) = qz.figure(W_FULL, 0.45 * W_FULL, ncols=2)
    for ax in (a1, a2):
        ax.axvspan(0, vlim, color=qz.MAL_TINT, lw=0)
        ax.axvline(vlim, color=qz.GREY, lw=0.6, ls=":")
    a1.plot(vl, smax, color=qz.MAL, lw=1.4)
    a1.axhline(20, color=qz.CAR, lw=1.1, ls="--")
    a1.plot([vlim], [20], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9)
    a1.text(0.1, 21, "20 mm", fontsize=FS, va="bottom")
    a1.set_xlim(0, 3)
    a1.set_ylim(0, 36)
    qz.label(a1, "Volume loss, $V_L$ (%)", r"Settlement, $S_{\max}$ (mm)")
    a2.plot(vl, slope, color=qz.MAL, lw=1.4)
    a2.axhline(2.0, color=qz.CAR, lw=1.1, ls="--")
    a2.plot([vlim], [20 * math.exp(-0.5) / i], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9)
    a2.text(0.1, 2.06, "1 in 500", fontsize=FS, va="bottom")
    a2.set_xlim(0, 3)
    a2.set_ylim(0, 2.6)
    qz.label(a2, "Volume loss, $V_L$ (%)", r"Slope, $\mathrm{d}S/\mathrm{d}y$ ($10^{-3}$)")
    fig.tight_layout()
    print(f"  S_max per 1 % {per:.2f} mm, V_L limit {vlim:.2f} %, slope there {20 * math.exp(-0.5) / i:.3f}e-3")
    _save(fig, "ch28", "settlement-volume-loss", {"V_L_pct": vl, "S_max_mm": smax, "slope_1e-3": slope})


def inflow_permeability():
    """Steady inflow per kilometre against permeability, with the 1 L/s per km condition."""
    k = np.logspace(-10, -5, 300)
    r0 = 3.0
    fig, ax = qz.figure(W_HALF, 0.85 * W_HALF)
    cols = {"k_w_m_s": k}
    for H, col, ls in ((10, qz.MAL_SHADES[3], ":"), (20, qz.MAL, "-"), (40, qz.MAL_SHADES[0], "--")):
        q = 2 * math.pi * k * H / math.log(2 * H / r0) * 1e6       # L/s per km
        ax.plot(k, q, color=col, ls=ls, lw=1.4, label=f"$H = {H}$ m")
        cols[f"q_L_s_km_H{H}"] = q
    ax.axhline(1.0, color=qz.CAR, lw=1.1, ls="--")
    kc = 1e-6 / (2 * math.pi * 20 / math.log(40 / r0))
    ax.plot([kc], [1.0], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9)
    ax.axvline(kc, color=qz.GREY, lw=0.6, ls=":")
    ax.text(7e-6, 0.7, "1 L/s per km", ha="right", va="top", fontsize=FS, color=qz.CAR)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-10, 1e-5)
    ax.set_ylim(1e-3, 1e3)
    qz.label(ax, "Permeability, $k_w$ (m/s)", "Inflow, $q$ (L/s per km)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    print(f"  inflow: k limit {kc:.3e} m/s")
    _save(fig, "ch28", "inflow-permeability", cols)


def baseline_mechanism():
    """A testable baseline: cumulative length of Class V ground against chainage (illustrative)."""
    x = np.linspace(0, 5, 501)
    zones = ((0.8, 0.06, 60), (1.9, 0.09, 90), (3.1, 0.08, 80), (4.2, 0.07, 70))
    L = np.zeros_like(x)
    for xc, w, length in zones:
        L += length * np.clip((x - (xc - w / 2)) / w, 0, 1)
    fig, ax = qz.figure(W_HALF, 0.85 * W_HALF)
    ax.axhspan(0, 200, color=qz.MAL_TINT, lw=0)
    ax.axhspan(200, 400, color=qz.CAR_TINT, lw=0)
    ax.axhline(200, color=qz.INK, lw=0.9, ls="--")
    ax.plot(x, L, color=qz.INK, lw=1.4)
    ax.text(0.15, 130, "Allowed for\nin the price", fontsize=FS, va="center", color=qz.MAL)
    ax.text(0.15, 275, "Changed\ncondition", fontsize=FS, va="center", color=qz.CAR)
    ax.text(4.9, 205, "Baseline", fontsize=FS, va="bottom", ha="right")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 340)
    qz.label(ax, "Chainage, $x$ (km)", "Class V encountered, $L_V$ (m)")
    fig.tight_layout()
    _save(fig, "ch28", "baseline-mechanism", {"x_km": x, "L_V_m": L})


def cost_distribution():
    """A skewed outturn distribution bounded below, with its P50 and P90 (illustrative)."""
    lo, med, sig = 0.8, 0.2, 0.6                  # C = lo + lognormal(median med, sigma sig)
    c = np.linspace(0.8, 2.0, 600)
    xx = np.clip(c - lo, 1e-9, None)
    pdf = norm.pdf(np.log(xx / med) / sig) / (xx * sig)
    cdf = norm.cdf(np.log(xx / med) / sig)
    p50 = lo + med
    p90 = lo + med * math.exp(norm.ppf(0.9) * sig)
    fig, (a1, a2) = qz.figure(W_FULL, 0.45 * W_FULL, ncols=2)
    a1.fill_between(c, 0, pdf, where=c >= p90, color=qz.CAR_TINT, lw=0)
    a1.plot(c, pdf, color=qz.MAL, lw=1.4)
    a2.plot(c, cdf, color=qz.MAL, lw=1.4)
    for ax in (a1, a2):
        for p, t in ((p50, "P50"), (p90, "P90")):
            ax.axvline(p, color=qz.GREY, lw=0.6, ls=":")
    ymax1 = float(pdf.max()) * 1.15
    for p, t in ((p50, "P50"), (p90, "P90")):
        a1.text(p + 0.02, ymax1 * 0.93, t, fontsize=FS, va="top")
    a2.plot([p50, p90], [0.5, 0.9], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9)
    _dim(a2, (p50, 0.2), (p90, 0.2), "", off=(0, 0))
    a2.text(p90 + 0.04, 0.2, "Risk allowance", ha="left", va="center", fontsize=FS)
    a1.set_xlim(0.8, 2.0)
    a1.set_ylim(0, ymax1)
    a2.set_xlim(0.8, 2.0)
    a2.set_ylim(0, 1.02)
    qz.label(a1, "Outturn cost, $C/C_{50}$ (-)", "Probability density, $f$ (-)")
    qz.label(a2, "Outturn cost, $C/C_{50}$ (-)", "Probability, $F$ (-)")
    fig.tight_layout()
    print(f"  cost: P50 {p50:.3f}, P90 {p90:.3f}")
    _save(fig, "ch28", "cost-distribution", {"C_over_C50": c, "pdf": pdf, "cdf": cdf})


ALL = {
    "shaft_stress": shaft_stress, "shaft_base_stability": shaft_base_stability, "portal_section": portal_section,
    "cavern_sequence": cavern_sequence, "zone_of_influence": zone_of_influence, "pillar_stress": pillar_stress,
    "groutability": groutability, "jet_grouting": jet_grouting, "compensation_grouting": compensation_grouting,
    "freeze_wall": freeze_wall, "presupport": presupport,
    "subsurface_trough": subsurface_trough, "deflection_ratio": deflection_ratio,
    "damage_interaction": damage_interaction, "void_migration": void_migration,
    "instrument_section": instrument_section, "trigger_levels": trigger_levels,
    "handover_record": handover_record, "deterioration_inspection": deterioration_inspection,
    "relining_clearance": relining_clearance, "discounting": discounting,
    "settlement_volume_loss": settlement_volume_loss, "inflow_permeability": inflow_permeability,
    "baseline_mechanism": baseline_mechanism, "cost_distribution": cost_distribution,
}

if __name__ == "__main__":
    qz.use()
    wanted = sys.argv[1:]
    for name, fn in ALL.items():
        if not wanted or any(w in name for w in wanted):
            fn()
