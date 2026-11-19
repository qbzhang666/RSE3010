"""Figures for Chapter 8 (in-situ stress), Chapter 9 (structural analysis) and
Appendix E (stereographic net constructions).

Every stereonet is drawn with the projection functions of the companion
package (``geo_platform.slopes``), so the plotted orientations are the ones the
application computes. Figures are sized at their printed width (117 mm text
block) in the house style of ``tools/qz_figure.py`` and written as PDF and
300 dpi PNG with the plotted data as CSV beside them.

    python tools/figs/f3.py              # all figures
    python tools/figs/f3.py kinematic    # figures whose name contains "kinematic"
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
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle  # noqa: E402

import qz_figure as qz  # noqa: E402
from geo_platform import slopes as sl  # noqa: E402

FIG = os.path.join(BOOK, "figures")
W_FULL = 117 / 25.4
W_HALF = 0.62 * W_FULL
EA, EQ = sl.EQUAL_ANGLE, sl.EQUAL_AREA
ROCK = "#EDEDED"          # light grey for rock in schematics
GAMMA = 0.027             # unit weight, MPa/m (27 kN/m3, lecture material)


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
    if isinstance(v, (int, float, np.floating, np.integer)):
        return f"{v:.6g}"
    return str(v)


# ---------------------------------------------------------------------------
# Stereonet helpers
# ---------------------------------------------------------------------------

def _tp(v):
    """Trend and plunge (deg) of an array of lower-hemisphere vectors (n, 3)."""
    v = np.atleast_2d(v)
    tr = np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360.0
    pl = np.degrees(np.arcsin(np.clip(-v[:, 2], -1, 1)))
    return tr, pl


def _net_frame(ax, lim=1.13, north=True, rot=0.0):
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=qz.INK, lw=0.8, zorder=5))
    ax.plot([-0.03, 0.03], [0, 0], color=qz.INK, lw=0.5, zorder=5)
    ax.plot([0, 0], [-0.03, 0.03], color=qz.INK, lw=0.5, zorder=5)
    if north:
        _index(ax, rot)


def _index(ax, rot=0.0, label="N"):
    """North index of the tracing, drawn at azimuth -rot."""
    a = math.radians(-rot)
    x0, y0 = math.sin(a), math.cos(a)
    ax.plot([x0, 1.07 * x0], [y0, 1.07 * y0], color=qz.INK, lw=0.9, zorder=6)
    ax.text(1.14 * x0, 1.14 * y0, label, ha="center", va="center", fontsize=8, zorder=6)


def _grid(ax, net=EA, step=10, color=None, lw=0.35, bold=None, bold_lw=0.6):
    """Meridional net: great circles striking N-S and small circles about the N-S axis."""
    color = color or qz.GREY_LIGHT
    t = np.linspace(0, math.pi, 241)
    for a in range(step, 180, step):
        ar = math.radians(a)
        v = np.c_[math.sin(ar) * np.cos(t), np.full_like(t, math.cos(ar)), -math.sin(ar) * np.sin(t)]
        x, y = sl.project(*_tp(v), net=net)
        w = bold_lw if bold and a % bold == 0 else lw
        ax.plot(x, y, color=color, lw=w, zorder=1)
    for d in range(step, 90, step):
        major = bool(bold) and d % bold == 0
        w = bold_lw if major else lw
        for dd in (90, 270):
            x, y = sl.great_circle(d, dd, net=net)
            if bold and not major:
                # fine meridians stop 10 degrees short of the poles, as on a printed net
                tt = np.linspace(0.0, math.pi, len(x))
                x = np.where(np.abs(np.cos(tt)) > math.cos(math.radians(10)), np.nan, x)
            ax.plot(x, y, color=color, lw=w, zorder=1)
    w = bold_lw if bold else lw
    ax.plot([-1, 1], [0, 0], color=color, lw=w, zorder=1)
    ax.plot([0, 0], [-1, 1], color=color, lw=w, zorder=1)


def _gc(ax, dip, dd, net=EA, rot=0.0, **kw):
    x, y = sl.great_circle(dip, dd - rot, net=net)
    ax.plot(x, y, zorder=4, **kw)
    return x, y


def _pt(ax, trend, plunge, net=EA, rot=0.0, **kw):
    x, y = sl.project(trend - rot, plunge, net=net)
    ax.plot([x], [y], zorder=7, **kw)
    return float(x), float(y)


def _grid_dirs(n=601, lim=1.0, net=EA):
    """Grid of net coordinates inside the primitive with their trend and plunge."""
    g = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(g, g)
    inside = np.hypot(X, Y) <= 1.0
    tr, pl = sl.unproject(X, Y, net=net)
    return X, Y, tr, pl, inside


def _zone(ax, X, Y, mask, color, hatch=None, alpha=1.0):
    Z = np.where(mask, 1.0, 0.0)
    if hatch:
        cs = ax.contourf(X, Y, Z, levels=[0.5, 1.5], colors="none", hatches=[hatch], zorder=2)
        for c in getattr(cs, "collections", [cs]):
            try:
                c.set_edgecolor(color)
                c.set_linewidth(0.0)
            except Exception:
                pass
        ax.contour(X, Y, Z, levels=[0.5], colors=[color], linewidths=0.5, zorder=2)
    else:
        ax.contourf(X, Y, Z, levels=[0.5, 1.5], colors=[color], alpha=alpha, zorder=2)
        ax.contour(X, Y, Z, levels=[0.5], colors=[qz.CAR], linewidths=0.5, zorder=2)


# ---------------------------------------------------------------------------
# Chapter 8: in-situ stress
# ---------------------------------------------------------------------------

def k_depth():
    """Hoek and Brown envelope and Sheorey's relation for k, and the stresses they imply."""
    z = np.linspace(20, 3000, 600)
    lo = 100 / z + 0.3
    hi = 1500 / z + 0.5
    fig, (a1, a2) = qz.figure(W_FULL, 0.56 * W_FULL, ncols=2, sharey=True)
    a1.fill_betweenx(z, lo, hi, color=qz.MAL_TINT, lw=0)
    a1.plot(lo, z, color=qz.MAL, lw=1.0)
    a1.plot(hi, z, color=qz.MAL, lw=1.0, label="Envelope")
    cols = {"z_m": z, "k_lower_HB": lo, "k_upper_HB": hi}
    for eh, ls, c in ((20, "-", qz.CAR_SHADES[3]), (50, "--", qz.CAR), (80, "-.", qz.CAR_SHADES[0])):
        k = 0.25 + 7 * eh * (0.001 + 1 / z)
        a1.plot(k, z, color=c, ls=ls, lw=1.2, label=f"$E_h = {eh}$ GPa")
        cols[f"k_sheorey_Eh{eh}"] = k
    a1.axvline(0.25 / 0.75, color=qz.GREY, lw=0.8, ls=(0, (1, 1.5)), label=r"$\nu/(1-\nu)$")
    a1.set_xlim(0, 3.5)
    a1.set_ylim(3000, 0)
    a1.legend(loc="lower right", handlelength=2.0)
    qz.label(a1, "Stress ratio, $k$ (-)", "Depth, $z$ (m)")

    sv = GAMMA * z
    a2.fill_betweenx(z, lo * sv, hi * sv, color=qz.MAL_TINT, lw=0)
    a2.plot(lo * sv, z, color=qz.MAL, lw=1.0)
    a2.plot(hi * sv, z, color=qz.MAL, lw=1.0, label=r"$\sigma_h$, envelope")
    sh50 = (0.25 + 7 * 50 * (0.001 + 1 / z)) * sv
    a2.plot(sh50, z, color=qz.CAR, ls="--", lw=1.2, label=r"$\sigma_h$, 50 GPa")
    a2.plot(sv, z, color=qz.INK, lw=1.4)
    a2.text(61.5, 2650, r"$\sigma_v$", ha="center", va="center", fontsize=8)
    a2.set_xlim(0, 100)
    a2.legend(loc="upper right", handlelength=1.0, handletextpad=0.5, borderaxespad=0.3)
    qz.label(a2, r"Stress, $\sigma$ (MPa)")
    cols.update({"sigma_v_MPa": sv, "sigma_h_lower_MPa": lo * sv, "sigma_h_upper_MPa": hi * sv,
                 "sigma_h_sheorey50_MPa": sh50})
    fig.tight_layout()
    _save(fig, "ch08", "k-depth-envelope", cols)


def erosion():
    """Horizontal stress retained after erosion of overburden (laterally restrained elastic unloading)."""
    nu, k0 = 0.25, 1.0
    z = np.linspace(10, 1500, 400)
    fig, ax = qz.figure(W_HALF, 0.95 * W_HALF)
    cols = {"z_m": z}
    styles = ((0, qz.GREY, "-"), (250, qz.MAL_SHADES[2], "--"), (500, qz.MAL, "-"), (1000, qz.CAR, "-."))
    for dz, c, ls in styles:
        k = k0 + (k0 - nu / (1 - nu)) * dz / z
        ax.plot(k, z, color=c, ls=ls, lw=1.2, label=rf"$\Delta z = {dz}$ m")
        cols[f"k_eroded_{dz}m"] = k
    ax.set_xlim(0, 6)
    ax.set_ylim(1500, 0)
    ax.legend(loc="lower right")
    qz.label(ax, "Stress ratio, $k$ (-)", "Present depth, $z$ (m)")
    fig.tight_layout()
    _save(fig, "ch08", "erosion-unloading", cols)


def overcoring():
    """Schematic of overcoring: section along the hole, and strain recovery as the bit passes the gauges."""
    fig = plt.figure(figsize=(W_FULL, 0.47 * W_FULL))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax.set_aspect("equal")
    ax.axis("off")
    R, r, bit = 75, 19, 250          # overcore and pilot radii (mm), bit position (mm)
    L0, L1 = -60, 560
    K = 9                            # kerf width (mm)
    ax.add_patch(Rectangle((L0, -130), L1 - L0, 260, fc=ROCK, ec="none"))
    ax.add_patch(Rectangle((L0, -R - K), bit - L0, 2 * (R + K), fc="white", ec="none"))     # access hole, kerf
    ax.add_patch(Rectangle((60, -R), bit - 60, 2 * R, fc=qz.MAL_TINT, ec="none"))           # core being cut
    ax.add_patch(Rectangle((60, -r), 470, 2 * r, fc="white", ec="none"))                    # pilot hole
    for yy in (R + K, -R - K):
        ax.plot([L0, bit], [yy, yy], color=qz.INK, lw=0.8)
    for sg in (1, -1):
        ax.plot([60, bit], [sg * R, sg * R], color=qz.INK, lw=0.6)
        ax.plot([60, 60], [sg * r, sg * R], color=qz.INK, lw=0.6)
        ax.plot([bit, bit], [sg * R, sg * (R + K)], color=qz.INK, lw=0.8)
        ax.add_patch(Rectangle((bit - 22, R if sg > 0 else -R - K), 22, K, fc=qz.GREY, ec="none"))
    for yy in (r, -r):
        ax.plot([60, 530], [yy, yy], color=qz.INK, lw=0.6)
    ax.plot([530, 530], [-r, r], color=qz.INK, lw=0.6)
    ax.add_patch(Rectangle((330, -r), 110, 2 * r, fc=qz.CAR_TINT, ec=qz.CAR, lw=0.8))      # cell
    ax.plot([385, 385], [-r - 12, r + 12], color=qz.CAR, lw=0.8, ls=":")
    ax.annotate("", xy=(bit + 45, R + K + 20), xytext=(bit - 35, R + K + 20),
                arrowprops=dict(arrowstyle="-|>", color=qz.INK, lw=0.8))
    ax.text(385, -R - 28, "Cell", ha="center", va="top", fontsize=8)
    ax.text(155, (R + r) / 2, "Core", ha="center", va="center", fontsize=8)
    ax.text(470, r + 22, "Pilot hole", ha="center", va="bottom", fontsize=8)
    ax.annotate("", xy=(385, -R - 26), xytext=(385, -r - 2),
                arrowprops=dict(arrowstyle="-", color=qz.INK, lw=0.5))
    # diameters: overcore D across the core, pilot hole d
    dim = dict(arrowstyle="<|-|>", color=qz.INK, lw=0.5, mutation_scale=5, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(100, R), xytext=(100, -R), arrowprops=dim, zorder=6)
    ax.text(110, -(R + r) / 2, "$D$", ha="left", va="center", fontsize=8, zorder=6)
    ax.annotate("", xy=(500, r), xytext=(500, -r), arrowprops=dim, zorder=6)
    ax.plot([500, 548], [-r - 4, -r - 30], color=qz.INK, lw=0.5)
    ax.text(552, -r - 32, "$d$", ha="left", va="center", fontsize=8)
    ax.set_xlim(L0, L1)
    ax.set_ylim(-135, 135)

    a2 = fig.add_subplot(gs[1])
    xi = np.linspace(-1.0, 2.0, 301)
    base = 0.5 * (1 + np.tanh((xi - 0.35) / 0.22)) - 0.22 * np.exp(-((xi + 0.05) / 0.2) ** 2)
    cols = {"x_over_D": xi}
    for amp, c, ls, name in ((1.0, qz.MAL, "-", "Circumferential"), (0.45, qz.MAL_SHADES[2], "--", "Inclined"),
                             (-0.25, qz.CAR, "-.", "Axial")):
        a2.plot(xi, amp * base, color=c, ls=ls, lw=1.2, label=name)
        a2.text(1.93, amp + (0.05 if amp > 0 else -0.06), name, ha="right",
                va="bottom" if amp > 0 else "top", fontsize=8)
        cols[f"strain_{name.lower()}"] = amp * base
    a2.axvline(0, color=qz.GREY, lw=0.6, ls=":")
    a2.axhline(0, color=qz.GREY_LIGHT, lw=0.6)
    a2.set_yticks([])
    a2.set_xlim(-1, 2)
    a2.set_ylim(-0.5, 1.3)
    qz.label(a2, "Bit position, $x/D$ (-)", r"Recovered strain, $\Delta\varepsilon$ (-)")
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.2, top=0.97)
    _save(fig, "ch08", "overcoring-schematic", cols)


def hydrofrac():
    """Schematic pressure record of a hydraulic fracturing test with three cycles."""
    sh, sH, T, p0 = 10.0, 16.0, 5.0, 1.0
    Pb = 3 * sh - sH + T - p0          # 18
    Pr = 3 * sh - sH - p0              # 13
    Ps = sh                            # 10
    Pp = 11.4                          # propagation pressure while pumping
    rate = 4.5                         # MPa/min while the interval is sealed

    def cycle(t0, peak, sharp):
        t = []
        p = []
        ramp = (peak - p0) / rate
        tt = np.linspace(0, ramp, 60)
        if sharp:
            pp = p0 + rate * tt
        else:
            tr_open = (Pr - p0) / rate
            pp = np.where(tt < tr_open, p0 + rate * tt,
                          Pr + (peak - Pr) * (1 - np.exp(-(tt - tr_open) / 0.12)))
            tt = np.linspace(0, tr_open + 0.45, 60)
            pp = np.where(tt < tr_open, p0 + rate * tt,
                          Pr + (peak - Pr) * (1 - np.exp(-(tt - tr_open) / 0.12)))
            ramp = tt[-1]
        t.append(t0 + tt)
        p.append(pp)
        # drop to propagation pressure while pumping continues
        tq = np.linspace(0, 1.4, 60)
        top = pp[-1]
        t.append(t0 + ramp + tq)
        p.append(Pp + (top - Pp) * np.exp(-tq / (0.07 if sharp else 0.25)))
        # shut-in: fast drop to the instantaneous shut-in pressure, then slow decay
        ts = np.linspace(0, 2.2, 80)
        t1 = t0 + ramp + 1.4
        t.append(t1 + ts)
        p.append(Ps - 0.9 * (1 - np.exp(-ts / 1.2)) + (Pp - Ps) * np.exp(-ts / 0.05))
        # bleed off
        tb = np.linspace(0, 1.2, 40)
        last = p[-1][-1]
        t.append(t1 + 2.2 + tb)
        p.append(p0 + (last - p0) * np.exp(-tb / 0.18))
        return np.concatenate(t), np.concatenate(p), t1 + 3.4

    fig, ax = qz.figure(W_FULL, 0.5 * W_FULL)
    T_all, P_all = [], []
    t0 = 0.4
    for i, (peak, sharp) in enumerate(((Pb, True), (Pr + 0.35, False), (Pr + 0.35, False))):
        tc, pc, tend = cycle(t0, peak, sharp)
        T_all.append(tc)
        P_all.append(pc)
        t0 = tend + 0.5
    t = np.concatenate([[0.0]] + T_all + [[t0]])
    p = np.concatenate([[p0]] + P_all + [[p0]])
    ax.plot(t, p, color=qz.MAL, lw=1.3)
    for val, lab in ((Pb, "$P_b$"), (Pr, "$P_r$"), (Ps, "$P_s$"), (p0, "$p_0$")):
        ax.axhline(val, color=qz.GREY, lw=0.6, ls=":")
        ax.text(t[-1] + 0.5, val + 0.25, lab, va="bottom", ha="left", fontsize=9)
    # extrapolate the first-cycle slope through the second cycle to show the reopening point
    t2 = T_all[1][0]
    tro = t2 + (Pr - p0) / rate
    ax.plot([tro, tro + 0.6], [Pr, Pr + rate * 0.6], color=qz.CAR, lw=0.8, ls="--")
    ax.plot([tro], [Pr], "o", ms=4, mfc="white", mec=qz.CAR, mew=1.0, zorder=5)
    ax.set_xlim(0, t[-1] + 1.6)
    ax.set_ylim(0, 20)
    qz.label(ax, "Time, $t$ (min)", "Interval pressure, $P$ (MPa)")
    fig.tight_layout()
    _save(fig, "ch08", "hydrofrac-record", {"t_min": t, "P_MPa": p,
                                             "Pb_Pr_Ps_p0": np.array([Pb, Pr, Ps, p0])})


def breakout():
    """Kirsch stress at a borehole wall and the breakout width it implies."""
    sh, C = 20.0, 80.0
    cases = ((35.0, qz.MAL, "-"), (45.0, qz.CAR, "--"))
    fig = plt.figure(figsize=(W_FULL, 0.44 * W_FULL))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.25], wspace=0.25)
    a0 = fig.add_subplot(gs[0])
    a0.set_aspect("equal")
    a0.axis("off")
    # section: sigma_H horizontal on the page, breakouts at +-90 deg from it
    sH = 45.0
    hw = 0.5 * math.degrees(math.acos((sH + sh - C) / (2 * (sH - sh))))
    hw = 90 - hw                     # half-width about the sigma_h direction
    a0.add_patch(Rectangle((-2.6, -2.6), 5.2, 5.2, fc=ROCK, ec="none"))
    th = np.linspace(0, 2 * math.pi, 361)
    a0.add_patch(Circle((0, 0), 1.0, fc="white", ec=qz.INK, lw=0.9, zorder=3))
    for c0 in (90, 270):
        a = np.radians(np.linspace(c0 - hw, c0 + hw, 80))
        rr = 1 + 0.45 * np.sqrt(np.clip(1 - ((np.degrees(a) - c0) / hw) ** 2, 0, 1)) ** 0.6
        xs = np.r_[np.cos(a), (rr * np.cos(a))[::-1]]
        ys = np.r_[np.sin(a), (rr * np.sin(a))[::-1]]
        a0.add_patch(Polygon(np.c_[xs, ys], closed=True, fc=qz.CAR_TINT, ec=qz.CAR, lw=0.8, zorder=4))
        a0.add_patch(Polygon(np.c_[rr * np.cos(a), rr * np.sin(a)], closed=False, fill=False, ec=qz.CAR,
                             lw=0.8, zorder=4))
    for sgn in (1, -1):
        a0.plot([sgn * 1.0, sgn * 1.35], [0, 0], color=qz.MAL, lw=1.4, zorder=4)
    for sgn in (1, -1):
        a0.annotate("", xy=(sgn * 1.75, 0), xytext=(sgn * 2.55, 0),
                    arrowprops=dict(arrowstyle="-|>", color=qz.INK, lw=1.6, mutation_scale=10))
        a0.annotate("", xy=(0, sgn * 1.85), xytext=(0, sgn * 2.45),
                    arrowprops=dict(arrowstyle="-|>", color=qz.INK, lw=0.9, mutation_scale=8))
    a0.text(2.15, 0.22, r"$\sigma_H$", ha="center", va="bottom", fontsize=10)
    a0.text(0.2, 2.2, r"$\sigma_h$", ha="left", va="center", fontsize=10)
    a0.text(1.08, -0.2, "Tensile\nfracture", ha="left", va="top", fontsize=8, color=qz.MAL_SHADES[0],
            linespacing=1.0, zorder=6)
    a0.text(0, 1.62, "Breakout", ha="center", va="bottom", fontsize=8, color=qz.CAR_DARK, zorder=6)
    a0.set_xlim(-2.6, 2.6)
    a0.set_ylim(-2.6, 2.6)

    ax = fig.add_subplot(gs[1])
    theta = np.linspace(0, 180, 361)
    cols = {"theta_deg": theta}
    for sH_, c, ls in cases:
        s = sH_ + sh - 2 * (sH_ - sh) * np.cos(np.radians(2 * theta))
        ax.plot(theta, s, color=c, ls=ls, lw=1.3, label=rf"$\sigma_H = {sH_:g}$ MPa")
        ax.fill_between(theta, C, s, where=s > C, color=qz.CAR_TINT if c == qz.CAR else qz.MAL_TINT,
                        lw=0, zorder=1 if c == qz.CAR else 2)
        cols[f"sigma_theta_sH{sH_:g}_MPa"] = s
        hw_ = 90 - 0.5 * math.degrees(math.acos((sH_ + sh - C) / (2 * (sH_ - sh))))
        print(f"  breakout width at sigma_H {sH_:g}: {2 * hw_:.1f} deg")
    ax.axhline(C, color=qz.INK, lw=0.7, ls=":")
    ax.text(3, C + 2, r"$C_0$", va="bottom", ha="left", fontsize=9)
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_ylim(0, 130)
    ax.legend(loc="lower center")
    qz.label(ax, r"Angle from $\sigma_H$, $\theta$ ($^\circ$)", r"Wall stress, $\sigma_\theta$ (MPa)")
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.2, top=0.97)
    _save(fig, "ch08", "borehole-breakout", cols)


def stress_bracket():
    """Kirsch boundary stress at crown and sidewall across the k range bracketed at 1000 m."""
    k = np.linspace(0, 3, 301)
    crown = 3 * k - 1
    wall = 3 - k
    klo, khi = 100 / 1000 + 0.3, 1500 / 1000 + 0.5
    fig, ax = qz.figure(W_HALF, 0.92 * W_HALF)
    ax.axvspan(klo, khi, color=qz.MAL_TINT, lw=0)
    ax.axhspan(-2, 0, color=qz.CAR_TINT, lw=0)
    ax.plot(k, crown, color=qz.MAL, lw=1.3, label="Crown")
    ax.plot(k, wall, color=qz.CAR, lw=1.3, ls="--", label="Sidewall")
    for kk in (klo, khi):
        ax.plot([kk, kk], [3 * kk - 1, 3 - kk], "o", ms=3.5, mfc="white", mec=qz.INK, mew=0.8, zorder=5)
    ax.axhline(0, color=qz.GREY, lw=0.6)
    ax.set_xlim(0, 3)
    ax.set_ylim(-1.5, 8)
    ax.legend(loc="upper left")
    qz.label(ax, "Stress ratio, $k$ (-)", r"Boundary stress, $\sigma_\theta/\sigma_v$ (-)")
    fig.tight_layout()
    _save(fig, "ch08", "stress-bracket-kirsch", {"k": k, "crown": crown, "sidewall": wall,
                                                 "k_bracket_1000m": np.array([klo, khi])})


# ---------------------------------------------------------------------------
# Chapter 9: structural analysis
# ---------------------------------------------------------------------------

def nets_compared():
    """The same uniform scatter of directions on the equal-angle and equal-area nets."""
    rng = np.random.default_rng(3010)
    n = 600
    u = rng.uniform(0, 1, n)
    tr = rng.uniform(0, 360, n)
    pl = np.degrees(np.arcsin(u))            # uniform on the lower hemisphere
    fig, axes = qz.figure(W_FULL, 0.52 * W_FULL, ncols=2)
    cols = {"trend_deg": tr, "plunge_deg": pl}
    for ax, net, name in zip(axes, (EA, EQ), ("Equal angle", "Equal area")):
        _net_frame(ax)
        _grid(ax, net=net, step=10)
        x, y = sl.project(tr, pl, net=net)
        ax.plot(x, y, "o", ms=1.6, mfc=qz.MAL, mec="none", zorder=4)
        # circle of radius 0.5 of the net to compare counts
        ax.add_patch(Circle((0, 0), 0.5, fill=False, ec=qz.CAR, lw=0.9, ls="--", zorder=6))
        inner = np.hypot(x, y) <= 0.5
        ax.set_title(name, fontsize=10, pad=12)
        cols[f"x_{net}"] = x
        cols[f"y_{net}"] = y
        print(f"  {name}: {inner.sum()} of {n} points inside half radius ({100 * inner.mean():.0f} %)")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.08)
    _save(fig, "ch09", "net-density-bias", cols)


def _fisher(rng, mean, kappa, n):
    """Fisher-distributed unit vectors about ``mean``."""
    u = rng.uniform(0, 1, n)
    w = 1 + np.log(u + (1 - u) * np.exp(-2 * kappa)) / kappa
    phi = rng.uniform(0, 2 * math.pi, n)
    s = np.sqrt(np.clip(1 - w ** 2, 0, 1))
    m = mean / np.linalg.norm(mean)
    a = np.cross(m, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.array([1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(m, a)
    return np.outer(w, m) + np.outer(s * np.cos(phi), a) + np.outer(s * np.sin(phi), b)


def pole_contours():
    """Synthetic poles from three Fisher sets, contoured with a 1 per cent counting circle."""
    rng = np.random.default_rng(11)
    sets = (((40, 225), 30, 70), ((65, 225), 30, 60), ((80, 45), 50, 50))
    vs = []
    for (dip, dd), kap, n in sets:
        tr, pl = sl.pole(dip, dd)
        vs.append(_fisher(rng, sl.line_vector(tr, pl), kap, n))
    rnd = rng.normal(size=(20, 3))
    vs.append(rnd / np.linalg.norm(rnd, axis=1, keepdims=True))
    v = np.vstack(vs)
    v[v[:, 2] > 0] *= -1
    tr, pl = _tp(v)
    X, Y, gtr, gpl, inside = _grid_dirs(241, 1.0, EQ)
    G = np.stack([sl.line_vector(a, b) for a, b in zip(gtr.ravel(), gpl.ravel())])
    cosa = math.cos(math.radians(8.11))     # cone of 1 per cent of the hemisphere
    count = (np.abs(G @ v.T) >= cosa).sum(axis=1).reshape(X.shape)
    from scipy.ndimage import gaussian_filter
    pct = gaussian_filter(100 * count / len(v), 2.5, mode="nearest")   # light smoothing of the step edges
    pct = np.where(inside, pct, np.nan)
    fig, ax = qz.figure(W_HALF, W_HALF)
    _net_frame(ax)
    levels = [2, 4, 6, 8, 12]
    cs = ax.contourf(X, Y, pct, levels=levels + [100], cmap=qz.SEQ_MAL, vmin=-2, vmax=13, zorder=1)
    # key to the contour bands, in the lower right corner outside the primitive
    kx, ky, kh = 0.80, -1.11, 0.075
    names = [f"{a}-{b}" for a, b in zip(levels[:-1], levels[1:])] + [f"$>${levels[-1]}"]
    shown = [j for j, lv in enumerate(levels) if lv < np.nanmax(pct)]      # bands that occur
    for j, (fc, nm) in enumerate(zip(cs.cmap(cs.norm(cs.cvalues))[shown], [names[k] for k in shown])):
        ax.add_patch(Rectangle((kx, ky + j * kh), 0.085, kh, fc=fc, ec=qz.MAL_SHADES[0], lw=0.4, zorder=3))
        ax.text(kx + 0.115, ky + (j + 0.5) * kh, nm, ha="left", va="center", fontsize=6.5)
    ax.text(kx + 0.115, ky + (len(shown) + 0.55) * kh, "%", ha="left", va="center", fontsize=6.5)
    ax.contour(X, Y, pct, levels=levels, colors=[qz.MAL_SHADES[0]], linewidths=0.4, zorder=2)
    x, y = sl.project(tr, pl, net=EQ)
    ax.plot(x, y, "o", ms=1.8, mfc=qz.INK, mec="none", zorder=4)
    for i, ((dip, dd), _, _) in enumerate(sets, 1):
        ptr, ppl = sl.pole(dip, dd)
        px, py = sl.project(ptr, ppl, net=EQ)
        ax.plot([px], [py], "+", ms=7, mew=1.2, color=qz.CAR, zorder=6)
        off = {1: (-0.13, -0.08), 2: (0.14, 0.04), 3: (0.12, 0.12)}[i]
        ax.text(px + off[0], py + off[1], f"J{i}", fontsize=8, color=qz.CAR_DARK, zorder=6,
                ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    print(f"  {len(v)} poles, max density {np.nanmax(pct):.1f} %")
    _save(fig, "ch09", "pole-contouring", {"pole_trend_deg": tr, "pole_plunge_deg": pl})


def terzaghi():
    """Terzaghi weight against the angle between plane and sampling line, with cut-offs."""
    d = np.linspace(0.2, 90, 450)
    w = 1 / np.sin(np.radians(d))
    fig, ax = qz.figure(W_HALF, 0.8 * W_HALF)
    ax.axvspan(0, 15, color=qz.CAR_TINT, lw=0)
    ax.plot(d, w, color=qz.GREY, lw=0.9, ls=":", label="Uncapped")
    for cut, ls, c in ((20, "-", qz.MAL), (15, "--", qz.CAR)):
        wc = np.minimum(w, 1 / math.sin(math.radians(cut)))
        ax.plot(d, wc, color=c, ls=ls, lw=1.3, label=rf"Cut-off ${cut}^\circ$")
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_ylim(0, 8)
    ax.legend(loc="upper right")
    qz.label(ax, r"Angle to sampling line, $\delta$ ($^\circ$)", "Weight, $w$ (-)")
    fig.tight_layout()
    _save(fig, "ch09", "terzaghi-weight", {"delta_deg": d, "w": w,
                                           "w_cap20": np.minimum(w, 1 / math.sin(math.radians(20))),
                                           "w_cap15": np.minimum(w, 1 / math.sin(math.radians(15)))})


def removability():
    """Schematic sections of a removable, a tapered and an infinite block above a roof."""
    fig, axes = qz.figure(W_FULL, 0.36 * W_FULL, ncols=3)
    shapes = {
        "Removable": ([(-1.6, 0), (1.6, 0), (0.2, 1.9)], [((-1.6, 0), (0.2, 1.9)), ((1.6, 0), (0.2, 1.9))]),
        "Tapered": ([(-0.6, 0), (0.6, 0), (1.35, 1.5), (-1.25, 1.5)],
                    [((-0.6, 0), (-1.25, 1.5)), ((0.6, 0), (1.35, 1.5)), ((-1.25, 1.5), (1.35, 1.5))]),
        "Infinite": ([(-0.9, 0), (0.9, 0), (1.35, 2.4), (-0.45, 2.4)],
                     [((-0.9, 0), (-0.45, 2.4)), ((0.9, 0), (1.35, 2.4))]),
    }
    fills = {"Removable": (qz.CAR_TINT, qz.CAR), "Tapered": (qz.MAL_TINT, qz.MAL), "Infinite": ("white", qz.GREY)}
    cols = {"case": [], "x": [], "y": []}
    for ax, (name, (poly, joints)) in zip(axes, shapes.items()):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Rectangle((-2.2, 0), 4.4, 2.4, fc=ROCK, ec="none"))
        fc, ec = fills[name]
        if name == "Infinite":
            plt.rcParams["hatch.color"] = qz.GREY_MID
            plt.rcParams["hatch.linewidth"] = 0.4
            ax.add_patch(Polygon(poly, closed=True, fc=ROCK, ec=qz.GREY_MID, lw=0, hatch="////", zorder=2))
        else:
            ax.add_patch(Polygon(poly, closed=True, fc=fc, ec="none", zorder=2))
        for (x0, y0), (x1, y1) in joints:
            dx, dy = x1 - x0, y1 - y0
            ext = 0.5 if name != "Infinite" else 0.0
            L = math.hypot(dx, dy)
            ax.plot([x0 - 0.0 * dx, x1 + ext * dx / L], [y0, y1 + ext * dy / L], color=ec, lw=1.0, zorder=3)
        ax.plot([-2.2, 2.2], [0, 0], color=qz.INK, lw=1.2, zorder=4)
        if name == "Removable":
            ax.annotate("", xy=(0.0, -0.75), xytext=(0.0, 0.55),
                        arrowprops=dict(arrowstyle="-|>", color=qz.CAR, lw=1.1, mutation_scale=9), zorder=5)
        else:
            ax.annotate("", xy=(0.0, -0.75), xytext=(0.0, 0.55),
                        arrowprops=dict(arrowstyle="-|>", color=qz.GREY, lw=0.9, ls="--", mutation_scale=8),
                        zorder=5)
            ax.plot([-0.18, 0.18], [-0.38, -0.12], color=qz.INK, lw=1.0, zorder=6)
            ax.plot([-0.18, 0.18], [-0.12, -0.38], color=qz.INK, lw=1.0, zorder=6)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-0.9, 2.5)
        ax.set_title(name, fontsize=10, pad=12)
        for x, y in poly:
            cols["case"].append(name)
            cols["x"].append(x)
            cols["y"].append(y)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9, wspace=0.08)
    _save(fig, "ch09", "block-removability", {k: np.array(v, dtype=object) for k, v in cols.items()})


CROWN_SETS = ((65, 70), (60, 280), (40, 180))


def _crown_unit():
    """Unit crown wedge on a horizontal roof: base triangle vertices and apex height for d_i = 1."""
    n = np.array([sl.plane_normal(d, a) for d, a in CROWN_SETS])
    h = n[:, :2]
    verts = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        verts.append(np.linalg.solve(h[[i, j]], np.ones(2)))
    verts = np.array(verts)
    apex = np.linalg.solve(n, np.ones(3))
    e1, e2 = verts[1] - verts[0], verts[2] - verts[0]
    area = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
    return verts, apex, area


def crown_wedge():
    """Largest crown wedge of three sets in a flat-roofed 10 m span, against tunnel axis trend."""
    B = 10.0
    verts, apex, area = _crown_unit()
    assert apex[2] > 0
    trends = np.arange(0, 180.5, 1.0)
    vol, height = [], []
    for a in trends:
        w = np.array([math.cos(math.radians(a)), -math.sin(math.radians(a))])
        p = verts @ w
        s = B / (p.max() - p.min())
        vol.append(area * apex[2] / 3 * s ** 3)
        height.append(apex[2] * s)
    vol = np.array(vol)
    height = np.array(height)
    imax, imin = int(vol.argmax()), int(vol.argmin())
    print(f"  crown wedge: max {vol[imax]:.1f} m3 at {trends[imax]:.0f} deg (apex {height[imax]:.2f} m), "
          f"min {vol[imin]:.1f} m3 at {trends[imin]:.0f} deg (apex {height[imin]:.2f} m)")

    fig = plt.figure(figsize=(W_FULL, 0.46 * W_FULL))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], wspace=0.32, hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(trends, vol, color=qz.MAL, lw=1.3)
    ax.plot([trends[imax]], [vol[imax]], "o", ms=4.5, mfc=qz.CAR, mec=qz.CAR, zorder=5)
    ax.plot([trends[imin]], [vol[imin]], "o", ms=4.5, mfc="white", mec=qz.MAL, mew=1.0, zorder=5)
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_ylim(0, vol.max() * 1.15)
    qz.label(ax, r"Tunnel axis trend, $\theta$ ($^\circ$)", "Wedge volume, $V$ (m$^3$)")
    for k, (i, c, name) in enumerate(((imax, qz.CAR, "max"), (imin, qz.MAL, "min"))):
        a2 = fig.add_subplot(gs[k, 1])
        a = math.radians(trends[i])
        u = np.array([math.sin(a), math.cos(a)])
        w = np.array([math.cos(a), -math.sin(a)])
        p = verts @ w
        s = B / (p.max() - p.min())
        V = verts * s
        loc = np.c_[V @ u, V @ w - (p.max() + p.min()) / 2 * s]
        loc[:, 0] -= loc[:, 0].mean()
        a2.add_patch(Rectangle((-14, -B / 2), 28, B, fc=ROCK, ec="none"))
        a2.plot([-14, 14], [B / 2, B / 2], color=qz.INK, lw=0.9)
        a2.plot([-14, 14], [-B / 2, -B / 2], color=qz.INK, lw=0.9)
        a2.add_patch(Polygon(loc, closed=True, fc=qz.CAR_TINT if name == "max" else qz.MAL_TINT,
                             ec=c, lw=1.0))
        ap = np.array([apex[:2] @ u * s, apex[:2] @ w * s - (p.max() + p.min()) / 2 * s])
        ap[0] -= (V @ u).mean()
        for vtx in loc:
            a2.plot([vtx[0], ap[0]], [vtx[1], ap[1]], color=c, lw=0.6, ls="--")
        a2.set_aspect("equal")
        a2.set_xlim(-14, 14)
        a2.set_ylim(-6.5, 6.5)
        a2.set_xticks([-10, 0, 10])
        a2.set_yticks([-5, 0, 5])
        a2.tick_params(labelsize=8)
        a2.text(13.2, 4.4, rf"${trends[i]:.0f}^\circ$", ha="right", va="top", fontsize=8)
        if k == 1:
            a2.set_xlabel("Along axis (m)")
        a2.set_ylabel("Across (m)", fontsize=8)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.17, top=0.97)
    _save(fig, "ch09", "crown-wedge-orientation", {"axis_trend_deg": trends, "volume_m3": vol,
                                                   "apex_height_m": height,
                                                   "weight_kN": vol * 27.0})


def kinematic():
    """Week 11 cases: wedge sliding on a 70/135 face, planar sliding and toppling on a 55/220 face."""
    phi = 35.0
    fig, (a1, a2) = qz.figure(W_FULL, 0.55 * W_FULL, ncols=2)
    cols = {}
    # wedge case on great circles
    fdip, fdd = 70.0, 135.0
    _net_frame(a1)
    X, Y, tr, pl, inside = _grid_dirs(701)
    app = np.degrees(np.arctan(np.tan(math.radians(fdip)) * np.cos(np.radians(tr - fdd))))
    off = np.abs((tr - fdd + 180) % 360 - 180)
    crit = inside & (pl > phi) & (off < 90) & (pl < app)
    _zone(a1, X, Y, crit, qz.CAR_TINT)
    x, y = sl.plunge_circle(phi)
    a1.plot(x, y, color=qz.GREY, lw=0.8, ls="--", zorder=3)
    _gc(a1, fdip, fdd, color=qz.INK, lw=1.4)
    _gc(a1, 60, 90, color=qz.MAL, lw=1.1)
    _gc(a1, 60, 210, color=qz.MAL, lw=1.1, ls="-.")
    itr, ipl = sl.intersection_line(60, 90, 60, 210)
    ix, iy = _pt(a1, itr, ipl, marker="o", ms=5, mfc=qz.CAR, mec=qz.CAR)
    a1.text(ix + 0.07, iy + 0.02, "I", fontsize=9, color=qz.CAR_DARK, ha="left", va="bottom", zorder=8)
    a1.set_title("Wedge sliding", fontsize=10, pad=12)
    res = sl.wedge_kinematics((60, 90), (60, 210), fdip, fdd, phi)
    print(f"  wedge: I {itr:.0f}/{ipl:.1f}, possible {res['Possible']}, {res['Sliding on']}")
    cols.update({"I_trend": [itr], "I_plunge": [ipl]})
    # label the joint circles near the primitive
    a1.text(0.12, 0.93, "Ja", fontsize=8, color=qz.MAL_SHADES[0], ha="left", va="center")
    a1.text(-0.86, 0.32, "Jb", fontsize=8, color=qz.MAL_SHADES[0], ha="center", va="center")
    a1.text(0.80, 0.80, "Face", fontsize=8, color=qz.INK, ha="left", va="bottom")

    # planar sliding on poles
    fdip, fdd = 55.0, 220.0
    _net_frame(a2)
    ptr_face = (fdd + 180) % 360
    offp = np.abs((tr - ptr_face + 180) % 360 - 180)
    planar = inside & (pl > 90 - fdip) & (pl < 90 - phi) & (offp <= 20)
    _zone(a2, X, Y, planar, qz.CAR_TINT)
    x, y = sl.plunge_circle(90 - phi)
    a2.plot(x, y, color=qz.GREY, lw=0.8, ls="--", zorder=3)
    _gc(a2, fdip, fdd, color=qz.INK, lw=1.4)
    marks = (("J1", 40, 225, "o"), ("J2", 65, 225, "s"))
    for lab, d, a, m in marks:
        ptr, ppl = sl.pole(d, a)
        px, py = _pt(a2, ptr, ppl, marker=m, ms=5, mfc=qz.MAL, mec=qz.MAL_SHADES[0], mew=0.6)
        pk = sl.planar_kinematics(d, a, fdip, fdd, phi)["Possible"]
        tk = sl.toppling_kinematics(d, a, fdip, fdd, phi)["Possible"]
        print(f"  {lab} {d}/{a}: pole {ptr:.0f}/{ppl:.0f}, planar {pk}, toppling {tk}")
        a2.text(px + (0.27 if lab == "J1" else 0.09), py, lab, fontsize=8, ha="left", va="center", color=qz.MAL_SHADES[0], zorder=8)
        cols[f"{lab}_pole"] = [f"{ptr:.0f}/{ppl:.0f}"]
    a2.text(-0.82, 0.72, "Face", fontsize=8, ha="right", va="bottom")
    a2.set_title("Planar sliding", fontsize=10, pad=12)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.08)
    _save(fig, "ch09", "portal-kinematics", cols)


# ---------------------------------------------------------------------------
# Appendix E: constructions on the net
# ---------------------------------------------------------------------------

def _construction_axes(ncols=2):
    fig, axes = qz.figure(W_FULL, 0.53 * W_FULL, ncols=ncols)
    for ax in axes:
        _grid(ax, net=EA, step=10)
    return fig, axes


def construct_plane_pole():
    dip, dd = 60.0, 130.0
    rot = dd - 90.0
    fig, (a1, a2) = _construction_axes()
    # rotated tracing
    _net_frame(a1, rot=rot)
    _gc(a1, dip, dd, rot=rot, color=qz.MAL, lw=1.3)
    rg = math.tan(math.radians((90 - dip) / 2))
    rp = math.tan(math.radians(dip / 2))
    a1.plot([rg, 1.0], [0, 0], color=qz.CAR, lw=2.2, solid_capstyle="butt", zorder=6)
    a1.plot([-rp, rg], [0, 0], color=qz.CAR, lw=1.0, ls=(0, (3, 2)), zorder=6)
    a1.plot([1.0], [0], "v", ms=6, color=qz.INK, zorder=8)
    _pt(a1, (dd + 180) % 360, 90 - dip, rot=rot, marker="o", ms=5, mfc="white", mec=qz.CAR, mew=1.2)
    a1.text((1 + rg) / 2, 0.06, r"$\psi$", ha="center", va="bottom", fontsize=9, color=qz.CAR_DARK, zorder=8)
    a1.text((rg - rp) / 2, -0.07, r"$90^\circ$", ha="center", va="top", fontsize=8, color=qz.CAR_DARK, zorder=8)
    a1.text(-rp, 0.07, "P", ha="center", va="bottom", fontsize=9, zorder=8)
    a1.text(1.08, 0.0, r"$\alpha$", ha="left", va="center", fontsize=9)
    a1.set_title("Rotated to the east", fontsize=10, pad=12)
    # returned to north
    _net_frame(a2)
    _gc(a2, dip, dd, color=qz.MAL, lw=1.3)
    px, py = _pt(a2, (dd + 180) % 360, 90 - dip, marker="o", ms=5, mfc="white", mec=qz.CAR, mew=1.2)
    a2.text(px - 0.06, py + 0.05, "P", ha="right", va="bottom", fontsize=9, zorder=8)
    ax_, ay_ = math.sin(math.radians(dd)), math.cos(math.radians(dd))
    a2.plot([ax_], [ay_], marker=(3, 0, -dd + 180 + 0), ms=6, color=qz.INK, zorder=8)
    a2.text(1.1 * ax_, 1.1 * ay_, r"$\alpha$", ha="left", va="top", fontsize=9)
    a2.set_title("Returned to north", fontsize=10, pad=12)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.08)
    _save(fig, "appE", "construct-plane-pole", {"dip": [dip], "dip_dir": [dd],
                                                "pole_trend": [(dd + 180) % 360], "pole_plunge": [90 - dip],
                                                "rotation_deg": [rot]})


def construct_intersection():
    p1, p2 = (60.0, 90.0), (60.0, 210.0)
    itr, ipl = sl.intersection_line(*p1, *p2)
    rot = itr - 90.0
    fig, (a1, a2) = _construction_axes()
    _net_frame(a1)
    _gc(a1, *p1, color=qz.MAL, lw=1.3)
    _gc(a1, *p2, color=qz.MAL, lw=1.3, ls="-.")
    ix, iy = _pt(a1, itr, ipl, marker="o", ms=5, mfc=qz.CAR, mec=qz.CAR)
    a1.text(ix + 0.07, iy + 0.03, "I", ha="left", va="bottom", fontsize=9, color=qz.CAR_DARK, zorder=8)
    a1.set_title("Both planes plotted", fontsize=10, pad=12)
    _net_frame(a2, rot=rot)
    _gc(a2, *p1, rot=rot, color=qz.MAL, lw=1.3)
    _gc(a2, *p2, rot=rot, color=qz.MAL, lw=1.3, ls="-.")
    ri = math.tan(math.radians((90 - ipl) / 2))
    a2.plot([ri, 1.0], [0, 0], color=qz.CAR, lw=2.2, solid_capstyle="butt", zorder=6)
    _pt(a2, itr, ipl, rot=rot, marker="o", ms=5, mfc=qz.CAR, mec=qz.CAR)
    a2.plot([1.0], [0], "v", ms=6, color=qz.INK, zorder=8)
    a2.text((1 + ri) / 2, 0.06, rf"$\beta = {ipl:.0f}^\circ$", ha="center", va="bottom", fontsize=8,
            color=qz.CAR_DARK, zorder=8, bbox=dict(fc="white", ec="none", pad=0.4))
    a2.text(ri - 0.16, 0.0, "I", ha="center", va="center", fontsize=9, color=qz.CAR_DARK, zorder=8)
    a2.set_title("Rotated to read plunge", fontsize=10, pad=12)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.08)
    print(f"  intersection {itr:.1f}/{ipl:.1f}, tracing rotated {rot:.0f} deg")
    _save(fig, "appE", "construct-intersection", {"plane1": ["60/090"], "plane2": ["60/210"],
                                                  "I_trend": [itr], "I_plunge": [ipl]})


def _rotate_path(v, angle_deg, n=120):
    """Path of a vector rotated about the horizontal N-S axis (y) up to angle_deg."""
    out = []
    for a in np.linspace(0, math.radians(angle_deg), n):
        R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        out.append(R @ v)
    return np.array(out)


def construct_rotation():
    fig, ax = qz.figure(W_HALF, W_HALF)
    _grid(ax, net=EA, step=10)
    _net_frame(ax)
    ang = -40.0
    cols = {}
    for lab, (tr, pl) in (("P1", (310.0, 30.0)), ("P2", (80.0, 20.0))):
        path = _rotate_path(sl.line_vector(tr, pl), ang)
        up = path[:, 2] > 0
        segs = []
        cur = [path[0]]
        for k in range(1, len(path)):
            if up[k] != up[k - 1]:
                segs.append((np.array(cur), up[k - 1]))
                cur = []
            cur.append(path[k])
        segs.append((np.array(cur), up[-1]))
        for seg, flip in segs:
            s = -seg if flip else seg
            x, y = sl.project(*_tp(s))
            ax.plot(x, y, color=qz.CAR, lw=1.4, zorder=5)
            if len(x) > 6:
                m = len(x) // 2
                ax.annotate("", xy=(x[m + 2], y[m + 2]), xytext=(x[m - 2], y[m - 2]), zorder=6,
                            arrowprops=dict(arrowstyle="-|>", color=qz.CAR, lw=0, mutation_scale=11,
                                            shrinkA=0, shrinkB=0))
        end = path[-1] if path[-1][2] <= 0 else -path[-1]
        etr, epl = _tp(end)
        x0, y0 = _pt(ax, tr, pl, marker="o", ms=5, mfc="white", mec=qz.CAR, mew=1.1)
        x1, y1 = _pt(ax, etr[0], epl[0], marker="o", ms=5, mfc=qz.CAR, mec=qz.CAR)
        ax.text(x0, y0 + 0.07, lab, ha="center", va="bottom", fontsize=8, zorder=8)
        ax.text(x1, y1 - 0.07, lab + "'", ha="center", va="top", fontsize=8, zorder=8)
        cols[f"{lab}_start"] = [f"{tr:.0f}/{pl:.0f}"]
        cols[f"{lab}_end"] = [f"{etr[0]:.0f}/{epl[0]:.0f}"]
        print(f"  {lab}: {tr:.0f}/{pl:.0f} -> {etr[0]:.0f}/{epl[0]:.0f}")
    ax.plot([0, 0], [-1, 1], color=qz.INK, lw=0.9, zorder=4)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    _save(fig, "appE", "construct-rotation", cols)


def _full_project(v):
    """Equal-angle projection of any direction (upper hemisphere plots outside the primitive)."""
    v = np.atleast_2d(v)
    return v[:, 0] / (1 - v[:, 2]), v[:, 1] / (1 - v[:, 2])


def joint_pyramid():
    """Joint pyramids of three sets on the full equal-angle projection: roof and east-facing sidewall."""
    lim = 2.4
    n = np.array([sl.plane_normal(d, a) for d, a in CROWN_SETS])
    g = np.linspace(-2.9, 4.0, 1001)
    X, Y = np.meshgrid(g, g)
    r2 = X ** 2 + Y ** 2
    V = np.stack([2 * X / (r2 + 1), 2 * Y / (r2 + 1), (r2 - 1) / (r2 + 1)], axis=-1)
    dots = np.einsum("ijk,lk->ijl", V, n)
    # sample the sphere to find which joint pyramids are removable for each free surface
    rng = np.random.default_rng(1)
    S = rng.normal(size=(200000, 3))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    sd = S @ n.T
    codes = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]

    def removable(code, free):
        sgn = np.array([1 if s == 0 else -1 for s in code])
        m = np.all(sd * sgn >= 0, axis=1)
        return m.any() and np.all(free(S[m]))

    fig, axes = qz.figure(W_FULL, 0.53 * W_FULL, ncols=2)
    cols = {"free_surface": [], "removable_JP": []}
    for ax, name, free, epmask, (x0, x1) in (
            (axes[0], "Roof", lambda s: s[:, 2] < 0, r2 > 1, (-2.75, 2.75)),
            (axes[1], "Sidewall", lambda s: s[:, 0] > 0, X < 0, (-1.7, 3.8))):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(x0, x1)
        ax.set_ylim(-lim, lim)
        ax.contourf(X, Y, np.where(epmask, 1.0, 0.0), levels=[0.5, 1.5], colors=[ROCK], zorder=1)
        for code in codes:
            if removable(code, free):
                sgn = np.array([1 if s == 0 else -1 for s in code])
                m = np.all(dots * sgn >= 0, axis=-1)
                ax.contourf(X, Y, np.where(m, 1.0, 0.0), levels=[0.5, 1.5], colors=[qz.CAR_TINT], zorder=2)
                cy, cx = np.argwhere(m).mean(axis=0)
                ax.text(g[int(cx)], g[int(cy)], "".join(map(str, code)), ha="center", va="center",
                        fontsize=8, color=qz.CAR_DARK, zorder=8)
                cols["free_surface"].append(name)
                cols["removable_JP"].append("".join(map(str, code)))
                print(f"  {name}: JP {''.join(map(str, code))} removable")
        ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=qz.INK, lw=1.0, zorder=5))
        for (d, a), ls in zip(CROWN_SETS, ("-", "-.", "--")):
            nn = sl.plane_normal(d, a)
            c = (-nn[0] / nn[2], -nn[1] / nn[2])
            ax.add_patch(Circle(c, 1 / nn[2], fill=False, ec=qz.MAL, lw=1.1, ls=ls, zorder=4))
        if name == "Sidewall":
            ax.plot([0, 0], [-lim, lim], color=qz.INK, lw=1.0, zorder=5)
        ax.add_patch(Rectangle((x0, -lim), x1 - x0, 2 * lim, fill=False, ec=qz.GREY_LIGHT, lw=0.6, zorder=6))
        if name == "Sidewall":
            ax.text(-0.07, 1.07, "N", ha="right", va="bottom", fontsize=8, zorder=8)
        else:
            ax.text(0, 1.07, "N", ha="center", va="bottom", fontsize=8, zorder=8)
        ax.set_title(name, fontsize=10, pad=12)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.06)
    _save(fig, "appE", "joint-pyramid", cols)


def blank_nets():
    for net, stem in ((EA, "blank-wulff-net"), (EQ, "blank-schmidt-net")):
        fig, ax = qz.figure(W_FULL, W_FULL)
        _grid(ax, net=net, step=2, color=qz.GREY_MID, lw=0.18, bold=10, bold_lw=0.45)
        ax.set_aspect("equal")
        ax.axis("off")
        lim = 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.add_patch(Circle((0, 0), 1.0, fill=False, ec=qz.INK, lw=0.9, zorder=5))
        for az in range(0, 360, 10):
            a = math.radians(az)
            ax.plot([math.sin(a), 1.025 * math.sin(a)], [math.cos(a), 1.025 * math.cos(a)],
                    color=qz.INK, lw=0.6)
        for az, lab in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            a = math.radians(az)
            ax.text(1.065 * math.sin(a), 1.065 * math.cos(a), lab, ha="center", va="center", fontsize=9)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        _save(fig, "appE", stem, {"net": [net], "interval_deg": [2], "bold_deg": [10]})


ALL = {"k_depth": k_depth, "erosion": erosion, "overcoring": overcoring, "hydrofrac": hydrofrac,
       "breakout": breakout, "stress_bracket": stress_bracket,
       "nets_compared": nets_compared, "pole_contours": pole_contours, "terzaghi": terzaghi,
       "removability": removability, "crown_wedge": crown_wedge, "kinematic": kinematic,
       "construct_plane_pole": construct_plane_pole, "construct_intersection": construct_intersection,
       "construct_rotation": construct_rotation, "joint_pyramid": joint_pyramid, "blank_nets": blank_nets}

if __name__ == "__main__":
    qz.use()
    wanted = sys.argv[1:]
    for name, fn in ALL.items():
        if not wanted or any(w in name for w in wanted):
            fn()
