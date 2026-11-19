"""Figures for Chapters 13 to 15: ground behaviour, high-stress ground, empirical design.

Every curve is computed from the equations and worked examples of the three
chapters (and, for rock mass strength, from ``geo_platform.rock``), drawn at
printed width in the house style of ``qz_figure.py`` and written as PDF,
300 dpi PNG and a CSV of the plotted data.

    python tools/figs/f5.py              # all figures
    python tools/figs/f5.py wedge        # figures whose name contains "wedge"
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
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch  # noqa: E402

import qz_figure as qz  # noqa: E402
from geo_platform import rock  # noqa: E402

FIG = os.path.join(BOOK, "figures")
W_FULL = 117 / 25.4
W_HALF = 0.62 * W_FULL
GAMMA_W = 9.81


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


def _arrow(ax, p0, p1, color=qz.INK, lw=0.8, style="-|>", ms=7, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, **kw))


def _dim(ax, p0, p1, text, offset=(0, 0), color=qz.GREY, **kw):
    """A two-headed dimension line with its label."""
    _arrow(ax, p0, p1, color=color, lw=0.6, style="<|-|>", ms=6)
    mx, my = (p0[0] + p1[0]) / 2 + offset[0], (p0[1] + p1[1]) / 2 + offset[1]
    ax.text(mx, my, text, color=qz.INK, ha="center", va="center", **kw)


# ---------------------------------------------------------------------------
# Chapter 13: behaviour classes on strength ratio and block size
# ---------------------------------------------------------------------------

def behaviour_map():
    fig, ax = qz.figure(W_FULL, 0.52 * W_FULL)
    xmin, xmax, ysplit = 0.01, 10.0, 0.42
    # upper half: structure decides; lower half: material and stress decide
    regions = [
        (xmin, 0.1, ysplit, 1.0, qz.MAL_TINT, "Heavily\nfractured", "Equivalent\ncontinuum"),
        (0.1, 1 / 3, ysplit, 1.0, "white", "Transition", "Run both"),
        (1 / 3, 3.0, ysplit, 1.0, qz.MAL_TINT, "Blocky", "Limit\nequilibrium"),
        (3.0, xmax, ysplit, 1.0, "white", "Massive", "Boundary\nstress"),
    ]
    for x0, x1, y0, y1, fc, name, route in regions:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc, edgecolor="none", zorder=0))
        xc = math.sqrt(x0 * x1)
        ax.text(xc, y0 + 0.62 * (y1 - y0), name, ha="center", va="center", fontsize=9)
        ax.text(xc, y0 + 0.28 * (y1 - y0), route, ha="center", va="center", fontsize=7.5,
                color=qz.GREY, style="italic")
    # lower row: one band at any block size; the two classes in it are told apart by the
    # strength of the intact material (Section 13.1.1), not by a value on the horizontal axis
    ax.add_patch(Rectangle((xmin, 0.0), xmax - xmin, ysplit, facecolor=qz.CAR_TINT, edgecolor="none", zorder=0))
    low = [
        (xmin, xmax, 0.0, ysplit, "Weak and soft ground", "Intact material weak", "Face stability, deformation"),
        (xmin, xmax, 0.0, ysplit, "High-stress ground", "Intact material strong, brittle", "Spalling depth, energy"),
    ]
    for xc, (_, _, _, _, name, cond, route) in zip((10 ** -1.25, 10 ** 0.25), low):
        ax.text(xc, 0.315, name, ha="center", va="center", fontsize=9)
        ax.text(xc, 0.205, cond, ha="center", va="center", fontsize=7.5, color=qz.CAR_DARK)
        ax.text(xc, 0.095, route, ha="center", va="center", fontsize=7.5, color=qz.GREY, style="italic")
    ax.text(10 ** -0.5, 0.205, "or", ha="center", va="center", fontsize=7.5, color=qz.GREY)
    regions += [(x0, x1, y0, y1, qz.CAR_TINT, f"{name} ({cond.lower()})", route)
                for x0, x1, y0, y1, name, cond, route in low]
    ax.plot([0.1, 0.1], [ysplit, 1.0], color=qz.INK, lw=0.7, ls="--")
    ax.plot([1 / 3, 1 / 3], [ysplit, 1.0], color=qz.INK, lw=0.7, ls="--")
    ax.plot([3.0, 3.0], [ysplit, 1.0], color=qz.GREY, lw=0.6, ls=":")
    ax.axhline(ysplit, color=qz.INK, lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.01, 0.1, 1 / 3, 1, 10])
    ax.set_xticklabels(["0.01", "0.1", "1/3", "1", "10"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks([0.06, 0.94])
    ax.set_yticklabels(["Low", "High"])
    ax.tick_params(axis="y", length=0)
    qz.label(ax, "Block size to opening span, $l_b/B$ (-)", "Strength to stress ratio (-)")
    fig.tight_layout()
    _save(fig, "ch13", "behaviour-class-map",
          {"region": [r[5] for r in regions], "x0": [r[0] for r in regions], "x1": [r[1] for r in regions],
           "y0_schematic": [r[2] for r in regions], "y1_schematic": [r[3] for r in regions],
           "analysis_route": [r[6].replace("\n", " ") for r in regions]})


# ---------------------------------------------------------------------------
# Chapter 13: voussoir beam
# ---------------------------------------------------------------------------

def voussoir():
    fig, ax = qz.figure(W_FULL, 0.40 * W_FULL)
    s, t = 10.0, 1.0                      # schematic span and bed thickness (m)
    sag = 0.25                            # exaggerated midspan deflection
    x = np.linspace(0, s, 201)
    defl = sag * np.sin(np.pi * x / s)
    top = t - defl
    bot = -defl
    # abutment rock
    for xa, w in ((-2.2, 2.2), (s, 2.2)):
        ax.add_patch(Rectangle((xa, -1.2), w, 3.4, facecolor=qz.GREY_LIGHT, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((-2.2, t), s + 4.4, 1.2, facecolor=qz.GREY_LIGHT, edgecolor="none", zorder=0))
    # the two halves of the bed, cracked at midspan and at the abutments
    for xs in (x[x <= s / 2], x[x >= s / 2]):
        d = sag * np.sin(np.pi * xs / s)
        poly = np.column_stack([np.r_[xs, xs[::-1]], np.r_[t - d, (-d)[::-1]]])
        ax.add_patch(Polygon(poly, closed=True, facecolor="white", edgecolor=qz.INK, lw=0.9, zorder=2))
    ax.plot([0, s], [t, t], color=qz.INK, lw=0.6, ls=(0, (4, 2)), zorder=3)   # bedding plane above
    # thrust line: low at the abutments, high at midspan (parabola within the bed)
    n = 0.25                              # contact depth as a fraction of t
    yl = bot + n * t / 2
    yh = top - n * t / 2
    thrust = yl + (yh - yl) * (1 - (2 * x / s - 1) ** 2)
    ax.plot(x, thrust, color=qz.MAL, lw=1.4, zorder=4)
    # compression zones: top at midspan, bottom at the abutments
    for xc, yc in ((s / 2, top[100] - n * t / 2), (0.25, bot[0] + n * t / 2), (s - 0.25, bot[-1] + n * t / 2)):
        ax.add_patch(Rectangle((xc - 0.25, yc - n * t / 2), 0.5, n * t, facecolor=qz.CAR, alpha=0.85,
                               edgecolor="none", zorder=5))
    ax.annotate("Midspan contact", (s / 2 + 0.2, top[100] - 0.1), xytext=(s / 2 + 1.6, t + 0.55), fontsize=8,
                color=qz.CAR_DARK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=qz.CAR_DARK, lw=0.6))
    ax.annotate("Abutment contact", (0.4, bot[0] + 0.05), xytext=(1.6, -0.75), fontsize=8,
                color=qz.CAR_DARK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=qz.CAR_DARK, lw=0.6))
    ax.text(2.6, 0.12, "Thrust line", ha="center", va="center", fontsize=8, color=qz.MAL)
    _dim(ax, (0, -1.75), (s, -1.75), "$s$", offset=(0, -0.28))
    _dim(ax, (s + 0.6, 0), (s + 0.6, t), "$t$", offset=(0.35, 0))
    ax.set_xlim(-2.2, s + 2.2)
    ax.set_ylim(-2.25, t + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "ch13", "voussoir-roof", {"x_m": x, "bed_top_m": top, "bed_bottom_m": bot, "thrust_line_m": thrust})


# ---------------------------------------------------------------------------
# Chapter 13: suspension load and beam building
# ---------------------------------------------------------------------------

def roof_bolting():
    fig, (a1, a2) = qz.figure(W_FULL, 0.46 * W_FULL, ncols=2)
    gamma = 25.0
    sp = np.linspace(1.0, 2.5, 151)
    cols = {"spacing_m": sp}
    a1.axhspan(130, 180, color=qz.GREY_LIGHT, alpha=0.6, lw=0, zorder=0)
    a1.text(1.03, 155, "Rebar yield", fontsize=8, color=qz.GREY, va="center")
    for th, col, ls in ((0.5, qz.MAL_SHADES[3], ":"), (1.0, qz.MAL, "-"), (1.5, qz.MAL_SHADES[0], "--"),
                        (2.0, qz.CAR, "-.")):
        load = gamma * th * sp ** 2
        a1.plot(sp, load, color=col, ls=ls, lw=1.3, label=f"$h = {th:g}$ m")
        cols[f"load_h{th:g}_kN"] = load
    a1.plot([1.5], [gamma * 1.0 * 1.5 ** 2], "o", mfc="white", mec=qz.INK, ms=5, mew=1.0, zorder=5)
    a1.annotate("56 kN", (1.5, 56.25), xytext=(1.64, 12), fontsize=8, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=qz.GREY, lw=0.6))
    a1.set_xlim(1.0, 2.5)
    a1.set_ylim(0, 440)
    qz.label(a1, "Bolt spacing, $s_b$ (m)", "Bolt load, $W$ (kN)")
    a1.legend(loc="upper left", ncol=2, handlelength=1.5, columnspacing=0.6, handletextpad=0.4,
              borderaxespad=0.3, borderpad=0.2)

    nb = np.arange(1, 6)
    ratio = nb.astype(float) ** 2            # (n t)^3 / (n t^3)
    a2.bar(nb - 0.18, np.ones_like(ratio), width=0.34, color=qz.GREY_LIGHT, label="Beds acting separately")
    a2.bar(nb + 0.18, ratio, width=0.34, color=qz.MAL, label="Beds bolted as one")
    a2.text(4.18, 16.6, "16", ha="center", va="bottom", fontsize=8)
    a2.set_xticks(nb)
    a2.set_xlim(0.4, 5.6)
    a2.set_ylim(0, 30)
    a2.tick_params(axis="x", top=False)
    qz.label(a2, "Number of beds, $n$ (-)", "Relative stiffness (-)")
    a2.legend(loc="upper left")
    fig.tight_layout()
    cols.update({"n_beds": nb, "stiffness_bolted_over_separate": ratio})
    _save(fig, "ch13", "roof-bolt-load", cols)


# ---------------------------------------------------------------------------
# Chapter 13: stability ratio
# ---------------------------------------------------------------------------

def stability_ratio():
    fig, ax = qz.figure(W_HALF, 0.82 * W_HALF)
    gamma, ncrit = 20.0, 6.0
    su = np.linspace(0, 120, 241)
    cols = {"su_kPa": su}
    for z, col, ls in ((10, qz.MAL_SHADES[3], ":"), (15, qz.MAL, "-"), (20, qz.MAL_SHADES[0], "--"),
                       (30, qz.CAR, "-.")):
        st = np.clip(gamma * z - ncrit * su, 0, None)
        pos = su <= gamma * z / ncrit
        ax.plot(su[pos], st[pos], color=col, ls=ls, lw=1.3, label=f"$z = {z}$ m")
        cols[f"sigmaT_z{z}_kPa"] = st
    ax.plot([50], [0], "o", mfc="white", mec=qz.INK, ms=5, mew=1.0, zorder=5, clip_on=False)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 620)
    qz.label(ax, "Undrained strength, $s_u$ (kPa)", "Face pressure, $\\sigma_T$ (kPa)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "ch13", "stability-ratio-pressure", cols)


# ---------------------------------------------------------------------------
# Chapter 13: wedge-silo model
# ---------------------------------------------------------------------------

def wedge_silo(D, C, gamma, c, phi_deg, k_silo=0.8, k_wedge=0.4, n=400):
    """Required face support pressure (kPa) for a square face of side D, cover C.

    Wedge at the face loaded by a prism whose vertical stress follows Janssen's
    silo theory (Anagnostou and Kovari 1996); the result is the maximum over the
    inclination of the sliding plane.
    """
    tp = math.tan(math.radians(phi_deg))
    om = np.radians(np.linspace(15, 85, n))
    cot = 1 / np.tan(om)
    B = H = D
    A = B * H * cot
    U = 2 * (B + H * cot)
    R = A / U
    if C > 0:
        sv = (R * gamma - c) / (k_silo * tp) * (1 - np.exp(-k_silo * tp * C / R))
    else:
        sv = np.zeros_like(om)
    sv = np.clip(sv, 0, None)
    G = 0.5 * B * H ** 2 * cot * gamma
    V = sv * A
    Ts = c * H ** 2 * cot / 2 + tp * k_wedge * (sv * H ** 2 * cot / 2 + gamma * H ** 3 * cot / 6)
    S = ((G + V) * (np.sin(om) - np.cos(om) * tp) - c * B * H / np.sin(om) - 2 * Ts) / (np.cos(om) + np.sin(om) * tp)
    i = int(np.argmax(S))
    return max(float(S[i]) / (B * H), 0.0), math.degrees(float(om[i]))


def wedge_silo_fig():
    fig, (a1, a2) = qz.figure(W_FULL, 0.47 * W_FULL, ncols=2, gridspec_kw={"width_ratios": [1, 1.15]})
    # schematic: longitudinal section through the face
    D, C = 1.0, 1.6
    om = math.radians(58)
    L = D / math.tan(om)
    a1.add_patch(Rectangle((-2.2, -0.05), 2.2 + L + 0.9, C + D + 0.05, facecolor=qz.GREY_LIGHT, alpha=0.35,
                           edgecolor="none", zorder=0))
    a1.add_patch(Rectangle((0, 0), 1.0, D, facecolor="white", edgecolor="none", zorder=1))   # tunnel
    a1.plot([0, 1.0], [0, 0], color=qz.INK, lw=0.9)
    a1.plot([0, 1.0], [D, D], color=qz.INK, lw=0.9)
    wedge = np.array([[0, 0], [0, D], [-L, D]])
    a1.add_patch(Polygon(wedge, closed=True, facecolor=qz.CAR_TINT, edgecolor=qz.CAR, lw=1.1, zorder=3))
    a1.add_patch(Rectangle((-L, D), L, C, facecolor=qz.MAL_TINT, edgecolor=qz.MAL, lw=1.1, zorder=3))
    a1.plot([-2.2, L + 0.9], [C + D, C + D], color=qz.INK, lw=0.9)
    # arching: shear on the prism sides
    for y in np.linspace(D + 0.3, C + D - 0.3, 3):
        _arrow(a1, (-L - 0.12, y - 0.12), (-L - 0.12, y + 0.12), color=qz.MAL, lw=0.8, ms=5)
        _arrow(a1, (0.12, y - 0.12), (0.12, y + 0.12), color=qz.MAL, lw=0.8, ms=5)
    for yy in (0.2, 0.5, 0.8):
        _arrow(a1, (0.32, yy * D), (0.03, yy * D), color=qz.INK, lw=0.7, ms=5)
    a1.text(-L / 2, D + C / 2, "Prism", ha="center", va="center", fontsize=8, color=qz.MAL, rotation=90)
    a1.annotate("Wedge", (-0.12, 0.45 * D), xytext=(-0.75, 0.2 * D), fontsize=8, color=qz.CAR, ha="center",
                va="center", arrowprops=dict(arrowstyle="-", color=qz.CAR, lw=0.6))
    a1.text(0.42, 0.5 * D, "$\\sigma_T$", ha="left", va="center", fontsize=9, zorder=4)
    _dim(a1, (-1.15, D), (-1.15, D + C), "$C$", offset=(-0.17, 0))
    _dim(a1, (-1.15, 0), (-1.15, D), "$D$", offset=(-0.17, 0))
    a1.set_xlim(-1.5, L + 0.95)
    a1.set_ylim(-0.12, C + D + 0.08)
    a1.set_aspect("equal")
    a1.axis("off")

    Dm, gamma, phi = 10.0, 20.0, 30.0
    cd = np.linspace(0.0, 4.0, 81)
    cols = {"C_over_D": cd}
    for c, col, ls in ((0, qz.CAR, "-"), (2, qz.MAL_SHADES[0], "--"), (5, qz.MAL, "-."), (10, qz.MAL_SHADES[3], ":")):
        st = np.array([wedge_silo(Dm, x * Dm, gamma, c, phi)[0] for x in cd])
        a2.plot(cd, st, color=col, ls=ls, lw=1.3, label=f"$c' = {c}$ kPa")
        cols[f"sigmaT_c{c}_kPa"] = st
    a2.set_xlim(0, 4)
    a2.set_ylim(0, None)
    qz.label(a2, "Cover to diameter, $C/D$ (-)", "Face pressure, $\\sigma_T$ (kPa)")
    a2.legend(loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.8)
    lo, hi = a2.get_ylim()
    a2.set_ylim(0, hi * 1.4)
    fig.tight_layout()
    print("  wedge-silo c=0: " + ", ".join(f"C/D {x:g}: {wedge_silo(Dm, x * Dm, gamma, 0, phi)[0]:.1f}"
                                         for x in (0.25, 0.5, 1, 2, 4)))
    _save(fig, "ch13", "wedge-silo", cols)


def face_window():
    fig, ax = qz.figure(W_HALF, 0.85 * W_HALF)
    Dm, gsat, phi = 10.0, 20.0, 30.0
    gsub = gsat - GAMMA_W
    cov = np.linspace(0.0, 20.0, 201)
    eff = np.array([wedge_silo(Dm, C, gsub, 0.0, phi)[0] for C in cov])
    lower = eff + GAMMA_W * cov
    upper = gsat * cov
    ok = upper >= lower
    ax.fill_between(cov, lower, upper, where=ok, color=qz.MAL_TINT, lw=0, zorder=0)
    ax.plot(cov, upper, color=qz.CAR, lw=1.3, ls="--", label="Blow-out")
    ax.plot(cov, lower, color=qz.MAL, lw=1.3, label="Collapse")
    cross = cov[np.argmax(ok & (cov > 0.5))]
    ax.axvline(cross, color=qz.GREY, lw=0.6, ls=":")
    ax.text(12, 0.5 * (np.interp(12, cov, lower) + np.interp(12, cov, upper)), "Window", ha="center",
            va="center", fontsize=8, color=qz.MAL)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 420)
    qz.label(ax, "Cover to crown, $C$ (m)", "Pressure at crown (kPa)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    print(f"  face window closes at C = {cross:.1f} m")
    _save(fig, "ch13", "face-pressure-window", {"C_m": cov, "collapse_kPa": lower, "blowout_kPa": upper,
                                                 "effective_wedge_silo_kPa": eff})


# ---------------------------------------------------------------------------
# Chapter 14: spalling notch
# ---------------------------------------------------------------------------

UCS, S1, S3 = 224.0, 60.0, 11.0


def spalling_notch():
    smax = 3 * S1 - S3
    ra = 0.5 + 1.25 * smax / UCS
    thr = 0.4 * UCS
    # boundary stress against the angle from the sigma_1 direction; the arc where
    # it exceeds 0.4 UCS sets the angular extent of the notch
    th = np.linspace(0, 180, 1801)
    sth = (S1 + S3) - 2 * (S1 - S3) * np.cos(np.radians(2 * th))
    over = sth > thr
    th_lo = float(th[over][0])
    half = 90 - th_lo

    fig, (a1, a2) = qz.figure(W_FULL, 0.52 * W_FULL, ncols=2, gridspec_kw={"width_ratios": [1, 1.25]})
    phi = np.linspace(0, 2 * np.pi, 361)
    a1.fill(np.cos(phi), np.sin(phi), facecolor="white", edgecolor=qz.INK, lw=1.0, zorder=2)
    for sgn in (1, -1):
        a0, a1_ = math.radians(90 - half), math.radians(90 + half)
        notch = [(math.cos(a0), sgn * math.sin(a0)), (0, sgn * ra), (math.cos(a1_), sgn * math.sin(a1_))]
        arc = np.radians(np.linspace(90 + half, 90 - half, 60))
        pts = notch + list(zip(np.cos(arc), sgn * np.sin(arc)))
        a1.add_patch(Polygon(pts, closed=True, facecolor=qz.CAR_TINT, edgecolor=qz.CAR, lw=1.0, zorder=3))
    for sgn in (1, -1):
        _arrow(a1, (sgn * 2.05, 0), (sgn * 1.45, 0), color=qz.INK, lw=1.0, ms=8)
        _arrow(a1, (0.0, sgn * 2.05), (0.0, sgn * 1.8), color=qz.GREY, lw=0.7, ms=6)
    a1.text(1.75, 0.1, "$\\sigma_1$", ha="center", va="bottom", fontsize=9)
    a1.text(0.12, -2.0, "$\\sigma_3$", ha="left", va="center", fontsize=9, color=qz.GREY)
    _arrow(a1, (0.62, 0), (0.62, ra), color=qz.GREY, lw=0.6, style="<|-|>", ms=6, zorder=5)
    a1.text(0.7, 0.62 * ra, f"$r = {ra:.2f}a$", fontsize=8, ha="left", va="center")
    a1.plot([0, 0.62], [0, 0], color=qz.GREY, lw=0.5, ls=":")
    a1.plot([0, 0.68], [ra, ra], color=qz.GREY, lw=0.5, ls=":")
    a1.set_xlim(-2.15, 2.15)
    a1.set_ylim(-2.15, 2.15)
    a1.set_aspect("equal")
    a1.axis("off")

    a2.fill_between(th, thr / UCS, sth / UCS, where=over, color=qz.CAR_TINT, lw=0, interpolate=True)
    a2.plot(th, sth / UCS, color=qz.INK, lw=1.3)
    a2.axhline(thr / UCS, color=qz.CAR, lw=0.9, ls="--")
    a2.text(5, thr / UCS + 0.03, "$0.4\\,\\mathrm{UCS}$", fontsize=8, color=qz.CAR, va="bottom")
    a2.set_xlim(0, 180)
    a2.set_xticks([0, 45, 90, 135, 180])
    a2.set_ylim(-0.1, 0.9)
    a2.axhline(0, color=qz.GREY_LIGHT, lw=0.6)
    qz.label(a2, "Angle from $\\sigma_1$ direction, $\\theta$ ($^\\circ$)",
             "Boundary stress, $\\sigma_\\theta/\\mathrm{UCS}$ (-)")
    fig.tight_layout()
    print(f"  spalling notch: r/a {ra:.3f}, overstressed for theta {th_lo:.1f} to {180 - th_lo:.1f} deg")
    _save(fig, "ch14", "spalling-notch", {"theta_deg": th, "sigma_theta_MPa": sth, "sigma_theta_over_UCS": sth / UCS,
                                          "r_over_a": [ra], "notch_half_angle_deg": [half]})


# ---------------------------------------------------------------------------
# Chapter 14: squeezing strain and over-excavation allowance
# ---------------------------------------------------------------------------

def sigma_cm_hm(sci, mi, gsi):
    """Rock mass strength of Hoek and Marinos (2000)."""
    return 0.0034 * mi ** 0.8 * sci * (1.029 + 0.025 * np.exp(-0.1 * mi)) ** gsi


def _class_bands(ax, axis="y", text=True, xpos=0.98, lo_lim=0.3):
    bands = ((0, 1, "Few problems"), (1, 2.5, "Minor"), (2.5, 5, "Severe"), (5, 10, "Very severe"),
             (10, 30, "Extreme"))
    for i, (lo, hi, name) in enumerate(bands):
        span = ax.axhspan if axis == "y" else ax.axvspan
        span(lo, hi, color=qz.CAR_TINT if i >= 2 else qz.MAL_TINT, lw=0, zorder=0, alpha=1.0 if i % 2 == 0 else 0.55)
        if text:
            yc = {0: 0.55, 1: 1.6, 2.5: 3.3, 5: 8.4, 10: 16}[lo]
            ax.text(xpos, yc, name, ha="right", va="center", fontsize=7.5, color=qz.GREY,
                    transform=ax.get_yaxis_transform())


def squeezing_allowance():
    sci, mi, p0, a = 72.0, 7.0, 16.9, 4.75
    gsi = np.linspace(15, 50, 351)
    scm = sigma_cm_hm(sci, mi, gsi)
    eps = 0.2 * (scm / p0) ** -2
    fig, ax = qz.figure(W_FULL, 0.5 * W_FULL)
    _class_bands(ax)
    ax.axvspan(25, 35, color=qz.GREY_LIGHT, alpha=0.35, lw=0, zorder=0.5)
    ax.plot(gsi, eps, color=qz.INK, lw=1.4)
    e30 = float(0.2 * (sigma_cm_hm(sci, mi, 30) / p0) ** -2)
    ax.plot([30], [e30], "o", mfc="white", mec=qz.INK, ms=5, mew=1.0, zorder=5)
    ax.axhline(6.5, color=qz.CAR, lw=0.9, ls="--")
    ax.text(16, 6.5 * 1.10, "84th percentile, 6.5%", fontsize=8, color=qz.CAR, va="bottom")
    ax.text(16, e30 * 1.06, "Mean, 3.7%", fontsize=8, color=qz.INK, va="bottom")
    ax.axhline(e30, color=qz.INK, lw=0.5, ls=":", xmax=(30 - 15) / 35)
    ax.set_yscale("log")
    ax.set_xlim(15, 50)
    ax.set_ylim(0.3, 25)
    ticks = [0.5, 1, 2.5, 5, 10, 20]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.tick_params(axis="y", right=False)
    qz.label(ax, "Geological strength index, GSI (-)", "Closure strain, $\\varepsilon$ (%)")
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(0.3e-2 * a * 1000, 25e-2 * a * 1000)
    rt = [25, 50, 100, 176, 310, 500, 1000]
    ax2.set_yticks(rt)
    ax2.set_yticklabels([str(r) for r in rt])
    ax2.minorticks_off()
    ax2.set_ylabel("Radial allowance, $\\varepsilon a$ (mm)")
    fig.tight_layout()
    print(f"  squeezing: GSI 30 strain {e30:.2f}%, allowance {e30 / 100 * a * 1000:.0f} mm; "
          f"6.5% -> {0.065 * a * 1000:.0f} mm")
    _save(fig, "ch14", "squeezing-allowance", {"GSI": gsi, "sigma_cm_MPa": scm, "strain_pct": eps,
                                                "allowance_mm": eps / 100 * a * 1000})


# ---------------------------------------------------------------------------
# Chapter 14: stable and violent failure, relative stiffness
# ---------------------------------------------------------------------------

def burst_stiffness():
    fig, ax = qz.figure(W_HALF, 0.82 * W_HALF)
    d = np.linspace(0, 1, 101)
    pre = 1 - (1 - d) ** 2                           # pre-peak, peak at d = 1
    lam = 1.6                                        # post-peak softening slope (normalised)
    dp = np.linspace(1, 1 + 1 / lam, 60)
    post = 1 - lam * (dp - 1)
    ax.plot(d, pre, color=qz.INK, lw=1.4)
    ax.plot(dp, post, color=qz.INK, lw=1.4)
    # unloading lines of the surrounding rock from the peak
    for k, col, ls, name in ((4.0, qz.MAL, "-", "Stiff surround"), (0.8, qz.CAR, "--", "Soft surround")):
        dd = np.linspace(1, 1 + 1 / k, 60)
        ax.plot(dd, 1 - k * (dd - 1), color=col, ls=ls, lw=1.3, label=name)
    dd = np.linspace(1, 1 + 1 / 0.8, 120)
    ax.fill_between(dd, np.clip(1 - lam * (dd - 1), 0, None), 1 - 0.8 * (dd - 1), color=qz.CAR_TINT, lw=0)
    ax.text(1.79, 0.13, "Excess\nenergy", fontsize=7.5, linespacing=1.0, color=qz.CAR_DARK, ha="center", va="center")
    ax.set_xlim(0, 2.3)
    ax.set_ylim(0, 1.15)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 0.5, 1])
    qz.label(ax, "Displacement, $\\delta/\\delta_p$ (-)", "Load, $F/F_p$ (-)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "ch14", "burst-stiffness", {"delta_pre": d, "F_pre": pre, "delta_post": dp, "F_post": post})


# ---------------------------------------------------------------------------
# Chapter 14: kinetic energy demand and displacement compatibility
# ---------------------------------------------------------------------------

def burst_energy():
    rho = 2700.0
    v = np.linspace(0, 6, 121)
    fig, (a1, a2) = qz.figure(W_FULL, 0.47 * W_FULL, ncols=2, gridspec_kw={"width_ratios": [1.3, 1]})
    a1.axhspan(20, 50, color=qz.GREY_LIGHT, alpha=0.6, lw=0, zorder=0)
    a1.text(0.15, 21.5, "Yielding bolts", fontsize=8, color=qz.GREY, va="bottom")
    cols = {"v_m_s": v}
    for t, col, ls in ((0.3, qz.MAL, "-"), (0.6, qz.MAL_SHADES[0], "--"), (1.0, qz.CAR, "-.")):
        e = 0.5 * rho * t * v ** 2 / 1000
        a1.plot(v, e, color=col, ls=ls, lw=1.3, label=f"$t = {t:g}$ m")
        cols[f"E_t{t:g}_kJ_m2"] = e
    a1.plot([3], [0.5 * rho * 0.3 * 9 / 1000], "o", mfc="white", mec=qz.INK, ms=5, mew=1.0, zorder=5)
    a1.annotate("3.6 kJ/m$^2$", (3, 3.645), xytext=(1.45, 14), fontsize=8, va="center", ha="center",
                arrowprops=dict(arrowstyle="-", color=qz.GREY, lw=0.6))
    a1.set_xlim(0, 6)
    a1.set_ylim(0, 85)
    qz.label(a1, "Ejection velocity, $v$ (m/s)", "Kinetic energy, $E$ (kJ/m$^2$)")
    a1.legend(loc="upper left")

    names = ["Bolt", "Mesh", "Shotcrete"]
    disp = [200, 50, 20]
    y = np.arange(3)[::-1]
    a2.barh(y, disp, height=0.55, color=[qz.MAL, qz.MAL_SHADES[3], qz.CAR])
    a2.axvline(20, color=qz.CAR, lw=0.9, ls="--")
    a2.set_yticks(y)
    a2.set_yticklabels(names)
    a2.tick_params(axis="y", length=0, right=False)
    a2.set_xlim(0, 250)
    a2.set_ylim(-0.6, 2.6)
    for yi, dv in zip(y, disp):
        a2.text(dv + 5, yi, f"{dv}", va="center", fontsize=8)
    qz.label(a2, "Displacement, $\\delta$ (mm)", None)
    fig.tight_layout()
    _save(fig, "ch14", "burst-energy", {**cols, "element": names, "displacement_capacity_mm": disp})


# ---------------------------------------------------------------------------
# Chapter 15: spalling depth and severity
# ---------------------------------------------------------------------------

def spalling_depth():
    ratio = np.linspace(0.2, 1.2, 201)
    ra = np.where(ratio < 0.4, 1.0, 0.5 + 1.25 * ratio)
    smax = 3 * S1 - S3
    x0 = smax / UCS
    fig, (a1, a2) = qz.figure(W_FULL, 0.47 * W_FULL, ncols=2)
    a1.plot(ratio, ra, color=qz.INK, lw=1.4)
    a1.axvline(0.4, color=qz.GREY, lw=0.6, ls=":")
    a1.plot([x0], [0.5 + 1.25 * x0], "o", mfc="white", mec=qz.INK, ms=5, mew=1.0, zorder=5)
    a1.annotate(f"$r/a = {0.5 + 1.25 * x0:.2f}$", (x0, 0.5 + 1.25 * x0), xytext=(0.42, 1.6), fontsize=8,
                arrowprops=dict(arrowstyle="-", color=qz.GREY, lw=0.6))
    a1.set_xlim(0.2, 1.2)
    a1.set_ylim(0.9, 2.1)
    qz.label(a1, "Stress ratio, $\\sigma_{\\max}/\\mathrm{UCS}$ (-)", "Depth of failure, $r/a$ (-)")

    bands = ((0, 1, "Stable"), (1, 1.5, "Minor"), (1.5, 2, "Moderate"), (2, 3.5, "Serious"))
    for i, (lo, hi, name) in enumerate(bands):
        a2.axhspan(lo, hi, color=qz.MAL_TINT if i == 0 else qz.CAR_TINT, alpha=1.0 if i % 2 == 0 else 0.55,
                   lw=0, zorder=0)
        a2.text(0.04, {0: 0.13, 1: 1.3, 1.5: 1.75, 2: 2.5}[lo], name, ha="left", va="center", fontsize=7.5, color=qz.GREY,
                transform=a2.get_yaxis_transform())
    cols = {"smax_over_UCS": ratio, "r_over_a": ra}
    for f, col, ls in ((0.4, qz.CAR, "--"), (0.6, qz.MAL, "-")):
        sev = ratio / f
        a2.plot(ratio, sev, color=col, ls=ls, lw=1.3, label=f"$\\mathrm{{CI}} = {f:g}\\,\\mathrm{{UCS}}$")
        a2.plot([x0], [x0 / f], "o", mfc="white", mec=col, ms=5, mew=1.0, zorder=5)
        cols[f"smax_over_CI_{f:g}"] = sev
    a2.set_xlim(0.2, 1.2)
    a2.set_ylim(0, 3.0)
    qz.label(a2, "Stress ratio, $\\sigma_{\\max}/\\mathrm{UCS}$ (-)", "Severity, $\\sigma_{\\max}/\\mathrm{CI}$ (-)")
    a2.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "ch15", "spalling-depth", cols)


# ---------------------------------------------------------------------------
# Chapter 15: rock mass strength, Hoek and Marinos against uniaxial
# ---------------------------------------------------------------------------

def rock_mass_strength():
    gsi = np.linspace(10, 80, 141)
    fig, (a1, a2) = qz.figure(W_FULL, 0.47 * W_FULL, ncols=2)
    cols = {"GSI": gsi}
    for mi, col, ls in ((7, qz.MAL, "-"), (15, qz.MAL_SHADES[0], "--"), (25, qz.CAR, "-.")):
        hm = sigma_cm_hm(1.0, mi, gsi)
        uni = np.array([rock.rock_mass_ucs(1.0, rock.hb_constants(g, mi)) for g in gsi])
        a1.plot(gsi, hm, color=col, ls=ls, lw=1.3, label=f"$m_i = {mi}$")
        a2.plot(gsi, hm / uni, color=col, ls=ls, lw=1.3, label=f"$m_i = {mi}$")
        cols[f"scm_HM_over_sci_mi{mi}"] = hm
        cols[f"sci_s_a_over_sci_mi{mi}"] = uni
        cols[f"ratio_mi{mi}"] = hm / uni
    uni7 = np.array([rock.rock_mass_ucs(1.0, rock.hb_constants(g, 7)) for g in gsi])
    a1.plot(gsi, uni7, color=qz.GREY, lw=1.0, ls=":", label="$s^{a}$, $m_i = 7$")
    a1.set_yscale("log")
    a1.set_xlim(10, 80)
    a1.set_ylim(1e-3, 1)
    qz.label(a1, "GSI (-)", "Strength ratio, $\\sigma_{cm}/\\sigma_{ci}$ (-)")
    a1.legend(loc="lower right")
    r30 = float(sigma_cm_hm(72, 7, 30) / rock.rock_mass_ucs(72, rock.hb_constants(30, 7)))
    a2.plot([30], [r30], "o", mfc="white", mec=qz.INK, ms=5, mew=1.0, zorder=5)
    a2.set_xlim(10, 80)
    a2.set_ylim(0, None)
    a2.set_ylim(0, a2.get_ylim()[1] * 1.05)
    qz.label(a2, "GSI (-)", "Ratio to uniaxial value (-)")
    a2.legend(loc="upper right")
    fig.tight_layout()
    print(f"  rock mass strength ratio at GSI 30, mi 7: {r30:.2f}")
    _save(fig, "ch15", "rock-mass-strength-hm", cols)


# ---------------------------------------------------------------------------
# Chapter 15: RMR bolt length
# ---------------------------------------------------------------------------

def bolt_length():
    B = np.linspace(6, 18, 121)
    l1 = 0.125 * B + 2.25
    l2 = 0.3 * B
    fig, ax = qz.figure(W_HALF, 0.82 * W_HALF)
    ax.plot(B, np.maximum(l1, l2), color=qz.MAL_TINT, lw=5, solid_capstyle="butt", zorder=1, label="Adopted")
    ax.plot(B, l1, color=qz.MAL, lw=1.3, label="$L = 0.125B + 2.25$", zorder=2)
    ax.plot(B, l2, color=qz.CAR, lw=1.3, ls="--", label="$L = 0.3B$", zorder=2)
    bx = 2.25 / (0.3 - 0.125)
    ax.axvline(bx, color=qz.GREY, lw=0.6, ls=":")
    for y, c in ((4.25, qz.MAL), (4.8, qz.CAR)):
        ax.plot([16], [y], "o", mfc="white", mec=c, ms=5, mew=1.0, zorder=5)
    ax.set_xlim(6, 18)
    ax.set_ylim(1.5, 6)
    qz.label(ax, "Span, $B$ (m)", "Bolt length, $L$ (m)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    print(f"  bolt length relations cross at B = {bx:.2f} m")
    _save(fig, "ch15", "bolt-length-span", {"B_m": B, "L1_m": l1, "L2_m": l2})


# ---------------------------------------------------------------------------
# Chapter 15: Q unsupported span and wall rating
# ---------------------------------------------------------------------------

def q_span_wall():
    q = np.logspace(-3, 3, 301)
    fig, (a1, a2) = qz.figure(W_FULL, 0.47 * W_FULL, ncols=2)
    cols = {"Q": q}
    series = ((0.5, qz.MAL_SHADES[3], ":"), (0.8, qz.CAR, "-"), (1.0, qz.MAL, "-"), (1.6, qz.MAL_SHADES[0], "--"),
              (3.0, qz.GREY, "-."))
    for esr, col, ls in series:
        sp = 2 * esr * q ** 0.4
        a1.plot(q, sp, color=col, ls=ls, lw=1.3, label=f"$\\mathrm{{ESR}} = {esr:g}$")
        cols[f"span_ESR{esr:g}_m"] = sp
    a1.plot([5], [2 * 0.8 * 5 ** 0.4], "o", mfc="white", mec=qz.CAR, ms=5, mew=1.0, zorder=5)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlim(1e-3, 1e3)
    a1.set_ylim(0.001, 100)
    a1.set_xticks([1e-3, 1e-1, 1e1, 1e3])
    a1.set_yticks([0.001, 0.01, 0.1, 1, 10, 100])
    a1.set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
    qz.label(a1, "Rock mass quality, $Q$ (-)", "Span, $\\mathrm{Span}_{\\max}$ (m)")
    a1.legend(loc="lower right", fontsize=7.5, handlelength=1.8, labelspacing=0.2, borderaxespad=0.3)

    qw = np.where(q > 10, 5 * q, np.where(q > 0.1, 2.5 * q, q))
    for seg in (q <= 0.1, (q > 0.1) & (q <= 10), q > 10):
        a2.plot(q[seg], qw[seg], color=qz.MAL, lw=1.4)
    a2.plot(q, q, color=qz.GREY, lw=0.7, ls=":")
    a2.plot([5], [12.5], "o", mfc="white", mec=qz.INK, ms=5, mew=1.0, zorder=5)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlim(1e-3, 1e3)
    a2.set_ylim(1e-3, 1e4)
    a2.set_xticks([1e-3, 1e-1, 1e1, 1e3])
    a2.set_yticks([1e-3, 1e-1, 1e1, 1e3])
    qz.label(a2, "Rock mass quality, $Q$ (-)", "Wall rating, $Q_{\\mathrm{wall}}$ (-)")
    fig.tight_layout()
    cols["Q_wall"] = qw
    _save(fig, "ch15", "q-span-wall", cols)


# ---------------------------------------------------------------------------
# Chapter 15: station cavern, roof support by the two systems
# ---------------------------------------------------------------------------

def station_cavern():
    span, wall, rise = 16.0, 6.0, 4.0
    R = (rise ** 2 + (span / 2) ** 2) / (2 * rise)
    yc = wall + rise - R
    half_ang = math.asin((span / 2) / R)
    fig, ax = qz.figure(W_FULL, 0.56 * W_FULL)
    ax.add_patch(Rectangle((-13, -1.0), 26, wall + rise + 7.5, facecolor=qz.GREY_LIGHT, alpha=0.3,
                           edgecolor="none", zorder=0))
    ang = np.linspace(-half_ang, half_ang, 200)
    xs = np.r_[-span / 2, -span / 2, R * np.sin(ang), span / 2, span / 2]
    ys = np.r_[0, wall, yc + R * np.cos(ang), wall, 0]
    ax.add_patch(Polygon(np.column_stack([xs, ys]), closed=True, facecolor="white", edgecolor=qz.INK, lw=1.0,
                         zorder=2))
    rows = []
    for side, L, sb, tsh, col in ((-1, 4.8, 1.0, 0.15, qz.CAR), (1, 4.0, 2.1, 0.12, qz.MAL)):
        # shotcrete band on this half of the arch, drawn to scale
        a = np.linspace(0, side * half_ang, 80)
        xo, yo = R * np.sin(a), yc + R * np.cos(a)
        xi, yi = (R - tsh) * np.sin(a), yc + (R - tsh) * np.cos(a)
        ax.add_patch(Polygon(np.column_stack([np.r_[xo, xi[::-1]], np.r_[yo, yi[::-1]]]), closed=True,
                             facecolor=col, edgecolor="none", zorder=3))
        # bolts normal to the arch at the stated spacing, from the crown outwards
        arc = R * half_ang
        s = sb / 2
        while s <= arc - 0.3:
            th = side * s / R
            x0, y0 = R * np.sin(th), yc + R * np.cos(th)
            x1, y1 = (R + L) * np.sin(th), yc + (R + L) * np.cos(th)
            ax.plot([x0, x1], [y0, y1], color=col, lw=1.0, zorder=4)
            ax.plot([x1], [y1], "o", color=col, ms=1.8, zorder=4)
            rows.append((side, s, L))
            s += sb
    ax.plot([0, 0], [0.0, wall + rise + 5.5], color=qz.INK, lw=0.5, ls=(0, (3, 2)), zorder=5)
    for side, col, head, l1, l2 in ((-1, qz.CAR, "RMR", "4.8 m at 1.0 m", "100 to 150 mm"),
                                    (1, qz.MAL, "Q", "4.0 m at 2.1 m", "90 to 120 mm")):
        xt, ha = side * 0.6, ("right" if side < 0 else "left")
        ax.text(xt, wall + 0.9, head, fontsize=9, color=col, ha=ha)
        ax.text(xt, wall - 0.5, l1, fontsize=8, ha=ha)
        ax.text(xt, wall - 1.7, l2, fontsize=8, ha=ha)
    _dim(ax, (-span / 2, -0.55), (span / 2, -0.55), "16 m", offset=(0, -0.55), fontsize=8)
    _dim(ax, (span / 2 + 0.7, 0), (span / 2 + 0.7, wall), "6 m", offset=(0.75, 0), fontsize=8)
    ax.set_xlim(-12.2, 12.2)
    ax.set_ylim(-1.6, wall + rise + 5.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "ch15", "station-cavern-support",
          {"side": [r[0] for r in rows], "arc_distance_m": [r[1] for r in rows], "bolt_length_m": [r[2] for r in rows]})


# ---------------------------------------------------------------------------
# Chapter 15: ratings, class boundaries and their uncertainty
# ---------------------------------------------------------------------------

def rating_uncertainty():
    fig, (a1, a2) = qz.figure(W_FULL, 0.62 * W_FULL, nrows=2, gridspec_kw={"height_ratios": [1, 1.1]})
    # RMR classes
    for i, (lo, hi, nm) in enumerate(((0, 20, "V"), (20, 40, "IV"), (40, 60, "III"), (60, 80, "II"),
                                      (80, 100, "I"))):
        a1.axvspan(lo, hi, color=qz.MAL_TINT if i % 2 == 0 else "white", lw=0, zorder=0)
        a1.text((lo + hi) / 2, 3.35, nm, ha="center", va="center", fontsize=8, color=qz.GREY)
    rows = [(2, 55, 5, "$55 \\pm 5$", qz.MAL), (1, 42, 5, "$42 \\pm 5$", qz.CAR)]
    for y, m, e, lab, col in rows:
        a1.errorbar([m], [y], xerr=[[e], [e]], fmt="o", color=col, mfc="white", ms=5, mew=1.0, capsize=3, lw=1.1)
        a1.text(m + e + 2, y, lab, va="center", fontsize=8)
    _arrow(a1, (50, 0), (38, 0), color=qz.CAR, lw=1.0, ms=7)
    a1.plot([50], [0], "o", color=qz.GREY, ms=4)
    a1.text(52, 0, "Station, 50 adjusted to 38", va="center", fontsize=8)
    a1.set_xlim(0, 100)
    a1.set_ylim(-0.7, 3.8)
    a1.set_yticks([])
    a1.tick_params(axis="x", top=False)
    for x in (20, 40, 60, 80):
        a1.axvline(x, color=qz.GREY_LIGHT, lw=0.6)
    qz.label(a1, "Rock mass rating, RMR (-)", None)

    qb = [1e-3, 0.01, 0.1, 1, 4, 10, 40, 100, 400, 1e3]
    names = ["Exc. poor", "Ext. poor", "Very poor", "Poor", "Fair", "Good", "Very good", "Ext. good", "Exc. good"]
    for i in range(9):
        a2.axvspan(qb[i], qb[i + 1], color=qz.MAL_TINT if i % 2 == 0 else "white", lw=0, zorder=0)
        a2.text(math.sqrt(qb[i] * qb[i + 1]), 2.75, names[i], ha="center", va="center", fontsize=7,
                color=qz.GREY, rotation=90)
        if i:
            a2.axvline(qb[i], color=qz.GREY_LIGHT, lw=0.6)
    a2.plot([0.009, 43], [0.6, 0.6], color=qz.GREY, lw=3.0, solid_capstyle="butt")
    a2.text(0.3, 0.6 - 0.45, "Car park cavern, 0.009 to 43", ha="left", va="center", fontsize=8)
    a2.errorbar([5], [1.5], xerr=[[2], [2]], fmt="o", color=qz.CAR, mfc="white", ms=5, mew=1.0, capsize=3, lw=1.1)
    a2.text(9.0, 1.5, "$5 \\pm 2$", va="center", fontsize=8)
    a2.set_xscale("log")
    a2.set_xlim(1e-3, 1e3)
    a2.set_ylim(-0.3, 3.9)
    a2.set_yticks([])
    a2.tick_params(axis="x", top=False, which="both")
    a2.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
    a2.set_xticklabels(["0.001", "0.01", "0.1", "1", "10", "100", "1000"])
    a2.xaxis.set_minor_locator(plt.NullLocator())
    qz.label(a2, "Rock mass quality, $Q$ (-)", None)
    fig.tight_layout()
    _save(fig, "ch15", "rating-uncertainty",
          {"case": ["RMR 55+-5", "RMR 42+-5", "Station RMR 50->38", "Q 5+-2", "Car park Q min", "Car park Q max"],
           "value": [55, 42, 38, 5, 0.009, 43], "minus": [5, 5, 12, 2, "", ""], "plus": [5, 5, 0, 2, "", ""]})


ALL = {"behaviour_map": behaviour_map, "voussoir": voussoir, "roof_bolting": roof_bolting,
       "stability_ratio": stability_ratio, "wedge_silo": wedge_silo_fig, "face_window": face_window,
       "spalling_notch": spalling_notch, "squeezing_allowance": squeezing_allowance,
       "burst_stiffness": burst_stiffness, "burst_energy": burst_energy, "spalling_depth": spalling_depth,
       "rock_mass_strength": rock_mass_strength, "bolt_length": bolt_length, "q_span_wall": q_span_wall,
       "station_cavern": station_cavern, "rating_uncertainty": rating_uncertainty}

if __name__ == "__main__":
    qz.use()
    wanted = sys.argv[1:]
    for name, fn in ALL.items():
        if not wanted or any(w in name for w in wanted):
            fn()
