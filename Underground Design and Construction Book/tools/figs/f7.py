"""Figures for Chapters 18 to 20: support, linings and seismic design.

Every plotted quantity follows from the chapters' own expressions and worked
values (Table 18.x of support characteristics, Appendix F for the lining
solutions, the Chapter 16 tunnel of radius 4.75 m), or from the published
relation the caption attributes. Schematics are drawn to plausible, internally
consistent dimensions and are captioned as schematic in the book.

    python tools/figs/f7.py            # all figures of the group
    python tools/figs/f7.py seismic    # figures whose name contains "seismic"
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
from matplotlib.patches import Polygon, Rectangle, Wedge, FancyArrowPatch, Circle  # noqa: E402

import qz_figure as qz  # noqa: E402
from geo_platform import ccm, lining  # noqa: E402

FIG = os.path.join(BOOK, "figures")
W_FULL = 117 / 25.4          # SNmono text width
W_HALF = 0.62 * W_FULL
A_TUNNEL = 4.75              # the Chapter 16 tunnel


def _save(fig, chapter: str, stem: str, columns: dict[str, np.ndarray]) -> None:
    out = os.path.join(FIG, chapter)
    qz.save(fig, stem, out)
    keys = list(columns)
    n = max(len(np.atleast_1d(columns[k])) for k in keys) if keys else 0
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


def _arrow(ax, x0, y0, x1, y1, color=qz.INK, lw=0.8, ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def _blank(ax):
    ax.set_aspect("equal")
    ax.axis("off")


# ===========================================================================
# Chapter 18
# ===========================================================================

def _bolt_scc(d_b=0.025, l_b=4.0, t_bf=0.2, e_b=200e3, s_c=1.5, s_l=1.5, q=0.0):
    """Point-anchored bolt pattern (Section 18.1.3): p_sm (MPa), k_s (MPa/m)."""
    psm = t_bf / (s_c * s_l)
    ks = 1.0 / (s_c * s_l) / (4 * l_b / (math.pi * d_b ** 2 * e_b) + q)
    return psm, ks


def _set_scc(a_s=50e-4, fy=275.0, e_s=200e3, s=1.0, a=A_TUNNEL):
    return a_s * fy / (s * a), e_s * a_s / (s * a ** 2)


def _ring_scc(t, fcc, e_c_gpa, nu_c=0.25, a=A_TUNNEL):
    s = ccm.support_ring(a, t, fcc, e_c_gpa, nu_c)
    return s.psm, s.ks


def bolt_load_transfer():
    """Axial force along three families of reinforcement crossing an opening joint (schematic)."""
    L, xj = 3.0, 1.2                     # bar length and joint position from the collar (m)
    x = np.linspace(0, L, 601)
    # continuously mechanically coupled: bond mobilised close to the joint and decaying along the bar
    t_cmc = np.exp(-np.abs(x - xj) / 0.18)
    # continuously frictionally coupled: uniform friction, force grows from each end towards the joint
    t_cfc = np.minimum(x / xj, (L - x) / (L - xj)) * 0.45
    # discretely mechanically coupled: tie between the plate and the anchor, uniform along the free length
    xa = 2.7
    t_dmc = np.where(x <= xa, 0.30, 0.30 * np.clip((L - x) / (L - xa), 0, 1))

    fig = plt.figure(figsize=(W_FULL, 0.42 * W_FULL))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # left: three bars in boreholes, collar at x = 0, joint at x = xj
    ys = (2.0, 1.0, 0.0)
    ax0.add_patch(Rectangle((0, -0.45), L + 0.3, 2.9, fc=qz.GREY_LIGHT, ec="none", alpha=0.45))
    ax0.plot([xj, xj], [-0.5, 2.5], color=qz.CAR, lw=1.0, ls="--")
    for y, kind in zip(ys, ("cmc", "cfc", "dmc")):
        ax0.add_patch(Rectangle((0, y - 0.13), L + 0.05, 0.26, fc="white", ec=qz.GREY, lw=0.5))
        if kind == "cmc":
            ax0.add_patch(Rectangle((0, y - 0.13), L, 0.26, fc=qz.MAL_TINT, ec=qz.GREY, lw=0.5))
            ax0.plot([0, L], [y, y], color=qz.MAL, lw=2.0, solid_capstyle="butt")
        elif kind == "cfc":
            ax0.plot([0, L], [y + 0.1, y + 0.1], color=qz.MAL, lw=1.0)
            ax0.plot([0, L], [y - 0.1, y - 0.1], color=qz.MAL, lw=1.0)
        else:
            ax0.plot([0, xa], [y, y], color=qz.MAL, lw=1.2)
            ax0.add_patch(Rectangle((xa, y - 0.12), L - xa, 0.24, fc=qz.MAL, ec="none"))
        ax0.add_patch(Rectangle((-0.06, y - 0.3), 0.06, 0.6, fc=qz.INK, ec="none"))   # plate
    for y, name in zip(ys, ("Grouted", "Friction", "Anchored")):
        ax0.text(L / 2 + 0.6, y + 0.2, name, ha="center", va="bottom", fontsize=8)
    ax0.text(xj, -0.55, "Joint", ha="center", va="top", fontsize=8, color=qz.CAR)
    ax0.text(-0.06, -0.55, "Plate", ha="center", va="top", fontsize=8)
    ax0.set_xlim(-0.45, L + 0.35)
    ax0.set_ylim(-0.95, 2.65)
    _blank(ax0)

    for t, col, ls, name in ((t_cmc, qz.MAL, "-", "Grouted"), (t_cfc, qz.CAR, "--", "Friction"),
                             (t_dmc, qz.GREY, "-.", "Anchored")):
        ax1.plot(x, t, color=col, ls=ls, lw=1.3, label=name)
    ax1.axvline(xj, color=qz.CAR, lw=0.6, ls=":")
    ax1.text(xj - 0.06, 1.08, "Joint", ha="right", va="center", fontsize=8, color=qz.CAR)
    ax1.set_xlim(0, L)
    ax1.set_ylim(0, 1.15)
    qz.label(ax1, "Distance from the collar, $x$ (m)", "Axial force, $T/T_{\\max}$ (-)")
    ax1.legend(loc="upper right")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.2, top=0.97)
    _save(fig, "ch18", "bolt-load-transfer", {"x_m": x, "T_grouted": t_cmc, "T_friction": t_cfc,
                                              "T_anchored": t_dmc})


def support_element_scc():
    """Support characteristic lines of the Chapter 18 elements and of shotcrete on sets."""
    elems = {
        "Bolts": _bolt_scc(),
        "Steel sets": _set_scc(),
        "Sprayed concrete": _ring_scc(0.2, 30, 25),
    }
    for k, (p, kk) in elems.items():
        print(f"  {k}: p_sm {p:.3f} MPa, k_s {kk:.1f} MPa/m, u_sm {p / kk * 1000:.1f} mm")
    fig, (a1, a2) = qz.figure(W_FULL, 0.52 * W_FULL, ncols=2)
    cols = {}
    styles ={"Bolts": (qz.GREY, "-."), "Steel sets": (qz.CAR, "--"), "Sprayed concrete": (qz.MAL, "-")}
    for name, (p, k) in elems.items():
        usm = p / k * 1000
        u = np.array([0, usm, 10.0])
        pp = np.array([0, p, p])
        col, ls = styles[name]
        a1.plot(u, pp, color=col, ls=ls, lw=1.3, label=name)
        a1.plot([usm], [p], "o", color=col, ms=3.5)
        cols[f"u_{name}"] = u
        cols[f"p_{name}"] = pp
    a1.set_xlim(0, 10)
    a1.set_ylim(0, 2.2)
    qz.label(a1, "Displacement, $u$ (mm)", "Support pressure, $p_i$ (MPa)")
    a1.legend(loc="upper left")

    ps, ks = elems["Sprayed concrete"]
    pst, kst = elems["Steel sets"]
    ktot = ks + kst
    umin = min(ps / ks, pst / kst) * 1000
    ptot = ktot * umin / 1000
    print(f"  combined: k {ktot:.1f}, u_sm {umin:.2f} mm, p_sm {ptot:.2f} MPa vs sum {ps + pst:.2f}")
    a2.plot([0, ps / ks * 1000], [0, ps], color=qz.MAL, lw=1.1, label="Sprayed concrete")
    a2.plot([0, pst / kst * 1000], [0, pst], color=qz.CAR, ls="--", lw=1.1, label="Steel sets")
    a2.plot([0, umin], [0, ptot], color=qz.INK, lw=1.5, label="Together")
    a2.plot([umin], [ptot], "o", mfc="white", mec=qz.INK, ms=5, mew=1.0)
    a2.axhline(ps + pst, color=qz.GREY, lw=0.7, ls=":")
    a2.set_xlim(0, 8)
    a2.set_ylim(0, 2.4)
    qz.label(a2, "Displacement, $u$ (mm)", "Support pressure, $p_i$ (MPa)")
    a2.legend(loc="upper left")
    cols.update({"combined_u_mm": np.array([0, umin]), "combined_p_MPa": np.array([0, ptot])})
    fig.tight_layout(w_pad=1.5)
    _save(fig, "ch18", "support-element-scc", cols)


def shotcrete_interaction():
    """Thrust-moment envelope of a plain sprayed section (rectangular block), nominal and thin."""
    fcc = 30.0                                         # MPa, as Table 18.x
    fig, ax = qz.figure(W_HALF, 0.86 * W_HALF)
    cols = {}
    for h, col, ls, name in ((0.20, qz.MAL, "-", "$t_c = 200$ mm"), (0.16, qz.CAR, "--", "$t_c = 160$ mm")):
        n = np.linspace(0, fcc * h, 300)               # MN per metre
        m = n * (h - n / fcc) / 2 * 1000               # kNm per metre
        ax.plot(n, m, color=col, ls=ls, lw=1.3, label=name)
        cols[f"N_h{h * 1000:.0f}_MN"] = n
        cols[f"M_h{h * 1000:.0f}_kNm"] = m
    for e, n_end, (dx, dy) in ((0.02, 6.5, (-0.1, 10)), (0.05, 3.4, (0.1, 0))):
        nn = np.linspace(0, n_end, 50)
        ax.plot(nn, nn * e * 1000, color=qz.GREY, lw=0.6, ls=":")
        ax.text(n_end + dx, n_end * e * 1000 + dy, f"$e = {e * 1000:.0f}$ mm", fontsize=8, color=qz.GREY,
                ha="left" if dx > 0 else "right", va="center" if dx > 0 else "bottom")
        cols[f"M_e{e * 1000:.0f}mm_kNm"] = nn * e * 1000
    # thrust from a uniform pressure of about 0.63 MPa (shotcrete alone, GSI 60 ground of Chapter 16)
    n_d = 0.63 * A_TUNNEL
    ax.axvline(n_d, color=qz.INK, lw=0.6, ls="-.")
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 230)
    qz.label(ax, "Thrust, $N$ (MN/m)", "Moment, $M$ (kNm/m)")
    ax.legend(loc="upper right", borderaxespad=0.3)
    fig.tight_layout()
    _save(fig, "ch18", "shotcrete-interaction", cols)


def shotcrete_scc_age():
    """Support line of 200 mm of sprayed concrete at 28 days, at one day, and 40 mm thin."""
    cases = (("200 mm, 28 days", 0.20, 30, 25, qz.MAL, "-"),
             ("160 mm, 28 days", 0.16, 30, 25, qz.CAR, "--"),
             ("200 mm, 1 day", 0.20, 20, 10, qz.GREY, "-."))
    fig, ax = qz.figure(W_HALF, 0.86 * W_HALF)
    cols = {}
    for name, t, f, e, col, ls in cases:
        p, k = _ring_scc(t, f, e)
        usm = p / k * 1000
        print(f"  {name}: p_sm {p:.2f}, k_s {k:.1f}, u_sm {usm:.1f}")
        u = np.array([0, usm, 12])
        pp = np.array([0, p, p])
        ax.plot(u, pp, color=col, ls=ls, lw=1.3, label=name)
        ax.plot([usm], [p], "o", color=col, ms=3.5)
        cols[f"u_{name}"] = u
        cols[f"p_{name}"] = pp
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.6)
    qz.label(ax, "Displacement, $u$ (mm)", "Support pressure, $p_i$ (MPa)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "ch18", "shotcrete-scc-age", cols)


def yielding_support():
    """GSI 30 ground of Chapter 16: a stiff shell against a shell with yielding elements."""
    g = ccm.Ground(16.9, A_TUNNEL, 0.22, 2.12, 23.8, 1.72)
    res = ccm.analyse(g, [ccm.support_ring(A_TUNNEL, 0.5, 60, 40, 0.25)])
    us0 = res.us0
    p, u = ccm.grc_curve(g)
    ok = np.isfinite(u)
    ps, ks = _ring_scc(0.2, 30, 25)
    u_stiff = us0 + ps / ks * 1000
    p_y, slot = 0.5, 60.0                      # illustrative yield pressure (MPa) and radial shortening (mm)
    u_y = us0 + p_y / ks * 1000
    u_eq = ccm.grc_displacement(g, p_y)
    u_close = u_y + slot
    u_cap = u_close + (ps - p_y) / ks * 1000          # the closed shell reaches its own capacity
    print(f"  u_s0 {us0:.1f} mm; stiff shell limit at {u_stiff:.1f} mm where ground needs "
          f"{np.interp(u_stiff, u[ok], p[ok]):.2f} MPa; yielding eq at {u_eq:.1f} mm")
    fig, ax = qz.figure(W_FULL, 0.5 * W_FULL)
    ax.plot(u[ok], p[ok], color=qz.MAL, lw=1.4, label="Ground reaction, GSI 30")
    ax.plot([us0, u_stiff], [0, ps], color=qz.CAR, lw=1.3, ls="--", label="Stiff shell")
    ax.plot([u_stiff], [ps], "x", color=qz.CAR, ms=6, mew=1.4)
    yu = np.array([us0, u_y, u_close, u_cap, 140.0])
    yp = np.array([0, p_y, p_y, ps, ps])
    ax.plot(yu, yp, color=qz.INK, lw=1.3, label="Shell with yielding elements")
    ax.plot([u_eq], [p_y], "o", mfc="white", mec=qz.INK, ms=5, mew=1.0)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 4)
    qz.label(ax, "Wall displacement, $u_r$ (mm)", "Support pressure, $p_i$ (MPa)")
    ax.annotate("Elements closed", xy=(u_close, p_y), xytext=(u_close - 1, 1.75), fontsize=8, ha="center",
                va="bottom", arrowprops=dict(arrowstyle="-", color=qz.GREY, lw=0.5, shrinkA=1, shrinkB=3))
    ax.text(us0 - 1.5, 0.08, "$u_{s0}$", fontsize=9, ha="right", va="bottom")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "ch18", "yielding-support", {"u_grc_mm": u[ok], "p_grc_MPa": p[ok], "u_yield_mm": yu,
                                            "p_yield_MPa": yp, "u_stiff_mm": np.array([us0, u_stiff]),
                                            "p_stiff_MPa": np.array([0, ps])})


# ===========================================================================
# Chapter 19
# ===========================================================================

SEG = dict(radius=3.4, angle=67.5, width=1.5, thickness=0.3)   # the companion's default segment


def ring_build():
    """A 5 + key ring in section, and the developed view of four staggered rings (schematic)."""
    r, t = SEG["radius"], SEG["thickness"]
    ri, ro = r - t / 2, r + t / 2
    seg_deg, key_deg = SEG["angle"], 360 - 5 * SEG["angle"]
    fig = plt.figure(figsize=(W_FULL, 0.44 * W_FULL))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.25], wspace=0.12)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # left: section, key at the crown
    start = 90 + key_deg / 2
    edges = [90 - key_deg / 2, 90 + key_deg / 2]
    ax0.add_patch(Wedge((0, 0), ro, 90 - key_deg / 2, 90 + key_deg / 2, width=t, fc=qz.CAR_TINT, ec=qz.CAR, lw=0.9))
    for i in range(5):
        a0 = start + i * seg_deg
        ax0.add_patch(Wedge((0, 0), ro, a0, a0 + seg_deg, width=t, fc="white", ec=qz.INK, lw=0.7))
        edges.append(a0)
    ax0.text(0, ro + 0.12, "K", ha="center", va="bottom", fontsize=8, color=qz.CAR)
    # dimension of one segment angle
    a0, a1 = start + seg_deg, start + 2 * seg_deg
    th = np.radians(np.linspace(a0, a1, 50))
    rd = ri - 0.55
    ax0.plot(rd * np.cos(th), rd * np.sin(th), color=qz.GREY, lw=0.6)
    for a in (a0, a1):
        ax0.plot([ri * math.cos(math.radians(a)), (rd - 0.15) * math.cos(math.radians(a))],
                 [ri * math.sin(math.radians(a)), (rd - 0.15) * math.sin(math.radians(a))], color=qz.GREY, lw=0.4)
    am = math.radians((a0 + a1) / 2)
    ax0.text((rd - 0.15) * math.cos(am), (rd - 0.15) * math.sin(am), "$67.5^\\circ$", ha="left", va="center",
             fontsize=8, color=qz.GREY)
    ax0.set_xlim(-ro - 0.1, ro + 0.1)
    ax0.set_ylim(-ro - 0.1, ro + 0.6)
    _blank(ax0)

    # right: developed view, position measured clockwise from the crown, rings rotated between builds
    rots = (22.5, 337.5, 292.5, 180.0)       # no radial joint within 22.5 deg of the neighbouring ring's
    cols = {"ring": [], "joint_deg": []}
    b = SEG["width"]
    for i, rot in enumerate(rots):
        y0 = i * b
        joints = [(rot - key_deg / 2) % 360, (rot + key_deg / 2) % 360]
        joints += [(rot + key_deg / 2 + k * seg_deg) % 360 for k in range(1, 5)]
        k0, k1 = (rot - key_deg / 2) % 360, (rot + key_deg / 2) % 360
        spans = [(k0, k1)] if k0 < k1 else [(k0, 360), (0, k1)]
        for s0, s1 in spans:
            ax1.add_patch(Rectangle((s0, y0), s1 - s0, b, fc=qz.CAR_TINT, ec="none"))
        ax1.add_patch(Rectangle((0, y0), 360, b, fc="none", ec=qz.INK, lw=0.7))
        for j in joints:
            ax1.plot([j, j], [y0, y0 + b], color=qz.CAR if j in (k0, k1) else qz.INK, lw=0.9)
            cols["ring"].append(i + 1)
            cols["joint_deg"].append(j)
        kc = (k0 + key_deg / 2) % 360
        ax1.text(kc, y0 + b / 2, "K", ha="center", va="center", fontsize=8, color=qz.CAR)
    ax1.set_xlim(0, 360)
    ax1.set_ylim(0, len(rots) * b)
    ax1.set_xticks([0, 90, 180, 270, 360])
    ax1.set_yticks([(i + 0.5) * b for i in range(len(rots))])
    ax1.set_yticklabels([str(i + 1) for i in range(len(rots))])
    ax1.tick_params(axis="y", length=0)
    ax1.yaxis.set_ticks_position("left")
    qz.label(ax1, "Position from the crown, $\\theta$ ($^\\circ$)", "Ring (-)")
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.2, top=0.97)
    _save(fig, "ch19", "ring-build", {k: np.array(v) for k, v in cols.items()})


def segment_actions():
    seg = lining.Segment(SEG["radius"], SEG["angle"], SEG["width"], SEG["thickness"], 25.0, 0.0, 35.0)
    act = lining.segment_actions(seg)
    sts = lining.segment_stresses(seg, act)
    s = act["s"]
    print(f"  q {seg.q:.2f} kN/m, chord {seg.chord:.2f} m, reaction {act['reaction'][0]:.2f} kN, "
          f"M max {act['M'].max():.2f} kNm, N end {act['N'][0]:.2f} kN, "
          f"sigma_t {max((-sts['intrados']).max(), (-sts['extrados']).max()):.3f} MPa, "
          f"delta {lining.crown_deflection_mm(seg, act):.3f} mm")
    fig, (a1, a2) = qz.figure(W_FULL, 0.42 * W_FULL, ncols=2)
    a1.plot(s, act["N"], color=qz.MAL, lw=1.3, label="Axial, $N$")
    a1.plot(s, act["V"], color=qz.CAR, lw=1.3, ls="--", label="Shear, $V$")
    a1.axhline(0, color=qz.GREY_LIGHT, lw=0.6)
    a1.set_xlim(0, s[-1])
    qz.label(a1, "Arc length, $s$ (m)", "Force, $N$, $V$ (kN)")
    a1.legend(loc="lower left")
    a2.plot(s, act["M"], color=qz.MAL, lw=1.3)
    a2.axhline(0, color=qz.GREY_LIGHT, lw=0.6)
    a2.set_xlim(0, s[-1])
    a2.set_ylim(0, None)
    qz.label(a2, "Arc length, $s$ (m)", "Moment, $M$ (kNm)")
    fig.tight_layout()
    _save(fig, "ch19", "segment-actions", {"s_m": s, "N_kN": act["N"], "V_kN": act["V"], "M_kNm": act["M"],
                                           "sigma_intrados_MPa": sts["intrados"],
                                           "sigma_extrados_MPa": sts["extrados"]})


def _es_coeffs(c, f, nu):
    a0 = c * f * (1 - nu) / (c + f + c * f * (1 - nu))
    a2 = (f + 6) * (1 - nu) / (2 * f * (1 - nu) + 6 * (5 - 6 * nu))
    return a0, a2


def lining_ratios():
    """Einstein and Schwartz full-slip coefficients against F and C, with the Chapter 16 ring marked."""
    rings = {}
    for gsi, em in ((60, 13520.0), (30, 2120.0)):
        rr = lining.ring_ratios(lining.Ring(A_TUNNEL, 0.5, 40.0, 0.25, em, 0.22))
        rings[gsi] = rr
        print(f"  GSI {gsi}: C {rr['C']:.3f}, F {rr['F']:.0f}")
    F = np.logspace(-1, 4, 300)
    C = np.logspace(-2, 2, 300)
    fig, (a1, a2) = qz.figure(W_FULL, 0.52 * W_FULL, ncols=2)
    cols = {"F": F, "C": C}
    for nu, col, ls in ((0.22, qz.MAL, "-"), (0.40, qz.CAR, "--")):
        _, a2s = _es_coeffs(1.0, F, nu)
        a1.plot(F, 1 - 2 * a2s, color=col, ls=ls, lw=1.3, label=f"$\\nu = {nu:g}$")
        a0s, _ = _es_coeffs(C, 1e9, nu)
        a2.plot(C, 1 - a0s, color=col, ls=ls, lw=1.3, label=f"$\\nu = {nu:g}$")
        cols[f"moment_coeff_nu{nu:g}"] = 1 - 2 * a2s
        cols[f"thrust_coeff_nu{nu:g}"] = 1 - a0s
    for gsi, mk in ((60, "o"), (30, "s")):
        rr = rings[gsi]
        _, a2s = _es_coeffs(rr["C"], rr["F"], 0.22)
        a0s, _ = _es_coeffs(rr["C"], rr["F"], 0.22)
        a1.plot([rr["F"]], [1 - 2 * a2s], mk, mfc="white", mec=qz.INK, ms=4.5, mew=0.9, clip_on=False, zorder=5)
        a2.plot([rr["C"]], [1 - a0s], mk, mfc="white", mec=qz.INK, ms=4.5, mew=0.9, clip_on=False, zorder=5)
        print(f"  GSI {gsi}: moment coeff {1 - 2 * a2s:.4f}, thrust coeff {1 - a0s:.3f}")
    a1.set_xscale("log")
    a1.set_xlim(0.1, 1e4)
    a1.set_ylim(0, 0.7)
    qz.label(a1, "Flexibility ratio, $F$ (-)", "Moment ratio, $1-2a_2^{*}$ (-)")
    a1.legend(loc="upper right")
    a2.set_xscale("log")
    a2.set_xlim(0.01, 100)
    a2.set_ylim(0, 1.05)
    qz.label(a2, "Compressibility ratio, $C$ (-)", "Thrust ratio, $1-a_0^{*}$ (-)")
    a2.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, "ch19", "lining-ratios", cols)


def joint_stiffness():
    """Radial joint in section (schematic) and the rotational stiffness of Janssen's model."""
    fig = plt.figure(figsize=(W_FULL, 0.42 * W_FULL))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.2], wspace=0.25)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # left: two segment ends meeting at a radial joint; extrados at top
    t, hj = 1.0, 0.55
    gap = 0.06
    rec = (1 - hj) / 2 - 0.08          # contact band slightly towards the intrados
    for side in (-1, 1):
        x_in = side * gap / 2
        x_out = side * 1.0
        # outline with chamfers / recesses either side of the contact band
        y_c0, y_c1 = rec, rec + hj
        xs = [x_out, x_in + side * 0.10, x_in + side * 0.10, x_in, x_in, x_in + side * 0.10,
              x_in + side * 0.10, x_out]
        ys = [0, 0, y_c0 - 0.0, y_c0, y_c1, y_c1, t, t]
        ax0.add_patch(Polygon(list(zip(xs, ys)), closed=True, fc="white", ec=qz.INK, lw=0.8))
        # gasket groove near the extrados
        ax0.add_patch(Rectangle((x_in + side * 0.10 - (0.07 if side > 0 else 0), 0.82), 0.07, 0.09,
                                fc=qz.GREY, ec="none"))
    ax0.add_patch(Rectangle((-gap / 2, rec), gap, hj, fc=qz.MAL, ec="none"))
    ax0.text(0, 1.06, "Gasket", ha="center", va="bottom", fontsize=8, color=qz.GREY)
    ax0.plot([0, 0], [0.93, 1.05], color=qz.GREY, lw=0.5)
    # dimension of the contact band
    xd = 0.35
    ax0.annotate("", xy=(xd, rec), xytext=(xd, rec + hj),
                 arrowprops=dict(arrowstyle="<->", color=qz.GREY, lw=0.6, shrinkA=0, shrinkB=0))
    ax0.text(xd + 0.06, rec + hj / 2, "$h_j$", va="center", fontsize=9)
    ax0.annotate("", xy=(-0.6, 0), xytext=(-0.6, t),
                 arrowprops=dict(arrowstyle="<->", color=qz.GREY, lw=0.6, shrinkA=0, shrinkB=0))
    ax0.text(-0.54, t / 2, "$t_c$", va="center", ha="left", fontsize=9)
    # thrust at an eccentricity
    e = 0.17
    for side in (-1, 1):
        _arrow(ax0, side * 1.45, rec + hj / 2 + e, side * 1.05, rec + hj / 2 + e, color=qz.CAR, lw=0.9)
    ax0.text(1.45, rec + hj / 2 + e + 0.08, "$N$", ha="right", va="bottom", fontsize=9, color=qz.CAR)
    ax0.set_xlim(-1.5, 1.5)
    ax0.set_ylim(-0.12, 1.3)
    _blank(ax0)

    m = np.linspace(0, 0.5, 400)
    k = np.where(m <= 1 / 6, 1.0, 13.5 * m * (1 - 2 * m) ** 2)
    ax1.plot(m, k, color=qz.MAL, lw=1.4)
    ax1.axvline(1 / 6, color=qz.GREY, lw=0.6, ls=":")
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks([0, 1 / 6, 0.25, 1 / 3, 0.5])
    ax1.set_xticklabels(["0", "1/6", "1/4", "1/3", "1/2"])
    qz.label(ax1, "Eccentricity, $e/h_j$ (-)", "Stiffness ratio, $k_\\varphi/k_{\\varphi 0}$ (-)")
    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.2, top=0.97)
    _save(fig, "ch19", "joint-stiffness", {"e_over_hj": m, "k_over_k0": k})


def drained_tanked():
    """Half-sections of a drained and a tanked lining under a water table (schematic)."""
    r_i, r_o = 1.0, 1.12
    z_w = 2.1                                   # water table above the centre, in radii
    fig, ax = plt.subplots(figsize=(W_HALF, 0.86 * W_HALF))
    # lining annulus, both halves
    ax.add_patch(Wedge((0, 0), r_o, 90, 270, width=r_o - r_i, fc=qz.GREY_LIGHT, ec=qz.INK, lw=0.7))
    ax.add_patch(Wedge((0, 0), r_o, -90, 90, width=r_o - r_i, fc=qz.GREY_LIGHT, ec=qz.INK, lw=0.7))
    ax.plot([0, 0], [-1.6, 2.35], color=qz.GREY, lw=0.5, ls=(0, (4, 3)))
    # drained (left): drainage layer, invert drain, water led round the lining
    th = np.radians(np.linspace(95, 255, 120))
    ax.plot(1.2 * np.cos(th), 1.2 * np.sin(th), color=qz.MAL, lw=1.6, ls=(0, (1, 1.2)))
    ax.add_patch(Circle((1.2 * math.cos(math.radians(250)), 1.2 * math.sin(math.radians(250))), 0.09,
                        fc="white", ec=qz.MAL, lw=1.0))
    for a in (130, 165, 200):
        a0, a1 = math.radians(a), math.radians(a + 22)
        _arrow(ax, 1.36 * math.cos(a0), 1.36 * math.sin(a0), 1.36 * math.cos(a1), 1.36 * math.sin(a1),
               color=qz.MAL, lw=0.7, ms=6)
    # tanked (right): membrane and hydrostatic pressure normal to the lining
    th = np.radians(np.linspace(-90, 90, 120))
    ax.plot(1.16 * np.cos(th), 1.16 * np.sin(th), color=qz.CAR, lw=1.3)
    scale = 0.17
    for a in np.linspace(-80, 80, 9):
        c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
        head = z_w - 1.16 * s
        L = scale * head
        _arrow(ax, (1.18 + L) * c, (1.18 + L) * s, 1.18 * c, 1.18 * s, color=qz.CAR, lw=0.8, ms=6)
    # water table
    ax.plot([-1.8, 1.9], [z_w, z_w], color=qz.GREY, lw=0.7)
    ax.add_patch(Polygon([(1.45, z_w), (1.35, z_w + 0.13), (1.55, z_w + 0.13)], closed=True, fc="white",
                         ec=qz.GREY, lw=0.7))
    def _lead(txt, xt, yt, ang, r, col, ha):
        ax.annotate(txt, xy=(r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang))), xytext=(xt, yt),
                    fontsize=8, color=col, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5, shrinkA=1, shrinkB=1))
    _lead("Drainage layer", -0.75, 1.62, 108, 1.2, qz.MAL, "right")
    _lead("Invert drain", -1.0, -1.42, 250, 1.29, qz.MAL, "right")
    _lead("Membrane", 0.95, 1.62, 72, 1.16, qz.CAR, "left")
    ax.text(-0.95, -1.75, "Drained", ha="center", va="top", fontsize=9)
    ax.text(0.95, -1.75, "Tanked", ha="center", va="top", fontsize=9)
    ax.set_xlim(-1.9, 2.0)
    ax.set_ylim(-2.0, 2.4)
    _blank(ax)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    _save(fig, "ch19", "drained-tanked", {"water_table_over_a": np.array([z_w]),
                                          "pressure_scale": np.array([scale])})


def carbonation_cover():
    """Front depth x = K sqrt(t), normalised by the nominal cover and its time to reach it."""
    tt = np.linspace(0, 1.6, 400)
    x = np.sqrt(tt)
    fig, ax = qz.figure(W_HALF, 0.78 * W_HALF)
    ax.plot(tt, x, color=qz.MAL, lw=1.4)
    for c, col, ls in ((1.0, qz.MAL, "--"), (0.8, qz.CAR, "--")):
        ti = c ** 2
        ax.plot([0, ti], [c, c], color=col, lw=0.8, ls=ls)
        ax.plot([ti, ti], [0, c], color=col, lw=0.8, ls=ls)
        ax.plot([ti], [c], "o", color=col, ms=4)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.3)
    ax.set_xticks([0, 0.4, 0.64, 1.0, 1.2, 1.6])
    ax.set_xticklabels(["0", "0.4", "0.64", "1", "1.2", "1.6"])
    ax.set_yticks([0, 0.4, 0.8, 1.0, 1.2])
    ax.set_yticklabels(["0", "0.4", "0.8", "1", "1.2"])
    qz.label(ax, "Time, $t/t_{\\mathrm{nom}}$ (-)", "Front depth, $x/c_{\\mathrm{nom}}$ (-)")
    fig.tight_layout()
    _save(fig, "ch19", "carbonation-cover", {"t_over_tnom": tt, "x_over_cnom": x})


# ===========================================================================
# Chapter 20
# ===========================================================================

def seismic_modes():
    """Axial, curvature, ovaling and racking deformation of a tunnel (schematic, exaggerated)."""
    fig, axes = plt.subplots(1, 4, figsize=(W_FULL, 0.26 * W_FULL),
                             gridspec_kw={"width_ratios": [1.35, 1.35, 1, 1], "wspace": 0.12})
    # axial: plan of a tube whose ring lines bunch in compression and spread in extension
    ax = axes[0]
    x = np.linspace(0, 4, 200)
    ax.plot(x, 0.35 + 0 * x, color=qz.INK, lw=0.8)
    ax.plot(x, -0.35 + 0 * x, color=qz.INK, lw=0.8)
    xr0 = np.linspace(0.1, 3.9, 17)
    xr = xr0 + 0.13 * np.sin(2 * np.pi * xr0 / 4)
    for xi in xr:
        ax.plot([xi, xi], [-0.35, 0.35], color=qz.MAL, lw=0.8)
    ax.plot(x, -0.75 + 0.18 * np.cos(2 * np.pi * x / 4), color=qz.CAR, lw=0.8)
    ax.text(2.0, -1.0, "Axial strain", ha="center", va="top", fontsize=7, color=qz.CAR)
    _arrow(ax, 3.3, 0.6, 4.0, 0.6, color=qz.GREY, lw=0.8, ms=6)
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(-1.1, 1.1)
    _blank(ax)
    # curvature: centreline follows the wave
    ax = axes[1]
    yc = 0.35 * np.sin(2 * np.pi * x / 4)
    ax.plot(x, yc + 0.3, color=qz.MAL, lw=0.9)
    ax.plot(x, yc - 0.3, color=qz.MAL, lw=0.9)
    ax.plot(x, 0.3 + 0 * x, color=qz.GREY_LIGHT, lw=0.7, ls="--")
    ax.plot(x, -0.3 + 0 * x, color=qz.GREY_LIGHT, lw=0.7, ls="--")
    _arrow(ax, 3.3, 0.85, 4.0, 0.85, color=qz.GREY, lw=0.8, ms=6)
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(-1.1, 1.1)
    _blank(ax)
    # ovaling: circle to ellipse along the diagonal under shear
    ax = axes[2]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.62 * np.cos(th), 0.62 * np.sin(th), color=qz.GREY_LIGHT, lw=0.7, ls="--")
    ea, eb, rot = 0.74, 0.5, math.radians(45)
    ex, ey = ea * np.cos(th), eb * np.sin(th)
    ax.plot(ex * math.cos(rot) - ey * math.sin(rot), ex * math.sin(rot) + ey * math.cos(rot), color=qz.MAL, lw=1.0)
    _arrow(ax, -0.5, 0.95, 0.5, 0.95, color=qz.CAR, lw=0.8, ms=6)
    _arrow(ax, 0.5, -0.95, -0.5, -0.95, color=qz.CAR, lw=0.8, ms=6)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.1, 1.1)
    _blank(ax)
    # racking: rectangle to parallelogram
    ax = axes[3]
    w, h, d = 1.2, 0.8, 0.18
    ax.add_patch(Rectangle((-w / 2, -h / 2), w, h, fc="none", ec=qz.GREY_LIGHT, lw=0.7, ls="--"))
    ax.add_patch(Polygon([(-w / 2 - d / 2, -h / 2), (w / 2 - d / 2, -h / 2), (w / 2 + d / 2, h / 2),
                          (-w / 2 + d / 2, h / 2)], closed=True, fc="none", ec=qz.MAL, lw=1.0))
    _arrow(ax, -0.5, 0.95, 0.5, 0.95, color=qz.CAR, lw=0.8, ms=6)
    _arrow(ax, 0.5, -0.95, -0.5, -0.95, color=qz.CAR, lw=0.8, ms=6)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.1, 1.1)
    _blank(ax)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.86)
    for ax, name in zip(axes, ("Axial", "Curvature", "Ovaling", "Racking")):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.88, name, ha="center", va="bottom", fontsize=9)
    _save(fig, "ch20", "seismic-modes", {"ring_x": xr})


def free_field_strain():
    """Newmark's free-field axial strain against incidence and against particle velocity."""
    theta = np.linspace(0, 90, 361)
    v = 0.3
    fig, (a1, a2) = qz.figure(W_FULL, 0.52 * W_FULL, ncols=2)
    cols = {"theta_deg": theta}
    sty = ((500.0, qz.CAR, "--"), (1000.0, qz.MAL, "-"), (2000.0, qz.GREY, "-."))
    for cs, col, ls in sty:
        eps = v / cs * np.sin(np.radians(theta)) * np.cos(np.radians(theta))
        a1.plot(theta, eps * 1e4, color=col, ls=ls, lw=1.3, label=f"$C_s = {cs:.0f}$ m/s")
        cols[f"eps_Cs{cs:.0f}"] = eps
        a1.text(45, eps.max() * 1e4 + 0.09, f"$C_s = {cs:.0f}$ m/s", ha="center", va="bottom", fontsize=8,
                color=col)
    a1.plot([45], [v / 2000 * 1e4], "o", color=qz.MAL, ms=4)
    a1.set_xlim(0, 90)
    a1.set_xticks([0, 15, 30, 45, 60, 75, 90])
    a1.set_ylim(0, 3.6)
    qz.label(a1, "Angle to the tunnel axis, $\\theta$ ($^\\circ$)", "Axial strain, $10^4\\,\\varepsilon_a$ (-)")
    vv = np.logspace(-2, 0, 100)
    cols["v_m_s"] = vv
    for cs, col, ls in sty:
        a2.loglog(vv, vv / (2 * cs), color=col, ls=ls, lw=1.3, label=f"$C_s = {cs:.0f}$ m/s")
        cols[f"epsmax_Cs{cs:.0f}"] = vv / (2 * cs)
    a2.plot([0.3], [1.5e-4], "o", color=qz.MAL, ms=4)
    a2.set_xlim(0.01, 1)
    a2.set_ylim(1e-6, 1e-3)
    qz.label(a2, "Particle velocity, $v_{\\max}$ (m/s)", "Peak axial strain, $\\varepsilon_{a,\\max}$ (-)")
    a2.legend(loc="lower right", handlelength=1.8)
    fig.tight_layout(w_pad=1.2)
    _save(fig, "ch20", "free-field-strain", cols)


def ovaling_flexibility():
    """Wang's full-slip ovaling solution: lining to free-field distortion, and the moment coefficient."""
    F = np.logspace(-1, 4, 400)               # the book's F (Appendix F)
    fig, (a1, a2) = qz.figure(W_FULL, 0.52 * W_FULL, ncols=2)
    cols = {"F": F}
    for nu, col, ls in ((0.25, qz.MAL, "-"), (0.40, qz.CAR, "--")):
        fw = F * (1 - nu) / 6                 # Wang's ratio: E(1-nu_l^2)R^3 / (6 E_l I (1+nu))
        k1 = 12 * (1 - nu) / (2 * fw + 5 - 6 * nu)
        ratio = 2 / 3 * k1 * fw
        a1.semilogx(F, ratio, color=col, ls=ls, lw=1.3, label=f"$\\nu = {nu:g}$")
        a1.axhline(4 * (1 - nu), color=col, lw=0.5, ls=":")
        a2.loglog(F, k1 / 6, color=col, ls=ls, lw=1.3, label=f"$\\nu = {nu:g}$")
        cols[f"ratio_nu{nu:g}"] = ratio
        cols[f"K1_over_6_nu{nu:g}"] = k1 / 6
        print(f"  nu {nu}: ratio 1 at F = {6 / (1 - nu) * (5 - 6 * nu) / (8 * (1 - nu) - 2):.1f}")
    a1.axhline(1, color=qz.GREY, lw=0.6)
    a1.set_xlim(0.1, 1e4)
    a1.set_ylim(0, 3.4)
    qz.label(a1, "Flexibility ratio, $F$ (-)", "Distortion, $\\Delta D_l/\\Delta D_{ff}$ (-)")
    a1.legend(loc="lower right")
    a2.set_xlim(0.1, 1e4)
    qz.label(a2, "Flexibility ratio, $F$ (-)", "Moment coefficient, $K_1/6$ (-)")
    fig.tight_layout()
    _save(fig, "ch20", "ovaling-flexibility", cols)


def rayleigh_damping():
    xi_t, f1, f2 = 0.05, 1.0, 8.0                  # illustrative target and bracketing frequencies
    w1, w2 = 2 * math.pi * f1, 2 * math.pi * f2
    a_r = 2 * xi_t * w1 * w2 / (w1 + w2)
    b_r = 2 * xi_t / (w1 + w2)
    f = np.logspace(-1, 2, 400)
    w = 2 * np.pi * f
    xm, xk = a_r / (2 * w), b_r * w / 2
    fig, ax = qz.figure(W_HALF, 0.78 * W_HALF)
    ax.semilogx(f, (xm + xk) * 100, color=qz.MAL, lw=1.4, label="Rayleigh")
    ax.semilogx(f, xm * 100, color=qz.GREY, lw=0.8, ls="--", label="Mass part")
    ax.semilogx(f, xk * 100, color=qz.GREY, lw=0.8, ls=":", label="Stiffness part")
    ax.axhline(xi_t * 100, color=qz.CAR, lw=0.7, ls="-.", label="Target")
    ax.plot([f1, f2], [5, 5], "o", color=qz.CAR, ms=4)
    ax.set_xlim(0.3, 30)
    ax.set_ylim(0, 30)
    qz.label(ax, "Frequency, $f$ (Hz)", "Damping ratio, $\\xi$ (%)")
    ax.legend(loc="upper center", ncol=2, columnspacing=0.8, handlelength=1.8)
    fig.tight_layout()
    _save(fig, "ch20", "rayleigh-damping", {"f_Hz": f, "xi_total": xm + xk, "xi_mass": xm, "xi_stiffness": xk,
                                            "a_R": np.array([a_r]), "b_R": np.array([b_r])})


def fault_crossing():
    """Four ways of accommodating fault displacement along a tunnel (schematic long sections)."""
    fig, axes = plt.subplots(4, 1, figsize=(W_FULL, 0.62 * W_FULL), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    x = np.linspace(0, 12, 600)
    z0, z1 = 5.0, 7.0                            # fault zone
    d = 0.45                                     # offset across the zone
    h = 0.5                                      # half-height of the tunnel
    names = ("Oversize", "Articulate", "Ductile", "Accept and repair")
    for ax, name in zip(axes, names):
        ax.add_patch(Rectangle((z0, -1.4), z1 - z0, 2.5, fc=qz.GREY_LIGHT, ec="none", alpha=0.6))
        ax.plot([5.9, 6.1], [-1.4, 1.1], color=qz.CAR, lw=0.9)        # rupture plane
        ax.text(-0.15, 0, name, ha="right", va="center", fontsize=8)
        ax.set_xlim(0, 12)
        ax.set_ylim(-1.4, 1.1)
        ax.axis("off")
    step = np.where(x < 6.0, 0.0, -d)
    # oversize: an enlarged section spanning the zone and margins; the envelope survives the offset
    ax = axes[0]
    xa, xb = z0 - 0.8, z1 + 0.8
    big = np.where((x > xa) & (x < xb), 0.85, h)
    ax.plot(x, big + step, color=qz.MAL, lw=1.0)
    ax.plot(x, -big + step, color=qz.MAL, lw=1.0)
    # the operating envelope, realigned through the enlarged length after the offset
    env = -d * np.clip((x - xa) / (xb - xa), 0, 1)
    ax.plot(x, 0.4 + env, color=qz.INK, lw=0.6, ls="--")
    ax.plot(x, -0.4 + env, color=qz.INK, lw=0.6, ls="--")
    # articulate: short rings with flexible joints in the zone; the offset taken at a joint
    ax = axes[1]
    ax.plot(x, h + step, color=qz.MAL, lw=1.0)
    ax.plot(x, -h + step, color=qz.MAL, lw=1.0)
    for xj in np.arange(z0, z1 + 0.01, 0.5):
        off = 0 if xj < 6.0 else -d
        ax.plot([xj, xj], [-h + off, h + off], color=qz.CAR, lw=1.2)
    ax.plot([6.0, 6.0], [-h - d, h], color=qz.CAR, lw=1.2)
    # ductile: a continuous lining that spreads the offset over the zone
    ax = axes[2]
    smooth = -d * 0.5 * (1 + np.tanh((x - 6.0) / 0.45))
    ax.plot(x, h + smooth, color=qz.MAL, lw=1.0)
    ax.plot(x, -h + smooth, color=qz.MAL, lw=1.0)
    # accept and repair: a defined damaged length at the crossing
    ax = axes[3]
    ax.plot(x, h + step, color=qz.MAL, lw=1.0)
    ax.plot(x, -h + step, color=qz.MAL, lw=1.0)
    plt.rcParams["hatch.linewidth"] = 0.5
    plt.rcParams["hatch.color"] = qz.CAR
    ax.add_patch(Rectangle((5.3, -h - d), 1.4, 2 * h + d, fc="none", ec=qz.CAR, lw=0.8, hatch="///"))
    fig.subplots_adjust(left=0.2, right=0.99, bottom=0.01, top=0.99)
    _save(fig, "ch20", "fault-crossing", {"zone_start_m": np.array([z0]), "zone_end_m": np.array([z1]),
                                          "offset": np.array([d])})


ALL = {"bolt_load_transfer": bolt_load_transfer, "support_element_scc": support_element_scc,
       "shotcrete_interaction": shotcrete_interaction, "shotcrete_scc_age": shotcrete_scc_age,
       "yielding_support": yielding_support, "ring_build": ring_build, "segment_actions": segment_actions,
       "lining_ratios": lining_ratios, "joint_stiffness": joint_stiffness, "drained_tanked": drained_tanked,
       "carbonation_cover": carbonation_cover, "seismic_modes": seismic_modes,
       "free_field_strain": free_field_strain, "ovaling_flexibility": ovaling_flexibility,
       "rayleigh_damping": rayleigh_damping, "fault_crossing": fault_crossing}

if __name__ == "__main__":
    qz.use()
    wanted = sys.argv[1:]
    for name, fn in ALL.items():
        if not wanted or any(w in name for w in wanted):
            fn()
