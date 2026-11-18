"""Classic figures of the book, drawn from the book's own data and equations.

Every quantity is computed with the same functions as the companion
application (the ``geo_platform`` package at the repository root), so a figure
and the worked example it illustrates cannot drift apart. Figures are drawn at
their printed width (117 mm, the SNmono text block) in the house style of
``qz_figure.py`` and written as PDF and 300 dpi PNG, with the plotted data as
CSV beside them.

    python tools/classic_figures.py            # all figures
    python tools/classic_figures.py kirsch     # figures whose name contains "kirsch"
"""

from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
REPO = os.path.dirname(BOOK)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import matplotlib.pyplot as plt  # noqa: E402

import qz_figure as qz  # noqa: E402
from geo_platform import ccm, rock, settlement  # noqa: E402

FIG = os.path.join(BOOK, "figures")
W_FULL = 117 / 25.4          # SNmono text width
W_HALF = 0.62 * W_FULL


def _save(fig, chapter: str, stem: str, columns: dict[str, np.ndarray]) -> None:
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
    return f"{col[i]:.6g}" if i < len(col) and isinstance(col[i], (int, float, np.floating, np.integer)) else (
        str(col[i]) if i < len(col) else "")


# ---------------------------------------------------------------------------
# Chapter 16: Kirsch stresses around a circular opening
# ---------------------------------------------------------------------------

def kirsch():
    """Tangential stress on the boundary, and its decay along the springline and crown."""
    fig, (a1, a2) = qz.figure(W_FULL, 0.5 * W_FULL, ncols=2)
    theta = np.linspace(0, 90, 181)
    cols = {"theta_deg": theta}
    for k, ls in ((0.5, "--"), (1.0, "-"), (2.0, ":")):
        s = (1 + k) + 2 * (1 - k) * np.cos(np.radians(2 * theta))       # sigma_theta / sigma_v
        a1.plot(theta, s, color=qz.MAL if k != 2.0 else qz.CAR, ls=ls, lw=1.3, label=f"$k = {k:g}$")
        cols[f"k{k:g}_boundary"] = s
    a1.axhline(0, color=qz.GREY_LIGHT, lw=0.6)
    a1.set_xlim(0, 90)
    a1.set_xticks([0, 30, 60, 90])
    qz.label(a1, "Angle from springline, $\\theta$ ($^\\circ$)", "Stress ratio, $\\sigma_\\theta/\\sigma_v$ (-)")
    a1.set_ylim(-0.5, 6.5)
    a1.legend(loc="upper left", ncol=3, handlelength=1.6, columnspacing=0.8)

    r = np.linspace(1, 5, 200)
    cols["r_over_a"] = r
    k = 0.5
    for th, name, col, ls in ((0, "Springline", qz.MAL, "-"), (90, "Crown", qz.CAR, "--")):
        c2 = math.cos(math.radians(2 * th))
        s = 0.5 * ((1 + k) * (1 + r ** -2) + (1 - k) * (1 + 3 * r ** -4) * c2)
        a2.plot(r, s, color=col, ls=ls, lw=1.3, label=name)
        cols[f"{name.lower()}_k0.5"] = s
    a2.axhline(1, color=qz.GREY, lw=0.6, ls=":")
    a2.axhline(k, color=qz.GREY, lw=0.6, ls=":")
    a2.set_xlim(1, 5)
    qz.label(a2, "Distance from centre, $r/a$ (-)", "Stress ratio, $\\sigma_\\theta/\\sigma_v$ (-)")
    a2.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "ch16", "kirsch-stress", cols)


# ---------------------------------------------------------------------------
# Chapter 16: ground reaction and support characteristic, worked example
# ---------------------------------------------------------------------------

def grc():
    fig, axes = qz.figure(W_FULL, 0.52 * W_FULL, ncols=2)
    cols = {}
    for ax, (gsi, em, phi, c) in zip(axes, ((60, 13.52, 31.5, 2.85), (30, 2.12, 23.8, 1.72))):
        g = ccm.Ground(16.9, 4.75, 0.22, em, phi, c)
        res = ccm.analyse(g, [ccm.support_ring(4.75, 0.5, 60, 40, 0.25)])
        p, u = ccm.grc_curve(g, res.method)
        ok = np.isfinite(u)
        ax.plot(u[ok], p[ok], color=qz.MAL, lw=1.4, label="Ground reaction")
        s = res.support
        umax = float(np.nanmax(u[ok]))
        ux = [res.us0, res.us0 + s.usm, max(res.us0 + s.usm, umax) * 1.05]
        ax.plot(ux, [0, s.psm, s.psm], color=qz.CAR, lw=1.3, ls="--", label="Support")
        ax.plot([res.eq.u_eq], [res.eq.p_eq], "o", mfc="white", mec=qz.INK, ms=5, mew=1.0)
        ax.axhline(g.p_cr, color=qz.GREY, lw=0.6, ls=":")
        ax.set_xlim(0, umax * 1.05)
        ax.set_ylim(0, 17.5)
        ax.text(0.97, 0.95, f"GSI {gsi}", transform=ax.transAxes, ha="right", va="top")
        qz.label(ax, "Wall displacement, $u_r$ (mm)", "Support pressure, $p_i$ (MPa)")
        cols[f"u_gsi{gsi}_mm"] = u[ok]
        cols[f"p_gsi{gsi}_MPa"] = p[ok]
        print(f"  GSI {gsi}: u_max {res.u_max:.1f} mm, u_s0 {res.us0:.1f} mm, p_eq {res.eq.p_eq:.2f} MPa, "
              f"FoS {res.eq.fos:.2f}")
    axes[1].legend(loc="center right")
    fig.tight_layout()
    _save(fig, "ch16", "grc-scc", cols)


def ldp():
    g = ccm.Ground(16.9, 4.75, 0.22, 2.12, 23.8, 1.72)
    res = ccm.analyse(g, [ccm.support_ring(4.75, 0.5, 60, 40, 0.25)])
    rstar = res.rp0 / g.a
    x = np.linspace(-2 * g.a, 8 * g.a, 400)
    fig, ax = qz.figure(W_FULL, 0.5 * W_FULL)
    cols = {"X_m": x}
    styles = {ccm.LDP_METHODS[0]: (qz.MAL, "-", "Vlachopoulos and Diederichs"),
              ccm.LDP_METHODS[1]: (qz.CAR, "--", "Panet"),
              ccm.LDP_METHODS[2]: (qz.GREY, ":", "Panet, exponential")}
    for m, (col, ls, name) in styles.items():
        y = np.array([ccm.ldp(v / g.a, rstar, m) for v in x])
        ax.plot(x, y, color=col, ls=ls, lw=1.3, label=name)
        cols[name.replace(" ", "_").replace(",", "")] = y
    ax.axvline(0, color=qz.INK, lw=0.6)
    ax.plot([res.install_distance], [res.ustar], "o", color=qz.CAR, ms=4.5)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1.05)
    qz.label(ax, "Distance behind the face, $X$ (m)", "Displacement ratio, $u^*$ (-)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "ch16", "ldp-profiles", cols)


# ---------------------------------------------------------------------------
# Chapter 12: rock mass envelope and modulus
# ---------------------------------------------------------------------------

def hb_envelope():
    hb = rock.hb_constants(63, 17, 0.0)
    s3max = 21.87
    s3 = np.linspace(0, s3max * 1.2, 200)
    pts3, pts1 = rock.hb_points(110, hb.mb, hb.s, hb.a, s3max, 6)
    m1, m2 = rock.fit_mohr_circles(pts3, pts1), rock.fit_principal_line(pts3, pts1)
    fig, ax = qz.figure(W_FULL, 0.55 * W_FULL)
    intact = rock.hb_sigma1(s3, 110, 17)
    mass = rock.hb_sigma1(s3, 110, hb.mb, hb.s, hb.a)
    ax.plot(s3, intact, color=qz.GREY, ls=":", lw=1.1, label="Intact rock")
    ax.plot(s3, mass, color=qz.MAL, lw=1.5, label="Rock mass, GSI 63")
    ax.plot(s3, m2.intercept + m2.slope * s3, color=qz.CAR, ls="--", lw=1.1,
            label=f"Linear fit, $\\phi = {m2.phi_deg:.1f}^\\circ$, $c = {m2.c:.1f}$ MPa")
    ax.plot(pts3, pts1, "o", mfc="white", mec=qz.INK, ms=4, mew=0.8)
    ax.set_xlim(0, s3[-1])
    ax.set_ylim(0, 340)
    qz.label(ax, "Minor principal stress, $\\sigma_3$ (MPa)", "Major principal stress, $\\sigma_1$ (MPa)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "ch12", "hb-envelope-sandstone", {"sigma3": s3, "intact": intact, "rock_mass": mass,
                                                  "fit_s3": pts3, "fit_s1": pts1})


def em_gsi():
    gsi = np.linspace(10, 100, 181)
    fig, ax = qz.figure(W_HALF, 0.75 * W_HALF)
    cols = {"GSI": gsi}
    for d, col, ls in ((0.0, qz.MAL, "-"), (0.5, qz.MAL_SHADES[3], "--"), (1.0, qz.CAR, ":")):
        r = np.array([rock.modulus_hd_generalised(1.0, g, d) for g in gsi])
        ax.plot(gsi, r, color=col, ls=ls, lw=1.3, label=f"$D = {d:g}$")
        cols[f"D{d:g}"] = r
    ax.set_xlim(10, 100)
    ax.set_ylim(0, 1.05)
    qz.label(ax, "Geological Strength Index, GSI (-)", "Modulus ratio, $E_m/E_i$ (-)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "ch12", "modulus-gsi", cols)


# ---------------------------------------------------------------------------
# Chapter 15: squeezing strain
# ---------------------------------------------------------------------------

def squeezing():
    ratio = np.linspace(0.1, 1.0, 300)
    strain = 0.2 * ratio ** -2
    fig, ax = qz.figure(W_HALF, 0.8 * W_HALF)
    bands = ((0, 1, "Few problems"), (1, 2.5, "Minor"), (2.5, 5, "Severe"), (5, 10, "Very severe"), (10, 25, "Extreme"))
    for i, (lo, hi, name) in enumerate(bands):
        ax.axhspan(lo, hi, color=qz.CAR_TINT if i >= 2 else qz.MAL_TINT, lw=0, zorder=0)
        ax.text(0.98, math.sqrt(lo * hi) if lo else 0.7, name, ha="right", va="center",
                fontsize=8, color=qz.GREY, transform=ax.get_yaxis_transform())
    ax.plot(ratio, strain, color=qz.INK, lw=1.4)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2.5, 5, 10, 20])
    ax.set_yticklabels(["0.5", "1", "2.5", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.2, 25)
    qz.label(ax, "Strength to stress ratio, $\\sigma_{cm}/p_0$ (-)", "Closure strain, $\\varepsilon$ (%)")
    fig.tight_layout()
    _save(fig, "ch15", "squeezing-strain", {"scm_over_p0": ratio, "strain_pct": strain})


# ---------------------------------------------------------------------------
# Chapter 17 and 26: settlement trough and its development
# ---------------------------------------------------------------------------

def trough():
    b = settlement.Bore(0.0, 20.0, 6.0, 1.0, 0.5)
    y = np.linspace(-40, 40, 401)
    s = settlement.transverse([b], y)
    _, eps = settlement.horizontal([b], y)
    fig, ax = qz.figure(W_FULL, 0.45 * W_FULL)
    ax.plot(y, -s, color=qz.MAL, lw=1.5, label="Settlement")
    ax.set_ylim(-13, 6)
    qz.label(ax, "Offset from the axis, $y$ (m)", "Settlement, $S$ (mm)")
    ax2 = ax.twinx()
    ax2.plot(y, eps, color=qz.CAR, ls="--", lw=1.2, label="Horizontal strain")
    ax2.set_ylim(-650, 300)
    ax2.set_ylabel("Horizontal strain, $\\varepsilon_h$ ($\\mu\\varepsilon$)")
    for side in (-1, 1):
        ax.axvline(side * b.i, color=qz.GREY_LIGHT, lw=0.7, ls=":")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    ax.set_xlim(-40, 40)
    fig.tight_layout()
    print(f"  trough: i {b.i:.1f} m, S_max {b.smax_mm:.1f} mm")
    _save(fig, "ch17", "gaussian-trough", {"y_m": y, "S_mm": s, "eps_h_microstrain": eps})


def face():
    b = settlement.Bore(0.0, 20.0, 6.0, 1.0, 0.5)
    x = np.linspace(-4 * b.i, 4 * b.i, 300)
    f = settlement.longitudinal_factor(x, 0.0, b.i)
    fig, ax = qz.figure(W_HALF, 0.85 * W_HALF)
    ax.plot(-x / b.i, f, color=qz.MAL, lw=1.4)
    ax.axvline(0, color=qz.INK, lw=0.6)
    ax.axhline(0.5, color=qz.GREY, lw=0.6, ls=":")
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 1.05)
    qz.label(ax, "Distance behind the face, $X/i$ (-)", "Settlement ratio, $S/S_{\\max}$ (-)")
    fig.tight_layout()
    _save(fig, "ch26", "settlement-development", {"X_behind_over_i": -x / b.i, "S_over_Smax": f})


# ---------------------------------------------------------------------------
# Chapter 24: elliptical openings
# ---------------------------------------------------------------------------

def elliptical():
    q = np.linspace(0.5, 4, 300)                # W/H
    fig, ax = qz.figure(W_FULL, 0.45 * W_FULL)
    cols = {"W_over_H": q}
    for k, col in ((0.5, qz.MAL_SHADES[3]), (1.0, qz.MAL), (2.0, qz.CAR)):
        wall = 2 * q + 1 - k
        crown = k * (2 / q + 1) - 1
        ax.plot(q, wall, color=col, lw=1.3, label=f"Wall, $k = {k:g}$")
        ax.plot(q, crown, color=col, lw=1.3, ls="--", label=f"Crown, $k = {k:g}$")
        cols[f"wall_k{k:g}"], cols[f"crown_k{k:g}"] = wall, crown
    ax.axhline(0, color=qz.GREY_LIGHT, lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(0.5, 4)
    qz.label(ax, "Width to height ratio, $W/H$ (-)", "Boundary stress, $\\sigma/p_0$ (-)")
    ax.legend(loc="upper center", ncol=3)
    ax.set_ylim(-1.5, 14)
    fig.tight_layout()
    _save(fig, "ch24", "elliptical-stress", cols)


# ---------------------------------------------------------------------------
# Chapter 18: bolt inclination for a sliding block
# ---------------------------------------------------------------------------

def bolt_inclination():
    psi, phi = math.radians(45), math.radians(30)
    alpha = np.linspace(0, 90, 361)               # angle of the bar to the sliding plane
    beta = np.radians(90 - alpha)                 # angle from the plane normal
    fig, ax = qz.figure(W_HALF, 0.75 * W_HALF)
    cols = {"alpha_deg": alpha}
    for F, col, ls in ((1.0, qz.MAL, "-"), (1.5, qz.CAR, "--")):
        t = (F * math.sin(psi) - math.cos(psi) * math.tan(phi)) / (F * np.sin(beta) + math.tan(phi) * np.cos(beta))
        ax.plot(alpha, t, color=col, ls=ls, lw=1.3, label=f"$F = {F:g}$")
        a_opt = math.degrees(math.atan(math.tan(phi) / F))
        ax.plot([a_opt], [np.interp(a_opt, alpha, t)], "o", color=col, ms=4)
        cols[f"T_over_W_F{F:g}"] = t
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 30, 60, 90])
    ax.set_ylim(0, None)
    qz.label(ax, "Bar inclination to the plane, $\\alpha$ ($^\\circ$)", "Bolt force, $T/W$ (-)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "ch18", "bolt-inclination", cols)


# ---------------------------------------------------------------------------
# Data for the three-dimensional renders (tools/render_classic_3d.py)
# ---------------------------------------------------------------------------

def scenes():
    """Write the numbers the Blender scenes are built from, so the renders match the book."""
    import json

    g = ccm.Ground(16.9, 4.75, 0.22, 2.12, 23.8, 1.72)
    res = ccm.analyse(g, [ccm.support_ring(4.75, 0.5, 60, 40, 0.25)])
    rstar = res.rp0 / g.a
    xs = np.linspace(-6 * g.a, 4 * g.a, 81)          # X > 0 behind the face
    plastic = {"a": g.a, "rp_unsupported": res.rp0, "rp_supported": res.rp_eq, "lining_t": 0.5,
               "install_X": res.install_distance, "u_max_mm": res.u_max,
               "ldp_X": list(map(float, -xs)), "ldp_u": [float(ccm.ldp(-x / g.a, rstar)) for x in xs]}

    D, z0, K, vl, spacing = 6.0, 20.0, 0.5, 1.0, 15.0
    # the second bore through ground the first has disturbed: higher volume loss
    bores = [settlement.Bore(-spacing / 2, z0, D, vl, K), settlement.Bore(spacing / 2, z0, D, 1.3 * vl, K)]
    ys = np.linspace(-60, 60, 121)
    s = settlement.transverse(bores, ys)
    buildings = []
    for name, y0, y1, depth in (("A", -50, -32, 14), ("B", -24, -6, 20), ("C", 2, 16, 12), ("D", 24, 44, 18)):
        r = settlement.screen_building(bores, y0, y1)
        buildings.append({"name": name, "y0": y0, "y1": y1, "depth": depth, "height": 9 + (depth % 7) * 2,
                          "smax_mm": r["max_settlement_mm"], "slope": r["max_slope"],
                          "assess": not r["excluded"]})
    trough = {"D": D, "z0": z0, "K": K, "bores": [{"y": b.offset, "vl": b.vl_percent} for b in bores],
              "y": list(map(float, ys)), "S_mm": list(map(float, s)), "buildings": buildings}

    wedge = {"a": 3.0, "length": 12.0, "planes": [[55, 30], [55, 150]], "apex_height": 2.2,
             "bolts": [[-0.9, 3.0], [0.0, 3.0], [0.9, 3.0], [-0.9, 6.0], [0.0, 6.0], [0.9, 6.0]]}

    out = os.path.join(FIG, "render")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "scenes.json"), "w", encoding="utf-8") as fh:
        json.dump({"plastic_zone": plastic, "settlement_trough": trough, "crown_wedge": wedge}, fh, indent=1)
    print("wrote render/scenes.json:", ", ".join(f"{b['name']} {b['smax_mm']:.1f} mm" for b in buildings))


ALL = {"scenes": scenes, "kirsch": kirsch, "grc": grc, "ldp": ldp, "hb_envelope": hb_envelope, "em_gsi": em_gsi,
       "squeezing": squeezing, "trough": trough, "face": face, "elliptical": elliptical,
       "bolt_inclination": bolt_inclination}

if __name__ == "__main__":
    qz.use()
    wanted = sys.argv[1:]
    for name, fn in ALL.items():
        if not wanted or any(w in name for w in wanted):
            fn()
