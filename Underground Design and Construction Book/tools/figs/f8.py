"""Figures for Chapters 21 to 23: integrated design, excavation methods, mechanised tunnelling.

Every data figure is computed from the book's own equations and worked examples
(with the ``geo_platform`` package at the repository root where it holds the
calculation); schematic figures are drawn to plausible dimensions consistent
with the text. Figures are drawn at their printed width in the house style of
``qz_figure.py`` and written as PDF and 300 dpi PNG, with the plotted data as
CSV beside them.

    python tools/figs/f8.py              # all figures
    python tools/figs/f8.py vibration    # figures whose name contains "vibration"
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
from scipy.optimize import brentq  # noqa: E402
from scipy.stats import norm  # noqa: E402

import qz_figure as qz  # noqa: E402
from geo_platform import ccm, rock  # noqa: E402

FIG = os.path.join(BOOK, "figures")
W_FULL = 117 / 25.4          # SNmono text width
W_HALF = 0.62 * W_FULL


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


def _blank(ax):
    ax.set_axis_off()
    ax.set_aspect("equal")


# ===========================================================================
# Chapter 21: reliability, optimisation and integrated design
# ===========================================================================

def characteristic_value():
    """Point and spatially averaged distributions of a friction angle; variance reduction."""
    mu, sd = 30.0, 3.0                       # phi' (deg), coefficient of variation 10 %
    lam = np.logspace(-1, 2, 300)            # L / lambda

    def gamma2(r):                           # exponential correlation, averaging over length L
        return 2.0 / r ** 2 * (r - 1 + np.exp(-r))

    r_long = 10.0
    sd_long = sd * math.sqrt(gamma2(r_long))
    phi = np.linspace(18, 48, 400)
    p_pt = norm.pdf(phi, mu, sd)
    p_av = norm.pdf(phi, mu, sd_long)
    f_pt = mu - 1.645 * sd
    f_av = mu - 1.645 * sd_long

    fig, (a1, a2) = qz.figure(W_FULL, 0.45 * W_FULL, ncols=2)
    a1.plot(phi, p_pt, color=qz.CAR, ls="--", lw=1.3, label="Point value")
    a1.plot(phi, p_av, color=qz.MAL, lw=1.4, label="Averaged")
    for f, col, ls in ((f_pt, qz.CAR, "--"), (f_av, qz.MAL, "-")):
        a1.axvline(f, color=col, lw=0.7, ls=ls)
    a1.axvline(mu, color=qz.GREY, lw=0.6, ls=":")
    a1.set_xlim(18, 48)
    a1.set_ylim(0, 0.36)
    qz.label(a1, "Friction angle, $\\phi'$ ($^\\circ$)", "Density, $p$ (1/$^\\circ$)")
    a1.legend(loc="upper right", handlelength=1.8)

    g2 = gamma2(lam)
    a2.semilogx(lam, g2, color=qz.MAL, lw=1.4)
    a2.plot([r_long], [gamma2(r_long)], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9)
    a2.set_xlim(0.1, 100)
    a2.set_ylim(0, 1.05)
    qz.label(a2, "Averaging length, $L/\\lambda$ (-)", "Variance ratio, $\\Gamma^2$ (-)")
    fig.tight_layout()
    print(f"  5% fractile: point {f_pt:.2f}, averaged {f_av:.2f}; Gamma^2(10) = {gamma2(r_long):.3f}")
    _save(fig, "ch21", "characteristic-averaging",
          {"phi_deg": phi, "pdf_point": p_pt, "pdf_averaged": p_av,
           "L_over_lambda": lam, "Gamma2": g2})


def ec7_category():
    """Geotechnical Category from consequence and complexity class, and the four routes."""
    fig = plt.figure(figsize=(W_FULL, 0.42 * W_FULL))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 117)
    ax.set_ylim(0, 0.42 * 117)
    ax.set_axis_off()

    # matrix, lower left at (16, 10), cells 11 x 11 mm
    x0, y0, c = 17.0, 11.0, 11.0
    shade = {1: qz.MAL_TINT, 2: qz.MAL_SHADES[3], 3: qz.MAL}
    for i in range(3):            # complexity class, columns
        for j in range(3):        # consequence class, rows
            # only the diagonal is drawn with its category; the off-diagonal cells are set by
            # the code's table (a nationally determined choice) and are left hatched
            fc = shade[i + 1] if i == j else "white"
            ax.add_patch(mp.Rectangle((x0 + i * c, y0 + j * c), c, c, fc=fc, ec=qz.INK, lw=0.6))
            if i == j:
                ax.text(x0 + (i + 0.5) * c, y0 + (j + 0.5) * c, f"GC{i + 1}", ha="center", va="center",
                        fontsize=9, color="white" if i == 2 else qz.INK)
            else:
                ax.add_patch(mp.Rectangle((x0 + i * c, y0 + j * c), c, c, fc="none", ec=qz.GREY_LIGHT,
                                          lw=0, hatch="////"))
                ax.add_patch(mp.Rectangle((x0 + i * c, y0 + j * c), c, c, fc="none", ec=qz.INK, lw=0.6))
    for i, t in enumerate(("GCC1", "GCC2", "GCC3")):
        ax.text(x0 + (i + 0.5) * c, y0 - 2.0, t, ha="center", va="top", fontsize=8)
    for j, t in enumerate(("CC1", "CC2", "CC3")):
        ax.text(x0 - 1.5, y0 + (j + 0.5) * c, t, ha="right", va="center", fontsize=8)
    ax.text(x0 + 1.5 * c, y0 - 7.0, "Geotechnical complexity class", ha="center", va="top", fontsize=9)
    ax.text(x0 - 9.5, y0 + 1.5 * c, "Consequence class", ha="center", va="center", rotation=90, fontsize=9)
    # urban tunnel beneath a city: a carmine ring in the top cell
    ax.add_patch(mp.Rectangle((x0 + 2 * c + 0.9, y0 + 2 * c + 0.9), c - 1.8, c - 1.8, fc="none",
                              ec=qz.CAR, lw=1.4))

    # arrow to routes
    ax.annotate("", xy=(62.0, y0 + 1.5 * c), xytext=(x0 + 3 * c + 4.0, y0 + 1.5 * c),
                arrowprops=dict(arrowstyle="-|>", color=qz.INK, lw=0.8, mutation_scale=8))

    routes = (("Calculation", "Partial factors or reliability"),
              ("Prescriptive measures", "Empirical design systems"),
              ("Testing", "Pull-out tests, trial panels"),
              ("Observational method", "Monitoring and updating"))
    bx, bw, bh, gap = 63.0, 52.0, 9.0, 2.0
    top = y0 + 1.5 * c + 2 * bh + 1.5 * gap
    for k, (head, sub) in enumerate(routes):
        yb = top - (k + 1) * bh - k * gap
        ax.add_patch(mp.FancyBboxPatch((bx, yb), bw, bh, boxstyle="round,pad=0,rounding_size=1.2",
                                       fc="white", ec=qz.MAL, lw=0.8))
        ax.text(bx + 2.5, yb + 0.66 * bh, head, ha="left", va="center", fontsize=9)
        ax.text(bx + 2.5, yb + 0.28 * bh, sub, ha="left", va="center", fontsize=8, color=qz.GREY)
    _save(fig, "ch21", "ec7-category-routes")


def _scm_hm(gsi, sci, mi):
    """Rock mass strength of Hoek and Marinos (2000), as printed in Chapter 15."""
    return 0.0034 * mi ** 0.8 * sci * (1.029 + 0.025 * np.exp(-0.1 * mi)) ** gsi


def squeezing_uncertainty():
    """Monte Carlo and point estimates of the strength ratio; variance shares."""
    p0 = 16.9
    rng = np.random.default_rng(2027)
    n = 200_000
    g = rng.normal(30, 5, n)
    s = rng.normal(72, 10.8, n)
    m = rng.normal(7, 1, n)
    ratio = _scm_hm(g, s, m) / p0
    pem = np.array([_scm_hm(G, S, M) / p0 for G in (25, 35) for S in (61.2, 82.8) for M in (6, 8)])
    mu, sd = pem.mean(), pem.std()
    beta = (mu - 0.200) / sd
    print(f"  PEM mean {mu:.3f}, sd {sd:.3f}, beta {beta:.2f}; MC P(<0.2) {np.mean(ratio < 0.2):.3f}, "
          f"P(strain>10%) {np.mean(0.2 * ratio ** -2 > 10):.3f}, "
          f"84th pct strain {np.percentile(0.2 * ratio ** -2, 84):.2f}%")

    fig, (a1, a2) = qz.figure(W_FULL, 0.45 * W_FULL, ncols=2, gridspec_kw={"width_ratios": [1.55, 1]})
    bins = np.linspace(0.05, 0.50, 46)
    h, e = np.histogram(ratio, bins=bins, density=True)
    ctr = 0.5 * (e[1:] + e[:-1])
    a1.bar(ctr, h, width=e[1] - e[0], color=qz.MAL_TINT, edgecolor=qz.MAL_SHADES[3], lw=0.4,
           label="Monte Carlo")
    xs = np.linspace(0.05, 0.50, 300)
    a1.plot(xs, norm.pdf(xs, mu, sd), color=qz.MAL, lw=1.3, label="Normal fit")
    a1.plot(pem, np.full(8, 0.25), "|", color=qz.INK, ms=8, mew=1.0)
    a1.axvline(0.200, color=qz.CAR, lw=1.0, ls="--")
    a1.axvline(math.sqrt(0.02), color=qz.CAR, lw=0.8, ls=":")
    a1.annotate("", xy=(0.200, 8.0), xytext=(mu, 8.0),
                arrowprops=dict(arrowstyle="<->", color=qz.INK, lw=0.7, mutation_scale=6,
                                shrinkA=0, shrinkB=0))
    a1.text(0.5 * (0.2 + mu), 8.25, "$\\beta\\sigma$", ha="center", va="bottom", fontsize=9)
    a1.set_xlim(0.05, 0.50)
    a1.set_ylim(0, 10)
    qz.label(a1, "Strength ratio, $\\sigma_{cm}/p_0$ (-)", "Density, $p$ (-)")
    a1.legend(loc="upper right", handlelength=1.6, fontsize=7.5)

    names = ("GSI", "$\\sigma_{ci}$", "$m_i$")
    dlt = [(_scm_hm(35, 72, 7) - _scm_hm(25, 72, 7)) / 2 / p0,          # one standard deviation each
           (_scm_hm(30, 82.8, 7) - _scm_hm(30, 61.2, 7)) / 2 / p0,
           (_scm_hm(30, 72, 8) - _scm_hm(30, 72, 6)) / 2 / p0]
    print("  one-at-a-time changes " + ", ".join(f"{x:.3f}" for x in dlt))
    share = np.array(dlt) ** 2
    share = share / share.sum() * 100
    yy = np.arange(3)[::-1]
    a2.barh(yy, share, color=[qz.MAL, qz.MAL_SHADES[2], qz.MAL_SHADES[3]], height=0.6)
    for y_, v in zip(yy, share):
        a2.text(v + 2, y_, f"{v:.0f}", va="center", ha="left", fontsize=8)
    a2.set_yticks(yy)
    a2.set_yticklabels(names)
    a2.tick_params(axis="y", which="both", right=False, left=False)
    a2.set_xlim(0, 80)
    a2.set_ylim(-0.6, 2.6)
    qz.label(a2, "Share of variance (%)", None)
    fig.tight_layout()
    _save(fig, "ch21", "squeezing-uncertainty",
          {"ratio_bin": ctr, "mc_density": h, "pem_ratio": pem, "variable": ["GSI", "sigma_ci", "m_i"],
           "variance_share_pct": share})


POOR = dict(p0=16.9, a=4.75, nu=0.22, em_gpa=2.12, phi_deg=23.8, c=1.72)


def empirical_vs_ccm():
    """Q-system shotcrete shell against the 0.5 m lining on the poor-domain ground reaction curve."""
    g = ccm.Ground(**POOR)
    lining = ccm.support_ring(4.75, 0.5, 60, 40, 0.25)
    shell = ccm.support_ring(4.75, 0.09, 35, 30, 0.2)
    r_l = ccm.analyse(g, [lining])
    r_s = ccm.analyse(g, [shell])
    us0 = r_l.us0
    # demand on an unyielding shell of the same stiffness
    p_el = brentq(lambda p: ccm.grc_displacement(g, p) - (us0 + p / shell.ks * 1000.0), 0.0, g.p_cr)
    p, u = ccm.grc_curve(g)
    ok = np.isfinite(u)
    print(f"  lining p_eq {r_l.eq.p_eq:.2f} FoS {r_l.eq.fos:.2f}; shell p_sm {shell.psm:.2f}, "
          f"elastic demand {p_el:.2f}, plateau u {r_s.eq.u_eq:.1f} mm")

    fig, ax = qz.figure(W_FULL, 0.50 * W_FULL)
    ax.plot(u[ok], p[ok], color=qz.INK, lw=1.4, label="Ground reaction, GSI 30")
    ax.plot([us0, us0 + lining.usm, 145], [0, lining.psm, lining.psm], color=qz.MAL, lw=1.3,
            label="Lining, 0.5 m")
    ax.plot([us0, us0 + shell.usm, 145], [0, shell.psm, shell.psm], color=qz.CAR, lw=1.3, ls="--",
            label="Shotcrete shell, 90 mm")
    ax.plot([us0 + shell.usm, us0 + p_el / shell.ks * 1000.0], [shell.psm, p_el], color=qz.CAR, lw=0.8, ls=":")
    ax.plot([r_l.eq.u_eq], [r_l.eq.p_eq], "o", mfc="white", mec=qz.MAL, ms=5, mew=1.1)
    ax.plot([us0 + p_el / shell.ks * 1000.0], [p_el], "o", mfc="white", mec=qz.CAR, ms=5, mew=1.1)
    ax.plot([r_s.eq.u_eq], [shell.psm], "s", mfc=qz.CAR, mec=qz.CAR, ms=4)
    ax.set_xlim(0, 145)
    ax.set_ylim(0, 8.0)
    qz.label(ax, "Wall displacement, $u_r$ (mm)", "Support pressure, $p_i$ (MPa)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45))
    fig.tight_layout()
    _save(fig, "ch21", "empirical-vs-ccm",
          {"u_mm": u[ok], "p_MPa": p[ok], "us0_mm": [us0], "lining_psm": [lining.psm],
           "lining_peq": [r_l.eq.p_eq], "shell_psm": [shell.psm], "shell_elastic_demand": [p_el],
           "shell_plateau_u": [r_s.eq.u_eq]})


def controlled_deformation():
    """Lining thickness for a constant factor of safety, against installation distance."""
    g = ccm.Ground(**POOR)
    fos_target = ccm.analyse(g, [ccm.support_ring(4.75, 0.5, 60, 40, 0.25)]).eq.fos
    dist = np.linspace(1.0, 12.0, 45)
    t_req, u_eq, p_eq, vol = [], [], [], []
    for L in dist:
        def f(t):
            r = ccm.analyse(g, [ccm.support_ring(4.75, t, 60, 40, 0.25)], value=float(L))
            return r.eq.fos - fos_target
        t = brentq(f, 0.02, 2.0, xtol=1e-5)
        r = ccm.analyse(g, [ccm.support_ring(4.75, t, 60, 40, 0.25)], value=float(L))
        t_req.append(t)
        u_eq.append(r.eq.u_eq)
        p_eq.append(r.eq.p_eq)
        vol.append(math.pi * ((4.75 + t) ** 2 - 4.75 ** 2))
    t_req, u_eq, p_eq, vol = map(np.array, (t_req, u_eq, p_eq, vol))
    print(f"  FoS target {fos_target:.2f}; L=3 m: t {np.interp(3, dist, t_req):.3f} m; "
          f"L=1: t {t_req[0]:.3f}, u {u_eq[0]:.1f}; L=12: t {t_req[-1]:.3f}, u {u_eq[-1]:.1f}")

    fig, (a1, a2) = qz.figure(W_FULL, 0.45 * W_FULL, ncols=2)
    a1.plot(dist, t_req, color=qz.MAL, lw=1.4)
    a1.plot([3.0], [0.5], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9)
    a1.set_xlim(0, 12)
    a1.set_ylim(0, 0.9)
    qz.label(a1, "Installation distance, $L$ (m)", "Lining thickness, $t$ (m)")

    a2.plot(u_eq, vol, color=qz.CAR, lw=1.4)
    for L, mk in ((1.0, "s"), (3.0, "o"), (6.0, "^"), (12.0, "D")):
        i = int(np.argmin(abs(dist - L)))
        a2.plot([u_eq[i]], [vol[i]], mk, mfc="white", mec=qz.INK, ms=4.5, mew=0.9, label=f"$L = {L:g}$ m")
    a2.set_xlim(40, 120)
    a2.set_ylim(0, 35)
    qz.label(a2, "Wall displacement, $u_{eq}$ (mm)", "Concrete, $V_c$ (m$^3$/m)")
    a2.legend(loc="upper right", handletextpad=0.3)
    fig.tight_layout()
    _save(fig, "ch21", "controlled-deformation",
          {"L_m": dist, "t_req_m": t_req, "u_eq_mm": u_eq, "p_eq_MPa": p_eq, "V_m3_per_m": vol})


def bayesian_update():
    """Normal prior, likelihood and posterior for a modulus; posterior spread with repeated observations."""
    mu0, s0 = 2.12, 0.60
    z, sz = 1.50, 0.40
    mu1 = (sz ** 2 * mu0 + s0 ** 2 * z) / (s0 ** 2 + sz ** 2)
    s1 = (1 / s0 ** 2 + 1 / sz ** 2) ** -0.5
    e = np.linspace(0, 5.0, 500)
    print(f"  posterior mean {mu1:.3f} GPa, sd {s1:.3f} GPa")

    fig, (a1, a2) = qz.figure(W_FULL, 0.45 * W_FULL, ncols=2)
    a1.plot(e, norm.pdf(e, mu0, s0), color=qz.GREY, lw=1.2, ls="--", label="Prior")
    a1.plot(e, norm.pdf(e, z, sz), color=qz.CAR, lw=1.2, ls=":", label="Likelihood")
    a1.plot(e, norm.pdf(e, mu1, s1), color=qz.MAL, lw=1.5, label="Posterior")
    a1.set_xlim(0, 5)
    a1.set_ylim(0, 1.45)
    qz.label(a1, "Rock mass modulus, $E_m$ (GPa)", "Density, $p$ (1/GPa)")
    a1.legend(loc="upper right")

    nobs = np.arange(0, 21)
    cols = {"E_GPa": e, "prior": norm.pdf(e, mu0, s0), "likelihood": norm.pdf(e, z, sz),
            "posterior": norm.pdf(e, mu1, s1), "n_obs": nobs}
    for r, col, ls in ((0.5, qz.MAL, "-"), (1.0, qz.MAL_SHADES[3], "--"), (2.0, qz.CAR, ":")):
        ratio = 1 / np.sqrt(1 + nobs / r ** 2)
        a2.plot(nobs, ratio, color=col, ls=ls, lw=1.3, marker="o", ms=2.5,
                label=f"$\\sigma_z/\\sigma_0 = {r:g}$")
        cols[f"sigma1_over_sigma0_r{r:g}"] = ratio
    a2.set_xlim(0, 20)
    a2.set_ylim(0, 1.05)
    a2.set_xticks([0, 5, 10, 15, 20])
    qz.label(a2, "Number of observations, $n$ (-)", "Spread ratio, $\\sigma_n/\\sigma_0$ (-)")
    a2.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "ch21", "bayesian-update", cols)


# ===========================================================================
# Chapter 22: excavation methods
# ===========================================================================

def advance_core():
    """ADECO-RS: advance core, extrusion and preconvergence (schematic, longitudinal section)."""
    fig = plt.figure(figsize=(W_FULL, 0.36 * W_FULL))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(-26, 22)
    ax.set_ylim(-9.2, 8.1)
    _blank(ax)
    R = 5.0
    # excavated tunnel behind the face (x < 0), with convergence growing behind the face
    xb = np.linspace(-26, 0, 200)
    conv = 0.6 + 0.9 * (1 - np.exp(xb / 6.0))           # exaggerated, schematic
    # preconvergence ahead of the face
    xa = np.linspace(0, 12, 100)
    pre = 0.6 * np.exp(-xa / 3.0)
    for sgn in (1, -1):
        ax.fill_between(xb, sgn * (R - conv), sgn * (R + 0.9), color=qz.GREY_LIGHT, lw=0)   # lining
        ax.plot(xb, sgn * (R - conv), color=qz.INK, lw=0.9)
        ax.plot(xa, sgn * (R - pre), color=qz.CAR, lw=1.0, ls="--")
        ax.plot([-26, 12], [sgn * R, sgn * R], color=qz.GREY, lw=0.5, ls=":")
    # advance core
    ax.add_patch(mp.Rectangle((0, -R), 9.0, 2 * R, fc=qz.MAL_TINT, ec=qz.MAL, lw=0.9, ls="-"))
    # face reinforcement
    for yy in np.linspace(-3.8, 3.8, 7):
        ax.plot([0.0, 12.0], [yy, yy], color=qz.MAL, lw=0.6)
    # face, bulged by extrusion
    yf = np.linspace(-R + 0.25, R - 0.25, 100)
    ax.plot(-0.9 * (1 - (yf / R) ** 2), yf, color=qz.CAR, lw=1.2)
    ax.plot([0, 0], [-R, R], color=qz.INK, lw=0.6)
    for yy in (-2.5, 0.0, 2.5):
        ax.annotate("", xy=(-2.6, yy), xytext=(-0.3, yy),
                    arrowprops=dict(arrowstyle="-|>", color=qz.CAR, lw=0.8, mutation_scale=7))
    # labels
    ax.text(4.5, 0.0, "Advance core", ha="center", va="center", fontsize=9,
            bbox=dict(fc=qz.MAL_TINT, ec="none", pad=0.6))
    ax.text(-4.2, 0.0, "Extrusion", ha="right", va="center", fontsize=9)
    ax.annotate("Preconvergence", xy=(1.5, R - 0.45), xytext=(10.0, 7.3), fontsize=9, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=qz.INK, lw=0.5))
    ax.annotate("Convergence", xy=(-16, R - 1.4), xytext=(-21, 7.3), fontsize=9, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=qz.INK, lw=0.5))
    ax.annotate("Face reinforcement", xy=(10.6, -2.53), xytext=(13.0, -8.1), fontsize=9, ha="center",
                va="center", arrowprops=dict(arrowstyle="-", color=qz.INK, lw=0.5))
    ax.annotate("Lining", xy=(-11, -R - 0.45), xytext=(-11, -8.1), fontsize=9, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=qz.INK, lw=0.5))
    ax.annotate("", xy=(-20, 0), xytext=(-25, 0),
                arrowprops=dict(arrowstyle="-|>", color=qz.GREY, lw=0.8, mutation_scale=8))
    ax.text(-22.5, 0.6, "Drive", ha="center", va="bottom", fontsize=8, color=qz.GREY)
    _save(fig, "ch22", "advance-core")


def cavern_shapes():
    """Mushroom, horseshoe and elliptical machine-hall sections to one scale (schematic)."""
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 0.44 * W_FULL))
    H, B = 48.0, 25.0
    for ax in axes:
        _blank(ax)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-5, 49)

    # mushroom: wider arch on abutments, narrower walls below
    ax = axes[0]
    Bw, Ba, ha = 23.0, 28.0, 36.0               # wall span, arch span, springline height
    th = np.linspace(0, np.pi, 100)
    rise = 7.0
    xa = Ba / 2 * np.cos(th)
    ya = ha + rise * np.sin(th)
    outline_x = np.concatenate([[-Bw / 2, -Bw / 2, -Ba / 2], xa[::-1], [Ba / 2, Bw / 2, Bw / 2, -Bw / 2]])
    outline_y = np.concatenate([[0, ha - 4, ha], ya[::-1], [ha, ha - 4, 0, 0]])
    ax.fill(outline_x, outline_y, fc="white", ec=qz.INK, lw=0.9)
    # concrete arch
    ta = 1.6
    xo = (Ba / 2 + ta) * np.cos(th)
    yo = ha + (rise + ta) * np.sin(th)
    ax.fill(np.concatenate([xa, xo[::-1]]), np.concatenate([ya, yo[::-1]]), fc=qz.GREY_LIGHT, ec=qz.INK, lw=0.6)
    ax.text(0, -3.5, "Mushroom", ha="center", va="top", fontsize=9)

    # horseshoe: vertical walls, arched roof
    ax = axes[1]
    hs = H - 6.0
    xa = B / 2 * np.cos(th)
    ya = hs + 6.0 * np.sin(th)
    ax.fill(np.concatenate([[-B / 2, -B / 2], xa[::-1], [B / 2, B / 2]]),
            np.concatenate([[0, hs], ya[::-1], [hs, 0]]), fc="white", ec=qz.INK, lw=0.9)
    ax.text(0, -3.5, "Horseshoe", ha="center", va="top", fontsize=9)

    # ellipse
    ax = axes[2]
    t = np.linspace(0, 2 * np.pi, 200)
    ax.fill(B / 2 * np.cos(t), H / 2 + H / 2 * np.sin(t), fc="white", ec=qz.INK, lw=0.9)
    ax.text(0, -3.5, "Elliptical", ha="center", va="top", fontsize=9)

    # scale bar on the last panel
    axes[2].plot([15.0, 15.0], [0, 10], color=qz.INK, lw=1.0)
    axes[2].text(16.0, 5, "10 m", ha="left", va="center", fontsize=8)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.99, bottom=0.03, wspace=0.0)
    _save(fig, "ch22", "cavern-shapes")


def blast_round():
    """Face of a 6 m by 6 m drill-and-blast round: cut, stoping, contour and lifter holes (schematic)."""
    W, hw, r = 6.0, 3.0, 3.0              # span, wall height, arch radius
    th = np.linspace(0, np.pi, 200)
    ox = np.concatenate([[-W / 2, -W / 2], r * np.cos(th)[::-1] * (W / 2 / r), [W / 2, W / 2, -W / 2]])
    oy = np.concatenate([[0, hw], hw + r * np.sin(th)[::-1], [hw, 0, 0]])

    holes = {}
    # contour: 0.15 m inside the line, about 0.5 m apart on walls and arch
    off = 0.15
    rc = r - off
    n_arc = int(round(math.pi * rc / 0.5))
    a = np.linspace(0, np.pi, n_arc + 1)
    cx = list(rc * np.cos(a))
    cy = list(hw + rc * np.sin(a))
    for yy in np.arange(hw - 0.55, 0.9, -0.55):
        cx += [-(W / 2 - off), W / 2 - off]
        cy += [yy, yy]
    holes["Contour"] = (np.array(cx), np.array(cy))
    # lifters along the floor, about 0.9 m apart
    lx = np.linspace(-(W / 2 - off), W / 2 - off, 8)
    holes["Lifter"] = (lx, np.full_like(lx, 0.2))
    # cut: parallel burn cut centred at (0.3, 1.8)
    ccx, ccy = 0.3, 1.8
    holes["Relief (empty)"] = (np.array([ccx - 0.09, ccx + 0.09]), np.array([ccy, ccy]))
    cut = []
    for k, d in ((4, 0.32), (4, 0.62), (4, 0.98)):
        rot = 0 if d == 0.62 else np.pi / 4          # successive squares turned through 45 degrees
        for i in range(k):
            ang = rot + i * np.pi / 2
            cut.append((ccx + d * np.cos(ang), ccy + d * np.sin(ang)))
    cut = np.array(cut)
    holes["Cut"] = (cut[:, 0], cut[:, 1])
    # stoping: a grid of about 0.9 m inside the contour, clear of cut and lifters
    sx, sy = [], []
    for yy in np.arange(0.95, hw + r - 0.6, 0.85):
        for xx in np.arange(-W / 2 + 0.85, W / 2 - 0.6, 0.9):
            if math.hypot(xx - ccx, yy - ccy) < 1.35:
                continue
            if yy > hw and math.hypot(xx, yy - hw) > r - 0.75:
                continue
            sx.append(xx)
            sy.append(yy)
    holes["Stoping"] = (np.array(sx), np.array(sy))

    fig = plt.figure(figsize=(W_HALF, 0.645 * W_HALF))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    _blank(ax)
    ax.set_xlim(-3.3, 6.9)
    ax.set_ylim(-0.3, 6.3)
    ax.fill(ox, oy, fc="white", ec=qz.INK, lw=1.0)
    style = {"Cut": dict(marker="o", ms=4.0, mfc=qz.CAR, mec=qz.CAR),
             "Relief (empty)": dict(marker="o", ms=5.0, mfc="white", mec=qz.CAR, mew=1.0),
             "Stoping": dict(marker="o", ms=3.6, mfc=qz.MAL, mec=qz.MAL),
             "Contour": dict(marker="o", ms=3.6, mfc="white", mec=qz.MAL, mew=0.9),
             "Lifter": dict(marker="s", ms=3.6, mfc=qz.MAL_SHADES[0], mec=qz.MAL_SHADES[0])}
    cols = {}
    for name in ("Cut", "Relief (empty)", "Stoping", "Contour", "Lifter"):
        x, y = holes[name]
        ax.plot(x, y, ls="none", label=name, **style[name])
        cols[f"{name}_x"] = x
        cols[f"{name}_y"] = y
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.98), handletextpad=0.2, borderaxespad=0.2)
    # scale bar
    ax.plot([4.0, 5.0], [0.0, 0.0], color=qz.INK, lw=1.0)
    ax.text(4.5, 0.12, "1 m", ha="center", va="bottom", fontsize=8)
    print("  holes: " + ", ".join(f"{k} {len(v[0])}" for k, v in holes.items()))
    _save(fig, "ch22", "blast-round", cols)


def disturbance_strength():
    """Rock mass strength and modulus against the disturbance factor D, both domains."""
    d = np.linspace(0, 1, 101)
    fig, (a1, a2) = qz.figure(W_FULL, 0.46 * W_FULL, ncols=2)
    cols = {"D": d}
    for gsi, col, ls in ((60, qz.MAL, "-"), (30, qz.CAR, "--")):
        scm = np.array([rock.rock_mass_ucs(72, rock.hb_constants(gsi, 7, x)) for x in d])
        em = np.array([rock.modulus_hd_generalised(26, gsi, x) for x in d])
        a1.plot(d, scm, color=col, ls=ls, lw=1.4, label=f"GSI {gsi}")
        a2.plot(d, em, color=col, ls=ls, lw=1.4, label=f"GSI {gsi}")
        cols[f"scm_gsi{gsi}"] = scm
        cols[f"Em_gsi{gsi}"] = em
        s5, s8 = np.interp(0.5, d, scm), np.interp(0.8, d, scm)
        e5, e8 = np.interp(0.5, d, em), np.interp(0.8, d, em)
        print(f"  GSI {gsi}: scm D0 {scm[0]:.2f}, D0.5 {s5:.2f}, D0.8 {s8:.2f} MPa (+{(s5 / s8 - 1) * 100:.0f}%); "
              f"Em D0 {em[0]:.2f}, D0.5 {e5:.2f}, D0.8 {e8:.2f} GPa (+{(e5 / e8 - 1) * 100:.0f}%)")
    for ax in (a1, a2):
        for x in (0.5, 0.8):
            ax.axvline(x, color=qz.GREY, lw=0.6, ls=":")
        ax.set_xlim(0, 1)
    a1.set_ylim(0, 9)
    a2.set_ylim(0, 16)
    qz.label(a1, "Disturbance factor, $D$ (-)", "Strength, $\\sigma_{cm}$ (MPa)")
    qz.label(a2, "Disturbance factor, $D$ (-)", "Modulus, $E_m$ (GPa)")
    a1.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "ch22", "disturbance-strength", cols)


def vibration_criteria():
    """Peak particle velocity criteria against frequency, from the limits table."""
    f = np.logspace(0, 2, 300)

    def din(f, a, b, c):          # DIN 4150-3 Table 1, foundation: <10, 10-50, 50-100 Hz
        return np.interp(f, [1, 10, 50, 100], [a, a, b, c])

    def bs_res(f):                # BS 7385-2: 15 at 4 Hz, 20 at 15 Hz, 50 at 40 Hz and above
        return np.interp(f, [4, 15, 40, 100], [15, 20, 50, 50])

    fig, ax = qz.figure(W_FULL, 0.62 * W_FULL)
    series = {
        "DIN 4150-3, commercial": (din(f, 20, 40, 50), qz.MAL_SHADES[0], "-", f),
        "DIN 4150-3, residential": (din(f, 5, 15, 20), qz.MAL, "-", f),
        "DIN 4150-3, sensitive": (din(f, 3, 8, 10), qz.MAL_SHADES[3], "-", f),
        "BS 7385-2, residential": (bs_res(f), qz.CAR, "--", f),
    }
    cols = {"f_Hz": f}
    for name, (v, col, ls, x) in series.items():
        m = x >= (4 if name.startswith("BS") else 1)
        ax.loglog(x[m], v[m], color=col, ls=ls, lw=1.4 if "residential" in name else 1.1, label=name)
        cols[name.replace(" ", "_").replace(",", "")] = np.where(m, v, np.nan)
    ax.axhline(1.0, color=qz.GREY, lw=1.0, ls=":", label="Complaint likely, BS 5228-2")
    ax.axhspan(0.14, 0.3, color=qz.GREY_LIGHT, alpha=0.6, lw=0, label="Perception, BS 5228-2")
    ax.set_xlim(1, 100)
    ax.set_ylim(0.01, 100)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    qz.label(ax, "Frequency, $f$ (Hz)", "Peak particle velocity, $v$ (mm/s)")
    ax.legend(loc="lower right", ncol=2, handlelength=2.0, fontsize=7.5, columnspacing=1.0)
    fig.tight_layout()
    _save(fig, "ch22", "vibration-criteria", cols)


def continuous_vibration():
    """Attenuation of continuous vibration with distance, and the vibration dose over a day."""
    d_ref = 10.0
    d = np.linspace(5, 80, 300)
    fig, (a1, a2) = qz.figure(W_FULL, 0.50 * W_FULL, ncols=2)
    cols = {"d_m": d}
    beta = 0.012                      # companion's default damping coefficient (1/m), a placeholder
    for n, beta_, col, ls in ((0.5, 0.0, qz.MAL_SHADES[3], "--"), (0.5, beta, qz.MAL, "-"),
                              (1.0, 0.0, qz.CAR_SHADES[3], "--"), (1.0, beta, qz.CAR, "-")):
        v = (d / d_ref) ** -n * np.exp(-beta_ * (d - d_ref))
        lab = f"$n = {n:g}$" + (f", $\\beta = {beta_:g}$ /m" if beta_ else ", undamped")
        a1.loglog(d, v, color=col, ls=ls, lw=1.3, label=lab)
        cols[f"n{n:g}_beta{beta_:g}"] = v
    a1.set_xlim(5, 80)
    a1.set_ylim(0.01, 2.5)
    a1.set_xticks([5, 10, 20, 40, 80])
    a1.set_xticklabels(["5", "10", "20", "40", "80"])
    a1.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
    a1.set_yticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"])
    a1.minorticks_off()
    qz.label(a1, "Slant distance, $d$ (m)", "Velocity ratio, $v/v_{\\mathrm{ref}}$ (-)")
    a1.legend(loc="lower left", fontsize=7.5, handlelength=1.8)

    t_h = np.linspace(0.1, 16, 300)
    a2.axhspan(0.2, 0.4, color=qz.CAR_TINT, lw=0)
    cols["t_h"] = t_h
    for a, col, ls in ((0.02, qz.CAR, "-"), (0.01, qz.MAL, "-"), (0.005, qz.MAL_SHADES[3], "--")):
        vdv = 1.4 * a * (t_h * 3600) ** 0.25
        a2.plot(t_h, vdv, color=col, ls=ls, lw=1.3, label=f"$a = {a:g}$ m/s$^2$")
        cols[f"VDV_a{a:g}"] = vdv
    a2.set_xlim(0, 16)
    a2.set_ylim(0, 0.8)
    a2.set_xticks([0, 4, 8, 12, 16])
    qz.label(a2, "Exposure, $t$ (h)", "Dose value, VDV (m/s$^{1.75}$)")
    a2.legend(loc="upper left", fontsize=7.5, handlelength=1.8)
    fig.tight_layout()
    _save(fig, "ch22", "continuous-vibration", cols)


# ===========================================================================
# Chapter 23: mechanised tunnelling
# ===========================================================================

def _ground(ax, x0, x1, ytop, ybot, R):
    for y0, y1 in ((R, ytop), (-ybot, -R)):
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="white", ec=qz.GREY_LIGHT,
                                  hatch="////", lw=0))
    for y in (R, -R):
        ax.plot([x0, x1], [y, y], color=qz.GREY, lw=0.6)


def tbm_families():
    """Open gripper, single shield, double shield and closed-face machines (schematic sections)."""
    plt.rcParams["hatch.color"] = qz.GREY_LIGHT
    plt.rcParams["hatch.linewidth"] = 0.4
    fig, axes = plt.subplots(4, 1, figsize=(W_FULL, 0.98 * W_FULL))
    R = 3.0
    names = ("Open gripper", "Single shield", "Double shield", "Earth pressure balance")
    for ax, name in zip(axes, names):
        _blank(ax)
        ax.set_xlim(-25.5, 3.6)
        ax.set_ylim(-4.6, 4.6)
        _ground(ax, -25.5, 3.6, 4.6, 4.6, R)
        # face ahead
        ax.add_patch(mp.Rectangle((0.35, -R), 3.25, 2 * R, fc="white", ec=qz.GREY_LIGHT, hatch="////", lw=0))
        ax.plot([0.35, 0.35], [-R, R], color=qz.GREY, lw=0.6)
        # cutterhead
        ax.add_patch(mp.Rectangle((-0.35, -R + 0.05), 0.7, 2 * R - 0.1, fc=qz.CAR, ec=qz.CAR))
        for yy in np.linspace(-2.4, 2.4, 7):
            ax.add_patch(mp.Circle((0.38, yy), 0.13, fc=qz.INK, ec="none"))
        ax.text(-25.2, 0, name, ha="left", va="center", fontsize=9,
                bbox=dict(fc="white", ec="none", pad=0.5))

    def shield(ax, x0, x1, gap=0.0):
        for s in (1, -1):
            ax.add_patch(mp.Rectangle((x0, s * (R - 0.05) - (0.35 if s > 0 else 0)), x1 - x0, 0.35,
                                      fc=qz.MAL, ec=qz.MAL))

    def segments(ax, x0, x1, n):
        xs = np.linspace(x0, x1, n + 1)
        for s in (1, -1):
            for a, b in zip(xs[:-1], xs[1:]):
                ax.add_patch(mp.Rectangle((a + 0.03, s * (R - 0.1) - (0.4 if s > 0 else 0)), b - a - 0.06, 0.4,
                                          fc=qz.GREY_LIGHT, ec=qz.GREY, lw=0.4))

    def cylinders(ax, x0, x1):
        for y in (1.6, -1.6):
            ax.plot([x0, x1], [y, y], color=qz.INK, lw=2.0, solid_capstyle="butt")
            ax.plot([x0, x0 + 0.6 * (x1 - x0)], [y, y], color=qz.INK, lw=3.5, solid_capstyle="butt")

    def grippers(ax, xc):
        for s in (1, -1):
            ax.add_patch(mp.Rectangle((xc - 1.0, s * R - (0.45 if s > 0 else 0)), 2.0, 0.45,
                                      fc=qz.MAL_SHADES[0], ec=qz.MAL_SHADES[0]))
            ax.annotate("", xy=(xc, s * (R - 0.5)), xytext=(xc, 2.0 * s),
                        arrowprops=dict(arrowstyle="-|>", color=qz.MAL_SHADES[0], lw=0.8, mutation_scale=7))

    def body(ax, x0, x1):
        ax.plot([x0, x1], [0, 0], color=qz.INK, lw=1.4)

    # open gripper: short head shield, main beam, grippers, bolts behind
    ax = axes[0]
    shield(ax, -2.2, -0.35)
    body(ax, -14, -0.35)
    grippers(ax, -8.0)
    cylinders(ax, -7.0, -2.4)
    for xb in (-4.0, -11.0, -13.0, -15.0, -17.0, -19.0):
        for s in (1, -1):
            ax.plot([xb, xb], [s * R, s * (R + 1.2)], color=qz.INK, lw=1.0)
    ax.plot([-21.5, -12], [R - 0.08, R - 0.08], color=qz.GREY, lw=1.6)
    ax.plot([-21.5, -12], [-R + 0.08, -R + 0.08], color=qz.GREY, lw=1.6)
    # single shield: shield, thrust on segments
    ax = axes[1]
    shield(ax, -9.0, -0.35)
    cylinders(ax, -8.8, -2.8)
    segments(ax, -23.0, -8.9, 9)
    body(ax, -9, -0.35)
    ax.add_patch(mp.Rectangle((-8.6, -1.0), 0.4, 2.0, fc=qz.GREY, ec="none"))   # erector
    # double shield: front shield, telescopic joint, gripper shield, tail
    ax = axes[2]
    shield(ax, -5.2, -0.35)
    shield(ax, -12.5, -5.8)
    for s in (1, -1):
        ax.plot([-5.8, -5.2], [s * (R - 0.2), s * (R - 0.2)], color=qz.INK, lw=0.8)
    grippers(ax, -9.0)
    cylinders(ax, -11.8, -5.4)
    segments(ax, -23.0, -12.3, 7)
    body(ax, -12.4, -0.35)
    # earth pressure balance: chamber, bulkhead, screw, shield, segments
    ax = axes[3]
    ax.add_patch(mp.Rectangle((-1.6, -R + 0.3), 1.25, 2 * R - 0.6, fc=qz.CAR_TINT, ec="none"))
    ax.plot([-1.6, -1.6], [-R + 0.3, R - 0.3], color=qz.INK, lw=1.4)
    shield(ax, -9.5, -0.35)
    cylinders(ax, -9.3, -2.8)
    segments(ax, -23.0, -9.4, 9)
    ax.plot([-1.0, -9.0], [-2.2, 0.6], color=qz.INK, lw=3.0, solid_capstyle="butt")   # screw conveyor
    ax.plot([-1.0, -9.0], [-2.2, 0.6], color=qz.CAR_TINT, lw=1.2, ls=(0, (1.5, 1.5)))
    # label one instance of each component on the last panel and the first
    lbl = dict(fontsize=8, ha="center", va="center", color=qz.INK)
    axes[0].annotate("Grippers", xy=(-8.0, -R - 0.1), xytext=(-8.0, -4.15), arrowprops=dict(arrowstyle="-", lw=0.5), **lbl)
    axes[0].annotate("Bolts", xy=(-19.0, R + 0.9), xytext=(-22.0, 4.0), arrowprops=dict(arrowstyle="-", lw=0.5), **lbl)
    axes[1].annotate("Segments", xy=(-16.0, -R + 0.2), xytext=(-16.0, -4.15), arrowprops=dict(arrowstyle="-", lw=0.5), **lbl)
    axes[1].annotate("Shield", xy=(-5.0, R - 0.2), xytext=(-5.0, 4.15), arrowprops=dict(arrowstyle="-", lw=0.5), **lbl)
    axes[1].annotate("Thrust cylinders", xy=(-8.5, -1.6), xytext=(-14.0, -1.6), arrowprops=dict(arrowstyle="-", lw=0.5), **lbl)
    axes[2].annotate("Telescopic joint", xy=(-5.5, R - 0.2), xytext=(-5.5, 4.15), arrowprops=dict(arrowstyle="-", lw=0.5), **lbl)
    axes[3].annotate("Pressurised chamber", xy=(-1.0, 1.5), xytext=(-5.5, 4.15), arrowprops=dict(arrowstyle="-", lw=0.5), **lbl)
    axes[3].annotate("Screw conveyor", xy=(-8.5, 0.43), xytext=(-13.5, 1.5), arrowprops=dict(arrowstyle="-", lw=0.5), **lbl)
    axes[0].annotate("Cutterhead", xy=(0.0, R - 0.4), xytext=(1.9, 4.2), arrowprops=dict(arrowstyle="-", lw=0.5), **lbl)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.06)
    _save(fig, "ch23", "tbm-families")


def advance_rate():
    """Daily advance against utilisation, for three penetration rates."""
    u = np.linspace(0, 0.6, 121)
    fig, ax = qz.figure(W_HALF, 0.80 * W_HALF)
    cols = {"utilisation": u}
    for pr, col, ls in ((2.0, qz.MAL_SHADES[3], "--"), (3.0, qz.MAL, "-"), (4.0, qz.MAL_SHADES[0], ":")):
        adv = pr * u * 24
        ax.plot(u * 100, adv, color=col, ls=ls, lw=1.3, label=f"{pr:g} m/h")
        cols[f"advance_pr{pr:g}"] = adv
    ax.axvspan(0, 20, color=qz.CAR_TINT, lw=0)
    ax.axvspan(40, 50, color=qz.MAL_TINT, lw=0)
    for uu in (0.18, 0.35):
        ax.plot([uu * 100], [3.0 * uu * 24], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9, zorder=5)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)
    qz.label(ax, "Utilisation, $U$ (%)", "Advance rate (m/day)")
    ax.legend(loc="upper left", title="Penetration rate", title_fontsize=8)
    fig.tight_layout()
    _save(fig, "ch23", "advance-rate", cols)


def specific_energy():
    """Rotary specific energy against penetration rate, 6 m head at 8 rev/min."""
    A = math.pi * 3.0 ** 2
    N = 8 / 60
    u = np.linspace(0.5, 6.0, 200)                  # m/h
    fig, ax = qz.figure(W_HALF, 0.80 * W_HALF)
    cols = {"u_m_per_h": u}
    for T, col, ls in ((1500, qz.MAL_SHADES[3], "--"), (2000, qz.MAL, "-"), (2500, qz.CAR, ":")):
        es = 2 * math.pi * N * T / (A * u / 3600) / 1000        # MJ/m3
        ax.plot(u, es, color=col, ls=ls, lw=1.3, label=f"$T = {T}$ kN m")
        cols[f"es_T{T}_MJ_m3"] = es
    e0 = 2 * math.pi * N * 2000 / (A * 3 / 3600) / 1000
    print(f"  worked point: P {2 * math.pi * N * 2000:.0f} kW, Q {A * 3:.1f} m3/h, e_s {e0:.1f} MJ/m3")
    ax.plot([3.0], [e0], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9, zorder=5)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 250)
    qz.label(ax, "Penetration rate, $u$ (m/h)", "Specific energy, $e_s$ (MJ/m$^3$)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "ch23", "specific-energy", cols)


def face_pressure():
    """Face pressure window over the face height for a slurry machine (schematic numbers)."""
    zc, D = 10.0, 10.0                    # crown depth and face height (m)
    gw, g, gs = 10.0, 20.0, 12.0          # water, ground and slurry unit weights (kN/m3)
    K = 0.35                              # effective earth pressure ratio for the collapse bound
    z = np.linspace(zc, zc + D, 50)
    u = gw * z                            # water table at the surface
    lower = u + K * (g - gw) * z
    upper = g * z
    pc_min = max(lower[-1] - gs * D, lower[0])
    pc_max = upper[0]
    pc = 0.5 * (pc_min + pc_max)
    slurry = pc + gs * (z - zc)
    print(f"  crown: lower {lower[0]:.0f}, upper {upper[0]:.0f}; invert lower {lower[-1]:.0f}; "
          f"slurry crown window {pc_min:.0f}-{pc_max:.0f} kPa; slurry chosen {pc:.0f} at crown, "
          f"{slurry[-1]:.0f} at invert")

    fig, ax = qz.figure(W_FULL, 0.50 * W_FULL)
    ax.fill_betweenx(z, lower, upper, color=qz.MAL_TINT, lw=0)
    ax.plot(u, z, color=qz.GREY, lw=1.0, ls=":", label="Water pressure")
    ax.plot(lower, z, color=qz.CAR, lw=1.3, ls="--", label="Collapse bound")
    ax.plot(upper, z, color=qz.CAR_SHADES[0], lw=1.3, ls="-.", label="Blow-out bound")
    ax.plot(slurry, z, color=qz.MAL, lw=1.6, label="Slurry pressure")
    ax.set_ylim(zc + D + 0.5, zc - 0.5)
    ax.set_xlim(0, 450)
    ax.set_yticks([10, 12, 14, 16, 18, 20])
    qz.label(ax, "Pressure, $p$ (kPa)", "Depth, $z$ (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "ch23", "face-pressure-profile",
          {"z_m": z, "water_kPa": u, "collapse_kPa": lower, "blowout_kPa": upper, "slurry_kPa": slurry})


def annulus_grout():
    """Annulus between bore and segment extrados, and the volume loss implied by a grout shortfall."""
    Db, De, Lr = 6.2, 6.0, 1.5
    v_ann = math.pi / 4 * (Db ** 2 - De ** 2) * Lr
    v_exc = math.pi / 4 * Db ** 2 * Lr
    print(f"  annulus {v_ann:.2f} m3, excavated {v_exc:.1f} m3, 0.5 m3 -> {0.5 / v_exc * 100:.2f} %")

    fig = plt.figure(figsize=(W_FULL, 0.45 * W_FULL))
    a1 = fig.add_axes([0.0, 0.04, 0.40, 0.92])
    a2 = fig.add_axes([0.53, 0.19, 0.45, 0.77])
    _blank(a1)
    a1.set_xlim(-4.3, 4.3)
    a1.set_ylim(-4.0, 4.0)
    # schematic: annulus drawn at about three times its true width
    rb, re, ri = 3.1 + 0.2, 3.0, 2.7
    a1.add_patch(mp.Circle((0, 0), 3.95, fc="white", ec="none", hatch="////", lw=0))
    a1.add_patch(mp.Circle((0, 0), rb, fc=qz.CAR_TINT, ec=qz.GREY, lw=0.6))
    a1.add_patch(mp.Circle((0, 0), re, fc=qz.GREY_LIGHT, ec=qz.INK, lw=0.6))
    a1.add_patch(mp.Circle((0, 0), ri, fc="white", ec=qz.INK, lw=0.6))
    for k in range(6):
        ang = math.radians(k * 60 + 15)
        a1.plot([ri * math.cos(ang), re * math.cos(ang)], [ri * math.sin(ang), re * math.sin(ang)],
                color=qz.INK, lw=0.6)
    a1.annotate("", xy=(-rb, 0), xytext=(rb, 0),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=qz.INK, mutation_scale=6, shrinkA=0, shrinkB=0))
    a1.text(0, 0.15, "Bore 6.2 m", ha="center", va="bottom", fontsize=8)
    a1.annotate("", xy=(-re, -0.9), xytext=(re, -0.9),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=qz.INK, mutation_scale=6, shrinkA=0, shrinkB=0))
    a1.text(0, -1.05, "Extrados 6.0 m", ha="center", va="top", fontsize=8)
    a1.annotate("Annulus", xy=(rb * 0.72, rb * 0.69), xytext=(3.2, 3.75), fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.5))
    rings = np.linspace(0, v_ann, 100)
    vl = rings / v_exc * 100
    a2.axhspan(0.5, 1.5, color=qz.MAL_TINT, lw=0)
    a2.plot(rings, vl, color=qz.CAR, lw=1.4)
    a2.plot([0.5], [0.5 / v_exc * 100], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9, zorder=5)
    a2.axvline(v_ann, color=qz.GREY, lw=0.6, ls=":")
    a2.set_xlim(0, 3.0)
    a2.set_ylim(0, 7.0)
    qz.label(a2, "Grout shortfall (m$^3$/ring)", "Volume loss, $V_L$ (%)")
    _save(fig, "ch23", "annulus-grout", {"shortfall_m3": rings, "VL_pct": vl})


ALL = [characteristic_value, ec7_category, squeezing_uncertainty, empirical_vs_ccm, controlled_deformation,
       bayesian_update, advance_core, cavern_shapes, blast_round, disturbance_strength, vibration_criteria,
       continuous_vibration, tbm_families, advance_rate, specific_energy, face_pressure, annulus_grout]


if __name__ == "__main__":
    qz.use()
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    for fn in ALL:
        if key in fn.__name__:
            print(fn.__name__)
            fn()
