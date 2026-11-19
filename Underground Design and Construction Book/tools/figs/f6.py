"""Figures for Chapter 16 (analytical methods), Chapter 17 (numerical modelling)
and Appendix F (closed-form solutions): group f6.

Every quantity is computed from the book's own equations, through the
``geo_platform`` package at the repository root where it holds the
calculation, so a figure and the worked example it illustrates cannot drift
apart. The canonical deep-tunnel numbers (u_max 126.9 mm, u_s0 62.0 mm,
FoS 2.8 and 4.3) are asserted before anything is drawn. Figures are drawn at
their printed width (117 mm, the SNmono text block) in the house style of
``tools/qz_figure.py`` and written as PDF and 300 dpi PNG, with the plotted
data as CSV beside them.

    python tools/figs/f6.py              # all figures
    python tools/figs/f6.py surrogate    # figures whose name contains "surrogate"
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

import qz_figure as qz  # noqa: E402
from geo_platform import ccm, settlement  # noqa: E402

FIG = os.path.join(BOOK, "figures")
W_FULL = 117 / 25.4          # SNmono text width
W_HALF = 0.62 * W_FULL


def _save(fig, chapter: str, stem: str, columns: dict[str, np.ndarray] | None) -> None:
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


# ---------------------------------------------------------------------------
# The deep tunnel at 650 m (Chapter 16, Table ccm-worked-example)
# ---------------------------------------------------------------------------

G30 = ccm.Ground(16.9, 4.75, 0.22, 2.12, 23.8, 1.72)
G60 = ccm.Ground(16.9, 4.75, 0.22, 13.52, 31.5, 2.85)
RING = ccm.support_ring(4.75, 0.5, 60, 40, 0.25)


def _check_canonical():
    r30 = ccm.analyse(G30, [RING])
    r60 = ccm.analyse(G60, [RING])
    assert abs(r30.u_max - 126.9) < 0.05, r30.u_max
    assert abs(r30.us0 - 62.0) < 0.05, r30.us0
    assert abs(r30.eq.p_eq - 2.27) < 0.005 and round(r30.eq.fos, 1) == 2.8, r30.eq
    assert abs(r60.eq.p_eq - 1.47) < 0.005 and round(r60.eq.fos, 1) == 4.3, r60.eq
    return r30, r60


# ---------------------------------------------------------------------------
# Chapter 16: stresses around the plastic zone and the plastic radius
# ---------------------------------------------------------------------------

def _stresses(g: ccm.Ground, r_over_a: np.ndarray):
    """Radial and tangential stress (MPa) of the unsupported tunnel, plastic and elastic zones."""
    k, scm = g.k_phi, g.scm_mc
    rp = ccm.plastic_radius(g, 0.0) / g.a
    sr = np.where(r_over_a <= rp, scm / (k - 1) * (r_over_a ** (k - 1) - 1),
                  g.p0 - (g.p0 - g.p_cr) * (rp / r_over_a) ** 2)
    st = np.where(r_over_a <= rp, k * sr + scm, g.p0 + (g.p0 - g.p_cr) * (rp / r_over_a) ** 2)
    return sr, st, rp


def plastic_zone():
    r30, r60 = _check_canonical()
    fig, (a1, a2) = qz.figure(W_FULL, 0.47 * W_FULL, ncols=2)
    r = np.linspace(1, 5, 401)
    cols = {"r_over_a": r}
    for g, gsi, col in ((G60, 60, qz.MAL), (G30, 30, qz.CAR)):
        sr, st, rp = _stresses(g, r)
        a1.plot(r, st, color=col, lw=1.3, label=f"$\\sigma_\\theta$, GSI {gsi}")
        a1.plot(r, sr, color=col, lw=1.1, ls="--", label=f"$\\sigma_r$, GSI {gsi}")
        a1.axvline(rp, color=col, lw=0.6, ls=":")
        cols[f"sigma_theta_gsi{gsi}"], cols[f"sigma_r_gsi{gsi}"] = st, sr
        print(f"  GSI {gsi}: R_p/a {rp:.3f}, peak sigma_theta {st.max():.2f} MPa")
    a1.axhline(G30.p0, color=qz.GREY, lw=0.6, ls=":")
    a1.set_xlim(1, 5)
    a1.set_ylim(0, 50)
    qz.label(a1, "Radial distance, $r/a$ (-)", "Stress, $\\sigma$ (MPa)")
    a1.legend(loc="upper right", ncol=1, handlelength=1.8)

    for g, gsi, col, res in ((G60, 60, qz.MAL, r60), (G30, 30, qz.CAR, r30)):
        p = np.linspace(0, 12, 481)
        rp = np.array([ccm.plastic_radius(g, float(x)) for x in p]) / g.a
        a2.plot(p, rp, color=col, lw=1.3, ls="-" if gsi == 60 else "--", label=f"GSI {gsi}")
        a2.plot([res.eq.p_eq], [res.rp_eq / g.a], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9)
        cols[f"p_i_MPa"] = p
        cols[f"Rp_over_a_gsi{gsi}"] = rp
        print(f"  GSI {gsi}: R_p at p_eq {res.rp_eq:.2f} m ({res.rp_eq / g.a:.2f} a)")
    a2.set_xlim(0, 12)
    a2.set_ylim(0.9, 2.5)
    qz.label(a2, "Support pressure, $p_i$ (MPa)", "Plastic radius, $R_p/a$ (-)")
    a2.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "ch16", "plastic-zone", cols)


# ---------------------------------------------------------------------------
# Chapter 16: installation distance against factor of safety and convergence
# ---------------------------------------------------------------------------

def install_distance():
    _check_canonical()
    dist = np.linspace(0, 12, 121)
    fig, (a1, a2) = qz.figure(W_FULL, 0.47 * W_FULL, ncols=2)
    cols = {"L_m": dist}
    a1.axhspan(1.5, 2.0, color=qz.GREY_LIGHT, alpha=0.5, lw=0, zorder=0)
    for g, gsi, col, ls in ((G60, 60, qz.MAL, "-"), (G30, 30, qz.CAR, "--")):
        res = [ccm.analyse(g, [RING], value=float(x)) for x in dist]
        fos = np.array([r.eq.fos for r in res])
        strain = np.array([r.eq.u_eq for r in res]) / (g.a * 1000) * 100
        peq = np.array([r.eq.p_eq for r in res])
        a1.plot(dist, fos, color=col, ls=ls, lw=1.3, label=f"GSI {gsi}")
        a2.plot(dist, strain, color=col, ls=ls, lw=1.3, label=f"GSI {gsi}")
        r3 = ccm.analyse(g, [RING], value=3.0)
        a1.plot([3.0], [r3.eq.fos], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9)
        a2.plot([3.0], [r3.eq.u_eq / (g.a * 1000) * 100], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9)
        cols[f"FoS_gsi{gsi}"], cols[f"ueq_over_a_pct_gsi{gsi}"], cols[f"peq_MPa_gsi{gsi}"] = fos, strain, peq
        print(f"  GSI {gsi}: FoS at L=0 {fos[0]:.2f}, L=3 {r3.eq.fos:.2f}, L=12 {fos[-1]:.2f}; "
              f"strain {strain[0]:.2f} to {strain[-1]:.2f} %")
    a1.set_xlim(0, 12)
    a1.set_yscale("log")
    a1.set_ylim(1, 50)
    a1.set_yticks([1, 2, 5, 10, 20, 50])
    a1.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    a1.minorticks_off()
    a1.set_xticks([0, 3, 6, 9, 12])
    qz.label(a1, "Installation distance, $L$ (m)", "Factor of safety, FoS (-)")
    a1.legend(loc="upper left")
    a2.set_xlim(0, 12)
    a2.set_xticks([0, 3, 6, 9, 12])
    a2.set_ylim(0, 2.5)
    qz.label(a2, "Installation distance, $L$ (m)", "Convergence, $u_{eq}/a$ (%)")
    a2.legend(loc="center right")
    fig.tight_layout()
    _save(fig, "ch16", "install-distance", cols)


# ---------------------------------------------------------------------------
# Chapter 16: the motorway tunnel, three installation criteria
# ---------------------------------------------------------------------------

def motorway_timing():
    phi, scm = 25.64, 0.47
    s = math.sin(math.radians(phi))
    c = scm * (1 - s) / (2 * math.cos(math.radians(phi)))
    g = ccm.Ground(1.62, 6.0, 0.3, 0.353, phi, c)
    df, sal = ccm.GRC_METHODS[1], ccm.GRC_METHODS[0]
    ks, psm = 29.5, 0.354                       # MPa/m, MPa
    usm = psm / ks * 1000                        # 12.0 mm
    fig, ax = qz.figure(W_FULL, 0.56 * W_FULL)
    p, u_df = ccm.grc_curve(g, df)
    _, u_sal = ccm.grc_curve(g, sal)
    ax.plot(u_df, p, color=qz.MAL, lw=1.5, label="Ground reaction")
    ax.plot(u_sal, p, color=qz.GREY, lw=1.0, ls=":", label="Ground reaction, simplified")
    cols = {"p_MPa": p, "u_duncanfama_mm": u_df, "u_simplified_mm": u_sal}
    # published analysis (Table support-timing and the text): u_s0, u_eq, p_eq
    cases = ((53.67, 59.89, 0.183, "$L = 3$ m", qz.CAR, "--"),
             (40.0, 48.69, 0.256, "$u_{s0} = 40$ mm", qz.CAR_SHADES[3], "-."),
             (18.0, 38.3, 0.354, "$\\varepsilon = 0.3$ %", qz.CAR_DARK, "-"),
             (86.30, 88.62, 0.354 / 5.15, "$L = 3$ m, elastic profile", qz.GREY, "--"))
    for us0, ueq, peq, name, col, ls in cases:
        u_end = max(ueq, us0 + usm) + 6
        uu = np.array([us0, us0 + usm, u_end])
        pp = np.array([0.0, psm, psm])
        ax.plot(uu, pp, color=col, lw=1.2, ls=ls, label=name)
        ax.plot([ueq], [peq], "o", mfc="white", mec=qz.INK, ms=4.2, mew=0.9, zorder=5)
        e = ccm.equilibrium(g, ccm.Support("bolts", psm, ks, usm), us0, df)
        print(f"  {name}: published u_eq {ueq} p_eq {peq:.3f}; closed form u_eq {e.u_eq:.2f} "
              f"p_eq {e.p_eq:.3f}")
        cols[f"support_{us0:g}_u"], cols[f"support_{us0:g}_p"] = uu, pp
    # the 0.3 per cent case: the support line extended beyond yield, to its notional intersection
    ax.plot([30.0, 32.6], [psm, 0.43], color=qz.CAR_DARK, lw=0.7, ls=":")
    ax.plot([32.6], [0.43], "o", mfc="white", mec=qz.CAR_DARK, ms=3.2, mew=0.8, zorder=5)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 1.0)
    qz.label(ax, "Wall displacement, $u_r$ (mm)", "Support pressure, $p_i$ (MPa)")
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    _save(fig, "ch16", "motorway-timing", cols)


# ---------------------------------------------------------------------------
# Chapter 16: combined and sequential support
# ---------------------------------------------------------------------------

def _elements(a=4.75):
    """Illustrative elements within the ranges of Appendix G, for the 4.75 m radius."""
    shot = ccm.support_ring(a, 0.20, 30.0, 25.0, 0.20, "Shotcrete")
    area, fy, sp, es = 59e-4, 275.0, 1.0, 200.0
    sets = ccm.Support("Steel sets", area * fy / (sp * a), es * 1000 * area / (sp * a ** 2), 0.0)
    sets = ccm.Support(sets.name, sets.psm, sets.ks, sets.psm / sets.ks * 1000)
    t_bolt, spacing, strain = 0.250, 1.0, 0.0024
    b_psm = t_bolt / spacing ** 2
    b_usm = strain * a * 1000
    bolts = ccm.Support("Bolts", b_psm, b_psm / b_usm * 1000, b_usm)
    return shot, sets, bolts


def combined_support():
    shot, sets, bolts = _elements()
    elems = (shot, sets, bolts)
    for e in elems:
        print(f"  {e.name}: p_sm {e.psm:.3f} MPa, k_s {e.ks:.1f} MPa/m, u_sm {e.usm:.2f} mm")
    u = np.linspace(0, 14, 561)
    fig, (a1, a2) = qz.figure(W_FULL, 0.5 * W_FULL, ncols=2, sharey=True)
    cols = {"du_mm": u}
    styles = {"Shotcrete": "-", "Steel sets": "--", "Bolts": ":"}
    total_cap = sum(e.psm for e in elems)
    for ax, starts, tag in ((a1, (0.0, 0.0, 0.0), "together"), (a2, (0.0, 4.0, 2.0), "sequence")):
        parts = []
        for e, u0 in zip(elems, starts):
            pe = np.clip(e.ks * np.clip(u - u0, 0, None) / 1000, 0, e.psm)
            parts.append(pe)
            ax.plot(u, pe, color=qz.GREY, lw=0.9, ls=styles[e.name], label=e.name)
            cols[f"{tag}_{e.name.replace(' ', '_')}"] = pe
        # capacity is reached when the first element reaches its own limit
        u_first = min(u0 + e.usm for e, u0 in zip(elems, starts))
        total = np.sum(parts, axis=0)
        p_cap = float(np.interp(u_first, u, total))
        comb = np.where(u <= u_first, total, p_cap)
        ax.plot(u, comb, color=qz.MAL, lw=1.6, label="Combined")
        ax.plot([u_first], [p_cap], "o", mfc="white", mec=qz.MAL, ms=4.5, mew=1.0, zorder=5)
        ax.axhline(total_cap, color=qz.CAR, lw=0.9, ls=(0, (1, 1.5)), label="Sum of capacities")
        cols[f"{tag}_combined"] = comb
        print(f"  {tag}: combined capacity {p_cap:.3f} MPa at {u_first:.2f} mm "
              f"(sum of capacities {total_cap:.3f} MPa)")
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 2.2)
        qz.label(ax, "Displacement, $u_r - u_{s0}$ (mm)")
        ax.text(0.07, 0.95, "Together" if tag == "together" else "In sequence", transform=ax.transAxes,
                ha="left", va="top")
        # direct labels at the right-hand end of each curve
        kw = dict(ha="right", fontsize=8, color=qz.GREY)
        ax.text(13.8, total_cap + 0.05, "Sum of capacities", va="bottom", **{**kw, "color": qz.CAR})
        ax.text(13.8, p_cap + 0.05, "Combined", va="bottom", **{**kw, "color": qz.MAL})
        if tag == "together":
            ax.text(13.8, shot.psm + 0.04, "Shotcrete", va="bottom", **kw)
            ax.text(13.8, sets.psm + 0.03, "Steel sets", va="bottom", **kw)
            ax.text(13.8, bolts.psm - 0.03, "Bolts", va="top", **kw)
    qz.label(a1, y="Support pressure, $p_i$ (MPa)")
    fig.tight_layout()
    _save(fig, "ch16", "combined-support", cols)


# ---------------------------------------------------------------------------
# Chapter 16: stiff, soft and yielding support on the same ground
# ---------------------------------------------------------------------------

def stiff_yielding():
    r30, _ = _check_canonical()
    g, us0 = G30, r30.us0
    soft = ccm.support_ring(4.75, 0.5, 60, 10, 0.25, "Soft ring")        # young concrete, E_c = 10 GPa
    p_yield = 1.0                                                        # MPa, yielding support
    fig, ax = qz.figure(W_FULL, 0.5 * W_FULL)
    p, u = ccm.grc_curve(g)
    ok = np.isfinite(u)
    ax.plot(u[ok], p[ok], color=qz.MAL, lw=1.5, label="Ground reaction, GSI 30")
    cols = {"p_MPa": p[ok], "u_mm": u[ok]}
    uu = np.linspace(us0, 140, 600)
    for sup, name, col, ls in ((RING, "Stiff ring", qz.CAR, "-"), (soft, "Soft ring", qz.CAR_SHADES[3], "--")):
        ps = np.clip(sup.ks * (uu - us0) / 1000, 0, sup.psm)
        ax.plot(uu, ps, color=col, lw=1.2, ls=ls, label=name)
        e = ccm.equilibrium(g, sup, us0)
        ax.plot([e.u_eq], [e.p_eq], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9, zorder=5)
        cols[f"{name.replace(' ', '_')}_p"] = ps
        print(f"  {name}: k_s {sup.ks:.1f} MPa/m, p_eq {e.p_eq:.2f} MPa, u_eq {e.u_eq:.1f} mm, FoS {e.fos:.2f}")
    py = np.clip(RING.ks * (uu - us0) / 1000, 0, p_yield)
    u_rest = ccm.grc_displacement(g, p_yield)
    ax.plot(uu, py, color=qz.CAR_DARK, lw=1.2, ls=":", label="Yielding support")
    ax.plot([u_rest], [p_yield], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9, zorder=5)
    cols["yielding_p"] = py
    cols["support_u_mm"] = uu
    print(f"  yielding support: rest at {u_rest:.1f} mm, p {p_yield} MPa")
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 8)
    qz.label(ax, "Wall displacement, $u_r$ (mm)", "Support pressure, $p_i$ (MPa)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.47))
    fig.tight_layout()
    _save(fig, "ch16", "stiff-yielding", cols)


# ---------------------------------------------------------------------------
# Chapter 17: four settlement troughs of the teaching example
# ---------------------------------------------------------------------------

def trough_cases():
    y = np.linspace(-30, 30, 601)
    fig, ax = qz.figure(W_FULL, 0.45 * W_FULL)
    cols = {"y_m": y}
    for z0, col in ((5.0, qz.CAR), (20.0, qz.MAL)):
        for k, ls, ground in ((0.5, "-", "stiff clay"), (0.25, "--", "rock")):
            b = settlement.Bore(0.0, z0, 6.0, 1.0, k)
            s = settlement.transverse([b], y)
            ax.plot(y, -s, color=col, ls=ls, lw=1.3, label=f"$z_0 = {z0:g}$ m, $K = {k:g}$")
            cols[f"S_z{z0:g}_K{k:g}_mm"] = s
            print(f"  z0 {z0:g} m, K {k:g}: i {b.i:.2f} m, S_max {b.smax_mm:.1f} mm")
    ax.axhline(0, color=qz.GREY_LIGHT, lw=0.6)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-100, 8)
    qz.label(ax, "Offset from the axis, $y$ (m)", "Settlement, $S$ (mm)")
    ax.legend(loc="lower right", ncol=1)
    fig.tight_layout()
    _save(fig, "ch17", "trough-cases", cols)


# ---------------------------------------------------------------------------
# Chapter 17: five sources of volume loss (schematic)
# ---------------------------------------------------------------------------

def _num(ax, x, y, n, col=qz.INK):
    ax.plot([x], [y], "o", ms=10, mfc="white", mec=col, mew=0.8, zorder=8)
    ax.text(x, y, str(n), ha="center", va="center_baseline", fontsize=8, color=col, zorder=9)


def volume_loss_sources():
    fig = plt.figure(figsize=(W_FULL, 0.42 * W_FULL))
    ax = fig.add_axes([0.0, 0.0, 0.72, 1.0])
    bx = fig.add_axes([0.74, 0.0, 0.26, 1.0])
    ax.set_aspect(1.5)                       # vertical exaggeration of the longitudinal section
    bx.set_aspect("equal")
    for a in (ax, bx):
        a.axis("off")
    # radii exaggerated so the gaps can be seen: cut profile, shield front and tail, lining extrados
    R_ex, R_sh0, R_sh1, R_li, t_li = 3.70, 3.36, 3.16, 2.85, 0.40
    x_face, x_tail, x_end = 0.0, -9.0, -30.0
    surf = 9.0
    # ground surface and the settled surface (schematic, exaggerated)
    xs = np.linspace(x_end, 9, 400)
    ax.plot(xs, np.full_like(xs, surf), color=qz.INK, lw=0.9)
    from scipy.stats import norm
    ax.plot(xs, surf - 1.2 * norm.cdf((x_face - 2 - xs) / 5.0), color=qz.CAR, lw=1.0, ls="--")
    # the cut profile and the face
    for sgn in (1, -1):
        ax.plot([x_end, x_face], [sgn * R_ex] * 2, color=qz.GREY, lw=0.6, ls=(0, (3, 2)))
    ax.plot([x_face, x_face], [-R_ex, R_ex], color=qz.INK, lw=0.8)
    # the ground boundary closing in: onto the shield, then into the tail void until the grout acts
    xg = np.linspace(x_end, x_face, 400)
    rb = np.interp(xg, [x_end, x_tail - 3.5, x_tail, x_face - 1.5, x_face],
                   [R_li + 0.18, R_li + 0.18, R_sh1 + 0.03, R_ex - 0.15, R_ex])
    for sgn, f in ((1, 1.0), (-1, 0.3)):                     # the invert closes less than the crown
        r_def = R_ex - f * (R_ex - rb)
        ax.fill_between(xg, sgn * r_def, sgn * R_ex, color=qz.CAR_TINT, lw=0, zorder=1)
        ax.plot(xg, sgn * r_def, color=qz.CAR, lw=0.9, zorder=2)
        behind = xg <= x_tail
        ax.fill_between(xg[behind], sgn * R_li, sgn * r_def[behind], color=qz.MAL_SHADES[3], lw=0, zorder=1)
    # shield (tapered) and cutterhead
    ax.add_patch(patches.Polygon([[x_face - 0.45, R_sh0], [x_tail, R_sh1], [x_tail, -R_sh1],
                                  [x_face - 0.45, -R_sh0]], closed=True, fc=qz.GREY_LIGHT, ec=qz.INK,
                                 lw=0.8, zorder=3))
    ax.add_patch(patches.Rectangle((x_face - 0.45, -R_ex), 0.45, 2 * R_ex, fc=qz.GREY, ec=qz.INK,
                                   lw=0.7, zorder=4))
    # lining rings behind the shield
    ring_w = 1.5
    for k in range(int((x_tail - x_end) / ring_w)):
        x0 = x_tail - (k + 1) * ring_w
        for sgn in (1, -1):
            ax.add_patch(patches.Rectangle((x0, R_li - t_li if sgn > 0 else -R_li), ring_w, t_li,
                                           fc="white", ec=qz.INK, lw=0.5, zorder=3))
    arrow = dict(arrowstyle="-|>", color=qz.CAR, lw=0.8, mutation_scale=7)
    # 1: face, movement towards the face
    for yy in (-1.8, 0, 1.8):
        ax.annotate("", xy=(x_face + 0.2, yy), xytext=(x_face + 2.6, yy), arrowprops=arrow)
    _num(ax, x_face + 4.0, 0.0, 1)
    # 2: overcut along the shield
    ax.annotate("", xy=(-4.5, R_ex - 0.05), xytext=(-4.5, R_ex + 1.9), arrowprops=arrow)
    _num(ax, -4.5, R_ex + 2.8, 2)
    # 3: tail void
    ax.annotate("", xy=(x_tail - 1.5, R_li + 0.35), xytext=(x_tail - 1.5, R_ex + 1.9), arrowprops=arrow)
    _num(ax, x_tail - 1.5, R_ex + 2.8, 3)
    # 5: consolidation, seepage towards the lining long behind the face
    for xx in (-27.0, -24.0):
        for sgn in (1, -1):
            ax.annotate("", xy=(xx, sgn * (R_ex + 0.2)), xytext=(xx, sgn * (R_ex + 2.4)),
                        arrowprops=dict(arrowstyle="-|>", color=qz.MAL, lw=0.8, mutation_scale=7,
                                        ls=(0, (2, 1.5))))
    _num(ax, -21.0, R_ex + 2.8, 5, qz.MAL)
    # direction of advance
    ax.annotate("", xy=(8.5, -R_ex - 2.0), xytext=(4.0, -R_ex - 2.0),
                arrowprops=dict(arrowstyle="-|>", color=qz.INK, lw=0.8, mutation_scale=8))
    ax.text(3.6, -R_ex - 2.0, "Advance", ha="right", va="center", fontsize=8)
    ax.annotate("Grout", xy=(-16.0, -R_li - 0.3), xytext=(-16.0, -R_ex - 2.0), ha="center", va="center",
                fontsize=8, color=qz.MAL,
                arrowprops=dict(arrowstyle="-", color=qz.MAL, lw=0.6, shrinkA=1, shrinkB=0))
    ax.set_xlim(x_end - 0.5, 9.5)
    ax.set_ylim(-R_ex - 3.0, surf + 0.6)

    # cross-section through the lining: annulus, ovalised ring
    th = np.linspace(0, 2 * np.pi, 400)
    d = 0.22
    rr = R_li * (1 + d / R_li * np.cos(2 * th))                    # squat: wider at the springline
    rr2 = (R_li - t_li) * (1 + d / R_li * np.cos(2 * th))
    bx.add_patch(patches.Circle((0, 0), R_li + 0.16, fc=qz.MAL_SHADES[3], ec=qz.CAR, lw=0.9))
    bx.fill(rr * np.cos(th), rr * np.sin(th), fc="white", ec=qz.INK, lw=1.0)
    bx.fill(rr2 * np.cos(th), rr2 * np.sin(th), fc="white", ec=qz.INK, lw=1.0)
    bx.add_patch(patches.Circle((0, 0), R_li, fill=False, ec=qz.GREY, lw=0.6, ls=(0, (3, 2))))
    bx.annotate("", xy=(0, R_li - d - 0.05), xytext=(0, R_li + 1.9),
                arrowprops=dict(arrowstyle="-|>", color=qz.CAR, lw=0.8, mutation_scale=7))
    _num(bx, 0, R_ex + 2.8, 4)
    bx.set_xlim(-4.4, 4.4)
    bx.set_ylim(-R_ex - 1.5, R_ex + 4.0)
    _save(fig, "ch17", "volume-loss-sources", None)


# ---------------------------------------------------------------------------
# Chapter 17: model domain, boundary conditions and graded mesh (schematic)
# ---------------------------------------------------------------------------

def _graded_points(D, z0, half_w, depth, a, hs):
    """Point cloud for a graded triangulation: fine near the tunnel, coarse far away."""
    h_near, h_far = hs
    r_near, r_far = a + 2 * D, a + 4 * D

    def size(r):
        return float(np.interp(r, [r_near, r_far], [h_near, h_far]))

    pts = []
    # rings around the tunnel, graded outwards, clipped to the domain away from its edges
    r, k = a, 0
    while r < math.hypot(half_w, depth):
        h = size(r)
        n = max(12, int(round(2 * np.pi * r / h)))
        th = np.linspace(0, 2 * np.pi, n, endpoint=False) + (np.pi / n if k % 2 else 0.0)
        for t in th:
            x, y = r * np.cos(t), -z0 + r * np.sin(t)
            if abs(x) < half_w - 0.45 * h and -depth + 0.45 * h < y < -0.45 * h:
                pts.append((x, y))
        r += h * 0.87
        k += 1

    # boundary points, graded along the edges
    def edge(p0, p1):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        L = np.linalg.norm(p1 - p0)
        t, out = 0.0, []
        while t < L - 1e-9:
            q = p0 + (p1 - p0) * t / L
            out.append(tuple(q))
            t += size(math.hypot(q[0], q[1] + z0))
        return out
    pts += edge((-half_w, 0), (half_w, 0)) + edge((half_w, 0), (half_w, -depth))
    pts += edge((half_w, -depth), (-half_w, -depth)) + edge((-half_w, -depth), (-half_w, 0))
    return np.array(pts)


def model_domain():
    D, z0, contact = 6.0, 25.0, 2.5
    a = D / 2
    half_w = 5 * D
    depth = z0 + a + 5 * D
    pts = _graded_points(D, z0, half_w, depth, a, (1.1, 4.5))
    tri = Triangulation(pts[:, 0], pts[:, 1])
    cx = pts[tri.triangles].mean(axis=1)
    tri.set_mask(np.hypot(cx[:, 0], cx[:, 1] + z0) < a)
    fig, ax = qz.figure(W_FULL * 0.78, W_FULL * 0.80)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(patches.Rectangle((-half_w, -contact), 2 * half_w, contact, fc=qz.MAL_TINT, lw=0, zorder=0))
    ax.triplot(tri, color=qz.GREY_LIGHT, lw=0.35, zorder=1)
    ax.plot([-half_w, half_w], [-contact, -contact], color=qz.MAL, lw=0.8, zorder=2)
    ax.add_patch(patches.Rectangle((-half_w, -depth), 2 * half_w, depth, fill=False, ec=qz.INK, lw=0.9, zorder=3))
    ax.add_patch(patches.Circle((0, -z0), a, fc="white", ec=qz.CAR, lw=1.4, zorder=3))
    # zone within two diameters of the excavation
    ax.add_patch(patches.Circle((0, -z0), a + 2 * D, fill=False, ec=qz.GREY, lw=0.6, ls="--", zorder=2))
    # boundary conditions: rollers on the sides, fixed base
    rr = 0.75
    for x, sgn in ((-half_w, -1), (half_w, 1)):
        for y in np.arange(-depth + 4, -2, 6.0):
            ax.add_patch(patches.Circle((x + sgn * rr, y), rr, fc="white", ec=qz.INK, lw=0.6, zorder=4))
            ax.plot([x + sgn * 2 * rr] * 2, [y - 1.3, y + 1.3], color=qz.INK, lw=0.6)
    for x in np.arange(-half_w + 1, half_w, 2.0):
        ax.plot([x, x - 1.2], [-depth, -depth - 1.4], color=qz.INK, lw=0.5)
    # dimensions
    def dim(p0, p1, text, off, rot=0):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<|-|>", color=qz.INK, lw=0.6,
                                                          mutation_scale=6, shrinkA=0, shrinkB=0))
        mx, my = (p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1]
        ax.text(mx, my, text, ha="center", va="center", rotation=rot, fontsize=10,
                bbox=dict(fc="white", ec="none", pad=0.6))
    dim((0, -depth + 9), (half_w, -depth + 9), "$\\geq 5D$", (0, 0))
    dim((-half_w + 9, -z0 - a), (-half_w + 9, -depth), "$\\geq 5D$", (0, 0), rot=90)
    ax.plot([-half_w + 7.5, -1.5], [-z0 - a] * 2, color=qz.GREY, lw=0.5, ls=":")
    dim((half_w - 7, 0), (half_w - 7, -z0), "$z_0$", (0, 0))
    ax.plot([0.5, half_w - 6], [-z0] * 2, color=qz.GREY, lw=0.5, ls=":")
    ax.plot([0, 0], [-z0 - a - 0.5, -z0 + a + 0.5], color=qz.GREY, lw=0.4, ls="-.")
    ax.text(0, -z0, "$D$", ha="center", va="center", fontsize=10)
    t = math.radians(145)
    dim((a * math.cos(t), -z0 + a * math.sin(t)), ((a + 2 * D) * math.cos(t), -z0 + (a + 2 * D) * math.sin(t)),
        "$2D$", (0, 0))
    ax.set_xlim(-half_w - 3, half_w + 3)
    ax.set_ylim(-depth - 2.5, 1.0)
    print(f"  mesh: {len(pts)} nodes, {int((~tri.mask).sum())} triangles")
    _save(fig, "ch17", "model-domain", None)


# ---------------------------------------------------------------------------
# Chapter 17: boundary distance and mesh size, verified against the closed form
# ---------------------------------------------------------------------------

def _lame(a, R, nu, E, p, r):
    """Cavity under internal pressure p in a ring a..R with u(R) = 0, plane strain (tension positive)."""
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    B = p / (2 * mu / a ** 2 + 2 * (lam + mu) / R ** 2)
    A = -B / R ** 2
    u = A * r + B / r
    st = 2 * (lam + mu) * A + 2 * mu * B / r ** 2
    return u, st


def _fe_cavity(a, R, nu, E, p, h0, growth=1.15, hmax=None):
    """One-dimensional axisymmetric finite elements (linear, plane strain) for the same problem."""
    hmax = hmax or a
    nodes, h = [a], h0
    while nodes[-1] + h < R:
        nodes.append(nodes[-1] + h)
        h = min(h * growth, hmax)
    nodes.append(R)
    r = np.array(nodes)
    n = len(r)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    Dm = np.array([[lam + 2 * mu, lam], [lam, lam + 2 * mu]])
    K = np.zeros((n, n))
    gp = np.array([-1, 1]) / math.sqrt(3)
    for e in range(n - 1):
        r1, r2 = r[e], r[e + 1]
        L = r2 - r1
        ke = np.zeros((2, 2))
        for xi in gp:
            N = np.array([(1 - xi) / 2, (1 + xi) / 2])
            rr = N @ [r1, r2]
            B = np.array([[-1 / L, 1 / L], N / rr])
            ke += B.T @ Dm @ B * rr * L / 2
        K[e:e + 2, e:e + 2] += ke
    f = np.zeros(n)
    f[0] = p * a
    u = np.zeros(n)
    u[:-1] = np.linalg.solve(K[:-1, :-1], f[:-1])
    # tangential stress at the centroid of the element next to the wall, as a model reports it
    rc = 0.5 * (r[0] + r[1])
    er = (u[1] - u[0]) / (r[1] - r[0])
    et = 0.5 * (u[0] + u[1]) / rc
    st = lam * er + (lam + 2 * mu) * et
    return u[0], st, rc


def extent_mesh():
    a, E, p = 1.0, 1000.0, 1.0
    fig, (a1, a2) = qz.figure(W_FULL, 0.47 * W_FULL, ncols=2)
    cols = {}
    # boundary distance
    rd = np.linspace(1.0, 12.0, 300)                          # R / D
    for nu, col, ls in ((0.3, qz.MAL, "-"), (0.45, qz.CAR, "--")):
        R = rd * 2 * a
        u_inf = p * a * (1 + nu) / E
        ua = np.array([_lame(a, x, nu, E, p, a)[0] for x in R])
        err = (1 - ua / u_inf) * 100
        a1.plot(rd, err, color=col, ls=ls, lw=1.3, label=f"$\\nu = {nu:g}$")
        cols[f"R_over_D"] = rd
        cols[f"u_error_pct_nu{nu:g}"] = err
        # finite element points at a few distances
        rpts = np.array([1.5, 2.5, 5.0, 10.0])
        fe = np.array([(1 - _fe_cavity(a, x * 2 * a, nu, E, p, 0.02)[0] / u_inf) * 100 for x in rpts])
        a1.plot(rpts, fe, "o", mfc="white", mec=col, ms=3.8, mew=0.8)
        print(f"  nu {nu}: error at R/D=5 {np.interp(5, rd, err):.2f} %, at R/D=2 {np.interp(2, rd, err):.1f} %")
    a1.axhline(5, color=qz.GREY, lw=0.6, ls=":")
    a1.axvline(5, color=qz.GREY_LIGHT, lw=0.6)
    a1.set_xlim(1, 12)
    a1.set_ylim(0, 40)
    a1.set_xticks([1, 3, 5, 7, 9, 11])
    a1.text(11.8, 5.6, "5 %", ha="right", va="bottom", fontsize=8, color=qz.GREY)
    qz.label(a1, "Boundary distance, $R/D$ (-)", "Displacement error, $e_u$ (%)")
    a1.legend(loc="upper right")

    # mesh size at the wall, domain fixed at R = 10 D
    nu, R = 0.3, 20 * a
    hs = np.geomspace(0.01, 0.6, 14)
    u_ex, st_ex = _lame(a, R, nu, E, p, a)
    eu, es = [], []
    for h in hs:
        ua, st, rc = _fe_cavity(a, R, nu, E, p, h)
        eu.append(abs(ua / u_ex - 1) * 100)
        # total tangential stress after excavation is p0 + change; refer the error to the exact 2 p0 at the wall
        es.append(abs((p + st) - (p + st_ex)) / (p + st_ex) * 100)
    eu, es = np.array(eu), np.array(es)
    a2.loglog(hs, es, color=qz.CAR, lw=1.3, ls="--", marker="o", ms=3, mfc="white", label="Stress, $\\sigma_\\theta$")
    a2.loglog(hs, eu, color=qz.MAL, lw=1.3, marker="s", ms=3, mfc="white", label="Displacement, $u_a$")
    a2.axhline(5, color=qz.GREY, lw=0.6, ls=":")
    a2.set_xlim(0.01, 0.6)
    a2.set_ylim(0.05, 100)
    a2.text(0.0135, 5.6, "5 %", ha="left", va="bottom", fontsize=8, color=qz.GREY)
    qz.label(a2, "Element size, $h/a$ (-)", "Error, $e$ (%)")
    a2.legend(loc="upper left")
    cols.update({"h_over_a": hs, "u_error_pct": eu, "sigma_theta_error_pct": es})
    print(f"  h/a 0.1: stress error {np.interp(0.1, hs, es):.1f} %, disp error {np.interp(0.1, hs, eu):.4f} %")
    fig.tight_layout()
    _save(fig, "ch17", "extent-mesh", cols)


# ---------------------------------------------------------------------------
# Chapter 17: what each family of methods discretises (schematic)
# ---------------------------------------------------------------------------

def method_family():
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 0.40 * W_FULL))
    a, half = 1.0, 4.0
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-half - 0.9, half + 0.9)
        ax.set_ylim(-half - 0.9, half + 0.9)
    # finite element: whole domain meshed, truncated at an artificial boundary
    ax = axes[0]
    pts = []
    r = a
    while r < 2.6:
        n = max(10, int(2 * np.pi * r / (0.18 + 0.22 * (r - a))))
        th = np.linspace(0, 2 * np.pi, n, endpoint=False) + (np.pi / n if len(pts) % 2 else 0)
        pts += [(r * np.cos(t), r * np.sin(t)) for t in th]
        r += 0.18 + 0.2 * (r - a)
    g = np.linspace(-half, half, 9)
    for x in g:
        for y in g:
            if max(abs(x), abs(y)) > 2.9 or (abs(x) == half or abs(y) == half):
                pts.append((x, y))
    for x in np.linspace(-half, half, 13):
        pts += [(x, half), (x, -half), (half, x), (-half, x)]
    pts = np.unique(np.round(np.array(pts), 6), axis=0)
    tri = Triangulation(pts[:, 0], pts[:, 1])
    c = pts[tri.triangles].mean(axis=1)
    tri.set_mask(np.hypot(c[:, 0], c[:, 1]) < a)
    ax.triplot(tri, color=qz.GREY, lw=0.35)
    ax.add_patch(patches.Rectangle((-half, -half), 2 * half, 2 * half, fill=False, ec=qz.INK, lw=1.0))
    ax.add_patch(patches.Circle((0, 0), a, fc="white", ec=qz.CAR, lw=1.3))
    ax.set_title("Finite element", fontsize=9, pad=3)
    # boundary element: only the excavation surface, far field exact
    ax = axes[1]
    n = 16
    th = np.linspace(0, 2 * np.pi, n + 1)
    ax.plot(a * np.cos(th), a * np.sin(th), color=qz.CAR, lw=1.3)
    ax.plot(a * np.cos(th[:-1]), a * np.sin(th[:-1]), "o", ms=2.6, mfc=qz.CAR, mec=qz.CAR)
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        x0, y0 = 3.9 * np.cos(t), 3.9 * np.sin(t)
        ax.annotate("", xy=(3.0 * np.cos(t), 3.0 * np.sin(t)), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=qz.GREY, lw=0.7, mutation_scale=6))
    ax.set_title("Boundary element", fontsize=9, pad=3)
    # distinct element: blocks bounded by two joint sets
    ax = axes[2]
    box = patches.Rectangle((-half, -half), 2 * half, 2 * half, fill=False, ec=qz.INK, lw=1.0, zorder=5)
    ax.add_patch(box)
    # two joint sets, symmetric about the vertical, one vertex of the set placed just inside the crown
    s1, s2 = math.radians(50), math.radians(-50)
    n1, n2 = np.array([-math.sin(s1), math.cos(s1)]), np.array([-math.sin(s2), math.cos(s2)])
    y0, sp = a - 0.30, 1.15
    d0 = y0 * math.cos(s1)
    d1 = d0 + sp * np.arange(-9, 10)
    d2 = d0 + sp * np.arange(-9, 10)
    for d, s, nn in [(d, s1, n1) for d in d1] + [(d, s2, n2) for d in d2]:
        t = np.array([-12.0, 12.0])
        p = np.outer(t, [math.cos(s), math.sin(s)]) + d * nn
        ln, = ax.plot(p[:, 0], p[:, 1], color=qz.GREY, lw=0.6, zorder=1)
        ln.set_clip_path(box)
    ax.add_patch(patches.Circle((0, 0), a, fc="white", ec=qz.INK, lw=1.0, zorder=3))
    # the key block above the crown, its lower vertex inside the opening, shown displaced into it
    M = np.array([n1, n2])
    corners = [np.linalg.solve(M, [x, y]) for x, y in ((d0, d0), (d0 + sp, d0), (d0 + sp, d0 + sp), (d0, d0 + sp))]
    blk = np.array(corners)
    ax.add_patch(patches.Polygon(blk, closed=True, fill=False, ec=qz.CAR, lw=0.6, ls=":", zorder=4))
    ax.add_patch(patches.Polygon(blk + [0, -0.45], closed=True, fc=qz.CAR_TINT, ec=qz.CAR, lw=1.0, zorder=4))
    ax.set_title("Distinct element", fontsize=9, pad=3)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01, wspace=0.08)
    _save(fig, "ch17", "method-family", None)


# ---------------------------------------------------------------------------
# Chapter 17: sampling designs and confident extrapolation of a surrogate
# ---------------------------------------------------------------------------

def _gp(xt, yt, xs, noise=1e-6):
    """Gaussian process, squared-exponential kernel, constant mean; length scale and amplitude
    chosen by maximum marginal likelihood over a grid."""
    mu = yt.mean()
    y = yt - mu
    n = len(xt)

    def kern(x1, x2, ell, amp):
        return amp ** 2 * np.exp(-0.5 * (x1[:, None] - x2[None, :]) ** 2 / ell ** 2)

    best = None
    for ell in np.geomspace(0.2, 20, 80):
        for amp in np.geomspace(1, 2000, 120):
            L = np.linalg.cholesky(kern(xt, xt, ell, amp) + noise * np.eye(n))
            a = np.linalg.solve(L.T, np.linalg.solve(L, y))
            ll = -0.5 * y @ a - np.log(np.diag(L)).sum()
            if best is None or ll > best[0]:
                best = (ll, ell, amp)
    _, ell, amp = best
    Kt = kern(xt, xt, ell, amp) + noise * np.eye(n)
    Ks = kern(xs, xt, ell, amp)
    mean = mu + Ks @ np.linalg.solve(Kt, y)
    var = amp ** 2 - np.einsum("ij,ji->i", Ks, np.linalg.solve(Kt, Ks.T))
    print(f"  GP: length {ell:.2f}, amplitude {amp:.1f}")
    return mean, np.sqrt(np.clip(var, 0, None))


def surrogate():
    _check_canonical()
    fig, (a1, a2) = qz.figure(W_FULL, 0.47 * W_FULL, ncols=2,
                              gridspec_kw={"width_ratios": [1, 1.35]})
    n = 9
    # one at a time: a cross through the centre
    lv = np.linspace(0.1, 0.9, 5)
    oat = np.unique(np.vstack([np.column_stack([lv, np.full(5, 0.5)]), np.column_stack([np.full(5, 0.5), lv])]),
                    axis=0)
    # Latin hypercube: one point in each row and column of a 9 x 9 grid (seeded)
    rng = np.random.default_rng(3)
    best, score = None, -1
    for _ in range(400):
        perm1, perm2 = rng.permutation(n), rng.permutation(n)
        pts = np.column_stack([(perm1 + rng.uniform(0.2, 0.8, n)) / n, (perm2 + rng.uniform(0.2, 0.8, n)) / n])
        dmin = min(np.hypot(*(pts[i] - pts[j])) for i in range(n) for j in range(i + 1, n))
        if dmin > score:
            best, score = pts, dmin
    for v in np.arange(1, n) / n:
        a1.axhline(v, color=qz.GREY_LIGHT, lw=0.4)
        a1.axvline(v, color=qz.GREY_LIGHT, lw=0.4)
    a1.plot(best[:, 0], best[:, 1], "o", color=qz.MAL, ms=4.5, label="Latin hypercube", zorder=4)
    a1.plot(oat[:, 0], oat[:, 1], "s", mfc="none", mec=qz.INK, ms=5.5, mew=0.9, label="One at a time", zorder=5)
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.set_aspect("equal")
    a1.set_xticks([0, 0.5, 1])
    a1.set_yticks([0, 0.5, 1])
    qz.label(a1, "Input 1, $x_1$ (-)", "Input 2, $x_2$ (-)")

    # parent: the closed-form maximum convergence of the deep tunnel (GSI 30) against cohesion
    def parent(c):
        g = ccm.Ground(16.9, 4.75, 0.22, 2.12, 23.8, c)
        return ccm.grc_displacement(g, 0.0)
    z = np.linspace(0.7, 3.2, 251)
    up = np.array([parent(v) for v in z])
    zt = np.array([1.45, 1.60, 1.72, 1.90, 2.05])
    ut = np.array([parent(v) for v in zt])
    assert abs(parent(1.72) - 126.9) < 0.05
    quad = np.polyval(np.polyfit(zt, ut, 2), z)
    gm, gs = _gp((zt - 1.72) / 0.3, ut, (z - 1.72) / 0.3)
    a2.axvspan(zt.min(), zt.max(), color=qz.MAL_TINT, lw=0, zorder=0)
    a2.fill_between(z, gm - 2 * gs, gm + 2 * gs, color=qz.GREY_LIGHT, alpha=0.6, lw=0,
                    label="Gaussian process, $\\pm 2$ sd")
    a2.plot(z, up, color=qz.MAL, lw=1.5, label="Parent model")
    a2.plot(z, quad, color=qz.CAR, lw=1.2, ls="--", label="Quadratic fit")
    a2.plot(z, gm, color=qz.GREY, lw=1.0, ls="-.", label="Gaussian process, mean")
    a2.plot(zt, ut, "o", mfc="white", mec=qz.INK, ms=4.2, mew=0.9, zorder=5)
    a2.set_xlim(0.7, 3.2)
    a2.set_ylim(0, 400)
    qz.label(a2, "Cohesion, $c$ (MPa)", "Convergence, $u_{\\max}$ (mm)")
    a2.legend(loc="upper right", handlelength=1.8)
    print(f"  u_max at c=0.7: parent {up[0]:.0f} mm, quadratic {quad[0]:.0f} mm, GP {gm[0]:.0f} +- {2 * gs[0]:.0f}; "
          f"at c=3.2: parent {up[-1]:.0f}, quadratic {quad[-1]:.0f}, GP {gm[-1]:.0f} +- {2 * gs[-1]:.0f}")
    fig.tight_layout()
    _save(fig, "ch17", "surrogate", {"c_MPa": z, "parent_mm": up, "quadratic_mm": quad, "process_mean_mm": gm,
                                     "process_sd_mm": gs, "train_c_MPa": zt, "train_u_mm": ut,
                                     "lhs_x1": best[:, 0], "lhs_x2": best[:, 1],
                                     "oat_x1": oat[:, 0], "oat_x2": oat[:, 1]})


# ---------------------------------------------------------------------------
# Appendix F: zone of influence around a circular opening
# ---------------------------------------------------------------------------

def zone_of_influence():
    x = np.linspace(0.0, 7.0, 701)
    X, Y = np.meshgrid(x, x)
    R = np.maximum(np.hypot(X, Y), 0.5)
    TH = np.arctan2(Y, X)
    q2, q4 = 1 / R ** 2, 1 / R ** 4
    fig, ax = qz.figure(W_HALF, W_HALF * 0.92)
    cols = {}
    specs = ((1.0, "p0", qz.MAL, "-", "$k = 1$"),
             (0.5, "p0", qz.CAR, "-", "$k = 0.5$, $p_0$"),
             (0.5, "local", qz.CAR, "--", "$k = 0.5$, far field"))
    handles = []
    for k, ref, col, ls, name in specs:
        dst = 0.5 * ((1 + k) * q2 + (1 - k) * 3 * q4 * np.cos(2 * TH))      # change in sigma_theta / p0
        far = 0.5 * ((1 + k) + (1 - k) * np.cos(2 * TH)) if ref == "local" else 1.0
        z = np.abs(dst) / far
        # the outer contour bounds the zone; at k = 0.5 the crown stress happens to equal the far-field
        # value at the wall itself, which closes a second, local contour there that is not drawn
        z[R < 1.5] = np.nan
        cs = ax.contour(X, Y, z, levels=[0.05], colors=[col], linewidths=1.3, linestyles=[ls])
        handles.append(Line2D([], [], color=col, ls=ls, lw=1.3, label=name))
        # radii on the axes, for the record
        rh = x[np.nanargmin(np.abs(z[0, :] - 0.05) + (x < 1.5) * 9)]
        rv = x[np.nanargmin(np.abs(z[:, 0] - 0.05) + (x < 1.5) * 9)]
        cols[f"k{k:g}_{ref}_r_horizontal"] = np.array([rh])
        cols[f"k{k:g}_{ref}_r_vertical"] = np.array([rv])
        print(f"  {name}: r = {rh:.2f} a horizontal, {rv:.2f} a vertical")
    ax.add_patch(patches.Wedge((0, 0), 1.0, 0, 90, fc=qz.GREY_LIGHT, ec=qz.INK, lw=0.8))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    qz.label(ax, "Horizontal distance, $x/a$ (-)", "Vertical distance, $y/a$ (-)")
    ax.legend(handles=handles, loc="upper right", handlelength=1.8)
    fig.tight_layout()
    _save(fig, "appF", "influence-contours", cols)


# ---------------------------------------------------------------------------
# Appendix F: Einstein and Schwartz thrust and moment coefficients
# ---------------------------------------------------------------------------

def einstein_schwartz():
    nu, t_over_a = 0.3, 0.1
    F = np.geomspace(1, 1e4, 400)
    C = F * t_over_a ** 2 / 12                     # solid section: F / C = 12 (a/t)^2
    a0 = C * F * (1 - nu) / (C + F + C * F * (1 - nu))
    a2 = (F + 6) * (1 - nu) / (2 * F * (1 - nu) + 6 * (5 - 6 * nu))
    fig, (a1, ax2) = qz.figure(W_FULL, 0.47 * W_FULL, ncols=2)
    cols = {"F": F, "C": C, "a0_star": a0, "a2_star": a2}
    t1 = (1 - a0)                                   # k = 1
    k = 0.5
    tw = 0.5 * ((1 + k) * (1 - a0) + (1 - k) * (1 - 2 * a2))     # springline, theta = 0
    tc = 0.5 * ((1 + k) * (1 - a0) - (1 - k) * (1 - 2 * a2))     # crown, theta = 90
    a1.semilogx(F, t1, color=qz.MAL, lw=1.4, label="$k = 1$")
    a1.semilogx(F, tc, color=qz.CAR, lw=1.3, ls="--", label="$k = 0.5$, crown")
    a1.semilogx(F, tw, color=qz.CAR, lw=1.3, label="$k = 0.5$, springline")
    cols.update({"T_k1": t1, "T_k0.5_springline": tw, "T_k0.5_crown": tc})
    a1.set_xlim(1, 1e4)
    a1.set_ylim(0, 1.2)
    qz.label(a1, "Flexibility ratio, $F$ (-)", "Thrust, $T/(p_0 a)$ (-)")
    a1.legend(loc="lower left", handlelength=1.6, borderaxespad=0.3)
    for kk, col, ls in ((0.25, qz.CAR_SHADES[0], "-"), (0.5, qz.CAR, "--"), (0.75, qz.CAR_SHADES[3], ":")):
        m = 0.5 * (1 - kk) * (1 - 2 * a2)
        ax2.semilogx(F, m, color=col, lw=1.3, ls=ls, label=f"$k = {kk:g}$")
        cols[f"M_k{kk:g}"] = m
    ax2.set_xlim(1, 1e4)
    ax2.set_ylim(0, 0.25)
    qz.label(ax2, "Flexibility ratio, $F$ (-)", "Moment, $|M|/(p_0 a^2)$ (-)")
    ax2.legend(loc="upper right")
    print(f"  F=10: a0* {np.interp(10, F, a0):.3f}, a2* {np.interp(10, F, a2):.3f}; "
          f"F=1000: a0* {np.interp(1000, F, a0):.3f}, a2* {np.interp(1000, F, a2):.3f}")
    fig.tight_layout()
    _save(fig, "appF", "einstein-schwartz", cols)


ALL = {"plastic_zone": plastic_zone, "install_distance": install_distance, "motorway_timing": motorway_timing,
       "combined_support": combined_support, "stiff_yielding": stiff_yielding, "trough_cases": trough_cases,
       "volume_loss_sources": volume_loss_sources, "model_domain": model_domain, "extent_mesh": extent_mesh,
       "method_family": method_family, "surrogate": surrogate, "zone_of_influence": zone_of_influence,
       "einstein_schwartz": einstein_schwartz}


if __name__ == "__main__":
    qz.use()
    want = sys.argv[1:]
    for name, fn in ALL.items():
        if not want or any(w in name for w in want):
            print(name)
            fn()
