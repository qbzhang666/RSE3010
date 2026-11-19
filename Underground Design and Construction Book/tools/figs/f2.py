"""Figures for Chapters 5 to 7: site investigation, ground modelling, hydrogeology.

Every plotted quantity comes from the chapter's own equations, tables and
worked examples, or is a schematic or synthetic illustration declared as such
in the caption. Figures are drawn at their printed width (117 mm text block)
in the house style of ``qz_figure.py`` and written as PDF and 300 dpi PNG, with
the plotted data as CSV beside them.

    python tools/figs/f2.py              # all figures
    python tools/figs/f2.py rqd          # figures whose name contains "rqd"
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402
from scipy.interpolate import RBFInterpolator  # noqa: E402
from scipy.special import erfc  # noqa: E402

import qz_figure as qz  # noqa: E402
from geo_platform import rock  # noqa: E402

FIG = os.path.join(BOOK, "figures")
W_FULL = 117 / 25.4          # SNmono text width
W_HALF = 0.62 * W_FULL


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


# ===========================================================================
# Chapter 5: site investigation
# ===========================================================================

# Core run of Figure rqd-core-run (mm, kind): illustrative, 1.5 m run.
CORE_RUN = [(250, "sound"), (60, "short"), (40, "short"), (180, "sound"), (130, "soft"),
            (70, "short"), (30, "short"), (110, "sound"), (90, "short"), (120, "loss"),
            (300, "sound"), (45, "short"), (75, "short")]


def rqd_core_run():
    """A 1.5 m core run, with the pieces that count towards RQD distinguished."""
    run = sum(L for L, _ in CORE_RUN)
    assert run == 1500
    counted = sum(L for L, k in CORE_RUN if k == "sound")
    rqd = 100 * counted / run
    tcr = 100 * (run - sum(L for L, k in CORE_RUN if k == "loss")) / run
    print(f"  core run {run} mm, counted {counted} mm, RQD {rqd:.0f} %, TCR {tcr:.0f} %")

    style = {"sound": dict(fc=qz.MAL_TINT, ec=qz.MAL, hatch=None, ls="-"),
             "short": dict(fc="white", ec=qz.GREY, hatch=None, ls="-"),
             "soft": dict(fc=qz.CAR_TINT, ec=qz.CAR, hatch="////", ls="-"),
             "loss": dict(fc="white", ec=qz.GREY, hatch=None, ls=(0, (2, 2)))}
    fig, ax = qz.figure(W_FULL, 0.33 * W_FULL)
    x = 0.0
    xs, ls_, ks = [], [], []
    h = 60.0                                             # core diameter drawn at about NQ size
    for L, kind in CORE_RUN:
        st = style[kind]
        r = Rectangle((x + 2, 0), L - 4, h, facecolor=st["fc"], edgecolor=st["ec"], lw=0.8,
                      hatch=st["hatch"], ls=st["ls"])
        if st["hatch"]:
            r.set_hatch_linewidth(0.4) if hasattr(r, "set_hatch_linewidth") else None
        ax.add_patch(r)
        if kind == "sound":
            ax.text(x + L / 2, h / 2, f"{L}", ha="center", va="center", fontsize=7, color=qz.MAL)
        xs.append(x)
        ls_.append(L)
        ks.append(kind)
        x += L
    ax.set_xlim(-10, run + 10)
    ax.set_ylim(-5, h + 5)
    ax.set_yticks([])
    ax.tick_params(axis="x", top=False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.set_xticks(np.arange(0, run + 1, 250))
    qz.label(ax, "Distance along core run, $x$ (mm)")
    handles = [Rectangle((0, 0), 1, 1, facecolor=style[k]["fc"], edgecolor=style[k]["ec"],
                         hatch=style[k]["hatch"], ls=style[k]["ls"], lw=0.8)
               for k in ("sound", "short", "soft", "loss")]
    labels = ["Sound, $\\geq 100$ mm (counted)", "Shorter than 100 mm", "Soft or weathered", "Core loss"]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4,
              handlelength=1.4, columnspacing=1.0, fontsize=8)
    ax.set_aspect("auto")
    fig.tight_layout()
    _save(fig, "ch05", "rqd-core-run", {"start_mm": xs, "length_mm": ls_, "kind": ks})


def borehole_rqd():
    """RQD logs of the two illustrative boreholes of Table tab:borehole against the Deere classes."""
    holes = {"BH01": [72, 85], "BH02": [45, 68]}
    depth = [0, 10, 20]
    fig, ax = qz.figure(W_HALF, 0.95 * W_HALF)
    bands = [(0, 25, "Very poor"), (25, 50, "Poor"), (50, 75, "Fair"), (75, 90, "Good"),
             (90, 100, "Excellent")]
    for i, (lo, hi, name) in enumerate(bands):
        if i % 2 == 0:
            ax.axvspan(lo, hi, color="#F2F2F2", lw=0)
        ax.text((lo + hi) / 2, 0.4, name, rotation=90, ha="center", va="top", fontsize=7,
                color=qz.GREY)
    cols = {"depth_top_m": depth[:-1], "depth_base_m": depth[1:]}
    for (name, vals), col, ls in zip(holes.items(), (qz.MAL, qz.CAR), ("-", "--")):
        xv = np.repeat(vals, 2)
        yv = np.array([0, 10, 10, 20])
        ax.plot(xv, yv, color=col, ls=ls, lw=1.5, label=name)
        cols[name + "_RQD_pct"] = vals
    ax.set_xlim(0, 100)
    ax.set_ylim(20, 0)
    ax.set_xticks([0, 25, 50, 75, 90, 100])
    qz.label(ax, "Rock Quality Designation, RQD (%)", "Depth, $z$ (m)")
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, "ch05", "borehole-rqd-logs", cols)


def _rqd_from_frequency(lam):
    """RQD (%) for randomly spaced joints at mean frequency lam (1/m), threshold 0.1 m."""
    x = 0.1 * np.asarray(lam, dtype=float)
    return 100 * np.exp(-x) * (1 + x)


def rqd_orientation():
    """Orientation bias: joint frequency and RQD seen by a borehole at different inclinations."""
    lam_b, lam_v = 6.8, 12.0                  # 1/m: horizontal bedding, sub-vertical joints
    theta = np.linspace(0, 90, 181)           # borehole inclination from vertical
    lam = lam_b * np.cos(np.radians(theta)) + lam_v * np.sin(np.radians(theta))
    rqd = _rqd_from_frequency(lam)
    i_max = int(np.argmax(lam))
    print(f"  vertical RQD {rqd[0]:.0f} %, horizontal {rqd[-1]:.0f} %, "
          f"min {rqd[i_max]:.0f} % at {theta[i_max]:.0f} deg")

    fig = plt.figure(figsize=(W_FULL, 0.45 * W_FULL))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.72, 1.2], wspace=0.6)
    a1 = fig.add_subplot(gs[0])
    a2 = fig.add_subplot(gs[1])

    # Schematic section, joint spacing exaggerated (x 2.5) for legibility.
    sb, sv = 2.5 / lam_b, 2.5 / lam_v
    B = 2.0
    for z in np.arange(sb / 2, B, sb):
        a1.plot([0, B], [z, z], color=qz.GREY, lw=0.7)
    for xx in np.arange(sv / 2, B, sv):
        a1.plot([xx, xx + 0.05], [0, B], color=qz.GREY_LIGHT, lw=0.7)
    a1.add_patch(Rectangle((0, 0), B, B, fill=False, ec=qz.INK, lw=0.6))
    # vertical hole and hole inclined 60 degrees from vertical
    a1.plot([0.35, 0.35], [B, 0], color=qz.MAL, lw=2.4, solid_capstyle="butt")
    t = math.radians(60)
    x0, z0 = 0.75, B
    L = (B - 0.0) / math.cos(t)
    x1, z1 = x0 + L * math.sin(t), z0 - L * math.cos(t)
    if x1 > B:
        frac = (B - x0) / (x1 - x0)
        x1, z1 = B, z0 - frac * (z0 - z1)
    a1.plot([x0, x1], [z0, z1], color=qz.CAR, lw=2.4, ls="-", solid_capstyle="butt")
    a1.text(0.35, B + 0.06, "$0^\\circ$", ha="center", va="bottom", fontsize=8, color=qz.MAL)
    a1.text(x0, B + 0.06, "$60^\\circ$", ha="center", va="bottom", fontsize=8, color=qz.CAR)
    a1.set_xlim(-0.02, B + 0.02)
    a1.set_ylim(-0.02, B + 0.25)
    a1.set_aspect("equal")
    a1.axis("off")

    a2.plot(theta, rqd, color=qz.INK, lw=1.4)
    a2.plot([0], [rqd[0]], "o", color=qz.MAL, ms=5)
    j60 = int(np.argmin(abs(theta - 60)))
    a2.plot([theta[j60]], [rqd[j60]], "s", color=qz.CAR, ms=5)
    for lo, hi, name in ((50, 75, "Fair"), (75, 90, "Good"), (90, 100, "Excellent")):
        if name != "Good":
            a2.axhspan(lo, hi, color="#F2F2F2", lw=0)
        a2.text(88, {"Fair": 53.5, "Good": 86.5, "Excellent": 95}[name], name, ha="right", va="center",
                fontsize=7, color=qz.GREY)
    a2.set_xlim(0, 90)
    a2.set_xticks([0, 15, 30, 45, 60, 75, 90])
    a2.set_ylim(50, 100)
    qz.label(a2, "Borehole inclination from vertical, $\\theta$ ($^\\circ$)", "Rock Quality Designation,\nRQD (%)")
    a3 = a2.twinx()
    a3.plot(theta, lam, color=qz.GREY, lw=1.0, ls="--")
    a3.set_ylim(0, 25)
    a3.set_ylabel("Joint frequency, $\\lambda$ (1/m)", color=qz.GREY)
    a3.tick_params(axis="y", colors=qz.GREY)
    a2.tick_params(axis="y", right=False)
    a2.text(27, 62.5, "RQD", ha="center", va="center", fontsize=8)
    a2.text(60, 80.8, "$\\lambda$", ha="center", va="center", fontsize=9, color=qz.GREY)
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.2, top=0.95)
    _save(fig, "ch05", "rqd-orientation-bias", {"theta_deg": theta, "lambda_per_m": lam, "RQD_pct": rqd})


def gsi_sensitivity():
    """Rock mass uniaxial strength against GSI, Table tab:sensitivity."""
    sci, mi = 110.0, 17.0
    gsi = np.linspace(25, 80, 221)
    cols = {"GSI": gsi}
    fig, ax = qz.figure(W_HALF, 0.8 * W_HALF)
    for d, col, ls, lab in ((0.0, qz.MAL, "-", "$D = 0$"), (0.5, qz.CAR, "--", "$D = 0.5$")):
        scm = np.array([rock.rock_mass_ucs(sci, rock.hb_constants(g, mi, d)) for g in gsi])
        ax.plot(gsi, scm, color=col, ls=ls, lw=1.3, label=lab)
        cols[f"sigma_cm_D{d:g}_MPa"] = scm
    pts = [45, 50, 55]
    vals = [rock.rock_mass_ucs(sci, rock.hb_constants(g, mi, 0.0)) for g in pts]
    print("  table check:", [f"{g}: {v:.2f}" for g, v in zip(pts, vals)],
          f"drop {100 * (1 - vals[0] / vals[2]):.0f} %, rise {100 * (vals[2] / vals[0] - 1):.0f} %")
    ax.plot(pts, vals, "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9, zorder=5)
    ax.annotate("", xy=(45, vals[0]), xytext=(45, vals[2]),
                arrowprops=dict(arrowstyle="->", color=qz.INK, lw=0.8))
    ax.plot([45, 55], [vals[2], vals[2]], color=qz.INK, lw=0.6, ls=":")
    ax.text(43.5, (vals[0] + vals[2]) / 2, "$-44\\%$", ha="right", va="center", fontsize=8)
    ax.set_xlim(25, 80)
    ax.set_ylim(0, 30)
    ax.set_xticks([30, 40, 50, 60, 70, 80])
    qz.label(ax, "Geological Strength Index, GSI (-)", "Rock mass strength,\n$\\sigma_{cm}$ (MPa)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    cols["table_GSI"] = pts
    cols["table_sigma_cm_MPa"] = vals
    _save(fig, "ch05", "gsi-strength-sensitivity", cols)


def specific_energy():
    """Teale specific energy for the worked interval: thrust and rotary terms against penetration rate."""
    d, F, T, N = 0.076, 20e3, 300.0, 120 / 60          # m, N, N m, rev/s
    A = math.pi * d ** 2 / 4
    u = np.linspace(0.15, 2.0, 200)                    # m/min
    thrust = F / A / 1e6 * np.ones_like(u)
    rotary = 2 * math.pi * N * T / (A * u / 60) / 1e6
    total = thrust + rotary
    u0 = 0.6
    r0 = 2 * math.pi * N * T / (A * u0 / 60) / 1e6
    print(f"  thrust {F / A / 1e6:.1f}, rotary {r0:.1f}, total {F / A / 1e6 + r0:.1f} MJ/m3")
    fig, ax = qz.figure(W_HALF, 0.8 * W_HALF)
    ax.plot(u, total, color=qz.MAL, lw=1.5, label="Total, $e_s$")
    ax.plot(u, rotary, color=qz.MAL_SHADES[3], lw=1.2, ls="--", label="Rotary term")
    ax.plot(u, thrust, color=qz.CAR, lw=1.2, ls=":", label="Thrust term")
    ax.plot([u0], [F / A / 1e6 + r0], "o", mfc="white", mec=qz.INK, ms=4.5, mew=0.9, zorder=5)
    ax.set_yscale("log")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(1, 500)
    qz.label(ax, "Penetration rate, $u$ (m/min)", "Specific energy, $e_s$ (MJ/m$^3$)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "ch05", "mwd-specific-energy", {"u_m_per_min": u, "thrust_MJ_m3": thrust,
                                               "rotary_MJ_m3": rotary, "total_MJ_m3": total})


def mwd_signature():
    """Idealised drilling-parameter log through a clay-filled seam and an open joint (schematic)."""
    rng = np.random.default_rng(5)
    z = np.linspace(0, 20, 801)
    n = len(z)
    u = 1.0 + 0.06 * rng.standard_normal(n)
    p = 1.0 + 0.04 * rng.standard_normal(n)
    # clay-filled seam 7.5 to 8.5 m: rate up moderately, flush pressure up (cuttings not clearing)
    seam = np.exp(-0.5 * ((z - 8.0) / 0.35) ** 2)
    u += 0.6 * seam
    p += 0.7 * seam
    # open joint at 13.2 m: abrupt jump in rate, collapse of flush pressure
    op = (z > 13.1) & (z < 13.45)
    u[op] = 3.2 + 0.1 * rng.standard_normal(op.sum())
    p[op] = 0.15 + 0.03 * rng.standard_normal(op.sum())
    # weathered zone 16 to 18 m: rate up gradually
    wz = 1 / (1 + np.exp(-(z - 16) / 0.3)) * (1 - 1 / (1 + np.exp(-(z - 18) / 0.3)))
    u += 0.5 * wz
    p += 0.1 * wz
    fig, (a1, a2) = qz.figure(W_HALF, 1.0 * W_HALF, ncols=2, sharey=True)
    for ax, v, col, lab, lim in ((a1, u, qz.MAL, "Penetration rate,\n$u/\\bar{u}$ (-)", (0, 3.6)),
                                 (a2, p, qz.CAR, "Flush pressure,\n$p_f/\\bar{p}_f$ (-)", (0, 2.0))):
        ax.axhspan(7.5, 8.5, color=qz.GREY_LIGHT, alpha=0.45, lw=0)
        ax.axhspan(13.1, 13.45, color=qz.CAR_TINT, lw=0)
        ax.axhspan(16, 18, color="#F2F2F2", lw=0)
        ax.plot(v, z, color=col, lw=0.8)
        ax.set_xlim(*lim)
        ax.set_xlabel(lab)
    a1.set_ylim(20, 0)
    a1.set_ylabel("Depth along hole, $z$ (m)")
    a1.set_yticks([0, 5, 10, 15, 20])
    a2.tick_params(axis="y", labelleft=False)
    fig.tight_layout(w_pad=0.6)
    _save(fig, "ch05", "mwd-open-feature", {"z_m": z, "u_norm": u, "p_norm": p})


# ===========================================================================
# Chapter 6: ground modelling
# ===========================================================================

def support_scale():
    """Characteristic length of each data source against the resolution each decision needs."""
    rows = [  # (label, lo m, hi m, group)
        ("Laboratory specimen", 0.04, 0.12, "data"),
        ("Borehole core", 0.05, 0.1, "data"),
        ("TBM advance", 1.0, 2.0, "data"),
        ("Mapped face", 5.0, 15.0, "data"),
        ("Seismic refraction", 10.0, 100.0, "data"),
        ("Construction control", 0.5, 3.0, "dec"),
        ("Detailed design", 0.5, 5.0, "dec"),
        ("Reference design", 2.0, 10.0, "dec"),
        ("Route selection", 500.0, 10000.0, "dec"),
    ]
    fig, ax = qz.figure(W_FULL, 0.5 * W_FULL)
    y = np.arange(len(rows))[::-1].astype(float)
    y[5:] -= 0.6
    for yi, (name, lo, hi, g) in zip(y, rows):
        col = qz.MAL if g == "data" else qz.CAR
        ax.plot([lo, hi], [yi, yi], color=col, lw=5, solid_capstyle="butt")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0, right=False)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 20000)
    ax.set_ylim(y[-1] - 0.7, y[0] + 0.7)
    ax.axhline((y[4] + y[5]) / 2, color=qz.GREY_LIGHT, lw=0.6)
    ax.grid(axis="x", which="major")
    ax.text(13000, y[0], "Data support", ha="right", va="center", fontsize=8, color=qz.MAL)
    ax.text(13000, y[6], "Decision resolution", ha="right", va="center", fontsize=8, color=qz.CAR)
    qz.label(ax, "Characteristic length, $L$ (m)")
    fig.tight_layout()
    _save(fig, "ch06", "support-scale", {"item": [r[0] for r in rows], "lo_m": [r[1] for r in rows],
                                         "hi_m": [r[2] for r in rows], "group": [r[3] for r in rows]})


def _implicit_field(bh_x, bh_z, xg, zg, dip_pt=None):
    """Signed-distance scalar field fitted by RBF to contacts and above/below constraints."""
    pts, val = [], []
    for x, zc in zip(bh_x, bh_z):
        for dz in (-12, -6, 0, 6, 12):
            pts.append((x, zc + dz))
            val.append(dz)          # positive above the contact
    if dip_pt is not None:          # an orientation measurement: field gradient along the normal
        x, zc, dip = dip_pt
        nx, nz = -math.sin(math.radians(dip)), math.cos(math.radians(dip))
        for s in (-6, 6):
            pts.append((x + s * nx, zc + s * nz))
            val.append(s)
    pts = np.array(pts, float)
    rbf = RBFInterpolator(pts / [1.0, 1.0], np.array(val, float), kernel="thin_plate_spline",
                          smoothing=0.0)
    X, Z = np.meshgrid(xg, zg)
    F = rbf(np.column_stack([X.ravel(), Z.ravel()])).reshape(X.shape)
    return X, Z, F


def implicit_field():
    """Implicit modelling: contact extracted as the zero isoline of a fitted scalar field."""
    bh_x = [40, 150, 260, 360]
    bh_z = [-14.0, -20.0, -11.0, -17.0]      # contact elevation, m below ground (illustrative)
    new_x, new_z = 205.0, -27.0
    dip = (310, -15.5, -12.0)                 # x, z, apparent dip (deg) measured at one hole
    xg = np.linspace(0, 400, 401)
    zg = np.linspace(-45, 0, 181)
    fig, axes = qz.figure(W_FULL, 0.62 * W_FULL, nrows=2, sharex=True)
    cols = {"x_m": xg}
    prev = None
    for ax, (bx, bz), title in zip(axes, ((bh_x, bh_z), (bh_x + [new_x], bh_z + [new_z])),
                                   ("Four boreholes and one dip", "Fifth borehole added")):
        X, Z, F = _implicit_field(bx, bz, xg, zg, dip)
        ax.contourf(X, Z, F, levels=[-1e6, 0, 1e6], colors=[qz.CAR_TINT, qz.MAL_TINT])
        ax.contour(X, Z, F, levels=np.arange(-40, 41, 5), colors=qz.GREY_LIGHT, linewidths=0.4, linestyles="solid")
        cs = ax.contour(X, Z, F, levels=[0], colors=qz.INK, linewidths=1.4)
        line = max(cs.allsegs[0], key=len)
        zc = np.interp(xg, line[:, 0], line[:, 1])
        cols[title.split()[0].lower() + "_contact_z_m"] = zc
        if prev is not None:
            ax.plot(xg, prev, color=qz.INK, lw=0.9, ls="--")
        prev = zc
        for x, z in zip(bx, bz):
            new = (x == new_x)
            ax.plot([x, x], [0, -42], color=qz.CAR if new else qz.INK, lw=1.6, solid_capstyle="butt")
            ax.plot([x], [z], "o", mfc="white", mec=qz.CAR if new else qz.INK, ms=4.5, mew=1.0, zorder=5)
        dx = 14
        ax.plot([dip[0] - dx, dip[0] + dx],
                [dip[1] + dx * math.tan(math.radians(-dip[2])), dip[1] - dx * math.tan(math.radians(-dip[2]))],
                color=qz.MAL, lw=2.0)
        ax.set_title(title, loc="left", fontsize=8, pad=3)
        ax.set_ylim(-45, 0)
        ax.set_xlim(0, 400)
        ax.set_ylabel("Elevation, $z$ (m)")
    axes[1].set_xlabel("Distance along section, $x$ (m)")
    fig.tight_layout(h_pad=0.4)
    _save(fig, "ch06", "implicit-scalar-field", cols)


def _spherical(h, a, sill=1.0, nugget=0.0):
    h = np.asarray(h, float)
    g = np.where(h < a, nugget + (sill - nugget) * (1.5 * h / a - 0.5 * (h / a) ** 3), sill)
    return np.where(h == 0, 0.0, g)


def variogram():
    h = np.linspace(0, 250, 501)
    a, c0 = 100.0, 0.2
    sph = _spherical(h, a, 1.0, c0)
    expo = np.where(h == 0, 0, c0 + (1 - c0) * (1 - np.exp(-3 * h / a)))
    fig, ax = qz.figure(W_HALF, 0.75 * W_HALF)
    ax.plot(h, sph, color=qz.MAL, lw=1.5, label="Spherical")
    ax.plot(h, expo, color=qz.CAR, lw=1.2, ls="--", label="Exponential")
    ax.plot([0], [0], "o", color=qz.MAL, ms=3.5, clip_on=False, zorder=6)
    ax.axhline(1.0, color=qz.GREY, lw=0.6, ls=":")
    ax.axvline(a, color=qz.GREY, lw=0.6, ls=":")
    ax.annotate("", xy=(4, 0), xytext=(4, c0), arrowprops=dict(arrowstyle="<->", color=qz.INK, lw=0.7))
    ax.text(11, c0 / 2, "nugget", va="center", fontsize=8)
    ax.text(245, 1.03, "sill", ha="right", va="bottom", fontsize=8)
    ax.text(a + 3, 0.08, "range", ha="left", va="bottom", fontsize=8)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 1.25)
    qz.label(ax, "Separation, $h$ (m)", "Semivariance, $\\gamma(h)/\\sigma^2$ (-)")
    ax.legend(loc="center right")
    fig.tight_layout()
    _save(fig, "ch06", "variogram-model", {"h_m": h, "spherical": sph, "exponential": expo})


def _cov_sph(h, a, sd):
    return sd ** 2 * (1 - _spherical(np.abs(h), a, 1.0, 0.0))


def _cov_matern52(h, ell, sd):
    r = np.sqrt(5) * np.abs(h) / ell
    return sd ** 2 * (1 + r + r ** 2 / 3) * np.exp(-r)


def _gp_condition(xs, xo, zo, mean, a, sd, cov=_cov_sph):
    """Simple kriging mean and covariance on xs given exact observations zo at xo."""
    Koo = cov(xo[:, None] - xo[None, :], a, sd)
    Kso = cov(xs[:, None] - xo[None, :], a, sd)
    Kss = cov(xs[:, None] - xs[None, :], a, sd)
    w = np.linalg.solve(Koo, Kso.T)
    mu = mean + w.T @ (zo - mean)
    C = Kss - Kso @ w
    return mu, C


def _draw(mu, C, n, rng):
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)
    L = vecs * np.sqrt(np.clip(vals, 0, None))
    return mu[:, None] + L @ rng.standard_normal((len(mu), n))


def kriging_simulation():
    """Kriging smooths; conditional simulation reproduces variability (synthetic 1D field)."""
    rng = np.random.default_rng(11)
    x = np.linspace(0, 1000, 501)
    mean, sd, a = 50.0, 8.0, 150.0
    truth = _draw(np.full(len(x), mean), _cov_sph(x[:, None] - x[None, :], a, sd), 1, rng)[:, 0]
    xo = np.array([60, 230, 390, 520, 700, 940.0])
    io = np.searchsorted(x, xo)
    zo = truth[io]
    mu, C = _gp_condition(x, xo, zo, mean, a, sd)
    s = np.sqrt(np.clip(np.diag(C), 0, None))
    sims = _draw(mu, C, 3, rng)
    print(f"  sd truth {truth.std():.1f}, sd kriged {mu.std():.1f}, sd sims "
          f"{', '.join(f'{v:.1f}' for v in sims.std(axis=0))}")
    fig, (a1, a2) = qz.figure(W_FULL, 0.66 * W_FULL, nrows=2, sharex=True)
    for ax in (a1, a2):
        ax.plot(x, truth, color=qz.GREY, lw=0.8, label="True field")
        ax.axhline(mean, color=qz.GREY_LIGHT, lw=0.6, ls=":")
    a1.fill_between(x, mu - 2 * s, mu + 2 * s, color=qz.MAL_TINT, lw=0, label="$\\pm 2$ standard deviations")
    a1.plot(x, mu, color=qz.MAL, lw=1.5, label="Kriged estimate")
    for k, (col, ls) in enumerate(((qz.CAR_SHADES[1], "-"), (qz.CAR_SHADES[3], "--"), (qz.MAL_SHADES[2], "-."))):
        a2.plot(x, sims[:, k], color=col, lw=1.0, ls=ls, label="Realisations" if k == 0 else None)
    for ax in (a1, a2):
        ax.plot(xo, zo, "o", mfc="white", mec=qz.INK, ms=4.5, mew=1.0, zorder=5, label="Boreholes")
        ax.set_ylim(20, 95)
        ax.set_yticks([20, 40, 60, 80])
    a1.legend(loc="upper left", ncol=2, handlelength=1.6, columnspacing=1.2)
    a2.legend(loc="upper left", ncol=3, handlelength=1.6, columnspacing=1.2)
    a1.text(0.99, 0.05, "Kriging", transform=a1.transAxes, ha="right", va="bottom", fontsize=8)
    a2.text(0.99, 0.05, "Conditional simulation", transform=a2.transAxes, ha="right", va="bottom", fontsize=8)
    a2.set_xlim(0, 1000)
    a2.set_xlabel("Chainage, $x$ (m)")
    fig.supylabel("Geological Strength Index, GSI (-)", fontsize=10)
    fig.tight_layout(h_pad=0.4)
    _save(fig, "ch06", "kriging-simulation", {"x_m": x, "truth": truth, "kriged": mu, "kriging_sd": s,
                                              "sim1": sims[:, 0], "sim2": sims[:, 1], "sim3": sims[:, 2],
                                              "bh_x_m": xo, "bh_GSI": zo})


def contact_probability():
    """Probability of rock below a simulated contact, and the confidence class along the section."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1000, 401)
    xo = np.array([40, 180, 260, 600, 960.0])
    zo = np.array([-16.0, -21.0, -19.5, -12.0, -18.0])          # illustrative contact elevations (m)
    mean, sd, ell = -17.0, 5.0, 150.0
    mu, C = _gp_condition(x, xo, zo, mean, ell, sd, cov=_cov_matern52)
    s = np.sqrt(np.clip(np.diag(C), 0, None))
    sims = _draw(mu, C, 200, rng)
    zg = np.linspace(-40, 0, 161)
    P = np.mean(zg[:, None, None] < sims[None, :, :], axis=2)    # P(rock) at (z, x)
    fig = plt.figure(figsize=(W_FULL, 0.7 * W_FULL))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], width_ratios=[1, 0.025], hspace=0.12,
                          wspace=0.03)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    cax = fig.add_subplot(gs[0, 1])
    cmap = LinearSegmentedColormap.from_list("white_mal_light", ["#FFFFFF", qz.MAL_SHADES[3]])
    im = a1.pcolormesh(x, zg, P, cmap=cmap, vmin=0, vmax=1, shading="auto", rasterized=True)
    for k in range(6):
        a1.plot(x, sims[:, k], color=qz.GREY, lw=0.4)
    a1.contour(x, zg, P, levels=[0.05, 0.95], colors=qz.CAR, linewidths=0.8, linestyles="--")
    a1.plot(x, mu, color=qz.INK, lw=1.0)
    for xi, zi in zip(xo, zo):
        a1.plot([xi, xi], [0, -38], color=qz.INK, lw=1.4, solid_capstyle="butt")
        a1.plot([xi], [zi], "o", mfc="white", mec=qz.INK, ms=4, mew=0.9, zorder=5)
    a1.set_ylim(-40, 0)
    a1.set_ylabel("Elevation, $z$ (m)")
    a1.tick_params(axis="x", labelbottom=False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Probability of rock, $P$ (-)")
    cb.outline.set_linewidth(0.5)
    hi, mod = 1.0, 3.0
    a2.axhspan(0, hi, color=qz.MAL_TINT, lw=0)
    a2.axhspan(mod, 6, color=qz.CAR_TINT, lw=0)
    a2.plot(x, s, color=qz.INK, lw=1.2)
    a2.plot(xo, np.zeros_like(xo), "o", mfc="white", mec=qz.INK, ms=4, mew=0.9, clip_on=False, zorder=5)
    a2.set_ylim(0, 6)
    a2.set_xlim(0, 1000)
    for yv, name, col in ((0.5, "High", qz.MAL), (2.0, "Moderate", qz.GREY), (4.5, "Low", qz.CAR)):
        a2.text(1.01, yv / 6, name, transform=a2.transAxes, ha="left", va="center", fontsize=8, color=col)
    a2.set_ylabel("Standard deviation,\n$s_z$ (m)")
    a2.set_xlabel("Chainage, $x$ (m)")
    fig.subplots_adjust(left=0.1, right=0.84, bottom=0.13, top=0.97)
    _save(fig, "ch06", "contact-probability", {"x_m": x, "kriged_contact_z_m": mu, "sd_m": s,
                                               "bh_x_m": xo, "bh_contact_z_m": zo})


# ===========================================================================
# Chapter 7: hydrogeology
# ===========================================================================

def goodman_q(k, H, r0):
    """Steady inflow per unit length (m2/s), Goodman et al. (1965)."""
    return 2 * math.pi * k * H / np.log(2 * H / r0)


TO_LMIN = 1000 * 60          # m3/s per m to L/min per m


def inflow_permeability():
    k = np.logspace(-9, -4, 200)
    r0 = 3.0
    fig, (a1, a2) = qz.figure(W_FULL, 0.48 * W_FULL, ncols=2)
    cols = {"k_m_s": k}
    for H, col, ls in ((10, qz.MAL_SHADES[3], ":"), (30, qz.MAL, "-"), (100, qz.CAR, "--")):
        q = goodman_q(k, H, r0) * TO_LMIN
        a1.loglog(k, q, color=col, ls=ls, lw=1.3, label=f"$H = {H}$ m")
        cols[f"q_H{H}_L_min_m"] = q
    a1.set_xlim(1e-9, 1e-4)
    a1.set_ylim(1e-4, 1e3)
    qz.label(a1, "Hydraulic conductivity, $k_w$ (m/s)", "Inflow, $q$ (L/min/m)")
    a1.legend(loc="lower right")
    top = a1.secondary_xaxis("top", functions=(lambda v: v / 1.3e-7, lambda v: v * 1.3e-7))
    top.set_xticks([1e-2, 1e-1, 1, 10, 100])
    top.set_xlabel("Lugeon value (Lu)")
    a1.tick_params(axis="x", which="both", top=False)
    H = np.linspace(10, 200, 200)
    k0 = 1e-7
    cols2 = {"H_m": H}
    for r, col, ls in ((2.0, qz.MAL_SHADES[3], ":"), (3.0, qz.MAL, "-"), (6.0, qz.CAR, "--")):
        q = goodman_q(k0, H, r) * TO_LMIN
        a2.plot(H, q, color=col, ls=ls, lw=1.3, label=f"$r_0 = {r:g}$ m")
        cols2[f"q_r{r:g}_L_min_m"] = q
    a2.set_xlim(0, 200)
    a2.set_ylim(0, 2.2)
    qz.label(a2, "Depth of axis, $H$ (m)", "Inflow, $q$ (L/min/m)")
    a2.legend(loc="upper left")
    print(f"  q(k=1e-7, H=30, r0=3) = {goodman_q(1e-7, 30, 3) * TO_LMIN:.3f} L/min/m")
    fig.tight_layout(w_pad=1.0)
    cols.update(cols2)
    _save(fig, "ch07", "inflow-goodman", cols)


def inflow_discrete():
    """Equivalent continuum against discrete-feature inflow with the same total (synthetic)."""
    rng = np.random.default_rng(21)
    Lt, H, r0, k = 500.0, 50.0, 3.0, 1e-7
    q_uniform = goodman_q(k, H, r0) * TO_LMIN          # L/min per m
    x = np.arange(0, Lt, 1.0) + 0.5
    total = q_uniform * Lt
    matrix = 0.1 * total / Lt * np.ones_like(x)        # 10 % from the matrix, evenly
    nf = rng.poisson(Lt / 45)
    xf = np.sort(rng.uniform(0, Lt, nf))
    wf = rng.lognormal(0, 1.2, nf)
    qf = 0.9 * total * wf / wf.sum()
    q = matrix.copy()
    for xi, qi in zip(xf, qf):
        q[min(int(xi), len(q) - 1)] += qi
    print(f"  uniform {q_uniform:.3f} L/min/m, total {total:.0f} L/min, {nf} features, "
          f"peak {q.max():.1f} L/min/m ({q.max() / q_uniform:.0f} x uniform)")
    fig, (a1, a2) = qz.figure(W_FULL, 0.66 * W_FULL, nrows=2, sharex=True)
    a1.plot(x, q, color=qz.CAR, lw=0.8, drawstyle="steps-mid", label="Discrete features")
    a1.axhline(q_uniform, color=qz.MAL, lw=1.3, ls="--", label="Equivalent continuum")
    a1.set_yscale("log")
    a1.set_ylim(0.01, 1000)
    a1.set_yticks([0.01, 1, 100])
    a1.set_ylabel("Inflow,\n$q$ (L/min/m)")
    a1.legend(loc="upper left", ncol=2)
    a2.plot(x, np.cumsum(q), color=qz.CAR, lw=1.3, label="Discrete features")
    a2.plot(x, q_uniform * (x + 0.5), color=qz.MAL, lw=1.3, ls="--", label="Equivalent continuum")
    a2.set_ylim(0, total * 1.1)
    a2.set_xlim(0, Lt)
    a2.set_ylabel("Cumulative inflow,\n$Q$ (L/min)")
    a2.set_xlabel("Distance along drive, $x$ (m)")
    a2.legend(loc="upper left")
    fig.tight_layout(h_pad=0.4)
    fig.align_ylabels([a1, a2])
    _save(fig, "ch07", "inflow-discrete", {"x_m": x, "q_discrete_L_min_m": q,
                                           "q_uniform_L_min_m": q_uniform * np.ones_like(x)})


def drained_tanked():
    """Drained and tanked linings: water table and pressure on the lining (schematic)."""
    r, t, hw = 3.0, 0.35, 7.0                # internal radius, lining thickness, water table above axis (m)
    ro = r + t
    fig, axes = qz.figure(W_FULL, 0.52 * W_FULL, ncols=2)
    th = np.linspace(0, 2 * np.pi, 361)
    rows = {}
    for ax, kind in zip(axes, ("Drained", "Tanked")):
        ax.add_patch(Circle((0, 0), ro, fc=qz.GREY_LIGHT, ec=qz.INK, lw=0.6))
        ax.add_patch(Circle((0, 0), r, fc="white", ec=qz.INK, lw=0.6))
        xw = np.linspace(-9, 9, 200)
        if kind == "Drained":
            ax.plot((ro + 0.12) * np.cos(th), (ro + 0.12) * np.sin(th), color=qz.MAL, lw=1.6, ls=(0, (3, 1.5)))
            for ang in (-35, -145):
                a_ = math.radians(ang)
                ax.add_patch(Circle(((ro + 0.35) * math.cos(a_), (ro + 0.35) * math.sin(a_)), 0.28,
                                    fc="white", ec=qz.MAL, lw=1.0))
            zw = hw - 3.8 * np.exp(-(xw / 4.5) ** 2)
            ax.plot(xw, zw, color=qz.MAL, lw=1.2)
            ax.plot(xw, hw * np.ones_like(xw), color=qz.GREY, lw=0.7, ls=":")
            for xa in (-6.5, -4.5, 4.5, 6.5):
                ax.add_patch(FancyArrowPatch((xa, 1.5), (np.sign(xa) * (abs(xa) - 1.3), 0.7),
                                             arrowstyle="-|>", mutation_scale=7, color=qz.MAL, lw=0.8))
            ax.plot([-7.0], [hw - 3.8 * math.exp(-(7.0 / 4.5) ** 2) + 0.35], marker="v", color=qz.MAL, ms=6)
            ax.text(8.8, hw + 0.25, "Original water table", ha="right", va="bottom", fontsize=8, color=qz.GREY)
            ax.text(-8.8, 4.9, "Drawn-down\nwater table", ha="left", va="top", fontsize=8, color=qz.MAL)
            a_ = math.radians(-145)
            ax.plot([-5.0, (ro + 0.63) * math.cos(a_)], [-3.7, (ro + 0.63) * math.sin(a_)], color=qz.GREY, lw=0.5)
            ax.text(-5.1, -3.8, "Drain", ha="right", va="center", fontsize=8)
            a_ = math.radians(-62)
            ax.plot([2.8, (ro + 0.25) * math.cos(a_)], [-4.6, (ro + 0.25) * math.sin(a_)], color=qz.GREY, lw=0.5)
            ax.text(2.9, -4.75, "Drainage layer", ha="left", va="center", fontsize=8)
            rows["drained_wt_x_m"] = xw
            rows["drained_wt_z_m"] = zw
        else:
            ax.plot((ro + 0.12) * np.cos(th), (ro + 0.12) * np.sin(th), color=qz.CAR, lw=1.6)
            ax.plot(xw, hw * np.ones_like(xw), color=qz.CAR, lw=1.2)
            ax.plot([-7.0], [hw + 0.35], marker="v", color=qz.CAR, ms=6)
            ang = np.radians(np.arange(0, 360, 22.5))
            head = hw - (ro + 0.2) * np.sin(ang)       # head above each point of the extrados (m)
            scale = 0.17
            for a_, hd in zip(ang, head):
                p0 = ro + 0.2
                p1 = p0 + scale * hd
                ax.add_patch(FancyArrowPatch((p1 * math.cos(a_), p1 * math.sin(a_)),
                                             (p0 * math.cos(a_), p0 * math.sin(a_)),
                                             arrowstyle="-|>", mutation_scale=7, color=qz.CAR, lw=0.8))
            tips = ro + 0.2 + scale * (hw - (ro + 0.2) * np.sin(th))    # smooth envelope of the arrow tails
            ax.plot(tips * np.cos(th), tips * np.sin(th), color=qz.CAR, lw=0.6)
            ax.text(8.8, hw + 0.25, "Water table", ha="right", va="bottom", fontsize=8, color=qz.CAR)
            a_ = math.radians(33.75)
            ax.plot([5.3, (ro + 0.2) * math.cos(a_)], [3.7, (ro + 0.2) * math.sin(a_)], color=qz.GREY, lw=0.5)
            ax.text(6.3, 4.2, "Membrane", ha="center", va="center", fontsize=8)
            ax.text(5.2, -4.2, "Water\npressure", ha="left", va="top", fontsize=8, color=qz.CAR)
            rows["tanked_angle_deg"] = np.degrees(ang)
            rows["tanked_head_m"] = head
            rows["tanked_pressure_kPa"] = 9.81 * head
        ax.set_xlim(-9, 9)
        ax.set_ylim(-7.0, 8.8)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0, -6.6, kind, ha="center", va="bottom", fontsize=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05)
    _save(fig, "ch07", "drained-tanked-lining", rows)


def drawdown_trough():
    """Width of a drawdown settlement trough against a volume-loss trough (schematic parameters)."""
    x = np.logspace(0, np.log10(3000), 600)
    z0, K = 20.0, 0.5
    i = K * z0
    gauss = np.exp(-x ** 2 / (2 * i ** 2))
    D = 2e-3                                   # hydraulic diffusivity T/S (m2/s)
    yr = 365.25 * 86400
    cols = {"x_m": x, "volume_loss": gauss}
    fig, ax = qz.figure(W_FULL, 0.45 * W_FULL)
    ax.plot(x, gauss, color=qz.INK, lw=1.5, label=f"Volume loss, $i = {i:g}$ m")
    for t, lab, col, ls in ((yr / 12, "Drawdown, 1 month", qz.MAL_SHADES[3], ":"),
                            (yr, "Drawdown, 1 year", qz.MAL, "--"),
                            (10 * yr, "Drawdown, 10 years", qz.CAR, "-")):
        sdd = erfc(x / (2 * np.sqrt(D * t)))
        ax.plot(x, sdd, color=col, ls=ls, lw=1.3, label=lab)
        cols[lab.replace(", ", "_").replace(" ", "")] = sdd
        print(f"  {lab}: half-width at 0.5 = {2 * np.sqrt(D * t) * 0.4769:.0f} m")
    ax.set_xscale("log")
    ax.set_xlim(1, 3000)
    ax.set_ylim(1.05, 0)
    qz.label(ax, "Distance from tunnel axis, $x$ (m)", "Settlement, $S/S_{\\max}$ (-)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    _save(fig, "ch07", "drawdown-trough", cols)


FIGURES = [rqd_core_run, borehole_rqd, rqd_orientation, gsi_sensitivity, specific_energy, mwd_signature,
           support_scale, implicit_field, variogram, kriging_simulation, contact_probability,
           inflow_permeability, inflow_discrete, drained_tanked, drawdown_trough]


if __name__ == "__main__":
    qz.use()
    keys = sys.argv[1:]
    for f in FIGURES:
        if not keys or any(k in f.__name__ for k in keys):
            f()
