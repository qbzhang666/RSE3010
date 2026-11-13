"""Matplotlib versions of the platform's interactive charts, for the PDF report.

The pages draw these charts with plotly; a printed report needs static vector
figures in the house style, so each is drawn again here from the same results.
"""

from __future__ import annotations

import math

import numpy as np

from . import ccm, rock, settlement, style


def _figure(width: float = 6.4, height: float = 3.8):
    import matplotlib.pyplot as plt

    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def _finish(ax, xlabel: str, ylabel: str, legend: str | None = "best") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, lw=0.4, color=style.GREY_LIGHT)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend, frameon=False, fontsize=8)


def envelope(sigma_ci: float, mb: float, s: float, a: float, s3max: float, fits, s3_pts, s1_pts,
             hb_label: str, intact_mi: float | None = None):
    """Hoek–Brown envelope with the fitted Mohr–Coulomb lines and the fitting points."""
    import matplotlib.pyplot as plt

    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        s3 = np.linspace(0, max(s3max * 1.25, 1.0), 200)
        ax.plot(s3, rock.hb_sigma1(s3, sigma_ci, mb, s, a), color=style.MALACHITE, lw=1.8, label=hb_label)
        for idx, (label, fit) in enumerate(fits):
            sty = style.series_style(idx + 1)
            y = fit.intercept + fit.slope * s3 if label.startswith("Method 2") else rock.mc_sigma1(s3, fit.c, fit.phi_deg)
            ax.plot(s3, y, color=sty["color"], ls=sty["linestyle"], lw=1.2, label=label)
        if intact_mi is not None:
            ax.plot(s3, rock.hb_sigma1(s3, sigma_ci, intact_mi), color=style.GREY, ls=":", lw=1.0, label="Intact rock")
        ax.plot(s3_pts, s1_pts, "o", mfc="none", color=style.INK, ms=5, label="Points used in the fit")
        ax.set_xlim(left=0)
        _finish(ax, style.axis_label("minor principal stress", r"$\sigma_3$", "MPa"),
                style.axis_label("major principal stress at failure", r"$\sigma_1$", "MPa"), "upper left")
    return fig


def ground_support(res: ccm.CCMResult):
    """Ground reaction curve, support characteristic curve and their equilibrium."""
    import matplotlib.pyplot as plt

    g = res.ground
    p, u = ccm.grc_curve(g, res.method)
    finite = np.isfinite(u)
    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(u[finite], p[finite], color=style.MALACHITE, lw=1.8, label="Ground reaction curve")
        umax = float(np.nanmax(u[finite])) if finite.any() else 1.0
        if res.support is not None:
            s = res.support
            ux = [res.us0, res.us0 + s.usm, max(res.us0 + s.usm, umax * 1.05)]
            ax.plot(ux, [0, s.psm, s.psm], color=style.CARMINE, lw=1.6, label="Support characteristic curve")
            if res.eq and res.eq.p_eq is not None and res.eq.u_eq is not None:
                ax.plot([res.eq.u_eq], [res.eq.p_eq], "o", mfc="none", mec=style.INK, ms=8, mew=1.4, label="Equilibrium")
        if not g.elastic:
            ax.axhline(g.p_cr, color=style.GREY, ls="--", lw=0.9)
            ax.annotate(rf"$p_{{cr}}$ = {g.p_cr:.2f} MPa", (umax, g.p_cr), xytext=(-4, 3),
                        textcoords="offset points", ha="right", fontsize=8, color=style.GREY_DARK)
        ax.axvline(res.us0, color=style.GREY, ls=":", lw=0.9)
        ax.annotate(rf"$u_{{s0}}$ = {res.us0:.1f} mm", (res.us0, g.p0), xytext=(4, -10),
                    textcoords="offset points", fontsize=8, color=style.GREY_DARK)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        _finish(ax, style.axis_label("radial wall displacement", r"$u_r$", "mm"),
                style.axis_label("support pressure", r"$p_i$", "MPa"), "upper right")
    return fig


def longitudinal_profile(res: ccm.CCMResult):
    """Longitudinal deformation profiles, the chosen one in colour, with the installation point."""
    import matplotlib.pyplot as plt

    g = res.ground
    rstar = res.rp0 / g.a if math.isfinite(res.rp0) else 20.0
    x = np.linspace(-4 * g.a, 12 * g.a, 300)
    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for idx, m in enumerate(ccm.LDP_METHODS):
            y = [ccm.ldp(v / g.a, rstar, m) * res.u_max for v in x]
            if m == res.ldp_method:
                ax.plot(x, y, color=style.MALACHITE, lw=1.8, label=m)
            else:
                ax.plot(x, y, color=style.GREY, ls=":", lw=1.0, label=m)
        ax.axvline(0, color=style.INK, lw=0.8)
        ax.annotate("Face", (0, 0), xytext=(3, 3), textcoords="offset points", fontsize=8)
        if math.isfinite(res.install_distance):
            ax.plot([res.install_distance], [res.us0], "o", color=style.CARMINE, ms=6, label="Support installed")
        _finish(ax, style.axis_label("distance behind the face", "$X$", "m"),
                style.axis_label("radial wall displacement", r"$u_r$", "mm"), "lower right")
    return fig


def transverse_trough(bores, y, s, eps):
    """Settlement trough and horizontal strain across the drive."""
    import matplotlib.pyplot as plt

    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(y, -s, color=style.MALACHITE, lw=1.8, label="Settlement")
        ax2 = ax.twinx()
        ax2.plot(y, eps, color=style.CARMINE, ls="--", lw=1.2, label="Horizontal strain (tension positive)")
        ax2.set_ylabel(style.axis_label("horizontal strain", r"$\varepsilon_h$", r"$\mu\varepsilon$"))
        for b in bores:
            for side in (-1, 1):
                ax.axvline(b.offset + side * b.i, color=style.GREY, ls=":", lw=0.8)
        lines = ax.get_legend_handles_labels()
        lines2 = ax2.get_legend_handles_labels()
        ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="lower left", frameon=False, fontsize=8)
        _finish(ax, style.axis_label("offset from the first axis", "$y$", "m"),
                style.axis_label("settlement (down negative)", "$S$", "mm"), None)
    return fig


def face_development(bore: settlement.Bore):
    """Settlement above the axis as the face passes (cumulative normal profile)."""
    import matplotlib.pyplot as plt

    x = np.linspace(-4 * bore.i, 4 * bore.i, 300)
    with plt.rc_context(style.matplotlib_rcparams()):
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        ax.plot(x, -bore.smax_mm * settlement.longitudinal_factor(x, 0.0, bore.i), color=style.MALACHITE, lw=1.8)
        ax.axvline(0, color=style.INK, lw=0.8)
        ax.annotate("Face", (0, 0), xytext=(3, -12), textcoords="offset points", fontsize=8)
        _finish(ax, style.axis_label("position relative to the face (drive towards +x)", "$x$", "m"),
                style.axis_label("settlement", "$S$", "mm"), None)
    return fig
