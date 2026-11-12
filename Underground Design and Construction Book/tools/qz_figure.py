"""Figure style and palette for redrawn figures: malachite and carmine.

The house style of qz-figure-style, as a module rather than a document, so a
figure script sets it in one line and every figure in the three book projects
comes out the same.

    import qz_figure as qz
    qz.use()
    fig, ax = qz.figure(width="single")      # 90 mm, or "double" for 190 mm
    ax.plot(x, y, color=qz.MAL, label="Supported")
    ax.plot(x, z, color=qz.CAR, ls="--", label="Unsupported")
    qz.label(ax, x="Distance from face, $X$ (m)",
                 y="Wall convergence, $u_r$ (mm)")
    qz.save(fig, "ldp")                      # writes ldp.pdf and ldp.png

Two hues only. Further series vary by shade, line style, marker or hatching,
never by a new hue; a reference or screening series is grey. Computer Modern
throughout, sized at print width so text is never scaled afterwards.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ----------------------------------------------------------------- the palette
MAL = "#2F6B57"        # malachite: good, pass, design, supported
CAR = "#8C2B3E"        # carmine: fail, below unity, tension, alert
CAR_DARK = "#6E1F30"   # the severe end, and labels on carmine areas
MAL_TINT = "#E6EFEC"   # fills and bands
CAR_TINT = "#F3E6E9"
INK = "#1A1A1A"        # text
GREY = "#5A5A5A"       # secondary text, reference series
GREY_MID = "#8A8A8A"   # hatching
GREY_LIGHT = "#C8C8C8"  # context: buildings, mesh, background objects
BLUE = "#006DAE"       # Monash blue, slides only, sparingly

#: Ordered shades for when more than two series are unavoidable. Vary by shade
#: of the two hues before reaching for line style, and never add a hue.
MAL_SHADES = ["#1C4034", "#2F6B57", "#4F907A", "#8FBCAC"]
CAR_SHADES = ["#541724", "#6E1F30", "#8C2B3E", "#B8697A"]

#: Diverging, for quantities with a meaningful centre: factor of safety about
#: unity, stress sign, over and under prediction. Pair with centred() so the
#: centre lands on the critical value rather than the middle of the data.
DIVERGING = LinearSegmentedColormap.from_list(
    "malachite_carmine", [CAR, CAR_TINT, MAL_TINT, MAL])

#: Sequential, for magnitude without a centre. White to malachite in general;
#: white to carmine for damage and settlement magnitude, where larger is worse.
SEQ_MAL = LinearSegmentedColormap.from_list("white_malachite", ["#FFFFFF", MAL])
SEQ_CAR = LinearSegmentedColormap.from_list("white_carmine", ["#FFFFFF", CAR])

# Print widths, in inches. Single column 90 mm, double column 190 mm.
WIDTH_SINGLE = 90 / 25.4
WIDTH_DOUBLE = 190 / 25.4

RC = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,     # cmr10 carries no Unicode minus
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "grid.color": GREY_LIGHT,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.prop_cycle": mpl.cycler(color=[MAL, CAR, GREY, "#4F907A", "#B8697A"]),
}


def use():
    """Apply the house style to the current matplotlib session."""
    mpl.rcParams.update(RC)


def figure(width="single", height=None, **kwargs):
    """A figure sized at its printed width, so text is never scaled after."""
    w = WIDTH_DOUBLE if width == "double" else (
        width if isinstance(width, (int, float)) else WIDTH_SINGLE)
    if height is None:
        height = w * 0.62
    return plt.subplots(figsize=(w, height), **kwargs)


def label(ax, x=None, y=None):
    """Axis labels in house form: capitalised name, symbol, unit.

    Write them as "Thrust, $T$ (MN)" or "Lowest $F$ (-)" for a dimensionless
    quantity. The symbol must match the equations in the text.
    """
    if x is not None:
        ax.set_xlabel(x)
    if y is not None:
        ax.set_ylabel(y)


def centred(vmin, vmax, centre=1.0):
    """A norm for the diverging map, pinned to a critical value."""
    return TwoSlopeNorm(vmin=vmin, vcenter=centre, vmax=vmax)


def save(fig, stem, outdir="."):
    """Write the vector PDF and the 300 dpi PNG the house style asks for."""
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, stem + ".pdf")
    png = os.path.join(outdir, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    return pdf, png


def greyscale_check(pdf_path, out_stem=None):
    """Render a greyscale PNG so the figure can be checked before committing.

    Malachite and carmine are close in lightness by design, so a figure that
    separates two series by hue alone will fail here. That is the point: fix
    it with line style, marker or hatching rather than with a third hue.
    """
    import subprocess
    out_stem = out_stem or os.path.splitext(pdf_path)[0] + "-grey"
    subprocess.run(["pdftocairo", "-png", "-gray", "-r", "150",
                    "-singlefile", pdf_path, out_stem], check=True)
    return out_stem + ".png"
