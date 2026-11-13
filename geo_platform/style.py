"""House style of the RSE3010 Geotechnical Platform: palette and figure helpers.

The same palette as the book *Underground Design and Construction* and the
pumped hydro platform:

* two hues carry meaning: malachite (design, pass, supported, ground) and
  carmine (alert, fail, demand, loss); everything else is a grey;
* more than two series are told apart by line style or marker, not by adding
  hues; a reference or limit series is grey and dashed;
* axis labels read "Name, symbol (unit)" with a capitalised name;
* figure text is Computer Modern (``cmr10`` ships with matplotlib).

Streamlit is never imported here; matplotlib and plotly are imported inside the
functions that use them.
"""

from __future__ import annotations

from typing import Any

MALACHITE = "#2F6B57"
CARMINE = "#8C2B3E"
CARMINE_DARK = "#6E1F30"
MALACHITE_TINT = "#E6EFEC"
CARMINE_TINT = "#F3E6E9"
INK = "#1A1A1A"
GREY_DARK = "#5A5A5A"
GREY = "#8A8A8A"
GREY_LIGHT = "#C8C8C8"
WHITE = "#FFFFFF"

REFERENCE_COLOUR = GREY

_DASHES_MPL = ("-", "--", ":", "-.")
_DASHES_PLOTLY = ("solid", "dash", "dot", "dashdot")
_MARKERS_MPL = ("o", "s", "^", "D")
_MARKERS_PLOTLY = ("circle", "square", "triangle-up", "diamond")


def _rgb(colour: str) -> tuple[int, int, int]:
    value = colour.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def tint(colour: str, fraction: float) -> str:
    """Blend a palette colour towards white: 0 keeps it, 1 gives white."""
    share = min(max(float(fraction), 0.0), 1.0)
    mixed = [int(c + (255 - c) * share + 0.5) for c in _rgb(colour)]
    return "#{:02X}{:02X}{:02X}".format(*mixed)


def rgba(colour: str, alpha: float) -> str:
    """``rgba(r, g, b, a)`` string for a translucent fill."""
    red, green, blue = _rgb(colour)
    return f"rgba({red}, {green}, {blue}, {min(max(float(alpha), 0.0), 1.0):g})"


def series_style(index: int, library: str = "matplotlib") -> dict[str, str]:
    """Colour, line style and marker of the ``index``-th data series.

    Series alternate malachite and carmine; each further pair changes the line
    style and the marker, so the series stay distinct in greyscale.
    """
    position = int(index)
    colour = (MALACHITE, CARMINE)[position % 2]
    variant = (position // 2) % len(_DASHES_MPL)
    if library == "plotly":
        return {"color": colour, "dash": _DASHES_PLOTLY[variant], "symbol": _MARKERS_PLOTLY[variant]}
    return {"color": colour, "linestyle": _DASHES_MPL[variant], "marker": _MARKERS_MPL[variant]}


def reference_style(library: str = "matplotlib") -> dict[str, str]:
    """Grey dashed style for a reference, limit or screening series."""
    if library == "plotly":
        return {"color": REFERENCE_COLOUR, "dash": "dash", "symbol": "circle-open"}
    return {"color": REFERENCE_COLOUR, "linestyle": "--", "marker": ""}


def axis_label(name: str, symbol: str | None = None, unit: str | None = None, *, math: bool = False) -> str:
    """Axis label in the house form "Name, symbol (unit)"; ``(-)`` if dimensionless."""
    label = str(name or "").strip()
    if label:
        label = label[0].upper() + label[1:]
    sym = str(symbol or "").strip()
    if sym:
        if math and not (sym.startswith("$") and sym.endswith("$")):
            sym = f"${sym}$"
        label = f"{label}, {sym}" if label else sym
    unit_text = str(unit or "").strip() or "-"
    return f"{label} ({unit_text})"


def matplotlib_rcparams() -> dict[str, Any]:
    """House rcParams; use with ``plt.rc_context(matplotlib_rcparams())``."""
    from matplotlib import cycler

    return {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": GREY_LIGHT,
        "grid.linewidth": 0.4,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "axes.prop_cycle": cycler(color=[MALACHITE, CARMINE, GREY_DARK, GREY]),
    }


def plotly_layout(**overrides: Any) -> dict[str, Any]:
    """Base plotly layout in the house style; keyword arguments override it."""
    axis = dict(
        showline=True, linecolor=INK, linewidth=0.8, mirror=True,
        ticks="inside", gridcolor=GREY_LIGHT, gridwidth=0.4, zeroline=False,
        title_font=dict(size=13), tickfont=dict(size=11),
    )
    layout: dict[str, Any] = dict(
        template="simple_white",
        font=dict(family="'CMU Serif', 'Latin Modern Roman', cmr10, Georgia, serif", color=INK, size=13),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        xaxis=dict(axis),
        yaxis=dict(axis),
        legend=dict(bgcolor="rgba(255,255,255,0)", borderwidth=0, font=dict(size=11)),
        margin=dict(l=60, r=20, t=30, b=55),
        hoverlabel=dict(bgcolor=WHITE, font_color=INK),
    )
    layout.update(overrides)
    return layout
