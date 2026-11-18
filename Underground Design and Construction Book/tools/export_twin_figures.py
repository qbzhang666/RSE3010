r"""Book figures derived from the twin demonstrator, at the book's own measure
and in the book's own palette.

The demonstrator that produces these panels lives in the X-Twin repository and
already exports figures for a journal page (its own
``scripts/export_book_chapter_figures.py``). A journal page is not this page,
and a screen palette is not this book's palette, so three things are corrected
on the way through.

**Measure.** ``SNmono`` sets a 117 mm text width, so a figure authored 13 inches
wide arrives at about a third scale and its 10 pt axis labels land at 3.5 pt.
Only panels whose authored width survives the reduction are exported. Every
figure written here reports the point size its labels reach on the page, and one
that falls below :data:`MIN_EFFECTIVE_PT` is skipped rather than shipped small.

**Palette.** The demonstrator draws in Okabe-Ito, which is a sound screen
palette and a different system from ``palette.tex``. This book uses the house
style of ``qz-figure-style``: two hues, malachite for what passes or is
designed and carmine for what fails, with further series varying by shade,
line style or grey rather than by a new hue. The panels are therefore
recoloured against those tokens: series take the cycle below, the diverging
rate scale is rebuilt asymmetrically so lightness still orders magnitude in
greyscale, and the damage categories become a monotone white-to-carmine ramp
so that severity is ordered in greyscale as well as in colour. Check the
result the way the README says to, on the PDF this writes::

    pdftocairo -png -gray -r 150 figures/ch26/damage-map.pdf out

**Caption.** The in-figure title is removed. The caption states the subject
once, below the figure, which is the Springer convention, what ``\caption`` is
for, and the only one of the two that can be edited in proof.

Each figure is written three times from the same drawing: PDF, SVG, and a PNG
that exists only to be glanced at. The two vector files are siblings rather
than conversions --- neither is made from the other, and neither is made from a
raster --- so the geometry is identical in both and the type is real type in
both. Which one the book includes is a switch in ``preamble.tex``; see
``tools/README.md``.

Nothing is re-derived here. ``build_default_state`` and the label anonymiser are
imported from the demonstrator's own exporter, so the corridor, the receptors
and the trained model are the same objects the platform draws from, and a change
to the physics reaches this book without anyone remembering to copy it.

Absent by intent: the wide panoramas (the geological long-section, the corridor
prediction) and the composed overview, which at a 117 mm measure need a
different aspect ratio rather than a different scale --- a redraw, not an
export. The IFC information-flow schematic is absent for a different reason: it
is a drawn diagram, and drawn diagrams in this book are TikZ against
``palette.tex``.

Run with the X-Twin virtual environment, which carries torch and geopandas.
From the book folder, in PowerShell::

    & (Join-Path $env:USERPROFILE ".venvs\srl-piml-313\Scripts\python.exe") tools\export_twin_figures.py

Set ``XTWIN_ROOT`` if the X-Twin checkout is not beside the RSE3010 one.
``tools/README.md`` carries the full runbook.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[1]

#: 117 mm of ``SNmono`` text width, in inches. A panel authored at 5.0 in and
#: included at ``width=\textwidth`` is reduced by 0.92, so its 10 pt labels
#: arrive at 9.2 pt. That is the whole test a figure has to pass to be here.
TEXTWIDTH_IN = 117.0 / 25.4

#: Below this, the figure is not readable on the page and belongs in a redraw
#: rather than in this export.
MIN_EFFECTIVE_PT = 7.5

# --------------------------------------------------------------------------
# palette.tex, in Python
# --------------------------------------------------------------------------
#: The tokens of ``palette.tex``, copied here and nowhere else in this file.
#: They are not adjusted in one project alone: change them in every project or
#: in none, as that file says. Malachite and carmine, the house style adopted
#: November 2026; L* is given because lightness is what a greyscale print
#: keeps.
MAL = "#2F6B57"       # L* 40.9 -- malachite: good, pass, design, supported
CAR = "#8C2B3E"       # L* 33.2 -- carmine: fail, below unity, alert
CAR_DARK = "#6E1F30"  # L* 25.3 -- the severe end
MAL_LIGHT = "#8FBCAC"  # L* 73.0 -- light malachite shade, for scale ends
MAL_TINT = "#E6EFEC"  # L* 93.7 -- fills and bands
CAR_TINT = "#F3E6E9"  # L* 92.4 -- fills and bands
CINK = "#1A1A1A"      # L*  9.3 -- axes, frames, primary label text
CTEXT = "#5A5A5A"     # L* 38.2 -- secondary annotation, reference series
CRULE = "#C8C8C8"     # L* 80.6 -- grid lines, light rules, context

#: The series order behind ``C0``..``C3``.
#:
#: The demonstrator's plots ask for ``C0`` and ``C3`` by name, so the order is
#: chosen to make those two land on the right meaning rather than merely on
#: distinct colours: ``C0`` is the principal series, and ``C3`` is what fails ---
#: the siloed baseline the federated model is measured against, and the share of
#: receptors at or above Slight damage.
SERIES_CYCLE = (MAL, CTEXT, MAL_LIGHT, CAR, CINK, MAL_TINT, CRULE, CAR_DARK)

#: The diverging rate scale, with white at the centre. Replaces ``RdBu``, and
#: in the same direction: fast and negative at the carmine end, slow and
#: positive at the malachite end.
#:
#: The house diverging map runs carmine to malachite symmetrically, and
#: :data:`tools.qz_figure.DIVERGING` provides it for quantities read in
#: colour. This figure needs the asymmetric form instead. Carmine is L* 33 and
#: malachite L* 41, so a symmetric ramp prints its two extremes as nearly the
#: same dark tone and a photocopied figure cannot tell fast settlement from
#: heave. Stopping the malachite end at the light shade, L* 73, restores the
#: reading that matters: the darker the mark, the faster the ground is going
#: down. The sign is then carried by hue in colour and by position relative to
#: white in greyscale.
RATE_RAMP = (CAR_DARK, CAR, CAR_TINT, "#FFFFFF", MAL_TINT, MAL_LIGHT)

#: The sequential scale, light to dark. Replaces ``YlOrRd``. White to carmine,
#: which is the house scale for damage and settlement magnitude, and monotone
#: in lightness (L* 100 to 92 to 33 to 25) so severity is ordered in a
#: greyscale print without reference to hue.
MAGNITUDE_RAMP = ("#FFFFFF", CAR_TINT, CAR, CAR_DARK)


def _locate_xtwin() -> Path:
    """The X-Twin checkout: ``XTWIN_ROOT``, or a sibling of this repository."""
    override = os.environ.get("XTWIN_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
    else:
        root = (BOOK_ROOT.parents[1] / "Digital-Twin").resolve()
    if not (root / "scripts" / "export_book_chapter_figures.py").exists():
        raise SystemExit(
            f"X-Twin exporter not found under {root}.\n"
            "Set XTWIN_ROOT to the Digital-Twin checkout."
        )
    return root


def _load_exporter(xtwin_root: Path):
    """Import the demonstrator's exporter as a module, by path.

    By path rather than by package name because the X-Twin checkout is not on
    ``sys.path`` and should not be installed to make a figure: the book depends
    on the demonstrator for its data, not for its packaging.
    """
    if str(xtwin_root) not in sys.path:
        sys.path.insert(0, str(xtwin_root))
    spec = importlib.util.spec_from_file_location(
        "xtwin_export", xtwin_root / "scripts" / "export_book_chapter_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_book_palette(plotting) -> None:
    """Recolour the demonstrator's template against ``palette.tex``.

    Three interceptions, because the demonstrator states its colours in three
    different ways and none of them should be edited there --- the platform is
    entitled to its own screen palette, and a book-specific fork of its plotting
    module would drift the first time either changed.

    1. ``rcParams``, for everything drawn from the cycle or the template: the
       series colours, the ink, the grid.
    2. The module constant for the damage categories, which is a list the
       plotting functions read at call time.
    3. ``Axes.scatter``, for the two colour scales, which the plotting functions
       request by the builtin names ``RdBu`` and ``YlOrRd``. Matplotlib refuses
       to let a builtin name be re-registered, so the substitution is made where
       the scale is asked for instead --- which is also the only place early
       enough to be right: a colour bar is built from the mappable at draw time,
       so a cmap swapped after the fact leaves the scale beside the figure
       describing the old one. The patch lives in this process only, which draws
       nothing but these seven panels.
    """
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=list(SERIES_CYCLE)),
        "text.color": CINK,
        "axes.labelcolor": CINK,
        "axes.edgecolor": CINK,
        "axes.titlecolor": CINK,
        "xtick.color": CINK,
        "ytick.color": CINK,
        "xtick.labelcolor": CINK,
        "ytick.labelcolor": CINK,
        "legend.labelcolor": CINK,
        "legend.edgecolor": CRULE,
        "grid.color": CRULE,
    })

    magnitude = LinearSegmentedColormap.from_list(
        "book_magnitude", list(MAGNITUDE_RAMP))
    rate = LinearSegmentedColormap.from_list("book_rate", list(RATE_RAMP))
    substitutions = {"YlOrRd": magnitude, "RdBu": rate}

    original_scatter = mpl.axes.Axes.scatter

    def scatter(self, *args, **kwargs):
        requested = kwargs.get("cmap")
        if isinstance(requested, str) and requested in substitutions:
            kwargs["cmap"] = substitutions[requested]
        return original_scatter(self, *args, **kwargs)

    mpl.axes.Axes.scatter = scatter

    # Five ordered damage categories, sampled off the magnitude ramp so that
    # Negligible to Severe is a monotone lightness sequence, and so that
    # Slight, the category at which an assessment stops being a screening
    # exercise, lands in the carmine half of the ramp. The sampling starts
    # above zero because the foot of the ramp is white and vanishes in print;
    # the plotting helper already holds Negligible back with a smaller marker,
    # so it does not also need to be invisible.
    plotting.DAMAGE_COLORS = [
        mpl.colors.to_hex(magnitude(value))
        for value in (0.10, 0.34, 0.55, 0.80, 1.0)
    ]


def _recolour_literals(fig) -> None:
    """Replace the colours the plotting helpers state as literals.

    Station lines are drawn navy, reference lines black, and the axes are given
    a faintly tinted panel. None of the three comes from the template, so none
    is caught by ``rcParams``: they are remapped here, on the built figure,
    against the same tokens.
    """
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    literal = {
        mpl.colors.to_rgba("navy"): CINK,
        mpl.colors.to_rgba("black"): CINK,
    }
    for axis in fig.axes:
        # The tinted panel: a book page is white, and a figure that brings its
        # own near-white ground reads as a pasted screenshot beside one that
        # does not.
        if mpl.colors.to_hex(axis.get_facecolor()) != "#ffffff":
            axis.set_facecolor("white")
        for collection in axis.collections:
            edges = collection.get_edgecolors()
            if len(edges) and tuple(edges[0]) in literal:
                collection.set_edgecolors(CINK)
        for patch in axis.patches:
            if mpl.colors.to_rgba(patch.get_edgecolor()) in literal:
                patch.set_edgecolor(CINK)
    for line in fig.findobj(Line2D):
        replacement = literal.get(mpl.colors.to_rgba(line.get_color()))
        if replacement is not None:
            line.set_color(replacement)


#: Wording the demonstrator uses on screen, and the wording this book uses.
#:
#: A figure has to speak the vocabulary of the text around it. The platform
#: labels its axes for an operator who knows the run --- ``Federated``,
#: ``Siloed``, ``TBM-2``, ``LOS`` --- and none of those terms is established
#: anywhere in this book, so a reader would meet them first in a legend, which
#: is the one place a term cannot be defined. The substitutions say the same
#: thing in the words the chapters use. ``(pattern, replacement)``, matched
#: against the whole label.
BOOK_LABELS = (
    (r"^Federated \(TBMs.*\)$", "Trained on the other machines"),
    (r"^Siloed \(TBM-?\s?\d+ only\)$", "Trained on the withheld machine only"),
    (r"^Federated \+ finetune \((.*)\)$", r"With local calibration (\1)"),
    (r"^RMSE on held-out TBM \(mm\)$", "RMSE on the withheld machine (mm)"),
    (r"^Federated round$", "Training round"),
    (r"^LOS rate \(mm/yr\)$", "Line-of-sight rate (mm/yr)"),
    (r"^TBM contribution \(mm\)$", "Construction contribution (mm)"),
    (r"^TBM \(mm\)$", "Construction\ncontribution (mm)"),
    (r"^1:1 line\n\(no TBM impact\)$", "1:1 line\n(no construction effect)"),
    (r"^Pre-construction cumulative LOS \(mm\)$",
     "Movement before construction (mm)"),
    (r"^During-construction LOS \(mm\)$", "Cumulative movement at end of construction (mm)"),
    # The demonstrator bins the settlement it computes at 5, 10, 25 and 50 mm
    # and gives the bins Burland's damage-category names. The bins are a
    # reasonable first-stage screen; they are not a damage assessment, which
    # Chapter 26 defines on limiting tensile strain and which needs the
    # building's own stiffness, its foundation and its orientation to the
    # trough. A figure labelled "Slight" beside that text would contradict it,
    # so the categories are relabelled as the settlement bands they are.
    (r"^Negligible \((\d+)\)$", r"Below 5 mm (\1)"),
    (r"^Very slight \((\d+)\)$", r"5 to 10 mm (\1)"),
    (r"^Slight \((\d+)\)$", r"10 to 25 mm (\1)"),
    (r"^Moderate \((\d+)\)$", r"25 to 50 mm (\1)"),
    (r"^Severe \((\d+)\)$", r"Above 50 mm (\1)"),
    (r"^Negligible$", "Below 5"),
    (r"^Very slight$", "5 to 10"),
    (r"^Slight$", "10 to 25"),
    (r"^Moderate$", "25 to 50"),
    (r"^Severe$", "Above 50"),
    (r"^% buildings . Slight damage$",
     "Buildings settling more than 10 mm (%)"),
)

#: Axis labels the demonstrator leaves unset because the category names carried
#: the meaning on screen. Once those become numeric bands, the axis has to say
#: what the numbers are. Keyed by output stem.
AXIS_LABELS = {
    "damage-categories": {"xlabel": "Settlement caused by construction (mm)"},
}


def _relabel_for_book(fig, stem: str = "") -> None:
    """Rewrite screen wording into the book's own, on the built figure."""
    import re

    import matplotlib as mpl

    def rewrite(label: str) -> str:
        for pattern, replacement in BOOK_LABELS:
            new, count = re.subn(pattern, replacement, label)
            if count:
                return new
        return label

    # Categorical tick labels are created lazily, so they do not exist to be
    # rewritten until the figure has been drawn once.
    fig.canvas.draw()
    for axis in fig.axes:
        current = [item.get_text() for item in axis.get_xticklabels()]
        rewritten = [rewrite(item) for item in current]
        if rewritten != current:
            axis.set_xticks(axis.get_xticks(), labels=rewritten)
        current = [item.get_text() for item in axis.get_yticklabels()]
        rewritten = [rewrite(item) for item in current]
        if rewritten != current:
            axis.set_yticks(axis.get_yticks(), labels=rewritten)

    artists = list(fig.findobj(mpl.text.Text))
    for axis in fig.axes:
        artists.extend([axis.xaxis.label, axis.yaxis.label])
        artists.extend(axis.get_xticklabels())
        artists.extend(axis.get_yticklabels())
        if axis.get_legend() is not None:
            artists.extend(axis.get_legend().get_texts())
    for artist in {id(item): item for item in artists}.values():
        artist.set_text(rewrite(artist.get_text()))

    for name, value in AXIS_LABELS.get(stem, {}).items():
        getattr(fig.axes[0], f"set_{name}")(value)


def _strip_titles(fig) -> None:
    """Remove in-figure titles; the LaTeX caption carries the subject.

    A title inside the artwork and a caption below it say the same thing twice,
    and the one inside cannot be edited in proof.
    """
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    for axis in fig.axes:
        axis.set_title("")


def _effective_pt(fig, base_pt: float = 10.0) -> float:
    """Point size a 10 pt label reaches once the figure is set to the measure."""
    width_in = fig.get_size_inches()[0]
    return base_pt * min(1.0, TEXTWIDTH_IN / width_in)


#: ``(chapter directory, output stem, builder name, arguments)``.
#:
#: The chapter each panel is written for. A file lands in ``figures/chNN`` and is
#: included by that chapter only once the text is written around it.
MANIFEST = (
    ("ch02", "twin-transfer", "plot_training_curves", ("result",)),
    ("ch02", "twin-anomaly", "plot_anomaly_detection", ("buildings",)),
    ("ch26", "insar-baseline", "plot_baseline", ("buildings",)),
    ("ch26", "settlement-attributable", "plot_tbm_attributable", ("buildings",)),
    ("ch26", "damage-map", "plot_damage_map", ("buildings",)),
    ("ch26", "damage-categories", "plot_damage_histogram", ("buildings",)),
    ("ch28", "receptor-risk", "plot_station_risk", ("buildings",)),
)


def export(output_root: Path, preview: bool = True) -> None:
    xtwin_root = _locate_xtwin()
    exporter = _load_exporter(xtwin_root)
    try:                                  # package renamed in the X-Twin repository
        from x_twin_piml import plotting as P
    except ImportError:
        from srl_piml import plotting as P

    exporter._configure_vector_output()   # vector template, embedded fonts
    _apply_book_palette(P)                # then this book's colours over it
    corridor, buildings, result = exporter.build_default_state()
    state = {"corridor": corridor, "buildings": buildings, "result": result}

    import matplotlib.pyplot as plt

    for chapter, stem, builder_name, argnames in MANIFEST:
        builder = getattr(P, builder_name)
        produced = builder(*(state[name] for name in argnames))
        fig = produced[0] if isinstance(produced, tuple) else produced

        # Anonymise first: the corridor identifiers become case labels, which is
        # also what keeps project names out of the book.
        exporter._anonymise_publication_labels(fig)
        _relabel_for_book(fig, stem)
        _strip_titles(fig)
        _recolour_literals(fig)

        effective = _effective_pt(fig)
        if effective < MIN_EFFECTIVE_PT:
            print(f"SKIP {stem}: {fig.get_size_inches()[0]:.1f} in wide, "
                  f"labels would set at {effective:.1f} pt", flush=True)
            plt.close(fig)
            continue

        target = output_root / chapter
        target.mkdir(parents=True, exist_ok=True)
        # Both vector formats, from the same figure object. Neither is derived
        # from the other and neither is derived from a raster: each is written
        # by Matplotlib straight from the drawing, so the geometry is identical
        # and the type is real type in both.
        fig.savefig(target / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(target / f"{stem}.svg", format="svg", bbox_inches="tight")
        if preview:
            fig.savefig(target / f"{stem}.png", dpi=200, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)
        print(f"exported {chapter}/{stem}  ({effective:.1f} pt on the page)",
              flush=True)

    print(f"demonstrator state: fine-tuned RMSE={result.rmse_fed_finetune:.2f} mm, "
          f"siloed RMSE={result.history_silo[-1]:.2f} mm", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BOOK_ROOT / "figures",
                        help="Figure root (default: the book's figures/)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Write only the PDF the book includes")
    args = parser.parse_args()
    export(args.output_dir.resolve(), preview=not args.no_preview)


if __name__ == "__main__":
    main()
