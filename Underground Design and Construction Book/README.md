# Underground Design and Construction

Springer book manuscript. **28 chapters in six parts**, 6 appendices, 328 pages, builds clean.

The manuscript is fully drafted: no outline or TODO markers remain anywhere, in either the chapters or the appendices.

## Format

Uses the official Springer Nature **SNmono** class, v6.00 (2025-11-11), with the same setup as the pumped hydro book in `RSE3241/Springer_Handbook`: `\documentclass[graybox,envcountchap,sectrefs]{SNmono}`, the 155 × 235 mm trim with the 117 × 191 mm text block centred on it, and newtxtext/newtxmath. `SNmono.cls` and `svind.ist` are byte-identical to the copies in that book. References close each chapter as an unnumbered section, as in that book; here they are produced by biblatex with biber (`refsection=chapter`) from the single `references.bib`, and every chapter and appendix file ends with `\chapterreferences`, which prints nothing where a chapter cites nothing. Entries and citations follow the Springer author--year (basic) style, set by `bibliography-style.tex`: `Hoek E, Diederichs MS (2006) Empirical estimation of rock mass modulus. Int J Rock Mech Min Sci 43(2):203--215`, cited as (Hoek and Diederichs 2006). `springer_book.sty` is this book's own semantic layer.

A full-width figure is therefore 117 mm wide, not the 190 mm double-column width of the journal figure style.

**No boxes.** The 121 framed asides inherited from the teaching notes were dissolved into running text: a meaningful title became a bold run-in lead-in, a generic label was dropped. `tcolorbox` is gone, and although the `graybox` class option is set for parity with the pumped hydro book, no box environment is used, so no passage of running text is shaded. Springer's own `important` and `backgroundinformation` environments remain available through `springer_book.sty` if ever wanted.

**Colour in figures only.** `palette.tex` holds the shared figure palette — malachite and carmine, the house style of `qz-figure-style`, adopted November 2026 — identical token for token to `RSE3241/Springer_Handbook/slide_palette_tikz.tex` and `Tri-HB/Springer_Book/figure_sources/palette.tex`, so a figure carries between the three projects without recolouring. Two hues only: malachite for what passes, is designed or is supported, carmine for what fails, falls below unity or raises an alert. Further series vary by shade of those two hues, by line style, marker or hatching, never by adding a hue; a reference or screening series is grey. The values are not adjusted in one project alone: change them everywhere or nowhere.

This replaced a four-hue palette (red, teal, green, gold over four tinted fills) on two measurements. In greyscale its four fills collapsed to within about 4 L\* of one another, so a printed reader could not tell a computation box from an adopted-value box; and two of its four stroke-on-own-fill pairs measured 2.7:1 and 2.8:1, below the 3:1 floor for graphical objects that the palette itself claimed to honour. Every stroke now reads at least 6.2:1 on white and 5.3:1 on its own tint.

Figures redrawn outside TikZ use `tools/qz_figure.py`, which carries the same tokens, the Computer Modern rc settings, the 90 mm and 190 mm print widths, the diverging and sequential colour maps, and a `greyscale_check()` helper.

## Build

```bash
latexmk -pdf -outdir=build main.tex
```

Output lands in `build/main.pdf`. Clean with `latexmk -C -outdir=build main.tex`. To draft one chapter, uncomment `\includeonly{...}` in `main.tex`; cross-references and page numbers stay correct provided `build/main.aux` is kept.

## Structure

`main.tex` defines the book order. Part order follows the project lifecycle.

| Part | Chapters |
| --- | --- |
| I. Challenges, Opportunities and Digital Practice | Challenges and Opportunities in Modern Practice · Digital Engineering and the Lifecycle Digital Twin · Sustainability, Carbon and Resilience |
| II. Site and Ground Characterisation | Engineering Geology · Site Investigation and Data Reliability · Three-Dimensional Ground Modelling · Groundwater and Hydrogeology · In-Situ Stress · Discontinuities and Structural Analysis · Rock Mass Classification |
| III. Geomechanical Parameters and Ground Behaviour | Stress, Strain and Intact Rock Behaviour · Rock Mass Strength and Design Parameters · Ground Behaviour and Face Stability · Squeezing, Swelling and High-Stress Ground |
| IV. Design and Analysis: From Preliminary to Detailed Design | Empirical Design Methods · Analytical Methods and Ground–Support Interaction · Numerical Modelling · Support and Reinforcement · Linings, Waterproofing and Durability · Seismic and Dynamic Design · Reliability, Optimisation and Integrated Design |
| V. Construction | Excavation Methods and Design Philosophies · Mechanised Tunnelling · Shafts, Caverns and Multiple Openings · Ground Improvement, Pre-Support and Construction Hazards |
| VI. Performance and Asset Life | Ground Movement, Monitoring and the Observational Method · Operation and Asset Management · Risk, Contracts and Project Delivery |

Part I establishes the criteria that now govern how projects are conceived. Part II characterises the ground; Part III derives parameters and behaviour; Part IV designs, running the three method families in order of increasing resolution and cost; Part V builds; Part VI operates.

## Layout

| Path | Contents |
| --- | --- |
| `main.tex` | Document class, part structure, chapter includes |
| `preamble.tex` | Packages, notation macros, TikZ styles, draft aids |
| `palette.tex` | Shared figure palette, common to the slide set and the Tri-HB book |
| `SNmono.cls`, `springer_book.sty`, `svind.ist` | Springer class and support files |
| `frontmatter/` | Preface, how to use this book, notation and units, list of symbols |
| `chapters/` | 28 chapter files, `c01`–`c28`, numbered in book order |
| `backmatter/` | Appendices A–F; A describes the companion open-access application |
| `references.bib` | Bibliography, 91 entries. Article titles in sentence case with proper nouns braced; German and French titles carry `langid`; each article has a `shortjournal` (ISO 4 abbreviation), which is what prints |
| `bibliography-style.tex` | Springer author--year reference style on biblatex, loaded by `main.tex` |
| `figures/` | Chapter figures in `chNN/`, as PDF and SVG, plus a PNG preview |
| `tools/` | `export_twin_figures.py`, the demonstrator figure export |
| `CLASSICAL-CROSSCHECK.md` | Mapping against Hoek & Brown (1980) and Brady & Brown (2006) |

## Demonstrator figures

Seven data figures come from the twin demonstrator in the X-Twin repository, exported by `tools/export_twin_figures.py`. Run it with that project's environment, which carries torch and geopandas — the plain `python` on `PATH` has matplotlib but neither of the others.

```powershell
& (Join-Path $env:USERPROFILE ".venvs\srl-piml-313\Scripts\python.exe") tools\export_twin_figures.py
```

`tools/README.md` is the runbook: prerequisites, the rebuild and greyscale-check steps that follow, the options, and what to do when it fails. The book compiles from the committed figures, so none of this is needed to build.

Each figure is written as PDF and as SVG from the same drawing — siblings rather than conversions, so both are vector and both carry real type. The chapters call `\datafigure{ch02/twin-transfer}` with no extension, and a switch in `preamble.tex` chooses the route: `\svgfiguresfalse` includes the PDF directly, `\svgfigurestrue` includes the SVG through the `svg` package, which needs Inkscape on `PATH` and `-shell-escape`. PDF is the default because it needs no external program; the SVG route is there for when an SVG is being hand-edited and those edits must reach the page without a re-export.

The export is not a copy. Three things are corrected on the way into the book, and the reasons are in the script's own docstring: the figure is held to a size whose type still sets at 9 pt or better against the 117 mm measure, and one that would set smaller is skipped rather than shipped; the Okabe-Ito screen palette is replaced by the tokens of `palette.tex`, with the two colour scales rebuilt from the rate ramp that file already defines; and the screen vocabulary is rewritten into the book's own. That last one is not cosmetic — the demonstrator bins settlement at 5, 10, 25 and 50 mm and labels the bins with Burland's damage-category names, which is a defensible first-stage screen and is not what Chapter 26 defines a damage category to be, so the bands are exported as the settlement magnitudes they are.

Two figures are deliberately not exported. The geological long-section and the corridor settlement prediction are authored at 13 inches and 4:1, so at this measure they need a new aspect ratio rather than a smaller scale; that is a redraw. The IFC information-flow schematic is a drawn diagram, and drawn diagrams here are TikZ against `palette.tex`.

All seven carry synthetic data, and every caption says so.

## Editorial conventions

Enforced throughout; new material must respect them.

- **No teaching apparatus.** No learning objectives, assignment tasks, key takeaways, week or lecture references, no second-person address, and no reference to a particular design study or brief.
- **Tool-neutral.** No method depends on a named program. Every calculation is stated so it can be followed by hand, with the longer ones set out as numbered sequences in Appendix H. Where a calculation is tedious by hand, the text points to the companion open-access application (Appendix C, `\companionname` / `\companionurl` in `preamble.tex`), never to a commercial package. Commercial software is discussed only in Appendix C, as a class rather than by name.
- **No project names in headings.** Individual projects appear only as examples in body text, sourced from published environmental statements, baseline reports, owner technical reports or peer-reviewed papers, never from news reporting. Headings read "a shallow urban metro tunnel" or "a deep pumped-storage scheme".
- **No "underground excavation" or "underground works" in chapter titles.** The whole book is underground.
- **At most 8 sections per chapter**, currently 2–6.
- **Every section has exactly 0 or 3 subsections**, never 1 or 2. Finer divisions use run-in `\paragraph` headings, so numbering never reaches a third level.
- **Chapter labels are semantic**, not numeric: `\label{ch:ccm}`, not `\label{ch:7}`, so references survive reordering.
- **Figure colour is semantic, never decorative.** A workflow node takes one size style and one role style, `[smallworkflowbox, wfstep]`. Because the palette carries two hues and the diagrams carry four roles, role is encoded as stroke hue × fill: `wfinput` (grey stroke, white fill — what the sequence starts from), `wfstep` (malachite, white fill — a computation), `wfadopt` (malachite, tinted fill — the value that enters the design), `wfcheck` (carmine, tinted fill — a comparison against a threshold) and `feedbackarrow` (carmine dashed — the return path). An empty box is a step; a tinted box is an outcome.
- **No figure depends on colour alone.** Malachite and carmine are only 7.7 L\* apart, so the hues read alike in greyscale by design and the fill carries the distinction. Steps are numbered, the forward path is solid and the return path dashed. Check a new figure before committing it: `pdftocairo -png -gray -r 150 file.pdf out`, or `qz_figure.greyscale_check()` for a redrawn one.
- **Figure text is Computer Modern.** Set on TikZ nodes through the `wffont` style and native in `qz_figure.py`. Mathematics inside a TikZ node still takes the document's `newtxmath`, because the Springer class fixes the document font; that mixture is transitional and resolves as figures are redrawn outside TikZ.

## Referencing

Citations are placed at the first substantive mention of a source in each chapter. `\citet` where the sentence needs the author's name — "the idealisation of Burland (1974)" — and `\citep` where it does not, typically after a named method or criterion: "the Terzaghi correction (Terzaghi, 1965)". The bibliography resolves through biber and renders. 44 citations across 28 chapters.

Two caveats stand before typesetting.

1. **Bibliographic detail needs verification.** Author names and titles are right; volume, issue, page and year fields were written from the standard citations rather than looked up, and must be checked in proof.
2. **Values in Appendices F and G need verification.** The closed-form expressions and the tabulated material constants were compiled from the standard references. Appendix G carries that warning to the reader as well, since its tables are intended for checking a measured value rather than for design.

## Outstanding work

1. **Figures.** Five workflow diagrams in Chapters 10 to 12, drawn in TikZ, and seven data figures in Chapters 2, 26 and 28 exported from the twin demonstrator (see *Demonstrator figures* below). Part-opening lifecycle diagrams, the ground reaction and support characteristic curves, the settlement trough, the stereonet constructions of Appendix E, and the blank nets are all referenced in the text and not yet drawn. New figures use `palette.tex` and the conventions above.
2. **Citation density.** 44 citations across 28 chapters is thin for a monograph. Passages presenting empirical relations, case-history evidence and code provisions should carry attribution more consistently.
3. **Worked numerical examples.** The calculation sequences in Appendix H are procedures rather than worked arithmetic; several would be more useful carried through with numbers.
4. **Index.** `makeidx` is loaded and no `\index` entries have been placed.
