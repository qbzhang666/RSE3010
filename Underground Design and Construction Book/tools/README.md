# Regenerating the demonstrator figures

`export_twin_figures.py` rebuilds the seven data figures that Chapters 2, 26 and 28 include. Run it whenever the demonstrator's physics, data or plotting changes; nothing else in the book depends on it, and the book compiles from the committed PDFs without it.

Everything below is PowerShell, run from **this repository's book folder**.

## Prerequisites

| Needs | Where it is | If missing |
| --- | --- | --- |
| Python with torch, geopandas, shapely | `%USERPROFILE%\.venvs\srl-piml-313` | See *Rebuilding the environment* below |
| The X-Twin checkout | `..\..\Digital-Twin`, beside `RSE3010` | Set `XTWIN_ROOT` (below) |
| MiKTeX (`latexmk`, `pdftocairo`) | on `PATH` | Only needed for steps 2 and 3 |

The plain `python` on `PATH` will **not** work — it has matplotlib but not torch or geopandas, and the script needs all three.

## 1. Export the figures

```powershell
Set-Location "C:\Users\qianbinz\Monash Uni Enterprise Dropbox\Qianbing Zhang\00-GitHub\RSE3010\Underground Design and Construction Book"
$py = Join-Path $env:USERPROFILE ".venvs\srl-piml-313\Scripts\python.exe"
& $py tools\export_twin_figures.py
```

Takes about two minutes. Nearly all of it is the federated training run, which happens before any figure is drawn, so expect a long silence and then seven lines at once.

Expected output:

```
exported ch02/twin-transfer  (9.2 pt on the page)
exported ch02/twin-anomaly  (9.2 pt on the page)
exported ch26/insar-baseline  (9.2 pt on the page)
exported ch26/settlement-attributable  (9.2 pt on the page)
exported ch26/damage-map  (9.2 pt on the page)
exported ch26/damage-categories  (10.0 pt on the page)
exported ch28/receptor-risk  (10.0 pt on the page)
demonstrator state: fine-tuned RMSE=1.98 mm, siloed RMSE=3.74 mm
```

The last line is the reproducibility check. The demonstrator is seeded at 7, so those two numbers should come back the same every run; if they move, the demonstrator changed underneath the book and the captions in Chapter 2 need re-reading, since one of them prints 1.98 mm in the figure legend.

The point size beside each name is what a 10 pt label in that figure reaches on the 117 mm page. A figure that would land under 7.5 pt prints `SKIP` instead of exporting — that is the guard working, not a failure. See *Two figures it will not export* below.

## 2. Rebuild the book

```powershell
latexmk -pdf -outdir=build main.tex
```

Output is `build\main.pdf`. To confirm the new figures were picked up:

```powershell
Select-String -Path build\main.log -Pattern "figures/ch\d+/[a-z-]+\.pdf" -AllMatches | ForEach-Object { $_.Matches.Value } | Sort-Object -Unique
```

Seven paths should come back.

## 3. Check the greyscale

Required by the book's figure conventions — see the three rules at the top of `palette.tex`. Colour must never be the only carrier of meaning, and the check is on the exported PDF, not on the screen preview.

```powershell
$out = Join-Path $env:TEMP "greycheck"; New-Item -ItemType Directory -Force $out | Out-Null
Get-ChildItem figures\ch*\*.pdf | ForEach-Object { pdftocairo -png -gray -r 150 $_.FullName (Join-Path $out $_.BaseName) }
Invoke-Item $out
```

Look for two things: that no two categories which mean different things have collapsed to the same tone, and that any colour bar still reads monotonically from one end to the other.

## What it writes

| File | Chapter | Figure |
| --- | --- | --- |
| `figures/ch02/twin-transfer.pdf` | 2 | Validation split at the machine |
| `figures/ch02/twin-anomaly.pdf` | 2 | Movement attribution against the 1:1 line |
| `figures/ch26/insar-baseline.pdf` | 26 | Pre-construction line-of-sight rates |
| `figures/ch26/settlement-attributable.pdf` | 26 | The same buildings, baseline removed |
| `figures/ch26/damage-map.pdf` | 26 | Settlement band by position |
| `figures/ch26/damage-categories.pdf` | 26 | The population by settlement band |
| `figures/ch28/receptor-risk.pdf` | 28 | Exposure aggregated by zone |

Each is written three times from the same drawing: `.pdf`, `.svg` and a `.png` at 200 dpi. The two vector files are siblings, not conversions — neither is made from the other, and neither is made from a raster — so the geometry is identical in both and the type is real type in both. The PNG exists only to be glanced at; `--no-preview` suppresses it.

Which vector file the book includes is a switch in `preamble.tex`, described next.

## PDF or SVG

The chapters call `\datafigure[<width>]{<chapter>/<name>}` with no extension. `preamble.tex` defines it one of two ways:

```latex
\svgfiguresfalse   % PDF via \includegraphics  -- the default
\svgfigurestrue    % SVG via \includesvg
```

**The switch does not decide whether the figure is vector.** It is vector either way; it decides which program converts the drawing for the page. `pdflatex` embeds a PDF natively. The SVG route hands the file to the `svg` package, which calls Inkscape at compile time, has Inkscape write a PDF, and includes that — so it needs Inkscape on `PATH` and the whole document compiled with `-shell-escape`:

```powershell
latexmk -pdf -shell-escape -outdir=build main.tex
```

Inkscape is **not installed on this machine**, so turning the switch on today fails with `Package svg Error: Inkscape version not detected`. Install it, put `inkscape.exe` on `PATH`, and the switch works with no other change.

Reach for the SVG route when an SVG is being edited by hand — recolouring a single mark, moving an annotation, fixing a label the export cannot reach — and those edits have to land on the page without a re-export. For everything the export already handles, the PDF route is faster, needs no external program, and is what the committed build uses.

## Options

```powershell
& $py tools\export_twin_figures.py --output-dir D:\scratch\figs   # write somewhere else
& $py tools\export_twin_figures.py --no-preview                   # PDFs only
$env:XTWIN_ROOT = "D:\path\to\Digital-Twin"                       # non-standard checkout location
```

`XTWIN_ROOT` lasts for the session. To set it permanently:

```powershell
[Environment]::SetEnvironmentVariable("XTWIN_ROOT", "D:\path\to\Digital-Twin", "User")
```

## Two figures it will not export

The geological long-section and the corridor settlement prediction are authored 13 inches wide at roughly 4:1. Reduced to a 117 mm measure their labels set at about 3.5 pt, so they need a different aspect ratio rather than a smaller scale. That is a redraw in the demonstrator's plotting module, not a setting here. The IFC information-flow schematic is excluded for a different reason: it is a drawn diagram, and drawn diagrams in this book are TikZ against `palette.tex`.

## Troubleshooting

**`X-Twin exporter not found under ...`** — the Digital-Twin checkout is not where the script expects. Set `XTWIN_ROOT`.

**`ModuleNotFoundError: No module named 'torch'`** — the wrong interpreter. Check `$py` resolves:

```powershell
Test-Path (Join-Path $env:USERPROFILE ".venvs\srl-piml-313\Scripts\python.exe")
```

**`Re-registering the builtin cmap ... is not allowed`** — a matplotlib version that refuses colormap overrides. The script does not do that any more; if this appears, an older copy is being run.

**`Package svg Error: Inkscape version not detected`** — `\svgfigurestrue` is set in `preamble.tex` but Inkscape is not on `PATH`. Either install it and compile with `-shell-escape`, or set `\svgfiguresfalse` and use the PDFs. Both routes draw the same vector figure.

**A figure prints `SKIP`** — its authored width grew. Either the plotting helper changed its `figsize`, or a new entry was added to `MANIFEST` that was never sized for a book page.

**Warnings about `streamlit.runtime.caching` and `No runtime found`** — harmless. The demonstrator's data helpers carry Streamlit cache decorators that degrade to plain functions outside a Streamlit process.

**`MiKTeX requires Windows 10 (or greater)`, once per file in step 3** — harmless, and not a Windows problem. MiKTeX prints it to stderr on every `pdftocairo` call; the PNGs are written regardless. They land as `<name>-1.png`, the `-1` being the page number.

## Rebuilding the environment

If the venv is gone, recreate it from the X-Twin repository, which owns these dependencies:

```powershell
py -3.13 -m venv $env:USERPROFILE\.venvs\srl-piml-313
& "$env:USERPROFILE\.venvs\srl-piml-313\Scripts\python.exe" -m pip install -r "..\..\Digital-Twin\requirements.txt"
```
