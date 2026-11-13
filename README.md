# RSE3010 — companion applications

This repository serves two audiences, and it is worth saying which file is for
whom before anything else.

- **Readers of *Underground Design and Construction*.** The book is tool
  neutral: no method in it depends on a program, every calculation is stated so
  it can be followed by hand, and the longer ones are set out as numbered
  sequences in its calculation-sequences appendix. The scripts here exist so
  that the worked examples can be reproduced and varied, not so that they can
  be trusted. They contain no judgement, and each returns an answer whether or
  not its assumptions apply to the problem in hand.
- **Students of RSE3010, Mine Geotechnical Engineering at Monash.** Several
  scripts carry a week number in their title because they are used in a
  particular week of that unit. Those titles are deliberate and are not an
  error; a reader of the book can ignore them.

## The platform

`rse3010_geo.py` gathers the corrected calculations into one Streamlit
application, the RSE3010 Geotechnical Platform. It has one shared ground model,
so the values set on the Ground model page are the ones the intact rock, rock
mass and convergence–confinement pages use. Its presets reproduce the data sets
and worked examples of the book. The pages are:

- ground model, intact rock and rock mass strength and stiffness;
- convergence–confinement, with ground reaction curve, longitudinal
  deformation profile and support;
- settlement above single and twin bores, and the segment and ring checks of
  the lining;
- construction vibration screening;
- the RSE3010 rock slope tools: slope mass rating, kinematic analysis and limit
  equilibrium;
- an equation reference;
- a design report: the inputs, equations, results, checks and figures of
  every page opened in the session. The report is set in LaTeX, compiled to
  PDF with pdflatex, and can also be downloaded as LaTeX source with its
  figures. On Streamlit Community Cloud, `packages.txt` installs TeX Live.

The calculations are in the `geo_platform` package. Every page is tested
against the book's worked values in `tests/`.

```bash
python -m pip install -r requirements.txt
streamlit run rse3010_geo.py
python -m pytest tests -q
```

The platform is deployed on Streamlit Community Cloud from the `main` branch,
with `rse3010_geo.py` as the main file and Python 3.12. The standalone scripts
below are kept as they were used in teaching. Where the platform corrects one
of them, the platform is the version to use.

## Running a standalone script

```bash
python -m pip install -r requirements.txt
```

Every script except one is a Streamlit application:

```bash
streamlit run "CCM Tunnel.py"
```

`TBM_Vibration.py` is the single command-line script:

```bash
python TBM_Vibration.py
```

Typing `python "CCM Tunnel.py"` on a Streamlit script produces nothing useful.

## What each script is for

### Used by the book

| Script | Supports | State |
| --- | --- | --- |
| `CCM Tunnel.py` | Convergence–confinement, ground behaviour, support, high-stress ground, integrated design | The most complete application here. Known issues listed below. |
| `Mohr-Coulomb.py` | Intact rock behaviour, rock-mass parameters | |
| `Generalised Hoek-Brown.py` | Rock-mass parameters, intact rock | Only script that draws the tensile cut-off |
| `MC Parameters from HB.py` | Rock-mass parameters, in-situ stress | Overlaps heavily with the above |
| `GRC.py`, `Hoek1998.py` | Ground reaction curve | Plastic branch corrected November 2026; see below |
| `arc-beam.py` | Linings | Known issues listed below |
| `settlement_app.py`, `tbm_settlement.py` | Settlement, movement monitoring, mechanised excavation | |
| `Retaining-Wall.py` | Shafts and openings, excavation methods | Known issues listed below |
| `TBM_Vibration.py` | Movement monitoring, excavation methods | Command line, not Streamlit |

### RSE3010 teaching, not referenced by the book

`Kinematic Slope.py`, `LEM Slope.py`, `SMR Slope.py`, `smr_streamlit.py`,
`Stereonet and LEM.py`. These treat slope stability, which the book does not
cover. `SMR Slope.py` and `smr_streamlit.py` hold the only stereographic
projection code in the repository.

## Known issues

These are recorded rather than hidden, because a tool that returns a confident
wrong answer is worse than no tool.

**Corrected, November 2026.** `GRC.py` and `Hoek1998.py` scaled the plastic
branch of the ground reaction curve by `(p_cr/p_i)^((k-1)/2)`. The
zero-dilation solution scales it by `(R_p/r0)^2`, which both scripts already
had in hand. The two forms agree near the critical pressure and diverge as the
support pressure falls, so at each script's own defaults the unsupported wall
convergence came out as 206.5 m instead of 37.7 mm, and both plots capped the
displacement axis so the divergence sat off-screen. The corrected chain now
reproduces the book's own worked example to 0.1 per cent.

**Corrected, November 2026.** `CCM Tunnel.py`, five defects.
`grc_duncan_fama` dropped the `2(1-nu)` factor and the `(1-2nu)(p0-p_i)` term,
so it computed the incompressible special case rather than Duncan Fama;
`grc_salencon` raised `(R_p/a)` to `(k+1)/(k-1)`, an exponent in no published
solution, which diverged as the friction angle fell and returned displacements
of order 1e21 mm at 5 degrees. Both are now as published, and they coincide at
`nu = 0.5` as they must. `plastic_radius_mc` divided by a quantity that is
exactly zero for cohesionless ground at zero support pressure, which the
sidebar allows, taking the page down; it now returns an unbounded radius. The
longitudinal deformation profile applied one distance divisor to two formulas
needing different ones, so the app computed `exp(-0.75 X/a)` for Panet while
its own Workflow tab showed the student `exp(-1.5 X/a)`; each formula is now
in its published form. The sensitivity sweep compared the support name against
`"Concrete lining"` while it is registered as `"Shotcrete / concrete lining"`,
so the lining-thickness sweep plotted a flat line.

**Which method matches the book.** The book derives the zero-dilation form,
which is Salençon. For its own worked example (rock mass modulus 2.12 GPa at
GSI 30) the corrected Salençon method returns 126.92 mm against the book's
126.9 mm, while the corrected Duncan Fama method returns 172.12 mm for the
same inputs. The Vlachopoulos and Diederichs profile, in its published
radius-normalised form, then gives 61.95 mm released at 3 m behind the face,
against the book's 62.0 mm. Both are right for their own
assumptions, and the 36 per cent gap is the compressibility terms, not an
error. Select Salençon when reproducing the book.

**Corrected in the platform, not in the standalone scripts.** The defects
below are corrected in `rse3010_geo.py`. The standalone scripts still contain
them:
`ccm_ldp_scc.py` uses a rock-mass strength expression and a dimensionless group
that appear in no published Hoek–Brown solution, and its ground reaction curve
is discontinuous at the critical pressure; `arc-beam.py` uses the arc length as
the beam span and applies straight simply-supported statics to a curved member,
computing no hoop thrust; `tbm_settlement.py` and `TBM_Vibration.py` use a
Gaussian rather than a cumulative-normal longitudinal profile, so settlement
recovers to zero behind the face. Treat results from these scripts as
illustrative. The platform replaces them with the following:

- the convergence–confinement page;
- a curved-member segment check with hoop thrust;
- a cumulative-normal longitudinal profile in both the settlement and
  vibration pages.

## Third-party material

Two items in this repository are not the author's to license and are therefore
excluded from the licence below.

- `mi_reference.png` is a reproduction of the suggested `m_i` table of Hoek and
  Marinos (2000). It is displayed by `Generalised Hoek-Brown.py` and
  `MC Parameters from HB.py`. Consult the published source rather than relying
  on the image.
- `Retaining-Wall.py` encodes wall-selection content derived from CIRIA C760.
  That report is licensed to its purchasers and the values here should be
  checked against it.

Both should be replaced by references to their sources before this repository
is distributed more widely than a teaching cohort. Neither is used by the
platform: its typical `m_i` values are taken from the book's own appendix
table, and it has no retaining-wall page.

## Licence

See `LICENSE`. The licence covers the author's own code and documentation and
does not extend to the two third-party items named above.
