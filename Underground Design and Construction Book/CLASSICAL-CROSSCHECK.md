# Crosscheck against Hoek & Brown (1980) and Brady & Brown (2006)

Source: `../Textbooks/classical_textbooks_toc_topics_overview.tex`.

That document maps the two classics against the **10-chapter** version of this book. The book has since grown to 39 chapters and 8 appendices, which closes most of its gaps and changes which ones remain. This file re-runs the comparison against the current structure and records what was done about each finding.

## 1. Hoek & Brown (1980), *Underground Excavations in Rock*

| H&B chapter | Where it lands now | Status |
| --- | --- | --- |
| 1. Planning considerations | Infrastructure and Geotechnical Challenges; Risk, Contracts and Delivery | Covered |
| 2. Classification of rock masses | Rock Mass Classification | Covered. Terzaghi rock load survives only as a passing mention in the construction chapter — worth a short historical section |
| 3. Geological data collection | Site Investigations; **Engineering Geology** | Covered |
| 4. Graphical presentation of geological data | **Ch. Stereographic Projection and Structural Analysis**, with net constructions in **Appendix E** | **Added as a full chapter.** Initially placed in an appendix; corrected after review — see [note](#note-on-stereographic-projection) |
| 5. Stresses around underground excavations | Stress and Intact Rock; **Continuum Mechanics**; CCM; **Appendix F** | Covered |
| 5. …multiple excavations, pillars, stress shadows | **Multiple Excavations, Pillars and Stress Interaction** | **Added** — was a complete gap |
| 6. Strength of rock and rock masses | Stress and Intact Rock; Rock Mass Design Parameters | Covered |
| 7. Underground excavation failure mechanisms | Failure Mechanisms and Empirical Design; **Excavation Design by Ground Type** | **Strengthened** — ground-type organisation added |
| 8. Underground excavation support design | **Rock–Support Interaction and Support System Design** | **Added** — this was your flagged high-priority gap |
| 9. Rockbolts, shotcrete and mesh | Rock Reinforcement; Shotcrete and SCL | Covered |
| 10. Blasting in underground excavations | Drill and Blast Engineering | Covered |
| 11. Instrumentation | Monitoring and the Observational Method | Covered |
| 12. Appendices | **Appendices F, G, H** | **Added** |

## 2. Brady & Brown (2006), *Rock Mechanics for Underground Mining*

| B&B chapter | Where it lands now | Status |
| --- | --- | --- |
| 1. Rock mechanics and mining engineering | Infrastructure and Geotechnical Challenges | Covered |
| 2. Stress and infinitesimal strain | **Stress, Strain and Constitutive Relations** | **Added** — the book previously used Kirsch, elasto-plasticity and FEM without ever defining strain or constitutive relations |
| 3. Rock mass structure and characterisation | Engineering Geology; **Stereographic Projection and Structural Analysis**; Rock Mass Classification | Covered |
| 4. Rock strength and deformability | Stress and Intact Rock; Rock Mass Design Parameters | Covered |
| 5. Pre-mining state of stress | In-situ Stress: Estimation and Measurement | Covered |
| 6. Methods of stress analysis | Numerical Modelling; Advanced Numerical Methods | **Strengthened** — boundary element and linked/hybrid schemes added; these were missing entirely |
| 7. Excavation design in massive elastic rock | **Excavation Design by Ground Type** §§3–8 | **Added** |
| 8. Excavation design in stratified rock | **Excavation Design by Ground Type** §§9–12 | **Added** — voussoir beam, roof separation, large deflection |
| 9. Excavation design in blocky rock | **Stereographic Projection and Structural Analysis** (identification) + **Excavation Design by Ground Type** (stability) | **Added** — split so geometry and mechanics are distinct steps |
| 10. Energy, stability, seismicity, rockbursts | Squeezing, Swelling and High-Stress Ground | **Strengthened** — energy change and energy release rate sections added |
| 11. Rock support and reinforcement | **Rock–Support Interaction**; Rock Reinforcement | **Added / covered** |
| 12–16. Mining methods, backfill, longwall, caving | Not adopted | Deliberately out of scope, per your guidance to use mining content selectively |
| 17. Mining-induced surface subsidence | Settlement and Building Damage | **Partially adopted** — chimney caving, sinkholes and legacy workings added as a civil problem; tabular-extraction subsidence not adopted |
| 18. Blasting mechanics | Drill and Blast Engineering | Covered |
| 19. Monitoring rock mass performance | Monitoring and the Observational Method | Covered |
| Appendices | **Appendices F, H** | **Added** |

## 3. Your seven recommendations

| Recommendation | Action |
| --- | --- |
| Make the design workflow explicit at the start of every part | **Not yet done.** Needs a part-opening page or figure. Flagged as the top remaining item below |
| Separate ground model, material model and construction model | **Done structurally.** Ground model → Engineering Geology §1; material model → Rock Mass Design Parameters and Continuum Mechanics; construction model → Part IV |
| Expand support design beyond chart reading | **Done.** New chapter treats support as stiffness, capacity and timing, with a worked verification example |
| Strengthen construction-induced damage | **Done.** Drill and Blast covers overbreak and the damage zone, and ties it explicitly to the disturbance factor *D*; support installation quality has its own section |
| Add a monitoring feedback loop | **Done.** Monitoring chapter covers trigger levels, contingency, back-analysis and design updating, plus a section on how monitoring programmes fail |
| Use mining-specific content selectively | **Done.** Pillars, energy release and chimney caving adopted and reframed for civil work; stoping, backfill, longwall and caving not adopted |
| Keep removed topics outside the main narrative | **Partly.** *Slope* kinematic analysis (planar, wedge and toppling tests on a slope face) stays out. Stereographic projection itself is a main-narrative chapter — see the note below |

### Note on stereographic projection

Recommendation 7 of the source document reads: *"Orientation-analysis content may be held as optional background or appendix material only if needed later."* That was first applied to stereographic projection as a whole, which was wrong.

Projection is a primary underground design tool, not slope-specific background. Structurally controlled instability is one of the three main underground failure modes; crown and sidewall wedges, blocks at junctions and cross-passages, portal and shaft-collar blocks are all identified by the same construction. Brady & Brown's blocky-rock chapter is built on it, Hoek & Brown give it a full chapter plus a wedge-analysis appendix, and the UnWedge assessment of the Snowy 2.0 power station in your own Week 12 material is exactly this analysis.

It is now a chapter in Part I, positioned after Site Investigations. What remains excluded is the *slope* kinematic content — the planar, wedge and toppling feasibility tests against a slope face — which belongs with the companion slope material.

One consequence to note: `chapters/ch09-integrated-design.tex` still carries the line *"Related orientation-analysis material is intentionally left out of this edition to keep the book aligned with the current chapter scope."* That sentence now contradicts the structure and should be reworded or deleted.

## 4. What was added

**Five chapters**

- *Stereographic Projection and Structural Analysis of Underground Excavations* (Part I) — projection, block removability, roof and sidewall wedges, excavation orientation as a design variable
- *Stress, Strain and Constitutive Relations* (Part II) — the continuum foundation the later solutions assume
- *Excavation Design by Ground Type* (Part III) — massive, stratified and blocky rock, the organising idea the book most lacked
- *Multiple Excavations, Pillars and Stress Interaction* (Part III) — interaction, tributary area, pillar strength, twin-tunnel pillars
- *Rock–Support Interaction and Support System Design* (opens Part V) — support as stiffness, capacity and timing

**Four appendices**

- E — Stereographic Net Constructions (step-by-step procedures supporting the chapter)
- F — Closed-Form Solutions and Design Charts
- G — Typical Material Constants and Parameter Ranges
- H — Worked Calculation Sequences

**Sections added to existing chapters**

- Advanced Numerical Methods: the numerical method family, boundary element methods, linked/hybrid schemes
- Squeezing and High-Stress Ground: energy changes on excavation, energy release rate
- Settlement and Building Damage: subsidence over voids and legacy workings, investigating and treating legacy voids

## 5. Remaining gaps

1. **Part-opening workflow figures.** Your highest-priority recommendation, and the one thing here that needs drawing rather than writing. Seven part openers, each showing where that part sits in the investigation → classification → parameters → analysis → support → construction → monitoring chain.
2. **Terzaghi rock load.** Present only as a mention. A short historical section in the classification chapter would complete the lineage from rock load through RMR and Q to GSI.
3. **Anisotropic rock strength.** Both classics treat it explicitly. Currently only a constitutive-relations section; the strength side (bedded and foliated rock, strength as a function of bedding angle) is not covered.
4. **Excavation influence on existing openings.** Covered for new-tunnel-near-existing-tunnel, but not the reverse case of an existing opening disturbing a new one.
5. **Trial excavations.** Hoek & Brown treat test adits and trial excavations as a design tool. Not covered anywhere.
