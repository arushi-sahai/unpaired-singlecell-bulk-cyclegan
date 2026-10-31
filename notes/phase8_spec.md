# Phase 8 spec: report figures, tables and results webpage (agreed)

Phase 8 turns the finished results of Phases 0–7 into the report's figures and tables and into the results webpage that the course asks for ("a results webpage showcasing qualitative results (visualizations, images/videos, figures)"). **No analysis result changes.**

**Rules for every number.**
- It comes from a saved `results/phaseN/*.csv`, or is recomputed from saved models on the training or validation splits only.
- **Test rows are never read.** The notebook loads only training and validation rows (asserted), and `results/phase7/test_touch.json` must still hold exactly its 2 entries after the run (asserted).
- Figures that need test-only data that was never saved reuse the Phase 7 figure files unchanged.

## Report figures (CVPR, at most 8 pages)
| Report | Content | Source | Split (also stated in the figure) |
|---|---|---|---|
| Fig. 1 (column) | The pairing ladder: a schematic of the three rungs (strictly unpaired M1, weakly paired M2 and B-cf, paired ridge + spread), what each uses, and the unpaired frontier as the information-free comparator | new drawing, no data | — |
| Fig. 2 (full width) | The accuracy–realism plane on test, with the no-information reference and OT-s marked | Phase 7, 7.1 (file reused unchanged) | test |
| Fig. 3 (full width, headline) | What pairs buy on test, with validation faint | Phase 7, 7.3 (file reused unchanged) | test, with validation faint |
| Fig. 4 (full width) | Mechanism: (a, b) marker-probe composition over training, M1 and M2 (validation); (c) sample-specific excess of the move from the start (validation); (d) within-donor top-1 of M1 against its start, ComBat, CORAL and its matched blend (**test**; its caption states the reversal from validation: in single-cell → bulk M1 is below its start, −0.090, 95% CI −0.162 to −0.019) | Phase 4 run logs, Phase 6 CSV, Phase 7 CSV | per panel, in its title |
| Fig. 5 (full width, qualitative) | Partner-line PCA, one panel per translator (start, M1, M2, ridge + spread, the no-information reference); single-cell → bulk in the report, bulk → single-cell in the supplement and on the webpage. Top 2 target PCs fit on training targets; real validation targets as a gray cloud; each translation drawn with a thin segment to its true partner. Cloud overlap gives a two-component view of realism (measured on 10 components); segment length shows the distance to the true partner within these two components (accuracy, centered R², is measured over all genes) | saved generators (locked, seed 0, raw), Phase 1 baselines, training-half draws | validation |

- **Fig. 5** uses every validation sample, so nothing is selected. Its caption says: validation donors, which also chose the models; qualitative only, and no number is cited from it. Test results are in Figs. 2–4.
- **Supplement and webpage only**, each keeping its split in the title:
  - the domain gap (Phase 0, 0.11, reused unchanged; all samples, descriptive, computed before any model);
  - the primary intervals (7.2, test);
  - the frontier on validation (3.1);
  - ablations (4.1) and Track A (5.1), validation;
  - what pairs buy on validation (6.1);
  - the ablation table.

## Report tables (LaTeX, booktabs)
1. **Main test results**, per direction.
   - Rows:
     - H1: margin, CI and the pre-stated reading, with the halving-1 margin and the qualifier status;
     - H2: difference, CI and reading, with the single-cell → bulk label and its validation point-estimate note;
     - H3: accuracy and composition differences with CIs;
     - B-cf (exploratory): composition and accuracy against M1 (averaged weights), with its label in both directions and "not compared with H1";
     - the no-information reference's margin.
   - Footnotes:
     - the superseded per-metric H1 ("superseded on 2026-10-07");
     - settings compared (M1 12 × 3, M2 9 × 3, exploratory 56 settings / 160 runs);
     - "95% CI, 1,000 donor resamples, uncorrected; 4 primary intervals".
   - Every reading and label string is taken verbatim from the CSVs.
2. **Models on test.** Centered R², realism, margin with its kind, marker composition, within-donor top-1 and nearest-neighbor ratio. Each row says raw or averaged weights; the paired rows' margins are marked descriptive.

## Numbers in the report
`report/numbers.tex` holds one `\newcommand` per cited value, generated from the CSVs. The report prose uses only those macros. A test checks that the macro writer and the table writer round exactly as the notebooks do (Python's f"{x:.3f}"; e.g. −0.089636 shows as −0.090).

## Webpage (`webpage/`, static, self-contained)
- **Format:** one `index.html` with local `assets/` and a `data.json` written by the notebook. No hand-typed numbers, no external scripts. It works offline and on GitHub Pages later; the author decides hosting.
- **Lead:** the result in plain words, built from the pre-stated readings. H1 is "not distinguishable from the unpaired frontier" (single-cell → bulk) and "inside the unpaired frontier" (bulk → single-cell), and H2 is "supported" (in single-cell → bulk, with its label "the generator partly fits its training readout"). The one-line gloss (fixed text): "Without pairs, the CycleGAN trades accuracy for realism and does no better than blending simple baselines; a random real profile passes the same realism test; pairs buy accuracy and composition."
- **Order:**
  - the result;
  - the pairing ladder;
  - the test plane (Fig. 2), with a short "how to read realism" box (the classifier test, 1 − 2|AUROC − 0.5|, and why a real profile scores well);
  - what pairs buy (Fig. 3);
  - the qualitative partner-line PCA with a model selector (Fig. 5 panels);
  - mechanism (Fig. 4);
  - Table 1 (HTML, from `data.json`);
  - a "development (validation)" section grouping every validation figure, including what did not work in Phases 4–5;
  - limitations (linking `results/phase7/test_touch.json`: one scoring run plus one presentation re-run);
  - links to the code and the report.
- **Accessibility:** the `dgm.plotting` palette, alt text on every figure, phone width supported.

## Notebook and code
- **`notebooks/phase8_figures.ipynb`.**
  - Builds Figs. 1, 4 and 5, copies Figs. 2 and 3, writes the LaTeX tables, `report/numbers.tex` and the webpage `data.json` and assets.
  - Recomputes only Fig. 5 and Fig. 4(a–c), from saved models and logs on validation, with the Phase 6 regeneration checks.
  - Asserts that every number written equals its source CSV value.
- **`dgm/report.py`:** formatting helpers for the LaTeX tables and the macros, with known-answer tests in `tests/test_phase8.py`.
- **`report/figure_register.md`:** updated to the final mapping.

## Process
1. Spec drafted by Claude from the author's goals and decision rules, then revised and approved by the author.
2. Code and run (minutes).
3. Results audit (figures, tables, macros and page against the CSVs; captions; split labels).
4. Polish by Claude.
5. A final check by Claude, then push.

Nothing is published by this phase.
