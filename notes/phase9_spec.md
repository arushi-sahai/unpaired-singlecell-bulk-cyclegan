# Phase 9 spec: biological interpretation (agreed)

**Status.** Post-hoc and exploratory: decided after the test results, to describe *what biology* the translations carry. All analyses use **validation** (135 pairs, 14 donors, 8–10 draws each) or are descriptive on **training** targets; no test row is read. The validation donors also chose the CycleGAN settings and checkpoints, so every CycleGAN-versus-start difference is biased in the CycleGAN's favor, and that is stated next to each one. There are no claims, no significance marks and no multiplicity correction. If the phase overruns, Part D is cut first, to program-level summaries.

**Leakage guard.**
- The guard is applied to **every pairs object the code uses**, including `ctx.pairs` from `SoundLife`. Test rows' expression is set to NaN, and the code asserts this.
- The Phase 8 `rows()` assert refuses `test`.
- The bulk obs metadata (timepoint, sex, CMV) is joined on training and validation `sample_ids` only, and this is asserted.
- No `results/phase7/*.csv` is read except `test_touch.json`, which must still hold 2 entries.

## Translators and references (validation, both directions)
Each **member** is scored, and the scores are averaged. An averaged translation is never scored.

| Translator | Members | Check against (centered R² and marker composition R², tolerance 1e-6) |
|---|---|---|
| Start (mean shift for single-cell → bulk, ComBat for bulk → single-cell) | 1 | `results/phase3/models_vs_baselines_val.csv` |
| Unpaired CycleGAN, CycleGAN + composition labels (locked) | seeds 0–2, chosen checkpoints | Phase 3 runs and registry hashes; `models_vs_baselines_val.csv` |
| Same two, with averaged weights | seeds 6–8 × steps 15k/17.5k/20k | `results/phase5/runs_val.csv` (per seed); M1 also against `results/phase6/pairs_buy_references_val.csv` (−0.2294 / −0.1881); per-class composition against `results/phase6/lineage_composition_val.csv` |
| CycleGAN + cross-fitted labels (averaged weights) | seeds 6–8 × steps 15k/17.5k/20k | `results/phase5/runs_val.csv` (row `B-cf cross-fitted labels`); `lineage_composition_val.csv` |
| Paired ridge regression (ridge + spread, all 582 training pairs) | 1 | `results/phase1/baselines_val.csv` |

- **No-information reference.** This is Phase 7's random real profile on validation: a training-half target drawn for each source. Check: every drawn index lies in the correct training half. Drawn with 200 seeds, it gives the **null band** (2.5–97.5%) for every r below.
- **Same-donor reference** (`other_draw`). Check: centered R² equals `results/phase6/same_donor_reference_val.csv` (−0.0115 and −0.1405). It is a mechanical reference, not a ceiling. Its between-donor r is about 1 by construction, and its within-donor r is about −1/(n_d − 1). It appears in tables only.

## Measures
- **Tracking r.** The Pearson r across samples between translated and true values of a gene or a program score.
  - It is reported with the slope of the true score on the translated score, because r ignores scale.
  - Each translator is also reported as **Δr against its start**, with paired ridge alongside.
  - A start is a per-gene affine map, so its per-gene r equals the raw correlation of that gene between the two assays.
- **Program score.** For each sample, the mean over the set's panel genes of each gene's expression z-scored with the **training-target** mean and SD.
  - Exception: the **granulocyte score** keeps Phase 6's definition, the plain mean of the 7 genes. It is reported for single-cell → bulk only, and checked against `lineage_composition_val.csv` (paired ridge regression 0.507).
- **Person vs visit.**
  - **Between-donor variance share:** the one-way ICC of each program's true score, computed on **all 582 training rows** of each assay. It orders programs; a share ≥ 0.5 is labeled *person-level*.
  - On validation, each translator's tracking is split into a **between-donor r**, across the 14 donor means (bootstrap and Fisher-z intervals), and a **within-donor r**, after subtracting each donor's mean from the translated and the true score separately.
  - The **recovered share** of a component is r² within that component. A component is shown as *not estimable* when it holds less than 10% of the true score's validation variance.
- **Composition (bulk → single-cell).** A **partial r**. Regress the true target score on the true PBMC composition in the **training** rows. Apply those coefficients to the translated and to the true validation score, then correlate the residuals.
  - In single-cell → bulk, the neutrophil fraction is unmeasured, so composition-dependent programs (granulocyte, erythroid, inflammatory) are labeled, not adjusted.
- **Uncertainty.** Paired donor-bootstrap 95% CIs (1,000 resamples of validation donors) for the short list and its Δr. Part C stratifies the resamples by sex or CMV status. The 50 Hallmark sets get point estimates and the null band only.
- **Decision criteria, fixed now.**
  - **"Tracked":** the 95% CI's lower bound lies above the upper end of the null band.
  - **"Inherited from the alignment":** the CI of Δr against the start includes 0 and the start itself is "tracked"; when the start is not tracked, the wording is "no different from the start".
  - **Part B "present":** the validation true change has a donor-bootstrap CI that excludes 0.
  - **Part B gate:** below.

## Part A. Programs and genes
- **Short list, fixed now.** Only these programs get figures and CIs:
  - Granulocyte (single-cell → bulk only).
  - Erythroid: ALAS2, HBB, HBA1, HBA2, CA1, SLC4A1, HBD, AHSP.
  - Platelet: PPBP, PF4.
  - Interferon response: IFI27, IFI44L, IFI44, ISG15, RSAD2, MX1, SIGLEC1, OAS1, OAS3, IFIT1, IFIT3, HERC5, LY6E.
  - Plasmablast and immunoglobulin: JCHAIN, MZB1, TNFRSF17, TXNDC5, IGHG1–4, IGHA1, IGKC, IGLC2, XBP1, PRDM1, CD38.
  - The T, B, NK, monocyte and dendritic-cell marker programs. These are the marker-probe genes, so they overlap Phase 6.3.
  - The CMV-associated cytotoxic program: KLRC2, GZMH, B3GAT1, GZMB, PRF1, GNLY, FCGR3A.
  - Identity: XIST, HLA-DRB5, HLA-DQA1, HLA-DQB1, HLA-DQA2.
  - Ribosomal: the 34 RPL and RPS genes, as an assay-artefact control.
- **MSigDB Hallmark.** Pinned release `h.all.v2024.1.Hs`, with its URL and SHA-256 recorded (CC BY 4.0; Liberzon et al. 2015), stored in `data/raw/msigdb/`.
  - Sets with ≥ 10 panel genes go in a **supplementary table only**, as "N of 50 sets", with each set's coverage and the symbols that don't match the panel.
  - The project sets are exempt from the size rule.
  - The Hallmark sets overlap, so their r values are not independent.
- **Genes.** Supplement only: the distribution of per-gene r by gene class against the per-gene null, labeled "not validated". No named top or bottom genes, consistent with Phase 6.

## Part B. The vaccine response
- **Unit.** The **donor-year** with both Day 0 and Day 7 of **Flu Year 1 or 2**. That gives 116 units in training and 26 in validation, from 14 donors.
  - "Immune Variation" visits are excluded: they are a separate study arm with no Day 7 plasmablast rise in training (−0.27 bulk, −0.13 single-cell), so they are not used as a vaccine response.
  - Every interval resamples **donors**, keeping all of a donor's years together.
- **Gate, per target assay.** The Day 7 − Day 0 change in the plasmablast program in **training** targets must have a donor-bootstrap 95% CI with its lower bound above 0. It is judged separately in bulk and in single-cell, and each direction uses its own target assay. If the gate fails, that direction reports "no detectable response at this resolution."
- **Programs.** Plasmablast is the primary program.
  - Interferon is secondary, with an expected null: the interferon response to inactivated vaccine peaks around Day 1, and in training it falls by Day 7 (bulk −0.18 / −0.08).
- **On validation.**
  - The true change, with its CI; it is "present" or not under the rule above.
  - Each translator's change: its mean, its CI, its r with the true change across donor-years, and its Δ against the start.
  - The start reproduces the input's change exactly (mean shift) or as a fixed per-gene rescaling (ComBat).
  - If the true change is not present on validation, the translators are reported as "cannot be judged," not as "not tracked."

## Part C. Traits of the person (one table, no figure)
- **Sex.** The female − male difference in translated XIST, in training-target SD units, against the true difference. The donor-level AUROC is secondary, since it saturates. Validation has 9 female and 5 male donors.
- **CMV.** The CMV-positive − CMV-negative difference in the CMV-associated program, true against translated. Validation has 8 positive and 6 negative donors.
- **HLA.** Between-donor r of the HLA genes, which serve as genotype markers.
- Bootstrap resamples are stratified by the trait. These are identity checks: keeping them means the translator preserves who the person is, not that it learned biology.

## Part D. What adversarial training changes
- **Definitions.**
  - The *move* is the CycleGAN's output minus its start's.
  - The *needed correction* is the true partner minus the start.
  - Both contain the start, so an information-free shrinkage of the start toward the training mean already gives a positive r between move and needed correction (about 0.24 per gene in single-cell → bulk).
- **What is reported.** Per program:
  - the mean absolute move;
  - the **excess r**: a translator's r between move and needed correction, minus the same r with another donor's real profile in place of the partner.
  - The **shrinkage reference** (the start moved 30% toward the training-half targets' mean, the same mean the frontier uses) is the comparator. Its excess is negative on real data (median per gene about −0.14 in single-cell → bulk and −0.11 in bulk → single-cell), because the start comes from the same draw as the partner. Each translator is reported as its excess minus the shrinkage reference's.
- This is the per-gene, across-sample analogue of Phase 6.2's move decomposition (`results/phase6/move_decomposition_ci_val.csv`, Phase 8 Fig. 4(c)), which was a per-sample cosine across genes. It is broken down by program and compared with paired ridge's move, on validation.

## Outputs and figures
- **Tables:** `results/phase9/` holds CSV tables for programs, Δr, the between-donor shares, the vaccine contrasts, the traits table, the moves and the Hallmark table. Per-gene values go in one compressed file.
- **Figures:**
  1. **Person vs visit.** The short-list programs, ordered by between-donor share in training. Each translator's recovered between-donor and within-donor share, with the null band; non-estimable components are marked.
  2. **Δr against the start, with paired ridge.** The short list, with CIs. Bulk → single-cell may move to the supplement.
  3. **Signal the input lacks (single-cell → bulk).** Translated against true granulocyte and erythroid scores, colored by donor.
  4. **Vaccine response.** Day 7 − Day 0 plasmablast change, true against translated, per donor-year, with the two years of each donor marked. The training response is shown alongside, and interferon in a secondary panel.

## Wording rules (README, page, report)
- Say "tracks the person's level of X" or "inherited from the alignment". Never "learns X" or "recovers biology".
- Every CycleGAN Δ carries "validation; post-hoc" and the selection bias.
- Cite only short-list results or summaries across sets, each with its interval and the null band, and give the count of intervals shown.
- Sentence forms are fixed now:
  - "Translations track [program] at r = … (95% CI …; null …), mostly between donors (share …)".
  - "The Day 7 plasmablast response is [present / not present] in the true validation profiles and [tracked / not tracked / cannot be judged] by …".

## Code and checks
- `dgm/biology.py` and `notebooks/phase9_biology.ipynb` (minutes on CPU).
- **Known-answer tests** (`tests/test_phase9.py`):
  - donor signal plus independent visit noise gives within-donor r ≈ 0 and between-donor r ≈ 1;
  - a visit signal centered within donors, plus donor noise, gives between-donor r ≈ 0;
  - a constant translation gives NaN;
  - shuffled rows give r ≈ 0;
  - r is invariant to per-gene centering;
  - z-score parameters come from training rows only;
  - a duplicated donor counts twice in a resample;
  - a composition-only signal is removed by the partial r;
  - with a synthetic start independent of the partner, pure shrinkage gives a Part D excess of 0; when start and partner are correlated, its excess is below 0;
  - donor-year bootstrap resamples keep a donor's years together;
  - the GMT parser handles duplicates, panel restriction and the size threshold;
  - the row guard refuses test rows, and the masked `ctx.pairs` holds NaN for test rows.
- **Process.**
  - Steps: spec drafted by Claude from the author's goals, then revised and approved by the author, code, code audit by Claude and code review by the author, run, results audit by Claude, polish and a final check by Claude, push.
  - The project plan gains Phase 9 (biological interpretation), and the talk and the report become Phases 10 and 11.
  - The phase is recorded as deviation #24.

## Not in scope
- Test-split analyses.
- New training.
- Enrichment tests.
- CBC data.
- Breakdowns by age group: age explains at most 2.9% of the top principal components, and there are too few donors.

## Amendment (2026-10-23, after the code audit; validation only, test untouched)
**Part D's headline measure changes.** The agreed "excess minus the shrinkage reference's excess" is not a valid null correction. The reference's per-gene excess approaches −1 for genes whose start already correlates with the target (−0.60 to −1.24 for the interferon genes in single-cell → bulk, where r(start, target) is 0.54–0.94), and correlations are bounded, so the subtraction inflated every translator (for example, +1.43 for the CycleGAN on interferon against an own excess of 0.34).
- **New headline:** the per-gene **partial r of the move with the true partner, controlling for the start**. Both are regressed on the start across validation samples, per gene, and the residuals correlated. Any move that is an affine function of the start, including the shrinkage reference, has a zero residual, so this is null by construction (asserted in the run, with a known-answer test).
- **Context only:** the move excess and the shrinkage reference's excess are reported as context, with no subtraction.

**Other changes:**
- The per-gene null is pooled over seeds, not averaged.
- The Hallmark null uses 200 seeds.
- "Inherited from the alignment" applies only to the CycleGANs; paired ridge reads "no different from the start".
- Part C adds the translated / true XIST difference.
- Paired ridge regression is also checked on marker composition R².
- Part D covers the locked models and paired ridge regression, summarized by the median over each program's genes.
