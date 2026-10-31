# Phase 7 spec: the test evaluation (agreed)

Phase 7 scores the locked models, the exploratory models declared in Phase 6 and every baseline **once** on the test split: 119 pairs from 14 donors, untouched so far. The hypotheses, metrics, comparators and wordings are those fixed in `notes/phase3_plan.md` (with its amendments) and in `notes/phase6_spec.md`, amendment A1/A2, and written out in Phase 6, Implications. Nothing is selected, tuned or refit on test.

## Before test is touched
- **Everything is frozen:** generators (by hash), evaluators (Phase 1, fit on training targets), composition networks and marker probes (Phase 2), the frontier members (fit on the training halves), the paired baselines (fit on all training pairs), the OT-s map (ε = 0.3 on the halving-0 halves), the analysis code and the bootstrap seed.
- **Dry run on validation.** The whole notebook runs with validation standing in for test (flag `PHASE7_DRY=1`, which never loads a test row) to check the code, its outputs and the run time. The validation outputs are compared with the Phase 3–6 values they should reproduce:
  - the locked models' margins and composition;
  - the B-cf and M1 averaged-weight readouts;
  - the frontier.
- **Code audit** of the code and the dry run, then the real run.
- **The test run is the only run that loads test rows.**
  - A guard asserts that test indices appear only when `PHASE7_DRY` is unset.
  - No fit, selection or threshold call receives a test index; this covers the evaluators, the frontier members, the paired baselines and OT-s.
  - The test run's code hash must equal the code-audited commit; a mismatch aborts before any test row is loaded.
  - `results/phase7/test_touch.json` is append-only: one entry per run that loads test rows (time, commit, code hash, flags).
  - Re-running reproduces the same numbers (deterministic) and is allowed for presentation only. Any change to the analysis after the test run is logged as a deviation and reported.

## Models on test
| Group | Models | Readout |
|---|---|---|
| Hypotheses | Locked M1, M2 (halving 0) | Chosen checkpoints, raw weights, seeds 0–2 |
| Split sensitivity | Locked M1, M2 (halving 1) | The same, halving 1 |
| Exploratory (A1) | B-cf; M1 and M2 as comparators | Averaged weights, seeds 6–8, steps 15,000/17,500/20,000 |
| Exploratory (A2) | OT-s, ε = 0.3, halving 0 | The same 3 draw seeds as Phase 6 (0–2) |
| Pair budget (descriptive) | Weakly paired GAN with labels from k = 4, 8, 16 donors and the k = 33 cross-fit; ridge at k = 8, 16, 33 and ridge + spread at k = 16, 33 | Averaged weights, seeds 6–8, kept steps; regressions on Phase 5's donor sets per seed. Nothing is chosen. These settings are scored on test only, so they are counted separately from the 56 settings compared on validation |
| No-information reference | One real training-half target drawn uniformly per source | 3 draws (seeds 0–2) |
| Unpaired baselines | Mean shift, ComBat, CORAL, training-target mean (halving 0 and 1) | Single fits |
| Paired reference | Ridge, ridge + spread, ridge + PC calibration, paired MLP (seeds 0–2) | Fit on all 582 training pairs (Phase 1) |
| References | Another draw of the same donor; real targets read by the probes | — |

Each generator's canonical hash is checked against its registry (Phase 3) or its run's provenance (Phase 5, as in Phase 6) before scoring.

## Primary analyses

Per direction, 1,000 donor resamples of the 14 test donors (seed 0). Every metric is recomputed on each resample, and a model's members are averaged inside each resample. A resampled donor's copies keep their original donor ID, so they fall in the same fold of the grouped classifier test and of every other donor-grouped metric. Realism is bootstrapped for the frontier members, M1 (H1), M2 and ridge + spread (H3); it is a point estimate for every other model.
- **H1: M1's margin beyond the unpaired frontier.**
  - On each resample, all blend-path points of the halving-0 members (mean shift, ComBat, CORAL, training mean; weights 0, 0.1, …, 1, interpolated) are re-scored for centered R² and realism, giving that resample's frontier.
  - Each M1 seed's margin is computed against that frontier, and the three margins are averaged in their own units whatever their kind, as in Phase 3. The H1 decision uses this average; on the full test set, each seed's kind is reported.
  - **Reported:**
    - the margin with its percentile CI;
    - M1's accuracy and realism, and the frontier value at M1's accuracy;
    - the share of resamples in which all three seeds' margins are of the realism kind, and in which any is of the accuracy kind;
    - the realism-margin CI over the all-realism-kind resamples.
  - **Wording:** CI above 0, "supported on halving 0"; CI including 0, "not distinguishable from the unpaired frontier"; CI below 0, "inside the unpaired frontier".
  - **Qualifier:** "not robust to the training split" when H1 is supported but the halving-1 test margin's point estimate is at or below 0. That margin is computed against the halving-1 frontier on the full test set, without a bootstrap.
  - **Next to H1 (Phase 6 implication):** M1's within-donor top-1 minus that of its start, of ComBat, of CORAL and of its matched blend, with CIs. The blend is matched per M1 seed on the full test set by the Phase 6 bisection, against the halving-0 test frontier; the matched weights are then held fixed across resamples, as in Phase 6. Also the same-donor reference's and the no-information reference's margins.
  - **Settings compared:** reported with H1 and H2. M1 came from 12 settings × 3 seeds and M2 from 9 settings × 3 seeds (Phase 6, Table 6.20).
- **The superseded per-metric H1, one line, labeled "superseded on 2026-10-07".**
  - It reports M1 minus the fixed comparators, separately on centered R² and on realism, with CIs. It "would have been supported" only if both CIs lie above 0.
  - The comparators are those of the version the frontier amendment replaced (`notes/phase3_plan.md` as of commit b85a79e to e9daa5c): single-cell → bulk, mean shift (accuracy) and CORAL (realism); bulk → single-cell, ComBat (accuracy) and mean shift (realism).
  - An earlier draft (b81c90e, the single baseline with the best validation selection score) had already been replaced on the same morning.
  - The fixed comparators are used even if another baseline scores best on test.
  - Partial results use the pre-stated words of that version: "better [realism / accuracy] than any unpaired baseline; [accuracy / realism] not distinguishable", or "worse" if that CI lies below 0.
- **H2: M2 minus M1 in marker-probe composition R²** (locked, seeds averaged inside resamples).
  - Wording: "supported" if the CI is above 0, "not supported" otherwise.
  - In single-cell → bulk, the Phase 6 label "the generator partly fits its training readout" is attached. The ratio behind it is a validation point estimate, and the locked comparison's CI includes the threshold.
  - In both directions, the statement that M2's composition networks were trained on pairs that include the training donors' own.
- **Multiplicity:** four primary intervals (H1 and H2 × 2 directions), per direction and uncorrected.
- **H3 (no claim):** ridge + spread minus M1, and minus M2, in centered R², realism and marker-probe composition R², with CIs.

## Secondary and exploratory (reported, no claims)
- **All locked models and baselines.** Every evaluator metric, as in Phase 3, including:
  - within-donor top-1;
  - composition R² under the ridge probe and under $C_B$/$C_S$ (M2's gain under $C$ that the marker probe does not confirm is labeled suspect, per Phase 3);
  - granulocyte gap and correlation;
  - diversity, nearest-neighbor ratio, MMD and the selection score.
- **Shrinkage:** validation minus test for the margins, centered R² and composition of M1 and M2, both locked (selected checkpoints) and with averaged weights (no checkpoint selection). The difference between the two estimates the optimism from checkpoint selection, confounded with weight averaging and the seed sets, and apart from the shift between donor sets.
- **Pair budget on test (descriptive):** the Phase 6 "what pairs buy" readouts on test for the grid above.
- **Halving 1:** point estimates of M1 and M2 against the halving-1 frontier.
- **B-cf (A1).**
  - *Primary:* paired CIs of Δ marker composition R² and Δ centered R² against M1 (averaged weights).
  - *Secondary:*
    - the same against M2 (averaged weights);
    - realism and within-donor top-1;
    - the frontier margin, as a point estimate (descriptive);
    - the differences against the locked models, labeled as mixing raw and averaged weights.
  - *Labels:* exploratory, the §4.2 label in both directions, 56 settings and 160 runs compared on validation, no halving-1 check, no multiplicity correction, not compared with H1.
- **OT-s and the no-information reference:** point estimates of margin, centered R², realism, composition and within-donor top-1, read as retrieval.
- **Exploitation ratios on test** may be shown descriptively. The labels are validation decisions and stay fixed: a test ratio cannot add or remove a label.
- **Same-donor reference:** its test margin and centered R², as the realism context for H1.

## Outputs
- `notebooks/phase7_test_evaluation.ipynb` and `results/phase7/*.csv`.
- **Figures:**
  - (7.1) the test accuracy–realism plane with the test frontier. Locked and averaged-weight M1 and M2 and B-cf are drawn as arrows from their validation to their test positions. OT-s, the no-information reference and the same-donor reference are marked separately.
  - (7.2) the primary intervals and H3 differences, one panel per metric (margin; composition; centered R²; realism), with B-cf's primary intervals in a separate panel titled exploratory.
  - (7.3) "what pairs buy" on test, in Phase 6's layout, with the validation values faint behind it.
- **Text:** the readings follow the pre-written rules in Phase 6, Implications, plus the following. No new reading is added after seeing the results; anything unexpected is described as such.
  - *OT-s or the no-information reference beyond the frontier:* read as the realism measure rewarding real profiles whatever information they carry, not as a non-adversarial method passing the frontier. If the no-information reference passes, the H1 text says so next to M1's margin.
  - *Halving-1 qualifier:* the rule is "at or below 0", applied exactly. A halving-1 margin between 0 and 0.05 gets one sentence noting that seed SDs of point estimates in this range are about 0.03–0.04 (validation, Phase 6 Table 6.21).
  - *Secondary metrics the text may cite:* within-donor top-1, composition under the ridge probe and under C, the granulocyte correlation, and the nearest-neighbor ratio. Every other secondary metric is reported in a supplementary table only.

## Code and tests
- **New `dgm/testeval.py`:**
  - the resample-level frontier and margin, reusing `dgm.frontier.trace` and `frontier_margin` on index subsets;
  - the seed-averaged statistics and the paired-difference bootstrap;
  - parallel resamples over worker processes, each pinned to 4 threads and given deterministic chunks of the resample list, so results don't depend on the number of workers.
- **Tests:**
  - on a resample equal to the full set, the bootstrap margin equals the direct margin;
  - with workers 1 and 6 the CI is identical;
  - the frontier on a resample equals `trace` on the subset;
  - a synthetic model placed on the frontier gets a margin of about 0;
  - the guard refuses test rows when `PHASE7_DRY=1`;
  - on a resample with a duplicated donor, no donor ID appears in both a training and a held-out classifier-test fold.
- **Compute:** about 65 classifier tests per resample and direction (58 frontier points, M1, M2 and ridge + spread), at about 0.3 s each, gives about 11 CPU-hours: about 2 hours with 6 workers, to be measured in the dry run. Everything else takes minutes. If the dry run's projected time exceeds the overnight window, the run is postponed, not cut.

## Process
1. Spec drafted by Claude from the author's goals and decision rules, then revised and approved by the author.
2. Code.
3. Dry run on validation.
4. Code audit by Claude and code review by the author.
5. The test run, tonight, on the laptop under `systemd-inhibit` with the lid-switch inhibitor.
6. Results audit by Claude.
7. Polish and a final check by Claude, then push.
