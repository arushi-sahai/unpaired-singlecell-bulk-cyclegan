# Phase 5 spec: exploratory improvements (final)

Phase 5 tries to improve on the locked models, using what Phase 4 found. It is **exploratory**: settings are chosen on the same 14 validation donors that chose the locked models and informed Phase 4.

**Link to test.**
- The Phase 3 hypotheses on test apply only to the locked M1 and M2.
- At most one model per track may go to test. Its exact configuration and seeds are fixed before Phase 7.
- That model is reported as exploratory, with the total number of configurations compared on validation across Phases 3–5 and the statement that its design used Phases 3–4 validation results.
- A Track B model is reported as weakly paired, against M1 and M2, never against H1.

## Default and reference
- **Weight averaging on.** Every run keeps an exponential moving average of the generator weights (decay 0.999). The averaged weights are the primary readout.
- **References:** the locked M1 and M2, both with weight averaging, on seeds 6–8. M1 also runs on seeds 9–11 for the confirmation. Seeds 0–5 informed the design.
- **Readouts** (validation, 135 pairs, 14 donors), as in Phase 4:
  - *Primary:* last-quarter means (steps 15,000–20,000) of the averaged weights: margin beyond the frontier, centered R², realism, marker-probe composition R².
  - *Also:* the sample-specific component of the move, i.e. the cosine with the needed correction minus the cosine with another donor's profile, at the kept checkpoints.
  - *Secondary:* the selected checkpoint (Phase 3 rule) and the within-donor top-1 rate.
- **Uncertainty:** seed SDs; donor-bootstrap CIs for centered R² and marker composition at the kept checkpoints.

## Track A: the strictly unpaired model (can it get beyond the frontier?)

Phase 4 found that the discriminator buys realism, but the generators' moves are mostly orthogonal to the needed correction, and that the identity loss costs single-cell → bulk accuracy and composition. Track A changes the losses and the generator's capacity to change the direction of the move, not its length.

**Zero-run pre-check for A-dist.** On the validation pairs, compute the correlation between the input's pairwise distances (after the start map) and the true partners' pairwise distances, and the same correlation for M1's outputs (Phase 3 and 4 checkpoints). A-dist runs at two weights only if the correlation for the start is at least 0.3 and exceeds M1's by at least 0.05; otherwise at one weight (thresholds fixed before computing it).

**Weight calibration (before any Phase 5 run).** New loss weights are set from the Phase 4 full-model logs, so that each term is about 0.1× and 1× the adversarial term at M1's plateau.

| # | Change | Rationale |
|---|---|---|
| A-idoff | **Identity loss off** (both directions) | The control: Phase 4 showed this helps single-cell → bulk. The anchored loss counts only if it beats this arm |
| A-idS | **Identity loss off for single-cell → bulk only** | Phase 4: identity off helps single-cell → bulk and hurts bulk → single-cell composition |
| A-anc | **Start-anchored identity loss** at the locked weight 10: the target becomes the start's output on the other domain's input, ‖G_{S→B}(x_B) − s_{S→B}(x_B)‖₁ + ‖G_{B→S}(x_S) − s_{B→S}(x_S)‖₁ | Removes the conflict with the start. It constrains the generators only on the other domain's inputs, so it may act much like A-idoff |
| A-dist | **Distance preservation** (DistanceGAN; Benaim & Wolf 2017): for every pair in a batch, the pairwise distance between translations should match the pairwise distance between the start's outputs, each set standardized by its own batch mean and SD (64 × 63 / 2 pairs per batch). L1 over pairs; both directions; 2 calibrated weights | Each side is standardized by its own batch statistics, so only relative geometry is constrained and the overall spread is free (it does not pin realism to the start's under-dispersion) |
| A-bn | **Bottleneck residual:** f routed through an 8-dimensional layer (5000 → 512 → 8 → 5000) | Between affine-only (unrealistic) and the full residual (spread orthogonal to the truth): forces the added variation into a few dimensions driven by the input |
| A-both | The best loss change combined with A-bn, run only if both beat A-idoff on their own | Whether they add up |

Runs: 3 each for A-idoff, A-idS, A-anc and A-bn, 6 for A-dist (3 if the pre-check says one weight), and 0 or 3 for A-both: 18 to 21.

**Confirmation.** The best Track A setting by the primary readout is re-run on seeds 9–11, alongside M1 on seeds 9–11. It is called **promising** only if, on the confirmation seeds:
- the mean over directions of its per-seed paired difference from M1 in last-quarter margin exceeds 2 × the SD of that difference;
- no direction is worse than M1 by more than 1 SD;
- its last-quarter margin is above 0 in at least one direction.

The SD of a difference over 3 seeds is itself noisy; this is stated with the result.

## Track B: the weakly paired model (how much honest pair information does composition supervision need?)

The single-cell composition network $C_S$ needs no pairs, so it is held fixed at Phase 4's half-A $C_S$ in every Track B arm. Only the bulk network $C_B$ varies.

| # | Change | Question |
|---|---|---|
| B-cf | **Cross-fitted $C_B$.** The 68 training donors are split into 5 donor-grouped, age-stratified folds, and $C_B$ is trained 5 times, each time leaving one fold out. Every training row, in both task terms, is read by the fold network that never saw its donor | What do honest labels buy at about 80% of the full training size? |
| B-k | **Label budget.** $C_B$ trained on the bulk pairs of k half-A donors, nested (k = 4 ⊂ 8 ⊂ 16) inside one fixed, age-stratified 16-donor pool drawn per seed (donor lists saved). The single-cell → bulk task term is applied only to the other 17 half-A donors, at every k. A separate point, k = 33 (all of half A, $C_B$ cross-fitted within half A for that term), is reported but not on the same footing. The bulk → single-cell term reads half B, which is out of sample for every k | How many paired donors do the labels need, per direction |

- **Why the single-cell → bulk term is restricted.** A $C_B$ trained on a donor's bulk pairs has effectively memorized them, so applying the term to that donor's single-cell rows would pull the generator toward each sample's true bulk partner. The restriction keeps the pair information acting only through labels. The same path existed in Phase 4's B3 and in the locked M2, whose $C_B$ was trained on all 582 training pairs, half A included: in the locked M2, pair information entered through both task terms. That is recorded in the Phase 3 and Phase 4 logs and texts (it changes how H2 is read, not the test). With the nested design the single-cell → bulk term trains on the same 17 donors' rows at k = 4, 8 and 16 (all of half A at k = 33, through cross-fitting); the row counts are reported next to each k.
- **Cross-fitting in B-cf** keeps each row's own label out of sample. The shared generator still learns from other rows' labels, whose networks saw that donor; this is standard.
- **Per-direction mechanisms.** In single-cell → bulk, k changes the readout's quality (the targets are measured compositions). In bulk → single-cell, it changes the pseudo-labels' quality. So "paired donors needed" is reported per direction.
- **Comparator at no GAN cost:** ridge + spread trained on the same k donors' pairs, scored on validation. This tests whether weak supervision is a better use of k pairs than a paired translator.

Runs: B-cf 3 and B-k 12, so 15 (one donor draw per seed; the seed SD then includes the donor draw, as in A4).

## Runs, code and tests
- **Runs:**

  | Item | Runs |
  |---|---|
  | References (M1, M2 on seeds 6–8) | 6 |
  | Track A | 18–21 |
  | Confirmation (candidate and M1 on seeds 9–11) | 6 |
  | Track B | 15 |
  | Equivalence | 3 |
  | **Total** | **48–51 (about 12 hours)** |

- **`dgm/translation.py`:**
  - `identity_target` ("input" by default, or "start");
  - per-direction identity switches;
  - `lambda_dist` with the two standardizers;
  - `bottleneck` (default None);
  - `task_nets` gains "crossfit" and "halfA_k{k}", with per-row network selection and a mask for the single-cell → bulk term.
  - All defaults reproduce Phase 3 bit for bit.
- **New `dgm/labels.py`:** the cross-fitted and budgeted $C_B$ networks (Phase 2 configuration), with hashes in `results/phase5/models.json`.
- **Tests:**
  - the anchored identity loss is zero at initialization;
  - the distance loss is zero when the generator equals the start and stays zero under a pure rescaling of the outputs (batch-standardized), and is positive for a permuted output;
  - the bottleneck has the stated shape and starts at the baseline;
  - the cross-fit folds partition the training donors and no network reads its own donors' rows;
  - the B-k networks see only k half-A donors, and the masked task term never touches their rows;
  - defaults are bit-identical to Phase 3.
- **Notebook** `phase5_exploratory.ipynb`:
  - the pre-check;
  - the Track A table and accuracy–realism figure with the frontier;
  - the sample-specific component per condition;
  - the confirmation;
  - the Track B table, per-direction composition gain against k with the ridge + spread comparator;
  - Implications.

## Not in Phase 5
Noise-input generators and diffusion (future work); a contrastive input–output loss (the outputs already identify their inputs: within-donor top-1, and the cycle loss); discriminator changes; halving 1; test.

## Amendment (2026-10-13, after the pre-run code audit; before any Phase 5 run)
- **Paired comparator:** ridge + spread cannot be fit on fewer than about 10 donors (its stretch factor is scored on held-out folds by an evaluator whose classifier test needs at least 2 donors per fold). The comparator is plain ridge at k = 8, 16, 33 and ridge + spread at k = 16, 33; none at k = 4. The table says so.
- **A-dist pre-check:** the thresholds are applied to the means over the two directions (one weight serves both directions); the table stays per direction. Result: start 0.335 vs M1 0.325, so A-dist runs at one weight, the 1× weight (0.56; adversarial term 0.591, distance term 1.062 at M1's plateau). Since the start's geometry is no more faithful than M1's, a null A-dist result is the expected outcome.
- **Label-network training:** the cross-fit and budget networks use the Phase 2 configuration's 184 epochs at every training-set size. With training steps matched to the full-size network, label quality at k = 4 and 8 changes by at most 0.02 (audit check), so small-k label quality is limited by data, not by training steps.
- **CIs** only for centered R² and marker composition, as in Phase 4.
- **The masked single-cell → bulk term** averages over the eligible rows of each batch (about 17/33 of 64), so it is noisier than the unmasked term at the same weight.
- **If the confirmation picks A-idoff** (the control), it is reported as "identity loss off", not as a new method.
