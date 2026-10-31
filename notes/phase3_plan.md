# Phase 3 analysis plan (fixed before any translation model is trained)

Copied verbatim into the Methods of `phase3_translation.ipynb`. Any later change is recorded here with its reason and date, and reported.

## Setting
- **Training data:** halving 0 of the training donors (`dgm.data.train_halves(pairs, 0)`): single-cell profiles of half A (33 donors, 288 draws), bulk profiles of half B (35 donors, 294 draws). No pair is seen in training. The final configuration of each model is also trained on halving 1 and reported as a sensitivity check.
- **Development:** validation pairs (135 pairs, 14 donors), scored by the Phase 1 evaluators. **Test** (119 pairs, 14 donors): once, in Phase 6.
- **Models:**
  - **M1, strictly unpaired:** CycleGAN (Zhu et al., 2017) with least-squares adversarial losses (Mao et al., 2017) and an L1 cycle loss; no composition task loss (the pair-free readout failed its criterion, Phase 2 §4.5).
  - **M2, weakly paired:** M1 plus the composition task loss through the frozen composition networks (the proposal's CyCADA [Hoffman et al., 2018]): $\lVert C_B(G_{S\to B}(x_S)) - c^*(x_S)\rVert_1$ and $\lVert C_S(G_{B\to S}(x_B)) - C_B(x_B)\rVert_1$.
  - **Paired reference:** ridge, ridge + spread, ridge + PC calibration, paired MLP (Phase 1).

## Hypotheses, primary metrics and comparators
All comparisons on test, per direction, as the paired donor-bootstrap 95% CI (1,000 resamples of the test donors, every metric recomputed on each resample; a model's 3 seeds are averaged inside each resample) of the difference *model − comparator*. Comparators are refit on the same halving as the model.

| | Claim | Primary metric | Comparator (fixed now, from validation on halving 0) | Supported if |
|---|---|---|---|---|
| **H1** | M1 lies beyond what unpaired alignment can reach | the margin beyond the unpaired frontier: realism(M1) − F(R²(M1)), where F(r) is the best realism on any blend path (interpolated between weights 0, 0.1, …, 1) of two of mean shift, ComBat, CORAL and the training-target mean (all fit on the same halving) with centered R² ≥ r (Phase 1 §3.8, `dgm.frontier`); if no path point is as accurate as M1, its accuracy lead over the most accurate point | the unpaired frontier, recomputed on each bootstrap resample | the margin's CI lies above 0 in that direction |
| **H2** | Pair-trained composition supervision helps | composition R² under the **marker probe** | M1 | CI above 0 in that direction |
| **H3** | What pairs buy | centered R², realism, marker-probe composition R² | ridge + spread | reported with CIs; no claim |

**Reporting H1.** Report the margin with its CI, the model's accuracy and realism, and the frontier value at the model's accuracy, with:
- the share of bootstrap resamples whose margin is of the realism kind and of the accuracy kind, and the realism-margin CI over the realism-kind resamples only (the decision is sign-based, so mixed kinds don't affect it, but its magnitude is only interpretable within one kind);
- the wording: CI above 0, "supported on halving 0"; CI including 0, "not distinguishable from the unpaired frontier"; CI below 0, "inside the unpaired frontier";
- the qualifier "not robust to the training split" when H1 is supported on halving 0 but the halving-1 margin's point estimate is at or below 0;
- the superseded per-metric H1 (beat the best unpaired baseline on centered R² and on realism separately, both CIs above 0), reported in one line on test and labeled as the version superseded on 2026-10-07.

With 14 test donors a CI including 0 is a likely outcome.

H1 and H2 are reported per direction (single-cell → bulk, bulk → single-cell); a hypothesis is supported in a direction only by its own CIs. No multiplicity correction is applied; all direction-level tests are reported together. The halving-1 results are reported next to halving 0 as a sensitivity check (the halvings share 19 of 33 half-A donors), not as an independent replicate.

**Secondary (reported, no claims):** within-donor top-1 rate; composition R² under the ridge probe and under $C_B$ (for M2, a gain under $C_B$ not matched by the marker probe is labeled suspect); granulocyte gap (single-cell → bulk); diversity; nearest-neighbor ratio; MMD; the selection score.

## Selection and budget
- **Checkpoint and configuration selection for M1 and M2** (aligned with H1, since the selection score's accuracy term is constant at 0 for unpaired models): the checkpoint with the largest mean margin beyond the validation frontier over the two directions, where each checkpoint's margin is averaged with its two neighbors (500 steps before and after) to damp the checkpoint-to-checkpoint noise seen in the pilot (about ±0.1 in realism). Never by a loss value. The paired reference keeps the selection score. Because the rule maximizes realism over many checkpoints on validation, validation realism is optimistic; some shrinkage on test is expected and is reported in Phase 6.
- **Budget:** at most 12 settings per model (M1, M2) in Phase 3, each run with seeds 0–2 (36 runs per model), and at most 4 ablations × 3 seeds in Phase 4. The number of settings actually compared is reported with every result.
- **Metrics for a setting** are the mean over its 3 seeds (each seed's translations scored separately).
- **Compute:** the H1 bootstrap scores about 58 blend points plus the model per resample and direction; computing only centered R² and the C2ST (about 0.3 s each), 1,000 resamples take about 5 hours per model, run once, in the background, with a fixed seed. The other comparisons compute all metrics with 1,000 resamples.

## Pilot and negative controls (before tuning)
- Discriminator trained with shuffled real/translated labels: the adversarial term carries no signal; M1 must beat this.
- The untrained generator (identity residual, bias set to the halves' mean shift): a floor equal to the mean-shift baseline.
- Real held-out target profiles scored as translations: the ceiling of the realism term.
- Decision point Oct 20: if M1's validation margin beyond the frontier (the same neighbor-averaged margin used for checkpoint selection) is at most 0 in both directions, the project reports the negative result and the ladder of pairing as its findings.

## Amendments

**2026-10-07, after the pilot (validation data only; test untouched).** *H1 changed from "beat the best unpaired baseline on each metric separately" to "lie beyond the unpaired accuracy–realism frontier"; the checkpoint rule and the Oct 20 decision rule changed with it.* Reason: the pilot (`notes/phase3_pilot.md`) showed that the CycleGAN moves along the accuracy–realism trade-off from wherever it starts, buying realism for a small loss of accuracy. No unpaired method, the GAN included, can beat mean shift's accuracy and CORAL's realism at once, so the per-metric version would fail any trade-off, however favorable, and could not distinguish a useful model from a useless one. The frontier of blends is the information-free version of that trade-off, it contains every single baseline, and it is stricter than any one baseline. The switch was made after seeing the pilot sit favorably against a first version of the new criterion in single-cell → bulk (+0.14 to +0.15 for the best checkpoints). On review that version had two flaws that flattered the model: shrinking toward the training mean could pass it through the accuracy fallback, and the coarse weight grid understated the frontier by about 0.05 where the model sits. With both fixed (the training mean added to the blend set; paths interpolated), the best pilot checkpoints are +0.04 to +0.08 beyond the frontier in single-cell → bulk and at most +0.02 in bulk → single-cell, and −0.03 to −0.06 and about −0.03 with the three-checkpoint averaging adopted for selection: the pilot GAN sits on the frontier, not beyond it. Biases: the frontier is a maximum over many noisy points (biased up, against the model); checkpoint selection on validation is biased toward the model.

**2026-10-11, after the pre-run code audit (before any Phase 3 run; validation and test untouched by Phase 3).** Implementation details fixed before training:
- For M2, only checkpoints at or after step 8,000 (when the task loss reaches full weight) are eligible for selection; before the warm-up ends an M2 run is identical to M1, so H2 could otherwise compare M1 with itself.
- The M2 grid uses the three best M1 settings by validation margin; the task loss ramps linearly from 0 at 20% of training to full weight at 40%.
- Next to the planned H2 comparison (chosen M2 against chosen M1), a descriptive comparison of M2 against M1 at M2's own base setting isolates the task loss.
- Reported with the Oct 20 decision: the chosen M1's margin without checkpoint selection (mean over all eligible checkpoints, and over the last quarter of training), since the selected margin is a maximum over about 38 checkpoints and 12 settings.
- The shuffled-label negative control is re-run under the final setup (chosen M1 setting, seed 0).
- The halving-1 models are frozen with the halving-0 models, for the Phase 6 "not robust to the training split" qualifier.

## References

- Hoffman, J., Tzeng, E., Park, T., Zhu, J.-Y., Isola, P., Saenko, K., Efros, A. A. & Darrell, T. CyCADA: cycle-consistent adversarial domain adaptation. *ICML* (2018).
- Mao, X., Li, Q., Xie, H., Lau, R. Y. K., Wang, Z. & Smolley, S. P. Least squares generative adversarial networks. *ICCV*, 2813–2821 (2017).
- Zhu, J.-Y., Park, T., Isola, P. & Efros, A. A. Unpaired image-to-image translation using cycle-consistent adversarial networks. *ICCV* (2017).
