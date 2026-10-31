# Phase 6 spec: analysis (final)

Phase 6 explains the locked models and the Phase 5 results and prepares the claims for test.

**Scope.**
- No neural translation model is trained. The optimal-transport map (amendment A2) is fit on the training halves (Sinkhorn, minutes). Everything is computed from saved generators, the Phase 1–5 result tables and the composition networks, on validation (135 pairs, 14 donors).
- The one new training is a retrained copy of each small composition network, used as a control in 6.1.
- The phase follows a review of the project's story on 2026-10-20.

**Framing it serves (validation, point estimates).**
- Without pairs, adversarial training buys realism but not information. The CycleGAN reaches what blending simple unpaired baselines reaches, and it gives up composition fidelity for that realism.
- In accuracy (centered R²), pairs pay off far more as regression targets than as composition labels.
- In composition, the comparison depends on the number of paired donors k and on the direction (6.4).
- In bulk → single-cell, composition labels partly undo the loss that the adversarial objective causes, but they never get back above the no-pairs ComBat start.

## Conventions
- **Translations analyzed:**
  - **Locked M1 and M2** at the chosen checkpoints, raw weights, seeds 0–2. Selection-biased; labeled so.
  - **Phase 4 fresh seeds 3–5**, at the last-quarter kept checkpoints (15,000, 17,500, 20,000), raw weights. Free of checkpoint selection, but the setting was still chosen on these validation donors.
  - **Phase 5 models** (B-cf, the budget models B-k): averaged (EMA) weights at the same kept steps, seeds 6–8.
  - Runs are resolved through each phase's `models.json` or registry entries, never by matching configs.
- **Raw and EMA weights.** Any table or panel with a Phase 5 model also carries M1 and M2 as EMA, seeds 6–8, at the last-quarter kept steps. Raw rows (locked or fresh seeds) appear only in a separately labeled column or panel.
- **Uncertainty:**
  - paired donor-bootstrap 95% CIs over 1,000 resamples of the 14 validation donors (donors are resampled, not samples), with seeds and checkpoints averaged inside each resample;
  - matched blend weights (6.2) are held fixed across resamples;
  - every CI is labeled "validation; these donors also chose the models".
- **Realism** always means the classifier test on the top 10 target principal components (with squares). Figure 3 cites the Phase 4 alternatives (`results/phase4/p1_margins_val.csv`): M1's margin is negative under logistic-50, about 0 to negative under boosted-50 (single-cell → bulk chosen: −0.001), and positive under −MMD. The limitations say what this measure doesn't see.
- **Space:** log1p CP10k over the 5,000-gene panel, unless stated otherwise.
- **Other donor $y'$:** a seeded draw of a real target from a different validation donor, as in Phase 5's sample-specific component.

## Analyses

### 6.1 Exploitation (attack-and-transfer) check
Required before H2 is reported on test (Phase 2 spec, gaming guard 3). Question: does the generator move the composition network it trained against more than it moves the evaluation probes?

- **Inputs and pairings:**

  | Model | Paired with |
  |---|---|
  | Locked M2 (chosen checkpoint) | locked M1 (chosen checkpoint); this is also M1 at M2's base setting (Phase 3 §4.2) |
  | Fresh M2, each seed and kept step | M1 on the same seed and kept step |
  | B-cf, each seed and kept step | M1 EMA on the same seed and kept step |

  Both directions are covered.
- **Target $c^*$:** what the task loss trained toward.
  - Single-cell → bulk: the measured composition of the source draw.
  - Bulk → single-cell: the pseudo-label $C_B(x_B)$. For locked and fresh M2 this comes from the locked $C_B$. For B-cf it is the mean of its 5 fold networks; these are out of sample on validation.
  - Every reader uses the same $c^*$, so the comparison measures transfer.
  - The pair-truth version (H2's metric) is reported separately.
- **Gap closure** per sample: $g = 1 - \lVert R(G_{M2}(x)) - c^*\rVert_1 / \lVert R(G_{M1}(x)) - c^*\rVert_1$. It is NaN when the denominator is 0.
- **Readers $R$:**
  - $C$, the network the model trained against:
    - in single-cell → bulk, $C_B$ (locked; for B-cf, the mean of its 5 fold networks);
    - in bulk → single-cell, $C_S$ (locked for M2; Phase 4's half-A $C_S$ for B-cf);
  - $C'_B$ and $C'_S$, retrained copies for the locked and fresh M2 only:
    - Phase 2 configuration and epochs (184 / 102), the locked networks' training rows;
    - member seeds 3–5, read as the 3-member mean;
    - B-cf gets no $C'$ arm, because retraining its 5 fold networks and the half-A $C_S$ is out of scope;
  - the marker probe, which is the decision probe;
  - the ridge probe, secondary, since it shares $C$'s training data.
- **Reported:**
  - the median $g$ and the share with $g > 0.5$, as in Phase 2;
  - the ratio-of-sums version, for robustness;
  - the transfer ratio, median $g_{marker}$ / median $g_C$.
- **Reading:**
  - $g_C \gg g_{C'} \approx g_{marker}$: the generator exploits $C$'s particular weights.
  - $g_C \approx g_{C'} \gg g_{marker}$: the composition-network family and the probe disagree.
- **Decision rule, fixed now:** if the marker-probe transfer ratio is below 0.5 in a direction, H2 on test in that direction carries the label "the generator partly fits its training readout". This mirrors Phase 2 rule 2.
- **Guard on the rule:** if median $g_C \le 0.1$ in a direction (including $\le 0$, where the generator did not move $C$ toward $c^*$), the ratio is not computed. That direction reports "no closure under $C$" and gets no label.
- **ε50** of $C$ at the translated inputs, compared with ε50 at real inputs (reused from `results/phase2/val_attack.csv`). Computed for the locked M2 seeds 0–2 and B-cf at step 20,000, seeds 6–8, both directions: 12 attack runs.

### 6.2 Is M1 a blend? (with the move decomposition, formerly 6.3)
Question: does M1 add anything sample-specific beyond an information-free map?

- **Matched blend**, per seed and checkpoint, against the halving-0 validation frontier:
  - On every blend path (two of mean shift, ComBat, CORAL and the training-target mean) whose re-scored centered R² crosses M1's, solve by bisection on the actual blended translations for the weight $w$ at which R²(blend) = R²(M1).
  - Re-score realism at that $w$ and keep the crossing with the highest realism. Ties go to the lower-index pair in `sorted(names)`, then the smaller $w$.
  - If no path crosses, report "no matched blend" (NaN), with no fallback.
- **Primary statistic:** partner discrimination of M1 minus that of the matched blend: rank_norm and within-donor top-1 from `Evaluator.score`, with a CI. Matching on accuracy does not fix this statistic, and it reads directly as "does the translation know which sample it came from".
- **Secondary statistic:**
  - Residual $r = G(x) - \text{blend}(x)$, with unit residual $\hat r = r/\lVert r\rVert$.
  - Sample-specific alignment $\Delta_i = \langle \hat r_i, y_i - \bar y_{-i}\rangle$, where $\bar y_{-i}$ is the mean of the other donors' validation targets.
  - Computed in genes and, as a check, in the top 50 centered target PCs.
  - **Not used:** regressing the needed correction $d$ on $r$. Matching on accuracy roughly fixes $\langle r, d\rangle \approx \lVert r\rVert^2/2$, so the fit reflects the residual's size, not its direction.
- **Scale references** next to the M1-vs-blend correlation and L2 distance:
  - the M1 seed-to-seed distance;
  - the distance from the matched blend to the second-best crossing blend (NaN when only one path crosses).
- **Best information-free map:**
  - Per gene, a least-squares fit of G's outputs on the training-half sources to the mean-shift, ComBat and CORAL translations of the same inputs, plus an intercept: 4 coefficients per gene, no pairs.
  - Share explained on validation = 1 − SSE / SST, with SST taken about the mean of G's validation outputs.
- **Move decomposition with CIs:** Phase 4 Table 4.9 (along/across the needed correction) and the Phase 5 sample-specific excess (own partner minus $y'$), recomputed with the same per-sample statistics and nulls as `p0_move_decomposition_val.csv` and `sample_specific_val.csv`, with CIs. Covers M1, M2, B-cf and ridge + spread. This is the same own-minus-other statistic measured from the source rather than from the blend, so 6.2 shows both in one table.
- **Readouts:** fresh seeds 3–5 are primary; the chosen checkpoints are secondary.

### 6.3 Per-lineage composition (marker probe)
Question: where does adversarial training lose composition, and where do labels win it back?

- One table per direction, giving the R² for each of the 6 classes and the mean over classes with a CI.
- **Rows:** the start (mean shift and ComBat), M1, M2, B-cf, ridge + spread, and real targets as a reference.
- **The real-target row:**
  - In bulk → single-cell it is the probe's ceiling (0.763).
  - In single-cell → bulk it is only a reference (0.463). Translations above it are flagged, since they can carry source composition (Phase 2 guard 1).
- The table notes that T cells lack canonical markers in the panel (Phase 0).
- **Granulocyte column (single-cell → bulk):** the Pearson correlation over validation samples between the translated and real granulocyte score (the mean of the 7 genes), with a CI. Ridge + spread is the positive control.

### 6.4 What pairs buy, in absolute units
Question: how do the uses of k paired donors compare?

- One row per direction, with 3 panels: centered R², realism and marker composition R², each plotted against paired donors k = 4, 8, 16, 33, 68.
- **Lines:**

  | Line | k values | Source |
  |---|---|---|
  | Ridge | 8–68 | k = 68 re-scored with the marker probe, since Phase 1 has only the ridge probe |
  | Ridge + spread | 16–68 | k = 68 from Phase 3 Table 3.6 |
  | Weakly paired GAN, EMA | 4, 8, 16 nested | Phase 5 budget models |
  | Weakly paired GAN, EMA | 33 | cross-fitted within half A; networks saw 26–27 donors |
  | Weakly paired GAN, EMA | 68 | B-cf; networks saw 54–55 donors |

  - There is no paired comparator at k = 4.
  - The k = 33 and 68 points are marked as not on the same footing as the nested ones.
- **References** (horizontal lines): M1 (EMA, seeds 6–8), the start, and real targets (a ceiling only in bulk → single-cell).
- **The caption states:**
  - at k = 4–16 the single-cell → bulk term is applied only to the other 17 half-A donors;
  - the GAN also trains on both full unpaired halves, while ridge uses only its k donors' pairs.
- **Error bars:**
  - seed SD, which includes the donor draw, for the GAN points and for ridge at k < 33 (the legend distinguishes seed SDs from bootstrap CIs);
  - ridge at 33 and 68 is deterministic, marked so, with a donor-bootstrap CI only.

### 6.5 Donor-level reference and within- vs. between-donor recovery (formerly 6.6 and 6.8)
Question: where does a realistic profile carrying only donor-level information sit, and is each model's signal donor-level or sample-level?

- **Same-donor reference.** Each source is scored against another draw of the same donor (the Phase 1 Table 1.9 construction, `default_rng(0)`), with the marker probe for composition.
  - Phase 1 values: centered R² −0.011 / −0.140, realism 0.954 / 0.991.
  - This is not a ceiling: plain ridge is more accurate.
- **Similarity:** the centered correlation (centered by the training-target mean), taking the median over samples.
  - Computed for each model against the true partner, the same donor's other draws, and other donors.
  - Within-donor top-1 is reported alongside.
  - The real-pair reference is recomputed on the 135 validation pairs with the same centering, next to Phase 0's all-sample 0.269 / 0.214.
- **Positive control:** ridge + spread at k = 68.

### 6.6 Cycle steganography check
Question: does the cycle pass source information through a hidden low-amplitude channel (Chu et al. 2017)?

- **Fit set:** the PCs and SDs are fit on the training-half targets of the direction, i.e. the discriminator's real set (half B bulk for single-cell → bulk, half A single-cell for bulk → single-cell).
- **Noise** (Gaussian; κ ∈ {0.01, 0.03, 0.1}; 10 seeded draws per sample), in three conditions:
  - **top-50:** in the top 50 target PCs, per-PC SD = κ × that PC's SD;
  - **complement:** in the orthogonal complement of those PCs, scaled to the same total norm as top-50 at each κ;
  - **per-gene:** per-gene SD = κ × that gene's SD on the fit set (genes with zero SD get no noise).
- **Ratio:** the sensitivity ratio ‖F(G(x)+ε) − F(G(x))‖ / ‖F(y+ε) − F(y)‖, where y is the source's real validation partner and both arms receive the same ε vector.
  - No clamping on either arm.
  - Aggregated as the median over samples of the per-sample ratio, averaged over draws.
- **Also reported:**
  - the cycle error with and without noise;
  - the same ratio for the untrained generators (the start), which is the baseline: a nonlinear F can be steeper off the data manifold without a hidden channel.
- **Reading (descriptive, no threshold claim):**
  - A ratio well above the start's at small κ means the reconstruction leans on a low-amplitude signal that real targets do not carry.
  - If that happens only under the complement or per-gene conditions, the signal sits outside the top 50 PCs.
- **Coverage:** M1 and M2, fresh seeds 3–5, raw weights at the 3 kept steps, both cycles. Forward passes only.

### 6.7 Memorization
- Per model: the nearest-neighbor distance from each translation to the training-half targets the model's discriminator saw (half B bulk, half A single-cell), relative to the same distance for real validation targets.
  - This training-half ratio is primary; the Evaluator's all-training version is secondary.
  - Both are the existing nearest-neighbor ratio, tabulated per model.
- **Reading:** a ratio well below 1 points to memorization, or to collapse toward training profiles.

**Dropped from the plan:** top genes of the learned residual. Gene-level findings cannot be validated here, and the story doesn't need them.

## Figures (report; the talk reuses 2–4)
1. **The pairing ladder** (small method schematic).
2. **What pairs buy** (6.4), full width: 2 directions × 3 metrics. This is the headline; Fig. 5.2 moves to the supplement.
3. **The accuracy–realism plane** (Fig. 3.1, updated). It adds:
   - B-cf, M1 and M2 as EMA (seeds 6–8, last-quarter kept steps), marked separately from the raw locked points;
   - the same-donor reference.
4. **Mechanism:**
   - (a) the composition trajectory (Fig. 4.2 style), with the ComBat and mean-shift starts as lines;
   - (b) along/across bars with CIs and nulls, plus the matched-blend row (6.2).

The per-lineage results are a table. In the talk, the fourth results slide is the steganography or exploitation result, whichever is clearer. Supplement: Figs. 3.2, 3.3, 4.3, 5.1, 5.2.

## Written before test (`phase6_analysis.ipynb`, Implications for Phase 7)
- **H1 on test:** draft all three readings from the prereg:
  - supported;
  - not distinguishable from the unpaired frontier, the likely reading with 14 donors;
  - inside the unpaired frontier, plausible in bulk → single-cell, where the last-quarter margin is −0.057.
- **The H1 write-up also reports** the share of resamples of the realism kind vs. the accuracy kind, and the one-line superseded per-metric H1.
- **Halving-1 qualifier.** The validation halving-1 bulk → single-cell margin (−0.332) makes "not robust to the training split" likely, but the qualifier is applied from the halving-1 test margin.
- **H2 on test:** draft both readings, each with the 6.1 label if it applies, and the caveat that the locked M2 had pair information in both task terms.
- **Multiplicity:** 4 primary CIs (H1 and H2 × 2 directions), per direction and uncorrected, as fixed in `notes/phase3_plan.md`. H3 (ridge + spread) is reported with CIs and no claim.
- **Configuration count** for any exploratory model sent to test: settings × seeds compared on validation across Phases 3–5.
- **Test compute:**
  - the H1 frontier bootstrap takes about 5 hours for M1;
  - the OT comparator's frontier bootstrap takes about 5 hours more;
  - B-cf and the other comparisons take minutes.
  - Plan two laptop nights.

## Code, tests, audit
- **New `dgm/analysis.py`:**
  - the matched blend (bisection on re-scored paths);
  - partner discrimination of the matched blend;
  - $\Delta_i$ and the move decomposition;
  - the least-squares information-free map;
  - gap closure and transfer (NaN-safe);
  - the donor bootstrap for per-sample statistics;
  - the steganography sensitivity ratio;
  - the run lookup through the registries.

  The attack reuses the Phase 2 code; $C'$ uses `dgm.composition.freeze_nets`.
- **Tests:**
  - the matched blend's re-scored R² equals M1's to within the bisection tolerance, and is NaN when M1 lies beyond every path;
  - the residual is 0 and $\Delta$ is NaN when the input is the blend itself;
  - $g = 0$ when $G_{M2} = G_{M1}$, and $g = 1$ when $R(G_{M2}(x)) = c^*$;
  - the same-donor reference never pairs a sample with its own draw;
  - the sensitivity ratio is 1 when F is linear, in all three noise conditions;
  - regenerated translations from saved states reproduce the cached Phase 3–5 metrics, and the 6.2 move decomposition reproduces the Phase 4 and 5 point estimates.
- **Compute:**
  - forward passes and bootstraps: minutes each;
  - $C'$: 2 small networks;
  - 12 attack runs (6.1), the only part likely to exceed 20 minutes. They get a code audit first.
- **Review:** a one-round results audit by Claude, then polish and a final check by Claude.

## Amendment (2026-10-20, the author's decisions on the two open items, before any Phase 6 code ran)

**A1. B-cf goes to test as an exploratory, weakly paired model.**
- **Declared configuration:**
  - the Phase 5 B-cf setting: M2 with bulk composition networks cross-fitted over the 68 training donors, and Phase 4's half-A $C_S$;
  - seeds 6–8;
  - EMA weights at steps 15,000, 17,500 and 20,000.
  - There is no checkpoint selection. Metrics are computed for each seed and step, then averaged over the 9 inside each bootstrap resample.
- **Primary, per direction:** paired donor-bootstrap CIs of Δ marker composition R² and Δ centered R², for B-cf minus M1 in EMA form (seeds 6–8, kept steps).
- **Secondary:**
  - the same differences against M2 in EMA form;
  - realism and within-donor top-1;
  - the frontier margin, which is descriptive and is not an H1 test;
  - the differences against the locked M1 and M2, labeled as mixing raw and EMA weights.
- No multiplicity correction is applied. Never compared against H1.
- **Reporting:**
  - labeled exploratory;
  - the configuration count across Phases 3–5, computed in Phase 6, is stated;
  - the write-up says the design used the Phases 3–5 validation results;
  - the 6.1 label, if it applies, travels with B-cf to test.
- **Limitation:** B-cf has no halving-1 runs, so no split sensitivity check is reported for it.

**A2. One non-adversarial unpaired comparator: entropic optimal transport (OT).** Exploratory, scored on the same plane, frontier and readouts as M1.

*The map:*
- Each direction starts from the GAN's own start map s (mean shift for single-cell → bulk, ComBat for bulk → single-cell), fit on the training halves.
- Costs are squared Euclidean distances between $s(x)$, for the training-half sources, and the other half's targets. They are computed in the top 50 PCs of that half's targets; the PCA is fit on those training-half targets only.
- Entropic OT with uniform weights, solved by log-domain Sinkhorn to a marginal tolerance of 1e-9.
- The regularization is ε ∈ {0.01, 0.03, 0.1, 0.3} × the median cost: 4 settings. One ε serves both directions.
- The Sinkhorn potentials are computed at the same ε.

*Variant OT-b (barycentric; the declared comparator).* A new source x maps to

$$T(x) = \sum_j w_j(x)\, y_j, \qquad w_j(x) \propto \exp\!\big((g_j - c(s(x), y_j))/\varepsilon\big).$$

- This is the entropic out-of-sample extension; with uniform weights the $\log b_j$ term cancels.
- The sum runs over the training-half targets in full gene space, so outputs are convex combinations of real training profiles.
- Small ε approaches copying the nearest target; large ε approaches the training mean.

*Variant OT-s (conditional sampling; descriptive only).*
- For each source, draw one target j with probability $w_j(x)$ and output $y_j$.
- 3 seeded draws are scored and averaged like seeds, with the SD across draws.
- Its outputs are real training profiles. Its realism will be near that of real profiles, and its memorization ratio is 0 by construction. Its information shows only in accuracy and in partner discrimination (6.2). It is read as a retrieval model, not as evidence about generation.
- OT-b and OT-s bracket what this OT plan can reach: the conditional mean (most accuracy, least spread) and a conditional sample (full spread).

*Selection:*
- For each variant, the ε with the best validation margin averaged over the two directions.
  - As in Phase 3, a direction whose margin is of the accuracy kind is averaged in its own units.
- For OT-b, an ε is eligible only if its training-half memorization ratio (6.7) is at least 0.9 in both directions. This keeps the rule from picking near-copies, which a classifier test scores as realistic whatever they carry.
  - If no ε is eligible, OT-b is reported on validation as "no eligible setting", with all 4 settings shown on the path. It then does not go to test.
- There are no checkpoints. All settings are reported, and the 4 + 4 count toward the configuration count.
- OT's validation margin rests on 4 settings and no checkpoints, against M1's roughly 38 checkpoints × 12 settings. The two validation margins are therefore not equally optimistic.

*Readouts and reporting:*
- **Readouts:** margin, centered R², realism, marker composition R², within-donor top-1, 6.2 partner discrimination against the matched blend, and the 6.7 memorization ratio.
- **Uncertainty:** donor-bootstrap CIs. OT-b is deterministic, so it has no seed SD.
- **On the plane:** all 4 OT-b settings are plotted as a path, so that a winner at the shrinkage end is visibly that end and not a property of OT in general.
- **Sensitivity check:** OT is fit on halving 1 at the ε chosen on halving 0, with no re-selection, and reported as for the locked models.
- **Test:** OT-b at the chosen ε goes to test next to M1, as an exploratory comparator. Its frontier margin is reported with a CI, labeled exploratory, and it is not part of H1. OT-s is reported on test descriptively.

*Reading (validation level, exploratory):*
- An OT margin above 0, together with a memorization ratio well below 1 or partner discrimination no better than the matched blend's, is read as copying, not information. This applies to OT-s by construction, so OT-s cannot support the reading below.
- If OT also lands on or inside the frontier, the result is consistent with the limit lying in the unpaired setting under these readouts, rather than in adversarial training. One non-adversarial method does not show that no unpaired method can pass the frontier.
- If OT passes the frontier without copying while M1 does not, a non-adversarial unpaired map can do what M1 cannot. That is evidence against the adversarial objective being needed for M1's position, and suggests that the adversarial method, rather than the unpaired setting, limits M1.

*Code:* `dgm/ot.py`, Sinkhorn and both maps in numpy (no new dependency). Tests:
- the marginals match to tolerance on a synthetic problem;
- with n = m and a unique optimal permutation, at very small ε each training source maps to its permutation partner, within tolerance;
- at very large ε every OT-b output equals the target mean;
- the out-of-sample map reproduces the in-sample barycentric projection on the training sources;
- OT-s draw frequencies on training sources match the plan's rows within sampling error.

*Figures:*
- OT joins the plane (Fig. 3) with its own markers and the OT-b path.
- OT appears in the "is it a blend" panel (Fig. 4b).

## Schedule
Phase 5 finished early, so Phase 6 can start now. There are no overnight runs. The plan's Nov 9–15 window becomes slack for the report draft and the figures.
