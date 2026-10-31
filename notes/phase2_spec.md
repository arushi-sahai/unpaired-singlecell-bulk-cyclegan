# Phase 2 spec: composition task nets (v3.1, revised after three spec reviews, 2026-10-06)

**Status:** spec only, no code yet. v1 and v2 were reviewed and **not approved**. The author set the goals, Claude drafted the spec and ran the reviews and audits, and the author revised and approved the final version. Changes for the first review are tagged **[audit N]**, and changes for the second review (v2 → v3) are tagged **[audit2 X]**. The overview plan (Phase 2) is the starting point.

## Goal

Train and freeze two composition networks. The Phase 3 translation model's task loss uses them:
- $C_{bulk}$: whole-blood bulk profile → the draw's PBMC cell-type fractions. Applied to $G_{S\to B}(x)$.
- $C_{sc}$: single-cell pseudobulk profile → the same fractions. Applied to $G_{B\to S}(x)$.

Targets are the 6-class AIFI_L1 fractions in `soundlife_pairs.npz` (T, Monocyte, B, NK, DC, Other). Train means: 0.63, 0.18, 0.088, 0.083, 0.013, 0.007. Inputs are log1p(CP10k) on the 5,000-gene panel.

A task net is useful only if it:
1. predicts composition from **real** profiles of its domain about as well as the ridge probe;
2. behaves sensibly on **translated** profiles;
3. **can't be satisfied by small non-biological input changes** (exploitability);
4. **isn't the evaluation probe**: the selection score's biology term is the Phase 1 ridge probe's comp R², so the task signal must be a different model, and Phase 3 must be checked against a second, independent probe.

## Data, splits and the use of val

- **Training and all model selection use the 582 train pairs only.**
- **Inner folds:** 5-fold, grouped by donor, stratified by age group, with a fixed seed. Every fitted quantity is computed **per inner fold from that fold's training part**: the standardization statistics [audit 9], the probe used for comparison, and the baselines used for robustness. The fold's held-out donors are used for scoring.
- **[audit 1] Selection, the exploitability check and the robustness check all run inside the inner folds.** Val is never used to choose or re-choose anything.
- **Val (135 pairs) is used once, after freezing, for the report.**
  - If the frozen nets fail a criterion on val, that goes to the author for a decision. **There is no re-selection.**
  - Test stays untouched.
- **[audit 9] Code guard:** the training and selection functions take the pairs object and **refuse any row whose split isn't train** (a hard assert, with a test).

## Model family (the grid)

`CompositionNet` (in `dgm/models.py`): **input scaling** → [Linear(5000 → h), LayerNorm, LeakyReLU(0.2), dropout p] → Linear(→ 6) → softmax. With h = 0 the hidden block is dropped, giving a linear-softmax model.

**[audit2 A] Output-layer initialization (every arm):** the final layer's **weights start at zero** and its **bias at log(mean training fractions)** of the data being fit (the fold's training part, or all train for the freeze). The model then predicts the training mean composition for **every** input at initialization (with nonzero weights, the hidden layer's output would shift the logits), not 1/6 per class. Gradients still flow: the final weights move on the first step, and the hidden weights after that. Without this, the linear-softmax arm never reached positive comp R², because comp R² weights DC (1.3%) and Other (0.7%) equally with the large classes.

| Axis | Values | Why |
|---|---|---|
| input scaling **[audit 2]** | `center` (subtract the train mean); `floor` (center, divide by max(SD, median SD)); `full` (center, divide by SD) | Full standardization makes the ~3,700 near-constant single-cell genes (SD as low as 0.0012) dominate the gradient. It was exploitable in the spec audit (65% / 88% closure at ε = 0.03). |
| architecture | h = 256; h = 0 (linear-softmax) | A linear softmax link is differentiable and still differs from the ridge probe (different link, loss and regularization) [audit 2, nets fell below ridge] |
| training loss | L1 on fractions; soft-target KL | Only the selection metric below compares them |
| weight decay | 1e-5, 1e-4, 1e-3 | 1e-2 collapsed in the audit, so it's excluded |
| dropout p (h = 256 only) | 0, 0.2 | [audit 2: p was unspecified] |
| input noise | none; Gaussian with σ = 0.1 × the median gene SD, **in raw units, the same for every gene** | Noise scaled by each gene's SD doesn't protect low-variance genes [audit 2] |

Training: Adam (lr 1e-3), batch 64, up to 300 epochs, with the held-out comp R² recorded each epoch.

**[audit2 F] One epoch count per config:** take the epoch that maximizes the **fold-averaged** comp R² curve, score every fold at that epoch, and freeze at it. Each fold's own best epoch would be picked on the donors it's scored on, which flatters it.

**Determinism:** CPU, float32, `torch.use_deterministic_algorithms(True)` [audit 6], and **`torch.set_num_threads(4)` pinned** [audit2 suggestion], since CPU reduction order depends on the thread count. **One seed per config** in the folds; 3 seeds only for the freeze.

## Selection (one objective, inside the folds)

1. **[audit 2] Selection metric:** out-of-fold comp R² (`comp_r2`: per-class R² across samples, mean over the 6 classes), averaged over the 5 folds, every fold at the config's single epoch count [audit2 F]. The same metric works for both losses.
2. **Robustness constraint [audit 1]:** in each fold, Phase 1 baselines (ridge + spread with its train-chosen factor, CORAL, mean shift) are fit on the fold's training part and translate the fold's held-out *sources*. **[audit2 E]** Clip each comp R² at 0 first, net and probe alike, then take per fold and baseline the difference d = net − probe. A config passes if both:
   - the mean of d over folds and baselines is ≥ −0.10;
   - each baseline's d, averaged over folds, is ≥ −0.20.

   Configs that fail are excluded. Clipping matters: one baseline where the probe is far below 0 (B→S mean shift was −5.48 in Phase 1) would otherwise make the constraint trivially true.
3. **Exploitability constraint [audit 2, 3]:** for the **top 5** configs by (1) that pass (2), run the attack below on each fold's held-out donors. A config passes if its median ε50 ≥ max(0.03, 0.5 × ε50 of the ridge probe). **Aggregation:** "median ε50" pools the per-sample ε50s of all held-out samples across the 5 folds, for the net and for the probe alike. Each fold's probe is fit on that fold's training part.
   - **[audit2 B]** The probe's ε50 is measured with the **same PGD and bisection** as the nets: the probe is wrapped as a linear torch layer and its output put through the same g ≥ 0.5 rule.
   - The closed form overstates the probe's ε50 (bulk 0.119 vs 0.055 by PGD in the second audit), so it's reported only as an upper bound.
   - Second audit's estimate: ridge probe PGD ε50 0.055 (bulk), 0.067 (single-cell). The threshold is therefore about 0.03 / 0.034, which center-scaled configs met in both domains.
4. **Pick** the passing config with the best (1).
   - If none of the top 5 passes, extend to the next 5.
   - If still none passes, **stop and decide with the author** (Phase 3 fallbacks are listed at the end).
5. **Freeze:** refit the chosen config on all 582 train pairs with **3 seeds**, for the config's fold-averaged epoch count [audit2 F]. The frozen artifact is the **mean of the 3 seeds' outputs**.
   - The ensemble is for smoothing and seed stability. **It isn't claimed as a robustness mitigation** [audit 9]; its ε50 is measured like any other model's.
   - (Alternative considered: a cross-fitted 5-fold ensemble. Not chosen, because the refit uses all train data.)

## The attack and its metrics [audit 3]

For a frozen or candidate model $C$ with inputs in domain $D$:
- **ε scale:** $s_D$ = median over **train** samples of $\|x_i - \bar x_{train}\|_2$, fixed per domain. Budgets are $\epsilon \cdot s_D$.
- **Target:** for each input $x_i$, a target composition $c_t$ = the measured fractions of a sample drawn (seeded) from a **different donor** in the same scoring set.
- **PGD:**
  - loss $\|C(x+\delta) - c_t\|_1$ on the model's output (for the frozen ensemble, its mean output);
  - 200 steps, step size $2.5\,\epsilon s_D / 200$, 3 random restarts (keep the best);
  - after every step, project δ onto the L2 ball $\|\delta\|_2 \le \epsilon s_D$ and $x + \delta$ onto $\ge 0$ (generator outputs are clamped at 0).
- **Gap closure per sample:** $g = 1 - \|C(x+\delta) - c_t\|_1 / \|C(x) - c_t\|_1$. Report the median and the share with $g > 0.5$, at ε ∈ {0.01, 0.03, 0.1, 0.3, 1.0}.
- **Primary metric, ε50:** per sample, the smallest ε with $g \ge 0.5$. **[audit2 C]** Found by bisection on **log ε over [1e-3, 2], 12 iterations**, which resolves ε to within about 0.2%. Report the median.
  - The probe's ε50 is measured the same way [audit2 B].
  - The closed form $\|W^{+}\, 0.5\,(c_t - \hat c(x))\|_2 / s_D$ (minimum-norm δ moving the probe's prediction exactly half-way) is reported as an **upper bound** only.
  - Report **ε50(net) / ε50(probe)** and aim for ≥ 0.5.
- **Transfer:** the ridge probe's own $g$ on the same attacked inputs, and the ratio $g_{probe}/g_{net}$. Also reported for the marker probe (below).
- **Natural-direction reference:** $g$ when the same budget is spent moving $x$ straight toward the target sample's *real profile*, $\delta = \epsilon s_D \cdot (x_t - x)/\|x_t - x\|$. This compares adversarial with biological sensitivity.
- **Gradient-masking check [audit2 G]:** at ε = 1.0 run PGD with **1,000 steps**. A median $g$ below 0.8 is **flagged for investigation**, not an automatic failure, since a weak attack could understate exploitability. (In the second audit a plainly unmasked model reached 0.87–0.89 with 200 steps.)
- **[audit 3] Character of δ** (replaces the C2ST check, which can't see 3% perturbations outside its top-10 PCs):
  - the share of $\|\delta\|^2$ in the top-50 train PCs of domain D;
  - the share of $\|\delta\|^2$ on genes with SD below the median;
  - the 20 genes with the largest |δ|;
  - the change in `r2_centered` if attacked inputs were scored as translations of the clean ones.

## Baselines and the val report (after freezing)

- **Floor:** train-mean composition. **Reference:** the frozen Phase 1 ridge probe (`Evaluator.comp_probe_`), reused, not refit.
- **[audit 7] Pass criterion on val,** per domain: a **paired donor bootstrap** (1,000 resamples of the 14 val donors, recomputing comp R² for net and probe on each resample) of the difference (net − probe).
  - **Pass** if the point estimate is ≥ −0.05 and the 95% CI of the difference includes 0 or lies above it.
  - **Applies to both $C_{bulk}$ and $C_{sc}$** [audit 2]. A fail goes to the author for a decision; there's no re-selection.
- **Also reported:**
  - per-class R² and Pearson r;
  - mean absolute error;
  - calibration (predicted vs true mean per class);
  - the val attack metrics (ε50, transfer, natural reference, δ character);
  - robustness on val translations from the frozen Phase 1 baselines.

## Second, independent evaluation probe [audit 4]

- The kNN probe in v1 is **dropped**: it scored 0.016 / 0.185 in the audit, since composition signal sits mostly outside the top PCs.
- **Marker probe instead:** ridge on about 30 canonical lineage markers present in the panel, each lineage represented by its established markers. The list is fixed in code before fitting and printed in the notebook. Examples: CD2, LCK, IL7R, CD5, CD6 (T); MS4A1, CD79A, CD79B, CD19, BANK1 (B); CD14, FCN1, LYZ, S100A9, VCAN (monocyte); KLRF1, NCAM1, SH2D1B, NKG7, GNLY (NK); FCER1A, CLEC10A, CD1C, LILRA4 (DC); PPBP, PF4 and HBB / HBA2 if present (Other).
  - Alpha by donor-grouped CV on train; one probe per domain.
  - Audit estimate: 0.324 / 0.772 comp R².
  - **Fit and frozen in Phase 2** in `dgm/composition.py`. **[audit2 suggestion]** It's hashed with the same canonical scheme as the nets: its coefficients, intercept, marker list and alpha as named arrays. It's **reported only and not added to the frozen selection score.**
- **Fixed for Phase 3** [audit 5]: report comp R² under both probes. A gain over the baselines under the ridge probe that the marker probe doesn't confirm (marker comp R² up by at least half as much) is labeled **suspect**, and isn't counted toward "beats the baselines".

## Gaming guards carried into Phase 3 [audit 5]

The frozen selection score isn't changed. Phase 3 must also:
1. report the biology term next to the real-target reference, and flag gains above it, since S→B can exceed real bulk by carrying source information;
2. require marker-probe agreement (above);
3. re-run the attack/transfer test on its chosen checkpoints' $C$ usage (does $G$ move $C$ more than it moves the probes?);
4. consider $C(y_{real})$ as the task target in the paired variant: a **Phase 4 ablation**, not changed now.

## Phase 3 task loss (fixed now)

- **L1 on fractions, summed over the 6 classes, averaged over the batch:** $\mathcal{L}_{task} = \frac{1}{B}\sum_b \sum_k |C(G(x_b))_k - c^*_{b,k}|$. The scale is fixed now because $\lambda_{task}$ depends on it [audit 8].
- **Note:** L1 on fractions weights T and monocytes most (DC and Other barely count), while comp R² weights each class 1/6. This mismatch is **accepted and recorded**: it keeps the task loss less aligned with the score's probe. KL is a Phase 4 ablation.

## Freezing and artifacts [audit 6]

- **Weights:** `data/models/phase2/{C_bulk,C_sc}_seed{0,1,2}.pt` and the marker probes (gitignored).
- **Canonical hash** (file hashes aren't stable across torch saves): SHA-256 over the sorted state-dict keys, then for each tensor its name, dtype, shape and raw little-endian bytes.
- **`results/phase2/models.json`** (committed) holds for each artifact:
  - the canonical hash;
  - the config;
  - the gene list hash and the class list;
  - the pairs-file content hash;
  - the training commit and notebook source hash;
  - the chosen epoch count;
  - the selection table reference.
- **`dgm.models.load_frozen(name)`:**
  - verifies the canonical hash, gene order, class list and pairs-file hash, and **raises on any mismatch**;
  - returns modules with `requires_grad=False` and `eval()`;
  - stays in eval even if a parent module calls `.train()` (override `train()` to a no-op).
- **Reuse:** the notebook loads existing weights unless `FORCE = True`, so re-running doesn't retrain. Training is deterministic on CPU, so `FORCE` reproduces the same canonical hashes.

## Code layout

- `dgm/models.py`: `CompositionNet`, `Ensemble`, `canonical_hash`, `save_frozen`, `load_frozen`.
- `dgm/composition.py`:
  - inner folds;
  - grid training and selection;
  - PGD and ε50 bisection (closed-form ε50 for linear probes);
  - transfer, natural-direction reference and δ character;
  - the marker probe;
  - the paired donor bootstrap of a comp R² difference.

  Shared because Phase 3 re-runs the attack and the bootstrap.
- `notebooks/phase2_composition_nets.ipynb`: runs it and keeps results, with `setup(notebook=...)`.
- The first Phase 2 commit also updates the README and overview checklist to mark Phase 1 done.

## Tests [audit tests]

`tests/test_models.py` and `tests/test_composition.py`, known-answer:
- **Output and fitting:** the net output is on the simplex; ensemble too.
- **Data guard and folds:** the standardization statistics come from the training part only; the training functions refuse non-train rows; inner folds are donor-disjoint and age-stratified.
- **Learning and determinism:** a planted linear-softmax composition is recovered on synthetic data; CPU training is deterministic (same canonical hash twice).
- **Canonical hash and loader:**
  - the canonical hash is stable across save → load → save, and changes after a one-ulp edit to one weight;
  - `load_frozen` rejects a hash, gene-order, class-list or pairs-hash mismatch;
  - it leaves no trainable parameters, stays in eval under a parent's `.train()`, and passes gradients to inputs while parameter grads stay None.
- **PGD and ε50:**
  - PGD respects $\|\delta\| \le \epsilon s$ and $x + \delta \ge 0$ exactly;
  - closure is 0 at ε = 0 and non-decreasing in ε (up to restart noise);
  - **[audit2 D]** on a **single-output** linear model, where ε50 = 0.5·|c_t − ĉ| / ‖w‖ / s exactly, the bisected ε50 matches within tolerance. The inputs sit far inside the positive orthant so the x + δ ≥ 0 projection never binds;
  - on a **multi-output** linear model, PGD ε50 ≤ closed form × 1.02 (it's an upper bound; the 2% allows for PGD being a heuristic);
  - **[audit2 A]** at initialization (before any training step), the model's prediction equals the training mean composition, for every grid arm.
- **Transfer and noise:**
  - transfer is 0 for a perturbation in the probe's null space;
  - planted low-variance exploitable feature: with `full` scaling it's exploitable, with `center` or `floor` it isn't. This replaces the v1 noise-reduces-closure test, which the audit showed doesn't hold on real data.
- **Bootstrap:** the paired-bootstrap CI contains 0 for two identical predictors and excludes 0 for a clearly worse one.

## Acceptance (Phase 2 sign-off)

1. Both frozen ensembles meet the val pass criterion (paired bootstrap), or the author has decided.
2. Selection, robustness and exploitability ran in the inner folds only. The code reads no val rows before freezing (checked by reading the code and by the split guard).
3. A config passing the exploitability constraint was found, or the author has decided. The val attack metrics and the gradient-masking check are reported.
4. The marker probe is frozen and its val comp R² reported. The Phase 3 rules (marker agreement, reference flag, transfer re-run) are recorded in the notebook.
5. Canonical hashes in `models.json`; `load_frozen` verifies them; tests pass; results byte-reproducible; provenance clean.

## Fallbacks if no config passes (for the author to choose)

1. Accept the least-exploitable config and add a probe-disagreement penalty to the Phase 3 task loss.
2. Use the task nets on noise-augmented $G$ outputs (randomized smoothing) inside the task loss.
3. Use the marker probe (its gradients only move markers) as the task net, with a third probe for evaluation.

## Effort

- Grid: 3 scalings × 2 architectures × 2 losses × 3 weight decays × (2 dropouts for h = 256) × 2 noise settings = 108 configs per domain.
- Each config trains in the 5 inner folds with one seed, so 540 small CPU trainings per domain.
- The attack runs on ≤ 10 configs per domain.
- Estimated at a few hours of compute in total, within the Oct 12–18 window.

## Deviations made during coding (for the Phase 2 audit)

1. **PGD step schedule.**
   - The spec fixed the step at 2.5 ε s / 200. The code instead decays the step linearly from 2 × 2.5 ε s / steps to 0 (the same total path, 2.5 ε s) and keeps the best iterate per sample.
   - Reason, from the known-answer tests:
     - Fixed steps overshoot when ε is much larger than needed, so a sample with exact ε50 0.003 came out as ε50 = ∞.
     - An accept-only-if-better variant stalls at the L1 loss's kinks; more steps made it worse.
   - The decaying schedule recovers the exact single-output ε50 (0.2%), stays within the multi-output closed-form bound at 200 and 1,000 steps, and reaches g ≥ 0.5 at the bound.
   - Everything else about the attack (budgets, targets, restarts, projections, bisection) is as specified.

2. **ε50 search direction.**
   - The spec bisected log ε over [1e-3, 2] starting from the ends. The code scans upward from 1e-3, doubling until PGD first succeeds, then bisects log ε inside that factor-of-2 bracket (8 halvings, within about 0.3%).
   - Reason: at large budgets the first PGD steps can saturate the softmax, the gradient vanishes and PGD stalls at δ = 0. A planted exploitable model closed 98% of the gap at ε = 0.1 and 0% at ε = 1–2, so testing the top of the range first returned ε50 = ∞ for the most exploitable models.
   - The true optimum's success is monotone in ε, so the first success from below brackets correctly.
   - Each budget's PGD is **warm-started** from the best δ found at the previous, smaller budget (feasible at the larger one), in the scan, the bisection and the ε grid. Without it, PGD at moderate budgets still saturated and lost ground: 2 of 40 planted samples came out ∞ although one was 91% closed at ε = 0.1.
3. **Closure reported over the ε grid** is the monotone envelope: the best over budgets ≤ ε, since a δ feasible at a smaller budget is feasible at a larger one. Alongside it is an independent cold run at each budget (superseded wording; see deviation 5).
   - The 1,000-step check at ε = 1 still uses the raw attack at that budget, so a model where PGD stalls there is flagged, which is what that check is for.
4. **Floor scaling's protection depends on the data.** It divides by max(SD, median SD). In the single-cell training data, 74% of panel genes have SD < 0.05 and the median SD is 0.03, so floor gives little protection there; in bulk, 17% are below 0.05 and the median is 0.09. The selection's exploitability constraint, not the scaling choice, is what guards against this. The known-answer test plants near-constant genes as a minority, where floor does protect.

**Changes after the pre-run code audit (2026-10-06, by a separate Claude session):**

5. **Per-budget raw closure.**
   - The warm-started run can't fall below the envelope, so the old "raw" column always equalled it.
   - `attack_summary` now reports `median_g_cold_this_budget`, a cold PGD at that budget alone, next to the envelope.
6. **The probes' closed-form ε50 is now reported** on val (it had been omitted). It's an upper bound only while x + δ ≥ 0 doesn't bind.
7. **Train-only guard.**
   - The freeze goes through `freeze_nets(pairs, idx, ...)`, which refuses non-train rows (as the spec's [audit 9] requires).
   - The selection functions get their data only from `prepare_folds`, which has the same guard.
8. **Sanity check on selection candidates.**
   - The masking check (cold PGD at ε = 1, 1,000 steps) now also runs on every candidate in selection, and is reported with ε50.
   - PGD's ε50 is an upper bound on the true value, so a weak attack would inflate it.
   - A flag is reported for investigation, not as an automatic failure, as the spec says.
9. **Cache keys.**
   - The cache key now includes the torch, sklearn and numpy versions and the thread count.
   - The exploitability cache is also keyed on its candidate list.
   - Cache files are written atomically.
10. **Determinism:** the BLAS/OpenMP threads (numpy, scipy, sklearn) are pinned to 4 as well as torch's.
11. **Phase 1 probe and baselines are refit, not loaded.**
   - The val report uses the Phase 1 evaluator probe and baselines refit on all train with the same code and data. They're deterministic, so this is equivalent to the frozen Phase 1 objects, but it's a deviation from "reused, not refit".
   - The ε scale in selection is each fold's training-part median deviation (fold-only fitting); on val it's the full train set's.
12. **Loaders.**
   - `load_frozen` also checks each member's config against the registry and the member count.
   - The marker probes have a verifying loader, `load_marker`, and their registry entries carry the genes hash and class list.
13. **`l1_task_loss(probs, target)`** gives Phase 3 the task loss on the frozen ensemble's probabilities.
