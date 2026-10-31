# Phase 3 spec: translation models (final)

Hypotheses and budget: `notes/phase3_plan.md`.

## Code
- `dgm/translation.py`: generators, discriminators, the training loop, checkpointing. `tests/test_translation.py`: known-answer tests (below). Notebook: `notebooks/phase3_translation.ipynb`.
- Training reads **only** halving-0 rows of the training split (single-cell from half A, bulk from half B); a guard refuses any other row, as `freeze_nets` does in Phase 2.

## Architecture (both directions)
- **Generator:** residual MLP with a per-gene affine skip, $G(x) = \mathrm{ReLU}(a \odot x + f(x))$, $f$: 5000 → 512 → 512 → 5000 with LayerNorm and LeakyReLU(0.2); $a$ is learnable. Each generator starts at the unpaired baseline that is most accurate in its direction (Phase 1, §4.4): single-cell → bulk at mean shift ($a = 1$, the last layer's bias = the halves' per-gene mean shift), bulk → single-cell at ComBat ($a = 1/\delta$ and the bias set so $a \odot x + b$ equals ComBat fit on the halves). The last layer's weights start at 0, so training starts exactly at those baselines (and $C_S$ never sees raw bulk; Phase 2 §5, Implications).
- **Discriminator:** MLP 5000 → 256 → 64 → 1 with spectral normalization and LeakyReLU(0.2), no dropout; Gaussian instance noise on its inputs (SD annealed from 0.1 to 0 over the first half of training). The pilot rejected a discriminator on the top 50 principal components (outputs drift outside its subspace; realism near 0) and found dropout 0.3 no better.

## Objective
- $\mathcal{L}_{D} = \mathbb{E}[(D(y)-1)^2] + \mathbb{E}[D(G(x))^2]$ per domain; $\mathcal{L}_{G}^{adv} = \mathbb{E}[(D(G(x))-1)^2]$ per direction (LSGAN, as corrected in `proposal/ERRATA.md`).
- Cycle: $\lambda_{cyc}\,(\lVert G_{B\to S}(G_{S\to B}(x_S)) - x_S\rVert_1 + \lVert G_{S\to B}(G_{B\to S}(x_B)) - x_B\rVert_1)$, mean over genes and samples.
- M2 only: the task loss of `notes/phase3_plan.md`, weight $\lambda_{task}$, ramped linearly from 0 after a warm-up of 20% of training.
- Optional identity loss, $\lambda_{id} = 0.5\,\lambda_{cyc}$ (part of the configuration budget).

## Training
- Adam, learning rate 2e-4 (generators and discriminators), β = (0.5, 0.999); batch 64, single-cell and bulk minibatches drawn independently each step.
- 20,000 steps (realism was still rising at 10,000 in the pilot); validation scored every 500 steps with the Phase 1 evaluators; the checkpoint is the one with the largest mean margin beyond the validation frontier (`notes/phase3_plan.md`), not the selection score, whose accuracy term is constant at 0 for unpaired models.
- CPU, deterministic algorithms, 4 threads (as in Phase 2); seeds 0–2.
- Logged every 100 steps: each loss term, discriminator accuracy on real and translated batches, output diversity.

## Configuration grid (within the budget of 12 settings × 3 seeds per model)
$\lambda_{cyc} \in \{5, 10, 20\}$ × discriminator learning rate ∈ {2e-4, 5e-5} × identity loss on/off = 12 settings for M1; for M2, the best M1 setting × $\lambda_{task} \in \{0.1, 1, 10\}$, plus the two nearest M1 settings × the same three = 9 settings. Each setting runs seeds 0–2.

## Known-answer tests
- At initialization, $G_{S\to B}$ equals the mean-shift baseline and $G_{B\to S}$ equals `ComBat().fit` on the halves (`translate`), each within 1e-5 on validation inputs (zero residual weights); the cycle loss at initialization equals that of the composition mean shift ∘ ComBat (and ComBat ∘ mean shift).
- One training step changes generator and discriminator weights; the train-only guard rejects validation rows.
- Deterministic: two runs with the same seed give identical weights (canonical hash).
- Synthetic unpaired recovery. Two domains differ by a per-gene affine map with positive scales, on skewed, correlated marginals; the generator starts with scale $a = 1$ and zero bias (away from the answer). Assert M1's centered R² is at least ComBat's minus 0.05 on the same data (one-sided) (a per-gene affine map is exactly what ComBat recovers from marginals, and CycleGAN is not identifiable in general, so matching ComBat is the right bar). An exploratory case adds a low-rank cross-gene component, which ComBat cannot capture; it is reported with no threshold.
- The task loss of M2 is zero when $C_B$ reads the source composition exactly (planted case).

## Amendments

**2026-10-07, after the pilot** (`notes/phase3_pilot.md`): per-direction initialization at the most accurate unpaired baseline with a learnable per-gene scale (was: mean shift in both directions, fixed scale 1), because the ComBat start lifted bulk → single-cell accuracy from about −0.9 to about −0.1 while the GAN still raised its realism; spectral-norm MLP discriminator without dropout, the principal-component discriminator dropped; 20,000 steps instead of 10,000; checkpoint rule changed with H1. The configuration grid is unchanged (its discriminator learning rate factor addresses the discriminator overpowering seen in one pilot run).

**2026-10-11, after the pre-run code audit (before any Phase 3 run):**
- *M2 checkpoint eligibility.* With a task loss, only checkpoints at or after (warm-up + ramp) × steps (step 8,000) are eligible. Reason: until the warm-up ends an M2 run is bit-identical to the M1 run with the same setting and seed, so a margin-maximizing rule could choose an M1 checkpoint and H2 would compare M1 with itself.
- *Task-loss ramp.* 0 until 20% of training, then linear to full weight at 40% (the spec gave no ramp length).
- *M2 grid.* "The best M1 setting plus the two nearest" is implemented as the three best M1 settings by validation margin.
- *Selection details.* The initialization and the last evaluation are never eligible; margins are averaged over the two directions first, then with the neighboring evaluations.
- *Synthetic recovery test* runs at hidden width 64 (40 genes). At width 512 the affine case drifts with training (0.921 at 3,000 steps, 0.870 at 10,000, 0.853 at 20,000, no selection), so Sound Life runs rely on checkpoint selection against the frontier; recorded in the Phase 3 notebook, §3.2.
- *Negative control* under the final setup: the chosen M1 setting with shuffled discriminator targets, seed 0.
- *Logging:* discriminator accuracy on real and on translated batches separately; diverged runs (non-finite outputs) are recorded as failed and never selected.
