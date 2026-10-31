# Phase 3a pilot: strictly unpaired CycleGAN (exploratory, 2026-10-07)

Not a result for the report: one configuration, validation only, best checkpoints chosen with hindsight. Its purpose was to see whether unpaired adversarial training works on about 290 samples per domain, and to fix design choices before the Phase 3 grid. The Phase 3 code was written fresh from the amended spec.

**Setup.** Halving 0 (single-cell from half A, bulk from half B). CycleGAN per `notes/phase3_spec.md` (draft before amendment): residual MLP generators, LSGAN, λ_cyc = 10, Adam 2e-4, batch 64, instance noise annealed over the first half, 10,000 steps, CPU 4 threads, validation every 500 steps with the Phase 1 evaluators.

| Run | Single-cell → bulk (centered R², realism) | Bulk → single-cell (centered R², realism) |
|---|---|---|
| Start: mean shift (halves) | −0.093, 0.154 | −1.750, 0.575 |
| Spectral-norm MLP discriminator, 2 seeds, final | −0.207 / −0.226, 0.662 / 0.702 | −0.985 / −0.883, 0.516 / 0.551 |
| + dropout 0.3, final | −0.303, 0.632 | −0.990, 0.536 |
| Discriminator on top 50 PCs, final | −1.544, 0.026 | −6.464, 0.136 |
| Shuffled-label control, final | −3.564, 0.131 | −12.968, 0.000 |
| Start: ComBat (halves) | −0.520, 0.417 | −0.007, 0.123 |
| ComBat start, spectral-norm MLP, 2 seeds, final | −0.630 / −0.618, 0.610 / 0.689 | −0.149 / −0.166, 0.261 / 0.529 |

**Against the unpaired frontier** (Phase 1 §3.8: blends of mean shift, ComBat, CORAL and the training-target mean, interpolated; halving 0; validation). Best single checkpoint / best three-checkpoint average (the selection rule):
- single-cell → bulk: +0.078 / −0.028 and +0.042 / −0.057 (mean-shift start, 2 seeds); −0.095 / −0.111 and −0.090 / −0.134 (ComBat start);
- bulk → single-cell: −0.031 / −0.039 and −0.007 / −0.025 (mean-shift start); −0.080 / −0.122 and +0.024 / −0.026 (ComBat start).

A first version of the frontier (single baselines only, step function on the 0.1 grid) put the best checkpoints at +0.14 to +0.15 in single-cell → bulk; it was too low, because shrinking toward the training mean and blends between grid weights were missing. The pilot GAN sits on the corrected frontier, not beyond it.

**Conclusions.**
- Adversarial training works with a spectral-norm MLP discriminator: it trades a small loss of accuracy for a large gain in realism from either start, and the shuffled-label control is far worse, so the gain comes from the adversarial signal.
- The principal-component discriminator fails (the generator drifts outside its 50-dimensional view); dropout does not help.
- The start matters by direction: mean shift for single-cell → bulk, ComBat for bulk → single-cell.
- Realism fluctuates by about ±0.1 between checkpoints and was still rising at 10,000 steps; one run's discriminator reached 100% accuracy.
- The per-metric H1 could not be met by any trade-off; H1 was changed to the frontier margin (`notes/phase3_plan.md`, Amendments). Against the corrected frontier the untuned pilot does not beat it, so Phase 3 has to find a configuration that does, or report that adversarial training reaches but does not pass what blending the unpaired baselines gives for free.
