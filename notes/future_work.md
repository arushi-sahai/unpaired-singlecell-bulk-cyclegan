# Future work: other model families on the same unpaired problem (from 2026-10-11)

For the author and the report's future-work paragraph. Not part of the planned study; nothing here changes Phase 3.

## The question

Does the result of Phase 3 (the CycleGAN sits at or just beyond the unpaired accuracy–realism frontier) belong to the GAN, or to the unpaired setting? If every model family lands near the same frontier, the limit is information (the two marginals do not identify which output belongs to which input); if one family clears it, the limit was the model.

## Candidate families

| Family | Unpaired translation method | Expected strength | Note |
|---|---|---|---|
| GAN (this project) | CycleGAN | realism for a small loss of accuracy | Phase 3 |
| Diffusion | DDIB (one diffusion model per domain, encode to noise with one, decode with the other); SDEdit (partial noising) | realism; the noising level is a built-in accuracy–realism dial, so the model traces a curve, not a point | data-hungry: train in principal-component space (top 50–100 PCs) at ~300 samples per domain |
| Bridges / flow matching | unpaired Schrödinger bridge (UNSB), flow matching with OT couplings | principled "move as little as possible" transport | |
| Optimal transport | neural OT (e.g. CellOT) | accuracy near mean shift | likely close to the frontier by construction |
| Shared-latent VAE | scGen-style latent shift; cross-domain VAE | standard single-cell tool | baseline family |

**If time in Phase 4:** a small SDEdit/DDIB translator in PC space, as (or next to) the stochastic-generator stretch goal. Scored like everything else, plus the energy score.

## What a fair comparison needs

**Same inputs and budget**
- Same training rows (the donor-disjoint halves; both halvings), same validation and test pairs, pairs used only as the answer key.
- Same tuning budget: the number of settings × seeds × checkpoints compared on validation, reported per family. Selection by the same frontier-margin rule; the test set used once.
- Compute reported per family (wall-clock, parameters); matched where feasible.

**Same scoring space**
- Every model is scored in gene space (log1p CP10k, 5,000 genes) by the Phase 1 evaluators.
- A family that works in PC space pays for the projection: add "PCA round-trip of the baselines" to the frontier members, so the frontier isn't beaten by the projection itself (PCA smoothing alone can change realism).

**Curves, not points**
- Families with a trade-off dial (SDEdit noise level, OT regularization, cycle weight) are compared as curves against the frontier: the area of the curve above the frontier, and the margin at fixed accuracies (e.g. at mean shift's accuracy).

**Stochastic models**
- Accuracy from the mean of K samples; realism from single samples (a mean of samples is unrealistically smooth); the energy score reported for both. Say which is used in each plot.

## What data would be needed

- **More paired donors for evaluation.** 14 test donors is the binding constraint on the CIs; the differences between families are likely smaller than between GAN and baseline. A cohort with paired PBMC single-cell and whole-blood bulk from many more donors (ideally ≥ 50 held out), same assays.
- **Larger unpaired pools in each domain**, to ask whether the gap to the frontier changes with unpaired sample size (diffusion should gain most). Best from the same cohort or protocol; pooling external cohorts (e.g. large single-cell-only or bulk-only blood cohorts) adds a cohort shift that would need its own baseline (a frontier refit with the extra data).
- **A data-scaling design:** unpaired N per domain ∈ {50, 150, 300, all} × pairs ∈ {0, few, all}, each family at each cell; the ladder of pairing crossed with data size.
- **A synthetic benchmark with known truth:** bulk simulated from the single-cell profiles with a known per-sample mixing (including granulocytes) and known gene-specific biases, with and without cross-gene structure. Separates "the marginals don't identify the map" from "the model can't learn it".
- **Repeated draws per donor** (as in Sound Life) for the within-donor metrics.

**The most informative single experiment:** the data-scaling grid, run on the synthetic benchmark and on Sound Life. If every family stays near the frontier at every unpaired sample size, the limit is the unpaired setting, not the GAN.

## Getting enough data to train a generative model

The scarce unit is the **sample (donor or draw)**, not the cell. Cells are abundant (public atlases hold tens of millions); samples with both measurements number in the hundreds. Sample-to-sample variation, which is what translation must preserve, can only come from more samples, and nothing below substitutes for more donors in the *evaluation*. Options, roughly by how much real information they add:

1. **Model cells, aggregate to samples.** Treat each sample as a set of cells; train the generator at the cell level (about 3,000 cells per draw here, so ~1M training cells instead of ~300 pseudobulks) and form pseudobulk by summation. Single-cell → bulk then becomes: cell-level model, aggregation, and a learned whole-blood correction (granulocytes, and the bulk assay's gene-specific biases). Donor-grouped splits stay mandatory. **Limit:** only the single-cell side has cells; the bulk side stays at ~300 samples, and bulk → single-cell would have to generate a whole cell population from one bulk vector (a set-generation problem close to deconvolution, whose pair-free version failed here, Phase 2 §4.5). So it mainly strengthens the single-cell model, not the bulk one. Out of scope for this project (new model class and data pipeline); stated as future work in the report.
2. **Pretrain on public data, fine-tune on the cohort.**
   - Single cell: atlas-scale data (CELLxGENE Census, OneK1K: ~1.27M PBMCs from ~1,000 donors) and the foundation models trained on it (scGPT, Geneformer, scFoundation). Use their latent space as the space the generator works in, or as a frozen encoder.
   - Bulk: uniformly reprocessed public bulk RNA-seq (ARCHS4, recount3), including GTEx whole blood. Pretrain a bulk prior, fine-tune on the cohort's half.
   - Cost: a cohort shift between public data and the cohort, which the baselines and frontier must be refit to see.
3. **Simulate.** Mix single-cell profiles from atlases at sampled compositions (as deconvolution methods do for their training sets), optionally adding granulocyte profiles, to make unlimited semi-synthetic bulk. Good for pretraining and for the known-truth benchmark; the gap between simulated and real bulk is itself part of what translation has to learn, so simulation can't replace real bulk.
4. **Resample within samples** (several pseudobulks per draw from subsets of its cells). Cheap augmentation that regularizes against cell-sampling noise; it adds no donor variation, so it inflates N without adding information about between-sample differences.
5. **Make the model need less data.** Work in a low-dimensional space (PCs, gene programs, a pretrained latent), use residual or structured generators that start at a baseline (as here), and keep the parameter count matched to ~300 samples.
6. **Collect paired data cheaply, where possible.** Split one blood draw into a PBMC single-cell aliquot and a whole-blood bulk aliquot; even a few dozen extra paired donors would sharpen evaluation more than any augmentation.

In this project: option 5 is already used (baseline starts; PC space for a diffusion add-on); option 4 is proposed for Phase 4 as part of the data-scaling experiment below; options 1–3 and 6 are future work (option 1 named in the report's future work).

## Proposed for Phase 4: does the GAN need more donors or just more samples?

Exploratory, validation only, no hypothesis; to be specified and added to the plan (dated amendment) before any Phase 4 run. The chosen M1 setting, 3 seeds, halving 0; all baselines and the frontier refit on each reduced training set.

| Arm | What varies | Levels | Question |
|---|---|---|---|
| Donors | donors per half, all their draws kept | ¼, ½, all | does accuracy/realism/margin fall as donors fall? |
| Draws | draws per donor, all donors kept, matched to the Donors arm's sample counts | ¼, ½, all | same sample count, more donors: better? |
| Bagging | extra single-cell pseudobulks per draw from random cell subsets (single-cell side only; bulk has no cells) | ×1, ×4, ×16 | do synthetic extra samples help? |

Reading: at matched sample count, if Donors falls faster than Draws, donors are what matters; if Bagging helps, the GAN was sample-starved in a way resampling can fix; if neither helps, the bottleneck is not data size at this scale. Limit: we can only go down from the current data, so this shows the trend, not what more data would do. About 18 runs (6 cells × 3 seeds), roughly 5 hours at 6 workers; bagging needs pseudobulks rebuilt from the slim files with the Phase 0 pipeline.

## Evaluation data (separate from training data)

Evaluation must stay real paired data; augmentation and simulation cannot substitute. The way to tighter conclusions is another cohort with paired PBMC single-cell and whole-blood bulk from many more donors. To look for: public or collaborator cohorts with both assays on the same blood draws.
