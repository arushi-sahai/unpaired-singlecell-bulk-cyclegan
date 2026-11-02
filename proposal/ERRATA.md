# Errata: submitted proposal (2026-09-30)

`proposal_final.tex` is the source as submitted and stays unchanged. These are the known errors in it. The code and results use the corrected versions.

## 1. LSGAN adversarial loss (math error)

**Submitted (Approach, Eq. 1):**

> The $S \to B$ direction is optimized as $\min_{G_{S \to B}} \max_{D_B}$:
> $\mathcal{L}_{\text{adv}}^{S \to B} = \mathbb{E}_{x_B}[(D_B(x_B)-1)^2] + \mathbb{E}_{x_S}[D_B(G_{S \to B}(x_S))^2]$

**What's wrong:** that expression is the discriminator's own LSGAN loss, which $D_B$ *minimizes*, not maximizes. If the generator minimized it, the generator would push $D_B(G(x_S))$ toward 0 (the "fake" label), which helps the discriminator. LSGAN (Mao et al. 2017) is not a single minimax objective; it is two separate least-squares losses.

**Correct:**

$$\mathcal{L}_{D_B} = \mathbb{E}_{x_B}[(D_B(x_B)-1)^2] + \mathbb{E}_{x_S}[D_B(G_{S \to B}(x_S))^2] \qquad \text{(minimized by } D_B)$$

$$\mathcal{L}_{G}^{S \to B} = \mathbb{E}_{x_S}[(D_B(G_{S \to B}(x_S))-1)^2] \qquad \text{(minimized by } G_{S \to B})$$

$B \to S$ mirrors these with $D_S$, $G_{B \to S}$. In the full objective $\mathcal{L} = \mathcal{L}_{\text{adv}} + \lambda_{\text{cyc}}\mathcal{L}_{\text{cyc}} + \lambda_{\text{task}}\mathcal{L}_{\text{task}}$ (minimized by the generators), $\mathcal{L}_{\text{adv}} = \mathcal{L}_G^{S\to B} + \mathcal{L}_G^{B\to S}$. The discriminators are updated separately on $\mathcal{L}_{D_B} + \mathcal{L}_{D_S}$. CycleGAN's code optimizes it this way. Many implementations also use $\tfrac12$ factors on the D loss; that doesn't matter here.

**Status:** found after submission (2026-09-30); the proposal was not resubmitted. The report and the code use the corrected form.

## 2. Pair count

The proposal says "∼840 pairs". The exact count is **836** pairs (868 single-cell samples, 836 with a same-draw bulk) from 96 donors. This comes from our own preprocessing of the raw files (`notebooks/phase0_preprocessing.ipynb`), which uses all donors. The GAN's own split is 582 / 135 / 119 pairs.

## 3. Figure 1 panel titles (cosmetic)

The panel titles baked into `proposal_fig1.png` say "SoundLife"; the caption says "AIFI Sound Life". The figure is kept as submitted.

## 4. Sound Life citation (first author)

**Submitted:** `\cite{gustafson2025soundlife}`, with the BibTeX author field "Gustafson, Claire E. and others" (`proposal/main.bib`), and a source comment saying the lead author is Gustafson, not Gong.

**Correct:** the first author is Qiuyu Gong; Claire E. Gustafson is the last and corresponding author (PubMed 41162704; Nature 648:696–706, 2025; doi:10.1038/s41586-025-09686-5). Cite as Gong et al. 2025. The report uses the key `gong2025soundlife` (`report/references.bib`).

## 5. Data license

**Submitted:** the data are described as "public, CC BY".

**Correct:** the Human Immune System Explorer portal and the DataCite record for doi:10.57785/bn17-3h16 state no license for the Sound Life data (checked 2026-11-02). The portal asks users to cite Gong et al. (2025) and Gustafson et al., "Longitudinal Dynamics of Health and Age" (doi:10.57785/bn17-3h16). The data are not redistributed in this repository.

## Not an error, but worth stating precisely in the report

- **What $c^*$ is.** $c^*$ is the AIFI_L1 composition of the PBMC single-cell sample (computed from all post-QC slim cells, ~3,000 per sample: 9 measured classes, merged to 6 targets, with Platelet, Erythrocyte, Progenitor and ILC as "Other"). For the bulk side, it is the composition of the *mononuclear fraction* of that draw, not of whole blood, which also holds granulocytes. So $C_{\text{bulk}}$ learns to read PBMC-relative proportions out of a whole-blood profile. That is learnable, but the proposal's "one invariant ground-truth composition vector supervises both directions" should say this.
- **Figure 1 is in CytoSet-embedding space.** The GAN works in gene space. In gene space on our processed panel the uncorrected silhouette is 0.78 (PC1 η² domain 0.99). Final ComBat/CORAL/Harmony numbers come from a recompute in the author's earlier, unpublished pipeline. Say which space a number comes from.

# Deviations from the proposal (design changes, not errors)

The proposal was a starting point. These are the changes made since, each with its reason; the report lists them.

1. **Unpaired training is the primary setting.** The proposal trained the translator on all paired samples. Because paired single-cell and bulk data from the same draw is rare, and pairs make regression a strong competitor that a GAN has to justify itself against, the models are now trained on donor-disjoint halves of the training donors (single-cell from one half, bulk from the other), and the pairs are used only to evaluate. Models form a ladder of pairing: strictly unpaired (headline), weakly paired, and a fully paired reference.
2. **The composition task loss, and the proposal's stated contribution.** The proposal supervised both directions with the measured composition $c^*$. For a bulk sample, $c^*$ exists only through its paired single-cell sample, so that loss uses pairs. The weakly paired model keeps the proposal's design through $C_{bulk}$ (trained on pairs) and a pseudo-label for the bulk → single-cell direction. A pair-free readout (reference deconvolution) did not meet its pre-stated accuracy criterion (Phase 2, §4.5), so the strictly unpaired model has no composition task loss. The proposal's stated mathematical contribution, the task-consistency term, therefore moves from the headline model to the weakly paired variant, where it is tested directly (does pair-derived composition supervision improve composition?). The project's contribution is the controlled study of what pairs, and supervision derived from them, add to unpaired adversarial translation.
3. **The downstream evaluation.** The proposal's CytoSet classifier and CytoSet-embedding space are replaced by gene-space metrics (accuracy, realism by a classifier two-sample test, composition by a probe) and an exploratory age classifier on the target principal components. The proposed ablation with a CytoSet task loss is dropped.
4. **How results are judged.** A selection score chooses checkpoints; claims are made per pre-specified hypothesis, on test, with paired donor-bootstrap confidence intervals (Phase 1, §3.6–3.7).
5. **Composition classes.** The 9 measured AIFI_L1 classes are merged into 6 targets (T, monocyte, B, NK, DC, and "Other" for the four classes under 0.5% of cells).
