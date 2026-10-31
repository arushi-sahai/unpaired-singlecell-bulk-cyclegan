# Phase 0 figure captions

Keyed by figure file name. Each caption is shown under its figure in `notebooks/phase0_preprocessing.ipynb` and reused by the report.

## 0.1_cell_qc

**Per-cell quality-control metrics of the subsampled single cells.** Histograms over all 2,586,999 cells in the 868 single-cell samples (up to 3,000 cells per sample): (a) genes detected per cell and (b) counts per cell, both on a log₁₀ scale, and (c) the percentage of a cell's counts from the 13 mitochondrial genes. Dashed lines mark the quality-control thresholds: at least 200 genes detected and less than 20% mitochondrial counts. Every cell passes both thresholds, consistent with the provider's files already being filtered, so quality control removes no cells and leaves the composition unchanged.

## 0.2_samples_donors

**Samples and donors of the Sound Life cohort.** (a) Cells per single-cell sample in the subsample (log-scale count of samples). All but 9 of the 868 samples contribute 3,000 cells; the 9 smaller samples contribute every cell they have (minimum 984). (b) Number of paired draws per donor for the 96 donors (median 10, range 1–10). (c) Samples by age at draw, for the 868 single-cell pseudobulk samples and the 836 whole-blood bulk samples. The cohort has two age groups, young (25–37 years) and older (55–67 years), with no donors aged 38–54 (shaded).

## 0.3_composition

**Measured PBMC composition of the 836 paired samples, by donor age group.** Box plots of the fraction of peripheral blood mononuclear cells (PBMC) in each of the six target classes: T cells (T), monocytes (Mono), B cells (B), natural killer cells (NK), dendritic cells (DC), and Other, which pools platelets, erythrocytes, progenitor cells, and innate lymphoid cells. Fractions are computed from all cells of each single-cell sample that pass quality control. Boxes span the interquartile range, the black line marks the median, and whiskers extend to 1.5 times the interquartile range; outliers are not drawn. The y axis is logarithmic, and samples with no cells in a class are drawn at 10⁻⁴. Sample counts per age group are in the legend. T cells make up most of PBMC in both age groups, and the composition differs little between them.

## 0.5_bulk_qc

**Library quality of the 836 whole-blood bulk samples.** (a) Library size, (b) genes detected, (c) percentage of counts from mitochondrial genes, and (d) percentage of counts from hemoglobin (globin) genes, per sample kit; the 2 kits sequenced twice are shown with their libraries summed. The dashed line in (a) marks the 1-million-count threshold; every library lies above it (minimum 1,220,532 counts). The globin fraction is at most 2.1%, so globin transcripts do not dominate any library.

## 0.7_expression

**Expression of the 5,000 panel genes in the two domains.** (a) Distribution of nonzero log(1 + CP10k) values (counts per 10,000, log-transformed) in single-cell pseudobulk and whole-blood bulk, over the 836 paired samples. (b) Mean expression of each panel gene in pseudobulk (x) against bulk (y); the dashed line is equality. The eight genes with the largest excess in bulk and the four with the largest excess in pseudobulk are numbered and named in the key. Genes far above the line are granulocyte genes (for example CXCR2, CSF3R, FCGR3B) and erythroid genes (ALAS2, HBB), which whole blood contains and PBMC lacks; genes below it are ribosomal protein genes.

## 0.8_panel

**Selection of the 5,000-gene panel on the training donors.** Normalized dispersion against mean expression for the 13,775 genes detected in at least 50% of training samples in both domains, computed separately on the 582 training samples of (a) single-cell pseudobulk and (b) whole-blood bulk. Dispersion is the variance-to-mean ratio of a gene, normalized within bins of mean expression (the Seurat flavor of highly variable gene selection in scanpy). Each gene is ranked by normalized dispersion within each domain, the two ranks are averaged, and the 5,000 best genes form the panel (colored); the other candidates are gray. No validation or test donor is used.

## 0.9_markers_vs_composition

**Marker expression tracks measured composition in both domains.** For T cells, monocytes, and B cells, the marker gene with the highest Spearman rank correlation (ρ) with its lineage's fraction in single-cell pseudobulk (named on the y axis), plotted against that measured fraction for the 836 paired samples from 96 donors. (a–c) Expression in single-cell pseudobulk; (d–f) expression of the same gene in whole-blood bulk from the same draws. Expression is log(1 + CP10k); fractions are of PBMC. Each panel gives Spearman's ρ across the 836 pairs. Agreement is strong in pseudobulk and weaker but clear in whole blood, so the PBMC composition can be read from a bulk profile; the NK and dendritic-cell markers follow the same pattern.

## 0.10_granulocyte_genes

**Granulocyte genes are expressed in whole-blood bulk and nearly absent from PBMC pseudobulk.** Mean expression (dot) ± one standard deviation (bars) across the 836 paired samples, in log(1 + CP10k), for seven granulocyte genes (above the line) and three lineage markers as controls (below: CD3E for T cells, MS4A1 for B cells, CD14 for monocytes), in single-cell pseudobulk and whole-blood bulk. Five granulocyte genes are 2.26–2.77 units higher in bulk; the more weakly expressed CMTM2 and MME are 0.95 and 0.59 units higher. PBMC isolation removes granulocytes, so single-cell → bulk translation must generate this signal without any counterpart in its input.

## 0.11_domain_gap

**The gap between single-cell pseudobulk and whole-blood bulk lies on one axis, and pairing carries sample-level signal.** (a) The first two principal components (PCs) of a principal component analysis fit to the 1,672 panel profiles of both domains (836 paired samples from 96 donors in each). PC1 explains 82.0% of the variance and separates the domains completely. (b) η², the share of each of the top 10 PCs' variance explained by domain or by donor age group (young, 25–37 years, or older, 55–67 years). Domain explains 99.0% of PC1 and at most 0.4% of PCs 2–5; age group explains at most 2.9% of any of the top five. The domain silhouette on the top 10 PCs is 0.777. (c) Centered correlation between a single-cell profile and a bulk profile, after each domain's per-gene mean is removed, for a sample's own pair (n = 836), for other draws of the same donor, and for draws of other donors. The medians are 0.269, 0.214, and −0.003. A sample's own bulk profile is its best match among the 836 for 17.5% of samples (median rank 8). Most of the pairing signal is shared across a donor's draws, and a smaller part is specific to the draw.
