"""Reference deconvolution: a composition readout for bulk profiles that uses no pairs.

Cell-type signatures are built from labeled single cells; a profile is modeled as a
non-negative mix of the signatures on a set of marker genes, solved by non-negative least
squares (NNLS; Lawson & Hanson 1974), and the weights are normalized to fractions.

  signatures       per-class mean profile over cells, on the panel genes. ``counts`` (raw UMI
                   counts per cell, the default) makes the NNLS weights proportional to cell
                   numbers, because cells with more mRNA contribute more to a mixture; ``cp10k``
                   (per-cell normalized) makes them mRNA-weighted.
  bulk_high_genes  genes whose mean in bulk exceeds the mean in pseudobulk by more than a fixed
                   fold change: whole-blood signal (granulocytes, red cells) that PBMC signatures
                   can't explain. Uses only each domain's marginal means, never pairs.
  select_markers   per class, the top-k genes by fold change over the highest other class.
  Deconvolver      NNLS on the markers, each gene scaled by its mean signature value so highly
                   expressed genes don't dominate; optionally a non-negative column holding the
                   same expression level in every marker gene absorbs unexplained background.

Inputs to ``bulk_high_genes`` and ``Deconvolver.predict`` are on the linear CP10k scale
(expm1 of the log1p(CP10k) profiles in the pairs file).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import nnls


def signatures(counts, labels: np.ndarray, classes: list, gene_idx: np.ndarray, kind: str = "counts") -> np.ndarray:
    """(genes x classes) mean profile per class over cells. counts: cells x all genes, raw UMIs;
    gene_idx: the columns of the panel genes. kind: "counts" or "cp10k" (normalized over all genes)."""
    assert kind in ("counts", "cp10k"), kind
    X = sp.csr_matrix(counts, dtype=np.float64)
    labels = np.asarray(labels)
    if kind == "cp10k":
        X = sp.diags(1e4 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)) @ X
    X = X[:, gene_idx]
    out = []
    for c in classes:
        rows = labels == c
        assert rows.any(), f"no cells of class {c}"
        out.append(np.asarray(X[rows].mean(0)).ravel())
    return np.stack(out, 1)


def bulk_high_genes(bulk_mean: np.ndarray, pseudobulk_mean: np.ndarray, log2fc: float = 1.0) -> np.ndarray:
    """Boolean mask of genes with log2((bulk + 1) / (pseudobulk + 1)) > log2fc (linear CP10k means)."""
    return np.log2((np.asarray(bulk_mean) + 1) / (np.asarray(pseudobulk_mean) + 1)) > log2fc


def select_markers(sig_cp10k: np.ndarray, exclude: np.ndarray, k: int = 50, min_expr: float = 0.5) -> np.ndarray:
    """Union over classes of the top-k genes by log2 fold change over the highest other class,
    among genes not excluded and with at least ``min_expr`` CP10k in the class. Sorted indices."""
    picked = set()
    for j in range(sig_cp10k.shape[1]):
        other = np.delete(sig_cp10k, j, 1).max(1)
        lfc = np.log2((sig_cp10k[:, j] + 0.1) / (other + 0.1))
        ok = ~np.asarray(exclude) & (sig_cp10k[:, j] >= min_expr)
        order = np.argsort(-np.where(ok, lfc, -np.inf), kind="stable")
        picked |= set(order[:min(k, int(ok.sum()))].tolist())
    return np.array(sorted(picked))


class Deconvolver:
    """Fractions of each signature class in a profile, by NNLS on the marker genes."""

    def __init__(self, signature: np.ndarray, markers: np.ndarray, residual: bool = False):
        self.markers = np.asarray(markers)
        S = signature[self.markers]
        self.scale_ = 1.0 / np.maximum(S.mean(1), 1e-12)
        M = S * self.scale_[:, None]
        self.M_ = np.hstack([M, self.scale_[:, None]]) if residual else M   # a constant level in every marker gene
        self.k_ = signature.shape[1]

    def weights(self, Y: np.ndarray) -> np.ndarray:
        """Raw NNLS weights (samples x classes [+ residual])."""
        Y = np.atleast_2d(np.asarray(Y, np.float64))[:, self.markers] * self.scale_
        return np.array([nnls(self.M_, y)[0] for y in Y])

    def predict(self, Y: np.ndarray) -> np.ndarray:
        """Fractions over the signature classes (rows sum to 1; the residual column is dropped)."""
        W = self.weights(Y)[:, :self.k_]
        return W / np.maximum(W.sum(1, keepdims=True), 1e-12)
