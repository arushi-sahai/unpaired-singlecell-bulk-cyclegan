"""Phase 6, amendment A2: entropic optimal transport as a non-adversarial unpaired translator
(spec: notes/phase6_spec.md).

  pca_basis   top principal components of the training-half targets (the cost space)
  sinkhorn    log-domain Sinkhorn for entropic OT with uniform weights; returns the potentials
  EntropicOT  fit on unpaired training halves: the GAN's start map s, costs in the targets' top
              PCs, potentials at regularization eps x median cost. Two maps of a new source x:
                barycentric  T(x) = sum_j w_j(x) y_j  (OT-b), the conditional mean of the plan
                sample       y_j with probability w_j(x)  (OT-s), a conditional draw
              with w_j(x) proportional to exp((g_j - c(s(x), y_j)) / eps), the entropic
              out-of-sample extension (the uniform log b_j term cancels in the normalization).

Outputs are convex combinations (OT-b) or copies (OT-s) of real training-half targets in the
full gene space; small eps approaches copying the nearest target, large eps the target mean.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

EPS_GRID = (0.01, 0.03, 0.1, 0.3)       # x the median cost
N_PCS = 50


def pca_basis(y: np.ndarray, k: int = N_PCS) -> tuple[np.ndarray, np.ndarray]:
    """(mean, components) of the top k PCs of y; components are orthonormal rows."""
    mu = y.mean(0)
    _, _, vt = np.linalg.svd(y - mu, full_matrices=False)
    return mu, vt[:k]


def sq_costs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)


def sinkhorn(cost: np.ndarray, eps: float, tol: float = 1e-9, max_iter: int = 100000) -> tuple[np.ndarray, np.ndarray, int]:
    """Potentials (f, g) of entropic OT between uniform weights on the rows and columns of `cost`:
    plan P_ij = exp((f_i + g_j - C_ij) / eps) / (n m). Log-domain updates until both marginals are
    within `tol` (max absolute error, in probability units). Returns (f, g, iterations)."""
    n, m = cost.shape
    la, lb = -np.log(n) * np.ones(n), -np.log(m) * np.ones(m)
    f, g = np.zeros(n), np.zeros(m)
    for it in range(1, max_iter + 1):
        f = -eps * logsumexp((g[None, :] - cost) / eps + lb[None, :], axis=1)
        g = -eps * logsumexp((f[:, None] - cost) / eps + la[:, None], axis=0)
        if it % 10 == 0 or it == max_iter:
            logp = (f[:, None] + g[None, :] - cost) / eps + la[:, None] + lb[None, :]
            err = max(np.abs(np.exp(logsumexp(logp, axis=1)) - 1 / n).max(), np.abs(np.exp(logsumexp(logp, axis=0)) - 1 / m).max())
            if err < tol:
                return f, g, it
    raise RuntimeError(f"Sinkhorn did not reach tolerance {tol} in {max_iter} iterations (eps {eps})")


def plan(f, g, cost, eps) -> np.ndarray:
    n, m = cost.shape
    return np.exp((f[:, None] + g[None, :] - cost) / eps) / (n * m)


class EntropicOT:
    """Entropic OT from start-mapped sources to targets (unpaired). `start` maps raw sources to the
    target domain (the GAN's start map); `eps_rel` is eps as a multiple of the median training cost."""

    def __init__(self, eps_rel: float, start, n_pcs: int = N_PCS, tol: float = 1e-9):
        self.eps_rel, self.start, self.n_pcs, self.tol = eps_rel, start, n_pcs, tol

    def _z(self, x_target_space: np.ndarray) -> np.ndarray:
        return (x_target_space - self.mu_) @ self.V_.T

    def fit(self, src: np.ndarray, tgt: np.ndarray) -> "EntropicOT":
        self.y_ = np.asarray(tgt, np.float64)
        self.mu_, self.V_ = pca_basis(self.y_, self.n_pcs)
        self.zy_ = self._z(self.y_)
        cost = sq_costs(self._z(self.start(src)), self.zy_)
        self.eps_ = float(self.eps_rel * np.median(cost))
        self.f_, self.g_, self.iters_ = sinkhorn(cost, self.eps_, self.tol)
        self.cost_ = cost
        return self

    def weights(self, src: np.ndarray) -> np.ndarray:
        """w_j(x) for each new source (rows sum to 1)."""
        logw = (self.g_[None, :] - sq_costs(self._z(self.start(src)), self.zy_)) / self.eps_
        return np.exp(logw - logsumexp(logw, axis=1, keepdims=True))

    def translate(self, src: np.ndarray) -> np.ndarray:
        """OT-b: the barycentric projection."""
        return self.weights(src) @ self.y_

    def sample(self, src: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """OT-s: one training target per source, drawn from w(x). Returns (outputs, chosen indices)."""
        w = self.weights(src)
        rng = np.random.default_rng(seed)
        u = rng.random(len(w))[:, None]
        j = np.minimum((np.cumsum(w, axis=1) < u).sum(1), w.shape[1] - 1)
        return self.y_[j], j
