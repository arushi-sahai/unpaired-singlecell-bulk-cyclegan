"""Phase 6: analysis of the locked and Phase 5 models (spec: notes/phase6_spec.md). Validation only.

  frontier_members   the unpaired frontier's members (mean shift, ComBat, CORAL, training-target
                     mean) fit on the training halves, as translations of given sources
  matched_blend      the blend at a model's accuracy: on every blend path whose centered R^2
                     crosses the model's, the weight solving R^2(blend) = R^2(model) by bisection
                     on the actual blended translations, realism re-scored there; the best and
                     second-best crossings by realism (6.2)
  alignment          Delta_i = <r_i / |r_i|, y_i - mean of the other donors' targets>, the
                     sample-specific alignment of a residual r (6.2)
  info_free_map      per-gene least squares of a generator's training-half outputs on the
                     mean-shift, ComBat and CORAL translations of the same inputs (+ intercept),
                     and its share of the generator's validation variance explained (6.2)
  gap_closure        g = 1 - |R(y_a) - c*|_1 / |R(y_b) - c*|_1 per sample, and its summaries (6.1)
  noise              the three steganography noise conditions (6.6)
  sensitivity        |F(G(x) + e) - F(G(x))| / |F(y + e) - F(y)| per sample (6.6)
  other_draw         another draw of the same donor (the Phase 1, Table 1.9 construction; 6.5)
  similarity         cosine of centered profiles to the true partner, the same donor's other
                     draws and other donors, per sample (6.5; Phase 0 and Evaluator's measure)
  donor_boot         donor-bootstrap resamples (row indices) of a set of samples
  ci                 point estimate and 95% percentile interval of a statistic over resamples
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from dgm.frontier import MEAN, WEIGHTS


# ---------------------------------------------------------------- bootstrap helpers
def donor_boot(donors: np.ndarray, n_boot: int = 1000, seed: int = 0) -> list[np.ndarray]:
    """Row indices of n_boot donor resamples (all of a resampled donor's rows come along); the
    same construction as the Phase 4 and 5 notebooks, so a seed gives the same resamples."""
    donors = np.asarray(donors)
    groups = [np.flatnonzero(donors == d) for d in pd.unique(donors)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)]


def ci(stat, n: int, boots: list[np.ndarray], level: float = 0.95) -> tuple[float, float, float]:
    """(estimate, low, high) of stat(idx) on all n rows and over the resamples; NaN resamples are dropped."""
    est = float(stat(np.arange(n)))
    vals = np.array([stat(b) for b in boots], float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return est, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return est, float(lo), float(hi)


# ---------------------------------------------------------------- 6.2: the matched blend
def frontier_members(pairs, rows_s: np.ndarray, rows_b: np.ndarray, src: np.ndarray, direction: str) -> dict:
    """{name: translations of src} for the frontier's members fit on the training halves
    (dgm.ablation.frontier_points uses the same members on validation sources)."""
    from dgm.baselines import CORAL, ComBat, MeanShift
    s_all, t_all = (pairs.x_sc, pairs.x_bulk) if direction == "S->B" else (pairs.x_bulk, pairs.x_sc)
    s_idx, t_idx = (rows_s, rows_b) if direction == "S->B" else (rows_b, rows_s)
    out = {m.name: m.fit(s_all[s_idx], t_all[t_idx]).translate(src) for m in (MeanShift(), ComBat(), CORAL())}
    out[MEAN] = np.tile(t_all[t_idx].mean(0), (len(src), 1))
    return out


def matched_blend(members: dict, r2_fn, realism_fn, target: float, tol: float = 1e-5, weights=WEIGHTS) -> dict:
    """The blend at accuracy `target`. Each pair (a, b) of sorted member names traces
    y(w) = w * y_a + (1 - w) * y_b over `weights`; on every segment where R^2(y(w)) - target changes
    sign (or is 0 at an end), w is found by bisection to |R^2 - target| <= tol and realism is
    re-scored there. Returns {"best", "second", "crossings"}: best = the crossing with the highest
    realism (ties: the lower-index pair, then the smaller w); second = the best crossing on another
    path (None if only one path crosses); both None if no path crosses."""
    names = sorted(members)
    cross = []
    for pi, (a, b) in enumerate(itertools.combinations(names, 2)):
        blend = lambda w, a=a, b=b: w * members[a] + (1 - w) * members[b]
        ws = np.asarray(weights, float)
        h = np.array([r2_fn(blend(w)) - target for w in ws])
        found = []
        for i in range(len(ws) - 1):
            if h[i] == 0:
                found.append(ws[i])
            elif h[i] * h[i + 1] < 0:
                lo, hi, hlo = ws[i], ws[i + 1], h[i]
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    hm = r2_fn(blend(mid)) - target
                    if abs(hm) <= tol:
                        break
                    if np.sign(hm) == np.sign(hlo):
                        lo, hlo = mid, hm
                    else:
                        hi = mid
                found.append(mid)
        if h[-1] == 0:
            found.append(ws[-1])
        for w in found:
            y = blend(w)
            cross.append({"pair": pi, "a": a, "b": b, "w": float(w), "r2": float(r2_fn(y)), "realism": float(realism_fn(y)), "y": y})
    if not cross:
        return {"best": None, "second": None, "crossings": []}
    order = sorted(cross, key=lambda c: (-c["realism"], c["pair"], c["w"]))
    best = order[0]
    second = next((c for c in order[1:] if c["pair"] != best["pair"]), None)
    return {"best": best, "second": second, "crossings": order}


def alignment(r: np.ndarray, y: np.ndarray, donors: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Delta_i = <r_i / |r_i|, y_i - ybar_{-i}>, ybar_{-i} the mean of the targets of the other donors.
    With `basis` (orthonormal rows, e.g. target PCs) both vectors are projected onto it first.
    NaN where |r_i| = 0."""
    donors = np.asarray(donors)
    ybar = np.vstack([y[donors != d].mean(0) for d in donors])
    v, rr = y - ybar, np.asarray(r, float)
    if basis is not None:
        v, rr = v @ basis.T, rr @ basis.T
    nr = np.linalg.norm(rr, axis=1)
    out = np.full(len(rr), np.nan)
    ok = nr > 0
    out[ok] = (rr[ok] * v[ok]).sum(1) / nr[ok]
    return out


def info_free_map(g_train: np.ndarray, reg_train: list, g_val: np.ndarray, reg_val: list) -> float:
    """Share of the generator's validation output variance explained by a per-gene affine
    combination of baseline translations: for each gene, least squares of g_train on
    [1, reg_1, ..., reg_k] (training-half sources; minimum-norm solution, since mean shift and
    ComBat are both affine in the input gene and can be collinear), applied to the validation
    regressors. Returns 1 - SSE / SST, SST about each gene's validation mean of g_val."""
    n, g = g_train.shape
    pred = np.empty_like(g_val, dtype=float)
    for j in range(g):
        A = np.column_stack([np.ones(n)] + [r[:, j] for r in reg_train])
        beta = np.linalg.lstsq(A, g_train[:, j], rcond=None)[0]
        pred[:, j] = np.column_stack([np.ones(len(g_val))] + [r[:, j] for r in reg_val]) @ beta
    sse = ((g_val - pred) ** 2).sum()
    sst = ((g_val - g_val.mean(0)) ** 2).sum()
    return float(1 - sse / sst)


# ---------------------------------------------------------------- 6.1: gap closure
def gap_closure(read_a: np.ndarray, read_b: np.ndarray, target: np.ndarray) -> np.ndarray:
    """g = 1 - |read_a - c*|_1 / |read_b - c*|_1 per sample (a = the model, b = its reference);
    NaN where the reference is already at the target."""
    num, den = np.abs(read_a - target).sum(1), np.abs(read_b - target).sum(1)
    out = np.full(len(num), np.nan)
    ok = den > 0
    out[ok] = 1 - num[ok] / den[ok]
    return out


def closure_summary(read_a, read_b, target) -> dict:
    """Median g, the share with g > 0.5 (as in Phase 2), and the ratio-of-sums version."""
    g = gap_closure(read_a, read_b, target)
    den = np.abs(read_b - target).sum()
    return {"median g": float(np.nanmedian(g)), "share g > 0.5": float(np.mean(g[np.isfinite(g)] > 0.5)),
            "ratio-of-sums g": float(1 - np.abs(read_a - target).sum() / den) if den > 0 else float("nan")}


def transfer_ratio(g_probe: float, g_c: float, guard: float = 0.1) -> float:
    """median g under the probe / median g under C; NaN ("no closure under C") if g_C <= guard."""
    return float(g_probe / g_c) if g_c > guard else float("nan")


# ---------------------------------------------------------------- 6.6: steganography
def noise(kind: str, kappa: float, n: int, basis: np.ndarray, pc_sd: np.ndarray, gene_sd: np.ndarray, rng) -> np.ndarray:
    """Gaussian noise vectors (n x genes):
      "top50"       in the span of `basis` (orthonormal rows), per-PC SD kappa * pc_sd
      "complement"  isotropic Gaussian projected onto the orthogonal complement of `basis`, each
                    row rescaled to the norm of an independent "top50" draw (same total norm)
      "per-gene"    per-gene SD kappa * gene_sd (genes with SD 0 get no noise)"""
    if kind == "top50":
        return (rng.standard_normal((n, len(basis))) * kappa * pc_sd) @ basis
    if kind == "complement":
        ref = np.linalg.norm(rng.standard_normal((n, len(basis))) * kappa * pc_sd, axis=1)
        z = rng.standard_normal((n, basis.shape[1]))
        z -= (z @ basis.T) @ basis
        return z / np.linalg.norm(z, axis=1, keepdims=True) * ref[:, None]
    if kind == "per-gene":
        return rng.standard_normal((n, len(gene_sd))) * kappa * gene_sd
    raise ValueError(kind)


def sensitivity(F, gx: np.ndarray, y: np.ndarray, e: np.ndarray) -> np.ndarray:
    """|F(G(x) + e) - F(G(x))| / |F(y + e) - F(y)| per sample, F a callable on arrays (no clamping)."""
    num = np.linalg.norm(F(gx + e) - F(gx), axis=1)
    den = np.linalg.norm(F(y + e) - F(y), axis=1)
    return num / np.maximum(den, 1e-12)


# ---------------------------------------------------------------- 6.5: donor-level reference
def other_draw(donors: np.ndarray, seed: int = 0) -> np.ndarray:
    """For each sample, another draw of the same donor (seeded; Phase 1, Table 1.9)."""
    donors = np.asarray(donors)
    assert all((donors == x).sum() > 1 for x in donors), "a donor has a single draw"
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(np.flatnonzero((donors == donors[i]) & (np.arange(len(donors)) != i))) for i in range(len(donors))])


def similarity(y_hat_c: np.ndarray, y_c: np.ndarray, donors: np.ndarray, exclude: np.ndarray | None = None) -> pd.DataFrame:
    """Per sample, the cosine of centered profiles (inputs already centered) to its true partner,
    the median to the same donor's other draws and the median to other donors' targets. `exclude`
    (one target index per row) drops that target from the row's comparison sets, e.g. the real
    draw a same-donor reference was copied from."""
    from dgm.metrics import cosine_matrix
    donors = np.asarray(donors)
    S = cosine_matrix(y_hat_c, y_c)
    same = donors[:, None] == donors[None, :]
    np.fill_diagonal(same, False)
    other = donors[:, None] != donors[None, :]
    if exclude is not None:
        rows = np.arange(len(S))
        same[rows, exclude] = False
        other[rows, exclude] = False
    return pd.DataFrame({"partner": np.diag(S),
                         "same donor, other draws": [np.median(S[i, same[i]]) if same[i].any() else np.nan for i in range(len(S))],
                         "other donors": [np.median(S[i, other[i]]) for i in range(len(S))]})
