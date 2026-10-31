"""Phase 4: helpers for the ablations of the locked models (spec: notes/phase4_spec.md).

  subset_rows        reduced training halves for the data-size ablation: half of each half's
                     donors (all their draws, age-stratified), or all donors with fewer draws,
                     matched to the same sample count
  frontier_points    the unpaired accuracy-realism frontier (Phase 1, section 3.8) refit on any
                     pair of training row sets: mean shift, ComBat, CORAL and the training-target
                     mean, blended pairwise and scored on validation
  half_nets          the composition networks retrained on half A only (ablation B3)
  c2st_alt           alternative realism measures for the metric-robustness check (P1)
  move_decomposition the move of a translation from its start, split along the needed correction
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUBSET_KINDS = ("donors", "draws")


# ---------------------------------------------------------------- data-size subsets
def _donor_half(pairs, rows: np.ndarray, frac: float, rng) -> np.ndarray:
    """Rows of round(frac * n) donors from each age group of `rows`, all their draws."""
    donors = pairs.donor_ids[rows]
    keep = []
    for g in np.unique(pairs.age_group[rows]):
        ds = np.unique(donors[pairs.age_group[rows] == g])
        k = max(1, int(round(frac * len(ds))))
        keep += list(rng.choice(ds, k, replace=False))
    return np.sort(rows[np.isin(donors, keep)])


def _draw_matched(pairs, rows: np.ndarray, n: int, rng) -> np.ndarray:
    """Every donor of `rows`, each with a share of its draws, n rows in total: draws per donor in
    proportion to the donor's count (at least 1), then single draws added or removed at random to
    hit n exactly."""
    donors = pairs.donor_ids[rows]
    ds = np.unique(donors)
    n_d = np.array([(donors == d).sum() for d in ds])
    k = np.maximum(1, np.round(n_d * n / len(rows)).astype(int))
    while k.sum() > n:
        i = rng.choice(np.flatnonzero(k > 1)); k[i] -= 1
    while k.sum() < n:
        i = rng.choice(np.flatnonzero(k < n_d)); k[i] += 1
    keep = [rng.choice(rows[donors == d], kk, replace=False) for d, kk in zip(ds, k)]
    return np.sort(np.concatenate(keep))


def subset_rows(pairs, half_a: np.ndarray, half_b: np.ndarray, kind: str, seed: int, frac: float = 0.5):
    """(rows of half A, rows of half B) for one subset draw. kind "donors": round(frac * n) donors
    per age group in each half, all their draws. kind "draws": all donors, with draws removed so
    that each half has exactly as many rows as the "donors" subset of the same seed."""
    assert kind in SUBSET_KINDS, kind
    out = []
    for h, rows in enumerate((np.asarray(half_a), np.asarray(half_b))):
        don = _donor_half(pairs, rows, frac, np.random.default_rng([seed, h, 0]))
        out.append(don if kind == "donors" else _draw_matched(pairs, rows, len(don), np.random.default_rng([seed, h, 1])))
    return tuple(out)


# ---------------------------------------------------------------- frontier on any training rows
def frontier_points(pairs, rows_s: np.ndarray, rows_b: np.ndarray, evaluators: dict, va: np.ndarray, classes,
                    directions=("S->B", "B->S")) -> dict:
    """{direction: frontier points} for unpaired baselines fit on single-cell rows rows_s and bulk
    rows rows_b (Phase 1, section 3.8; the same members, weights and scoring)."""
    from dgm.baselines import CORAL, ComBat, MeanShift
    from dgm.frontier import MEAN, trace
    out = {}
    for d in directions:
        src, tgt = (pairs.x_sc, pairs.x_bulk) if d == "S->B" else (pairs.x_bulk, pairs.x_sc)
        s_idx, t_idx = (rows_s, rows_b) if d == "S->B" else (rows_b, rows_s)

        def acc_real(y, d=d, tgt=tgt):
            m, _ = evaluators[d].score(y, tgt[va], pairs.composition[va], pairs.age_group[va], pairs.donor_ids[va], classes)
            return m["r2_centered"], 1 - 2 * abs(m["c2st_auroc"] - 0.5)
        members = {m.name: m.fit(src[s_idx], tgt[t_idx]).translate(src[va]) for m in (MeanShift(), ComBat(), CORAL())}
        members[MEAN] = np.tile(tgt[t_idx].mean(0), (len(va), 1))
        out[d] = trace(members, acc_real)
    return out


# ---------------------------------------------------------------- B3: composition networks on half A
def half_nets(pairs, half_a: np.ndarray, halving: int, out_dir, registry):
    """Retrain the locked Phase 2 composition networks (same configuration, epochs and seeds) on
    half A's training rows only, save them as C_bulk_halfA{halving} / C_sc_halfA{halving} and
    return their registry entries. C_B uses half A's bulk pairs, C_S half A's single-cell profiles."""
    import json
    from pathlib import Path
    from dgm import paths
    from dgm.composition import freeze_nets
    from dgm.models import list_hash, pairs_hash, save_frozen
    reg2 = json.loads((paths.results_dir(2) / "models.json").read_text())
    entries = {}
    for name, domain in (("C_bulk", "bulk"), ("C_sc", "sc")):
        cfg, epochs = reg2[name]["config"], reg2[name]["epochs"]
        nets, mean, inv, prior = freeze_nets(pairs, np.asarray(half_a), domain, cfg, epochs)
        entries[name] = save_frozen(f"{name}_halfA{halving}", nets, cfg, mean, inv, prior, Path(out_dir), Path(registry),
                                    {"epochs": epochs, "classes": pairs.l1_classes.tolist(), "genes_hash": list_hash(pairs.genes.tolist()),
                                     "pairs_hash": pairs_hash(pairs), "trained_on": f"half A of halving {halving} ({len(half_a)} rows)",
                                     "half_a_rows_hash": list_hash(np.sort(half_a).tolist())})
    return entries


# ---------------------------------------------------------------- P1: alternative realism measures
def _pcs(evaluator, y, k):
    return evaluator.pca_.transform(y)[:, :k]


def c2st_alt(kind: str, evaluator, y_hat: np.ndarray, y: np.ndarray, donors: np.ndarray) -> float:
    """C2ST AUROC of translations against their paired real targets under an alternative classifier,
    with the evaluator's donor-grouped 5-fold CV:
      "logistic50"  the H1 classifier (logistic regression on PCs and their squares) on the top 50 PCs
      "boosted50"   gradient-boosted trees on the top 50 PCs (fixed settings, seeded)"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from dgm.metrics import c2st
    if kind == "logistic50":
        return c2st(_pcs(evaluator, y_hat, 50), _pcs(evaluator, y, 50), donors)
    assert kind == "boosted50", kind
    z = np.vstack([_pcs(evaluator, y_hat, 50), _pcs(evaluator, y, 50)])
    lab = np.r_[np.ones(len(y_hat)), np.zeros(len(y))]
    grp = np.r_[donors, donors]
    proba = np.zeros(len(z))
    for tr, te in GroupKFold(n_splits=min(5, len(np.unique(donors)))).split(z, lab, grp):
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1, random_state=0)
        proba[te] = clf.fit(z[tr], lab[tr]).predict_proba(z[te])[:, 1]
    return float(roc_auc_score(lab, proba))


def c2st_alt_null(kind: str, evaluator, y_hat: np.ndarray, y: np.ndarray, donors: np.ndarray, n_perm: int = 50,
                  seed: int = 0) -> tuple[float, float]:
    """Mean and SD of c2st_alt when, within each pair, which profile is "translated" is swapped at
    random (the paired null, as dgm.metrics.c2st_null)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        swap = rng.random(len(y_hat)) < 0.5
        a, b = np.where(swap[:, None], y, y_hat), np.where(swap[:, None], y_hat, y)
        vals.append(c2st_alt(kind, evaluator, a, b, donors))
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


# ---------------------------------------------------------------- P0: signal or spread
def move_decomposition(start: np.ndarray, out: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    """Per sample: the move delta = out - start against the needed correction r = target - start.
    Columns: |delta|, |r|, cos(delta, r), the length of delta along r (signed) and across it."""
    delta, r = out - start, target - start
    nd, nr = np.linalg.norm(delta, axis=1), np.linalg.norm(r, axis=1)
    along = (delta * r).sum(1) / np.maximum(nr, 1e-12)
    return pd.DataFrame({"move": nd, "needed": nr, "cos": along / np.maximum(nd, 1e-12), "along": along,
                         "across": np.sqrt(np.maximum(nd ** 2 - along ** 2, 0))})
