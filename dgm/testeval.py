"""Phase 7: the test evaluation's bootstrap of hypothesis H1 (spec: notes/phase7_spec.md).

On every donor resample the unpaired frontier is recomputed from scratch: each blend
w * y_a + (1 - w) * y_b of two frontier members (weights 0, 0.1, ..., 1, as dgm.frontier) is
re-scored for centered R^2 and realism on the resampled rows, and each model member's margin
is taken against that resample's frontier. Blends are scored exactly without forming them:

  centered R^2   per sample, the blend's squared error is w^2 G_aa + (1 - w)^2 G_bb + 2 w (1 - w) G_ab,
                 with G the Gram matrix of the members' residuals (y_m - y) for that sample
  realism        the C2ST on the top 10 target principal components of the blend itself, formed in
                 gene space exactly as dgm.frontier.trace forms it (blending the members' PC scores
                 instead differs by rounding, which can flip a tie in the AUROC)

  DirectionData  precomputed per-sample Gram matrices, PC scores and targets of one direction
  resample       frontier points, model members' (R^2, realism) and H1 margins on one resample
  run            the resamples over worker processes (each pinned to the deterministic thread
                 count; every resample's result depends only on its rows, so not on the workers)
  split_rows     the evaluation rows: validation for a dry run, test otherwise (guarded)
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context

import numpy as np
import pandas as pd

from dgm.frontier import WEIGHTS, frontier_margin
from dgm.metrics import DIST_PCS, c2st


def split_rows(pairs, dry: bool) -> np.ndarray:
    """Validation rows for a dry run, test rows otherwise; a dry run can never return test rows."""
    rows = pairs.idx("val" if dry else "test")
    if dry:
        assert not np.isin(rows, pairs.idx("test")).any()
    return rows


@dataclass
class DirectionData:
    """Everything the bootstrap needs for one direction, without the gene-level arrays.
    members: frontier members {name: translations}; models: {model: [member translations]}."""
    members: dict
    models: dict
    target: np.ndarray
    mu: np.ndarray
    pca: object
    donors: np.ndarray
    names: list = field(init=False)

    def __post_init__(self):
        y = np.asarray(self.target, np.float64)
        self.names = sorted(self.members)
        res = np.stack([np.asarray(self.members[n], np.float64) - y for n in self.names])     # members x n x g
        self.gram = np.einsum("ang,bng->nab", res, res)                                    # n x members x members
        self.sst = ((y - self.mu) ** 2).sum(1)
        pcs = lambda a: self.pca.transform(np.asarray(a, np.float64))[:, :DIST_PCS]
        self.z_members = np.stack([pcs(self.members[n]) for n in self.names])
        self.y_members = np.stack([np.asarray(self.members[n]) for n in self.names])           # for blends, in gene space
        self.z_target = pcs(y)
        self.model_r2 = {m: [1 - ((np.asarray(a, np.float64) - y) ** 2).sum(1) / self.sst for a in ys] for m, ys in self.models.items()}
        self.model_z = {m: [pcs(a) for a in ys] for m, ys in self.models.items()}
        del self.members, self.models                              # keep the object small for the workers
        self.target = None


def _realism(z, zt, donors):
    return 1 - 2 * abs(c2st(z, zt, donors) - 0.5)


def resample(dd: DirectionData, idx: np.ndarray, h1_model: str = "M1") -> dict:
    """Frontier points on rows idx (the format of dgm.frontier.trace), every model member's
    (centered R^2, realism), and the H1 model's per-member margins and kinds."""
    idx = np.asarray(idx)
    donors, zt, sst, G = dd.donors[idx], dd.z_target[idx], dd.sst[idx], dd.gram[idx]
    pure = {}
    for i, n in enumerate(dd.names):
        r2 = float(np.median(1 - G[:, i, i] / sst))
        pure[n] = (r2, _realism(dd.z_members[i][idx], zt, donors))
    rows = []
    for (i, a), (j, b) in itertools.combinations(list(enumerate(dd.names)), 2):
        for w in WEIGHTS:
            if w == 1.0:
                r2, real = pure[a]
            elif w == 0.0:
                r2, real = pure[b]
            else:
                sse = w * w * G[:, i, i] + (1 - w) ** 2 * G[:, j, j] + 2 * w * (1 - w) * G[:, i, j]
                r2 = float(np.median(1 - sse / sst))
                blend = w * dd.y_members[i][idx] + (1 - w) * dd.y_members[j][idx]
                real = _realism(dd.pca.transform(np.asarray(blend, np.float64))[:, :DIST_PCS], zt, donors)
            rows.append({"a": a, "b": b, "w": float(w), "r2_centered": r2, "realism": real})
    points = pd.DataFrame(rows)
    models = {m: [(float(np.median(r[idx])), _realism(z[idx], zt, donors)) for r, z in zip(dd.model_r2[m], dd.model_z[m])]
              for m in dd.model_r2}
    out = {"points": points, "pure": pure, "models": models}
    if h1_model in models:
        mk = [frontier_margin(points, r2, real) for r2, real in models[h1_model]]
        out["margins"] = [m for m, _ in mk]
        out["kinds"] = [k for _, k in mk]
    return out


_DD: dict = {}


def _init(dd_by_dir, threads):
    from dgm.composition import deterministic
    deterministic(threads)
    _DD.update(dd_by_dir)


def _work(job):
    d, k, idx = job
    r = resample(_DD[d], idx)
    r.pop("points")                                                 # only the summaries travel back
    return d, k, r


def run(dd_by_dir: dict, boots: list, workers: int = 6, threads: int = 4) -> dict:
    """{direction: [resample summaries in resample order]}; deterministic and independent of `workers`."""
    jobs = [(d, k, b) for d in dd_by_dir for k, b in enumerate(boots)]
    out = {d: [None] * len(boots) for d in dd_by_dir}
    if workers <= 1:
        _init(dd_by_dir, threads)
        results = map(_work, jobs)
        for d, k, r in results:
            out[d][k] = r
        return out
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init,
                             initargs=(dd_by_dir, threads)) as ex:
        for d, k, r in ex.map(_work, jobs, chunksize=4):
            out[d][k] = r
    return out


# ---------------------------------------------------------------- pre-stated reading rules (prereg; notes/phase7_spec.md)
def h1_wording(lo: float, hi: float) -> str:
    """H1 per direction, from the margin's 95% CI."""
    if lo > 0:
        return "supported on halving 0"
    if hi < 0:
        return "inside the unpaired frontier"
    return "not distinguishable from the unpaired frontier"


def h1_qualifier(lo: float, halving1_margin: float) -> str:
    """"not robust to the training split" when H1 is supported but the halving-1 point estimate is at or below 0."""
    return "not robust to the training split" if (lo > 0 and halving1_margin <= 0) else ""


def ci_word(lo: float, hi: float) -> str:
    return "better" if lo > 0 else "worse" if hi < 0 else "not distinguishable"


def superseded_wording(acc_ci: tuple, real_ci: tuple) -> str:
    """The per-metric H1 superseded on 2026-10-07, in that version's pre-stated words."""
    wa, wr = ci_word(*acc_ci), ci_word(*real_ci)
    if wa == wr == "better":
        return "would have been supported"
    if wr == "better":
        return f"better realism than any unpaired baseline; accuracy {wa}"
    if wa == "better":
        return f"better accuracy than any unpaired baseline; realism {wr}"
    return f"would not have been supported (realism {wr}; accuracy {wa})"


def suspect(gain_marker: float, gain_other: float) -> bool:
    """Phase 3's rule: a composition gain under another readout that the marker probe does not confirm by at least half."""
    return bool(gain_other > 0 and gain_marker < 0.5 * gain_other)
