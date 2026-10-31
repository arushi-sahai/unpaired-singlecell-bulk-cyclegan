"""The accuracy-realism frontier of the unpaired baselines, the comparator for hypothesis H1.

Blending the outputs of two translators adds no information, so every blend
w * y_i + (1 - w) * y_j (log1p(CP10k) space) of two members of the unpaired set is something an
unpaired method gets for free. The set holds the unpaired baselines and the training-target mean
(a constant output): blends with the mean are the baselines shrunk toward it, which raises
centered R^2 without adding information, so shrinkage alone can never beat the frontier.

Each pair's blends are scored on a grid of weights (w = 0, 0.1, ..., 1) for accuracy (centered
R^2) and realism (1 - 2 |C2ST AUROC - 0.5|), and the path between neighbouring grid points is
interpolated linearly, since blends are continuous in w. The frontier at accuracy r is the best
realism reachable on any path with accuracy at least r:

    F(r) = max { realism(p) : p on a blend path, R^2(p) >= r }

A model at (r, s) beats the frontier by the margin s - F(r) (a realism difference). If no point
on any path is as accurate as the model, F(r) is undefined and the margin is the model's
accuracy lead over the most accurate point instead (an R^2 difference); ``frontier_margin`` says
which kind it returned.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

WEIGHTS = tuple(np.round(np.linspace(0, 1, 11), 2))
MEAN = "training mean"   # name of the constant member


def trace(outputs: dict, score, weights=WEIGHTS) -> pd.DataFrame:
    """Score every blend of every pair of members: score(y) -> (centered R^2, realism).
    Each pure member is scored once; every pair's path lists all its weights, pure ends
    included (w = 1 is member a, w = 0 is member b), in increasing w."""
    names = sorted(outputs)
    pure = {n: score(outputs[n]) for n in names}
    rows = []
    for a, b in itertools.combinations(names, 2):
        for w in weights:
            if w == 1.0:
                r2, real = pure[a]
            elif w == 0.0:
                r2, real = pure[b]
            else:
                r2, real = score(w * outputs[a] + (1 - w) * outputs[b])
            rows.append({"a": a, "b": b, "w": float(w), "r2_centered": float(r2), "realism": float(real)})
    return pd.DataFrame(rows)


def _segments(points: pd.DataFrame):
    """(r2_start, realism_start, r2_end, realism_end) for every segment between neighbouring
    weights on each pair's path."""
    for _, path in points.sort_values("w").groupby(["a", "b"], sort=False):
        r, s = path["r2_centered"].to_numpy(), path["realism"].to_numpy()
        for i in range(len(r) - 1):
            yield r[i], s[i], r[i + 1], s[i + 1]


def frontier_realism(points: pd.DataFrame, r: float) -> float:
    """F(r): the best realism on any (interpolated) blend path at accuracy >= r; NaN if none.
    On a straight segment the maximum subject to R^2 >= r is at an end or where R^2 = r."""
    best = -np.inf
    for r0, s0, r1, s1 in _segments(points):
        for rr, ss in ((r0, s0), (r1, s1)):
            if rr >= r:
                best = max(best, ss)
        lo, hi = min(r0, r1), max(r0, r1)
        if lo < r < hi:
            best = max(best, s0 + (s1 - s0) * (r - r0) / (r1 - r0))
    return float(best) if np.isfinite(best) else float("nan")


def frontier_margin(points: pd.DataFrame, r2: float, realism: float) -> tuple[float, str]:
    """(margin, kind): realism - F(r2) with kind "realism" if some path point is as accurate;
    otherwise r2 minus the best path accuracy, with kind "accuracy". Positive = beyond the frontier."""
    f = frontier_realism(points, r2)
    if np.isnan(f):
        return float(r2 - points["r2_centered"].max()), "accuracy"
    return float(realism - f), "realism"


def envelope(points: pd.DataFrame, n: int = 400) -> pd.DataFrame:
    """The frontier F(r) on a grid of n accuracies spanning the blend paths, for plotting and tables."""
    rs = np.linspace(points["r2_centered"].min(), points["r2_centered"].max(), n)
    return pd.DataFrame({"r2_centered": rs, "realism": [frontier_realism(points, r) for r in rs]})
