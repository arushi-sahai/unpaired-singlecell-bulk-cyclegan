"""Known-answer checks for dgm.frontier: python -m tests.test_frontier"""
import numpy as np
import pandas as pd

from dgm.frontier import MEAN, envelope, frontier_margin, frontier_realism, trace


def path(a, b, pts):
    """A pair's path from (w, r2, realism) triples."""
    return pd.DataFrame([{"a": a, "b": b, "w": w, "r2_centered": r, "realism": s} for w, r, s in pts])


def test_trace_scores_each_member_once_and_lists_full_paths():
    calls = []
    def score(y):
        calls.append(float(y.mean()))
        return float(y.mean()), float(1 - y.mean())
    out = {"a": np.zeros((4, 2)), "b": np.ones((4, 2)), "c": 2 * np.ones((4, 2))}
    t = trace(out, score)
    assert len(t) == 3 * 11                                   # every pair's full path, ends included
    assert len(calls) == 3 + 3 * 9                            # pure members scored once, interior weights per pair
    ab = t[(t.a == "a") & (t.b == "b")].sort_values("w")
    assert np.allclose(ab["r2_centered"], 1 - ab["w"])        # w * a + (1 - w) * b has mean 1 - w


def test_frontier_interpolates_between_grid_weights():
    pts = path("x", "y", [(0.0, -0.6, 0.8), (0.5, -0.3, 0.6), (1.0, -0.1, 0.2)])
    assert frontier_realism(pts, -0.3) == 0.6
    assert abs(frontier_realism(pts, -0.2) - 0.4) < 1e-12     # midway along the segment from (-0.3, 0.6) to (-0.1, 0.2)
    assert abs(frontier_realism(pts, -0.5) - (0.8 - 0.2 / 3)) < 1e-12   # on the segment from (-0.6, 0.8) to (-0.3, 0.6)
    assert np.isnan(frontier_realism(pts, 0.0))
    m, kind = frontier_margin(pts, -0.2, 0.5)
    assert kind == "realism" and abs(m - 0.1) < 1e-12
    m, kind = frontier_margin(pts, 0.05, 0.1)                 # more accurate than any path point
    assert kind == "accuracy" and abs(m - 0.15) < 1e-12
    env = envelope(pts, n=5)
    assert env["realism"].is_monotonic_decreasing


def test_shrinking_toward_the_mean_cannot_beat_the_frontier():
    """Without the mean as a member, a baseline shrunk toward the training mean is more accurate than
    every blend and 'beats' the frontier by the accuracy fallback; with the mean as a member it can't."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=(200, 30))                            # real targets, training mean 0
    mu = np.zeros(30)
    def score(yh):                                             # centered R2 (median) and a realism proxy: spread match
        r2 = np.median(1 - ((yh - y) ** 2).sum(1) / ((y - mu) ** 2).sum(1))
        return r2, float(min(yh.std() / y.std(), 1.0))
    base = {"noisy": y + rng.normal(scale=1.5, size=y.shape), "shuffled": y[rng.permutation(200)]}
    shrunk = 0.3 * base["noisy"]                              # shrink toward the mean: more accurate, less realistic
    r2, real = score(shrunk)
    without = trace(base, score)
    assert frontier_margin(without, r2, real)[1] == "accuracy" and frontier_margin(without, r2, real)[0] > 0
    with_mean = trace({**base, MEAN: np.tile(mu, (200, 1))}, score)
    m, kind = frontier_margin(with_mean, r2, real)
    assert kind == "realism" and m <= 1e-9, (m, kind)          # on or inside the noisy-mean path


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
