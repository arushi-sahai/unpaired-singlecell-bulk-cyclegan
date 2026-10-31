"""Known-answer checks for the Phase 7 bootstrap (notes/phase7_spec.md): python -m tests.test_phase7"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from dgm import analysis as an
from dgm import composition as comp
from dgm import testeval as te
from dgm.frontier import frontier_margin, trace
from dgm.metrics import DIST_PCS, c2st

comp.deterministic()
RNG = np.random.default_rng(0)
N, G = 60, 30
DONORS = np.repeat(np.arange(12), 5)
Y = RNG.normal(size=(N, G)) + 3
TRAIN = RNG.normal(size=(200, G)) + 3
PCA_ = PCA(n_components=12, random_state=0).fit(TRAIN)
MU = TRAIN.mean(0)
MEMBERS = {"mean shift": Y + RNG.normal(scale=0.7, size=(N, G)), "ComBat": Y + RNG.normal(scale=1.0, size=(N, G)) + 0.3,
           "CORAL": 0.5 * Y + RNG.normal(scale=1.3, size=(N, G)) + 1.5, "training mean": np.tile(MU, (N, 1))}
M1 = [Y + RNG.normal(scale=s, size=(N, G)) for s in (0.9, 1.0, 1.1)]


def direct_score(idx):
    def score(y):
        r2 = np.median(1 - ((y - Y[idx]) ** 2).sum(1) / ((Y[idx] - MU) ** 2).sum(1))
        auc = c2st(PCA_.transform(y)[:, :DIST_PCS], PCA_.transform(Y[idx])[:, :DIST_PCS], DONORS[idx])
        return r2, 1 - 2 * abs(auc - 0.5)
    return score


def dd():
    return te.DirectionData(members=dict(MEMBERS), models={"M1": list(M1)}, target=Y, mu=MU, pca=PCA_, donors=DONORS)


def test_full_set_resample_matches_trace_and_direct_margin():
    idx = np.arange(N)
    r = te.resample(dd(), idx)
    want = trace({k: v[idx] for k, v in MEMBERS.items()}, direct_score(idx))
    cols = ["a", "b", "w", "r2_centered", "realism"]
    pd.testing.assert_frame_equal(r["points"][cols].reset_index(drop=True), want[cols].reset_index(drop=True), atol=1e-9, rtol=0)
    for (m, k), y in zip(zip(r["margins"], r["kinds"]), M1):
        m2, k2 = frontier_margin(want, *direct_score(idx)(y))
        assert abs(m - m2) < 1e-9 and k == k2


def test_resample_with_duplicates_matches_trace_on_the_subset():
    boot = an.donor_boot(DONORS, 3, seed=1)[0]
    r = te.resample(dd(), boot)
    want = trace({k: v[boot] for k, v in MEMBERS.items()}, direct_score(boot))
    assert np.allclose(r["points"]["realism"].to_numpy(), want["realism"].to_numpy(), atol=1e-12)
    assert np.allclose(r["points"]["r2_centered"].to_numpy(), want["r2_centered"].to_numpy(), atol=1e-9)


def test_duplicated_donor_copies_share_a_fold():
    """During a bootstrap resample with duplicated donors, every classifier-test split keeps each donor's
    copies in one fold (the splits are recorded from inside te.resample)."""
    from dgm import metrics
    boot = next(b for b in an.donor_boot(DONORS, 50, seed=2) if len(np.unique(DONORS[b])) < 12)
    seen = []
    class Recording(GroupKFold):
        def split(self, X, y=None, groups=None):
            for tr_, te_ in super().split(X, y, groups):
                seen.append(not set(np.asarray(groups)[tr_]) & set(np.asarray(groups)[te_]))
                yield tr_, te_
    orig = metrics.GroupKFold
    metrics.GroupKFold = Recording
    try:
        te.resample(dd(), boot)
    finally:
        metrics.GroupKFold = orig
    assert seen and all(seen)


def test_workers_do_not_change_results():
    boots = an.donor_boot(DONORS, 4, seed=3)
    one = te.run({"S->B": dd()}, boots, workers=1)
    two = te.run({"S->B": dd()}, boots, workers=6)
    for a, b in zip(one["S->B"], two["S->B"]):
        assert a["margins"] == b["margins"] and a["pure"] == b["pure"] and a["models"] == b["models"]


def test_model_on_the_frontier_has_zero_margin():
    """A model equal to a blend that is on the frontier has margin 0 (its own point is a frontier point)."""
    idx = np.arange(N)
    r = te.resample(dd(), idx)
    pts = r["points"]
    best = pts.loc[pts["realism"].idxmax()]
    y = best["w"] * MEMBERS[best["a"]] + (1 - best["w"]) * MEMBERS[best["b"]]
    d2 = te.DirectionData(members=dict(MEMBERS), models={"M1": [y]}, target=Y, mu=MU, pca=PCA_, donors=DONORS)
    m = te.resample(d2, idx)["margins"][0]
    assert abs(m) < 1e-9


def test_dry_run_never_returns_test_rows():
    class P:
        split = np.array(["train"] * 5 + ["val"] * 3 + ["test"] * 2)
        def idx(self, s):
            return np.flatnonzero(self.split == s)
    rows = te.split_rows(P(), dry=True)
    assert set(P.split[rows]) == {"val"}
    assert set(P.split[te.split_rows(P(), dry=False)]) == {"test"}


def test_reading_rules():
    assert te.h1_wording(0.01, 0.2) == "supported on halving 0" and te.h1_wording(-0.2, -0.01) == "inside the unpaired frontier"
    assert te.h1_wording(-0.1, 0.1) == "not distinguishable from the unpaired frontier"
    assert te.h1_qualifier(0.01, 0.0) == "not robust to the training split" and te.h1_qualifier(0.01, 0.01) == "" and te.h1_qualifier(-0.1, -1) == ""
    assert te.superseded_wording((0.1, 0.2), (0.1, 0.3)) == "would have been supported"
    assert te.superseded_wording((-0.3, -0.1), (0.1, 0.3)) == "better realism than any unpaired baseline; accuracy worse"
    assert te.superseded_wording((0.1, 0.2), (-0.1, 0.3)) == "better accuracy than any unpaired baseline; realism not distinguishable"
    assert te.superseded_wording((-0.3, -0.1), (-0.1, 0.3)) == "would not have been supported (realism not distinguishable; accuracy worse)"
    assert te.suspect(0.1, 0.4) and not te.suspect(0.3, 0.4) and not te.suspect(-0.03, -0.02) and not te.suspect(-0.1, 0.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
