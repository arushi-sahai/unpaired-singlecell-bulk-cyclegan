"""Known-answer checks for dgm.baselines: python -m tests.test_baselines"""
import numpy as np

from dgm.baselines import CORAL, ComBat, Harmony, Identity, MeanShift, PairedMLP, Ridge, RidgePCCalibration, RidgeSpread
from tests.synthetic import paired

D = paired()
tr, te = D["train"], D["test"]


def test_identity_and_mean_shift():
    rng = np.random.default_rng(0)
    src = rng.random((50, 10)) + 1
    tgt = src + 3.0                                       # a pure shift
    assert np.allclose(Identity().fit(src, tgt).translate(src), src)
    ms = MeanShift().fit(src, tgt)
    assert np.allclose(ms.translate(src), tgt)
    assert (MeanShift().fit(tgt, src).translate(tgt) >= 0).all()   # clipped at 0


def test_combat_location_scale():
    rng = np.random.default_rng(1)
    tgt = rng.normal(5, 1, (400, 30))
    src = rng.normal(5, 1, (400, 30)) * 2 + 3            # every gene scaled x2 and shifted
    out = ComBat(eb=False).fit(src, tgt).translate(src)
    assert np.allclose(out.mean(0), tgt.mean(0), atol=1e-8) and np.allclose(out.std(0, ddof=1), tgt.std(0, ddof=1), atol=1e-8)
    eb = ComBat().fit(src, tgt).translate(src)           # with many samples, EB shrinkage barely moves it
    assert np.abs(eb.mean(0) - tgt.mean(0)).max() < 0.05 and np.abs(eb.std(0) / tgt.std(0) - 1).max() < 0.05
    assert np.allclose(ComBat().fit(tgt, tgt).translate(tgt), tgt, atol=0.05)   # reference to itself: ~unchanged


def test_coral_matches_covariance():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(5, 5))
    src = rng.normal(size=(5000, 5)) + 10
    tgt = rng.normal(size=(5000, 5)) @ A + 20
    out = CORAL(shrinkage=0.0).fit(src, tgt).translate(src)
    assert np.allclose(np.cov(out, rowvar=False), np.cov(tgt, rowvar=False), atol=0.15)
    assert np.allclose(out.mean(0), tgt.mean(0), atol=0.05)


def test_harmony_corrects_overlapping_domains():
    # Three cell-type-like groups present in both domains, plus a batch shift smaller than the
    # group separation: Harmony's clusters then mix the domains and it should remove most of the
    # shift. (On data where the domains don't overlap, like SoundLife, it corrects nothing.)
    rng = np.random.default_rng(4)
    centers = rng.normal(0, 6, (3, 30))
    def domain(n, shift):
        lab = rng.integers(0, 3, n)
        return np.maximum(10 + centers[lab] + shift + rng.normal(0, 1, (n, 30)), 0)
    shift = np.r_[np.full(15, 2.0), np.zeros(15)]
    tgt_tr, src_tr, src_ev = domain(300, 0), domain(300, shift), domain(100, shift)
    h = Harmony(n_pcs=10)
    out = h.fit_translate(src_tr, tgt_tr, src_ev)
    d = h.diagnostics_
    assert d["mixed_clusters"] > 0 and d["centroid_gap_after"] < 0.5 * d["centroid_gap_before"], d
    assert out.shape == src_ev.shape and (out >= 0).all()
    gap_before = np.abs(src_ev.mean(0) - tgt_tr.mean(0)).mean()
    gap_after = np.abs(out.mean(0) - tgt_tr.mean(0)).mean()
    assert gap_after < 0.5 * gap_before, (gap_before, gap_after)


def test_coral_absolute_shrinkage_option():
    c = CORAL(shrinkage=1e-3, relative=False)
    assert c.name == "CORAL (abs 0.001)" and CORAL().name == "CORAL"
    rng = np.random.default_rng(5)
    src, tgt = rng.normal(size=(3000, 4)) + 10, rng.normal(size=(3000, 4)) * 3 + 20   # positive, so the clip at 0 never bites
    out = c.fit(src, tgt).translate(src)
    assert np.allclose(np.cov(out, rowvar=False), np.cov(tgt, rowvar=False), atol=0.3)


def test_spread_ridge_factor_is_cross_fitted():
    import dgm.metrics
    kw = dict(comp=D["comp"][tr], age=D["age"][tr], genes=D["genes"], direction="B->S")
    real_evaluator = dgm.metrics.Evaluator
    scored_against_own_targets = []

    class Recording(real_evaluator):
        # Every Evaluator RidgeSpread builds remembers its training targets, and every score
        # call checks that none of the targets it's scored against are among them.
        def __post_init__(self):
            super().__post_init__()
            self.seen_ = {r.tobytes() for r in np.asarray(self.target_train)}

        def score(self, y_hat, y, *args, **kwargs):
            scored_against_own_targets.append(any(r.tobytes() in self.seen_ for r in np.asarray(y)))
            return super().score(y_hat, y, *args, **kwargs)

    dgm.metrics.Evaluator = Recording
    try:
        rs = RidgeSpread().fit(D["src"][tr], D["tgt"][tr], D["donors"][tr], **kw)
    finally:
        dgm.metrics.Evaluator = real_evaluator
    assert scored_against_own_targets and not any(scored_against_own_targets), "an Evaluator scored its own training targets"
    assert len(scored_against_own_targets) == 5 * len(rs.factors)   # 5 folds x every factor
    r = Ridge().fit(D["src"][tr], D["tgt"][tr], D["donors"][tr])
    assert rs.alpha_ == r.alpha_ and rs.factor_ in rs.factors
    assert rs.factor_scores_[rs.factor_] == max(rs.factor_scores_.values())
    one = RidgeSpread(factors=(1.0,)).fit(D["src"][tr], D["tgt"][tr], D["donors"][tr], **kw)   # factor 1.0 is plain ridge
    assert np.allclose(one.translate(D["src"][te]), r.translate(D["src"][te]), atol=1e-6)


def test_ridge_recovers_a_linear_map():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 20))
    W = rng.normal(size=(20, 8))
    Y = X @ W + 5
    donors = np.repeat(np.arange(40), 3)
    r = Ridge(alphas=(1e-6, 1e-3)).fit(X, Y, donors)
    assert np.allclose(r.translate(X[:10]), np.maximum(Y[:10], 0), atol=1e-2)


def test_paired_mlp_learns():
    m = PairedMLP(hidden=32, max_epochs=200, patience=20, device="cpu").fit(D["src"][tr], D["tgt"][tr], D["donors"][tr])
    before = np.abs(D["src"][te] - D["tgt"][te]).mean()
    after = np.abs(m.translate(D["src"][te]) - D["tgt"][te]).mean()
    assert after < 0.5 * before, (before, after)



def test_ridge_spread_translate_at():
    kw = dict(comp=D["comp"][tr], age=D["age"][tr], genes=D["genes"], direction="B->S")
    rs = RidgeSpread(factors=(1.0, 1.5)).fit(D["src"][tr], D["tgt"][tr], D["donors"][tr], **kw)
    assert np.array_equal(rs.translate(D["src"][te]), rs.translate_at(D["src"][te], rs.factor_))
    r = Ridge().fit(D["src"][tr], D["tgt"][tr], D["donors"][tr])
    assert np.allclose(rs.translate_at(D["src"][te], 1.0), r.translate(D["src"][te]), atol=1e-6)


def test_pc_calibration_matches_component_spread():
    """Ridge on a noisy linear map shrinks its predictions; calibrated held-out predictions have
    top-component SDs close to the real targets' (plain ridge's are well below)."""
    rng = np.random.default_rng(4)
    n, g = 600, 30
    X = rng.normal(size=(n, g))
    Y = X @ rng.normal(size=(g, g)) * 0.3 + rng.normal(size=(n, g)) + 5    # half the variance is unpredictable noise
    donors = np.repeat(np.arange(n // 3), 3)
    trn, tst = slice(0, 450), slice(450, None)
    pc = RidgePCCalibration(n_pcs=5, alphas=(1e2, 1e3)).fit(X[trn], Y[trn], donors[trn])
    r = Ridge(alphas=(1e2, 1e3)).fit(X[trn], Y[trn], donors[trn])
    assert pc.alpha_ == r.alpha_ and (pc.ratio_ > 1).all(), pc.ratio_
    sd = lambda x: pc.pca_.transform(x).std(0)
    target = sd(Y[tst])
    err_cal = np.abs(np.log(sd(pc.translate(X[tst])) / target)).mean()
    err_raw = np.abs(np.log(sd(r.translate(X[tst])) / target)).mean()
    assert err_cal < 0.15 < err_raw, (err_cal, err_raw)
    pc.ratio_ = np.ones(5)                                     # ratio 1 is plain ridge
    assert np.allclose(pc.translate(X[tst]), r.translate(X[tst]), atol=1e-8)

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
