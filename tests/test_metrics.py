"""Known-answer checks for dgm.metrics: python -m tests.test_metrics"""
import numpy as np

from dgm.metrics import (Evaluator, c2st_null, composite, cosine_matrix, donor_auroc, donor_bootstrap, eta2, mmd_rbf,
                         partner_ranks, rowwise_pearson, rowwise_spearman, within_donor_ranks)
from tests.synthetic import paired

D = paired()
tr, te = D["train"], D["test"]
EV = Evaluator(D["tgt"][tr], D["comp"][tr], D["age"][tr], D["donors"][tr], D["genes"], "B->S", n_pcs=20)


def score(y_hat):
    return EV.score(y_hat, D["tgt"][te], D["comp"][te], D["age"][te], D["donors"][te], D["classes"])[0]


def test_building_blocks():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5, 20))
    assert np.allclose(rowwise_pearson(a, a), 1) and np.allclose(rowwise_pearson(a, -a), -1)
    assert np.allclose(rowwise_pearson(a, 3 * a + 7), 1)                      # invariant to scale and offset
    assert np.allclose(rowwise_spearman(a, np.exp(a)), 1)                    # monotone transform
    assert (partner_ranks(cosine_matrix(a, a)) == 1).all()
    assert partner_ranks(np.array([[0.1, 0.9], [0.2, 0.8]])).tolist() == [2, 1]
    assert eta2(np.r_[np.zeros(5), np.ones(5)], np.r_[np.zeros(5), np.ones(5)]) == 1.0
    assert abs(eta2(rng.normal(size=1000), rng.integers(0, 2, 1000))) < 0.01
    x = rng.normal(size=(200, 3))
    # identical sets: ~0 (slightly negative, unbiased); far-apart sets: 2 E k(a,b) = 2 (1+2)^-1.5 = 0.38 here
    assert abs(mmd_rbf(x, x + 0.0, 1.0)) < 0.02 and mmd_rbf(x, x + 3, 1.0) > 0.25
    donors = np.repeat(["a", "b", "c", "d"], 3)
    assert donor_auroc(np.repeat([0.1, 0.2, 0.8, 0.9], 3), np.repeat([0, 0, 1, 1], 3), donors) == 1.0
    est, lo, hi = donor_bootstrap(rng.normal(size=12), donors, n_boot=200)
    assert lo <= est <= hi


def test_perfect_translation():
    m = score(D["tgt"][te])
    assert m["r_centered"] > 0.999 and m["rank_median"] == 1 and m["top1"] == 1.0
    assert m["mae"] == 0 and abs(m["c2st_auroc"] - 0.5) < 0.15 and m["mmd"] < 0.02
    assert abs(m["diversity"] - 1) < 1e-9 and abs(m["nn_ratio"] - 1) < 1e-9
    reference = EV.fit_reference(D["tgt"][te], D["comp"][te], D["age"][te], D["donors"][te], D["classes"])
    assert abs(m["comp_r"] - reference["comp_r"]) < 1e-9 and reference["comp_r"] > 0.8   # the probe works on this data


def test_shuffled_partners_lose_pairing():
    y = D["tgt"][te]
    m = score(y[np.random.default_rng(2).permutation(len(y))])
    assert m["rank_median"] > 5 and m["r_centered"] < 0.6  # the pairing is gone


def test_independent_sample_from_same_distribution():
    # translations that are real profiles of *other* donors from the same distribution (the
    # training targets): the two-sample test should be near chance, the pairing metrics poor.
    y = D["tgt"][te]
    other = D["tgt"][tr][np.random.default_rng(1).permutation(tr.sum())[: len(y)]]
    m = score(other)
    assert abs(m["c2st_auroc"] - 0.5) < 0.15, m["c2st_auroc"]
    assert m["top1"] < 0.2


def test_shifted_translation_is_detected():
    m = score(D["tgt"][te] + 1.0)
    assert m["c2st_auroc"] > 0.95 and m["mmd"] > 0.1
    # a uniform offset is invisible to per-sample correlation (it's offset-invariant), so the
    # distribution metrics and MAE are what catch it
    assert m["r_centered"] > 0.999 and abs(m["mae"] - 1.0) < 1e-9


def test_shrinkage_toward_the_mean_is_detected():
    # outputs with the right mean but half the spread (what regression does) must not look real
    y = D["tgt"][te]
    m = score(EV.mean_ + 0.5 * (y - EV.mean_))
    assert m["c2st_auroc"] > 0.8 and m["diversity"] < 0.6


def test_collapsed_translation():
    y = D["tgt"][te]
    m = score(np.tile(y.mean(0), (len(y), 1)))
    assert m["diversity"] < 1e-9 and m["top1"] < 0.2


def test_composite_definition():
    assert composite({"r2_centered": 1.0, "c2st_auroc": 0.5, "comp_r2": 1.0}) == 1.0
    assert composite({"r2_centered": 0.0, "c2st_auroc": 1.0, "comp_r2": 0.0}) == 0.0
    assert composite({"r2_centered": 0.0, "c2st_auroc": 0.0, "comp_r2": 0.0}) == 0.0   # perfectly separable either way
    assert composite({"r2_centered": -5.0, "c2st_auroc": 0.5, "comp_r2": -2.0}) == 1 / 3  # negative R^2 floors at 0


def test_r2_penalizes_inflation_and_noise_where_correlation_does_not():
    y = D["tgt"][te]
    perfect = score(y)
    inflated = score(EV.mean_ + 1.5 * (y - EV.mean_))
    noisy = score(np.maximum(y + np.random.default_rng(3).normal(0, 0.5, y.shape), 0))
    assert abs(perfect["r2_centered"] - 1) < 1e-9
    assert abs(perfect["comp_r2"] - EV.fit_reference(y, D["comp"][te], D["age"][te], D["donors"][te], D["classes"])["comp_r2"]) < 1e-9
    assert inflated["r_centered"] > 0.999                     # correlation can't see the inflation...
    assert inflated["r2_centered"] < 0.9 and inflated["comp_r2"] < perfect["comp_r2"]   # ...R^2 can
    assert noisy["r2_centered"] < perfect["r2_centered"] and noisy["composite"] < perfect["composite"]


def test_c2st_null_is_centred_at_chance():
    zh = EV.pca_.transform(D["tgt"][te] * 0.9 + 0.1)[:, :10]
    zy = EV.pca_.transform(D["tgt"][te])[:, :10]
    mean, sd = c2st_null(zh, zy, D["donors"][te], n_perm=40)
    assert abs(mean - 0.5) < 0.06 and 0 < sd < 0.15


def test_scoring_never_changes_the_fitted_evaluator():
    import copy
    state = lambda ev: [ev.mean_.copy(), ev.pca_.components_.copy(), ev.comp_probe_.coef_.copy(),
                        ev.age_clf_[-1].coef_.copy(), ev.comp_alpha_]
    before = state(EV)
    y = D["tgt"][te]
    first = score(y * 0.9)
    EV.score(y * 0.9, y * 3 + 7, D["comp"][te][::-1], 1 - D["age"][te], D["donors"][te], D["classes"])   # wildly different held-out targets
    after = state(EV)
    assert all(np.array_equal(a, b) for a, b in zip(before, after)), "scoring changed the evaluator"
    again = score(y * 0.9)
    assert again.keys() == first.keys() and all(np.isclose(again[k], first[k], equal_nan=True, rtol=0, atol=0) for k in first), \
        "same inputs gave different scores"



def test_within_donor_ranks():
    sim = np.array([[0.9, 0.1, 0.0], [0.5, 0.4, 0.0], [0.0, 0.0, 1.0]])
    r = within_donor_ranks(sim, np.array(["a", "a", "b"]))
    assert r[0] == 1 and r[1] == 2 and np.isnan(r[2])         # donor b has one draw: not ranked
    m = score(D["tgt"][te])                                    # perfect translations: every partner ranks first
    assert m["within_top1"] == 1.0 and abs(m["within_chance"] - 1 / 3) < 1e-12, m
    rng = np.random.default_rng(0)
    m = score(D["tgt"][te][rng.permutation(te.sum())])        # shuffled: near chance
    assert m["within_top1"] < 0.6

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
