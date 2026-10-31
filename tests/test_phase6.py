"""Known-answer checks for the Phase 6 analyses and the OT comparator (notes/phase6_spec.md):
python -m tests.test_phase6"""
import numpy as np

from dgm import analysis as an
from dgm import ot

RNG = np.random.default_rng(0)


def r2_of(y_true, mu):
    return lambda y: float(np.median(1 - ((y - y_true) ** 2).sum(1) / ((y_true - mu) ** 2).sum(1)))


def toy():
    n, g = 40, 12
    y = RNG.normal(size=(n, g)) + 3
    mu = y.mean(0) + 0.1
    members = {"a": y + RNG.normal(scale=0.8, size=(n, g)), "b": y + RNG.normal(scale=1.5, size=(n, g)) + 0.5,
               "training mean": np.tile(mu, (n, 1))}
    return y, mu, members


def test_matched_blend_hits_the_target():
    y, mu, members = toy()
    r2 = r2_of(y, mu)
    real = lambda yy: -float(np.abs(yy.std(0) - y.std(0)).mean())      # any deterministic score
    vals = [r2(m) for m in members.values()]
    target = float(np.mean([min(vals), max(vals)]))
    out = an.matched_blend(members, r2, real, target, tol=1e-6)
    assert out["best"] is not None and abs(out["best"]["r2"] - target) <= 1e-6
    assert all(abs(c["r2"] - target) <= 1e-6 for c in out["crossings"])
    assert out["best"]["realism"] == max(c["realism"] for c in out["crossings"])
    if out["second"] is not None:
        assert out["second"]["pair"] != out["best"]["pair"]
    none = an.matched_blend(members, r2, real, max(vals) + 1.0)          # beyond every path: no matched blend
    assert none["best"] is None and none["second"] is None


def test_residual_of_the_blend_itself():
    y, mu, members = toy()
    donors = np.repeat(np.arange(10), 4)
    r = members["a"] - members["a"]
    assert np.isnan(an.alignment(r, y, donors)).all()
    d = an.alignment(y - members["a"], y, donors)                         # the residual pointing at the truth is positive
    assert np.isfinite(d).all() and np.median(d) > 0


def test_gap_closure_known_answers():
    c = RNG.dirichlet(np.ones(6), size=30)
    b = RNG.dirichlet(np.ones(6), size=30)
    assert np.allclose(an.gap_closure(b, b, c), 0)
    assert np.allclose(an.gap_closure(c, b, c), 1)
    s = an.closure_summary(c, b, c)
    assert s["median g"] == 1 and s["share g > 0.5"] == 1 and s["ratio-of-sums g"] == 1
    assert np.isnan(an.gap_closure(b, c, c)).all()                         # reference already at the target
    assert np.isnan(an.transfer_ratio(0.3, 0.05)) and an.transfer_ratio(0.3, 0.6) == 0.5


def test_other_draw_is_never_the_sample():
    donors = np.repeat(np.arange(14), 5)[:-2]
    od = an.other_draw(donors)
    assert (od != np.arange(len(donors))).all() and (donors[od] == donors).all()


def test_sensitivity_is_one_for_a_linear_map():
    g = 20
    W = RNG.normal(size=(g, g))
    F = lambda x: x @ W
    basis = np.linalg.qr(RNG.normal(size=(g, 5)))[0].T
    for kind in ("top50", "complement", "per-gene"):
        e = an.noise(kind, 0.1, 15, basis, np.ones(5), np.r_[np.ones(g - 1), 0.0], np.random.default_rng(1))
        gx, y = RNG.normal(size=(15, g)), RNG.normal(size=(15, g))
        assert np.allclose(an.sensitivity(F, gx, y, e), 1)
        if kind == "complement":
            assert np.allclose(e @ basis.T, 0, atol=1e-10)
        if kind == "per-gene":
            assert (e[:, -1] == 0).all()


def test_similarity_excludes_the_copied_draw():
    """A same-donor reference copied from draw od[i] must not count od[i] among its comparisons."""
    donors = np.repeat(np.arange(4), 3)
    y = RNG.normal(size=(12, 8))
    od = np.array([1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9])
    with_self = an.similarity(y[od], y, donors)
    without = an.similarity(y[od], y, donors, exclude=od)
    assert (with_self["same donor, other draws"] >= without["same donor, other draws"]).all()
    from dgm.metrics import cosine_matrix
    S = cosine_matrix(y[od], y)
    for i in range(12):
        same = [j for j in range(12) if donors[j] == donors[i] and j != i and j != od[i]]
        assert np.isclose(without["same donor, other draws"][i], np.median(S[i, same]))


def test_info_free_map_recovers_an_affine_combination():
    regs_tr = [RNG.normal(size=(50, 6)) for _ in range(3)]
    regs_va = [RNG.normal(size=(20, 6)) for _ in range(3)]
    coef = RNG.normal(size=(3, 6))
    f = lambda regs: 0.5 + sum(c * r for c, r in zip(coef, regs))
    assert np.isclose(an.info_free_map(f(regs_tr), regs_tr, f(regs_va), regs_va), 1.0)


def test_noise_scales():
    g, k = 30, 4
    basis = np.linalg.qr(RNG.normal(size=(g, k)))[0].T
    pc_sd, gene_sd = np.array([1.0, 2.0, 0.5, 3.0]), np.linspace(0, 2, g)
    rng = np.random.default_rng(3)
    e = an.noise("top50", 0.1, 40000, basis, pc_sd, gene_sd, rng)
    assert np.allclose((e @ basis.T).std(0), 0.1 * pc_sd, rtol=0.03)
    e = an.noise("per-gene", 0.1, 40000, basis, pc_sd, gene_sd, rng)
    assert np.allclose(e.std(0), 0.1 * gene_sd, rtol=0.03, atol=1e-12)


def test_transfer_guard_and_bootstrap():
    assert np.isnan(an.transfer_ratio(0.05, 0.1)) and np.isnan(an.transfer_ratio(0.05, -0.2))
    donors = np.repeat(np.arange(7), 3)
    boots = an.donor_boot(donors, 50, seed=0)
    assert all(len(b) % 3 == 0 and all((donors[b] == d).sum() % 3 == 0 for d in range(7)) for b in boots)
    e, lo, hi = an.ci(lambda idx: float(np.mean(idx)), len(donors), boots)
    assert lo <= e <= hi


def ident(x):
    return x


def test_sinkhorn_marginals():
    a, b = RNG.normal(size=(30, 4)), RNG.normal(size=(25, 4)) + 1
    C = ot.sq_costs(a, b)
    eps = 0.1 * np.median(C)
    f, g, _ = ot.sinkhorn(C, eps)
    P = ot.plan(f, g, C, eps)
    assert np.abs(P.sum(1) - 1 / 30).max() < 1e-9 and np.abs(P.sum(0) - 1 / 25).max() < 1e-9


def test_small_eps_maps_to_the_permutation_partner():
    y = RNG.normal(size=(20, 6)) * 3
    perm = RNG.permutation(20)
    x = y[perm] + RNG.normal(scale=0.01, size=(20, 6))                     # source i's unique optimal partner is perm[i]
    m = ot.EntropicOT(1e-3, ident, n_pcs=6).fit(x, y)
    assert np.allclose(m.translate(x), y[perm], atol=1e-3)


def test_large_eps_gives_the_target_mean():
    x, y = RNG.normal(size=(20, 6)), RNG.normal(size=(24, 6))
    m = ot.EntropicOT(1e6, ident, n_pcs=6).fit(x, y)
    assert np.allclose(m.translate(RNG.normal(size=(5, 6))), y.mean(0), atol=1e-4)


def test_out_of_sample_matches_in_sample():
    x, y = RNG.normal(size=(30, 6)), RNG.normal(size=(26, 6)) + 0.5
    m = ot.EntropicOT(0.1, ident, n_pcs=6).fit(x, y)
    P = ot.plan(m.f_, m.g_, m.cost_, m.eps_)
    assert np.allclose(m.translate(x), (P / P.sum(1, keepdims=True)) @ y, atol=1e-8)


def test_conditional_samples_follow_the_plan():
    x, y = RNG.normal(size=(3, 4)), RNG.normal(size=(5, 4))
    m = ot.EntropicOT(0.5, ident, n_pcs=4).fit(x, y)
    w = m.weights(x)
    xs = np.repeat(x, 4000, axis=0)
    _, j = m.sample(xs, seed=0)
    freq = np.stack([np.bincount(j[i * 4000:(i + 1) * 4000], minlength=5) / 4000 for i in range(3)])
    assert np.abs(freq - w).max() < 4 * np.sqrt(0.25 / 4000)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
