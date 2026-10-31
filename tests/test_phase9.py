"""Known-answer checks for the Phase 9 measures (notes/phase9_spec.md): python -m tests.test_phase9"""
import tempfile
from pathlib import Path

import numpy as np

from dgm import analysis as an
from dgm import biology as bi

RNG = np.random.default_rng(0)
DONORS = np.repeat([f"d{i}" for i in range(40)], 10)          # 40 donors x 10 draws


def _donor_signal(sd=1.0):
    lvl = {d: RNG.normal(0, sd) for d in np.unique(DONORS)}
    return np.array([lvl[d] for d in DONORS])


def test_donor_signal_tracks_between_not_within():
    donor = _donor_signal()
    true = donor + RNG.normal(0, 0.5, len(DONORS))             # a person-level program with visit noise
    pred = donor + RNG.normal(0, 0.5, len(DONORS))             # the translation keeps the person, not the visit
    t = bi.tracking(true, pred, DONORS)
    assert t["between r"] > 0.9 and abs(t["within r"]) < 0.15, t


def test_visit_signal_tracks_within_not_between():
    visit = bi.within(RNG.normal(0, 1, len(DONORS)), DONORS)   # a visit-level signal, centered within donors
    true = visit + _donor_signal()
    pred = visit + _donor_signal()                              # independent donor noise in the translation
    t = bi.tracking(true, pred, DONORS)
    assert t["within r"] > 0.99 and abs(t["between r"]) < 0.35, t


def test_constant_translation_gives_nan():
    true = RNG.normal(0, 1, len(DONORS))
    t = bi.tracking(true, np.zeros(len(DONORS)), DONORS)
    assert np.isnan(t["r"]) and np.isnan(t["between r"]) and np.isnan(t["within r"]) and np.isnan(t["slope"])


def test_shuffled_rows_give_r_near_zero():
    true = _donor_signal() + RNG.normal(0, 1, len(DONORS))
    rs = [bi.pearson(true, RNG.permutation(true)) for _ in range(200)]
    assert abs(np.mean(rs)) < 0.02 and np.std(rs) < 0.1


def test_r_is_invariant_to_centering_and_slope_is_not_to_scale():
    a, b = RNG.normal(size=(300, 4)), RNG.normal(size=(300, 4))
    b[:, 0] += 2 * a[:, 0]
    assert np.allclose(bi.colwise_r(a, b), bi.colwise_r(a - 5.0, b + 3.0))
    assert abs(bi.slope(b[:, 0], a[:, 0]) - 2 * bi.slope(b[:, 0], 2 * a[:, 0])) < 1e-9


def test_zscore_parameters_refuse_masked_rows():
    Y = RNG.normal(size=(20, 3))
    mu, sd = bi.zparams(Y)
    assert np.allclose(mu, Y.mean(0)) and np.allclose(sd, Y.std(0))
    Y[3] = np.nan
    try:
        bi.zparams(Y)
    except AssertionError:
        return
    raise AssertionError("zparams accepted rows with NaN (masked test rows)")


def test_duplicated_donor_counts_twice():
    donors = np.repeat(["a", "b", "c", "d"], 3)
    boots = bi.donor_boot_labeled(donors, 50, seed=1)
    assert all((i == j).all() for i, (j, _) in zip(an.donor_boot(donors, 50, seed=1), boots))
    idx, lab = next(b for b in boots if len(np.unique(donors[b[0]])) < 4)   # a resample with a repeated donor
    assert len(np.unique(lab)) == 4 and len(bi.donor_means(np.arange(len(idx), dtype=float), lab)[0]) == 4


def test_partial_r_removes_a_composition_only_signal():
    comp = RNG.dirichlet(np.ones(4), 400)
    beta = np.array([3.0, -2.0, 1.0, 0.0])
    true = comp @ beta + RNG.normal(0, 0.05, 400)               # a program that is only composition
    pred = comp @ beta + RNG.normal(0, 0.05, 400)
    tr, va = np.arange(300), np.arange(300, 400)
    coefs = bi.partial_coefs(true[tr], comp[tr])
    rt, rp = bi.residualize(true[va], comp[va], coefs), bi.residualize(pred[va], comp[va], coefs)
    assert bi.pearson(true[va], pred[va]) > 0.9 and abs(bi.pearson(rt, rp)) < 0.3


def test_shrinkage_move_excess():
    n, g = 300, 50
    donors = np.repeat(np.arange(30), 10)
    mean = np.zeros(g)
    s = RNG.normal(size=(n, g))                                # a start independent of the partner
    y = RNG.normal(size=(n, g))
    other = bi.other_donor(donors, seed=0)
    G = s + bi.SHRINK * (mean - s)
    assert abs(np.nanmedian(bi.move_excess(G, s, y, y[other]))) < 0.05
    y_corr = s + RNG.normal(0, 0.5, (n, g))                    # a partner correlated with the start (as on real data)
    assert np.nanmedian(bi.move_excess(G, s, y_corr, y_corr[other])) < -0.1


def test_donor_year_units_and_their_bootstrap():
    donors = np.array(["a"] * 4 + ["b"] * 2 + ["c"] * 2)
    tp = np.array(["Flu Year 1 Day 0", "Flu Year 1 Day 7", "Flu Year 2 Day 0", "Flu Year 2 Day 7",
                   "Flu Year 1 Day 0", "Flu Year 1 Day 90", "Immune Variation Day 0", "Immune Variation Day 7"])
    u = bi.donor_years(tp, donors)
    assert list(u["donor"]) == ["a", "a"] and list(u["day0"]) == [0, 2] and list(u["day7"]) == [1, 3]
    units = u.assign(donor=["a", "b"])                         # two donors, so a resample can miss one
    for b in bi.unit_boot(units.iloc[[0, 1]].assign(donor=["a", "a"]), 20):
        assert sorted(b.tolist()) == [0, 1]                    # a donor's years always travel together


def test_stratified_bootstrap_keeps_every_class():
    donors = np.repeat([f"d{i}" for i in range(14)], 2)
    strata = {f"d{i}": ("F" if i < 9 else "M") for i in range(14)}
    for b in bi.strat_boot(donors, strata, 200):
        drawn = {strata[d] for d in donors[b]}
        assert drawn == {"F", "M"}


def test_gmt_parser():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "x.gmt"
        f.write_text("A\turl\tG1\tG2\tG2\tG3\tZZ\nB\turl\tG1\n")
        sets, cov = bi.read_gmt(f, ["G1", "G2", "G3"], min_size=3, check=False)
    assert sets == {"A": ["G1", "G2", "G3"]}
    assert cov.set_index("set").loc["A", "unmatched"] == 1 and not cov.set_index("set").loc["B", "kept"]


def test_mask_test_and_row_guard():
    class P:
        split = np.array(["train", "val", "test", "test"])
        def idx(self, split):
            return np.flatnonzero(self.split == split)
        x_sc = RNG.normal(size=(4, 3)).astype(np.float32)
        x_bulk = RNG.normal(size=(4, 3)).astype(np.float32)
        composition = RNG.dirichlet(np.ones(2), 4).astype(np.float32)
    bi.mask_test(P)
    assert np.isnan(P.x_sc[2:]).all() and np.isnan(P.x_bulk[2:]).all() and np.isnan(P.composition[2:]).all()
    assert np.isfinite(P.x_sc[:2]).all()
    rows = bi.row_guard(P())
    assert list(rows("val")) == [1]
    try:
        rows("test")
    except AssertionError:
        return
    raise AssertionError("the row guard returned test rows")


def test_between_share():
    donor = _donor_signal(1.0)
    x = donor + RNG.normal(0, 1.0, len(DONORS))                # equal person and visit variance: share about 0.5
    assert 0.35 < bi.between_share(x, DONORS) < 0.65
    assert bi.between_share(RNG.normal(size=len(DONORS)), DONORS) < 0.1


def test_partial_move_r():
    n, g = 300, 40
    s = RNG.normal(size=(n, g)); y = s + RNG.normal(size=(n, g))
    affine = s + 0.3 * (np.zeros(g) - s) + 0.1                 # an affine function of the start: null by construction
    r = bi.partial_move_r(affine, s, y)
    assert np.all(np.isnan(r) | (np.abs(r) < 1e-6))
    planted = s + 0.5 * (y - s) + RNG.normal(0, 0.1, (n, g))  # a move carrying the partner's own signal
    assert np.nanmedian(bi.partial_move_r(planted, s, y)) > 0.5


def test_vectorized_tracking_matches():
    true, pred = RNG.normal(size=len(DONORS)), RNG.normal(size=len(DONORS)) + _donor_signal()
    t = bi.tracking(true, pred, DONORS)
    v = bi.tracking_codes(true, pred, bi.codes_of(DONORS))
    assert np.allclose([t["r"], t["between r"], t["within r"]], v)
    idx, lab = bi.donor_boot_labeled(DONORS, 3, seed=2)[0]
    t = bi.tracking(true[idx], pred[idx], lab)
    assert np.allclose([t["r"], t["between r"], t["within r"]], bi.tracking_codes(true[idx], pred[idx], bi.codes_of(lab)))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
