"""Known-answer checks for dgm.deconv: python -m tests.test_deconv"""
import numpy as np
import scipy.sparse as sp

from dgm.deconv import Deconvolver, bulk_high_genes, select_markers, signatures


def cells(rng, n_per=(200, 100, 50), g=60):
    """Three cell types, each with its own block of 20 specific genes and a different mRNA content."""
    content = (1.0, 3.0, 0.5)                          # relative counts per cell
    X, lab = [], []
    for c, (n, m) in enumerate(zip(n_per, content)):
        mu = np.full(g, 0.2)
        mu[20 * c:20 * (c + 1)] = 5.0
        X.append(rng.poisson(mu * m, (n, g)))
        lab += [f"C{c}"] * n
    return sp.csr_matrix(np.vstack(X)), np.array(lab)


def test_signatures_and_markers():
    rng = np.random.default_rng(0)
    X, lab = cells(rng)
    sig = signatures(X, lab, ["C0", "C1", "C2"], np.arange(60))
    assert sig.shape == (60, 3) and sig[:20, 0].mean() > 4 and sig[20:40, 1].mean() > 12   # C1 carries 3x the mRNA
    cp = signatures(X, lab, ["C0", "C1", "C2"], np.arange(60), kind="cp10k")
    assert np.allclose(cp.sum(0), 1e4, rtol=1e-6)      # all genes kept, so each class profile sums to 1e4
    m = select_markers(cp, np.zeros(60, bool), k=5)
    assert len(m) == 15 and all(any(20 * c <= i < 20 * (c + 1) for c in range(3)) for i in m)
    excl = np.zeros(60, bool); excl[:20] = True
    assert not set(select_markers(cp, excl, k=5)) & set(range(20)), "excluded genes were picked"
    assert bulk_high_genes(np.array([10.0, 1.0]), np.array([1.0, 1.0])).tolist() == [True, False]


def test_count_signatures_recover_cell_fractions():
    """A pseudobulk of known cell numbers: count signatures recover the cell fractions even though
    the types differ 6-fold in mRNA content; mRNA-weighted (CP10k) signatures don't."""
    rng = np.random.default_rng(1)
    X, lab = cells(rng)
    classes = ["C0", "C1", "C2"]
    sig = signatures(X, lab, classes, np.arange(60))
    cp = signatures(X, lab, classes, np.arange(60), kind="cp10k")
    markers = select_markers(cp, np.zeros(60, bool), k=20)
    truth = np.array([0.5, 0.2, 0.3])
    mix = sig @ (truth * 1000)                          # 1000 cells in these proportions
    y = 1e4 * mix / mix.sum()                           # CP10k: the global scale doesn't matter
    est = Deconvolver(sig, markers).predict(y)[0]
    assert np.allclose(est, truth, atol=1e-6), est
    est_cp = Deconvolver(cp, markers).predict(y)[0]
    assert np.abs(est_cp - truth).max() > 0.1, est_cp   # mRNA-weighted: C1 over-, C2 under-counted
    est_r = Deconvolver(sig, markers, residual=True).predict(y + 5.0)[0]   # a constant offset is absorbed
    assert np.allclose(est_r, truth, atol=1e-6), est_r


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
