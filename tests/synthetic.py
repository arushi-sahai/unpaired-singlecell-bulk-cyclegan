"""Small synthetic paired data with known structure, shared by the metric and baseline tests:
target profiles are driven by composition (so a composition probe can work) plus noise, and
the source is a known transform of the target."""
import numpy as np


def paired(n_donors=30, draws=3, g=40, k=3, seed=0):
    rng = np.random.default_rng(seed)
    donors = np.repeat([f"D{i}" for i in range(n_donors)], draws)
    n = len(donors)
    age_group = np.repeat(rng.integers(0, 2, n_donors), draws)
    comp = rng.dirichlet(np.ones(k) * 3, n)
    loadings = rng.normal(0, 1, (k, g))
    donor_effect = np.repeat(rng.normal(0, 0.3, (n_donors, g)), draws, axis=0)
    age_effect = np.outer(age_group, rng.normal(0, 0.5, g))
    tgt = np.maximum(2 + comp @ loadings * 2 + donor_effect + age_effect + rng.normal(0, 0.2, (n, g)), 0)
    shift = rng.normal(0, 1, g)
    src = np.maximum(tgt * 0.8 + shift + 1, 0)            # a known distortion of the target
    genes = [f"G{i}" for i in range(g)]
    train = np.arange(n) < int(0.6 * n_donors) * draws    # first 60% of donors
    return dict(src=src, tgt=tgt, comp=comp, age=age_group, donors=donors, genes=genes,
                classes=[f"C{i}" for i in range(k)], train=train, test=~train)
