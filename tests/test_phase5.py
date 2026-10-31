"""Known-answer checks for the Phase 5 options (notes/phase5_spec.md): python -m tests.test_phase5"""
import json

import numpy as np
import torch

from dgm import composition as comp
from dgm import labels as lb
from dgm.models import canonical_hash
from dgm.translation import config, distance_loss, make_generator, start_map, train
from tests import test_ablation as t4

comp.deterministic()
XS, XB, INIT = t4.XS, t4.XB, t4.INIT


def run(cfg_kw, task=None, seed=0):
    return t4.run(cfg_kw, task=task, seed=seed)


def test_defaults_reproduce_phase3():
    """Every Phase 5 option at its default: bit-identical to the Phase 3 code (with and without a task loss)."""
    t4.test_defaults_reproduce_phase3()


def test_anchored_identity_is_zero_at_the_start():
    res = train(XS, XB, INIT, config(hidden=32, steps=1, log_every=1, identity=True, identity_target="start"), 0)
    assert res["log"][0]["identity"] < 1e-6
    plain = train(XS, XB, INIT, config(hidden=32, steps=1, log_every=1, identity=True), 0)
    assert plain["log"][0]["identity"] > 1e-3          # the plain identity loss is not zero at the mean-shift start


def test_identity_for_bulk_to_single_cell_only():
    """identity_dirs = "bs": with no adversarial and no cycle loss, G_sb receives no gradient at all."""
    kw = dict(identity=True, lambda_adv=0.0, lambda_cyc=0.0, lambda_id=10.0)
    only_bs, both = run({**kw, "identity_dirs": "bs"}), run({**kw})
    start = train(XS, XB, INIT, config(hidden=32, steps=0), 0)["state"]
    sb = lambda st: {k: v for k, v in st.items() if k.startswith("sb.")}
    assert canonical_hash(sb(only_bs["state"])) == canonical_hash(sb(start))
    assert canonical_hash(sb(both["state"])) != canonical_hash(sb(start))


def test_distance_loss():
    """Zero when the output is the reference, under rescaling and shifting of the output; positive
    for a permuted output."""
    x = torch.as_tensor(XS[:64])
    ref = start_map(INIT["S->B"], x)
    assert float(distance_loss(ref, ref)) < 1e-6
    assert float(distance_loss(3.0 * ref + 1.0, ref)) < 1e-4
    perm = ref[torch.randperm(len(ref), generator=torch.Generator().manual_seed(0))]
    assert float(distance_loss(perm, ref)) > 0.1
    res = run({"lambda_dist": 1.0, "log_every": 10})
    assert all(np.isfinite(e["dist"]) for e in res["log"])


def test_bottleneck_shape_and_start():
    g = make_generator(*INIT["S->B"], hidden=32, bottleneck=8)
    assert g.f[3].out_features == 8 and g.f[-1].in_features == 8
    x = torch.as_tensor(XB[:10])
    with torch.no_grad():
        assert torch.allclose(g(x), start_map(INIT["S->B"], x), atol=1e-6)


def test_task_mask_and_row_readouts():
    """A mask with no eligible rows makes the single-cell -> bulk term zero; a RowEnsemble reads each
    row with its own fold's network."""
    task = t4.task_dict()
    task["mask_s"] = torch.zeros(len(XS), dtype=torch.bool)
    res = run({"lambda_task": 1.0, "warmup": 0.0, "ramp": 0.0, "task_terms": "sb"}, task=task)
    assert all(e["task"] == 0 for e in res["log"])
    consts = [torch.nn.Module() for _ in range(3)]
    for i, m in enumerate(consts):
        m.forward = (lambda i: (lambda x: torch.full((len(x), 2), float(i))))(i)
    ens = lb.RowEnsemble(consts, [2, 0, 1, 2])
    out = ens(torch.zeros(3, 5), torch.tensor([0, 1, 3]))
    assert out[:, 0].tolist() == [2.0, 0.0, 2.0]


def test_folds_and_pool():
    """Folds partition the donors and are age-balanced; the budget pool is half-A only, age-balanced
    at every nested k, and differs between seeds."""
    from dgm.data import load_pairs, train_halves
    P = load_pairs()
    tr = P.idx("train")
    folds = lb.donor_folds(P, tr)
    assert set(folds) == {str(d) for d in np.unique(P.donor_ids[tr])} and set(folds.values()) == set(range(5))
    sizes = np.bincount(list(folds.values()))
    assert sizes.max() - sizes.min() <= 2
    half_a, half_b = train_halves(P, 0)
    for seed in (6, 7, 8):
        pool = lb.budget_pool(P, half_a, seed)
        assert len(pool) == 16 and set(pool) <= {str(d) for d in P.donor_ids[half_a]}
        age = {str(d): g for d, g in zip(P.donor_ids[half_a], P.age_group[half_a])}
        for k in (4, 8, 16):
            assert len({age[d] for d in pool[:k]}) == 2 and sum(age[d] == age[pool[0]] for d in pool[:k]) == k // 2
    assert lb.budget_pool(P, half_a, 6) != lb.budget_pool(P, half_a, 7)


def test_label_networks_never_see_their_rows():
    """Registry checks (if the Phase 5 networks have been trained): every cross-fit network leaves out
    exactly its fold's donors; the budget networks saw only the first k pool donors of half A."""
    from dgm import paths
    reg = paths.REPO / "results" / "phase5" / "models.json"
    if not reg.exists():
        print("  skipped (Phase 5 networks not trained yet)")
        return
    from dgm.data import load_pairs, train_halves
    P = load_pairs()
    half_a, _ = train_halves(P, 0)
    e = json.loads(reg.read_text())
    for name in ("C_bulk_cf68", "C_bulk_cfA"):
        folds = e[f"{name}_folds"]
        for f in range(5):
            seen = set(e[f"{name}_f{f}"]["donors"])
            assert not seen & {d for d, ff in folds.items() if ff == f}
            assert seen == {d for d, ff in folds.items() if ff != f}
    assert set(e["C_bulk_cfA_folds"]) == {str(d) for d in np.unique(P.donor_ids[half_a])}
    for seed in (6, 7, 8):
        pool = e[f"pool_s{seed}"]
        for k in (4, 8, 16):
            assert set(e[f"C_bulk_k{k}_s{seed}"]["donors"]) == set(pool[:k])


def test_budget_mask_on_real_contexts():
    """For every k4/k8/k16 task on seeds 6-8 (if trained): the single-cell -> bulk mask is False on
    exactly the 16 pool donors' rows and True on the other 17 half-A donors' rows, and the label
    network's donors all fall where the mask is False."""
    from dgm import paths
    from dgm.data import load_pairs, train_halves
    from dgm.models import pairs_hash
    from dgm.translation import SoundLife
    reg = paths.REPO / "results" / "phase5" / "models.json"
    if not reg.exists():
        print("  skipped (Phase 5 networks not trained yet)")
        return
    e = json.loads(reg.read_text())
    P = load_pairs()
    half_a, half_b = train_halves(P, 0)
    ctx = SoundLife.__new__(SoundLife)                  # only the fields _phase5_task reads
    ctx.pairs, ctx.half_a, ctx.half_b, ctx.subset = P, half_a, half_b, None
    ctx.genes, ctx.classes, ctx.comp_s = P.genes.tolist(), P.l1_classes.tolist(), P.composition[half_a]
    donors_a = np.array([str(d) for d in P.donor_ids[half_a]])
    for seed in (6, 7, 8):
        pool = set(e[f"pool_s{seed}"])
        for k in (4, 8, 16):
            task = ctx._phase5_task(f"k{k}", seed, 0, pairs_hash(P))
            m = task["mask_s"].numpy()
            assert (~m == np.isin(donors_a, list(pool))).all()
            assert len(set(donors_a[m])) == 17 and len(set(donors_a[~m])) == 16
            assert set(e[f"C_bulk_k{k}_s{seed}"]["donors"]) <= set(donors_a[~m])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
