"""Known-answer checks for dgm.composition (spec: notes/phase2_spec.md, Tests): python -m tests.test_composition"""
from types import SimpleNamespace

import numpy as np
import torch

from dgm.composition import (LinearProbe, MarkerProbe, MARKERS, assert_train_only, attack_summary, closed_form_eps50, closure,
                             comp_r2, deterministic, eps50, freeze_nets, grid, inner_folds, load_marker, paired_bootstrap_diff,
                             pgd, pick_targets, run_config, save_marker, train_net)
from dgm.models import canonical_hash

deterministic()


def planted(n=240, g=30, k=3, seed=0, low_var_genes=0):
    """Composition from a linear-softmax of a few high-variance genes, plus optional near-constant
    genes that carry the same signal at tiny scale (the exploitable kind)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(5, 1, (n, g)).astype(np.float32)
    W = np.zeros((g, k)); W[:6] = rng.normal(0, 1.5, (6, k))
    logits = (X - 5) @ W
    C = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    if low_var_genes:
        z = ((X[:, :6] - 5) @ W[:6, :1]).ravel()
        lv = 2 + 0.001 * (z[:, None] + rng.normal(0, 0.3, (n, low_var_genes)))
        X = np.hstack([X, lv.astype(np.float32)])
    donors = np.repeat([f"D{i}" for i in range(n // 4)], 4)
    return X, C.astype(np.float32), donors


CFG = {"scaling": "center", "hidden": 0, "loss": "l1", "weight_decay": 1e-5, "dropout": 0.0, "noise": 0.0}


def test_grid_has_108_configs():
    g = grid()
    assert len(g) == 108 and len({tuple(sorted(c.items())) for c in g}) == 108
    assert all(c["dropout"] == 0 for c in g if c["hidden"] == 0)


def test_train_only_guard():
    pairs = SimpleNamespace(split=np.array(["train", "train", "val", "test"]))
    assert_train_only(pairs, [0, 1])
    for bad in ([0, 2], [3]):
        try:
            assert_train_only(pairs, bad)
            raise AssertionError("accepted non-train rows")
        except AssertionError as e:
            assert "non-train" in str(e)


def test_inner_folds_donor_disjoint_and_stratified():
    donors = np.repeat([f"D{i}" for i in range(40)], 3)
    age = np.repeat(np.r_[np.zeros(20), np.ones(20)].astype(int), 3)
    folds = inner_folds(donors, age, 5)
    seen = set()
    for tr, te in folds:
        assert not set(donors[tr]) & set(donors[te])
        assert np.bincount(age[te][::3], minlength=2).tolist() == [4, 4]   # each fold: 4 young + 4 older donors
        seen |= set(te)
    assert seen == set(range(len(donors)))


def test_planted_composition_is_recovered_and_training_is_deterministic():
    X, C, _ = planted()
    # 180 samples is 3 batches per epoch and the planted targets are near 0/1, so it needs
    # ~600 epochs here (it reaches 0.80 by epoch 400); the real folds have 8 batches per epoch.
    a, hist = train_net(X[:180], C[:180], CFG, seed=0, epochs=600, eval_sets={"held": (X[180:], C[180:])})
    assert hist["held"][-1] > 0.8, hist["held"][-1]
    b, _ = train_net(X[:180], C[:180], CFG, seed=0, epochs=600)
    assert canonical_hash(a.state_dict()) == canonical_hash(b.state_dict())


def test_pgd_constraints_and_monotone_closure():
    X, C, donors = planted()
    net, _ = train_net(X, C, CFG, seed=0, epochs=60)
    x = torch.from_numpy(X[:40])
    c_t = torch.from_numpy(C[pick_targets(donors[:40])])
    gs = []
    for e in (0.0, 0.5, 2.0, 8.0):
        eps = torch.full((40,), e)
        d = pgd(net, x, c_t, eps, steps=100)
        assert (d.norm(dim=1) <= eps + 1e-4).all() and ((x + d) >= -1e-6).all()
        gs.append(np.median(closure(net, x, d, c_t)))
    assert gs[0] == 0 and all(b >= a - 0.02 for a, b in zip(gs, gs[1:])), gs


def test_single_output_linear_eps50_matches_exact():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(1, 20)).astype(np.float32)
    probe = LinearProbe(w, np.zeros(1, np.float32))
    x = torch.from_numpy((50 + rng.random((25, 20))).astype(np.float32))       # far inside x >= 0
    c_t = torch.from_numpy((probe(x).numpy() + rng.normal(0, 3, (25, 1))).astype(np.float32))
    scale = 10.0
    exact = 0.5 * np.abs(c_t.numpy() - probe(x).numpy()).ravel() / np.linalg.norm(w) / scale
    got = eps50(probe, x, c_t, scale, steps=200)
    ok = (exact > 1e-3) & (exact < 2)
    assert np.allclose(got[ok], exact[ok], rtol=0.01), np.c_[got[ok], exact[ok]][:5]


def test_multi_output_pgd_eps50_within_closed_form_bound():
    rng = np.random.default_rng(2)
    W = rng.normal(size=(3, 20)).astype(np.float32)
    probe = LinearProbe(W, np.zeros(3, np.float32))
    x = torch.from_numpy((50 + rng.random((20, 20))).astype(np.float32))
    c_t = torch.from_numpy((probe(x).numpy() + rng.normal(0, 3, (20, 3))).astype(np.float32))
    got = eps50(probe, x, c_t, 10.0)
    bound = closed_form_eps50(probe, x, c_t, 10.0)
    ok = bound < 2
    assert (got[ok] <= bound[ok] * 1.02).all()


def test_transfer_is_zero_in_the_probes_null_space():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(3, 20)).astype(np.float32)
    probe = LinearProbe(W, np.zeros(3, np.float32))
    _, _, vt = np.linalg.svd(W)
    null = vt[3:]                                         # orthonormal basis of W's null space
    x = torch.from_numpy((50 + rng.random((10, 20))).astype(np.float32))
    delta = torch.from_numpy((rng.normal(size=(10, 17)) @ null).astype(np.float32))
    c_t = torch.from_numpy(rng.normal(size=(10, 3)).astype(np.float32))
    assert np.allclose(closure(probe, x, delta, c_t), 0, atol=1e-4)


def test_low_variance_genes_make_full_scaling_exploitable():
    # 10 near-constant genes among 40: a minority, so the median gene SD (the floor) is set by the
    # ordinary genes. With near-constant genes in the majority the median itself is tiny and floor
    # behaves like full (true of the real single-cell panel: 74% of genes have SD < 0.05).
    X, C, donors = planted(low_var_genes=10)
    x, c_t = torch.from_numpy(X[:40]), torch.from_numpy(C[pick_targets(donors[:40])])
    scale = float(np.median(np.linalg.norm(X - X.mean(0), axis=1)))
    e = {}
    for s in ("full", "center", "floor"):
        net, _ = train_net(X, C, {**CFG, "scaling": s}, seed=0, epochs=100)
        e[s] = float(np.median(eps50(net, x, c_t, scale, steps=100, restarts=1)))
    assert e["full"] < 0.5 * e["center"] and e["full"] < 0.5 * e["floor"], e


def test_eps50_is_found_even_when_large_budgets_saturate():
    # The full-scaling planted model: PGD at eps 1-2 stalls (softmax saturation), but it's easily
    # fooled at small eps. eps50 must be small and finite, and the envelope must be monotone.
    X, C, donors = planted(low_var_genes=10)
    x, c_t = torch.from_numpy(X[:40]), torch.from_numpy(C[pick_targets(donors[:40])])
    scale = float(np.median(np.linalg.norm(X - X.mean(0), axis=1)))
    net, _ = train_net(X, C, {**CFG, "scaling": "full"}, seed=0, epochs=100)
    e = eps50(net, x, c_t, scale, steps=100, restarts=1)
    assert np.isfinite(e).all() and np.median(e) < 0.05, np.median(e)


def _tiny_folds(X, C, donors):
    folds = []
    for tr, te in inner_folds(donors, (np.arange(len(donors)) // 4) % 2, 3):
        folds.append({"X_tr": X[tr], "C_tr": C[tr], "X_te": X[te], "C_te": C[te], "donors_te": donors[te],
                      "translations": {"same": X[te], "shifted": X[te] + 3.0},
                      "probe_r2_trans": {"same": 0.5, "shifted": -4.0}, "scale": 1.0})
    return folds


def test_run_config_epoch_and_robustness_rules():
    X, C, donors = planted(n=120)
    folds = _tiny_folds(X, C, donors)
    r = run_config(CFG, folds, epochs=40)
    # one epoch for the config: the argmax of the fold-averaged curve, every fold scored there
    assert r["epochs"] == int(np.argmax(r["curve"])) + 1
    assert abs(r["oof_comp_r2"] - np.mean(r["fold_comp_r2"])) < 1e-12
    assert abs(r["oof_comp_r2"] - r["curve"][r["epochs"] - 1]) < 1e-12
    # clipping: the probe's -4.0 on "shifted" counts as 0, so d = max(net, 0) - 0 >= 0 there.
    # Without clipping d would be net + 4.0, so this also checks that it isn't net - (-4.0).
    assert 0 <= r["robust_d:shifted"] <= 1 + 1e-9
    assert -1 - 1e-9 <= r["robust_d:same"] <= 1 + 1e-9
    assert r["robust_pass"] == (r["robust_mean_d"] >= -0.10 and min(r["robust_d:same"], r["robust_d:shifted"]) >= -0.20)


def test_selection_model_is_reproduced_by_retraining_to_its_epoch():
    X, C, _ = planted(n=120)
    cfg = {**CFG, "hidden": 8, "dropout": 0.2, "noise": 0.1, "loss": "kl"}
    _, hist = train_net(X[:90], C[:90], cfg, seed=0, epochs=30, eval_sets={"held": (X[90:], C[90:])})
    for e in (5, 17, 30):
        net, _ = train_net(X[:90], C[:90], cfg, seed=0, epochs=e)
        with torch.no_grad():
            assert comp_r2(net(torch.from_numpy(X[90:])).numpy(), C[90:]) == hist["held"][e - 1]


def test_attack_summary_envelope_is_monotone_and_reports_cold_runs():
    X, C, donors = planted(low_var_genes=10)
    net, _ = train_net(X, C, {**CFG, "scaling": "full"}, seed=0, epochs=60)
    x, t = X[:24], pick_targets(donors[:24])
    probe = LinearProbe(np.zeros((3, X.shape[1]), np.float32), np.full(3, 1 / 3, np.float32))
    pcs = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:5]
    res = attack_summary(net, x, C[t], X[t], 1.0, {"p": probe}, pcs, X.std(0) < np.median(X.std(0)),
                         [f"G{i}" for i in range(X.shape[1])], X.mean(0))
    env = [res["by_eps"][e]["median_g"] for e in sorted(res["by_eps"])]
    assert all(b >= a - 1e-9 for a, b in zip(env, env[1:])), env
    assert all(res["by_eps"][e]["median_g_cold_this_budget"] <= res["by_eps"][e]["median_g"] + 1e-9 for e in res["by_eps"])
    assert res["by_eps"][0.03]["transfer_p"] == 0.0          # a constant probe can't be moved


def test_marker_probe_save_load_and_tamper():
    import json, tempfile
    from pathlib import Path
    from dgm.models import list_hash
    genes = [g for ms in MARKERS.values() for g in ms] + ["X1", "X2"]
    rng = np.random.default_rng(5)
    X = rng.random((60, len(genes))).astype(np.float32)
    C = rng.dirichlet(np.ones(6), 60)
    donors = np.repeat([f"D{i}" for i in range(20)], 3)
    mp = MarkerProbe(genes).fit(X, C, donors)
    with tempfile.TemporaryDirectory() as tmp:
        reg = Path(tmp) / "m.json"
        prov = {"genes_hash": list_hash(genes), "classes": list("abcdef"), "pairs_hash": "P"}
        save_marker(mp, "marker_test", tmp, reg, prov)
        back = load_marker("marker_test", tmp, reg, genes, list("abcdef"), "P")
        assert np.allclose(back.predict(X), mp.predict(X))
        r = json.loads(reg.read_text()); r["marker_test"]["canonical_hash"] = "0" * 64; reg.write_text(json.dumps(r))
        try:
            load_marker("marker_test", tmp, reg, genes, list("abcdef"), "P")
            raise AssertionError("accepted a tampered marker probe")
        except ValueError:
            pass


def test_freeze_refuses_non_train_rows():
    X, C, _ = planted(n=12)
    pairs = SimpleNamespace(split=np.array(["train"] * 10 + ["val"] * 2), x_bulk=X, composition=C)
    nets, *_ = freeze_nets(pairs, np.arange(10), "bulk", CFG, epochs=2, seeds=(0,))
    assert len(nets) == 1
    try:
        freeze_nets(pairs, np.arange(12), "bulk", CFG, epochs=2, seeds=(0,))
        raise AssertionError("froze on val rows")
    except AssertionError as e:
        assert "non-train" in str(e)


def test_paired_bootstrap():
    rng = np.random.default_rng(4)
    true = rng.dirichlet(np.ones(3), 60)
    donors = np.repeat([f"D{i}" for i in range(20)], 3)
    good = true + rng.normal(0, 0.01, true.shape)
    d, lo, hi = paired_bootstrap_diff(good, good, true, donors, n_boot=200)
    assert d == 0 and lo <= 0 <= hi
    bad = np.tile(true.mean(0), (60, 1)) + rng.normal(0, 0.05, true.shape)
    d, lo, hi = paired_bootstrap_diff(bad, good, true, donors, n_boot=200)
    assert d < 0 and hi < 0
    assert abs(comp_r2(true, true) - 1) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
