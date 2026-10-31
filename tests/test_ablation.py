"""Known-answer checks for the Phase 4 options (notes/phase4_spec.md): python -m tests.test_ablation
Set PHASE4_SLOW=1 to also check that the frontier refit on the full halves reproduces Phase 1's."""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import torch

from dgm import ablation as ab
from dgm import composition as comp
from dgm.models import canonical_hash
from dgm.translation import config, init_affine, train
from tests.synthetic import paired

comp.deterministic()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE3_COMMIT = "4562011"            # the pushed, locked Phase 3 code
D = paired(n_donors=60, draws=3, g=40, seed=0)
tr_rows = np.flatnonzero(D["train"])
XS = D["src"][tr_rows[: len(tr_rows) // 2]].astype(np.float32)
XB = D["tgt"][tr_rows[len(tr_rows) // 2:]].astype(np.float32)
INIT = {"S->B": init_affine("mean shift", XS, XB), "B->S": init_affine("ComBat", XB, XS)}
SRC_V = torch.as_tensor(D["src"][D["test"]], dtype=torch.float32)


def evaluate(step, sb, bs):
    """A cheap stand-in for the validation evaluator: deterministic, uses no random numbers."""
    y = sb(SRC_V).numpy()
    return {"margin": -float(np.abs(y - D["tgt"][D["test"]]).mean())}


def task_dict(cs=None):
    k = D["comp"].shape[1]
    cb = torch.nn.Sequential(torch.nn.Linear(40, k), torch.nn.Softmax(-1))
    torch.manual_seed(1); [p.requires_grad_(False) for p in cb.parameters()]
    return {"C_B": cb, "C_S": cs or cb, "comp_s": D["comp"][tr_rows[: len(tr_rows) // 2]]}


def phase3_module():
    """dgm/translation.py as committed for Phase 3, imported under another name."""
    src = subprocess.run(["git", "show", f"{PHASE3_COMMIT}:dgm/translation.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    spec = importlib.util.spec_from_loader("translation_phase3", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, "translation_phase3.py", "exec"), mod.__dict__)
    sys.modules["translation_phase3"] = mod
    return mod


def run(cfg_kw, task=None, seed=0):
    cfg = config(**{"hidden": 32, "steps": 60, "eval_every": 10, "log_every": 10, **cfg_kw})
    return train(XS, XB, INIT, cfg, seed, evaluate=evaluate, task=task)


def test_defaults_reproduce_phase3():
    """With every Phase 4 option at its default, training is bit-identical to the Phase 3 code:
    same chosen generators, discriminators, margins and chosen step (with and without a task loss)."""
    p3 = phase3_module()
    for task_kw, task in (({}, None), ({"lambda_task": 1.0, "warmup": 0.1, "ramp": 0.1}, task_dict())):
        base = dict(hidden=32, steps=60, eval_every=10, log_every=10, identity=True, lambda_cyc=20.0, **task_kw)
        old = p3.train(XS, XB, INIT, p3.config(**base), 3, evaluate=evaluate, task=task)
        new = train(XS, XB, INIT, config(**base), 3, evaluate=evaluate, task=task)
        assert canonical_hash(old["state"]) == canonical_hash(new["state"])
        assert canonical_hash(old["disc_state"]) == canonical_hash(new["disc_state"])
        assert [e["margin"] for e in old["evals"]] == [e["margin"] for e in new["evals"]]
        assert old["best_step"] == new["best_step"]


def test_adversarial_off_keeps_every_random_draw():
    """lambda_adv = 0: the generators no longer follow the discriminators, the discriminators still
    train, and the run makes exactly the same random draws (same batches and noise)."""
    on, off = run({"identity": True}), run({"identity": True, "lambda_adv": 0.0})
    assert on["rng_end"] == off["rng_end"]
    assert canonical_hash(on["state"]) != canonical_hash(off["state"])
    init_d = run({"identity": True, "lambda_adv": 0.0, "lr_d": 0.0})["disc_state"]
    assert canonical_hash(off["disc_state"]) != canonical_hash(init_d)


def test_affine_only_has_no_residual():
    res = run({"affine_only": True})
    assert sorted(res["state"]) == ["bs.a", "bs.b", "sb.a", "sb.b"]
    start = train(XS, XB, INIT, config(hidden=32, steps=0, affine_only=True), 0)["state"]
    assert torch.allclose(start["sb.b"], torch.as_tensor(INIT["S->B"][1], dtype=torch.float32))


def test_ema_changes_nothing_but_adds_scores():
    plain, with_ema = run({"identity": True}), run({"identity": True, "ema": 0.9})
    assert canonical_hash(plain["state"]) == canonical_hash(with_ema["state"])
    assert canonical_hash(plain["disc_state"]) == canonical_hash(with_ema["disc_state"])
    assert [e["margin"] for e in plain["evals"]] == [e["margin"] for e in with_ema["evals"]]
    assert plain["rng_end"] == with_ema["rng_end"]
    assert all("ema" in e for e in with_ema["evals"]) and with_ema["ema_state"] is not None


def test_identity_weight():
    """lambda_id = None is Phase 3's 0.5 * lambda_cyc; an explicit weight keeps the identity loss
    when the cycle loss is off."""
    tied, explicit = run({"identity": True, "lambda_cyc": 20.0}), run({"identity": True, "lambda_cyc": 20.0, "lambda_id": 10.0})
    assert canonical_hash(tied["state"]) == canonical_hash(explicit["state"])
    no_cyc = run({"identity": True, "lambda_cyc": 0.0, "lambda_id": 10.0})
    assert no_cyc["log"][-1]["cyc"] > 0 and canonical_hash(no_cyc["state"]) != canonical_hash(run({"identity": True, "lambda_cyc": 0.0})["state"])


def test_keep_states():
    res = run({"keep_states": (20, 60)})
    assert sorted(res["kept"]) == [20, 60]
    assert canonical_hash(res["kept"][20]) != canonical_hash(res["kept"][60])


def test_task_terms():
    """task_terms "sb" never calls C_S; "bs" and "both" do; "sb" and "bs" give different runs."""
    def boom(x):
        raise RuntimeError("C_S called")
    kw = {"lambda_task": 1.0, "warmup": 0.0, "ramp": 0.0}
    sb = run({**kw, "task_terms": "sb"}, task=task_dict(cs=boom))
    for terms in ("bs", "both"):
        try:
            run({**kw, "task_terms": terms}, task=task_dict(cs=boom))
        except RuntimeError:
            continue
        raise AssertionError(f"task_terms={terms} did not use C_S")
    bs = run({**kw, "task_terms": "bs"}, task=task_dict())
    assert canonical_hash(sb["state"]) != canonical_hash(bs["state"])


def test_subsets():
    """Subsets lie inside the halves; "donors" keeps all draws of about half of each age group's
    donors; "draws" keeps every donor and exactly the same number of rows; seeds differ."""
    from dgm.data import load_pairs, train_halves
    P = load_pairs()
    half_a, half_b = train_halves(P, 0)
    for seed in (3, 4):
        don = ab.subset_rows(P, half_a, half_b, "donors", seed)
        dra = ab.subset_rows(P, half_a, half_b, "draws", seed)
        for full, d, r in zip((half_a, half_b), don, dra):
            assert set(d) <= set(full) and set(r) <= set(full)
            assert len(d) == len(r)
            kept = np.unique(P.donor_ids[d])
            assert set(d) == set(full[np.isin(P.donor_ids[full], kept)])          # all draws of kept donors
            for g in np.unique(P.age_group[full]):
                n_g = len(np.unique(P.donor_ids[full][P.age_group[full] == g]))
                assert len(np.unique(P.donor_ids[d][P.age_group[d] == g])) == max(1, round(0.5 * n_g))
            assert set(np.unique(P.donor_ids[r])) == set(np.unique(P.donor_ids[full]))   # every donor
        assert not set(P.donor_ids[don[0]]) & set(P.donor_ids[don[1]])
    assert not np.array_equal(ab.subset_rows(P, half_a, half_b, "donors", 3)[0], ab.subset_rows(P, half_a, half_b, "donors", 4)[0])
    assert all(np.array_equal(x, y) for x, y in zip(ab.subset_rows(P, half_a, half_b, "draws", 3), ab.subset_rows(P, half_a, half_b, "draws", 3)))


def test_half_nets_never_see_half_b():
    """B3's networks (results/phase4/models.json, if trained) were fit on exactly halving 0's half A:
    no row and no donor of half B."""
    import json
    from dgm import paths
    from dgm.data import load_pairs, train_halves
    from dgm.models import list_hash
    reg = paths.REPO / "results" / "phase4" / "models.json"
    if not reg.exists():
        print("  skipped (half-A networks not trained yet)")
        return
    P = load_pairs()
    half_a, half_b = train_halves(P, 0)
    entries = json.loads(reg.read_text())
    for name in ("C_bulk_halfA0", "C_sc_halfA0"):
        assert entries[name]["half_a_rows_hash"] == list_hash(np.sort(half_a).tolist()), name
    assert not set(half_a) & set(half_b) and not set(P.donor_ids[half_a]) & set(P.donor_ids[half_b])


def test_move_decomposition():
    rng = np.random.default_rng(0)
    start, tgt = rng.normal(size=(5, 8)), rng.normal(size=(5, 8))
    d = ab.move_decomposition(start, start + 0.5 * (tgt - start), tgt)
    assert np.allclose(d["cos"], 1) and np.allclose(d["across"], 0, atol=1e-6) and np.allclose(d["along"], 0.5 * d["needed"])
    perp = np.array([[1.0, 0], [0, 1]]), np.array([[0.0, 1], [1, 0]])
    d = ab.move_decomposition(np.zeros((2, 2)), perp[0], perp[1])
    assert np.allclose(d["cos"], 0) and np.allclose(d["across"], 1)


def test_frontier_refit_reproduces_phase1():
    """Slow (PHASE4_SLOW=1): the frontier refit on halving 0's full halves equals Phase 1's."""
    if not os.environ.get("PHASE4_SLOW"):
        print("  skipped (set PHASE4_SLOW=1)")
        return
    import pandas as pd
    from dgm import paths
    from dgm.translation import SoundLife
    ctx = SoundLife(0)
    pts = ab.frontier_points(ctx.pairs, ctx.half_a, ctx.half_b, ctx.ev, ctx.va, ctx.classes)
    for d in pts:
        ref = ctx.frontier[d][["a", "b", "w", "r2_centered", "realism"]].reset_index(drop=True)
        new = pts[d][["a", "b", "w", "r2_centered", "realism"]].reset_index(drop=True)
        assert (ref[["a", "b", "w"]] == new[["a", "b", "w"]]).all().all()
        diff = (ref[["r2_centered", "realism"]] - new[["r2_centered", "realism"]]).abs().max()
        print(f"  {d}: max |difference| from Phase 1: centered R2 {diff['r2_centered']:.2e}, realism {diff['realism']:.2e}")
        assert (diff < 1e-3).all(), diff        # Phase 1 ran at a different thread count: ties in the AUROC can flip


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
