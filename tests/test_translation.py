"""Known-answer checks for dgm.translation (notes/phase3_spec.md): python -m tests.test_translation"""
import numpy as np
import torch

from dgm import composition as comp
from dgm.baselines import ComBat, MeanShift
from dgm.models import canonical_hash
from dgm.translation import Generator, assert_halves_only, config, init_affine, train
from tests.synthetic import paired

comp.deterministic()
D = paired(n_donors=200, draws=3, g=40, seed=0)
tr = np.flatnonzero(D["train"])
half_s, half_b = tr[: len(tr) // 2], tr[len(tr) // 2:]       # unpaired: source from one half of the donors, target from the other
XS, XB = D["src"][half_s].astype(np.float32), D["tgt"][half_b].astype(np.float32)
TEST = np.flatnonzero(D["test"])
SMALL = config(hidden=64, steps=20, eval_every=10 ** 9, log_every=10)


def init_real():
    return {"S->B": init_affine("mean shift", XS, XB), "B->S": init_affine("ComBat", XB, XS)}


def centered_r2(yh, y, mu):
    return float(np.median(1 - ((yh - y) ** 2).sum(1) / ((y - mu) ** 2).sum(1)))


def test_generators_start_at_their_baselines():
    init = init_real()
    sb, bs = Generator(*init["S->B"], hidden=64), Generator(*init["B->S"], hidden=64)
    ms, cb = MeanShift().fit(XS, XB), ComBat().fit(XB, XS)
    x_s = torch.as_tensor(D["src"][TEST], dtype=torch.float32)
    x_b = torch.as_tensor(D["tgt"][TEST], dtype=torch.float32)
    with torch.no_grad():
        assert np.abs(sb(x_s).numpy() - ms.translate(D["src"][TEST])).max() < 1e-5
        assert np.abs(bs(x_b).numpy() - cb.translate(D["tgt"][TEST])).max() < 1e-5
        # the cycle at initialization is the composition of the two baselines
        cyc = bs(sb(x_s)).numpy()
        assert np.abs(cyc - cb.translate(ms.translate(D["src"][TEST]))).max() < 1e-4
        cyc = sb(bs(x_b)).numpy()
        assert np.abs(cyc - ms.translate(cb.translate(D["tgt"][TEST]))).max() < 1e-4


def test_one_step_changes_generators_and_discriminators():
    init = init_real()
    cfg = config(hidden=64, steps=1, eval_every=10 ** 9)
    res = train(XS, XB, init, cfg, seed=0)
    start = train(XS, XB, init, dict(cfg, steps=0), seed=0)          # same seed, no step: the initial weights
    changed = [k for k, v in res["state"].items() if not torch.equal(v, start["state"][k])]
    assert any(k.startswith("sb.") for k in changed) and any(k.startswith("bs.") for k in changed), changed
    # spectral norm's u/v vectors move on any forward pass, so compare the underlying weights only
    orig = [k for k in res["disc_state"] if k.endswith(".original")]
    assert orig and any(not torch.equal(res["disc_state"][k], start["disc_state"][k]) for k in orig)
    frozen_d = train(XS, XB, init, dict(cfg, lr_d=0.0), seed=0)       # with lr_d = 0 they must not move
    assert all(torch.equal(frozen_d["disc_state"][k], start["disc_state"][k]) for k in orig)


def test_training_is_deterministic():
    a = train(XS, XB, init_real(), SMALL, seed=3)
    b = train(XS, XB, init_real(), SMALL, seed=3)
    c = train(XS, XB, init_real(), SMALL, seed=4)
    assert canonical_hash(a["state"]) == canonical_hash(b["state"]) != canonical_hash(c["state"])
    assert a["log"] == b["log"]


def test_selection_uses_the_neighbor_averaged_margin():
    margins = {0: 0.0, 10: 0.1, 20: 0.9, 30: 0.0, 40: 0.5, 50: 0.5, 60: 0.6, 70: 0.0}   # a lone spike at 20
    res = train(XS, XB, init_real(), config(hidden=64, steps=70, eval_every=10, log_every=10), seed=0,
                evaluate=lambda step, sb, bs: {"margin": margins[step]})
    avg = {e["step"]: e.get("margin_avg") for e in res["evals"]}
    assert abs(avg[20] - 1.0 / 3) < 1e-12 and abs(avg[50] - 1.6 / 3) < 1e-12
    assert res["best_step"] == 50, res["best_step"]           # not the spike, and never step 0 or the last step


def test_task_runs_cannot_select_before_the_task_loss_is_at_full_weight():
    """Before warmup + ramp, an M2 run is (or is close to) the M1 run; a planted margin spike there
    must not be chosen."""
    c0 = torch.tensor([0.5, 0.3, 0.2])
    const = lambda x: c0.expand(x.shape[0], 3) + 0 * x[:, :1]
    task = {"C_B": const, "C_S": const, "comp_s": np.tile(c0.numpy(), (len(XS), 1))}
    margins = {k * 10: 0.0 for k in range(11)}
    margins.update({10: 1.0, 20: 1.0, 30: 1.0, 70: 0.2})              # the spike sits inside the warm-up (steps < 40)
    cfg = config(hidden=64, steps=100, eval_every=10, log_every=10, lambda_task=1.0, warmup=0.2, ramp=0.2)
    res = train(XS, XB, init_real(), cfg, seed=0, evaluate=lambda step, sb, bs: {"margin": margins[step]}, task=task)
    assert res["best_step"] >= 40, res["best_step"]
    m1 = train(XS, XB, init_real(), dict(cfg, lambda_task=0.0), seed=0, evaluate=lambda step, sb, bs: {"margin": margins[step]})
    assert m1["best_step"] == 20                                      # without a task loss the spike is eligible


def test_task_loss_is_zero_when_composition_is_read_exactly():
    c0 = torch.tensor([0.5, 0.3, 0.2])
    const = lambda x: c0.expand(x.shape[0], 3) + 0 * x[:, :1]  # depends on x only through a zero term
    task = {"C_B": const, "C_S": const, "comp_s": np.tile(c0.numpy(), (len(XS), 1))}
    res = train(XS, XB, init_real(), config(hidden=64, steps=40, lambda_task=1.0, warmup=0.0, ramp=0.0, log_every=10), seed=0, task=task)
    assert all(r["task_weight"] > 0 and r["task"] == 0 for r in res["log"])


def _recovery(xs, xb, src_test, tgt_test, hidden, steps=20000):
    init = {"S->B": init_affine("zero", xs, xb), "B->S": init_affine("zero", xb, xs)}
    res = train(xs, xb, init, config(hidden=hidden, steps=steps, eval_every=10 ** 9, log_every=10 ** 6), seed=0)
    from dgm.translation import load_generators
    sb, _ = load_generators(res["state"], xs.shape[1], hidden)
    with torch.no_grad():
        yh = sb(torch.as_tensor(src_test, dtype=torch.float32)).numpy()
    mu = xb.mean(0)
    return centered_r2(yh, tgt_test, mu), centered_r2(ComBat().fit(xs, xb).translate(src_test), tgt_test, mu)


def test_unpaired_recovery_matches_combat():
    """Domains differ by a per-gene affine map with positive scales, on correlated marginals; no
    pairs; the generator starts at a = 1, zero bias, and trains for the spec's 20,000 steps with no
    checkpoint selection. M1's centered R2 must be at least ComBat's minus 0.05 (one-sided).
    Run at hidden width 64 (40 genes here). At the Sound Life width of 512 the same problem drifts
    away from the affine solution as training continues (0.921 at 3,000 steps, 0.853 at 20,000),
    which is why Sound Life runs select checkpoints on validation (Phase 3 notebook, §3.2)."""
    r2_gan, r2_combat = _recovery(XS, XB, D["src"][TEST], D["tgt"][TEST], hidden=64)
    print(f"  affine case: CycleGAN R2 {r2_gan:.3f}, ComBat {r2_combat:.3f}")
    assert r2_gan >= r2_combat - 0.05, (r2_gan, r2_combat)


def test_low_rank_case_is_reported():
    """Exploratory, no threshold: the source also carries a rank-2 cross-gene distortion of the
    target, which a per-gene map (ComBat) cannot undo."""
    rng = np.random.default_rng(1)
    L = rng.normal(0, 0.3, (40, 2)) @ rng.normal(0, 1, (2, 40))
    src = np.maximum(D["src"] + (D["tgt"] - D["tgt"].mean(0)) @ L, 0).astype(np.float32)
    r2_gan, r2_combat = _recovery(src[half_s], XB, src[TEST], D["tgt"][TEST], hidden=64)
    print(f"  low-rank case (reported only): CycleGAN R2 {r2_gan:.3f}, ComBat {r2_combat:.3f}")


def test_guard_rejects_rows_outside_the_halves():
    from dgm.data import load_pairs, train_halves
    from dgm.paths import PAIRS_FILE
    if not PAIRS_FILE.exists():
        return
    p = load_pairs()
    a, b = train_halves(p, 0)
    assert_halves_only(p, a, b, 0)
    for bad_s, bad_b in ((np.r_[a, p.idx("val")[:1]], b), (a, np.r_[b, a[:1]])):
        try:
            assert_halves_only(p, bad_s, bad_b, 0)
        except AssertionError:
            continue
        raise AssertionError("guard accepted a row outside the halves")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
