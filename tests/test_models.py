"""Known-answer checks for dgm.models (spec: notes/phase2_spec.md, Tests): python -m tests.test_models"""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from dgm.models import (SCALINGS, CompositionNet, Ensemble, Frozen, canonical_hash, input_scale, list_hash, load_frozen,
                        save_frozen)

torch.use_deterministic_algorithms(True)
RNG = np.random.default_rng(0)
X = (RNG.random((40, 12)) * 3).astype(np.float32)
PRIOR = np.array([0.6, 0.3, 0.07, 0.03], np.float32)


def net(scaling="center", hidden=8):
    mean, inv = input_scale(X, scaling)
    return CompositionNet(mean, inv, PRIOR, hidden=hidden)


def test_outputs_on_the_simplex():
    for h in (8, 0):
        n = net(hidden=h)
        torch.nn.init.normal_(n.head.weight)          # move off the initial state
        out = n(torch.from_numpy(X)).detach().numpy()
        assert (out >= 0).all() and np.allclose(out.sum(1), 1, atol=1e-6)
    e = Ensemble([net(), net(hidden=0)])
    out = e(torch.from_numpy(X)).detach().numpy()
    assert np.allclose(out.sum(1), 1, atol=1e-6)


def test_initial_prediction_is_the_training_mean_for_every_arm():
    for scaling in SCALINGS:
        for h in (8, 0):
            out = net(scaling, h)(torch.from_numpy(RNG.random((5, 12)).astype(np.float32) * 10)).detach().numpy()
            assert np.allclose(out, PRIOR, atol=1e-6), (scaling, h)


def test_input_scale_uses_only_the_given_rows():
    mean, inv = input_scale(X[:20], "full")
    assert np.allclose(mean, X[:20].mean(0)) and np.allclose(inv, 1 / X[:20].std(0))
    _, inv_floor = input_scale(X[:20], "floor")
    assert (1 / inv_floor >= np.median(X[:20].std(0)) - 1e-7).all()
    _, inv_c = input_scale(X[:20], "center")
    assert (inv_c == 1).all()


def test_canonical_hash_stable_and_sensitive():
    n = net()
    h1 = canonical_hash(n.state_dict())
    with tempfile.TemporaryDirectory() as tmp:
        torch.save(n.state_dict(), Path(tmp) / "a.pt")
        n2 = net()
        n2.load_state_dict(torch.load(Path(tmp) / "a.pt"))
        torch.save(n2.state_dict(), Path(tmp) / "b.pt")
        n3 = net()
        n3.load_state_dict(torch.load(Path(tmp) / "b.pt"))
    assert canonical_hash(n3.state_dict()) == h1
    sd = {k: v.clone() for k, v in n.state_dict().items()}
    w = sd["head.bias"]
    w[0] = torch.nextafter(w[0], torch.tensor(np.inf))   # one ulp
    assert canonical_hash(sd) != h1


def test_load_frozen_verifies_and_freezes():
    genes, classes = [f"G{i}" for i in range(12)], ["a", "b", "c", "d"]
    with tempfile.TemporaryDirectory() as tmp:
        d, reg = Path(tmp), Path(tmp) / "models.json"
        mean, inv = input_scale(X, "center")
        cfg = {"hidden": 8, "dropout": 0.0}
        nets = [CompositionNet(mean, inv, PRIOR, hidden=8) for _ in range(2)]
        for n in nets:
            torch.nn.init.normal_(n.head.weight)
        prov = {"genes_hash": list_hash(genes), "classes": classes, "pairs_hash": "P"}
        save_frozen("C_test", nets, cfg, mean, inv, PRIOR, d, reg, prov)
        m = load_frozen("C_test", d, reg, genes, classes, "P")
        x = torch.from_numpy(X).requires_grad_(True)
        assert np.allclose(m(x).detach().numpy(), Ensemble(nets)(torch.from_numpy(X)).detach().numpy(), atol=1e-6)
        assert not any(p.requires_grad for p in m.parameters())
        parent = torch.nn.Sequential(m)
        parent.train()
        assert not m.training and not m.model.training and all(not mod.training for mod in m.modules())
        m(x).sum().backward()
        assert x.grad is not None and all(p.grad is None for p in m.parameters())
        for bad in ({"genes": genes[::-1]}, {"classes": classes[::-1]}, {"pairs_digest": "Q"}):
            args = {"genes": genes, "classes": classes, "pairs_digest": "P", **bad}
            try:
                load_frozen("C_test", d, reg, **args)
                raise AssertionError(f"accepted {bad}")
            except ValueError:
                pass
        r0 = json.loads(reg.read_text())
        for field, bad in (("config", {"hidden": 16, "dropout": 0.0}), ("n_members", 3)):
            r = json.loads(json.dumps(r0)); r["C_test"][field] = bad
            reg.write_text(json.dumps(r))
            try:
                load_frozen("C_test", d, reg, genes, classes, "P")
                raise AssertionError(f"accepted a mismatched {field}")
            except ValueError:
                pass
        r = json.loads(json.dumps(r0))
        r["C_test"]["members"][0]["canonical_hash"] = "0" * 64
        reg.write_text(json.dumps(r))
        try:
            load_frozen("C_test", d, reg, genes, classes, "P")
            raise AssertionError("accepted a wrong canonical hash")
        except ValueError:
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
