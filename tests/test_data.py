"""dgm.data schema checks: a synthetic round trip, malformed files rejected, and the real
pairs file when present. Plain asserts (no pytest in the env): python -m tests.test_data"""
import tempfile
from pathlib import Path

import numpy as np

from dgm.data import HALVING_SEEDS, load_pairs, save_pairs, train_halves
from dgm.paths import PAIRS_FILE


def synthetic(n=12, g=5, k=3):
    rng = np.random.default_rng(0)
    comp = rng.random((n, k))
    donors = np.repeat([f"D{i}" for i in range(n // 2)], 2).astype("<U8")
    age = np.repeat(np.array([30, 60, 31, 61, 32, 62], dtype=np.float32), 2)    # constant within a donor
    split = np.array(["train"] * 8 + ["val"] * 2 + ["test"] * 2, dtype="<U8")
    return dict(x_sc=rng.random((n, g)).astype(np.float32), x_bulk=rng.random((n, g)).astype(np.float32),
                composition=(comp / comp.sum(1, keepdims=True)).astype(np.float32), genes=[f"G{i}" for i in range(g)],
                l1_classes=[f"C{i}" for i in range(k)], sample_ids=[f"S{i}" for i in range(n)], donor_ids=donors,
                age=age, age_group=(age >= 55).astype(np.int64), split=split)


def test_round_trip():
    a = synthetic()
    with tempfile.TemporaryDirectory() as tmp:
        save_pairs(Path(tmp) / "p.npz", **a)
        p = load_pairs(Path(tmp) / "p.npz")
    assert len(p) == 12 and p.genes.dtype.kind == "U" and len(p.idx("train")) == 8
    for name in ("x_sc", "x_bulk", "composition", "age", "age_group"):
        assert np.array_equal(getattr(p, name), a[name]), name
    assert p.sample_ids.tolist() == a["sample_ids"] and p.split.tolist() == a["split"].tolist()


def rejection(arrays):
    """The assertion message save_pairs raises for these arrays, or None if it accepts them."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            save_pairs(Path(tmp) / "p.npz", **arrays)
        except AssertionError as e:
            return str(e)
    return None


def test_malformed_rejected():
    # case -> (how to break the file, words the intended check's message must contain)
    cases = {
        "donor in two splits": (lambda a: a["donor_ids"].__setitem__(8, "D0"), "two splits"),    # row 8 is val, same age group as D0
        "composition off the simplex": (lambda a: a.update(composition=a["composition"] * 2), "simplex"),
        "shape mismatch": (lambda a: a.update(x_sc=a["x_sc"][:, :4]), "shape"),
        "NaN in x_bulk": (lambda a: a["x_bulk"].__setitem__((0, 0), np.nan), "non-finite expression"),
        "raw-count scale": (lambda a: a.update(x_sc=a["x_sc"] * 1e6), "log1p(CP10k) range"),
        "negative values": (lambda a: a["x_sc"].__setitem__((0, 0), -1.0), "log1p(CP10k) range"),
        "NaN age": (lambda a: a["age"].__setitem__(0, np.nan), "age has non-finite"),
        "age_group inverted": (lambda a: a.update(age_group=1 - a["age_group"]), "disagrees with age"),
        "float16 profiles": (lambda a: a.update(x_sc=a["x_sc"].astype(np.float16)), "expected float32"),
        "duplicate sample ids": (lambda a: a.update(sample_ids=["S0"] * 12), "duplicate sample_ids"),
        "duplicate genes": (lambda a: a.update(genes=["G0"] * 5), "duplicate genes"),
        "missing field": (lambda a: a.pop("split"), "fields differ"),
        "unknown split label": (lambda a: a["split"].__setitem__(0, "holdout"), "split labels"),
    }
    for name, (break_it, expected) in cases.items():
        a = synthetic()
        break_it(a)
        msg = rejection(a)
        assert msg is not None, f"accepted a file with: {name}"
        assert expected in msg, f"{name}: rejected by the wrong check ({msg!r})"


def test_missing_file_and_bad_split():
    try:
        load_pairs(Path("/nonexistent/pairs.npz"))
        raise AssertionError("loaded a missing file")
    except FileNotFoundError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        save_pairs(Path(tmp) / "p.npz", **synthetic())
        p = load_pairs(Path(tmp) / "p.npz")
    try:
        p.idx("bogus")
        raise AssertionError("accepted an unknown split")
    except AssertionError as e:
        assert "bogus" in str(e)


def test_source_hash_ignores_outputs():
    import json
    from dgm.notebook import source_hash
    nb = {"cells": [{"cell_type": "code", "source": ["x = 1\n"], "outputs": [], "execution_count": None, "metadata": {}}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    run = json.loads(json.dumps(nb)); run["cells"][0].update(outputs=[{"output_type": "stream", "text": "1"}], execution_count=3)
    edited = json.loads(json.dumps(nb)); edited["cells"][0]["source"] = ["x = 2\n"]
    assert source_hash(json.dumps(nb)) == source_hash(json.dumps(run)) != source_hash(json.dumps(edited))


def test_real_file():
    if not PAIRS_FILE.exists():
        print("  (skipped: no pairs file; run phase 0)")
        return
    p = load_pairs()
    assert p.x_sc.shape == (836, 5000) and p.composition.shape == (836, 6)
    assert p.l1_classes.tolist() == ["T cell", "Monocyte", "B cell", "NK cell", "DC", "Other"]
    assert [len(p.idx(s)) for s in ("train", "val", "test")] == [582, 135, 119]
    assert [len(set(p.donor_ids[p.idx(s)])) for s in ("train", "val", "test")] == [68, 14, 14]



def test_train_halves():
    """Disjoint donors, together exactly the train split, deterministic, balanced in age and draws."""
    import dataclasses
    from dgm.data import Pairs, validate
    rng = np.random.default_rng(3)
    n_donors = 20
    draws = rng.integers(1, 6, n_donors)
    donors = np.repeat([f"D{i}" for i in range(n_donors)], draws)
    age = np.repeat(np.where(np.arange(n_donors) % 2, 60.0, 30.0), draws).astype(np.float32)
    n = len(donors)
    split = np.where(np.isin(donors, [f"D{i}" for i in range(16)]), "train", np.where(np.isin(donors, ["D16", "D17"]), "val", "test"))
    a = synthetic(n=n)
    a.update(donor_ids=donors, age=age, age_group=(age >= 55).astype(np.int64), split=split.astype("<U8"),
             sample_ids=np.array([f"S{i}" for i in range(n)]), genes=np.array(a["genes"]), l1_classes=np.array(a["l1_classes"]))
    validate(a)
    p = Pairs(**a)
    for seed in HALVING_SEEDS:
        A, B = train_halves(p, seed)
        dA, dB = set(p.donor_ids[A]), set(p.donor_ids[B])
        assert not dA & dB, "a donor is in both halves"
        assert sorted(np.r_[A, B].tolist()) == p.idx("train").tolist(), "halves don't partition the train split"
        assert abs(len(A) - len(B)) <= draws.max(), (len(A), len(B))
        for g in (0, 1):
            nA = sum(p.age_group[p.donor_ids == d][0] == g for d in dA)
            nB = sum(p.age_group[p.donor_ids == d][0] == g for d in dB)
            assert abs(nA - nB) <= 2, (g, nA, nB)
        A2, B2 = train_halves(p, seed)
        assert np.array_equal(A, A2) and np.array_equal(B, B2), "not deterministic"
    assert not np.array_equal(train_halves(p, 0)[0], train_halves(p, 1)[0]), "seeds give the same halving"
    if PAIRS_FILE.exists():
        real = load_pairs()
        A, B = train_halves(real, 0)
        assert not set(real.donor_ids[A]) & set(real.donor_ids[B]) and len(A) + len(B) == len(real.idx("train"))

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
