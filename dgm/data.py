"""The model-ready pairs file (written by phase 0, read by every later phase).

One row per blood draw: the single-cell pseudobulk and the bulk profile of the same draw,
log1p(CP10k) on the gene panel, plus the AIFI_L1 composition of that draw's cells, donor
and age, and the GAN's donor-grouped split. ``save_pairs`` checks the schema on write and
``load_pairs`` on read, so a phase never trains on a malformed file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dgm.paths import PAIRS_FILE

SPLITS = ("train", "val", "test")
LOG_CP10K_MAX = float(np.log1p(1e4))   # a profile can't exceed log1p(1e4) in any gene
OLDER_FROM = 55                        # age_group 1 = older (55-67); the cohort has no ages 38-54
ARRAYS = {   # name -> (dtype kind, ndim)
    "x_sc": ("f", 2), "x_bulk": ("f", 2), "composition": ("f", 2),
    "genes": ("U", 1), "l1_classes": ("U", 1), "sample_ids": ("U", 1), "donor_ids": ("U", 1),
    "age": ("f", 1), "age_group": ("i", 1), "split": ("U", 1),
}


@dataclass(frozen=True)
class Pairs:
    x_sc: np.ndarray          # (n, g) pseudobulk, log1p(CP10k)
    x_bulk: np.ndarray        # (n, g) bulk, same scaling
    composition: np.ndarray   # (n, k) cell-type fractions, rows sum to 1
    genes: np.ndarray         # (g,) panel genes, column order of x_sc / x_bulk
    l1_classes: np.ndarray    # (k,) composition classes, column order of composition
    sample_ids: np.ndarray    # (n,) blood-draw id (sampleKitGuid)
    donor_ids: np.ndarray     # (n,)
    age: np.ndarray           # (n,) years
    age_group: np.ndarray     # (n,) 0 young (25-37), 1 older (55-67)
    split: np.ndarray         # (n,) train / val / test, donor-grouped

    def __len__(self):
        return len(self.sample_ids)

    def idx(self, split: str) -> np.ndarray:
        """Row indices of one split."""
        assert split in SPLITS, split
        return np.flatnonzero(self.split == split)


def validate(arrays: dict) -> None:
    """Schema and consistency checks shared by save and load."""
    assert set(arrays) == set(ARRAYS), f"fields differ: {set(arrays) ^ set(ARRAYS)}"
    for name, (kind, ndim) in ARRAYS.items():
        a = arrays[name]
        assert a.dtype.kind == kind and a.ndim == ndim, f"{name}: dtype {a.dtype}, ndim {a.ndim}"
    for name in ("x_sc", "x_bulk", "composition"):
        assert arrays[name].dtype == np.float32, f"{name}: {arrays[name].dtype}, expected float32"
    n, g = arrays["x_sc"].shape
    k = len(arrays["l1_classes"])
    assert arrays["x_bulk"].shape == (n, g) and len(arrays["genes"]) == g, "shape: x_bulk or genes don't match x_sc"
    assert arrays["composition"].shape == (n, k), "shape: composition doesn't match samples x l1_classes"
    assert all(len(arrays[f]) == n for f in ("sample_ids", "donor_ids", "age", "age_group", "split")), "shape: a per-sample field has the wrong length"
    assert len(set(arrays["sample_ids"])) == n, "duplicate sample_ids"
    assert len(set(arrays["genes"])) == g, "duplicate genes"
    assert np.isfinite(arrays["x_sc"]).all() and np.isfinite(arrays["x_bulk"]).all(), "non-finite expression"
    for name in ("x_sc", "x_bulk"):
        x = arrays[name]
        assert (x >= 0).all() and x.max() <= LOG_CP10K_MAX + 1e-3, f"{name}: outside the log1p(CP10k) range"
        # each row is the panel's share of a CP10k profile, so it can't sum past 1e4 on the count scale
        assert (np.expm1(x.astype(np.float64)).sum(1) <= 1e4 * (1 + 1e-4)).all(), f"{name}: rows exceed 1e4 CP10k"
    age, age_group = arrays["age"], arrays["age_group"]
    assert np.isfinite(age).all(), "age has non-finite values"
    assert (age_group == (age >= OLDER_FROM)).all(), "age_group disagrees with age"
    assert np.allclose(arrays["composition"].sum(1), 1, atol=1e-5) and (arrays["composition"] >= 0).all(), "composition rows aren't on the simplex"
    assert set(arrays["split"]) == set(SPLITS), "split labels aren't exactly train/val/test"
    for d in np.unique(arrays["donor_ids"]):
        assert len(np.unique(age_group[arrays["donor_ids"] == d])) == 1, f"donor {d} has two age groups"
    donors = {s: set(arrays["donor_ids"][arrays["split"] == s]) for s in SPLITS}
    assert not (donors["train"] & donors["val"] or donors["train"] & donors["test"] or donors["val"] & donors["test"]), \
        "a donor is in two splits"


def save_pairs(path: Path = PAIRS_FILE, **arrays) -> None:
    """Write the pairs file after checking the schema. String fields are stored as fixed-width
    unicode (never object arrays), so loading needs no pickle."""
    arrays = {k: (np.asarray(v).astype(str) if ARRAYS.get(k, ("",))[0] == "U" else np.asarray(v)) for k, v in arrays.items()}
    validate(arrays)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_pairs(path: Path = PAIRS_FILE) -> Pairs:
    """Read and check the pairs file. Build it with notebooks/phase0_preprocessing.ipynb."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing: run notebooks/phase0_preprocessing.ipynb")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    validate(arrays)
    return Pairs(**arrays)


HALVING_SEEDS = (0, 1)   # seed 0 is the primary halving; seed 1 checks sensitivity to the split


def train_halves(pairs: Pairs, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split the training donors into two disjoint halves for unpaired training: half A supplies
    only single-cell profiles and half B only bulk profiles, so no pair is ever seen together.
    Donors are shuffled within each age group (seeded) and each goes to whichever half has fewer
    draws so far (ties to the half with fewer donors, then to A), so the halves match in age mix
    and in number of draws. Returns the row indices of each half, sorted."""
    tr = pairs.idx("train")
    donors, first = np.unique(pairs.donor_ids[tr], return_index=True)
    group = pairs.age_group[tr][first]
    draws = {d: int((pairs.donor_ids[tr] == d).sum()) for d in donors}
    rng = np.random.default_rng(seed)
    halves, n_draws = ([], []), [0, 0]
    for g in np.unique(group):
        members = donors[group == g]
        for d in members[rng.permutation(len(members))]:
            h = 0 if (n_draws[0], len(halves[0])) <= (n_draws[1], len(halves[1])) else 1
            halves[h].append(d)
            n_draws[h] += draws[d]
    return tuple(np.sort(tr[np.isin(pairs.donor_ids[tr], h)]) for h in halves)
