"""Phase 5, track B: composition networks for honest weak labels (spec: notes/phase5_spec.md).

  donor_folds     donor-grouped, age-stratified fold labels
  train_crossfit  the bulk composition network C_B (frozen Phase 2 configuration) trained once per
                  fold, each time leaving that fold's donors out
  budget_pool     a fixed, age-stratified pool of 16 half-A donors per seed, ordered so that the
                  first k (k = 4, 8, 16) are nested and balanced across age groups
  train_budget    C_B trained on the bulk pairs of the first k pool donors
  RowEnsemble     a set of networks read per row: each row by the network whose training left out
                  that row's donor

The single-cell network C_S needs no pairs and is not trained here (Phase 4's half-A C_S is used).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

POOL = 16
KS = (4, 8, 16)


def donor_folds(pairs, rows: np.ndarray, n_folds: int = 5, seed: int = 0) -> dict:
    """{donor: fold} for the donors of `rows`: within each age group, donors are shuffled and dealt
    round-robin to the folds, so folds are donor-disjoint and balanced in age."""
    rows = np.asarray(rows)
    rng = np.random.default_rng(seed)
    out, start = {}, 0
    for g in np.unique(pairs.age_group[rows]):
        ds = np.unique(pairs.donor_ids[rows][pairs.age_group[rows] == g])
        for i, d in enumerate(ds[rng.permutation(len(ds))]):
            out[str(d)] = int((start + i) % n_folds)
        start += len(ds)
    return out


def _save(pairs, rows, name, out_dir, registry, extra):
    from dgm import paths
    from dgm.composition import freeze_nets
    from dgm.models import list_hash, pairs_hash, save_frozen
    reg2 = json.loads((paths.results_dir(2) / "models.json").read_text())
    cfg, epochs = reg2["C_bulk"]["config"], reg2["C_bulk"]["epochs"]
    nets, mean, inv, prior = freeze_nets(pairs, np.asarray(rows), "bulk", cfg, epochs)
    return save_frozen(name, nets, cfg, mean, inv, prior, Path(out_dir), Path(registry),
                       {"epochs": epochs, "classes": pairs.l1_classes.tolist(), "genes_hash": list_hash(pairs.genes.tolist()),
                        "pairs_hash": pairs_hash(pairs), "rows_hash": list_hash(np.sort(rows).tolist()), "n_rows": int(len(rows)),
                        "donors": sorted({str(d) for d in pairs.donor_ids[rows]}), **extra})


def _write(registry, key, value):
    registry = Path(registry)
    reg = json.loads(registry.read_text()) if registry.exists() else {}
    reg[key] = value
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")


def train_crossfit(pairs, rows: np.ndarray, name: str, out_dir, registry, n_folds: int = 5, seed: int = 0) -> dict:
    """C_B trained n_folds times on `rows`, each leaving one fold of donors out; saved as
    {name}_f{i}, with the fold map under {name}_folds. Returns the fold map."""
    rows = np.asarray(rows)
    folds = donor_folds(pairs, rows, n_folds, seed)
    for f in range(n_folds):
        keep = rows[np.array([folds[str(d)] != f for d in pairs.donor_ids[rows]])]
        _save(pairs, keep, f"{name}_f{f}", out_dir, registry, {"left_out_fold": f, "crossfit": name})
    _write(registry, f"{name}_folds", folds)
    return folds


def budget_pool(pairs, half_a: np.ndarray, seed: int, n: int = POOL) -> list:
    """n half-A donors, half from each age group, in an order whose first k are balanced in age
    (k even): alternate the groups' shuffled lists."""
    rng = np.random.default_rng([seed, 5])
    groups = [np.unique(pairs.donor_ids[half_a][pairs.age_group[half_a] == g]) for g in np.unique(pairs.age_group[half_a])]
    lists = [list(g[rng.permutation(len(g))][: n // len(groups)]) for g in groups]
    return [str(d) for pair in zip(*lists) for d in pair]


def train_budget(pairs, half_a: np.ndarray, seed: int, out_dir, registry, ks=KS) -> list:
    """C_B on the bulk pairs of the first k pool donors, saved as C_bulk_k{k}_s{seed}; the pool is
    recorded under pool_s{seed}. Returns the pool."""
    half_a = np.asarray(half_a)
    pool = budget_pool(pairs, half_a, seed)
    for k in ks:
        rows = half_a[np.isin(pairs.donor_ids[half_a], pool[:k])]
        _save(pairs, rows, f"C_bulk_k{k}_s{seed}", out_dir, registry, {"k": k, "seed": seed})
    _write(registry, f"pool_s{seed}", pool)
    return pool


class RowEnsemble(nn.Module):
    """Networks read per row: row r is read by nets[fold_of_row[r]] (a fold-out network)."""

    def __init__(self, nets: list, fold_of_row: np.ndarray):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.register_buffer("fold", torch.as_tensor(np.asarray(fold_of_row), dtype=torch.long))

    def forward(self, x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        out = torch.stack([n(x) for n in self.nets])          # (folds, batch, classes)
        return out[self.fold[rows], torch.arange(len(x))]
