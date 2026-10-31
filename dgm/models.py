"""Model building blocks shared by Phase 2 (composition task nets) and Phase 3 (translation).

  CompositionNet   profile (log1p CP10k on the panel) -> cell-type fractions (softmax)
  Ensemble         mean of several nets' softmax outputs
  canonical_hash   a hash of tensors/arrays that is stable across torch.save/load
  save_frozen      write weights + their canonical hash and provenance to a models.json
  load_frozen      read them back, verify everything, return a frozen (eval, no-grad) module

Spec: notes/phase2_spec.md (v3.1, approved by the spec audit 2026-10-06).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCALINGS = ("center", "floor", "full")


def input_scale(x_train: np.ndarray, scaling: str) -> tuple[np.ndarray, np.ndarray]:
    """(mean, inverse scale) for an input scaling, from training rows only.
      center  subtract the mean; inverse scale 1
      floor   subtract the mean, divide by max(SD, median SD)
      full    subtract the mean, divide by SD (SD 0 -> 1)
    Full scaling lets near-constant genes dominate (exploitable; spec audit finding 2)."""
    assert scaling in SCALINGS, scaling
    mean = x_train.mean(0)
    sd = x_train.std(0)
    if scaling == "center":
        inv = np.ones_like(sd)
    elif scaling == "floor":
        inv = 1.0 / np.maximum(sd, np.median(sd))
    else:
        inv = 1.0 / np.where(sd > 0, sd, 1.0)
    return mean.astype(np.float32), inv.astype(np.float32)


class CompositionNet(nn.Module):
    """input scaling -> [Linear(g, h), LayerNorm, LeakyReLU(0.2), Dropout(p)] -> Linear(-> k) -> softmax.
    hidden = 0 drops the hidden block (linear-softmax). The output layer starts with zero weights
    and bias log(mean training fractions), so at initialization the prediction is the training
    mean composition for every input (spec [audit2 A])."""

    def __init__(self, mean: np.ndarray, inv_scale: np.ndarray, prior: np.ndarray, hidden: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        g, k = len(mean), len(prior)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("inv_scale", torch.as_tensor(inv_scale, dtype=torch.float32))
        self.hidden = hidden
        if hidden > 0:
            self.body = nn.Sequential(nn.Linear(g, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2), nn.Dropout(dropout))
            self.head = nn.Linear(hidden, k)
        else:
            self.body = nn.Identity()
            self.head = nn.Linear(g, k)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.copy_(torch.log(torch.as_tensor(prior, dtype=torch.float32)))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body((x - self.mean) * self.inv_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x), dim=-1)


class Ensemble(nn.Module):
    """Mean of the members' softmax outputs (still on the simplex)."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.members]).mean(0)


class Frozen(nn.Module):
    """Wraps a model so it stays in eval mode with no trainable parameters, even when a parent
    module calls .train(). Gradients still flow to its *inputs* (Phase 3's task loss)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):     # never leaves eval mode
        self.training = False
        self.model.eval()
        return self

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------- hashing and freezing
def canonical_hash(arrays: dict) -> str:
    """SHA-256 over sorted names; per entry its name, dtype, shape and raw little-endian bytes.
    Stable across torch.save/load (file hashes aren't: spec audit finding 6)."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        v = arrays[name]
        a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        if a.dtype.kind == "U":
            a = np.char.encode(a, "utf-8")
        a = np.ascontiguousarray(a)
        if a.dtype.byteorder == ">":
            a = a.byteswap().view(a.dtype.newbyteorder("<"))
        for part in (name, str(a.dtype), str(a.shape)):
            h.update(part.encode() + b"\0")
        h.update(a.tobytes())
    return h.hexdigest()


def list_hash(items) -> str:
    return hashlib.sha256("\n".join(map(str, items)).encode()).hexdigest()


def pairs_hash(pairs) -> str:
    """Content hash of the pairs file the models were trained on."""
    return canonical_hash({k: getattr(pairs, k) for k in ("x_sc", "x_bulk", "composition", "genes", "l1_classes",
                                                           "sample_ids", "donor_ids", "age", "age_group", "split")})


def build_net(cfg: dict, mean, inv_scale, prior) -> CompositionNet:
    return CompositionNet(mean, inv_scale, prior, hidden=cfg["hidden"], dropout=cfg.get("dropout", 0.0))


def save_frozen(name: str, nets: list, cfg: dict, mean, inv_scale, prior, out_dir: Path, registry: Path,
                provenance: dict) -> dict:
    """Save each member's state dict (with the config needed to rebuild it) and record the
    canonical hash of every member plus the provenance in the registry (models.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = {"config": cfg, "members": [], "n_members": len(nets), **provenance}
    for i, net in enumerate(nets):
        path = out_dir / f"{name}_seed{i}.pt"
        torch.save({"state_dict": net.state_dict(), "cfg": cfg, "mean": mean, "inv_scale": inv_scale, "prior": prior}, path)
        entry["members"].append({"file": path.name, "canonical_hash": canonical_hash(net.state_dict())})
    reg = json.loads(registry.read_text()) if registry.exists() else {}
    reg[name] = entry
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
    return entry


def load_frozen(name: str, model_dir: Path, registry: Path, genes, classes, pairs_digest: str) -> Frozen:
    """Rebuild a frozen ensemble and verify it: every member's canonical hash, and that the gene
    order, class list and pairs file match what it was trained on. Raises on any mismatch."""
    reg = json.loads(Path(registry).read_text())
    if name not in reg:
        raise KeyError(f"{name} not in {registry}")
    entry = reg[name]
    if entry["genes_hash"] != list_hash(genes):
        raise ValueError(f"{name}: gene order differs from training")
    if entry["classes"] != list(classes):
        raise ValueError(f"{name}: class list differs from training ({entry['classes']})")
    if entry["pairs_hash"] != pairs_digest:
        raise ValueError(f"{name}: pairs file differs from the one it was trained on")
    if len(entry["members"]) != entry.get("n_members", -1):
        raise ValueError(f"{name}: {len(entry['members'])} members recorded, expected {entry.get('n_members')}")
    nets = []
    for m in entry["members"]:
        blob = torch.load(Path(model_dir) / m["file"], map_location="cpu", weights_only=False)
        if json.loads(json.dumps(blob["cfg"])) != entry["config"]:
            raise ValueError(f"{name}: {m['file']} was built with a different config than the registry records")
        net = build_net(blob["cfg"], blob["mean"], blob["inv_scale"], blob["prior"])
        net.load_state_dict(blob["state_dict"])
        if canonical_hash(net.state_dict()) != m["canonical_hash"]:
            raise ValueError(f"{name}: {m['file']} doesn't match its recorded canonical hash")
        nets.append(net)
    return Frozen(Ensemble(nets))
