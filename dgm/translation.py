"""Phase 3: unpaired translation between single-cell pseudobulk and whole-blood bulk (CycleGAN).

Spec: notes/phase3_spec.md; hypotheses, selection rule and budget: notes/phase3_prereg.md.

  Generator        G(x) = ReLU(a * x + f(x)), a per-gene learnable scale, f a residual MLP
                   (g -> h -> h -> g, LayerNorm, LeakyReLU(0.2)) whose last layer starts at zero
                   weights with bias b, so G starts exactly at the affine map x -> a * x + b
  init_affine      (a, b) of the most accurate unpaired baseline in each direction, fit on the
                   training halves: mean shift for S->B (a = 1), ComBat for B->S
  Discriminator    spectral-norm MLP g -> 256 -> 64 -> 1, LeakyReLU(0.2)
  train            LSGAN adversarial + L1 cycle (+ optional identity) losses, and for the weakly
                   paired model (M2) the composition task loss; instance noise on the
                   discriminators' inputs; validation every `eval_every` steps; returns the log
                   and the checkpoint chosen by the neighbor-averaged validation margin
  SoundLife        the Sound Life training halves, validation evaluators, frontier, frozen
                   composition networks and marker probes, and the H1 margin of a checkpoint
                   (full scoring adds the marker-probe and composition-network readouts)
  run / run_grid   one (config, seed) run, cached on disk; a grid of runs in worker processes

Training is deterministic on CPU at a fixed thread count (dgm.composition.deterministic).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

DIRECTIONS = ("S->B", "B->S")
INIT = {"S->B": "mean shift", "B->S": "ComBat"}     # each generator starts at its direction's most accurate unpaired baseline

DEFAULTS = {
    "steps": 20000, "batch": 64, "lr_g": 2e-4, "lr_d": 2e-4, "betas": (0.5, 0.999), "hidden": 512,
    "lambda_cyc": 10.0, "identity": False, "lambda_task": 0.0, "warmup": 0.2, "ramp": 0.2,
    "noise_sd": 0.1, "eval_every": 500, "log_every": 100, "shuffle_d": False,
    # Phase 4 ablations (notes/phase4_spec.md); the defaults reproduce Phase 3 exactly
    "lambda_adv": 1.0, "affine_only": False, "ema": 0.0, "keep_states": (), "task_terms": "both", "task_nets": "locked",
    "subset": None, "lambda_id": None,
    # Phase 5 (notes/phase5_spec.md); the defaults reproduce Phase 3 exactly
    "identity_target": "input", "identity_dirs": "both", "lambda_dist": 0.0, "bottleneck": None,
}


def config(**kw) -> dict:
    """A full configuration: DEFAULTS overridden by kw (unknown keys are an error)."""
    unknown = set(kw) - set(DEFAULTS)
    assert not unknown, f"unknown config keys: {unknown}"
    return {**DEFAULTS, **kw}


def config_name(cfg: dict) -> str:
    """Short name of the settings that differ from DEFAULTS."""
    diff = {k: v for k, v in cfg.items() if DEFAULTS.get(k) != v}
    return ",".join(f"{k}={v}" for k, v in sorted(diff.items())) or "default"


# ---------------------------------------------------------------- networks
class Generator(nn.Module):
    """G(x) = ReLU(a * x + f(x)); f's last layer starts at zero weights and bias b. With a
    bottleneck k, f's second hidden layer has k units (g -> hidden -> k -> g; Phase 5)."""

    def __init__(self, scale: np.ndarray, bias: np.ndarray, hidden: int = 512, bottleneck: int | None = None):
        super().__init__()
        g = len(scale)
        h2 = bottleneck or hidden
        self.a = nn.Parameter(torch.as_tensor(np.asarray(scale), dtype=torch.float32).clone())
        self.f = nn.Sequential(nn.Linear(g, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2),
                               nn.Linear(hidden, h2), nn.LayerNorm(h2), nn.LeakyReLU(0.2),
                               nn.Linear(h2, g))
        with torch.no_grad():
            self.f[-1].weight.zero_()
            self.f[-1].bias.copy_(torch.as_tensor(np.asarray(bias), dtype=torch.float32))

    def forward(self, x):
        return torch.relu(self.a * x + self.f(x))


class AffineGenerator(nn.Module):
    """G(x) = ReLU(a * x + b): the per-gene affine map alone, with no residual branch (ablation A2)."""

    def __init__(self, scale: np.ndarray, bias: np.ndarray, hidden: int = 512):
        super().__init__()
        self.a = nn.Parameter(torch.as_tensor(np.asarray(scale), dtype=torch.float32).clone())
        self.b = nn.Parameter(torch.as_tensor(np.asarray(bias), dtype=torch.float32).clone())

    def forward(self, x):
        return torch.relu(self.a * x + self.b)


def make_generator(scale, bias, hidden: int = 512, affine_only: bool = False, bottleneck: int | None = None) -> nn.Module:
    if affine_only:
        return AffineGenerator(scale, bias, hidden)
    return Generator(scale, bias, hidden) if bottleneck is None else Generator(scale, bias, hidden, bottleneck)


def start_map(init_ab, x: torch.Tensor) -> torch.Tensor:
    """The generator's starting map s(x) = ReLU(a0 * x + b0)."""
    a0, b0 = (torch.as_tensor(np.asarray(v), dtype=torch.float32) for v in init_ab)
    return torch.relu(a0 * x + b0)


def pair_distances(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Euclidean distances between all pairs of rows (upper triangle), differentiable at duplicates."""
    sq = (x * x).sum(1)
    d2 = (sq[:, None] + sq[None, :] - 2 * x @ x.T).clamp_min(0)
    i, j = torch.triu_indices(len(x), len(x), 1)
    return torch.sqrt(d2[i, j] + eps)


def distance_loss(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """DistanceGAN-style loss: L1 between the pairwise distances of `out` and of `ref`, each
    standardized by its own batch mean and SD, so only relative geometry counts (Phase 5)."""
    z = lambda d: (d - d.mean()) / (d.std() + 1e-8)
    return (z(pair_distances(out)) - z(pair_distances(ref))).abs().mean()


class Discriminator(nn.Module):
    """Spectral-norm MLP g -> 256 -> 64 -> 1 with LeakyReLU(0.2); outputs one score per profile."""

    def __init__(self, g: int, dims=(256, 64)):
        super().__init__()
        layers, d_in = [], g
        for d in dims:
            layers += [spectral_norm(nn.Linear(d_in, d)), nn.LeakyReLU(0.2)]
            d_in = d
        layers.append(spectral_norm(nn.Linear(d_in, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def init_affine(kind: str, src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(a, b) such that a * x + b equals the baseline `kind` fit on (src, tgt) before clipping:
    mean shift (a = 1, b = mean(tgt) - mean(src)) or ComBat (reference-batch EB; see dgm.baselines)."""
    from dgm.baselines import ComBat
    if kind == "mean shift":
        return np.ones(src.shape[1]), tgt.mean(0) - src.mean(0)
    if kind == "ComBat":
        cb = ComBat().fit(src, tgt)
        a = 1.0 / cb.delta_
        return a, cb.alpha_ - cb.alpha_ * a - cb.sigma_ * cb.gamma_ * a
    if kind == "zero":                                  # a = 1, b = 0: the synthetic test's start, away from the answer
        return np.ones(src.shape[1]), np.zeros(src.shape[1])
    raise ValueError(kind)


def canonical_state(sb: nn.Module, bs: nn.Module) -> dict:
    return {**{f"sb.{k}": v.detach().clone() for k, v in sb.state_dict().items()},
            **{f"bs.{k}": v.detach().clone() for k, v in bs.state_dict().items()}}


# ---------------------------------------------------------------- training
def train(xs: np.ndarray, xb: np.ndarray, init: dict, cfg: dict, seed: int, evaluate=None, task=None) -> dict:
    """Train one CycleGAN on unpaired single-cell profiles xs and bulk profiles xb.

    init      {"S->B": (a, b), "B->S": (a, b)}: the generators' starting affine maps
    evaluate  optional callable(step, G_sb, G_bs) -> dict of validation metrics with a "margin"
              key (the H1 margin, mean over directions); called every cfg["eval_every"] steps
              and at step 0
    task      optional dict for the weakly paired model: {"C_B", "C_S"} frozen composition
              networks (probability outputs) and "comp_s" (the measured composition of each
              row of xs); the task loss is
                L1(C_B(G_sb(s)), comp_s) + L1(C_S(G_bs(b)), C_B(b))   (summed over classes, mean over batch)
              weighted by lambda_task * ramp(step): 0 until warmup * steps, then rising linearly to
              1 over ramp * steps

    Selection: each evaluated checkpoint after step 0 and before the last evaluation is scored by
    its margin averaged with its two neighbors (one evaluation before and after); the checkpoint
    with the largest averaged margin is kept. With a task loss, only checkpoints at or after
    (warmup + ramp) * steps are eligible, when the task loss is at full weight: before the warm-up
    ends an M2 run is identical to the M1 run with the same setting and seed, so an earlier
    checkpoint would be an M1 model.

    shuffle_d (the negative control): the discriminators' real/translated targets are drawn at
    random each step, so the adversarial term carries no information.

    Phase 4 options (defaults reproduce Phase 3 bit for bit): lambda_adv multiplies the generators'
    adversarial loss (the discriminators still train and every random draw is unchanged);
    affine_only uses AffineGenerator; lambda_id sets the identity loss's weight
    (None: 0.5 * lambda_cyc, as in Phase 3); task_terms ("both", "sb", "bs") keeps one or both task
    terms; ema > 0 keeps an exponential moving average of the generator weights, scored at every
    evaluation under evals[i]["ema"] (selection and the returned state stay on the raw weights;
    the final average is returned as "ema_state"); keep_states lists evaluation steps whose raw
    generator weights are returned under "kept" (and the averaged weights, if kept, under
    "kept_ema"). Returns {"log", "evals", "best_step",
    "best_margin", "state", "disc_state"} (state: the chosen generators' weights; disc_state:
    the discriminators' final weights)."""
    from dgm.composition import l1_task_loss
    torch.manual_seed(seed)
    g = xs.shape[1]
    G_sb = make_generator(*init["S->B"], hidden=cfg["hidden"], affine_only=cfg["affine_only"], bottleneck=cfg["bottleneck"])
    G_bs = make_generator(*init["B->S"], hidden=cfg["hidden"], affine_only=cfg["affine_only"], bottleneck=cfg["bottleneck"])
    D_b, D_s = Discriminator(g), Discriminator(g)
    opt_g = torch.optim.Adam([*G_sb.parameters(), *G_bs.parameters()], lr=cfg["lr_g"], betas=cfg["betas"])
    opt_d = torch.optim.Adam([*D_b.parameters(), *D_s.parameters()], lr=cfg["lr_d"], betas=cfg["betas"])
    XS, XB = torch.as_tensor(xs, dtype=torch.float32), torch.as_tensor(xb, dtype=torch.float32)
    CS = torch.as_tensor(task["comp_s"], dtype=torch.float32) if task else None
    rng = torch.Generator().manual_seed(seed)
    mse = lambda p, t: ((p - t) ** 2).mean()
    steps, B = cfg["steps"], cfg["batch"]
    ones, zeros = torch.ones(B), torch.zeros(B)

    min_step = (cfg["warmup"] + cfg["ramp"]) * steps if task else 1

    def task_weight(step):
        start, width = cfg["warmup"] * steps, max(cfg["ramp"] * steps, 1)
        return cfg["lambda_task"] * float(np.clip((step - start) / width, 0, 1))

    log, evals, held = [], [], []          # held: (step, margin, state) of the last two evaluations
    best = {"step": None, "margin": -np.inf, "state": None}
    kept, kept_ema = {}, {}
    ema = None
    if cfg["ema"] > 0:
        import copy
        ema = (copy.deepcopy(G_sb), copy.deepcopy(G_bs))
        for e in ema:
            e.requires_grad_(False)

    def ema_update():
        with torch.no_grad():
            for e, g_ in zip(ema, (G_sb, G_bs)):
                for pe, p in zip(e.parameters(), g_.parameters()):
                    pe.mul_(cfg["ema"]).add_(p, alpha=1 - cfg["ema"])

    def do_eval(step):
        G_sb.eval(); G_bs.eval()
        with torch.no_grad():
            m = evaluate(step, G_sb, G_bs)
        G_sb.train(); G_bs.train()
        evals.append({"step": step, **m})
        if ema is not None:
            ema[0].eval(); ema[1].eval()
            with torch.no_grad():
                evals[-1]["ema"] = evaluate(step, *ema)
        if step in cfg["keep_states"]:
            kept[step] = canonical_state(G_sb, G_bs)
            if ema is not None:
                kept_ema[step] = canonical_state(*ema)
        held.append((step, m["margin"], canonical_state(G_sb, G_bs)))
        if len(held) == 3:                 # the middle one now has both neighbors
            (s0, m0, _), (s1, m1, st1), (s2, m2, _) = held
            avg = float(np.mean([m0, m1, m2]))
            evals[-2]["margin_avg"] = avg
            if s1 >= min_step and avg > best["margin"]:
                best.update(step=s1, margin=avg, state=st1)
            held.pop(0)

    if evaluate is not None:
        do_eval(0)
    for step in range(1, steps + 1):
        idx_s = torch.randint(len(XS), (B,), generator=rng)
        s = XS[idx_s]
        idx_b = torch.randint(len(XB), (B,), generator=rng)
        b = XB[idx_b]
        sd = cfg["noise_sd"] * max(0.0, 1 - step / (steps / 2))      # instance noise, annealed to 0 by mid-training
        noisy = (lambda x: x + sd * torch.randn(x.shape, generator=rng)) if sd > 0 else (lambda x: x)
        # generators
        fb, fs = G_sb(s), G_bs(b)
        l_adv = mse(D_b(noisy(fb)), ones) + mse(D_s(noisy(fs)), ones)
        l_cyc = (G_bs(fb) - s).abs().mean() + (G_sb(fs) - b).abs().mean()
        loss = cfg["lambda_adv"] * l_adv + cfg["lambda_cyc"] * l_cyc
        l_id = l_task = torch.zeros(())
        if cfg["identity"]:
            if cfg["identity_target"] == "input":
                t_b, t_s = b, s
            else:                                  # start-anchored: the start's output on the other domain's input
                assert cfg["identity_target"] == "start", cfg["identity_target"]
                with torch.no_grad():
                    t_b, t_s = start_map(init["S->B"], b), start_map(init["B->S"], s)
            if cfg["identity_dirs"] == "both":
                l_id = (G_sb(b) - t_b).abs().mean() + (G_bs(s) - t_s).abs().mean()
            else:                                  # identity loss for the bulk -> single-cell generator only
                assert cfg["identity_dirs"] == "bs", cfg["identity_dirs"]
                l_id = (G_bs(s) - t_s).abs().mean()
            loss = loss + (0.5 * cfg["lambda_cyc"] if cfg["lambda_id"] is None else cfg["lambda_id"]) * l_id
        l_dist = torch.zeros(())
        if cfg["lambda_dist"] > 0:
            with torch.no_grad():
                ref_b, ref_s = start_map(init["S->B"], s), start_map(init["B->S"], b)
            l_dist = distance_loss(fb, ref_b) + distance_loss(fs, ref_s)
            loss = loss + cfg["lambda_dist"] * l_dist
        w_task = task_weight(step) if task else 0.0
        if w_task > 0:
            # Phase 5 options: per-row readouts ("C_B_sb", "C_B_pseudo": callables of (x, rows)) and a
            # mask of the single-cell rows the single-cell -> bulk term may use ("mask_s")
            with torch.no_grad():
                pseudo = task["C_B_pseudo"](b, idx_b) if "C_B_pseudo" in task else task["C_B"](b)
            def term_sb():
                read = task["C_B_sb"](fb, idx_s) if "C_B_sb" in task else task["C_B"](fb)
                if "mask_s" not in task:
                    return l1_task_loss(read, CS[idx_s])
                m = task["mask_s"][idx_s]
                return l1_task_loss(read[m], CS[idx_s][m]) if bool(m.any()) else torch.zeros(())
            if cfg["task_terms"] == "both":
                l_task = term_sb() + l1_task_loss(task["C_S"](fs), pseudo)
            elif cfg["task_terms"] == "sb":
                l_task = term_sb()
            else:
                assert cfg["task_terms"] == "bs", cfg["task_terms"]
                l_task = l1_task_loss(task["C_S"](fs), pseudo)
            loss = loss + w_task * l_task
        opt_g.zero_grad(); loss.backward(); opt_g.step()
        if ema is not None:
            ema_update()
        # discriminators
        fb, fs = fb.detach(), fs.detach()
        pr_b, pf_b, pr_s, pf_s = D_b(noisy(b)), D_b(noisy(fb)), D_s(noisy(s)), D_s(noisy(fs))
        t_real, t_fake = ones, zeros
        if cfg["shuffle_d"]:
            lab = torch.randint(0, 2, (2 * B,), generator=rng).float()
            t_real, t_fake = lab[:B], lab[B:]
        l_d = mse(pr_b, t_real) + mse(pf_b, t_fake) + mse(pr_s, t_real) + mse(pf_s, t_fake)
        opt_d.zero_grad(); l_d.backward(); opt_d.step()
        if step % cfg["log_every"] == 0:
            real_acc, fake_acc = (lambda p: float((p > 0.5).float().mean())), (lambda p: float((p < 0.5).float().mean()))
            log.append({"step": step, "adv": l_adv.item(), "cyc": l_cyc.item(), "identity": l_id.item(), "dist": l_dist.item(),
                        "task": l_task.item(), "task_weight": w_task, "d": l_d.item(),
                        "d_acc_bulk": (real_acc(pr_b) + fake_acc(pf_b)) / 2, "d_acc_sc": (real_acc(pr_s) + fake_acc(pf_s)) / 2,
                        "d_acc_bulk_real": real_acc(pr_b), "d_acc_bulk_fake": fake_acc(pf_b),
                        "d_acc_sc_real": real_acc(pr_s), "d_acc_sc_fake": fake_acc(pf_s),
                        "diversity_bulk": float(fb.std(0).mean() / b.std(0).mean()),
                        "diversity_sc": float(fs.std(0).mean() / s.std(0).mean())})
        if evaluate is not None and step % cfg["eval_every"] == 0:
            do_eval(step)
    if evaluate is None:
        best.update(step=steps, margin=float("nan"), state=canonical_state(G_sb, G_bs))
    disc = {**{f"b.{k}": v.detach().clone() for k, v in D_b.state_dict().items()},
            **{f"s.{k}": v.detach().clone() for k, v in D_s.state_dict().items()}}
    return {"log": log, "evals": evals, "best_step": best["step"], "best_margin": best["margin"], "state": best["state"],
            "disc_state": disc, "kept": kept, "kept_ema": kept_ema, "ema_state": canonical_state(*ema) if ema is not None else None,
            "rng_end": float(torch.rand(1, generator=rng))}   # one draw after training: equal iff the run made the same draws


def load_generators(state: dict, g: int, hidden: int = 512, affine_only: bool = False, bottleneck: int | None = None) -> tuple[nn.Module, nn.Module]:
    """Rebuild (G_sb, G_bs) from a train() state, in eval mode."""
    sb, bs = (make_generator(np.ones(g), np.zeros(g), hidden, affine_only, bottleneck) for _ in range(2))
    sb.load_state_dict({k[3:]: v for k, v in state.items() if k.startswith("sb.")})
    bs.load_state_dict({k[3:]: v for k, v in state.items() if k.startswith("bs.")})
    return sb.eval(), bs.eval()


# ---------------------------------------------------------------- Sound Life setting
def assert_halves_only(pairs, idx_s: np.ndarray, idx_b: np.ndarray, halving: int) -> None:
    """Training may read single-cell profiles of half A and bulk profiles of half B only."""
    from dgm.composition import assert_train_only
    from dgm.data import train_halves
    idx_s, idx_b = np.asarray(idx_s), np.asarray(idx_b)
    assert_train_only(pairs, idx_s)
    assert_train_only(pairs, idx_b)
    assert not set(pairs.donor_ids[idx_s]) & set(pairs.donor_ids[idx_b]), "a donor contributes to both domains"
    half_a, half_b = train_halves(pairs, halving)
    assert set(idx_s) <= set(half_a), "single-cell rows outside half A"
    assert set(idx_b) <= set(half_b), "bulk rows outside half B"


class SoundLife:
    """Everything a Sound Life run needs, built once per process: the training halves, the
    starting affine maps, the validation evaluators and frontier (Phase 1), and for the weakly
    paired model the frozen composition networks (Phase 2).

    Phase 4: subset ("donors" or "draws", with subset_seed) trains on reduced halves
    (dgm.ablation.subset_rows), with the starts, the baselines and the frontier refit on them; the
    margin against the full halves' frontier is also reported ("margin_full"). task_nets="halfA"
    trains the weakly paired model against composition networks retrained on half A only
    (ablation B3); the readouts used for scoring stay the locked networks."""

    def __init__(self, halving: int = 0, with_task: bool = False, subset: str | None = None, subset_seed: int = 0,
                 task_nets: str = "locked"):
        from dgm import paths
        from dgm.composition import load_marker
        from dgm.data import load_pairs, train_halves
        from dgm.metrics import Evaluator
        from dgm.models import load_frozen, pairs_hash
        self.halving = halving
        P = self.pairs = load_pairs()
        self.genes, self.classes = P.genes.tolist(), P.l1_classes.tolist()
        self.half_a, self.half_b = train_halves(P, halving)
        self.subset = subset
        if subset is not None:
            from dgm.ablation import subset_rows
            self.half_a, self.half_b = subset_rows(P, self.half_a, self.half_b, subset, subset_seed)
        assert_halves_only(P, self.half_a, self.half_b, halving)
        self.xs, self.xb = P.x_sc[self.half_a], P.x_bulk[self.half_b]
        self.comp_s = P.composition[self.half_a]
        self.init = {"S->B": init_affine(INIT["S->B"], self.xs, self.xb), "B->S": init_affine(INIT["B->S"], self.xb, self.xs)}
        tr, self.va = P.idx("train"), P.idx("val")
        self.src = {"S->B": P.x_sc[self.va], "B->S": P.x_bulk[self.va]}
        self.tgt = {"S->B": P.x_bulk[self.va], "B->S": P.x_sc[self.va]}
        self.ev = {d: Evaluator((P.x_bulk if d == "S->B" else P.x_sc)[tr], P.composition[tr], P.age_group[tr],
                                P.donor_ids[tr], self.genes, d) for d in DIRECTIONS}
        fr = pd.read_csv(paths.results_dir(1) / "frontier_val_halves.csv")
        self.frontier = {d: fr[(fr.halving == halving) & (fr.direction == d)].reset_index(drop=True) for d in DIRECTIONS}
        assert all(len(self.frontier[d]) for d in DIRECTIONS), "frontier missing for this halving (run Phase 1)"
        self.frontier_full = None
        if subset is not None:                # the subset's own frontier is the comparator; the full one is reported too
            from dgm.ablation import frontier_points
            self.frontier_full = self.frontier
            self.frontier = frontier_points(P, self.half_a, self.half_b, self.ev, self.va, self.classes)
        digest = pairs_hash(P)
        reg2, mod2 = paths.results_dir(2) / "models.json", paths.REPO / "data" / "models" / "phase2"
        self.marker = {"S->B": load_marker("marker_bulk", mod2, reg2, self.genes, self.classes, digest),
                       "B->S": load_marker("marker_sc", mod2, reg2, self.genes, self.classes, digest)}
        self.nets = {"S->B": load_frozen("C_bulk", mod2, reg2, self.genes, self.classes, digest),
                     "B->S": load_frozen("C_sc", mod2, reg2, self.genes, self.classes, digest)}
        self.task = None
        if with_task:
            if task_nets == "locked":
                cb, cs = self.nets["S->B"], self.nets["B->S"]
            elif task_nets == "halfA":
                assert subset is None, (task_nets, subset)
                reg4, mod4 = paths.results_dir(4) / "models.json", paths.REPO / "data" / "models" / "phase4"
                cb = load_frozen(f"C_bulk_halfA{halving}", mod4, reg4, self.genes, self.classes, digest)
                cs = load_frozen(f"C_sc_halfA{halving}", mod4, reg4, self.genes, self.classes, digest)
            else:
                self.task = self._phase5_task(task_nets, subset_seed, halving, digest)
                return
            self.task = {"C_B": cb, "C_S": cs, "comp_s": self.comp_s}

    def _phase5_task(self, task_nets: str, seed: int, halving: int, digest: str) -> dict:
        """Track B of Phase 5 (notes/phase5_spec.md). C_S is Phase 4's half-A network throughout.
          "crossfit"   C_B cross-fitted over the 68 training donors; every row is read by the
                       network that left its donor out, in both task terms
          "k4/k8/k16"  C_B on the first k donors of this seed's 16-donor half-A pool; the
                       single-cell -> bulk term uses only the other 17 half-A donors' rows
          "k33"        all of half A: the single-cell -> bulk term reads each half-A row with C_B
                       cross-fitted within half A; the pseudo-labels on half B come from Phase 4's
                       half-A C_B"""
        import json
        from dgm import paths
        from dgm.labels import RowEnsemble
        from dgm.models import load_frozen
        assert halving == 0 and self.subset is None, (halving, self.subset)
        P = self.pairs
        reg4, mod4 = paths.results_dir(4) / "models.json", paths.REPO / "data" / "models" / "phase4"
        reg5, mod5 = paths.results_dir(5) / "models.json", paths.REPO / "data" / "models" / "phase5"
        load = lambda name, reg, mod: load_frozen(name, mod, reg, self.genes, self.classes, digest)
        cs = load("C_sc_halfA0", reg4, mod4)
        entries = json.loads(reg5.read_text())
        task = {"C_S": cs, "comp_s": self.comp_s}
        if task_nets in ("crossfit", "k33"):
            name = "C_bulk_cf68" if task_nets == "crossfit" else "C_bulk_cfA"
            folds = entries[f"{name}_folds"]
            nets = [load(f"{name}_f{f}", reg5, mod5) for f in range(5)]
            task["C_B_sb"] = RowEnsemble(nets, [folds[str(d)] for d in P.donor_ids[self.half_a]])
            if task_nets == "crossfit":
                task["C_B_pseudo"] = RowEnsemble(nets, [folds[str(d)] for d in P.donor_ids[self.half_b]])
            else:
                task["C_B"] = load("C_bulk_halfA0", reg4, mod4)
            return task
        k = int(task_nets[1:])
        assert k in (4, 8, 16), task_nets
        pool = entries[f"pool_s{seed}"]
        task["C_B"] = load(f"C_bulk_k{k}_s{seed}", reg5, mod5)
        task["mask_s"] = torch.as_tensor(~np.isin(P.donor_ids[self.half_a], pool))   # the 17 donors outside the pool
        return task

    def score(self, G_sb, G_bs, full: bool = False) -> dict:
        """Validation metrics of a checkpoint per direction, the H1 margin per direction and its mean,
        the marker-probe composition R² and the mean distance of the translations from the start
        ("drift"). full=True adds every Evaluator metric and the composition networks' readout."""
        from dgm.composition import comp_r2
        from dgm.frontier import frontier_margin
        P, out, margins = self.pairs, {}, []
        with torch.no_grad():
            yh = {"S->B": G_sb(torch.as_tensor(self.src["S->B"])).numpy(), "B->S": G_bs(torch.as_tensor(self.src["B->S"])).numpy()}
        if not all(np.isfinite(y).all() for y in yh.values()):   # a diverged run: never selectable, recorded as failed
            return {"margin": -np.inf, "nonfinite": True}
        for d in DIRECTIONS:
            m, _ = self.ev[d].score(yh[d], self.tgt[d], P.composition[self.va], P.age_group[self.va], P.donor_ids[self.va], self.classes)
            realism = 1 - 2 * abs(m["c2st_auroc"] - 0.5)
            margin, kind = frontier_margin(self.frontier[d], m["r2_centered"], realism)
            margins.append(margin)
            keep = m if full else {k: m[k] for k in ("r2_centered", "c2st_auroc", "comp_r2", "within_top1", "diversity", "gran_gap", "nn_ratio")}
            out.update({f"{d} {k}": v for k, v in keep.items()})
            out.update({f"{d} realism": realism, f"{d} margin": margin, f"{d} margin kind": kind})
            if self.frontier_full is not None:
                out[f"{d} margin_full"] = frontier_margin(self.frontier_full[d], m["r2_centered"], realism)[0]
            out[f"{d} marker comp_r2"] = comp_r2(self.marker[d].predict(yh[d]), P.composition[self.va])
            a0, b0 = self.init[d]
            out[f"{d} drift"] = float(np.linalg.norm(yh[d] - np.maximum(self.src[d] * a0 + b0, 0), axis=1).mean())
            if full:
                out[f"{d} net comp_r2"] = comp_r2(self.nets[d](torch.as_tensor(yh[d], dtype=torch.float32)).numpy(), P.composition[self.va])
        out["margin"] = float(np.mean(margins))
        return out


# ---------------------------------------------------------------- runs and grids
_CTX: dict = {}


def _context(halving: int, with_task: bool, subset: str | None = None, subset_seed: int = 0, task_nets: str = "locked") -> SoundLife:
    per_seed = subset is not None or task_nets.startswith("k")
    key = (halving, with_task, subset, subset_seed if per_seed else 0, task_nets)
    if key not in _CTX:
        _CTX[key] = SoundLife(halving, with_task, subset, subset_seed, task_nets)
    return _CTX[key]


def code_key() -> str:
    """Hash of the code and data a run depends on, so a cache never serves stale results."""
    from dgm import paths
    from dgm.data import load_pairs
    from dgm.models import pairs_hash
    import platform
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info
    files = ["dgm/paths.py", "dgm/translation.py", "dgm/metrics.py", "dgm/frontier.py", "dgm/baselines.py", "dgm/models.py",
             "dgm/composition.py", "dgm/data.py", "dgm/ablation.py", "results/phase1/frontier_val_halves.csv", "results/phase2/models.json"]
    files.append("dgm/labels.py")
    for ph in (4, 5):
        if (paths.REPO / "results" / f"phase{ph}" / "models.json").exists():
            files.append(f"results/phase{ph}/models.json")
    h = hashlib.sha256(b"".join((paths.REPO / f).read_bytes() for f in files))
    blas = sorted({(i.get("internal_api"), i.get("version")) for i in threadpool_info()}, key=str)
    h.update(f"{pairs_hash(load_pairs())}|python {platform.python_version()}|torch {torch.__version__}|sklearn {sklearn.__version__}"
             f"|numpy {np.__version__}|scipy {scipy.__version__}|blas {blas}".encode())
    return h.hexdigest()[:16]


def run(cfg: dict, seed: int, halving: int = 0, cache_dir: Path | None = None, key: str | None = None) -> dict:
    """One run: train with validation-based selection, then score the chosen checkpoint fully.
    Cached in cache_dir under (code key, config, seed, halving)."""
    import warnings
    from scipy.linalg import LinAlgWarning
    from dgm import composition as comp
    from dgm.models import canonical_hash
    warnings.filterwarnings("ignore", category=LinAlgWarning)   # ill-conditioned ridge solves inside the evaluators (harmless)
    comp.deterministic()
    cfg = config(**cfg)                       # fill Phase 4 keys missing from older configurations
    with_task = cfg["lambda_task"] > 0
    key = key or code_key()
    name = hashlib.sha256(json.dumps([key, cfg, seed, halving], sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"run_{name}.pkl" if cache_dir else None
    if path is not None and path.exists():
        return pickle.loads(path.read_bytes())
    ctx = _context(halving, with_task, cfg["subset"], seed, cfg["task_nets"])
    t0 = time.time()
    res = train(ctx.xs, ctx.xb, ctx.init, cfg, seed, evaluate=lambda step, sb, bs: ctx.score(sb, bs), task=ctx.task)
    res.update(cfg=cfg, seed=seed, halving=halving, code_key=key, minutes=(time.time() - t0) / 60, failed=res["state"] is None,
               rows_s=ctx.half_a, rows_b=ctx.half_b)
    if res["failed"]:                         # every checkpoint diverged: kept as a failed run, never selectable
        res.update(best={"margin": -np.inf}, best_margin=-np.inf, state_hash=None)
    else:
        sb, bs = load_generators(res["state"], ctx.xs.shape[1], cfg["hidden"], cfg["affine_only"], cfg["bottleneck"])
        res.update(best=ctx.score(sb, bs, full=True), state_hash=canonical_hash(res["state"]))
    if path is not None:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(res))
        tmp.replace(path)                     # atomic: an interrupted write never leaves a half-written cache
    return res


def _run_star(args):
    return run(*args)


def run_grid(cfgs: list[dict], seeds=(0, 1, 2), halving: int = 0, workers: int = 5, cache_dir: Path | None = None) -> list[dict]:
    """Every (config, seed) in `workers` spawned processes (each pinned to the deterministic thread
    count, so results don't depend on the number of workers). Returns the runs in input order."""
    key = code_key()
    jobs = [(c, s, halving, cache_dir, key) for c, s in itertools.product(cfgs, seeds)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
        return list(ex.map(_run_star, jobs))


def summarize(runs: list[dict]) -> pd.DataFrame:
    """One row per configuration: the seeds' chosen-checkpoint metrics averaged (the prereg's
    metrics for a setting), with the per-seed margins."""
    rows = []
    for r in runs:                            # a failed run has margin -inf, so its configuration can't be chosen
        rows.append({"config": config_name(r["cfg"]), "seed": r["seed"], "best_step": r["best_step"],
                     "margin_avg": r["best_margin"], "failed": r.get("failed", False), **r["best"]})
    df = pd.DataFrame(rows)
    num = df.drop(columns=["config", "seed"]).select_dtypes("number").columns
    agg = df.groupby("config", sort=False)[list(num)].mean()
    agg["seeds"] = df.groupby("config", sort=False).size()
    agg["failed runs"] = df.groupby("config", sort=False)["failed"].sum()
    agg["margin_avg_sd"] = df.groupby("config", sort=False)["margin_avg"].std()
    return agg.sort_values("margin_avg", ascending=False)
