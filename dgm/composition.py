"""Phase 2: composition task nets. Training, selection inside donor-grouped inner folds,
the adversarial-exploitability attack, the marker probe, and the paired donor bootstrap.

Spec: notes/phase2_spec.md (v3.1, approved by the spec audit 2026-10-06). Every rule here
follows it; section names below match the spec's.

Domains: "bulk" (C_bulk: inputs are bulk profiles, applied to G_S->B outputs in Phase 3) and
"sc" (C_sc: inputs are single-cell pseudobulk profiles, applied to G_B->S outputs).
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge as SkRidge
from sklearn.model_selection import GroupKFold

from dgm.models import CompositionNet, input_scale

SEED = 0
N_THREADS = 4            # pinned: CPU reduction order (and so the weights) depends on the thread count
DOMAINS = {              # domain -> (input field, source field for translations, Phase 1 direction)
    "bulk": ("x_bulk", "x_sc", "S->B"),
    "sc": ("x_sc", "x_bulk", "B->S"),
}
EPS_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
EPS50_FLOOR = 0.03
MARKERS = {   # fixed before fitting (spec: second evaluation probe); all present in the panel
    "T cell": ["CD2", "LCK", "IL7R", "CD5", "CD6"],
    "B cell": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1"],
    "Monocyte": ["CD14", "FCN1", "LYZ", "S100A9", "VCAN"],
    "NK cell": ["KLRF1", "NCAM1", "SH2D1B", "NKG7", "GNLY"],
    "DC": ["FCER1A", "CLEC10A", "CD1C", "LILRA4"],
    "Other": ["PPBP", "PF4", "HBB", "HBA2"],
}


_THREADPOOL = None


def deterministic(threads: int = N_THREADS):
    """Pin torch and the BLAS/OpenMP pools (numpy/scipy/sklearn) to `threads`, and use
    deterministic algorithms: reduction order, and so the exact weights, depends on the thread
    count (1 thread instead of 4 gave a different hash in the pre-run code audit)."""
    global _THREADPOOL
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    _THREADPOOL = threadpool_limits(limits=threads)   # kept alive so the limit persists


# ---------------------------------------------------------------- data guards and folds
def assert_train_only(pairs, idx) -> None:
    """Training and selection must only ever see train rows (spec: data, splits and the use of val)."""
    bad = np.asarray(pairs.split)[np.asarray(idx)] != "train"
    assert not bad.any(), f"{int(bad.sum())} non-train rows passed to a training/selection function"


def inner_folds(donors, age_group, n_folds: int = 5, seed: int = SEED):
    """Donor-grouped folds, stratified by age group: within each age group, donors (sorted) are
    shuffled with the seed and dealt round-robin. Returns [(train_pos, held_out_pos), ...] as
    positions into the given arrays."""
    donors, age_group = np.asarray(donors), np.asarray(age_group)
    rng = np.random.default_rng(seed)
    fold_of = {}
    for grp in np.unique(age_group):
        ds = np.array(sorted(set(donors[age_group == grp])))
        ds = ds[rng.permutation(len(ds))]
        for i, d in enumerate(ds):
            fold_of[d] = i % n_folds
    f = np.array([fold_of[d] for d in donors])
    return [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in range(n_folds)]


def comp_r2(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean over classes of 1 - SSE/SST (SST around these samples' mean), as dgm.metrics' comp_r2."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.mean([1 - ((pred[:, j] - true[:, j]) ** 2).sum() / ((true[:, j] - true[:, j].mean()) ** 2).sum()
                          for j in range(true.shape[1])]))


# ---------------------------------------------------------------- the grid and training
def grid() -> list[dict]:
    """The 108 configurations of the spec's model family."""
    out = []
    for scaling, hidden, loss, wd, noise in itertools.product(("center", "floor", "full"), (256, 0), ("l1", "kl"),
                                                              (1e-5, 1e-4, 1e-3), (0.0, 0.1)):
        for dropout in ((0.0, 0.2) if hidden else (0.0,)):
            out.append({"scaling": scaling, "hidden": hidden, "loss": loss, "weight_decay": wd, "dropout": dropout, "noise": noise})
    return out


def config_name(cfg: dict) -> str:
    return (f"{cfg['scaling']}|h{cfg['hidden']}|{cfg['loss']}|wd{cfg['weight_decay']:g}|p{cfg['dropout']:g}|n{cfg['noise']:g}")


def task_loss(pred_logits: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    """Training loss. L1: sum over classes, mean over the batch (the Phase 3 task-loss scale).
    KL: soft-target KL(c || softmax), sum over classes, mean over the batch."""
    if kind == "l1":
        return (torch.softmax(pred_logits, -1) - target).abs().sum(1).mean()
    logp = torch.log_softmax(pred_logits, -1)
    return (torch.special.xlogy(target, target) - target * logp).sum(1).mean()


def l1_task_loss(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The Phase 3 task loss on a frozen net's output probabilities (Ensemble/Frozen return
    probabilities, not logits): L1 summed over classes, averaged over the batch."""
    return (probs - target).abs().sum(1).mean()


def train_net(X: np.ndarray, C: np.ndarray, cfg: dict, seed: int, epochs: int, eval_sets: dict | None = None,
              batch: int = 64, lr: float = 1e-3):
    """Train one CompositionNet on (X, C) for `epochs` epochs, deterministically on CPU.
    eval_sets: name -> (X, C); each epoch's comp R² on each is recorded (history[name] list).
    Input noise (cfg["noise"] > 0): Gaussian with SD = noise x the median gene SD of X, in raw
    units, the same for every gene. Returns (net, history)."""
    X = np.asarray(X, np.float32)
    C = np.asarray(C, np.float32)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    mean, inv = input_scale(X, cfg["scaling"])
    net = CompositionNet(mean, inv, C.mean(0), hidden=cfg["hidden"], dropout=cfg["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    Xt, Ct = torch.from_numpy(X), torch.from_numpy(C)
    noise_sd = float(cfg["noise"] * np.median(X.std(0)))
    history = {k: [] for k in (eval_sets or {})}
    ev = {k: (torch.from_numpy(np.asarray(x, np.float32)), np.asarray(c)) for k, (x, c) in (eval_sets or {}).items()}
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            xb = Xt[b]
            if noise_sd > 0:
                xb = xb + noise_sd * torch.randn(xb.shape, generator=gen)
            loss = task_loss(net.logits(xb), Ct[b], cfg["loss"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ev:
            net.eval()
            with torch.no_grad():
                for k, (x, c) in ev.items():
                    history[k].append(comp_r2(net(x).numpy(), c))
    net.eval()
    return net, history


# ---------------------------------------------------------------- probes as torch modules
class LinearProbe(torch.nn.Module):
    """A fitted linear map x -> coef x + intercept (the ridge or marker probe) as a torch module,
    so the same PGD attack applies to it. Output: fractions (not softmaxed)."""

    def __init__(self, coef: np.ndarray, intercept: np.ndarray):
        super().__init__()
        self.register_buffer("coef", torch.as_tensor(np.asarray(coef), dtype=torch.float32))
        self.register_buffer("intercept", torch.as_tensor(np.asarray(intercept), dtype=torch.float32))

    def forward(self, x):
        return x @ self.coef.T + self.intercept


def probe_module(ridge) -> LinearProbe:
    return LinearProbe(ridge.coef_, ridge.intercept_)


class MarkerProbe:
    """Ridge on the fixed lineage markers (spec: second, independent evaluation probe). Alpha by
    donor-grouped 5-fold CV on the training rows (MAE)."""

    def __init__(self, genes, alphas=tuple(10.0 ** np.arange(-3, 5))):
        self.genes = list(genes)
        self.markers = [g for ms in MARKERS.values() for g in ms]
        missing = [g for g in self.markers if g not in self.genes]
        assert not missing, f"markers not in the panel: {missing}"
        self.idx = np.array([self.genes.index(g) for g in self.markers])
        self.alphas = alphas

    def fit(self, X, C, donors):
        Xm = np.asarray(X)[:, self.idx]
        errs = []
        for a in self.alphas:
            e = [np.abs(SkRidge(alpha=a).fit(Xm[tr], C[tr]).predict(Xm[te]) - C[te]).mean()
                 for tr, te in GroupKFold(n_splits=5).split(Xm, groups=donors)]
            errs.append(np.mean(e))
        self.alpha_ = float(self.alphas[int(np.argmin(errs))])
        self.ridge_ = SkRidge(alpha=self.alpha_).fit(Xm, C)
        return self

    def predict(self, X):
        return self.ridge_.predict(np.asarray(X)[:, self.idx])

    def module(self) -> LinearProbe:
        coef = np.zeros((self.ridge_.coef_.shape[0], len(self.genes)), np.float32)
        coef[:, self.idx] = self.ridge_.coef_
        return LinearProbe(coef, self.ridge_.intercept_)

    def state(self) -> dict:
        """The arrays its canonical hash is taken over."""
        return {"coef": self.ridge_.coef_, "intercept": self.ridge_.intercept_, "markers": np.array(self.markers),
                "alpha": np.array([self.alpha_])}


# ---------------------------------------------------------------- the attack
def pick_targets(donors, seed: int = SEED) -> np.ndarray:
    """For each sample, a target sample from a different donor in the same set (seeded)."""
    donors = np.asarray(donors)
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(np.flatnonzero(donors != d)) for d in donors])


def _gap(model, x, c_t):
    with torch.no_grad():
        return (model(x) - c_t).abs().sum(1)


def pgd(model, x: torch.Tensor, c_t: torch.Tensor, eps_abs: torch.Tensor, steps: int = 200, restarts: int = 3,
        seed: int = SEED, init: torch.Tensor | None = None) -> torch.Tensor:
    """Targeted L2 PGD: minimize ||model(x + delta) - c_t||_1 subject to ||delta||_2 <= eps_abs
    (per sample) and x + delta >= 0, along the normalized gradient.

    Step schedule (deviation from the spec's fixed step 2.5 eps / steps, found while testing,
    2026-10-06; recorded in notes/phase2_spec.md): the step decays linearly from
    2 x 2.5 eps / steps to 0, so the total path length is the spec's 2.5 eps, and the best iterate
    per sample is kept. Reasons, both shown by known-answer tests:
      - fixed steps overshoot when eps is much larger than needed and oscillate without
        converging, so a sample whose exact eps50 was 0.003 looked unexploitable (eps50 = inf);
      - the L1 loss has kinks, where accept-only-if-better steps stall (more steps made it worse).
    Diminishing steps with the best iterate is the standard fix for non-smooth objectives; it
    recovers exact eps50 on a single-output linear model (0.2%) and stays within the closed-form
    bound on a multi-output one. Restart 0 starts at `init` (projected; delta = 0 if None), the
    others at a random point in the ball; per sample the best restart (smallest gap) is kept.
    The eps50 search warm-starts each budget from the best delta of the previous, smaller one
    (feasible here too), so closure can't drop as the budget grows."""
    gen = torch.Generator().manual_seed(seed)
    eps = eps_abs.view(-1, 1)
    best_delta = torch.zeros_like(x)
    best_gap = torch.full((len(x),), float("inf"))

    def project(d):
        norm = d.norm(dim=1, keepdim=True).clamp_min(1e-12)
        d = d * torch.clamp(eps / norm, max=1.0)
        return torch.maximum(d, -x)                     # x + delta >= 0 (only shrinks |delta|)

    def per_sample_loss(d):
        return (model(x + d) - c_t).abs().sum(1)

    for r in range(restarts):
        if r == 0:
            d = torch.zeros_like(x) if init is None else project(init.detach().clone())
        else:
            z = torch.randn(x.shape, generator=gen)
            d = project(z / z.norm(dim=1, keepdim=True) * eps * torch.rand(len(x), 1, generator=gen))
        with torch.no_grad():
            run_best_gap = per_sample_loss(d)
        run_best = d.detach().clone()
        for t in range(steps):
            alpha = 2 * 2.5 * eps / steps * (1 - t / steps)
            d.requires_grad_(True)
            (grad,) = torch.autograd.grad(per_sample_loss(d).sum(), d)
            with torch.no_grad():
                d = project(d - alpha * grad / grad.norm(dim=1, keepdim=True).clamp_min(1e-12))
                gap = per_sample_loss(d)
                better = gap < run_best_gap
                run_best_gap = torch.where(better, gap, run_best_gap)
                run_best[better] = d[better]
        better = run_best_gap < best_gap
        best_gap = torch.where(better, run_best_gap, best_gap)
        best_delta[better] = run_best[better]
    return best_delta


def closure(model, x, delta, c_t) -> np.ndarray:
    """Gap closure g = 1 - ||C(x+delta) - c_t||_1 / ||C(x) - c_t||_1, per sample."""
    return (1 - _gap(model, x + delta, c_t) / _gap(model, x, c_t).clamp_min(1e-12)).numpy()


def eps50(model, x: torch.Tensor, c_t: torch.Tensor, scale: float, iters: int = 8, lo: float = 1e-3, hi: float = 2.0,
          steps: int = 200, restarts: int = 3, seed: int = SEED) -> np.ndarray:
    """Per sample, the smallest eps (in units of `scale`) at which PGD closes >= 50% of the gap.

    Deviation from the spec's plain bisection of log eps over [lo, hi] (found while testing,
    2026-10-06; recorded in notes/phase2_spec.md): the search scans **upward** from lo, doubling
    eps until PGD first succeeds, then bisects log eps inside that factor-of-2 bracket (`iters`
    halvings, 8 -> within ~0.3%). Starting from hi relied on the attack succeeding at the largest
    budget, but there the first steps can saturate the softmax, the gradient vanishes and PGD
    stalls at delta = 0 (a planted exploitable model closed 98% of the gap at eps 0.1 and 0% at
    eps 1-2), which made the most exploitable models look unexploitable (eps50 = inf). Success
    is monotone in eps for the true optimum (a delta feasible at eps is feasible at any larger
    eps), so the first success from below is the right bracket. inf if no budget up to hi
    succeeds; lo if lo already does."""
    n = len(x)
    levels = [lo * 2.0 ** k for k in range(int(np.ceil(np.log2(hi / lo))) + 1)]
    levels[-1] = hi
    first = np.full(n, -1)
    warm = torch.zeros_like(x)                    # best delta so far, carried up the scan
    lo_delta = torch.zeros_like(x)                # best delta at the bracket's lower end
    for k, e in enumerate(levels):
        todo = first < 0
        if not todo.any():
            break
        d = pgd(model, x, c_t, torch.full((n,), e * scale), steps, restarts, seed, init=warm)
        ok = closure(model, x, d, c_t) >= 0.5
        newly = todo & ok
        lo_delta[torch.from_numpy(newly)] = warm[torch.from_numpy(newly)]
        first[newly] = k
        warm = d
    out = np.full(n, np.inf)
    out[first == 0] = lo
    br = np.flatnonzero(first > 0)
    if len(br):
        lo_v = torch.tensor([levels[first[i] - 1] for i in br], dtype=torch.float32)
        hi_v = torch.tensor([levels[first[i]] for i in br], dtype=torch.float32)
        xs, cs, dl = x[br], c_t[br], lo_delta[br].clone()
        for _ in range(iters):
            mid = torch.sqrt(lo_v * hi_v)
            d = pgd(model, xs, cs, mid * scale, steps, restarts, seed, init=dl)
            ok = torch.from_numpy(closure(model, xs, d, cs) >= 0.5)
            hi_v = torch.where(ok, mid, hi_v)
            lo_v = torch.where(ok, lo_v, mid)
            dl = torch.where(ok.view(-1, 1), dl, d)   # failed mids raise the lower end: warm-start from there
        out[br] = hi_v.numpy()
    return out


def closed_form_eps50(probe: LinearProbe, x: torch.Tensor, c_t: torch.Tensor, scale: float) -> np.ndarray:
    """Upper bound for a linear probe: the minimum-norm delta moving its prediction exactly half-way."""
    W = probe.coef.numpy().astype(np.float64)
    with torch.no_grad():
        half = 0.5 * (c_t - probe(x)).numpy().astype(np.float64)
    delta = half @ np.linalg.pinv(W).T
    return np.linalg.norm(delta, axis=1) / scale


def attack_summary(model, x: np.ndarray, c_t: np.ndarray, x_target: np.ndarray, scale: float, probes: dict,
                   pcs: np.ndarray, low_var: np.ndarray, genes, mu: np.ndarray, seed: int = SEED) -> dict:
    """The full attack report for one model on one scoring set (spec: the attack and its metrics).
    probes: name -> torch module (for transfer). pcs: top-50 train PCs (rows orthonormal) of the domain."""
    x_t, c_tt = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(c_t, dtype=torch.float32)
    rows = {}
    best_g = np.full(len(x), -np.inf)
    best_d = torch.zeros_like(x_t)
    for e in EPS_GRID:
        eps_abs = torch.full((len(x),), e * scale)
        d = pgd(model, x_t, c_tt, eps_abs, seed=seed, init=best_d)     # warm start from the smaller budget
        g_warm = closure(model, x_t, d, c_tt)
        # monotone envelope: a delta found at a smaller budget is feasible here too, so report the
        # best over budgets <= e (PGD itself can stall at large budgets when the softmax saturates)
        better = g_warm > best_g
        best_g = np.where(better, g_warm, best_g)
        best_d[torch.from_numpy(better)] = d[torch.from_numpy(better)]
        g, d = best_g.copy(), best_d.clone()
        # an independent cold run at this budget alone (delta starts at 0), to show what the
        # envelope adds; the warm-started run can't fall below the envelope by construction
        g_cold = closure(model, x_t, pgd(model, x_t, c_tt, eps_abs, seed=seed), c_tt)
        row = {"median_g": float(np.median(g)), "share_g_gt_0.5": float(np.mean(g > 0.5)),
               "median_g_cold_this_budget": float(np.median(g_cold))}
        for pname, p in probes.items():
            gp = closure(p, x_t, d, c_tt)
            # transfer: the probe's median closure on the model's attacked inputs; the ratio is a
            # ratio of medians (median probe g / median model g), not a median of ratios
            row[f"transfer_{pname}"] = float(np.median(gp))
            row[f"transfer_ratio_{pname}"] = float(np.median(gp) / np.median(g)) if np.median(g) > 0 else float("nan")
        nat = torch.as_tensor(x_target - x, dtype=torch.float32)
        nat = nat / nat.norm(dim=1, keepdim=True).clamp_min(1e-12) * e * scale
        nat = torch.maximum(nat, -x_t)
        row["natural_g"] = float(np.median(closure(model, x_t, nat, c_tt)))
        if e == 0.03:
            dn = d.numpy().astype(np.float64)
            energy = (dn ** 2).sum(1)
            row["delta_share_top50_pcs"] = float(np.median(((dn @ pcs.T) ** 2).sum(1) / np.maximum(energy, 1e-30)))
            row["delta_share_low_var_genes"] = float(np.median((dn[:, low_var] ** 2).sum(1) / np.maximum(energy, 1e-30)))
            row["delta_r2_change"] = float(np.median(-energy / ((x - mu) ** 2).sum(1)))
            top = np.argsort(-np.abs(dn).mean(0))[:20]
            row["delta_top_genes"] = ",".join(np.asarray(genes)[top])
        rows[e] = row
    masking = closure(model, x_t, pgd(model, x_t, c_tt, torch.full((len(x),), 1.0 * scale), steps=1000, seed=seed), c_tt)
    e50 = eps50(model, x_t, c_tt, scale, seed=seed)
    return {"by_eps": rows, "eps50_median": float(np.median(e50)), "eps50": e50,
            "masking_g_eps1_1000steps": float(np.median(masking)), "masking_flag": bool(np.median(masking) < 0.8)}


# ---------------------------------------------------------------- freezing
def freeze_nets(pairs, idx, domain: str, cfg: dict, epochs: int, seeds=(0, 1, 2)):
    """Refit the chosen config on the given rows (which must all be train) with each seed, for
    `epochs` epochs. Returns (nets, mean, inv_scale, prior) for save_frozen."""
    assert_train_only(pairs, idx)
    X = np.asarray(getattr(pairs, DOMAINS[domain][0])[idx], np.float32)
    C = np.asarray(pairs.composition[idx], np.float32)
    nets = [train_net(X, C, cfg, seed=s, epochs=epochs)[0] for s in seeds]
    mean, inv = input_scale(X, cfg["scaling"])
    return nets, mean, inv, C.mean(0)


def save_marker(mp: "MarkerProbe", name: str, out_dir, registry, provenance: dict) -> str:
    """Save a fitted marker probe and record its canonical hash and provenance in the registry."""
    import json
    from pathlib import Path
    from dgm.models import canonical_hash
    out_dir, registry = Path(out_dir), Path(registry)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{name}.npz", **mp.state())
    digest = canonical_hash(mp.state())
    reg = json.loads(registry.read_text()) if registry.exists() else {}
    reg[name] = {"canonical_hash": digest, "markers": mp.markers, "alpha": mp.alpha_, "file": f"{name}.npz", **provenance}
    registry.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
    return digest


def load_marker(name: str, model_dir, registry, genes, classes, pairs_digest: str) -> "MarkerProbe":
    """Reload a marker probe and verify its canonical hash, gene order, class list and pairs file."""
    import json
    from pathlib import Path
    from dgm.models import canonical_hash, list_hash
    entry = json.loads(Path(registry).read_text())[name]
    if entry["genes_hash"] != list_hash(genes) or entry["classes"] != list(classes) or entry["pairs_hash"] != pairs_digest:
        raise ValueError(f"{name}: genes, classes or pairs file differ from training")
    z = np.load(Path(model_dir) / entry["file"], allow_pickle=False)
    state = {k: z[k] for k in z.files}
    if canonical_hash(state) != entry["canonical_hash"]:
        raise ValueError(f"{name}: file doesn't match its recorded canonical hash")
    mp = MarkerProbe(genes)
    if list(state["markers"]) != mp.markers:
        raise ValueError(f"{name}: marker list differs")
    mp.alpha_ = float(state["alpha"][0])
    mp.ridge_ = SkRidge(alpha=mp.alpha_)
    mp.ridge_.coef_, mp.ridge_.intercept_ = state["coef"], state["intercept"]
    mp.ridge_.n_features_in_ = len(mp.markers)
    return mp


# ---------------------------------------------------------------- the paired donor bootstrap
def paired_bootstrap_diff(pred_a, pred_b, true, donors, n_boot: int = 1000, seed: int = SEED, level: float = 0.95):
    """comp R²(a) - comp R²(b) with a percentile CI, resampling donors (all of a donor's samples
    come along) and recomputing both comp R² values on each resample."""
    donors = np.asarray(donors)
    groups = [np.flatnonzero(donors == d) for d in pd.unique(donors)]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        diffs.append(comp_r2(pred_a[idx], true[idx]) - comp_r2(pred_b[idx], true[idx]))
    lo, hi = np.quantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return comp_r2(pred_a, true) - comp_r2(pred_b, true), float(lo), float(hi)


# ---------------------------------------------------------------- selection inside the folds
def prepare_folds(pairs, domain: str, train_idx: np.ndarray, genes, n_folds: int = 5) -> list[dict]:
    """Everything each inner fold needs, fit on that fold's training part only: inputs/targets,
    the fold ridge probe (Phase 1 Evaluator probe, refit on the fold), and the Phase 1 baselines'
    translations of the fold's held-out sources (robustness constraint)."""
    from dgm.baselines import CORAL, MeanShift, RidgeSpread
    from dgm.metrics import Evaluator
    assert_train_only(pairs, train_idx)
    xf, sf, direction = DOMAINS[domain]
    X, S = getattr(pairs, xf)[train_idx], getattr(pairs, sf)[train_idx]
    C, donors, age = pairs.composition[train_idx], pairs.donor_ids[train_idx], pairs.age_group[train_idx]
    out = []
    for tr, te in inner_folds(donors, age, n_folds):
        ev = Evaluator(X[tr], C[tr], age[tr], donors[tr], list(genes), direction)
        trans = {}
        for model in (MeanShift(), CORAL()):
            trans[model.name] = model.fit(S[tr], X[tr], donors[tr]).translate(S[te])
        rs = RidgeSpread().fit(S[tr], X[tr], donors[tr], comp=C[tr], age=age[tr], genes=list(genes), direction=direction)
        trans[rs.name] = rs.translate(S[te])
        probe_r2 = {k: comp_r2(ev._comp_pred(v), C[te]) for k, v in trans.items()}
        out.append({"X_tr": X[tr], "C_tr": C[tr], "X_te": X[te], "C_te": C[te], "donors_te": donors[te],
                    "translations": trans, "probe": ev.comp_probe_, "probe_r2_real": comp_r2(ev._comp_pred(X[te]), C[te]),
                    "probe_r2_trans": probe_r2, "scale": float(np.median(np.linalg.norm(X[tr] - X[tr].mean(0), axis=1)))})
    return out


_FOLDS = None


def _init_worker(folds):
    global _FOLDS
    _FOLDS = folds
    deterministic()


def run_config(cfg: dict, folds: list[dict] | None = None, epochs: int = 300, seed: int = SEED) -> dict:
    """Train cfg in every fold (one seed), record per-epoch held-out comp R² on real profiles and
    on the fold's baseline translations; pick ONE epoch from the fold-averaged real curve; score
    every fold at it (spec [audit2 F]); apply the robustness constraint (spec [audit2 E])."""
    folds = folds if folds is not None else _FOLDS
    curves, trans_curves = [], []
    for f in folds:
        sets = {"real": (f["X_te"], f["C_te"]), **{f"trans:{k}": (v, f["C_te"]) for k, v in f["translations"].items()}}
        _, hist = train_net(f["X_tr"], f["C_tr"], cfg, seed, epochs, eval_sets=sets)
        curves.append(hist["real"])
        trans_curves.append({k.split(":", 1)[1]: v for k, v in hist.items() if k.startswith("trans:")})
    mean_curve = np.mean(curves, axis=0)
    e = int(np.argmax(mean_curve))                        # 0-based; train for e + 1 epochs
    per_fold = [c[e] for c in curves]
    d = {}
    for b in folds[0]["translations"]:
        d[b] = [max(tc[b][e], 0.0) - max(f["probe_r2_trans"][b], 0.0) for tc, f in zip(trans_curves, folds)]
    mean_d = float(np.mean([v for vs in d.values() for v in vs]))
    per_b = {b: float(np.mean(v)) for b, v in d.items()}
    return {"config": config_name(cfg), **cfg, "epochs": e + 1, "oof_comp_r2": float(np.mean(per_fold)),
            "fold_comp_r2": per_fold, "robust_mean_d": mean_d, **{f"robust_d:{b}": v for b, v in per_b.items()},
            "robust_pass": bool(mean_d >= -0.10 and min(per_b.values()) >= -0.20), "curve": mean_curve.tolist()}


def _pool(folds, workers):
    # spawn, not fork: forking a process whose torch thread pool is already running can deadlock
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                               initargs=(folds,))


def run_grid(folds: list[dict], configs: list[dict], workers: int = 5, epochs: int = 300) -> pd.DataFrame:
    """All configs, `workers` processes x N_THREADS threads each (results don't depend on the
    number of processes, only on the per-process thread count)."""
    with _pool(folds, workers) as ex:
        rows = list(ex.map(run_config, configs, itertools.repeat(None), itertools.repeat(epochs)))
    return pd.DataFrame(rows)


def fold_targets(folds: list[dict], seed: int = SEED) -> list[np.ndarray]:
    """The attack targets for each fold's held-out samples (shared by the nets and the probe)."""
    return [pick_targets(f["donors_te"], seed) for f in folds]


def probe_fold_eps50(folds: list[dict], seed: int = SEED) -> np.ndarray:
    """Pooled per-sample eps50 of each fold's ridge probe over the folds' held-out samples, by the
    same PGD + search as the nets (spec [audit2 B]). Computed once per domain."""
    out = []
    for f, t in zip(folds, fold_targets(folds, seed)):
        x = torch.as_tensor(f["X_te"], dtype=torch.float32)
        c_t = torch.as_tensor(f["C_te"][t], dtype=torch.float32)
        out.append(eps50(probe_module(f["probe"]), x, c_t, f["scale"], seed=seed))
    return np.concatenate(out)


def net_fold_eps50(cfg: dict, epochs: int, folds: list[dict] | None = None, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Pooled per-sample eps50 of cfg over all folds' held-out samples, and the pooled gradient-
    masking check (cold PGD at eps = 1 with 1,000 steps; median g < 0.8 is flagged). cfg is
    retrained in each fold for `epochs` epochs; training is deterministic, so this is the
    selection run's model at that epoch. eps50 found by PGD is an upper bound on the true
    eps50 (a weak attack inflates it), which is why the masking check runs here too."""
    folds = folds if folds is not None else _FOLDS
    e50, mask = [], []
    for f, t in zip(folds, fold_targets(folds, seed)):
        net, _ = train_net(f["X_tr"], f["C_tr"], cfg, seed, epochs)
        x = torch.as_tensor(f["X_te"], dtype=torch.float32)
        c_t = torch.as_tensor(f["C_te"][t], dtype=torch.float32)
        e50.append(eps50(net, x, c_t, f["scale"], seed=seed))
        mask.append(closure(net, x, pgd(net, x, c_t, torch.full((len(x),), 1.0 * f["scale"]), steps=1000, seed=seed), c_t))
    return np.concatenate(e50), np.concatenate(mask)


def run_exploitability(folds: list[dict], candidates: list[tuple[dict, int]], workers: int = 5) -> list[tuple]:
    """net_fold_eps50 for several (cfg, epochs) candidates in parallel."""
    with _pool(folds, workers) as ex:
        return list(ex.map(net_fold_eps50, [c for c, _ in candidates], [e for _, e in candidates]))
