"""Redraw the Development figures for the results page from saved validation results (no training, no test rows).

    python scripts/make_public_dev_figures.py

- assets/ablations.png: what each component of the unpaired CycleGAN does (results/phase4/ablation_means_val.csv).
- assets/changes.png: five changes to the unpaired CycleGAN (results/phase5/means_val.csv and sd_val.csv).

Both use the validation frontier (results/phase4/p1_frontiers_val.csv, the realism measure used throughout) and the page's
color scheme: the unpaired CycleGAN black, every other condition gray, the frontier a black dashed line.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dgm.frontier import envelope  # noqa: E402
from dgm.plotting import INK, INK_2, MUTED, apply_style  # noqa: E402

DIRS = [("S->B", "single-cell → bulk"), ("B->S", "bulk → single-cell")]
FRONT = pd.read_csv(ROOT / "results/phase4/p1_frontiers_val.csv")
FRONT = FRONT[FRONT.measure == "H1 (logistic, 10 PCs)"]


def frontier(ax, d):
    env = envelope(FRONT[FRONT.direction == d])
    ax.plot(env.r2_centered, env.realism, color=INK, lw=0.9, ls="--", zorder=1)


def panel(ax, d, title, rows, sds=None, xlim=None):
    frontier(ax, d)
    for name, label, marker, color in rows:
        x, y = rows_xy[name][d]
        kw = dict(color=color, ms=6.5 if color == INK else 5.5, mec="white", mew=0.5, zorder=4 if color == INK else 3)
        if sds is not None and name in sds:
            ax.errorbar(x, y, xerr=sds[name][d][0], yerr=sds[name][d][1], fmt=marker, elinewidth=0.6, capsize=0, **kw)
        else:
            ax.plot(x, y, marker, **kw)
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel("accuracy: centered R²", fontsize=7)
    ax.set_ylim(0, 0.9)
    if xlim:
        ax.set_xlim(*xlim)


def legend(fig, rows):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK, lw=0.9, ls="--", label="unpaired frontier (validation)")]
    h += [Line2D([], [], ls="", marker=mk, color=c, ms=5.5, mec="white", mew=0.5, label=lab) for _, lab, mk, c in rows]
    fig.legend(handles=h, loc="outside lower center", ncol=4, fontsize=6.3, frameon=False)


rows_xy = {}


def ablations():
    m = pd.read_csv(ROOT / "results/phase4/ablation_means_val.csv").set_index("condition")
    rows = [("full model", "unpaired CycleGAN", "o", INK),
            ("A1 adversarial loss off", "no adversarial loss", "s", MUTED),
            ("A2 affine-only", "per-gene scale and shift only", "^", MUTED),
            ("A2 identity loss off", "no identity loss", "v", MUTED),
            ("A3 cycle weight 0", "no cycle loss", "D", MUTED),
            ("A3 cycle weight 40", "cycle loss doubled", "<", MUTED)]
    for name, *_ in rows:
        rows_xy[name] = {d: (m.loc[name, f"{d} r2_centered"], m.loc[name, f"{d} realism"]) for d, _ in DIRS}
    fig, axs = plt.subplots(1, 2, figsize=(6.875, 3.0), constrained_layout=True)
    for ax, (d, lab), letter in zip(axs, DIRS, "ab"):
        panel(ax, d, lab, rows)
        ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    axs[0].set_ylabel("realism (1 = indistinguishable from real)", fontsize=7)
    legend(fig, rows)
    fig.savefig(ROOT / "assets/ablations.png", dpi=300, bbox_inches="tight", facecolor="white")


def changes():
    m = pd.read_csv(ROOT / "results/phase5/means_val.csv").set_index("condition")
    sd = pd.read_csv(ROOT / "results/phase5/sd_val.csv").set_index("condition")
    rows = [("M1", "unpaired CycleGAN", "o", INK),
            ("A-idoff identity loss off", "no identity loss", "s", MUTED),
            ("A-idS identity loss off for single-cell → bulk", "no identity loss, one direction", "D", MUTED),
            ("A-anc start-anchored identity loss", "identity loss toward the per-gene correction", "^", MUTED),
            ("A-dist distance preservation, weight 0.56", "distance-preserving loss", "v", MUTED),
            ("A-bn bottleneck residual (8)", "bottleneck generator", "<", MUTED)]
    sds = {}
    for name, *_ in rows:
        rows_xy[name] = {d: (m.loc[name, f"{d} r2_centered"], m.loc[name, f"{d} realism"]) for d, _ in DIRS}
        if name in sd.index:
            sds[name] = {d: (sd.loc[name, f"{d} r2_centered"], sd.loc[name, f"{d} realism"]) for d, _ in DIRS}
    fig, axs = plt.subplots(1, 2, figsize=(6.875, 3.0), constrained_layout=True)
    for ax, (d, lab), letter in zip(axs, DIRS, "ab"):
        panel(ax, d, lab, rows, sds=sds, xlim=(-0.6, 0.1) if d == "S->B" else (-0.6, 0.25))
        ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    axs[0].set_ylabel("realism (1 = indistinguishable from real)", fontsize=7)
    legend(fig, rows)
    fig.savefig(ROOT / "assets/changes.png", dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    apply_style()
    ablations()
    changes()
    print("assets/ablations.png, assets/changes.png")
