"""Draw the project overview schematic (assets/overview.png) for the README and the results page.

A drawing, not a result: panel (d) illustrates how the unpaired accuracy-realism frontier is built;
its points are placed by hand and carry no data. Run from the repository root:

    python scripts/make_overview_figure.py
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dgm.plotting import DOMAIN, GRID, INK, INK_2, MUTED, SERIES, SURFACE, apply_style  # noqa: E402

BLUE, ORANGE = DOMAIN["single-cell"], DOMAIN["bulk"]
VIOLET, GREEN = SERIES[6], SERIES[2]
PALE = {"blue": "#eaf2fb", "orange": "#fdf0ea", "gray": "#f3f2ef", "violet": "#efedf8", "green": "#e8f7f1"}
FS, FS_SMALL, FS_TITLE = 9.5, 8.5, 11


def box(ax, x, y, w, h, ec, fc=SURFACE, lw=1.4, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02",
                                ec=ec, fc=fc, lw=lw, ls=ls, transform=ax.transAxes))


def arrow(ax, a, b, color=INK_2, lw=1.3):
    ax.add_patch(FancyArrowPatch(a, b, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw))


def text(ax, x, y, s, size=FS, color=INK, **kw):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=size, color=color, **kw)


def title(ax, s):
    ax.text(0.0, 1.0, s, transform=ax.transAxes, fontsize=FS_TITLE, fontweight="bold", va="bottom", ha="left")


def panel_setup(ax):
    title(ax, "(a) Inputs and outputs")
    box(ax, 0.00, 0.50, 0.36, 0.38, BLUE, PALE["blue"])
    text(ax, 0.18, 0.80, "Single-cell pseudobulk", ha="center", fontweight="bold")
    text(ax, 0.18, 0.70, "PBMCs, summed per sample", ha="center", size=FS_SMALL, color=INK_2)
    text(ax, 0.18, 0.60, "33 training donors", ha="center", size=FS_SMALL, color=INK_2)
    text(ax, 0.18, 0.535, "5,000 genes", ha="center", size=FS_SMALL, color=INK_2)
    box(ax, 0.64, 0.50, 0.36, 0.38, ORANGE, PALE["orange"])
    text(ax, 0.82, 0.80, "Whole-blood bulk", ha="center", fontweight="bold")
    text(ax, 0.82, 0.70, "one profile per blood draw", ha="center", size=FS_SMALL, color=INK_2)
    text(ax, 0.82, 0.60, "35 other training donors", ha="center", size=FS_SMALL, color=INK_2)
    text(ax, 0.82, 0.535, "5,000 genes", ha="center", size=FS_SMALL, color=INK_2)
    arrow(ax, (0.375, 0.78), (0.625, 0.78), color=INK, lw=1.5)
    text(ax, 0.50, 0.81, "predict", ha="center", va="bottom", size=FS_SMALL)
    arrow(ax, (0.625, 0.60), (0.375, 0.60), color=INK, lw=1.5)
    text(ax, 0.50, 0.57, "predict back", ha="center", va="top", size=FS_SMALL)
    text(ax, 0.50, 0.69, "CycleGAN", ha="center", va="center", size=FS_SMALL, color=INK_2, style="italic")
    text(ax, 0.50, 0.43, "Different donors: the CycleGAN never sees a pair; each donor's other assay is withheld", ha="center", size=FS_SMALL, color=INK_2)
    box(ax, 0.00, 0.08, 1.00, 0.24, MUTED, PALE["gray"], ls="--")
    text(ax, 0.03, 0.25, "Scoring only: held-out donors, both assays from the same draw",
         fontweight="bold", size=FS_SMALL, va="center")
    text(ax, 0.03, 0.15, "validation: 135 pairs, 14 donors     test: 119 pairs, 14 donors (scored once)",
         size=FS_SMALL, color=INK_2, va="center")


def panel_ladder(ax):
    title(ax, "(b) Models: how much pair information each uses")
    rows = [
        ("Simple alignments", "mean shift, ComBat, CORAL", "pairs used: none", MUTED, PALE["gray"]),
        ("Unpaired CycleGAN", "adversarial + cycle-consistency losses, built on mean shift or ComBat", "pairs used: none", INK, SURFACE),
        ("CycleGAN + cross-fitted labels", "the same, plus a composition loss (panel c)",
         "pairs used: other donors' pairs", VIOLET, PALE["violet"]),
        ("CycleGAN + composition labels", "the same, plus a composition loss (panel c)",
         "pairs used: all training pairs, including each sample's own", VIOLET, PALE["violet"]),
        ("Paired ridge regression", "linear regression on matched profiles",
         "pairs used: all 582 training pairs", GREEN, PALE["green"]),
    ]
    h, gap, top = 0.165, 0.03, 0.96
    for i, (name, what, pairs, ec, fc) in enumerate(rows):
        y = top - (i + 1) * h - i * gap
        box(ax, 0.00, y, 0.86, h, ec, fc, lw=1.6 if i == 1 else 1.3)
        text(ax, 0.025, y + h * 0.74, name, fontweight="bold", va="center")
        text(ax, 0.025, y + h * 0.45, what, size=FS_SMALL, color=INK_2, va="center")
        text(ax, 0.025, y + h * 0.17, pairs, size=FS_SMALL, color=INK_2, va="center", style="italic")
    arrow(ax, (0.92, 0.95), (0.92, 0.03), color=INK_2, lw=1.4)
    text(ax, 0.95, 0.50, "more pair information", rotation=90, ha="left", va="center", size=FS_SMALL, color=INK_2)


def panel_labels(ax):
    title(ax, "(c) Composition labels")
    text(ax, 0.00, 0.93, "1. A composition network reads cell-type fractions", fontweight="bold", size=FS_SMALL, va="center")
    box(ax, 0.00, 0.74, 0.22, 0.13, MUTED, PALE["gray"], lw=1.0)
    text(ax, 0.11, 0.805, "a profile\n(5,000 genes)", ha="center", va="center", size=FS_SMALL)
    arrow(ax, (0.225, 0.805), (0.275, 0.805))
    box(ax, 0.28, 0.74, 0.30, 0.13, MUTED, PALE["gray"], lw=1.0)
    text(ax, 0.43, 0.805, "linear layer + softmax", ha="center", va="center", size=FS_SMALL)
    arrow(ax, (0.585, 0.805), (0.635, 0.805))
    left = 0.64
    for f, c in zip([0.45, 0.25, 0.12, 0.11, 0.02, 0.05], SERIES[:6]):
        ax.add_patch(Rectangle((left, 0.815), 0.30 * f, 0.055, fc=c, ec=SURFACE, lw=0.8, transform=ax.transAxes))
        left += 0.30 * f
    text(ax, 0.64, 0.78, "6 cell-type fractions", va="center", size=FS_SMALL)
    for k, (lab, c) in enumerate(zip(["T", "Mono", "B", "NK", "DC", "Other"], SERIES[:6])):
        x, y = 0.64 + (k % 3) * 0.10, 0.735 - (k // 3) * 0.035
        ax.add_patch(Rectangle((x, y - 0.012), 0.015, 0.024, fc=c, ec="none", transform=ax.transAxes))
        text(ax, x + 0.02, y, lab, size=7, color=INK_2, va="center")
    text(ax, 0.00, 0.645, "One network per assay, trained on fractions counted from each draw's cells", size=FS_SMALL, color=INK_2, va="center")
    text(ax, 0.00, 0.605, "(so the bulk network needs pairs)", size=FS_SMALL, color=INK_2, va="center")
    text(ax, 0.00, 0.52, "2. An extra loss on the CycleGAN", fontweight="bold", size=FS_SMALL, va="center")
    text(ax, 0.00, 0.465, "single-cell → bulk: bulk network's reading of the prediction ≈ the sample's counted fractions",
         size=FS_SMALL, va="center")
    text(ax, 0.00, 0.415, "bulk → single-cell: single-cell network's reading of the prediction ≈ bulk network's reading of the input",
         size=FS_SMALL, va="center")
    text(ax, 0.00, 0.31, "3. Which network reads each training sample", fontweight="bold", size=FS_SMALL, va="center")
    text(ax, 0.00, 0.255, "Composition labels:", size=FS_SMALL, color=VIOLET, fontweight="bold", va="center")
    text(ax, 0.245, 0.255, "one network, all 68 training donors (has seen each sample's pair)", size=FS_SMALL, color=INK_2, va="center")
    text(ax, 0.00, 0.205, "Cross-fitted labels:", size=FS_SMALL, color=VIOLET, fontweight="bold", va="center")
    text(ax, 0.245, 0.205, "five networks; each sample is read by one that never saw its donor", size=FS_SMALL, color=INK_2, va="center")
    text(ax, 0.00, 0.10, "Composition is scored separately, from 28 marker genes.", size=FS_SMALL, color=INK_2, style="italic", va="center")


def _bezier(p0, p1, c, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(c) + t ** 2 * np.array(p1)


def panel_frontier(ax):
    ax.set_title("(d) The unpaired accuracy–realism frontier (illustration)", loc="left", fontsize=FS_TITLE,
                 fontweight="bold", pad=8)
    base = {"training mean": (0.0, 0.0), "mean shift": (-0.08, 0.05), "ComBat": (-0.46, 0.27), "CORAL": (-0.72, 0.55)}
    ctrl = {("training mean", "mean shift"): (0.03, 0.03), ("training mean", "ComBat"): (-0.10, 0.22),
            ("training mean", "CORAL"): (-0.18, 0.42), ("mean shift", "ComBat"): (-0.25, 0.20),
            ("mean shift", "CORAL"): (-0.30, 0.40), ("ComBat", "CORAL"): (-0.62, 0.38)}
    pts = []
    for (a, b), c in ctrl.items():
        path = _bezier(base[a], base[b], c)
        ax.plot(path[:, 0], path[:, 1], color=GRID, lw=1.0, zorder=1)
        pts.append(path)
    pts = np.vstack(pts)
    r = np.linspace(-1.12, pts[:, 0].max(), 400)
    F = np.array([pts[pts[:, 0] >= ri, 1].max() for ri in r])
    ax.fill_between(r, 0, F, color=PALE["gray"], zorder=0, lw=0)
    ax.plot(r, F, color=INK, lw=1.6, ls="--", zorder=3)
    ax.text(-0.56, 0.14, "blends of two simple alignments (gray)", fontsize=FS_SMALL, color=INK_2, ha="center", va="center")
    for name, (x, y) in base.items():
        ax.plot(x, y, "o", color=INK_2, ms=5, zorder=4, mec=SURFACE, mew=1)
        dx, dy, ha = {"CORAL": (0.0, 0.04, "center"), "ComBat": (0.02, 0.035, "left"),
                      "mean shift": (-0.03, -0.01, "right"), "training mean": (-0.03, -0.045, "right")}[name]
        shown = {"training mean": "training mean (R² = 0)"}.get(name, name)
        ax.text(x + dx, y + dy, shown, fontsize=FS_SMALL, color=INK_2, ha=ha)
    ax.text(-1.10, 0.52, "frontier: the most realistic point\nat this accuracy or better", fontsize=FS_SMALL, color=INK, ha="left", va="top")
    mx, my = -0.24, 0.56
    f_m = np.interp(mx, r, F)
    ax.annotate("", xy=(mx, my), xytext=(mx, f_m),
                arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=1.4, mutation_scale=10), zorder=5)
    ax.plot(mx, my, "o", color=VIOLET, ms=7, zorder=6, mec=SURFACE, mew=1)
    ax.text(mx + 0.03, my + 0.02, "a model", fontsize=FS_SMALL, color=INK, ha="left", va="bottom")
    ax.text(mx + 0.03, (my + f_m) / 2, "margin: how much more\nrealistic than the frontier\nat the same accuracy",
            fontsize=FS_SMALL, color=INK, ha="left", va="center")
    ax.plot(-0.98, 0.71, "x", color=INK_2, ms=7, mew=1.6, zorder=6)
    ax.text(-0.95, 0.71, "random real profile (ignores the input)", fontsize=FS_SMALL, color=INK_2, ha="left", va="center")
    ax.set_xlim(-1.15, 0.10)
    ax.set_ylim(-0.08, 0.95)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("accuracy: centered R² against the true partner  →", fontsize=FS_SMALL)
    ax.set_ylabel("realism: a classifier can't tell it from real  →", fontsize=FS_SMALL)


def main():
    apply_style()
    fig = plt.figure(figsize=(12.5, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.30, wspace=0.12, left=0.03, right=0.985, top=0.95, bottom=0.07)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    fig.set_constrained_layout(False)
    for ax in axes[:3]:
        ax.set_axis_off()
    panel_setup(axes[0])
    panel_ladder(axes[1])
    panel_labels(axes[2])
    panel_frontier(axes[3])
    out = Path(__file__).resolve().parents[1] / "assets" / "overview.png"
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
