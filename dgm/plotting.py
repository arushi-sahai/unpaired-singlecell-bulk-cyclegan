"""Figure style, sizes, colors and display names shared by every phase notebook.

Figures are drawn at their final printed size for the CVPR two-column template (single column
3.25 in, full width 6.875 in), so 7-8 pt text stays legible in the report. Figures carry no
suptitle: the caption (results/phaseN/figures/captions.md) says what the figure shows.

Colors are from the validated reference palette (dataviz skill, light mode). The domain
pair (blue, orange) and the age pair (aqua, violet) pass the all-pairs colorblind and
normal-vision checks together. Aqua is below 3:1 contrast on white, so plots that use it
carry a legend or direct labels.

``LABELS`` maps code names to the display text used in figures, tables and prose, so the
three can't disagree.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]   # fixed order
DOMAIN = {"single-cell": SERIES[0], "bulk": SERIES[1]}
AGE = {"young (25-37)": SERIES[2], "older (55-67)": SERIES[6]}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
THRESHOLD = dict(color=INK_2, linestyle="--", linewidth=0.8)   # QC cut-offs and reference lines

SINGLE, DOUBLE = 3.25, 6.875   # CVPR column and full-page widths, inches

LABELS = {
    # directions and domains
    "S->B": "single-cell → bulk", "B->S": "bulk → single-cell",
    "sc": "single-cell pseudobulk", "bulk": "whole-blood bulk",
    # metrics
    "r2_centered": "centered R²", "r_centered": "centered correlation", "r_raw": "correlation",
    "c2st_auroc": "C2ST AUROC", "realism": "realism (1 − 2|AUROC − 0.5|)", "mmd": "MMD²",
    "comp_r2": "composition R²", "comp_r": "composition correlation", "marker_rho": "marker ρ",
    "gran_gap": "granulocyte gap", "age_auroc": "age AUROC", "diversity": "diversity ratio",
    "nn_ratio": "nearest-neighbor ratio", "top1": "top-1 partner rate", "within_top1": "within-donor top-1 rate",
    "composite": "selection score", "eps50": "ε₅₀",
    # composition networks and classes
    "C_sc": "C_S", "C_bulk": "C_B", "T cell": "T", "Monocyte": "Mono", "B cell": "B", "NK cell": "NK",
    # baselines
    "identity": "identity", "mean shift": "mean shift", "ComBat": "ComBat", "CORAL": "CORAL", "Harmony": "Harmony",
    "ridge (paired)": "ridge", "ridge + spread": "ridge + spread", "ridge + PC calibration": "ridge + PC calibration",
    "paired MLP": "paired MLP",
}


def label(name: str) -> str:
    """Display text for a code name (the name itself if it has none)."""
    return LABELS.get(name, name)


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.5, "lines.markersize": 4,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.prop_cycle": plt.cycler(color=SERIES), "figure.constrained_layout.use": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def figure(width: str = "single", height: float = 2.4, nrows: int = 1, ncols: int = 1, **kw):
    """A figure at its printed size: width "single" (3.25 in) or "double" (6.875 in)."""
    w = {"single": SINGLE, "double": DOUBLE}[width]
    return plt.subplots(nrows, ncols, figsize=(w, height), **kw)


def panel_labels(axes, x: float = -0.02, y: float = 1.02):
    """Bold (a), (b), ... at the top left of each axis."""
    for i, ax in enumerate(list(axes)):
        ax.text(x, y, f"({chr(97 + i)})", transform=ax.transAxes, fontweight="bold", ha="right", va="bottom")


def save_show(fig, out_dir: Path, name: str):
    """Save a figure to <out_dir>/<name>.png (300 dpi) and .pdf (vector), and show it inline.
    Returns nothing, so a figure cell doesn't echo the path."""
    fig.savefig(Path(out_dir) / f"{name}.png", bbox_inches="tight", metadata={"Software": None})
    fig.savefig(Path(out_dir) / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "Producer": None})   # no timestamp, so reruns are byte-identical
    plt.show()
