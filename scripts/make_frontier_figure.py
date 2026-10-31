"""Draw the conceptual frontier figure for the results page (assets/frontier.png).

    python scripts/make_frontier_figure.py

A drawing, not a result: the points are placed by hand and carry no data. The frontier is computed from the drawn
blend curves exactly as the real one is (at each accuracy, the best realism of any blend at that accuracy or better),
so its shape is right by construction. The layout and styling follow the BioRender draft of this figure.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FILL, CURVE, POINT, TEXT, RED = "#efefef", "#cfcfcf", "#8c8c8c", "#1a1a1a", "#c0392b"
FS, FS_AXIS, FS_TITLE = 15, 19, 23
FONTS = ["Noto Sans", "DejaVu Sans"]   # DejaVu supplies the arrow glyphs Noto Sans lacks


def bezier(p0, p1, c, n=1500):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(c) + t ** 2 * np.array(p1)


def main():
    plt.rcParams.update({"font.family": FONTS, "figure.constrained_layout.use": False})
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.10, 0.17, 0.87, 0.67])            # explicit placement: room for the title and axis labels
    base = {"CORAL": (-0.72, 0.56), "ComBat": (-0.46, 0.33), "mean shift": (-0.21, 0.17), "training mean": (0.0, 0.035)}
    ctrl = {("CORAL", "ComBat"): (-0.60, 0.42), ("CORAL", "mean shift"): (-0.42, 0.40), ("CORAL", "training mean"): (-0.22, 0.52),
            ("ComBat", "mean shift"): (-0.33, 0.24), ("ComBat", "training mean"): (-0.20, 0.30), ("mean shift", "training mean"): (-0.09, 0.14)}
    paths = [bezier(base[a], base[b], c) for (a, b), c in ctrl.items()]
    for p in paths:
        ax.plot(p[:, 0], p[:, 1], color=CURVE, lw=1.1, zorder=1)
    pts = np.vstack(paths)
    xs = np.linspace(-1.0, 0.0, 3000)
    frontier = np.array([pts[pts[:, 0] >= x, 1].max() for x in xs])      # best realism at this accuracy or better
    ax.fill_between(xs, 0, frontier, color=FILL, zorder=0, lw=0)
    ax.plot(xs, frontier, color="black", lw=3.4, ls=(0, (3.0, 1.7)), zorder=3)
    for name, (x, y) in base.items():
        ax.plot(x, y, "o", color=POINT, ms=13, zorder=4, mec="white", mew=1.4)
        dx, dy, ha, va = {"CORAL": (-0.012, -0.03, "right", "top"), "ComBat": (0.0, 0.03, "center", "bottom"),
                          "mean shift": (0.0, -0.035, "center", "top"), "training mean": (-0.025, 0.0, "right", "center")}[name]
        ax.text(x + dx, y + dy, name, fontsize=FS, color=TEXT, ha=ha, va=va, zorder=5)
    i = np.searchsorted(xs, -0.56)                                          # frontier label, along the curve right of CORAL
    ang = np.degrees(np.arctan2(frontier[i + 60] - frontier[i - 60], xs[i + 60] - xs[i - 60]))
    ax.text(xs[i], frontier[i] + 0.022, "frontier", fontsize=FS + 2, color=TEXT, style="italic",
            rotation=ang * 0.75, rotation_mode="anchor", ha="center", va="bottom")
    mx, my = -0.36, 0.68                                                    # a model above the frontier, and its margin
    fm = np.interp(mx, xs, frontier)
    ax.annotate("", xy=(mx, my - 0.02), xytext=(mx, fm),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=3.0, mutation_scale=22, shrinkA=0, shrinkB=0), zorder=5)
    ax.plot(mx, my, "o", color="black", ms=14, zorder=6)
    ax.text(mx, my + 0.035, "a model", fontsize=FS + 2, color=TEXT, ha="center", va="bottom", weight="bold")
    ax.text(mx + 0.025, (my + fm) / 2, "margin", fontsize=FS + 2, color=TEXT, ha="left", va="center", weight="bold")
    ax.plot(-0.96, 0.80, "x", color=RED, ms=13, mew=3.0, zorder=6)
    ax.text(-0.93, 0.80, "random real profile", fontsize=FS, color=RED, ha="left", va="center")
    ax.set_xlim(-1.0, 0.03)
    ax.set_ylim(0, 0.86)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black"); ax.spines[side].set_linewidth(1.6)
    ax.text(0.5, -0.03, "accuracy →", transform=ax.transAxes, fontsize=FS_AXIS, color=TEXT, ha="center", va="top")
    ax.text(0.5, -0.105, "closeness to the true partner", transform=ax.transAxes, fontsize=FS - 1, color=TEXT, ha="center", va="top")
    ax.text(-0.058, 0.5, "realism →", transform=ax.transAxes, fontsize=FS_AXIS, color=TEXT, ha="center", va="center", rotation=90)
    ax.text(-0.025, 0.5, "harder to tell from real profiles", transform=ax.transAxes, fontsize=FS - 1, color=TEXT,
            ha="center", va="center", rotation=90)
    fig.text(0.535, 0.93, "The unpaired accuracy–realism frontier", fontsize=FS_TITLE, color=TEXT, ha="center", va="center")
    out = Path(__file__).resolve().parents[1] / "assets" / "frontier.png"
    fig.savefig(out, dpi=250, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
