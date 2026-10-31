"""Public color scheme and layout fixes for the results page's notebook figures.

Imported by scripts/make_public_figures.py inside the notebook rerun. It wraps Figure.savefig so each public figure is
adjusted just before it is written. The data, the drawing code, and the class notebooks are unchanged.

One color per meaning, across every figure on the page:
    assays and directions   single-cell blue #2a78d6, bulk orange #eb6834; a direction takes the color of the assay
                            being predicted (single-cell → bulk orange, bulk → single-cell blue)
    models                  per-gene correction and simple alignments gray, unpaired CycleGAN black,
                            composition labels violet #4a3aa7, paired ridge regression green #1baf7a
                            (ridge without the spread step dark green #008300)
    references              random real profile and optimal-transport retrieval gray (no learned pairing; light gray in the partner-line plots),
                            another draw of the same donor pink #e87ba4
    chance level            light gray band; the frontier is a black dashed line
"""
import re
from pathlib import Path

import matplotlib.colors as mc
import matplotlib.figure
import matplotlib.text

INK_2, MUTED, LIGHT = "#52514e", "#8a8984", "#e4e3df"
RECOLOR = {   # figure file stem -> {old color: new color}
    "8.2_test_plane": {"#eda100": INK_2},                       # optimal-transport retrieval: amber read as the bulk orange
    "8.S3_validation_plane": {"#eda100": INK_2},
    "8.5_partner_lines": {"#eb6834": "#b5b3ad", "#c9c8c3": LIGHT},  # random real profile was bulk orange; lighten the real cloud
    "8.S2_partner_lines_bulk_to_single_cell": {"#eb6834": "#b5b3ad", "#c9c8c3": LIGHT},
    "8.3_what_pairs_buy_test": {"#2a78d6": "#008300"},          # ridge without the spread step was single-cell blue
}
WIDEN = {"8.S1_test_intervals": 1.3}                            # room for the comparison labels and the x-labels
PANEL = re.compile(r"^\([a-z]\)$")
SHORT_TICKS = {"plasmablast and immunoglobulin": "plasmablast (Ig)", "CMV-associated cytotoxic": "CMV cytotoxic",
               "identity (XIST, HLA)": "sex and HLA", "interferon response": "interferon", "Mono markers": "monocyte markers"}
TICK_FIGS = ("9.1_person_vs_visit", "9.2_delta_vs_start")   # program names on the x axis: short labels, less whitespace


def _swap(c, table):
    rgba = mc.to_rgba(c)
    new = table.get(mc.to_hex(rgba[:3]).lower())
    return (*mc.to_rgb(new), rgba[3]) if new else None


def _recolor(fig, table):
    pairs = [("color", "get_color"), ("facecolor", "get_facecolor"), ("edgecolor", "get_edgecolor"),
             ("markerfacecolor", "get_markerfacecolor"), ("markeredgecolor", "get_markeredgecolor")]
    for art in fig.findobj():
        for prop, getter in pairs:
            if not hasattr(art, getter) or not hasattr(art, "set_" + prop):
                continue
            try:
                c = getattr(art, getter)()
            except Exception:
                continue
            try:
                arr = mc.to_rgba_array(c)
            except (ValueError, TypeError):
                continue
            if len(arr) == 0:
                continue
            new = [_swap(row, table) for row in arr]
            if not any(n is not None for n in new):
                continue
            out = [n if n is not None else tuple(row) for n, row in zip(new, arr)]
            getattr(art, "set_" + prop)(out if hasattr(art, "get_offsets") else out[0])   # collections take a list


def _layout(fig, stem):
    if stem in ("8.2_test_plane", "8.S3_validation_plane"):
        found = [t for t in fig.findobj(matplotlib.text.Annotation) if t.get_text() == "training mean"]
        found.sort(key=lambda t: fig.axes.index(t.axes))
        for k, t in enumerate(found):                     # a short leader line into open space, clear of the frontier and R² = 0
            ax, xy, size, color = t.axes, t.xy, t.get_fontsize(), t.get_color()
            t.remove()
            dx, dy, ha = (12, 24, "left") if k == 0 else (-40, 8, "right")   # single-cell → bulk: up-right; bulk → single-cell: up-left
            ax.annotate("training mean", xy, xytext=(dx, dy), textcoords="offset points", fontsize=size, color=color,
                        ha=ha, va="bottom", arrowprops=dict(arrowstyle="-", color=color, lw=0.5, shrinkA=1, shrinkB=3))
    if stem in WIDEN:
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * WIDEN[stem], h)
        for t in fig.findobj(matplotlib.text.Text):
            if PANEL.match(t.get_text() or ""):            # panel letters left of the titles
                x, y = t.get_position()
                t.set_position((x - 0.1, y))


def _short_ticks(fig):
    fig.canvas.draw()
    for ax in fig.axes:
        labels = [t.get_text() for t in ax.get_xticklabels()]
        if any(l in SHORT_TICKS for l in labels):
            ax.set_xticks(ax.get_xticks(), [SHORT_TICKS.get(l, l) for l in labels], rotation=40, ha="right", rotation_mode="anchor")
        labels = [t.get_text() for t in ax.get_yticklabels()]
        if any(l in SHORT_TICKS for l in labels):          # program names on the y axis keep their orientation
            ax.set_yticks(ax.get_yticks(), [SHORT_TICKS.get(l, l) for l in labels])


def install():
    original = matplotlib.figure.Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        stem = Path(str(fname)).stem
        if not getattr(self, "_public_done", False):
            if stem in RECOLOR:
                _recolor(self, RECOLOR[stem])
            _layout(self, stem)
            if stem in TICK_FIGS:
                _short_ticks(self)
            self._public_done = True
        return original(self, fname, *args, **kwargs)

    matplotlib.figure.Figure.savefig = savefig
