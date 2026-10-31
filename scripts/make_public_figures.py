"""Redraw the results page's notebook figures with public wording ("predicted" for "translated").

    python scripts/make_public_figures.py

The class notebooks (Phases 8 and 9) keep the machine-learning term "translation". This script runs their code in
memory with two changes only: figure text is relabeled (see RELABEL: "predicted" for "translated", and the page's baseline and deviation terms) and every output goes to a temporary folder.
It checks that the rerun reproduces the committed result tables, then copies the public figures into assets/.
Run it after Phase 8 or Phase 9 is re-executed. The notebooks, results/ and report/ are not modified.
"""
import filecmp
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RELABEL = [("Translations", "Predicted profiles"), ("translations", "predicted profiles"),
           ("Translation", "Predicted profile"), ("translation", "predicted profile"),
           ("Translated", "Predicted"), ("translated", "predicted"),
           ("translators", "methods"), ("translator", "method"),
           ("between-donor (person)", "donor baselines"), ("within-donor (visit)", "visit deviations"),
           ("null band (no-information reference)", "chance level (random real profiles)"),
           ("starting alignment", "per-gene correction"), ("generator start", "per-gene correction"),
           ("(its start)", "(its per-gene correction)"), ("(start)", "(per-gene correction)"),
           ("paired ridge, not spread out", "ridge regression without the spread step"),
           ("paired ridge −", "paired ridge regression −"),
           ("margin beyond the frontier (realism units)", "margin (realism units)")]
PUBLIC = {                                   # assets/<name> <- the rerun's output, relative to its temporary root
    "phase8": {name: f"web/assets/{name}" for name in
               ("lines_sb.png", "lines_bs.png", "plane.png", "plane_val.png", "pairs.png", "mechanism.png", "intervals.png")},
    "phase9": {"person_visit.png": "results/phase9/figures/9.1_person_vs_visit.png",
               "delta_start.png": "results/phase9/figures/9.2_delta_vs_start.png",
               "absent_cells.png": "results/phase9/figures/9.3_absent_cells.png",
               "vaccine.png": "results/phase9/figures/9.4_vaccine_response.png"},
}
NOTEBOOK = {"phase8": "phase8_figures.ipynb", "phase9": "phase9_biology.ipynb"}

PATCH = r'''
import sys, re
from pathlib import Path
ROOT_ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "dgm").is_dir())
sys.path.insert(0, str(ROOT_))
from dgm import paths
TMP_ = Path({tmp!r})
_results_dir = paths.results_dir
def _redirected(phase, *sub):                 # this phase's outputs go to the temporary folder; inputs are read as usual
    if phase == {phase}:
        p = TMP_ / "results" / f"phase{{phase}}"
        p = p.joinpath(*sub)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return _results_dir(phase, *sub)
paths.results_dir = _redirected
import matplotlib.text
_RELABEL = {relabel!r}
_set_text = matplotlib.text.Text.set_text
def _public_text(self, s):
    if isinstance(s, str):
        for a, b in _RELABEL:
            s = re.sub((r"\b" if a[0].isalnum() else "") + re.escape(a) + (r"\b" if a[-1].isalnum() else ""), b, s)
    return _set_text(self, s)
matplotlib.text.Text.set_text = _public_text
sys.path.insert(0, str(ROOT_ / "scripts"))
import public_figure_style
public_figure_style.install()
'''


def run(key: str, tmp: Path) -> None:
    nb = nbformat.read(ROOT / "notebooks" / NOTEBOOK[key], as_version=4)
    phase = int(key[-1])
    if key == "phase8":                       # the report and page outputs go to the temporary folder too
        shutil.copytree(ROOT / "report", tmp / "report")
        (tmp / "web").mkdir()
        shutil.copytree(ROOT / "assets", tmp / "web" / "assets")
        shutil.copy2(ROOT / "index.html", tmp / "web" / "index.html")
        old = 'REPORT, WEB = ROOT / "report", ROOT'
        cell = next(c for c in nb.cells if c.cell_type == "code" and old in c.source)
        cell.source = cell.source.replace(old, f'REPORT, WEB = Path({str(tmp / "report")!r}), Path({str(tmp / "web")!r})')
    nb.cells.insert(0, nbformat.v4.new_code_cell(PATCH.format(tmp=str(tmp), phase=phase, relabel=RELABEL)))
    NotebookClient(nb, timeout=5000, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    # the rerun must reproduce the committed result tables
    for f in sorted((ROOT / "results" / key).glob("*.csv")):
        assert filecmp.cmp(f, tmp / "results" / key / f.name, shallow=False), f"rerun differs: {f.name}"


def class_outputs() -> dict:
    """Content hash of every file under the class outputs; the rerun must leave them unchanged."""
    return {f: hashlib.sha256(f.read_bytes()).hexdigest()
            for d in ("results", "report", "notebooks") for f in sorted((ROOT / d).rglob("*")) if f.is_file()}


def main():
    before = class_outputs()
    for key in sys.argv[1:] or ["phase8", "phase9"]:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            run(key, tmp)
            for dst, src in PUBLIC[key].items():
                shutil.copy2(tmp / src, ASSETS / dst)
                print(f"assets/{dst} <- {key} rerun ({src})")
    assert class_outputs() == before, "the rerun changed files under results/, report/ or notebooks/"


if __name__ == "__main__":
    main()
