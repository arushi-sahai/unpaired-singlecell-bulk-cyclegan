"""Setup every phase notebook runs first: the code version that produced its outputs, seeds,
and the figure style."""
from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess

import numpy as np

from dgm.paths import REPO


# What "+dirty" watches, untracked files included: the shared code. Each notebook carries its own
# code in its saved source, and its outputs change on every run, so notebooks/ is not watched
# (otherwise running one notebook would mark every later notebook's run as dirty).
CODE_PATHS = ("dgm", "tests")


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:   # git not installed
        return ""


def git_version() -> str:
    """Current commit, with "+dirty" when the shared code (dgm/, tests/) has uncommitted or
    untracked changes, so outputs never carry a clean hash for code that isn't in it."""
    run = _git
    commit = run("rev-parse", "--short", "HEAD") or "unknown"
    dirty = run("status", "--porcelain", "--", *CODE_PATHS)
    return commit + ("+dirty" if dirty else "")


def source_hash(notebook_json: str) -> str:
    """Hash of a notebook's cell sources (code and markdown) in order, ignoring outputs,
    execution counts and metadata, so it changes only when the notebook's content changes."""
    cells = json.loads(notebook_json)["cells"]
    joined = "\n\x1e".join(c["cell_type"] + "\x1f" + "".join(c["source"]) for c in cells)
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


def notebook_source(name: str) -> str:
    """The saved notebook's source hash and whether that source matches HEAD, e.g.
    "3f2a9c1b4d5e (= HEAD)" or "... (edited since HEAD)". This covers what git_version leaves
    out: a run from notebook source that isn't committed. It hashes the file as saved, so
    unsaved edits in an open editor aren't seen; runs via nbconvert always use the saved file."""
    path = REPO / "notebooks" / name
    here = source_hash(path.read_text())
    committed = _git("show", f"HEAD:notebooks/{name}")
    if not committed:
        return f"{here} (not in HEAD)"
    return f"{here} ({'= HEAD' if source_hash(committed) == here else 'edited since HEAD'})"


def setup(seed: int = 0, notebook: str | None = None) -> dict:
    """Seed numpy/random (and torch if installed), apply the figure style, print provenance:
    the commit of the shared code and, given the notebook's file name, its source hash."""
    random.seed(seed)
    np.random.seed(seed)
    info = {"commit": git_version()}
    if notebook:
        info["notebook_source"] = notebook_source(notebook)
    info.update({"python": platform.python_version(), "numpy": np.__version__, "seed": seed})
    try:
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    from dgm import plotting
    plotting.apply_style()
    print(" | ".join(f"{k}: {v}" for k, v in info.items()))
    return info
