"""Parallel execution of cached translation runs.

`dgm.translation.run_grid` returns each run's result (which holds torch tensors) from the worker
processes; torch passes those tensors between processes as shared file descriptors, and a long grid
broke the process pool ("received 0 items of ancdata"). Here the workers only write each run to the
cache and return nothing; the results are then read from the cache in the calling process. The runs
themselves are `dgm.translation.run`, unchanged, so the results and the cache are identical.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from dgm import translation as tr


def _run_to_cache(args) -> None:
    tr.run(*args)


def run_grid(cfgs: list[dict], seeds=(0, 1, 2), halving: int = 0, workers: int = 5, cache_dir: Path | None = None) -> list[dict]:
    """As `dgm.translation.run_grid` (same jobs, same cache, runs in input order), with results
    passed back through the cache instead of between processes."""
    if cache_dir is None:
        raise ValueError("run_grid passes results through the cache; cache_dir is required")
    key = tr.code_key()
    jobs = [(tr.config(**c), s, halving, cache_dir, key) for c, s in itertools.product(cfgs, seeds)]   # filled, as tr.run names them
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
        list(ex.map(_run_to_cache, jobs))
    for c, s, h, d, k in jobs:                 # every run must now be in the cache, so nothing retrains in this process
        name = hashlib.sha256(json.dumps([k, c, s, h], sort_keys=True, default=str).encode()).hexdigest()[:16]
        assert (Path(d) / f"run_{name}.pkl").exists(), f"run missing from the cache: {tr.config_name(c)}, seed {s}"
    return [tr.run(*job) for job in jobs]     # cache hits: loaded in this process
