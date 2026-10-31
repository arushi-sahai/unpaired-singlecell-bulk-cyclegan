"""Synthetic checks of unpaired recovery reported in the Phase 3 notebook (§3.2), written to
results/phase3/synthetic_checks.csv. Same data and training as the known-answer tests in
tests/test_translation.py (affine and low-rank cases at hidden width 64), plus the affine case at
the Sound Life width of 512 for 3,000, 10,000 and 20,000 steps, with no checkpoint selection.

    python -m scripts.phase3_synthetic_checks
"""
import numpy as np
import pandas as pd

from dgm import paths
from tests import test_translation as t

rng = np.random.default_rng(1)                       # the low-rank distortion of test_low_rank_case_is_reported
L = rng.normal(0, 0.3, (40, 2)) @ rng.normal(0, 1, (2, 40))
low_rank = np.maximum(t.D["src"] + (t.D["tgt"] - t.D["tgt"].mean(0)) @ L, 0).astype(np.float32)

rows = []
for case, src, width, steps in [("per-gene affine", t.D["src"], 64, 20000),
                                ("per-gene affine + rank-2 cross-gene distortion", low_rank, 64, 20000),
                                ("per-gene affine", t.D["src"], 512, 3000),
                                ("per-gene affine", t.D["src"], 512, 10000),
                                ("per-gene affine", t.D["src"], 512, 20000)]:
    xs = src[t.half_s].astype(np.float32)
    r2_gan, r2_combat = t._recovery(xs, t.XB, src[t.TEST], t.D["tgt"][t.TEST], hidden=width, steps=steps)
    rows.append({"case": case, "hidden width": width, "steps": steps, "CycleGAN centered R²": r2_gan, "ComBat centered R²": r2_combat})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(paths.results_dir(3) / "synthetic_checks.csv", index=False)
