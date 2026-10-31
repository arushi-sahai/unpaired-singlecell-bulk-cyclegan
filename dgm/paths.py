"""Where things live (see the README's layout table)."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw" / "soundlife"        # symlinks to the raw inputs (gitignored)
PROCESSED = REPO / "data" / "processed"          # phase 0 outputs (gitignored)
PAIRS_FILE = PROCESSED / "soundlife_pairs.npz"   # model-ready pairs, written by phase 0


def results_dir(phase: int, *sub: str) -> Path:
    """results/phase<N>/<sub...> (committed: small tables and figures), created if missing."""
    path = REPO / "results" / f"phase{phase}"
    path = path.joinpath(*sub)
    path.mkdir(parents=True, exist_ok=True)
    return path
