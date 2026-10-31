"""Phase 9: biological interpretation of the translations (post-hoc; validation and training rows only).

Program scores, tracking correlations split into a person (between-donor) and a visit (within-donor)
part, the training-row between-donor variance share, composition-adjusted (partial) correlations,
the vaccine-response unit, and the per-gene move analysis. Definitions are in notes/phase9_spec.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

GMT_SHA256 = "ee2463540042078bfa3f67828e1e223bb354446d9fbb4d22845866835ba5c772"   # h.all.v2024.1.Hs.symbols.gmt
GMT_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt"

# The short list, fixed in the spec before any Phase 9 run (lineage programs come from dgm.composition.MARKERS).
PROGRAMS = {
    "erythroid": ["ALAS2", "HBB", "HBA1", "HBA2", "CA1", "SLC4A1", "HBD", "AHSP"],
    "platelet": ["PPBP", "PF4"],
    "interferon response": ["IFI27", "IFI44L", "IFI44", "ISG15", "RSAD2", "MX1", "SIGLEC1", "OAS1", "OAS3", "IFIT1",
                            "IFIT3", "HERC5", "LY6E"],
    "plasmablast and immunoglobulin": ["JCHAIN", "MZB1", "TNFRSF17", "TXNDC5", "IGHG1", "IGHG2", "IGHG3", "IGHG4",
                                       "IGHA1", "IGKC", "IGLC2", "XBP1", "PRDM1", "CD38"],
    "CMV-associated cytotoxic": ["KLRC2", "GZMH", "B3GAT1", "GZMB", "PRF1", "GNLY", "FCGR3A"],
    "identity (XIST, HLA)": ["XIST", "HLA-DRB5", "HLA-DQA1", "HLA-DQB1", "HLA-DQA2"],
}
SHRINK = 0.3            # the Part D shrinkage reference: the start moved this far toward the training-half targets' mean
FLU_YEARS = ("Flu Year 1", "Flu Year 2")


# ---------------------------------------------------------------- guards and gene sets
def mask_test(pairs) -> None:
    """Set the test rows' expression and composition to NaN in place (every array a Phase 9 code path reads)."""
    t = pairs.split == "test"
    for name in ("x_sc", "x_bulk", "composition"):
        getattr(pairs, name)[t] = np.nan
    assert all(np.isnan(getattr(pairs, n)[t]).all() for n in ("x_sc", "x_bulk", "composition"))
    assert not np.isnan(pairs.x_sc[~t]).any() and not np.isnan(pairs.x_bulk[~t]).any()


def row_guard(pairs):
    """rows(split) for Phase 9: training and validation only; refuses test."""
    def rows(split):
        assert split in ("train", "val"), "Phase 9 never selects test rows"
        return pairs.idx(split)
    return rows


def read_gmt(path: Path, panel, min_size: int = 10, check: bool = True) -> tuple[dict, pd.DataFrame]:
    """Gene sets from a GMT file restricted to the panel: {name: panel genes (sorted, unique)} for sets with at
    least min_size panel genes, and a coverage table for every set (size, panel genes, kept, unmatched symbols)."""
    raw = Path(path).read_bytes()
    if check:
        assert hashlib.sha256(raw).hexdigest() == GMT_SHA256, "the Hallmark file differs from the pinned release"
    panel = set(panel)
    sets, rows = {}, []
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        name, _, *genes = line.split("\t")
        genes = sorted({g for g in genes if g})
        inside = [g for g in genes if g in panel]
        kept = len(inside) >= min_size
        if kept:
            sets[name] = inside
        rows.append({"set": name, "genes": len(genes), "panel genes": len(inside), "kept": kept,
                     "unmatched": len(genes) - len(inside)})
    return sets, pd.DataFrame(rows)


# ---------------------------------------------------------------- scores and correlations
def zparams(Y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene mean and SD of the training targets (the only rows z-scores are fit on)."""
    Y_train = np.asarray(Y_train, float)
    assert np.isfinite(Y_train).all(), "z-score parameters must come from training rows only"
    sd = Y_train.std(0)
    return Y_train.mean(0), np.where(sd > 0, sd, 1.0)


def program_score(Y: np.ndarray, cols, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Per sample, the mean over the program's genes of training-z-scored expression."""
    cols = np.asarray(cols)
    return ((np.asarray(Y, float)[:, cols] - mu[cols]) / sd[cols]).mean(1)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r; NaN when either input is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else float("nan")


def colwise_r(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-column Pearson r between two (samples x genes) arrays; NaN for constant columns."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    A, B = A - A.mean(0), B - B.mean(0)
    den = np.sqrt((A * A).sum(0) * (B * B).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-12, (A * B).sum(0) / np.where(den > 1e-12, den, 1), np.nan)


def slope(true: np.ndarray, pred: np.ndarray) -> float:
    """Least-squares slope of the true score on the translated score (r ignores scale; this does not)."""
    p = np.asarray(pred, float) - np.mean(pred)
    den = (p * p).sum()
    return float((p * (np.asarray(true, float) - np.mean(true))).sum() / den) if den > 1e-12 else float("nan")


def donor_means(x: np.ndarray, donors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(means, labels) per donor label (labels from donor_boot_labeled make a donor drawn twice count twice)."""
    labels = pd.unique(donors)
    return np.array([np.asarray(x)[donors == d].mean(0) for d in labels]), labels


def within(x: np.ndarray, donors: np.ndarray) -> np.ndarray:
    """x minus its donor's mean (per donor label)."""
    x = np.asarray(x, float).copy()
    for d in pd.unique(donors):
        m = donors == d
        x[m] = x[m] - x[m].mean(0)
    return x


def tracking(true: np.ndarray, pred: np.ndarray, donors: np.ndarray) -> dict:
    """Overall, between-donor (donor means) and within-donor (donor means removed) r, and the slope."""
    donors = np.asarray(donors)
    tm, _ = donor_means(true, donors)
    pm, _ = donor_means(pred, donors)
    return {"r": pearson(true, pred), "between r": pearson(tm, pm),
            "within r": pearson(within(true, donors), within(pred, donors)), "slope": slope(true, pred)}


def codes_of(labels: np.ndarray) -> np.ndarray:
    """Integer codes 0..k-1 of donor labels, in order of first appearance."""
    _, first, inv = np.unique(np.asarray(labels), return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return order[inv]


def tracking_codes(true: np.ndarray, pred: np.ndarray, codes: np.ndarray) -> tuple[float, float, float]:
    """(r, between r, within r) as in tracking(), vectorized over donor codes (for the bootstrap)."""
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    k = codes.max() + 1
    cnt = np.bincount(codes, minlength=k)
    mt, mp = np.bincount(codes, true, k) / cnt, np.bincount(codes, pred, k) / cnt
    return pearson(true, pred), pearson(mt, mp), pearson(true - mt[codes], pred - mp[codes])


def donor_boot_labeled(donors: np.ndarray, n_boot: int = 1000, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """The resamples of dgm.analysis.donor_boot (same seed, same draws), each with a donor label per row that is
    unique per drawn copy, so a donor drawn twice forms two donor groups in the between- and within-donor r."""
    donors = np.asarray(donors)
    uniq = pd.unique(donors)
    groups = [np.flatnonzero(donors == d) for d in uniq]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        lab = np.concatenate([np.full(len(groups[i]), f"{uniq[i]}#{j}") for j, i in enumerate(pick)])
        out.append((idx, lab))
    return out


def fisher_ci(r: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Fisher-z interval for a correlation from n points."""
    from scipy.stats import norm
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1:
        return float("nan"), float("nan")
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    q = norm.ppf(1 - (1 - level) / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def between_share(x: np.ndarray, donors: np.ndarray) -> float:
    """One-way ICC(1) of a score: the share of its variance between donors, with the unbalanced-design group size."""
    x, donors = np.asarray(x, float), np.asarray(donors)
    labels = pd.unique(donors)
    groups = [x[donors == d] for d in labels]
    n, k = len(x), len(groups)
    grand = x.mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n - k)
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    den = msb + (n0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else float("nan")


def component_share(true: np.ndarray, donors: np.ndarray) -> dict:
    """The shares of a score's (validation) variance between and within donors, for the not-estimable rule."""
    true = np.asarray(true, float)
    tot = ((true - true.mean()) ** 2).sum()
    w = (within(true, donors) ** 2).sum()
    return {"between": float((tot - w) / tot) if tot > 0 else float("nan"), "within": float(w / tot) if tot > 0 else float("nan")}


def partial_coefs(score_train: np.ndarray, comp_train: np.ndarray) -> np.ndarray:
    """Least-squares coefficients (intercept first) of a true training score on PBMC composition."""
    X = np.column_stack([np.ones(len(comp_train)), comp_train])
    return np.linalg.lstsq(X, np.asarray(score_train, float), rcond=None)[0]


def residualize(score: np.ndarray, comp: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """A score minus its composition fit (training coefficients applied as they are)."""
    return np.asarray(score, float) - np.column_stack([np.ones(len(comp)), comp]) @ coefs


# ---------------------------------------------------------------- Part B: the vaccine unit
def donor_years(timepoints: np.ndarray, donors: np.ndarray, years=FLU_YEARS) -> pd.DataFrame:
    """One row per donor-year with both Day 0 and Day 7 of a flu-vaccine year: row indices of the two draws."""
    tp, donors = np.asarray(timepoints).astype(str), np.asarray(donors)
    rows = []
    for d in pd.unique(donors):
        for y in years:
            i0 = np.flatnonzero((donors == d) & (tp == f"{y} Day 0"))
            i7 = np.flatnonzero((donors == d) & (tp == f"{y} Day 7"))
            assert len(i0) <= 1 and len(i7) <= 1, "one sample per donor and timepoint"
            if len(i0) and len(i7):
                rows.append({"donor": d, "year": y, "day0": int(i0[0]), "day7": int(i7[0])})
    return pd.DataFrame(rows)


def unit_boot(units: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> list[np.ndarray]:
    """Donor-bootstrap resamples of donor-year units: all of a resampled donor's years come along."""
    groups = [np.flatnonzero(units["donor"].to_numpy() == d) for d in pd.unique(units["donor"])]
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(n_boot)]


def strat_boot(donors: np.ndarray, strata: dict, n_boot: int = 1000, seed: int = 0) -> list[np.ndarray]:
    """Donor-bootstrap row resamples stratified by a donor-level trait ({donor: stratum}): each stratum's donors are
    resampled within it, so no resample loses a class."""
    donors = np.asarray(donors)
    labels = pd.unique(donors)
    by = {}
    for d in labels:
        by.setdefault(strata[d], []).append(np.flatnonzero(donors == d))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        parts = []
        for groups in by.values():
            parts += [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        out.append(np.concatenate(parts))
    return out


# ---------------------------------------------------------------- Part D: moves
def move_excess(G: np.ndarray, s: np.ndarray, y: np.ndarray, y_other: np.ndarray) -> np.ndarray:
    """Per gene, r(move, needed correction) minus r(move, the same with another donor's real profile): move = G - s,
    needed = y - s, across samples."""
    move = np.asarray(G, float) - s
    return colwise_r(move, np.asarray(y, float) - s) - colwise_r(move, np.asarray(y_other, float) - s)


def other_donor(donors: np.ndarray, seed: int = 0) -> np.ndarray:
    """For each sample, a sample of another donor (seeded): the 'another donor's real profile' of the move excess."""
    donors = np.asarray(donors)
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(np.flatnonzero(donors != donors[i])) for i in range(len(donors))])


def partial_move_r(G: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per gene, across samples, the partial r of the move (G - s) with the true partner y, controlling for the start s:
    both are regressed on s (with an intercept, per gene) and the residuals correlated. A move that is an affine
    function of the start has a zero residual (r undefined, NaN): the information-free null by construction."""
    s, y = np.asarray(s, float), np.asarray(y, float)
    move = np.asarray(G, float) - s
    sc = s - s.mean(0)
    ss = (sc * sc).sum(0)
    def resid(v):
        vc = v - v.mean(0)
        b = np.where(ss > 1e-12, (sc * vc).sum(0) / np.where(ss > 1e-12, ss, 1), 0.0)
        r = vc - b * sc
        return np.where(np.abs(r) < 1e-9 * (np.abs(vc).max(0) + 1e-12), 0.0, r)
    return colwise_r(resid(move), resid(y))
