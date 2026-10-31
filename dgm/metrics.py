"""Evaluation of a translation: the fixed yardstick every phase scores with.

A translation maps profiles of one domain (the source) into the other (the target): S->B
turns single-cell pseudobulk into bulk-like profiles, B->S the reverse. An ``Evaluator`` is
fit once per direction on the **training donors only**, and then scores any translation of
held-out samples against their real paired target profiles. Everything it fits (the
centering means, the PCA, the composition probe, the age classifier) is fixed at fit time,
so every baseline and every model is scored by exactly the same yardstick.

Metric groups (plan, Phase 1a); every name in ``Evaluator.score``'s output is listed here:

  accuracy        r_raw (median per-sample Pearson), r_centered (median per-sample Pearson
                  after removing the target domain's training mean per gene; the honest one,
                  since raw correlation is dominated by the shared mean profile), r2_centered
                  (median per-sample 1 - SSE/SST, with SST around the target's training mean:
                  0 = no better than predicting the training mean; unlike a correlation it
                  drops when outputs are rescaled or noised), rho_raw
                  (median per-sample Spearman), gene_mean_r / gene_sd_r (agreement of per-gene
                  means / SDs across samples), mse, mae
  pairing         rank_median (rank of the true partner among all real targets, by cosine of
                  centered profiles; 1 = best), rank_norm (rank rescaled to 0 best .. 1 worst,
                  median), top1, top5 (fraction of samples whose partner ranks 1st / in top 5),
                  within_top1 (among samples whose donor has 2+ draws in the scored set: the
                  fraction whose partner ranks 1st among that donor's draws only; this separates
                  draw-level signal from donor identity), within_chance (its chance level, the
                  mean of 1 / draws per donor over the same samples)
  distribution    c2st_auroc (classifier two-sample test: donor-grouped cross-validated
                  logistic regression telling translations from real targets on the top
                  DIST_PCS target PCs and their squares, so it sees a loss of spread as well as
                  a shift; 0.5 = indistinguishable; see ``c2st``), mmd (unbiased RBF-kernel
                  MMD^2 on the same top DIST_PCS PCs, median-heuristic bandwidth), silhouette
                  and pc1_eta2 (separation of translated vs real on their joint PCA)
  biology         comp_r (mean over classes of the Pearson between the composition probe's
                  prediction from the translation and the measured composition), comp_r2 (the
                  same as R^2, 1 - SSE/SST per class across samples, mean over classes; scale-
                  sensitive like r2_centered. Its SST is around the held-out samples' own mean
                  composition, slightly stricter than r2_centered's training-mean baseline:
                  predicting the training mean composition scores a little below 0), comp_mae,
                  marker_rho (mean Spearman of one marker per lineage vs that lineage's
                  fraction), gran_gap (S->B only: mean over granulocyte genes of translated
                  minus real target expression; 0 = matches whole blood)
  downstream      age_auroc (donor-level AUROC of a young/older classifier trained on real
                  target-domain training profiles, applied to the translations)
  non-collapse    diversity (mean pairwise distance of translations / of real targets; 1 =
                  as diverse as real), nn_ratio (median distance to the nearest training
                  target, translations / real targets; << 1 suggests copying training data)
  selection       composite (the checkpoint-selection score; see ``composite``)

``c2st_null`` gives the C2ST's null distribution for a given translation (labels swapped at
random within each pair), to judge whether a C2ST value is distinguishable from 0.5.

``Evaluator.reference`` holds the probes' scores on the real held-out targets: what a perfect
translation would get. They are references, not ceilings: a translation is built from the
source, and the source can carry more of a signal than the real target does (S->B
translations from single-cell profiles recover composition better than real bulk does).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# One marker per lineage: the first of each list present in the panel is used. CD3E/CD3D are
# not in the panel (phase 0), so T cells fall back to CD2/LCK.
MARKERS = {"T cell": ["CD3E", "CD3D", "CD2", "LCK", "IL7R"], "B cell": ["MS4A1", "CD79A", "CD19"],
           "Monocyte": ["CD14", "FCN1", "LYZ"], "NK cell": ["KLRF1", "SH2D1B", "NCAM1"], "DC": ["FCER1A", "CLEC10A", "CD1C"]}
GRANULOCYTE = ["FCGR3B", "CSF3R", "CXCR2", "CXCR1", "FPR1", "CMTM2", "MME"]
N_PCS = 50       # target PCA fit on training targets (age classifier input)
DIST_PCS = 10    # the leading PCs the distribution tests use. With quadratic features on all 50
                 # PCs, real training-donor profiles vs real val profiles scored 0.96 (S->B; ~0.73
                 # B->S): trailing PCs carry PCA inflation on training samples plus differences
                 # between donor sets. On the top 10 that check gives ~0.5-0.6; the proper null for
                 # the paired test (c2st_null) is centred at 0.50, SD ~0.04 (Phase 1 audit).
SEED = 0


# ---------------------------------------------------------------- per-sample building blocks
def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row of a with the same row of b."""
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.divide((a * b).sum(1), den, out=np.zeros(len(a)), where=den > 0)


def rowwise_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return rowwise_pearson(np.apply_along_axis(rankdata, 1, a), np.apply_along_axis(rankdata, 1, b))


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return an @ bn.T


def partner_ranks(sim: np.ndarray) -> np.ndarray:
    """For a square similarity matrix (translation i vs real target j, true partner j = i):
    the rank of the true partner in each row, 1 = most similar. Ties count against it."""
    return (sim >= np.diag(sim)[:, None]).sum(1)


def within_donor_ranks(sim: np.ndarray, donors: np.ndarray) -> np.ndarray:
    """Rank of the true partner (j = i) among the same donor's draws only, 1 = most similar.
    NaN for samples whose donor has a single draw in the set. Ties count against it."""
    donors = np.asarray(donors)
    out = np.full(len(donors), np.nan)
    for d in np.unique(donors):
        idx = np.flatnonzero(donors == d)
        if len(idx) > 1:
            sub = sim[np.ix_(idx, idx)]
            out[idx] = (sub >= np.diag(sub)[:, None]).sum(1)
    return out


def mmd_rbf(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """Unbiased MMD^2 with an RBF kernel exp(-|a-b|^2 / (2 bandwidth^2))."""
    def k(a, b):
        d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
        return np.exp(-np.maximum(d, 0) / (2 * bandwidth ** 2))
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    n, m = len(x), len(y)
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())


def eta2(values: np.ndarray, labels: np.ndarray) -> float:
    """One-way eta^2: share of the variance of *values* explained by *labels*."""
    grand = values.mean()
    total = ((values - grand) ** 2).sum()
    between = sum((labels == l).sum() * (values[labels == l].mean() - grand) ** 2 for l in np.unique(labels))
    return float(between / total) if total > 0 else 0.0


def donor_auroc(scores: np.ndarray, labels: np.ndarray, donors: np.ndarray) -> float:
    """AUROC after averaging scores per donor (one label per donor), so a donor's many draws
    count once."""
    t = pd.DataFrame({"s": scores, "y": labels, "d": donors}).groupby("d").agg(s=("s", "mean"), y=("y", "first"))
    return float(roc_auc_score(t["y"], t["s"])) if t["y"].nunique() == 2 else float("nan")


def donor_bootstrap(values: np.ndarray, donors: np.ndarray, stat=np.median, n_boot: int = 1000,
                    seed: int = SEED, level: float = 0.95) -> tuple[float, float, float]:
    """Point estimate and percentile CI of stat(values), resampling donors with replacement
    (all of a resampled donor's samples come along)."""
    values, donors = np.asarray(values, float), np.asarray(donors)
    groups = [np.flatnonzero(donors == d) for d in pd.unique(donors)]
    rng = np.random.default_rng(seed)
    boots = np.array([stat(values[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])])
                      for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(stat(values)), float(lo), float(hi)


def c2st(z_hat: np.ndarray, z_real: np.ndarray, donors: np.ndarray) -> float:
    """Classifier two-sample test AUROC between translations and real targets (row i of each
    is the same draw). Features are the given PCs and their squares: a linear classifier on the
    PCs alone only sees mean shifts, and missed outputs collapsed toward the mean (real profiles
    shrunk 50% toward the mean scored 0.61/0.52; with squares, 0.96/0.95). Logistic
    regression, donor-grouped 5-fold CV, so a donor's translations
    and real profiles always sit in the same fold. Both classes contain the same donors, so
    differences between donor groups (age mix, few donors) can't drive the score; only how
    translations differ from real profiles can. Caveat: translations that copy *other held-out*
    real profiles would bias it below 0.5 (the copies' originals sit in training folds with the
    opposite label). A model never sees held-out targets, so it can't do that; a test that
    shuffles real profiles can. Splitting donors into disjoint halves avoids that case but was
    too noisy with 14 validation donors (identical distributions scored 0.22-0.68)."""
    z = np.vstack([z_hat, z_real])
    z = np.hstack([z, z ** 2])
    lab = np.r_[np.ones(len(z_hat)), np.zeros(len(z_real))]
    grp = np.r_[donors, donors]
    proba = np.zeros(len(z))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    for tr, te in GroupKFold(n_splits=min(5, len(np.unique(donors)))).split(z, lab, grp):
        proba[te] = clf.fit(z[tr], lab[tr]).predict_proba(z[te])[:, 1]
    return float(roc_auc_score(lab, proba))


def c2st_null(z_hat: np.ndarray, z_real: np.ndarray, donors: np.ndarray, n_perm: int = 100,
              seed: int = SEED) -> tuple[float, float]:
    """Mean and SD of the C2ST AUROC when, within each pair, which profile is "translated" and
    which "real" is swapped at random: the null for this paired test. A C2ST value inside
    mean +- 2 SD can't be told from chance."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        swap = rng.random(len(z_hat)) < 0.5
        a = np.where(swap[:, None], z_real, z_hat)
        b = np.where(swap[:, None], z_hat, z_real)
        vals.append(c2st(a, b, donors))
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def composite(m: dict) -> float:
    """Checkpoint-selection score for one direction (approved 2026-10-05; see the Phase 1
    notebook for the reasoning). Equal-weight mean of three parts, each in [0, 1], 1 = ideal:
      accuracy      max(r2_centered, 0)               (sample-specific accuracy beyond the mean)
      distribution  1 - 2 |c2st_auroc - 0.5|          (1 = indistinguishable from real targets)
      biology       max(comp_r2, 0)                   (composition recoverable from the output)
    Accuracy and biology are R^2 (error-based), not correlations: correlations ignore scale, so
    inflating a collapsed output's spread raised the score without adding information (ridge x1.5
    spread: 0.50 -> 0.65 in S->B). With R^2, inflation and noise cost accuracy. R^2 below 0 (worse
    than predicting the training mean) counts as 0, so one bad part can't swamp the others.
    Deliberate calibration of spread still helps the distribution part; that's why the baselines
    include a spread-calibrated ridge, which a model must beat."""
    return float((max(m["r2_centered"], 0.0) + (1 - 2 * abs(m["c2st_auroc"] - 0.5)) + max(m["comp_r2"], 0.0)) / 3)


# ---------------------------------------------------------------- the evaluator
@dataclass
class Evaluator:
    """Fit on the training donors of one direction; score translations of held-out samples.

    target_train   real target-domain profiles of the training samples
    comp_train     their measured composition (rows on the simplex)
    age_train      their age group (0/1)
    donors_train   their donors (for grouped cross-validation)
    genes          panel gene names (column order)
    direction      "S->B" or "B->S" (the granulocyte gap is reported for S->B only)
    """
    target_train: np.ndarray
    comp_train: np.ndarray
    age_train: np.ndarray
    donors_train: np.ndarray
    genes: list
    direction: str
    n_pcs: int = N_PCS
    reference: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        assert self.direction in ("S->B", "B->S")
        self.genes = list(self.genes)
        self.mean_ = self.target_train.mean(0)
        k = min(self.n_pcs, self.target_train.shape[0] - 1, self.target_train.shape[1])
        self.pca_ = PCA(n_components=k, random_state=SEED).fit(self.target_train)
        folds = GroupKFold(n_splits=5)
        # Composition probe: ridge from target-domain profiles to composition, alpha chosen by
        # donor-grouped CV on the training donors. Independent of any task net a model trains against.
        alphas = 10.0 ** np.arange(-3, 5)
        cv_err = [np.mean([np.abs(Ridge(alpha=a).fit(self.target_train[tr], self.comp_train[tr]).predict(self.target_train[te])
                                   - self.comp_train[te]).mean()
                           for tr, te in folds.split(self.target_train, groups=self.donors_train)]) for a in alphas]
        self.comp_alpha_ = float(alphas[int(np.argmin(cv_err))])
        self.comp_probe_ = Ridge(alpha=self.comp_alpha_).fit(self.target_train, self.comp_train)
        # Age classifier: L2 logistic regression on standardized target PCs.
        self.age_clf_ = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(
            self.pca_.transform(self.target_train), self.age_train)
        self.markers_ = {ct: next((g for g in cands if g in self.genes), None) for ct, cands in MARKERS.items()}
        self.gran_idx_ = [self.genes.index(g) for g in GRANULOCYTE if g in self.genes] if self.direction == "S->B" else []

    def _comp_pred(self, x):
        return self.comp_probe_.predict(x)

    def score(self, y_hat: np.ndarray, y: np.ndarray, comp: np.ndarray, age: np.ndarray, donors: np.ndarray,
              classes: list) -> tuple[dict, pd.DataFrame]:
        """Metrics for translations y_hat of held-out samples whose real targets are y.
        Returns (metrics dict, per-sample table). Row i of y_hat, y, comp, age, donors is one draw."""
        y_hat, y = np.asarray(y_hat, np.float64), np.asarray(y, np.float64)
        assert y_hat.shape == y.shape and np.isfinite(y_hat).all(), "translations must match targets and be finite"
        m = {}
        # accuracy
        r_raw = rowwise_pearson(y_hat, y)
        r_cen = rowwise_pearson(y_hat - self.mean_, y - self.mean_)
        r2 = 1 - ((y_hat - y) ** 2).sum(1) / ((y - self.mean_) ** 2).sum(1)
        m.update(r_raw=np.median(r_raw), r_centered=np.median(r_cen), r2_centered=np.median(r2),
                 rho_raw=np.median(rowwise_spearman(y_hat, y)),
                 gene_mean_r=np.corrcoef(y_hat.mean(0), y.mean(0))[0, 1],
                 gene_sd_r=np.corrcoef(y_hat.std(0), y.std(0))[0, 1] if y_hat.std(0).std() > 0 else 0.0,
                 mse=((y_hat - y) ** 2).mean(), mae=np.abs(y_hat - y).mean())
        # pairing
        sim = cosine_matrix(y_hat - self.mean_, y - self.mean_)
        ranks = partner_ranks(sim)
        n = len(y)
        m.update(rank_median=np.median(ranks), rank_norm=np.median((ranks - 1) / (n - 1)),
                 top1=np.mean(ranks == 1), top5=np.mean(ranks <= 5))
        wranks = within_donor_ranks(sim, donors)
        multi = ~np.isnan(wranks)
        sizes = pd.Series(donors).map(pd.Series(donors).value_counts()).to_numpy()
        m.update(within_top1=np.mean(wranks[multi] == 1) if multi.any() else float("nan"),
                 within_chance=np.mean(1 / sizes[multi]) if multi.any() else float("nan"))
        # distribution (target PCA, fit on training targets)
        zh, zy = self.pca_.transform(y_hat), self.pca_.transform(y)
        dh, dy = zh[:, :DIST_PCS], zy[:, :DIST_PCS]
        m["c2st_auroc"] = c2st(dh, dy, donors)
        lab = np.r_[np.ones(n), np.zeros(n)]
        dists = np.sqrt(np.maximum(((dy[:, None, :] - dy[None, :, :]) ** 2).sum(-1), 0))
        m["mmd"] = mmd_rbf(dh, dy, bandwidth=float(np.median(dists[np.triu_indices(n, 1)])))
        joint = PCA(n_components=min(10, 2 * n - 1), random_state=SEED).fit_transform(np.vstack([y_hat, y]))
        m["silhouette"] = silhouette_score(joint, lab) if np.ptp(joint) > 0 else 0.0
        m["pc1_eta2"] = eta2(joint[:, 0], lab)
        # biology
        comp_hat = self._comp_pred(y_hat)
        per_class = [np.corrcoef(comp_hat[:, j], comp[:, j])[0, 1] if comp_hat[:, j].std() > 0 else 0.0 for j in range(comp.shape[1])]
        m["comp_r"] = float(np.mean(per_class))
        m["comp_r2"] = float(np.mean([1 - ((comp_hat[:, j] - comp[:, j]) ** 2).sum() / ((comp[:, j] - comp[:, j].mean()) ** 2).sum()
                                      for j in range(comp.shape[1])]))
        m["comp_mae"] = float(np.abs(comp_hat - comp).mean())
        rhos = [pd.Series(y_hat[:, self.genes.index(g)]).corr(pd.Series(comp[:, classes.index(ct)]), method="spearman")
                for ct, g in self.markers_.items() if g is not None and ct in classes]
        m["marker_rho"] = float(np.mean(rhos)) if rhos else float("nan")
        m["gran_gap"] = float((y_hat[:, self.gran_idx_].mean(0) - y[:, self.gran_idx_].mean(0)).mean()) if self.gran_idx_ else float("nan")
        # downstream
        m["age_auroc"] = donor_auroc(self.age_clf_.decision_function(zh), age, donors)
        # non-collapse
        def mean_pairwise(a):
            sq = (a * a).sum(1)
            d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * a @ a.T, 0))
            return d[np.triu_indices(len(a), 1)].mean()
        m["diversity"] = mean_pairwise(y_hat) / mean_pairwise(y)
        def nn_dist(a):
            d = (a * a).sum(1)[:, None] + (self.target_train ** 2).sum(1)[None, :] - 2 * a @ self.target_train.T
            return np.sqrt(np.maximum(d, 0)).min(1)
        m["nn_ratio"] = float(np.median(nn_dist(y_hat)) / np.median(nn_dist(y)))
        m = {k: float(v) for k, v in m.items()}
        m["composite"] = composite(m)
        per_sample = pd.DataFrame({"donor": donors, "r_raw": r_raw, "r_centered": r_cen, "r2_centered": r2, "rank": ranks,
                                   "rank_within": wranks})
        return m, per_sample

    def fit_reference(self, y: np.ndarray, comp: np.ndarray, age: np.ndarray, donors: np.ndarray, classes: list) -> dict:
        """The probes on real held-out targets: what a perfect translation would get from them
        (a reference, not a ceiling; see the module docstring)."""
        comp_hat = self._comp_pred(y)
        self.reference = {
            "comp_r": float(np.mean([np.corrcoef(comp_hat[:, j], comp[:, j])[0, 1] for j in range(comp.shape[1])])),
            "comp_r2": float(np.mean([1 - ((comp_hat[:, j] - comp[:, j]) ** 2).sum() / ((comp[:, j] - comp[:, j].mean()) ** 2).sum()
                                      for j in range(comp.shape[1])])),
            "comp_mae": float(np.abs(comp_hat - comp).mean()),
            "age_auroc": donor_auroc(self.age_clf_.decision_function(self.pca_.transform(y)), age, donors),
        }
        rhos = [pd.Series(y[:, self.genes.index(g)]).corr(pd.Series(comp[:, classes.index(ct)]), method="spearman")
                for ct, g in self.markers_.items() if g is not None and ct in classes]
        self.reference["marker_rho"] = float(np.mean(rhos)) if rhos else float("nan")
        return self.reference
