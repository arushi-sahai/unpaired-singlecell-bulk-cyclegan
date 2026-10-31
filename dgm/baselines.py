"""Baseline translators, the reference points every model is compared with (plan, Phase 1b).

Each translator maps source-domain profiles into the target domain (log1p CP10k on the
panel). ``fit`` sees **training samples only**; ``translate`` maps held-out source profiles.
All outputs are clipped at 0, like the generators will be, since log1p(CP10k) is non-negative.

  Identity       no correction (the floor)
  MeanShift      add the per-gene difference of the training means
  ComBat         empirical-Bayes location/scale adjustment with the target as the reference
                 batch (Johnson, Li & Rabinovic 2007, reference-batch variant): the target's
                 values define the scale, the source is standardized onto it
  CORAL          recolor the source covariance to the target covariance (Sun, Feng & Saenko
                 2016), with shrinkage since n < genes: by default 1.0 x the mean variance
                 (relative); sc2bulk-working's frozen choice, absolute 1e-3, is available with
                 relative=False and reported as a sensitivity row
  Harmony        Harmony (Korsunsky et al. 2019) on a joint PCA, mapped back to genes through
                 the PCA. harmonypy 2.0.2 has no reference-batch option, so both batches move
                 into a shared space; the target batch's mean correction is then undone, which
                 puts the corrected source back in target space (an approximation: the exact
                 per-cell correction of a target profile isn't defined for a source profile).
                 Transductive: it needs the held-out source profiles at fit time (never the
                 held-out targets), so it has ``fit_translate`` instead of fit and translate.
                 Its output is limited to what the PCA can represent. On this data Harmony
                 makes no correction at all: every one of its clusters holds a single domain
                 (the domains don't overlap), and Harmony only corrects within mixed clusters
                 (Phase 1 notebook, 2026-10-05).
  Ridge          supervised paired regression (dual-form ridge), alpha by donor-grouped CV
  RidgeSpread    ridge with its spread stretched by one factor, chosen on the training donors
                 by cross-fitting to maximize the selection score: the most that rescaling
                 alone (no new information) can buy. A model has to beat this, not plain ridge.
  RidgePCCalibration  ridge with the spread of each of its top 10 target principal components
                 matched, one factor per component, to the training targets' spread, using
                 cross-fitted (out-of-fold) predictions; like RidgeSpread, it adds no new
                 information, only per-component calibration
  PairedMLP      supervised paired regression with the generator's architecture (residual MLP,
                 5000 -> 512 -> 512 -> 5000, LayerNorm, LeakyReLU), L1 loss, early stopping on
                 a donor-held-out part of the training set

Unpaired baselines (MeanShift, ComBat, CORAL, Harmony) never use which source sample goes
with which target; the paired ones (Ridge, PairedMLP) do, as the GAN will.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

SEED = 0


def _clip(x):
    return np.maximum(np.asarray(x, dtype=np.float64), 0.0)


class Identity:
    name, paired = "identity", False

    def fit(self, src, tgt, donors=None):
        return self

    def translate(self, src):
        return _clip(src)


class MeanShift:
    name, paired = "mean shift", False

    def fit(self, src, tgt, donors=None):
        self.shift_ = tgt.mean(0) - src.mean(0)
        return self

    def translate(self, src):
        return _clip(src + self.shift_)


class ComBat:
    """Reference-batch ComBat with no covariates. The target is the reference: its per-gene
    mean and SD are kept, and the source's per-gene location and scale shifts (gamma, delta^2),
    shrunk toward their across-gene priors by empirical Bayes, are removed."""
    name, paired = "ComBat", False

    def __init__(self, max_iter: int = 100, tol: float = 1e-6, eb: bool = True):
        self.max_iter, self.tol, self.eb = max_iter, tol, eb

    def fit(self, src, tgt, donors=None):
        alpha = tgt.mean(0)
        sigma = tgt.std(0, ddof=1)
        sigma[sigma == 0] = 1.0
        z = (src - alpha) / sigma                      # source standardized on the reference's scale
        n = z.shape[0]
        g_hat, d_hat = z.mean(0), z.var(0, ddof=1)
        d_hat[d_hat == 0] = 1e-8
        if self.eb:
            g_bar, t2 = g_hat.mean(), g_hat.var(ddof=1)
            m, s2 = d_hat.mean(), d_hat.var(ddof=1)
            a = (2 * s2 + m ** 2) / s2                 # inverse-gamma prior on delta^2, method of moments
            b = (m * s2 + m ** 3) / s2
            g_star, d_star = g_hat.copy(), d_hat.copy()
            for _ in range(self.max_iter):
                g_new = (n * t2 * g_hat + d_star * g_bar) / (n * t2 + d_star)
                d_new = (b + 0.5 * ((z - g_new) ** 2).sum(0)) / (n / 2 + a - 1)
                done = max(np.abs(g_new - g_star).max() / np.abs(g_star).max(), np.abs(d_new - d_star).max() / d_star.max()) < self.tol
                g_star, d_star = g_new, d_new
                if done:
                    break
        else:
            g_star, d_star = g_hat, d_hat
        self.alpha_, self.sigma_, self.gamma_, self.delta_ = alpha, sigma, g_star, np.sqrt(d_star)
        return self

    def translate(self, src):
        z = (src - self.alpha_) / self.sigma_
        return _clip(self.sigma_ * (z - self.gamma_) / self.delta_ + self.alpha_)


def _sym_power(mat, power):
    w, v = np.linalg.eigh(mat)
    return (v * np.clip(w, 1e-12, None) ** power) @ v.T


class CORAL:
    """Source covariance recolored to the target's: x -> (x - mu_s) C_s^-1/2 C_t^1/2 + mu_t.
    Covariances get a ridge on the diagonal (n < genes): ``shrinkage`` x mean variance when
    ``relative`` (default), else ``shrinkage`` itself."""
    paired = False

    def __init__(self, shrinkage: float = 1.0, relative: bool = True):
        self.shrinkage, self.relative = shrinkage, relative
        self.name = "CORAL" if (shrinkage, relative) == (1.0, True) else f"CORAL ({'rel' if relative else 'abs'} {shrinkage:g})"

    def fit(self, src, tgt, donors=None):
        self.mu_s_, self.mu_t_ = src.mean(0), tgt.mean(0)
        def cov(x):
            c = np.cov(x - x.mean(0), rowvar=False)
            lam = self.shrinkage * np.mean(np.diag(c)) if self.relative else self.shrinkage
            return c + lam * np.eye(c.shape[0])
        self.A_ = _sym_power(cov(src), -0.5) @ _sym_power(cov(tgt), 0.5)
        return self

    def translate(self, src):
        return _clip((src - self.mu_s_) @ self.A_ + self.mu_t_)


class Harmony:
    """Joint PCA of training source + training target + held-out source, Harmony on the
    batch label, the target batch's mean correction undone, back to genes via the PCA."""
    name, paired = "Harmony", False

    def __init__(self, n_pcs: int = 50):
        self.n_pcs = n_pcs

    def fit_translate(self, src_train, tgt_train, src_eval):
        import harmonypy
        import pandas as pd
        X = np.vstack([src_train, tgt_train, src_eval])
        pca = PCA(n_components=self.n_pcs, random_state=SEED).fit(np.vstack([src_train, tgt_train]))
        Z = pca.transform(X)
        batch = np.r_[np.full(len(src_train), "source"), np.full(len(tgt_train), "target"), np.full(len(src_eval), "source")]
        ho = harmonypy.run_harmony(Z, pd.DataFrame({"batch": batch}), "batch", random_state=SEED, verbose=False)
        Zc = np.asarray(ho.Z_corr)
        Zc = Zc.T if Zc.shape[0] != len(X) else Zc
        t = slice(len(src_train), len(src_train) + len(tgt_train))
        back_to_target = (Z[t] - Zc[t]).mean(0)          # undo the target batch's average correction
        self.target_shift_norm_ = float(np.linalg.norm(back_to_target))
        # Diagnostics: does Harmony find clusters shared by both domains, and does it move anything?
        R = np.asarray(ho.R)
        R = R if R.shape[1] == len(X) else R.T
        hard = R.argmax(0)
        is_src = batch == "source"
        frac = np.array([is_src[hard == k].mean() for k in np.unique(hard)])
        self.diagnostics_ = {
            "clusters": int(len(frac)),
            "mixed_clusters": int(((frac > 0.05) & (frac < 0.95)).sum()),   # >= 5% of each domain
            "centroid_gap_before": float(np.linalg.norm(Z[is_src].mean(0) - Z[~is_src].mean(0))),
            "centroid_gap_after": float(np.linalg.norm(Zc[is_src].mean(0) - Zc[~is_src].mean(0))),
            "max_coordinate_change": float(np.abs(Zc - Z).max()),
        }
        return _clip(pca.inverse_transform(Zc[-len(src_eval):] + back_to_target))


class Ridge:
    """Paired multi-output ridge, solved in dual form (n << genes). Alpha by donor-grouped
    5-fold CV on the training pairs (mean absolute error)."""
    name, paired = "ridge (paired)", True

    def __init__(self, alphas=tuple(10.0 ** np.arange(0, 7))):
        self.alphas = alphas

    @staticmethod
    def _solve(X, Y, alpha):
        mx, my = X.mean(0), Y.mean(0)
        Xc = X - mx
        K = Xc @ Xc.T
        coef = Xc.T @ np.linalg.solve(K + alpha * np.eye(len(K)), Y - my)
        return mx, my, coef

    def fit(self, src, tgt, donors):
        errs = []
        for a in self.alphas:
            e = []
            for tr, te in GroupKFold(n_splits=5).split(src, groups=donors):
                mx, my, W = self._solve(src[tr], tgt[tr], a)
                e.append(np.abs((src[te] - mx) @ W + my - tgt[te]).mean())
            errs.append(np.mean(e))
        self.alpha_ = float(self.alphas[int(np.argmin(errs))])
        self.cv_mae_ = dict(zip(map(float, self.alphas), map(float, errs)))
        self.mx_, self.my_, self.W_ = self._solve(src, tgt, self.alpha_)
        return self

    def translate(self, src):
        return _clip((src - self.mx_) @ self.W_ + self.my_)


class PairedMLP:
    """The generator architecture trained as a plain supervised regressor on the pairs:
    G(x) = clip(x + f(x)), f = 5000 -> 512 -> 512 -> 5000 MLP with LayerNorm and LeakyReLU,
    L1 loss, Adam. Early stopping on a donor-held-out 15% of the training donors."""
    name, paired = "MLP (paired)", True

    def __init__(self, hidden: int = 512, lr: float = 1e-3, weight_decay: float = 1e-4, batch: int = 64,
                 max_epochs: int = 400, patience: int = 30, seed: int = SEED, device: str | None = None):
        self.hidden, self.lr, self.wd, self.batch = hidden, lr, weight_decay, batch
        self.max_epochs, self.patience, self.seed, self.device = max_epochs, patience, seed, device

    def _net(self, g):
        import torch.nn as nn
        h = self.hidden
        return nn.Sequential(nn.Linear(g, h), nn.LayerNorm(h), nn.LeakyReLU(0.2),
                             nn.Linear(h, h), nn.LayerNorm(h), nn.LeakyReLU(0.2), nn.Linear(h, g))

    def fit(self, src, tgt, donors):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        uniq = rng.permutation(np.unique(donors))
        stop_donors = set(uniq[:max(1, round(0.15 * len(uniq)))])
        stop = np.isin(donors, list(stop_donors))
        X = torch.tensor(src, dtype=torch.float32, device=dev)
        Y = torch.tensor(tgt, dtype=torch.float32, device=dev)
        tr, va = np.flatnonzero(~stop), np.flatnonzero(stop)
        net = self._net(src.shape[1]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, wait, self.history_ = np.inf, None, 0, []
        for epoch in range(self.max_epochs):
            net.train()
            order = tr[rng.permutation(len(tr))]          # one shuffle per epoch
            for i in range(0, len(order), self.batch):
                b = torch.tensor(order[i:i + self.batch], device=dev)
                loss = (torch.relu(X[b] + net(X[b])) - Y[b]).abs().mean()
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = float((torch.relu(X[va] + net(X[va])) - Y[va]).abs().mean())
            self.history_.append(v)
            if v < best - 1e-5:
                best, wait = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net_, self.dev_, self.best_epoch_, self.stop_mae_ = net.eval(), dev, int(np.argmin(self.history_)), float(best)
        return self

    def translate(self, src):
        import torch
        with torch.no_grad():
            X = torch.tensor(src, dtype=torch.float32, device=self.dev_)
            return torch.relu(X + self.net_(X)).cpu().numpy().astype(np.float64)


class RidgeSpread(Ridge):
    """Ridge with its spread stretched by the single factor that maximizes the selection score
    on the training donors: the most that rescaling alone, with no new information, can buy.
    Regression shrinks toward the mean, so stretching buys realism at some cost in accuracy;
    this baseline makes that trade explicitly, so a model has to beat it, not plain ridge.

    The factor is chosen by cross-fitting (donor-grouped 5-fold, at the chosen alpha). In each
    fold, ridge is fit on the other folds, its held-out predictions are stretched about the
    training-part mean, and they're scored by an Evaluator fit on the training part only. The
    per-fold scores are averaged per factor and the best factor wins. Scoring with an Evaluator
    fit on the targets being scored biased the factor upward: its PCA and probe are in-sample,
    and training profiles have inflated spread along their own PCs (Phase 1 audit). No held-out
    data is used. Translations are stretched about the training targets' mean."""
    name, paired = "ridge + spread", True

    def __init__(self, factors=tuple(np.round(np.arange(1.0, 2.51, 0.1), 2)), **kw):
        super().__init__(**kw)
        self.factors = factors

    def fit(self, src, tgt, donors, comp=None, age=None, genes=None, direction=None):
        from dgm.metrics import Evaluator
        super().fit(src, tgt, donors)
        assert comp is not None and age is not None and genes is not None and direction is not None, \
            "RidgeSpread chooses its factor with the selection score, so it needs the training composition, age, genes and direction"
        donors = np.asarray(donors)
        classes = list(range(comp.shape[1]))
        per_fold = {float(f): [] for f in self.factors}
        for tr, te in GroupKFold(n_splits=5).split(src, groups=donors):
            mx, my, W = self._solve(src[tr], tgt[tr], self.alpha_)
            pred = (src[te] - mx) @ W + my
            ev = Evaluator(tgt[tr], comp[tr], age[tr], donors[tr], genes, direction)   # never sees tgt[te]
            for f in self.factors:
                m, _ = ev.score(_clip(my + f * (pred - my)), tgt[te], comp[te], age[te], donors[te], classes)
                per_fold[float(f)].append(m["composite"])
        self.factor_scores_ = {f: float(np.mean(v)) for f, v in per_fold.items()}
        self.factor_ = max(self.factor_scores_, key=self.factor_scores_.get)
        return self

    def translate(self, src):
        return self.translate_at(src, self.factor_)

    def translate_at(self, src, factor: float):
        """Ridge stretched by a given factor (for tracing the accuracy-realism curve)."""
        raw = (src - self.mx_) @ self.W_ + self.my_
        return _clip(self.my_ + factor * (raw - self.my_))


class RidgePCCalibration(Ridge):
    """Ridge whose predictions are stretched separately along each of the top ``n_pcs``
    principal components of the training targets, by the ratio of the targets' SD to the SD of
    cross-fitted (donor-grouped 5-fold, out-of-fold) ridge predictions along that component.
    Out-of-fold predictions are used because in-sample ridge predictions are nearly exact on
    the training rows and would give ratios near 1. The stretch is about the training-target
    mean; components beyond the top ``n_pcs`` are left as ridge predicts them."""
    name, paired = "ridge + PC calibration", True

    def __init__(self, n_pcs: int = 10, **kw):
        super().__init__(**kw)
        self.n_pcs = n_pcs

    def fit(self, src, tgt, donors):
        super().fit(src, tgt, donors)
        self.pca_ = PCA(n_components=self.n_pcs, random_state=SEED).fit(tgt)
        oof = np.zeros_like(tgt, dtype=np.float64)
        for tr, te in GroupKFold(n_splits=5).split(src, groups=donors):
            mx, my, W = self._solve(src[tr], tgt[tr], self.alpha_)
            oof[te] = (src[te] - mx) @ W + my
        self.ratio_ = self.pca_.transform(tgt).std(0) / np.maximum(self.pca_.transform(oof).std(0), 1e-12)
        return self

    def translate(self, src):
        raw = (src - self.mx_) @ self.W_ + self.my_
        z = self.pca_.transform(raw)                       # scores about the training-target mean
        return _clip(raw + (z * (self.ratio_ - 1)) @ self.pca_.components_)


UNPAIRED = (Identity, MeanShift, ComBat, CORAL)   # fit/translate
TRANSDUCTIVE = (Harmony,)                         # fit_translate
PAIRED = (Ridge, RidgeSpread, RidgePCCalibration, PairedMLP)   # fit/translate, use the pairing
