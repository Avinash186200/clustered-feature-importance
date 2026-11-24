"""Feature importance: plain and clustered.

Every method returns one score per *unit*: a unit is a feature for the plain methods
and a cluster of features for the clustered ones.

* MDI: total split gain per feature in the fitted LightGBM model, normalised to sum
  to 1. This is the boosted analogue of mean decrease in impurity. Clustered MDI sums
  the gains of a cluster's members.
* MDA (permutation): drop in validation R^2 when a feature's column is shuffled.
  Clustered MDA shuffles every column of a cluster with one shared row permutation,
  so the cluster's internal structure is kept but its link to y is broken.
* SHAP: mean |TreeSHAP value| on validation rows, from LightGBM's ``pred_contrib``
  (no extra dependency). SHAP values are additive, so a cluster's SHAP value per row
  is the sum of its members'; clustered SHAP is the mean absolute of that sum.
* Drop-column: drop in validation R^2 when the model is refitted without the feature.

Permutations use one random generator per (seed, unit, repeat), so a parallel run
gives the same numbers as a serial one, bit for bit (tested).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from joblib import Parallel, delayed

from .model import ModelConfig, fit, predict, r2


def mdi(model, n_features: int) -> np.ndarray:
    g = np.asarray(model.feature_importance(importance_type="gain"), dtype=float)
    assert g.shape[0] == n_features
    tot = g.sum()
    return g / tot if tot > 0 else g


def clustered_mdi(model, clusters: Sequence[np.ndarray], n_features: int) -> np.ndarray:
    g = mdi(model, n_features)
    return np.array([g[c].sum() for c in clusters])


def _perm_chunk(model, X, y, base, units, unit_ids, n_repeats, seed):
    out = np.empty(len(units))
    Xp = X.copy()
    for i, (uid, cols) in enumerate(zip(unit_ids, units)):
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            rng = np.random.default_rng([seed, uid, r])
            perm = rng.permutation(X.shape[0])
            Xp[:, cols] = X[perm[:, None], cols]
            drops[r] = base - r2(y, predict(model, Xp))
            Xp[:, cols] = X[:, cols]
        out[i] = drops.mean()
    return out


def permutation_importance(model, X: np.ndarray, y: np.ndarray, units: Sequence[np.ndarray],
                           n_repeats: int = 5, seed: int = 0, n_jobs: int = 1,
                           backend: str = "loky") -> np.ndarray:
    """Mean drop in R^2 when each unit (a list of columns) is shuffled jointly."""
    base = r2(y, predict(model, X))
    units = [np.atleast_1d(np.asarray(u)) for u in units]
    ids = list(range(len(units)))
    if n_jobs == 1:
        return _perm_chunk(model, X, y, base, units, ids, n_repeats, seed)
    n_chunks = min(len(units), n_jobs * 4)
    bounds = np.linspace(0, len(units), n_chunks + 1).astype(int)
    parts = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_perm_chunk)(model, X, y, base, units[a:b], ids[a:b], n_repeats, seed)
        for a, b in zip(bounds[:-1], bounds[1:]) if b > a)
    return np.concatenate(parts)


def mda(model, X, y, n_repeats=5, seed=0, n_jobs=1, backend="loky") -> np.ndarray:
    return permutation_importance(model, X, y, [[j] for j in range(X.shape[1])],
                                  n_repeats=n_repeats, seed=seed, n_jobs=n_jobs, backend=backend)


def clustered_mda(model, X, y, clusters, n_repeats=5, seed=0, n_jobs=1, backend="loky") -> np.ndarray:
    return permutation_importance(model, X, y, clusters, n_repeats=n_repeats,
                                  seed=seed + 7919, n_jobs=n_jobs, backend=backend)


def shap_values(model, X: np.ndarray) -> np.ndarray:
    """TreeSHAP values (n_rows, n_features); the bias column is dropped."""
    contrib = model.predict(X, pred_contrib=True, num_threads=1)
    return np.asarray(contrib)[:, :-1]


def shap_importance(model, X) -> np.ndarray:
    return np.abs(shap_values(model, X)).mean(axis=0)


def clustered_shap(model, X, clusters) -> np.ndarray:
    phi = shap_values(model, X)
    return np.array([np.abs(phi[:, c].sum(axis=1)).mean() for c in clusters])


def _drop_one(Xtr, ytr, Xva, yva, cfg, seed, j):
    keep = np.delete(np.arange(Xtr.shape[1]), j)
    m = fit(Xtr[:, keep], ytr, cfg, seed)
    return r2(yva, predict(m, Xva[:, keep]))


def drop_column(Xtr, ytr, Xva, yva, cfg: ModelConfig, seed: int, base_r2: float,
                n_jobs: int = 1) -> np.ndarray:
    js = range(Xtr.shape[1])
    if n_jobs == 1:
        scores = [_drop_one(Xtr, ytr, Xva, yva, cfg, seed, j) for j in js]
    else:
        scores = Parallel(n_jobs=n_jobs)(delayed(_drop_one)(Xtr, ytr, Xva, yva, cfg, seed, j)
                                         for j in js)
    return base_r2 - np.asarray(scores)
