"""Scoring importance methods against the generator's ground truth."""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

from .clustering import medoid_order
from .data import Truth


def feature_attribution(unit_scores: np.ndarray, clusters: Sequence[np.ndarray] | None,
                        n_features: int) -> np.ndarray:
    """Per-feature scores. A cluster's score is split equally among its members."""
    if clusters is None:
        return np.asarray(unit_scores, dtype=float)
    out = np.zeros(n_features)
    for s, c in zip(unit_scores, clusters):
        out[c] = s / len(c)
    return out


def group_scores(attr: np.ndarray, truth: Truth) -> np.ndarray:
    """Sum of per-feature scores within each true group.

    For the plain methods this uses the true grouping, which the analyst would not
    have; it is the most generous way to read them at group level.
    """
    n_groups = len(truth.group_kind)
    return np.bincount(truth.feature_group, weights=attr, minlength=n_groups)


def rank_corr_informative(gs: np.ndarray, truth: Truth) -> float:
    """Spearman correlation between estimated and true importance over informative groups."""
    idx = truth.informative_groups
    rho = spearmanr(gs[idx], truth.group_importance[idx]).statistic
    return float(rho) if np.isfinite(rho) else 0.0


def feature_level_scores(unit_scores: np.ndarray, clusters: Sequence[np.ndarray] | None,
                         n_features: int) -> np.ndarray:
    """Score each feature as the analyst would read it when pruning.

    Plain methods: the feature's own score. Clustered methods: the score of the
    feature's cluster (undivided), since keeping any member keeps that information.
    """
    if clusters is None:
        return np.asarray(unit_scores, dtype=float)
    out = np.zeros(n_features)
    for s, c in zip(unit_scores, clusters):
        out[c] = s
    return out


def rank_corr_features(fs: np.ndarray, truth: Truth) -> float:
    """Spearman over all features between a feature's score and its group's true importance.

    Noise features have true importance 0. Low values mean credit is diluted across
    copies (a feature in a big important group looks unimportant) or given to noise.
    """
    target = truth.group_importance[truth.feature_group]
    rho = spearmanr(fs, target).statistic
    return float(rho) if np.isfinite(rho) else 0.0


def select(unit_scores: np.ndarray, clusters: Sequence[np.ndarray] | None, X: np.ndarray,
           k: int) -> np.ndarray:
    """Pick k features.

    Plain methods: the k highest-scoring features. Clustered methods: walk the clusters
    from the highest score down and take each cluster's most central member; if k is
    larger than the number of clusters, take the next most central members in a second
    pass, and so on. Ties are broken by index, so selection is deterministic.
    """
    unit_scores = np.asarray(unit_scores, dtype=float)
    if clusters is None:
        order = np.lexsort((np.arange(len(unit_scores)), -unit_scores))
        return np.sort(order[:k])
    corder = np.lexsort((np.arange(len(unit_scores)), -unit_scores))
    queues = [list(medoid_order(X, clusters[c])) for c in corder]
    chosen: list[int] = []
    depth = 0
    while len(chosen) < k and any(len(q) > depth for q in queues):
        for q in queues:
            if len(q) > depth:
                chosen.append(int(q[depth]))
                if len(chosen) == k:
                    break
        depth += 1
    return np.sort(np.asarray(chosen, dtype=int))


def recall_informative(selected: np.ndarray, truth: Truth) -> float:
    """Share of informative groups with at least one selected feature."""
    groups = set(truth.feature_group[selected].tolist())
    inf = set(truth.informative_groups.tolist())
    return len(groups & inf) / len(inf)


def noise_fdr(selected: np.ndarray, truth: Truth) -> float:
    """Share of selected features that belong to a noise group."""
    if len(selected) == 0:
        return 0.0
    return float(truth.is_noise_feature[selected].mean())


def mean_pairwise_spearman(vectors: Sequence[np.ndarray]) -> float:
    vals = []
    for a, b in combinations(vectors, 2):
        r = spearmanr(a, b).statistic
        vals.append(float(r) if np.isfinite(r) else 0.0)
    return float(np.mean(vals))


def mean_pairwise_jaccard(sets: Sequence[np.ndarray]) -> float:
    vals = []
    for a, b in combinations(sets, 2):
        sa, sb = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
        vals.append(len(sa & sb) / len(sa | sb) if sa | sb else 1.0)
    return float(np.mean(vals))


def adjusted_rand(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(labels_a, labels_b))
