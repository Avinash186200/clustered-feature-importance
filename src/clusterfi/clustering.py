"""Feature clustering by correlation distance and hierarchical linkage.

``d_ij = sqrt((1 - rho_ij) / 2)`` is 0 for identical features and 1 for perfectly
opposite ones. For standardised columns z_i, z_j over n rows,
``||z_i - z_j||^2 = 2 n (1 - rho_ij)``, so d is a scaled Euclidean distance: it is a
true metric, and Ward linkage on it is valid.

The number of clusters is chosen per dataset, from training rows only, by scanning
k = 2..k_max and keeping the k with the best silhouette score. The linkage and the
score (mean silhouette, or mean / std as in the "t-stat" variant) are fixed on
development seeds; see ``bench.choose_clustering``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples

LINKAGES = ("single", "average", "complete", "ward")
CRITERIA = ("silhouette", "tstat")


@dataclass(frozen=True)
class ClusterConfig:
    linkage: str = "single"
    criterion: str = "silhouette"
    k_max_frac: float = 0.5        # scan k up to this share of the feature count

    def to_dict(self) -> dict:
        return asdict(self)


def corr_distance(X: np.ndarray) -> np.ndarray:
    c = np.corrcoef(X, rowvar=False)
    d = np.sqrt(np.clip((1.0 - c) / 2.0, 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return (d + d.T) / 2.0


def _score(d: np.ndarray, labels: np.ndarray, criterion: str) -> float:
    s = silhouette_samples(d, labels, metric="precomputed")
    if criterion == "silhouette":
        return float(s.mean())
    sd = float(s.std())
    return float(s.mean() / sd) if sd > 0 else -np.inf


@dataclass
class Clustering:
    labels: np.ndarray             # cluster id per feature, 0..n_clusters-1
    k: int
    scores: dict[int, float]

    @property
    def clusters(self) -> list[np.ndarray]:
        return [np.flatnonzero(self.labels == c) for c in range(self.k)]


def cluster_features(X: np.ndarray, cfg: ClusterConfig) -> Clustering:
    d = corr_distance(X)
    p = d.shape[0]
    Z = linkage(squareform(d, checks=False), method=cfg.linkage)
    k_max = max(2, min(p - 1, int(cfg.k_max_frac * p)))
    scores: dict[int, float] = {}
    best_s, best_labels = -np.inf, None
    for k in range(2, k_max + 1):
        labels = fcluster(Z, k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        s = _score(d, labels, cfg.criterion)
        scores[k] = s
        if s > best_s:
            best_s, best_labels = s, labels
    # relabel to 0..k-1 in order of first appearance
    _, labels = np.unique(best_labels, return_inverse=True)
    return Clustering(labels=labels.astype(int), k=int(labels.max() + 1), scores=scores)


def medoid_order(X: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Members ordered by mean absolute correlation with the rest of the cluster."""
    if len(members) == 1:
        return members
    c = np.abs(np.corrcoef(X[:, members], rowvar=False))
    centrality = (c.sum(axis=1) - 1.0) / (len(members) - 1)
    return members[np.lexsort((members, -centrality))]
