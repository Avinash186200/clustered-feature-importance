import numpy as np

from clusterfi.clustering import ClusterConfig, cluster_features, corr_distance, medoid_order
from clusterfi.data import GenConfig, make_splits
from clusterfi.metrics import adjusted_rand


def test_corr_distance_is_scaled_euclidean():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 6))
    X[:, 1] = X[:, 0] + 0.1 * rng.standard_normal(500)
    X[:, 2] = -X[:, 0]
    d = corr_distance(X)
    Z = (X - X.mean(0)) / X.std(0)
    eu = np.sqrt(((Z[:, :, None] - Z[:, None, :]) ** 2).sum(0) / (4 * X.shape[0]))
    assert np.allclose(d, eu, atol=1e-9)
    assert d[0, 0] == 0 and abs(d[0, 2] - 1) < 1e-9 and d[0, 1] < 0.1


def test_clusters_recover_strong_structure():
    s = make_splits(GenConfig(rho=0.9, noise_frac=0.25), 7, n_train=2000, n_valid=10, n_test=10)
    cl = cluster_features(s.train.X, ClusterConfig())
    assert adjusted_rand(cl.labels, s.truth.feature_group) > 0.8
    assert sorted(np.concatenate(cl.clusters).tolist()) == list(range(s.train.X.shape[1]))


def test_both_criteria_and_all_linkages_run():
    s = make_splits(GenConfig(noise_frac=0.25), 8, n_train=400, n_valid=10, n_test=10)
    for lk in ("single", "average", "complete", "ward"):
        for cr in ("silhouette", "tstat"):
            cl = cluster_features(s.train.X, ClusterConfig(linkage=lk, criterion=cr))
            assert 2 <= cl.k <= s.train.X.shape[1]


def test_medoid_is_the_most_central_member():
    rng = np.random.default_rng(1)
    f = rng.standard_normal(3000)
    X = np.column_stack([f + s * rng.standard_normal(3000) for s in (1.0, 0.2, 0.6)])
    assert medoid_order(X, np.array([0, 1, 2]))[0] == 1
