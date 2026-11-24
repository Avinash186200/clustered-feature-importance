import numpy as np

from clusterfi import metrics as M
from clusterfi.data import Truth


def _truth():
    # groups: 0 (informative, imp .3), 1 (informative, imp .1), 2 (noise), 3 (noise)
    return Truth(names=[f"x{i}" for i in range(6)],
                 feature_group=np.array([0, 0, 0, 1, 2, 3]),
                 group_kind=["informative", "informative", "noise", "noise"],
                 group_importance=np.array([0.3, 0.1, 0.0, 0.0]),
                 feature_role=["copy"] * 4 + ["noise"] * 2,
                 n_informative_groups=2, sigma=1.0)


def test_plain_topk_picks_duplicates_and_misses_groups():
    t = _truth()
    scores = np.array([0.5, 0.4, 0.3, 0.05, 0.01, 0.0])
    sel = M.select(scores, None, None, 2)
    assert sel.tolist() == [0, 1]
    assert M.recall_informative(sel, t) == 0.5
    assert M.noise_fdr(sel, t) == 0.0


def test_clustered_select_takes_one_member_per_cluster_first():
    t = _truth()
    rng = np.random.default_rng(0)
    f = rng.standard_normal(500)
    X = np.column_stack([f + s * rng.standard_normal(500) for s in (0.5, 0.1, 0.5)] +
                        [rng.standard_normal(500) for _ in range(3)])
    clusters = [np.array([0, 1, 2]), np.array([3]), np.array([4]), np.array([5])]
    sel = M.select(np.array([0.9, 0.2, 0.05, 0.0]), clusters, X, 2)
    assert sel.tolist() == [1, 3]                    # medoid of cluster 0, then cluster 1
    assert M.recall_informative(sel, t) == 1.0
    sel6 = M.select(np.array([0.9, 0.2, 0.05, 0.0]), clusters, X, 6)
    assert sel6.tolist() == list(range(6))


def test_attribution_and_group_scores():
    t = _truth()
    clusters = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    attr = M.feature_attribution(np.array([0.6, 0.3]), clusters, 6)
    assert np.allclose(attr, [0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    gs = M.group_scores(attr, t)
    assert np.allclose(gs, [0.6, 0.1, 0.1, 0.1])
    fs = M.feature_level_scores(np.array([0.6, 0.3]), clusters, 6)
    assert np.allclose(fs, [0.6] * 3 + [0.3] * 3)
    assert abs(M.rank_corr_informative(gs, t) - 1.0) < 1e-12


def test_feature_rank_corr_penalises_dilution():
    t = _truth()
    diluted = np.array([0.1, 0.1, 0.1, 0.2, 0.0, 0.0])   # group 0 split across 3 copies
    inherited = np.array([0.3, 0.3, 0.3, 0.2, 0.0, 0.0])
    assert M.rank_corr_features(inherited, t) > M.rank_corr_features(diluted, t)


def test_stability_helpers():
    a, b = np.array([1, 2, 3.0]), np.array([1, 2, 3.0])
    assert abs(M.mean_pairwise_spearman([a, b, a]) - 1.0) < 1e-12
    assert M.mean_pairwise_jaccard([np.array([1, 2]), np.array([2, 3])]) == 1 / 3
