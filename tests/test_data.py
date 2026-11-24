import numpy as np

from clusterfi.data import KIND_INFORMATIVE, KIND_NOISE, GenConfig, make_splits


def test_layout_is_fixed_per_seed_and_rows_differ_between_splits():
    a = make_splits(GenConfig(), 3, n_train=300, n_valid=200, n_test=200)
    b = make_splits(GenConfig(), 3, n_train=300, n_valid=200, n_test=200)
    assert np.array_equal(a.train.X, b.train.X)
    assert np.array_equal(a.truth.feature_group, b.truth.feature_group)
    assert not np.allclose(a.train.X[:200], a.valid.X)


def test_group_importance_matches_variance_decomposition():
    cfg = GenConfig()
    s = make_splits(cfg, 1, n_train=200_000, n_valid=10, n_test=10)
    t = s.truth
    inf = t.informative_groups
    assert len(inf) == cfg.n_factors == t.n_informative_groups
    # true shares of Var(y) sum to the oracle R^2
    assert abs(t.group_importance[inf].sum() - cfg.r2_latent) < 1e-12
    assert np.all(t.group_importance[np.array(t.group_kind) == KIND_NOISE] == 0)
    # empirical check: Var(y) and the share of the first linear factor
    y, F = s.train.y, s.train.F
    var_y = (sum(b * b for b in cfg.betas) + cfg.gamma ** 2) / cfg.r2_latent
    assert abs(y.var() / var_y - 1) < 0.02
    share0 = np.cov(y, F[:, 0])[0, 1] ** 2 / y.var()
    assert abs(share0 - t.group_importance[0]) < 0.01


def test_copies_have_the_requested_correlation():
    cfg = GenConfig(rho=0.8, interaction=False)
    s = make_splits(cfg, 2, n_train=50_000, n_valid=10, n_test=10)
    t = s.truth
    roles = np.array(t.feature_role)
    g0 = np.flatnonzero((t.feature_group == 0) & (roles == "copy"))
    assert len(g0) >= 2
    c = np.corrcoef(s.train.X[:, g0[:2]], rowvar=False)[0, 1]
    assert abs(c - 0.8) < 0.02
    iso = np.flatnonzero(roles == "noise")
    assert abs(np.corrcoef(s.train.X[:, iso[0]], s.train.y)[0, 1]) < 0.02


def test_noise_fraction_and_roles():
    for nf in (0.25, 0.5, 0.75):
        s = make_splits(GenConfig(noise_frac=nf), 4, n_train=50, n_valid=10, n_test=10)
        share = s.truth.is_noise_feature.mean()
        assert abs(share - nf) < 0.02
    kinds = set(s.truth.group_kind)
    assert kinds == {KIND_INFORMATIVE, KIND_NOISE}


def test_interaction_factors_have_no_linear_signal():
    cfg = GenConfig()
    s = make_splits(cfg, 5, n_train=100_000, n_valid=10, n_test=10)
    lin = len(cfg.betas)
    for k in (lin, lin + 1):
        assert abs(np.corrcoef(s.train.F[:, k], s.train.y)[0, 1]) < 0.015
