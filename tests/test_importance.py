import numpy as np
import pytest

from clusterfi import importance as imp
from clusterfi.model import ModelConfig, fit, predict


@pytest.fixture(scope="module")
def toy():
    """y depends on f only; x0, x1 are near-copies of f, x2, x3 are noise."""
    rng = np.random.default_rng(0)
    n = 3000
    f = rng.standard_normal(n)
    X = np.column_stack([f + 0.3 * rng.standard_normal(n), f + 0.3 * rng.standard_normal(n),
                         rng.standard_normal(n), rng.standard_normal(n)])
    y = f + 0.5 * rng.standard_normal(n)
    cfg = ModelConfig(num_leaves=7, n_estimators=80, learning_rate=0.1, colsample_bytree=1.0)
    m = fit(X[:2000], y[:2000], cfg, 0)
    return m, X, y, cfg


def test_parallel_permutation_is_identical_to_serial(toy):
    m, X, y, _ = toy
    serial = imp.mda(m, X[2000:], y[2000:], n_repeats=3, seed=4, n_jobs=1)
    par = imp.mda(m, X[2000:], y[2000:], n_repeats=3, seed=4, n_jobs=2)
    assert np.array_equal(serial, par)
    units = [np.array([0, 1]), np.array([2, 3])]
    s2 = imp.clustered_mda(m, X[2000:], y[2000:], units, n_repeats=3, seed=4, n_jobs=1)
    p2 = imp.clustered_mda(m, X[2000:], y[2000:], units, n_repeats=3, seed=4, n_jobs=2)
    assert np.array_equal(s2, p2)


def test_substitution_effect_and_its_fix(toy):
    """Shuffling one copy barely hurts; shuffling the cluster hurts a lot."""
    m, X, y, _ = toy
    Xv, yv = X[2000:], y[2000:]
    single = imp.mda(m, Xv, yv, n_repeats=5, seed=1)
    joint = imp.clustered_mda(m, Xv, yv, [np.array([0, 1]), np.array([2, 3])], n_repeats=5, seed=1)
    assert joint[0] > single[0] + single[1]
    assert abs(joint[1]) < 0.02


def test_permutation_restores_input(toy):
    m, X, y, _ = toy
    Xv = X[2000:].copy()
    imp.mda(m, Xv, y[2000:], n_repeats=2, seed=0)
    assert np.array_equal(Xv, X[2000:])


def test_shap_is_additive_and_clustered_shap_sums_members(toy):
    m, X, y, _ = toy
    Xv = X[2000:2200]
    phi = imp.shap_values(m, Xv)
    full = m.predict(Xv, pred_contrib=True)
    assert np.allclose(full.sum(axis=1), predict(m, Xv), atol=1e-9)
    c = imp.clustered_shap(m, Xv, [np.array([0, 1]), np.array([2, 3])])
    assert np.isclose(c[0], np.abs(phi[:, 0] + phi[:, 1]).mean())


def test_mdi_normalised_and_clustered_mdi_sums(toy):
    m, X, _, _ = toy
    g = imp.mdi(m, 4)
    assert np.isclose(g.sum(), 1.0)
    c = imp.clustered_mdi(m, [np.array([0, 1]), np.array([2, 3])], 4)
    assert np.isclose(c[0], g[0] + g[1]) and c[0] > 0.8


def test_drop_column_serial_equals_parallel(toy):
    _, X, y, cfg = toy
    base_m = fit(X[:2000], y[:2000], cfg, 0)
    from clusterfi.model import r2
    base = r2(y[2000:], predict(base_m, X[2000:]))
    a = imp.drop_column(X[:2000], y[:2000], X[2000:], y[2000:], cfg, 0, base, n_jobs=1)
    b = imp.drop_column(X[:2000], y[:2000], X[2000:], y[2000:], cfg, 0, base, n_jobs=2)
    assert np.array_equal(a, b)
    # dropping one copy costs little: the other copy substitutes for it
    assert a[0] < 0.05
