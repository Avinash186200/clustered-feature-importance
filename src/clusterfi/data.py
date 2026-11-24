"""Synthetic wide feature panel with known ground truth.

The target depends on a few latent factors. The analyst never sees the factors, only
noisy copies of them (near-duplicates, like different look-backs of one idea), some
linear combinations of two factors, and pure noise, part of which is itself clustered
(several copies of a useless idea).

Ground truth is defined on *groups*:

* one group per informative factor: its copies plus the combinations whose dominant
  weight is on that factor;
* one group per noise factor (its copies);
* one group per isolated noise feature.

True importance of a group is its share of Var(y). Factors are independent standard
normals, so for ``y = sum_k beta_k f_k + gamma f_a f_b + sigma e`` the terms are
uncorrelated and the variance splits exactly: ``beta_k^2`` for a linear factor and
``gamma^2 / 2`` for each factor of the interaction (the Shapley split of the
interaction's variance, which is symmetric). Noise groups have importance 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

KIND_INFORMATIVE = "informative"
KIND_NOISE = "noise"


@dataclass(frozen=True)
class GenConfig:
    betas: tuple[float, ...] = (1.0, 0.85, 0.7, 0.55, 0.4, 0.25)
    interaction: bool = True          # two extra factors that act only through their product
    gamma: float = 1.0
    copies: tuple[int, int] = (2, 12)  # noisy copies per factor: drawn per seed in [lo, hi]
    rho: float = 0.7                  # correlation between two copies of the same factor
    n_combo: int = 8                  # redundant features: 0.8 f_k + 0.6 f_l, plus noise
    noise_frac: float = 0.5           # share of all features that are noise
    noise_clustered_share: float = 0.5  # share of noise features that come in clusters
    noise_cluster_size: tuple[int, int] = (2, 12)
    r2_latent: float = 0.5            # R^2 an oracle that sees the factors would get

    @property
    def n_factors(self) -> int:
        return len(self.betas) + (2 if self.interaction else 0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Truth:
    names: list[str]
    feature_group: np.ndarray          # group id per feature
    group_kind: list[str]              # per group: informative / noise
    group_importance: np.ndarray       # per group: share of Var(y)
    feature_role: list[str]            # copy / combo / noise_copy / noise
    n_informative_groups: int
    sigma: float
    factor_of_group: list[int] = field(default_factory=list)  # -1 for noise groups

    @property
    def informative_groups(self) -> np.ndarray:
        return np.flatnonzero(np.array(self.group_kind) == KIND_INFORMATIVE)

    @property
    def is_noise_feature(self) -> np.ndarray:
        kinds = np.array(self.group_kind)
        return kinds[self.feature_group] == KIND_NOISE


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    F: np.ndarray                      # latent factors (for the oracle only)


def _copy(rng, base: np.ndarray, rho: float) -> np.ndarray:
    return np.sqrt(rho) * base + np.sqrt(1.0 - rho) * rng.standard_normal(base.shape[0])


def make_truth_and_sampler(cfg: GenConfig, seed: int):
    """Fix the feature layout for a seed and return (truth, sampler).

    ``sampler(n, rng)`` draws rows from the same structure, so train, validation and
    test rows of one run share the layout but never share rows.
    """
    rng = np.random.default_rng([seed, 0])
    K = cfg.n_factors
    n_lin = len(cfg.betas)
    imp = np.array(list(np.square(cfg.betas)) + ([cfg.gamma ** 2 / 2] * 2 if cfg.interaction else []))
    var_signal = float(imp.sum())
    sigma = float(np.sqrt(var_signal * (1 - cfg.r2_latent) / cfg.r2_latent))
    var_y = var_signal + sigma ** 2

    # combos: pair factor k (dominant, weight 0.8) with factor k+1 (weight 0.6), cycling
    combo_pairs = [(i % K, (i + 1) % K) for i in range(cfg.n_combo)]
    lo, hi = cfg.copies
    copies = rng.integers(lo, hi + 1, size=K)          # unrelated to importance
    n_signal = int(copies.sum()) + cfg.n_combo
    n_noise = int(round(cfg.noise_frac / (1 - cfg.noise_frac) * n_signal)) if cfg.noise_frac < 1 else 0
    budget = int(round(n_noise * cfg.noise_clustered_share))
    noise_sizes: list[int] = []
    while True:
        sz = int(rng.integers(cfg.noise_cluster_size[0], cfg.noise_cluster_size[1] + 1))
        if sum(noise_sizes) + sz > budget:
            break
        noise_sizes.append(sz)
    n_noise_clusters = len(noise_sizes)
    n_noise_iso = n_noise - sum(noise_sizes)

    # spec per raw feature: (kind, payload)
    spec: list[tuple[str, tuple]] = []
    groups: list[int] = []
    roles: list[str] = []
    for k in range(K):
        for _ in range(int(copies[k])):
            spec.append(("copy", (k,)))
            groups.append(k)
            roles.append("copy")
    for k, k2 in combo_pairs:
        spec.append(("combo", (k, k2)))
        groups.append(k)
        roles.append("combo")
    gid = K
    for c in range(n_noise_clusters):
        for _ in range(noise_sizes[c]):
            spec.append(("noise_copy", (c,)))
            groups.append(gid)
            roles.append("noise_copy")
        gid += 1
    for _ in range(n_noise_iso):
        spec.append(("noise", ()))
        groups.append(gid)
        roles.append("noise")
        gid += 1

    group_kind = [KIND_INFORMATIVE] * K + [KIND_NOISE] * (gid - K)
    group_imp = np.concatenate([imp / var_y, np.zeros(gid - K)])
    factor_of_group = list(range(K)) + [-1] * (gid - K)

    # shuffle the column order so position carries no information
    order = rng.permutation(len(spec))
    spec = [spec[i] for i in order]
    groups_arr = np.asarray(groups)[order]
    roles = [roles[i] for i in order]
    names = [f"x{j:03d}" for j in range(len(spec))]

    truth = Truth(names=names, feature_group=groups_arr, group_kind=group_kind,
                  group_importance=group_imp, feature_role=roles,
                  n_informative_groups=K, sigma=sigma, factor_of_group=factor_of_group)

    betas = np.asarray(cfg.betas)
    rho = cfg.rho

    def sampler(n: int, rng_rows: np.random.Generator) -> Dataset:
        F = rng_rows.standard_normal((n, K))
        Fn = rng_rows.standard_normal((n, max(n_noise_clusters, 1)))
        y = F[:, :n_lin] @ betas
        if cfg.interaction:
            y = y + cfg.gamma * F[:, n_lin] * F[:, n_lin + 1]
        y = y + sigma * rng_rows.standard_normal(n)
        X = np.empty((n, len(spec)))
        for j, (kind, p) in enumerate(spec):
            if kind == "copy":
                X[:, j] = _copy(rng_rows, F[:, p[0]], rho)
            elif kind == "combo":
                X[:, j] = _copy(rng_rows, 0.8 * F[:, p[0]] + 0.6 * F[:, p[1]], rho)
            elif kind == "noise_copy":
                X[:, j] = _copy(rng_rows, Fn[:, p[0]], rho)
            else:
                X[:, j] = rng_rows.standard_normal(n)
        return Dataset(X=X, y=y, F=F)

    return truth, sampler


@dataclass
class Splits:
    train: Dataset
    valid: Dataset
    test: Dataset
    truth: Truth


def make_splits(cfg: GenConfig, seed: int, n_train: int = 2000, n_valid: int = 1000,
                n_test: int = 4000) -> Splits:
    """Independent train / validation / test rows from one fixed layout."""
    truth, sampler = make_truth_and_sampler(cfg, seed)
    return Splits(train=sampler(n_train, np.random.default_rng([seed, 1])),
                  valid=sampler(n_valid, np.random.default_rng([seed, 2])),
                  test=sampler(n_test, np.random.default_rng([seed, 3])),
                  truth=truth)
