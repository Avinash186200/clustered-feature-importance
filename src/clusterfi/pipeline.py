"""Run every importance method on one dataset and score it against the truth."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import importance as imp
from . import metrics as M
from .clustering import ClusterConfig, Clustering, cluster_features
from .data import Dataset, Truth
from .model import ModelConfig, fit, predict, r2

PLAIN = ("MDI", "MDA", "SHAP")
CLUSTERED = ("cMDI", "cMDA", "cSHAP")
DROP = "drop-column"
METHODS = PLAIN + (DROP,) + CLUSTERED


@dataclass
class MethodResult:
    unit_scores: np.ndarray
    clusters: list[np.ndarray] | None


@dataclass
class RunOutput:
    results: dict[str, MethodResult]
    clustering: Clustering
    valid_r2: float


def run_methods(train: Dataset, valid: Dataset, mcfg: ModelConfig, ccfg: ClusterConfig,
                seed: int, n_repeats: int = 5, with_drop: bool = False,
                n_jobs: int = 1) -> RunOutput:
    p = train.X.shape[1]
    model = fit(train.X, train.y, mcfg, seed)
    base = r2(valid.y, predict(model, valid.X))
    cl = cluster_features(train.X, ccfg)       # training rows only, no target
    clusters = cl.clusters
    res = {
        "MDI": MethodResult(imp.mdi(model, p), None),
        "MDA": MethodResult(imp.mda(model, valid.X, valid.y, n_repeats, seed, n_jobs), None),
        "SHAP": MethodResult(imp.shap_importance(model, valid.X), None),
        "cMDI": MethodResult(imp.clustered_mdi(model, clusters, p), clusters),
        "cMDA": MethodResult(imp.clustered_mda(model, valid.X, valid.y, clusters, n_repeats,
                                               seed, n_jobs), clusters),
        "cSHAP": MethodResult(imp.clustered_shap(model, valid.X, clusters), clusters),
    }
    if with_drop:
        res[DROP] = MethodResult(imp.drop_column(train.X, train.y, valid.X, valid.y, mcfg, seed,
                                                 base, n_jobs), None)
    return RunOutput(results=res, clustering=cl, valid_r2=base)


def score_methods(out: RunOutput, truth: Truth, X_train: np.ndarray, k: int) -> dict:
    p = X_train.shape[1]
    rows = {}
    for name, mr in out.results.items():
        attr = M.feature_attribution(mr.unit_scores, mr.clusters, p)
        gs = M.group_scores(attr, truth)
        sel = M.select(mr.unit_scores, mr.clusters, X_train, k)
        rows[name] = {
            "rank_corr": M.rank_corr_informative(gs, truth),
            "feature_rank_corr": M.rank_corr_features(
                M.feature_level_scores(mr.unit_scores, mr.clusters, p), truth),
            "recall_at_k": M.recall_informative(sel, truth),
            "noise_fdr_at_k": M.noise_fdr(sel, truth),
            "selected": sel.tolist(),
            "group_scores": gs.tolist(),
        }
    return rows


def true_labels(truth: Truth) -> np.ndarray:
    return truth.feature_group
