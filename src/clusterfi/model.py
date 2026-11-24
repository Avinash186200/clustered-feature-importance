"""One model for every importance method: a small, deterministic LightGBM regressor."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import lightgbm as lgb
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    num_leaves: int = 7
    n_estimators: int = 100
    learning_rate: float = 0.08
    min_child_samples: int = 20
    colsample_bytree: float = 0.5
    reg_lambda: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def fit(X: np.ndarray, y: np.ndarray, cfg: ModelConfig, seed: int = 0) -> lgb.Booster:
    params = {
        "objective": "regression",
        "num_leaves": cfg.num_leaves,
        "learning_rate": cfg.learning_rate,
        "min_data_in_leaf": cfg.min_child_samples,
        "feature_fraction": cfg.colsample_bytree,
        "lambda_l2": cfg.reg_lambda,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "verbose": -1,
    }
    ds = lgb.Dataset(X, label=y, params={"verbose": -1})
    return lgb.train(params, ds, num_boost_round=cfg.n_estimators)


def predict(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    return model.predict(X, num_threads=1)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot
