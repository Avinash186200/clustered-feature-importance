"""Speedup of permutation importance versus worker count (machine-specific).

Permutation importance is embarrassingly parallel: every (feature, repeat) pair is an
independent predict call on a shuffled copy of the validation rows. The features are
split into chunks and the chunks run on a joblib backend (loky processes by default).
LightGBM is pinned to one thread inside each worker, so the workers do not fight over
cores.

The numbers depend on the machine and the load on it, so tests never check them.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .data import GenConfig, make_splits
from .importance import mda
from .model import ModelConfig, fit


def machine_info() -> dict:
    info = {"python": platform.python_version(), "system": platform.system(),
            "release": platform.release(), "version": platform.version(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpus": os.cpu_count()}
    info["summary"] = (f"{info['processor']}, {info['logical_cpus']} logical CPUs, "
                       f"{info['system']} {info['release']} (build {info['version']}), Python {info['python']}")
    return info


def run_scale(workers=(1, 2, 4, 8), noise_frac: float = 0.75, n_valid: int = 4000,
              n_repeats: int = 3, passes: int = 2, backend: str = "loky", seed: int = 1) -> dict:
    s = make_splits(GenConfig(noise_frac=noise_frac, r2_latent=0.25), seed,
                    n_train=2000, n_valid=n_valid, n_test=10)
    model = fit(s.train.X, s.train.y, ModelConfig(num_leaves=7, n_estimators=100,
                                                  learning_rate=0.08), seed)
    X, y = s.valid.X, s.valid.y
    ref = mda(model, X, y, n_repeats=n_repeats, seed=seed, n_jobs=1)
    rows = []
    for w in workers:
        if w > 1:  # start the worker pool once so the timing excludes process start-up
            mda(model, X[:200], y[:200], n_repeats=1, seed=seed, n_jobs=w, backend=backend)
        best = np.inf
        for _ in range(passes):
            t0 = time.perf_counter()
            res = mda(model, X, y, n_repeats=n_repeats, seed=seed, n_jobs=w, backend=backend)
            best = min(best, time.perf_counter() - t0)
        rows.append({"workers": w, "seconds": round(best, 3),
                     "identical_to_serial": bool(np.array_equal(res, ref))})
    t1 = rows[0]["seconds"]
    for r in rows:
        r["speedup"] = round(t1 / r["seconds"], 3)
        r["efficiency"] = round(t1 / r["seconds"] / r["workers"], 3)
    return {"p": int(X.shape[1]), "n_valid": n_valid, "n_repeats": n_repeats, "passes": passes,
            "backend": backend, "machine": machine_info(), "rows": rows}


def write_scale(out: Path, **kw) -> dict:
    res = run_scale(**kw)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scale.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    return res
