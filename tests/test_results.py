"""The committed results are reproducible.

* Every summary (dev_choices, sweep, base) and RESULTS.md is recomputed from the
  per-task records in results/runs.json and must match the committed files.
* A fixed subset of tasks is rerun from scratch and must match its record: one
  held-out seed in every sweep cell (the base cell includes drop-column and both
  budgets), one stability seed, three development clustering tasks and every
  development model fit. Set FULL=1 to rerun every task (several minutes on a
  laptop with many cores; this is what ``make verify`` does).
"""
import json
import math
import os
from pathlib import Path

import pytest
from joblib import Parallel, delayed

from clusterfi import bench as B
from clusterfi.clustering import ClusterConfig
from clusterfi.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FULL = os.environ.get("FULL") == "1"
N_JOBS = min(4, os.cpu_count() or 1)


def _diff(a, b, path="$"):
    """Differences between two JSON trees; floats match to 1e-6 absolute."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(a.keys() | b.keys()):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: missing")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return []
        ok = isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
            math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
        return [] if ok else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def _norm(x):
    return json.loads(json.dumps(x))


@pytest.fixture(scope="module")
def runs():
    return json.loads((RES / "runs.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", B.SUMMARY_FILES)
def test_summaries_are_computed_from_the_records(runs, name):
    fresh = _norm(B.SUMMARIZERS[name](runs))
    committed = json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))
    diffs = _diff(fresh, committed)
    assert not diffs, "\n".join(diffs[:20])


def test_results_md_is_rendered_from_the_summaries():
    assert B.render(RES) == (RES / "RESULTS.md").read_text(encoding="utf-8")


def test_results_written_with_lf_only():
    for f in RES.iterdir():
        if f.suffix in (".json", ".md"):
            assert b"\r\n" not in f.read_bytes(), f.name


def test_library_defaults_are_the_frozen_development_choices():
    dev = json.loads((RES / "dev_choices.json").read_text(encoding="utf-8"))
    assert ModelConfig().to_dict() == dev["chosen_model"]
    assert ClusterConfig().to_dict() == dev["chosen_cluster"]


def test_heldout_and_development_seeds_are_disjoint(runs):
    dev = {r["seed"] for r in runs["dev_cluster"]} | {r["seed"] for r in runs["dev_model"]}
    held = {r["seed"] for r in runs["cells"]} | {r["seed"] for r in runs["stability"]}
    assert dev == set(B.DEV_SEEDS) and held == set(B.HELDOUT_SEEDS)
    assert not dev & held


def _subset(runs):
    mcfg, ccfg = B.frozen_choices(json.loads((RES / "dev_choices.json").read_text(encoding="utf-8")))
    first = B.HELDOUT_SEEDS[0]
    jobs = []
    for i, r in enumerate(runs["cells"]):
        if FULL or r["seed"] == first:
            jobs.append((("cells", i), B.cell_task, (r["rho"], r["noise_frac"], r["seed"], mcfg, ccfg)))
    for i, r in enumerate(runs["stability"]):
        if FULL or r["seed"] == first:
            jobs.append((("stability", i), B.stability_task, (r["seed"], mcfg, ccfg)))
    for i, r in enumerate(runs["dev_cluster"]):
        if FULL or (r["seed"] == B.DEV_SEEDS[0] and r["noise_frac"] == 0.5):
            jobs.append((("dev_cluster", i), B.dev_cluster_task, (r["rho"], r["noise_frac"], r["seed"])))
    for i, r in enumerate(runs["dev_model"]):
        jobs.append((("dev_model", i), B.dev_model_task, (r["model_idx"], r["seed"])))
    return jobs


@pytest.fixture(scope="module")
def reruns(runs):
    jobs = _subset(runs)
    outs = Parallel(n_jobs=N_JOBS)(delayed(fn)(*args) for _, fn, args in jobs)
    return [(key, out) for (key, _, _), out in zip(jobs, outs)]


def test_rerun_covers_every_kind_of_task(reruns):
    kinds = {k for (k, _), _ in reruns}
    assert kinds == {"cells", "stability", "dev_cluster", "dev_model"}
    cells = [(k, i) for (k, i), _ in reruns if k == "cells"]
    assert len(cells) >= len(B.RHOS) * len(B.NOISE_FRACS)


def test_rerun_tasks_match_their_records(runs, reruns):
    diffs = []
    for (kind, i), fresh in reruns:
        diffs += _diff(_norm(fresh), runs[kind][i], f"{kind}[{i}]")
    assert not diffs, "\n".join(diffs[:20])
