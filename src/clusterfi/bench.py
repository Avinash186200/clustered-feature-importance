"""The full, deterministic benchmark.

1. Development seeds: choose the clustering rule (linkage x k-selection score) by
   adjusted Rand index against the true groups, and the LightGBM settings by
   validation R^2 (one-standard-error rule). Nothing else is tuned here.
2. Held-out seeds, run once with the frozen choices:
   * a sweep over within-cluster correlation x noise share, every method except
     drop-column (one refit per feature), with the downstream check at k;
   * the base cell in detail: drop-column, a second budget, and bootstrap stability.

Every task is seeded and single-threaded inside, so results do not depend on
``n_jobs``. Per-task records go to ``results/runs.json``; ``dev_choices.json``,
``sweep.json`` and ``base.json`` are pure summaries of those records, and
``RESULTS.md`` is rendered from the summaries.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from . import metrics as M
from .clustering import CRITERIA, LINKAGES, ClusterConfig, cluster_features
from .data import Dataset, GenConfig, make_splits
from .model import ModelConfig, fit, predict, r2
from .pipeline import CLUSTERED, DROP, PLAIN, run_methods, score_methods

DEV_SEEDS = (1, 2, 3, 4, 5)
HELDOUT_SEEDS = (101, 102, 103, 104, 105, 106, 107, 108)
RHOS = (0.5, 0.7, 0.9)
NOISE_FRACS = (0.25, 0.5, 0.75)
BASE = {"rho": 0.7, "noise_frac": 0.5}
R2_LATENT = 0.25
N_REPEATS = 3
N_BOOT = 4
N_RANDOM = 3
MODEL_GRID = [ModelConfig(num_leaves=nl, n_estimators=ne, learning_rate=lr)
              for nl in (3, 7, 15) for ne, lr in ((100, 0.08), (300, 0.03))]
SUMMARY_FILES = ("dev_choices", "sweep", "base")
DEDUPE = "MDA+dedupe"
CMEANS = "cMDA, cluster means"
SCORE_KEYS = ("rank_corr", "feature_rank_corr", "recall_at_k", "noise_fdr_at_k")
STAB_KEYS = ("spearman", "jaccard", "group_jaccard")


def gen_cfg(rho: float, noise_frac: float) -> GenConfig:
    return GenConfig(rho=rho, noise_frac=noise_frac, r2_latent=R2_LATENT)


def k_informative() -> int:
    return gen_cfg(**BASE).n_factors


def is_base(rho: float, nf: float) -> bool:
    return rho == BASE["rho"] and nf == BASE["noise_frac"]


def budgets_for(rho: float, noise_frac: float) -> tuple[int, ...]:
    K = k_informative()
    return (K // 2, K) if is_base(rho, noise_frac) else (K,)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")


def _r(x: float, nd: int = 6) -> float:
    return float(round(float(x), nd))


def _round_tree(x):
    if isinstance(x, dict):
        return {k: _round_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round_tree(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return _r(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def mean_se(vals) -> dict:
    """Mean and standard error across seeds (each seed is an independent dataset)."""
    a = np.asarray(vals, dtype=float)
    se = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else float("nan")
    return {"mean": _r(a.mean()), "se": _r(se), "n": int(len(a))}


# ---------------------------------------------------------------- tasks (one record each)

def dev_cluster_task(rho: float, nf: float, seed: int) -> dict:
    s = make_splits(gen_cfg(rho, nf), seed)
    res = {}
    for lk in LINKAGES:
        for cr in CRITERIA:
            cl = cluster_features(s.train.X, ClusterConfig(linkage=lk, criterion=cr))
            res[f"{lk}/{cr}"] = {"ari": M.adjusted_rand(cl.labels, s.truth.feature_group),
                                 "k": cl.k}
    return _round_tree({"rho": rho, "noise_frac": nf, "seed": seed, "configs": res})


def dev_model_task(model_idx: int, seed: int) -> dict:
    s = make_splits(gen_cfg(**BASE), seed)
    m = fit(s.train.X, s.train.y, MODEL_GRID[model_idx], seed)
    return _round_tree({"model_idx": model_idx, "seed": seed,
                        "valid_r2": r2(s.valid.y, predict(m, s.valid.X))})


def _downstream(train: Dataset, test: Dataset, cols, mcfg, seed) -> float:
    cols = np.asarray(cols, dtype=int)
    m = fit(train.X[:, cols], train.y, mcfg, seed)
    return r2(test.y, predict(m, test.X[:, cols]))


def _downstream_cluster_means(train: Dataset, test: Dataset, mr, k: int, mcfg, seed) -> float:
    """Top-k clusters by score, each fed to the model as the mean of its standardised
    members. Standardisation uses training rows only."""
    order = np.lexsort((np.arange(len(mr.unit_scores)), -np.asarray(mr.unit_scores)))[:k]
    mu, sd = train.X.mean(axis=0), train.X.std(axis=0)

    def feats(X):
        Z = (X - mu) / sd
        return np.column_stack([Z[:, mr.clusters[c]].mean(axis=1) for c in order])

    m = fit(feats(train.X), train.y, mcfg, seed)
    return r2(test.y, predict(m, feats(test.X)))


def dedupe_select(scores: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Plain ranking, but at most one feature per found cluster (its top-scoring one)."""
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    seen, out = set(), []
    for j in order:
        if labels[j] not in seen:
            seen.add(labels[j])
            out.append(int(j))
            if len(out) == k:
                break
    return np.sort(np.asarray(out, dtype=int))


def cell_task(rho: float, nf: float, seed: int, mcfg: ModelConfig, ccfg: ClusterConfig) -> dict:
    s = make_splits(gen_cfg(rho, nf), seed)
    truth = s.truth
    K = truth.n_informative_groups
    out = run_methods(s.train, s.valid, mcfg, ccfg, seed, N_REPEATS, with_drop=is_base(rho, nf))
    scored = score_methods(out, truth, s.train.X, K)
    rec = {"rho": rho, "noise_frac": nf, "seed": seed,
           "p": int(s.train.X.shape[1]), "n_groups": len(truth.group_kind),
           "n_noise_features": int(truth.is_noise_feature.sum()),
           "found_clusters": int(out.clustering.k),
           "ari": M.adjusted_rand(out.clustering.labels, truth.feature_group),
           "valid_r2_all": out.valid_r2, "methods": {}, "downstream": {}}
    for name, row in scored.items():
        rec["methods"][name] = {k: row[k] for k in SCORE_KEYS}
    mda_scores = out.results["MDA"].unit_scores
    dd = dedupe_select(mda_scores, out.clustering.labels, K)
    rec["methods"][DEDUPE] = {"recall_at_k": M.recall_informative(dd, truth),
                              "noise_fdr_at_k": M.noise_fdr(dd, truth)}
    for kk in budgets_for(rho, nf):
        d = {}
        for name, mr in out.results.items():
            d[name] = _downstream(s.train, s.test, M.select(mr.unit_scores, mr.clusters,
                                                            s.train.X, kk), mcfg, seed)
        d[CMEANS] = _downstream_cluster_means(s.train, s.test, out.results["cMDA"], kk, mcfg, seed)
        d[DEDUPE] = _downstream(s.train, s.test, dedupe_select(mda_scores, out.clustering.labels, kk),
                                mcfg, seed)
        rng = np.random.default_rng([seed, 99, kk])
        d["random"] = float(np.mean([_downstream(s.train, s.test,
                                                 rng.choice(rec["p"], kk, replace=False), mcfg, seed)
                                     for _ in range(N_RANDOM)]))
        rec["downstream"][str(kk)] = d
    rec["downstream_all_features"] = _downstream(s.train, s.test, np.arange(rec["p"]), mcfg, seed)
    m_or = fit(s.train.F, s.train.y, mcfg, seed)
    rec["downstream_oracle_factors"] = r2(s.test.y, predict(m_or, s.test.F))
    return _round_tree(rec)


def stability_task(seed: int, mcfg: ModelConfig, ccfg: ClusterConfig) -> dict:
    """Rerun every method on bootstrap resamples of the training rows (base cell)."""
    s = make_splits(gen_cfg(**BASE), seed)
    p = s.train.X.shape[1]
    K = s.truth.n_informative_groups
    names = PLAIN + CLUSTERED
    feats = {m: [] for m in names}
    sels = {m: [] for m in names}
    gsels = {m: [] for m in names}
    n = s.train.X.shape[0]
    for b in range(N_BOOT):
        idx = np.random.default_rng([seed, 500, b]).integers(0, n, n)
        tr = Dataset(X=s.train.X[idx], y=s.train.y[idx], F=s.train.F[idx])
        out = run_methods(tr, s.valid, mcfg, ccfg, seed, N_REPEATS)
        for m in names:
            mr = out.results[m]
            feats[m].append(M.feature_level_scores(mr.unit_scores, mr.clusters, p))
            sel = M.select(mr.unit_scores, mr.clusters, tr.X, K)
            sels[m].append(sel)
            gsels[m].append(np.unique(s.truth.feature_group[sel]))
    return _round_tree({"seed": seed, "methods": {
        m: {"spearman": M.mean_pairwise_spearman(feats[m]),
            "jaccard": M.mean_pairwise_jaccard(sels[m]),
            "group_jaccard": M.mean_pairwise_jaccard(gsels[m])} for m in names}})


# ---------------------------------------------------------------- summaries (pure functions)

def summarize_dev(runs: dict) -> dict:
    grid = {}
    for key in runs["dev_cluster"][0]["configs"]:
        grid[key] = {"ari": mean_se([r["configs"][key]["ari"] for r in runs["dev_cluster"]]),
                     "mean_k": _r(np.mean([r["configs"][key]["k"] for r in runs["dev_cluster"]]), 2)}
    best = max(grid, key=lambda k: (grid[k]["ari"]["mean"], k))
    lk, cr = best.split("/")
    mgrid = []
    for i, mc in enumerate(MODEL_GRID):
        vals = [r["valid_r2"] for r in runs["dev_model"] if r["model_idx"] == i]
        mgrid.append({"config": mc.to_dict(), "valid_r2": mean_se(vals)})
    # one-standard-error rule: the cheapest setting (fewest rounds, then fewest leaves)
    # whose mean validation R^2 is within one s.e. of the best mean
    top = max(range(len(MODEL_GRID)), key=lambda i: mgrid[i]["valid_r2"]["mean"])
    floor = mgrid[top]["valid_r2"]["mean"] - mgrid[top]["valid_r2"]["se"]
    ok = [i for i in range(len(MODEL_GRID)) if mgrid[i]["valid_r2"]["mean"] >= floor]
    chosen = min(ok, key=lambda i: (MODEL_GRID[i].n_estimators, MODEL_GRID[i].num_leaves))
    return _round_tree({
        "dev_seeds": list(DEV_SEEDS),
        "clustering_tasks": len(runs["dev_cluster"]),
        "clustering_grid": grid,
        "chosen_cluster": ClusterConfig(linkage=lk, criterion=cr).to_dict(),
        "model_grid": mgrid,
        "model_rule": "cheapest setting within one s.e. of the best mean validation R^2",
        "best_model_by_mean": MODEL_GRID[top].to_dict(),
        "chosen_model": MODEL_GRID[chosen].to_dict(),
    })


def _agg(recs: list[dict], budgets) -> dict:
    out = {"methods": {}, "downstream": {}}
    names = list(recs[0]["methods"])
    for m in names:
        out["methods"][m] = {key: mean_se([r["methods"][m][key] for r in recs])
                             for key in SCORE_KEYS if key in recs[0]["methods"][m]}
    for kk in budgets:
        out["downstream"][str(kk)] = {n: mean_se([r["downstream"][str(kk)][n] for r in recs])
                                      for n in recs[0]["downstream"][str(kk)]}
    for key in ("downstream_all_features", "downstream_oracle_factors", "ari", "valid_r2_all",
                "found_clusters", "p", "n_groups", "n_noise_features"):
        out[key] = mean_se([r[key] for r in recs])
    return out


def summarize_sweep(runs: dict) -> dict:
    K = k_informative()
    sweep = {"heldout_seeds": list(HELDOUT_SEEDS), "k": K, "r2_latent": R2_LATENT,
             "n_repeats": N_REPEATS, "cells": []}
    for rho in RHOS:
        for nf in NOISE_FRACS:
            rs = [r for r in runs["cells"] if r["rho"] == rho and r["noise_frac"] == nf]
            cell = {"rho": rho, "noise_frac": nf}
            agg = _agg(rs, (K,))
            agg["methods"].pop(DROP, None)   # drop-column is reported at the base cell only
            agg["downstream"][str(K)].pop(DROP, None)
            cell.update(agg)
            sweep["cells"].append(cell)
    return _round_tree(sweep)


def summarize_base(runs: dict) -> dict:
    K = k_informative()
    rs = [r for r in runs["cells"] if is_base(r["rho"], r["noise_frac"])]
    base = {"cell": BASE, "heldout_seeds": list(HELDOUT_SEEDS), "k": K,
            "budgets": list(budgets_for(**BASE)), "n_boot": N_BOOT, "n_random": N_RANDOM,
            "n_repeats": N_REPEATS, "generator": gen_cfg(**BASE).to_dict()}
    base.update(_agg(rs, budgets_for(**BASE)))
    st = runs["stability"]
    base["stability"] = {m: {key: mean_se([s["methods"][m][key] for s in st]) for key in STAB_KEYS}
                         for m in st[0]["methods"]}
    # paired comparisons at k = K (same seed, same rows): difference in test R^2
    kk = str(K)

    def ds(r, name):
        return r["downstream_all_features"] if name == "all features" else r["downstream"][kk][name]

    pairs = [(CMEANS, "all features"), (CMEANS, "MDI"), (CMEANS, "SHAP"), ("cMDA", "MDA"),
             ("MDI", "cMDI"), (DEDUPE, "MDA")]
    base["paired_test_r2_at_k"] = []
    for a, b in pairs:
        d = [ds(r, a) - ds(r, b) for r in rs]
        base["paired_test_r2_at_k"].append({"a": a, "b": b, "diff": mean_se(d),
                                            "a_wins": int(sum(x > 0 for x in d)), "n": len(d)})
    base["per_seed"] = [{"seed": r["seed"], "p": r["p"], "found_clusters": r["found_clusters"],
                         "ari": r["ari"]} for r in rs]
    return _round_tree(base)


SUMMARIZERS = {"dev_choices": summarize_dev, "sweep": summarize_sweep, "base": summarize_base}


# ---------------------------------------------------------------- driver

def frozen_choices(dev: dict) -> tuple[ModelConfig, ClusterConfig]:
    return ModelConfig(**dev["chosen_model"]), ClusterConfig(**dev["chosen_cluster"])


def run(out: Path, n_jobs: int = -1, log=print) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    par = Parallel(n_jobs=n_jobs)
    log("development seeds: clustering rule and model settings")
    runs = {"dev_cluster": par(delayed(dev_cluster_task)(rho, nf, sd)
                               for rho in RHOS for nf in NOISE_FRACS for sd in DEV_SEEDS),
            "dev_model": par(delayed(dev_model_task)(i, sd)
                             for i in range(len(MODEL_GRID)) for sd in DEV_SEEDS)}
    dev = summarize_dev(runs)
    mcfg, ccfg = frozen_choices(dev)
    log(f"  chosen: {ccfg}; {mcfg}")
    log("held-out seeds: sweep and base cell (run once)")
    runs["cells"] = par(delayed(cell_task)(rho, nf, sd, mcfg, ccfg)
                        for rho in RHOS for nf in NOISE_FRACS for sd in HELDOUT_SEEDS)
    log("held-out seeds: bootstrap stability at the base cell")
    runs["stability"] = par(delayed(stability_task)(sd, mcfg, ccfg) for sd in HELDOUT_SEEDS)
    _write_json(out / "runs.json", runs)
    for name, fn in SUMMARIZERS.items():
        _write_json(out / f"{name}.json", fn(runs))
    (out / "RESULTS.md").write_text(render(out), encoding="utf-8", newline="\n")
    log(f"wrote {out}")


# ---------------------------------------------------------------- report

ORDER = list(PLAIN) + [DROP] + list(CLUSTERED) + [DEDUPE]


def _f(d: dict | None, nd: int = 2, se: bool = True) -> str:
    if d is None:
        return "-"
    return f"{d['mean']:.{nd}f} ± {d['se']:.{nd}f}" if se else f"{d['mean']:.{nd}f}"


def render(out: Path) -> str:
    out = Path(out)
    dev = json.loads((out / "dev_choices.json").read_text(encoding="utf-8"))
    sweep = json.loads((out / "sweep.json").read_text(encoding="utf-8"))
    base = json.loads((out / "base.json").read_text(encoding="utf-8"))
    K = base["k"]
    n_seeds = len(base["heldout_seeds"])
    L = ["# Results", "",
         "Synthetic data only, generated by `make bench` (`python -m clusterfi.cli bench --out results`).",
         "`±` is one standard error across seeds: sample sd / sqrt(n), treating seeds as independent "
         "datasets (they are: each seed draws its own layout and rows). n is given per table.", ""]
    L += [f"## Development choices (seeds {', '.join(map(str, dev['dev_seeds']))})", "",
          f"Clustering rule, scored by adjusted Rand index (ARI) against the true groups over "
          f"{dev['clustering_tasks']} development datasets (9 sweep cells x 5 seeds; n = "
          f"{dev['clustering_tasks']} per row):", "",
          "| linkage / k-score | ARI | mean clusters found |", "|---|---|---|"]
    for key, v in sorted(dev["clustering_grid"].items(), key=lambda kv: (-kv[1]["ari"]["mean"], kv[0])):
        L.append(f"| {key} | {_f(v['ari'], 3)} | {v['mean_k']:.1f} |")
    cc = dev["chosen_cluster"]
    L += ["", f"Chosen: **{cc['linkage']} / {cc['criterion']}** (highest mean ARI).", "",
          f"LightGBM settings, scored by validation R^2 at the base cell (n = {len(dev['dev_seeds'])} "
          f"seeds per row):", "",
          "| leaves | rounds | learning rate | validation R^2 |", "|---|---|---|---|"]
    for g in dev["model_grid"]:
        c = g["config"]
        L.append(f"| {c['num_leaves']} | {c['n_estimators']} | {c['learning_rate']} | "
                 f"{_f(g['valid_r2'], 4)} |")
    cm = dev["chosen_model"]
    L += ["", f"Rule: {dev['model_rule']}. Chosen: {cm['num_leaves']} leaves, {cm['n_estimators']} "
              f"rounds, learning rate {cm['learning_rate']}.", ""]

    L += [f"## Base cell: rho = {base['cell']['rho']}, noise share = {base['cell']['noise_frac']} "
          f"(held-out seeds, n = {n_seeds} per row)", "",
          f"Per dataset: {_f(base['p'], 1)} features, of which {_f(base['n_noise_features'], 1)} noise; "
          f"{_f(base['n_groups'], 1)} true groups; {_f(base['found_clusters'], 1)} clusters found; "
          f"clustering ARI {_f(base['ari'], 3)}; validation R^2 with all features "
          f"{_f(base['valid_r2_all'], 3)}. k = {K}, the number of informative groups.", "",
          "### Against the ground truth", "",
          "| method | group rank corr | feature rank corr | recall of informative groups @k | "
          "noise share of selected @k |", "|---|---|---|---|---|"]
    for m in ORDER:
        r = base["methods"].get(m)
        if r is not None:
            L.append(f"| {m} | {_f(r.get('rank_corr'))} | {_f(r.get('feature_rank_corr'))} | "
                     f"{_f(r['recall_at_k'])} | {_f(r['noise_fdr_at_k'])} |")
    L += ["", f"### Stability over {base['n_boot']} bootstrap resamples of the training rows "
              f"(mean over pairs, then over seeds)", "",
          "| method | feature-score Spearman | selected features Jaccard @k | selected groups Jaccard @k |",
          "|---|---|---|---|"]
    for m in ORDER:
        st = base["stability"].get(m)
        if st:
            L.append(f"| {m} | {_f(st['spearman'])} | {_f(st['jaccard'])} | {_f(st['group_jaccard'])} |")
    L += ["", "### Downstream: refit on the selected features, score once on 4,000 untouched test rows",
          "", "Test R^2.", "",
          "| selection | " + " | ".join(f"k = {b}" for b in base["budgets"]) + " |",
          "|---|" + "---|" * len(base["budgets"])]
    for m in ORDER + [CMEANS, "random"]:
        if m in base["downstream"][str(base["budgets"][0])]:
            L.append(f"| {m} | " + " | ".join(_f(base["downstream"][str(b)][m], 3)
                                              for b in base["budgets"]) + " |")
    L += [f"| all features (no selection) | {_f(base['downstream_all_features'], 3)} | |",
          f"| reference: the true latent factors | {_f(base['downstream_oracle_factors'], 3)} | |", "",
          f"The population R^2 of the true factors is {sweep['r2_latent']}; LightGBM on the factors "
          "is a reference, not a ceiling for this learner.", "",
          f"Paired differences in test R^2 at k = {K} (same seed and rows; s.e. of the paired "
          f"difference across n = {n_seeds} seeds):", "",
          "| a | b | a - b | a better in |", "|---|---|---|---|"]
    for pr in base["paired_test_r2_at_k"]:
        L.append(f"| {pr['a']} | {pr['b']} | {pr['diff']['mean']:+.3f} ± {pr['diff']['se']:.3f} | "
                 f"{pr['a_wins']} of {pr['n']} seeds |")
    L.append("")

    L += ["## Sweep: within-cluster correlation x noise share", "",
          f"Held-out seeds, n = {len(sweep['heldout_seeds'])} per cell, k = {sweep['k']}. Means only; "
          "every standard error is in `results/sweep.json`.", ""]
    cols = [m for m in ORDER if m != DROP]
    for metric, title, nd in (("recall_at_k", "Recall of informative groups @k", 2),
                              ("noise_fdr_at_k", "Noise share of the selected features @k", 2),
                              ("feature_rank_corr", "Feature rank correlation", 2),
                              ("rank_corr", "Group rank correlation (true groups)", 2),
                              ("test_r2", "Downstream test R^2 @k", 3)):
        names = cols + ([CMEANS, "all features"] if metric == "test_r2" else [])
        L += [f"### {title}", "", "| rho | noise | p | " + " | ".join(names) + " |",
              "|---|---|---|" + "---|" * len(names)]
        for c in sweep["cells"]:
            vals = []
            for m in names:
                if m == "all features":
                    v = c["downstream_all_features"]
                elif metric == "test_r2":
                    v = c["downstream"][str(sweep["k"])].get(m)
                else:
                    v = c["methods"].get(m, {}).get(metric)
                vals.append(_f(v, nd, se=False))
            L.append(f"| {c['rho']} | {c['noise_frac']} | {c['p']['mean']:.0f} | " + " | ".join(vals) + " |")
        L.append("")
    scale = out / "scale.json"
    if scale.exists():
        sc = json.loads(scale.read_text(encoding="utf-8"))
        L += ["## Parallel permutation importance (machine-specific, never checked by tests)", "",
              f"{sc['machine']['summary']}. p = {sc['p']} features, {sc['n_valid']} validation rows, "
              f"{sc['n_repeats']} repeats, best of {sc['passes']} passes, joblib backend {sc['backend']}.",
              "", "| workers | wall time (s) | speedup | efficiency | identical to serial |",
              "|---|---|---|---|---|"]
        for row in sc["rows"]:
            L.append(f"| {row['workers']} | {row['seconds']:.2f} | {row['speedup']:.2f}x | "
                     f"{row['efficiency']:.2f} | {row['identical_to_serial']} |")
        L.append("")
    return "\n".join(L)
