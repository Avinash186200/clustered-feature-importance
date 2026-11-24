# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-24

### Added

- Synthetic wide-panel generator with known ground truth: informative latent factors (six linear, two acting only through their product), each spawning a cluster of 2 to 12 correlated noisy copies, redundant two-factor combinations, clustered and isolated noise, and knobs for within-cluster correlation, noise share and signal strength. True group importance is the exact split of Var(y).
- Feature clustering by correlation distance `sqrt((1 - rho) / 2)` with hierarchical linkage and the number of clusters chosen per dataset by silhouette (mean or mean / std).
- Importance methods on one deterministic LightGBM model: MDI (split gain), permutation (MDA), TreeSHAP via `pred_contrib`, drop-column, and clustered MDI, clustered MDA and clustered SHAP.
- Metrics against the truth: group and feature rank correlation, recall of informative groups and noise share at k, bootstrap stability (score Spearman, feature and group Jaccard), and a downstream check that refits on the selected features and scores once on untouched test rows, with paired comparisons.
- A deduplicated-ranking ablation and a cluster-means selection variant.
- Benchmark with choices made on development seeds and held-out seeds run once; a sweep over correlation strength and noise share.
- Parallel permutation importance with joblib (loky), identical to serial bit for bit, and a speedup-versus-workers harness with machine info.
- Tests that recompute every summary from the per-task records and rerun a fixed subset of tasks (all of them with `FULL=1`), `constraints.txt` with exact versions, CI on ubuntu (Python 3.11) with a secret scan.
