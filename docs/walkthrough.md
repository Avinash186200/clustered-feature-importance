# Walkthrough

A guided tour, from one dataset to the full benchmark.

## 1. Install and run the tests

```bash
pip install -c constraints.txt -e ".[dev]"
pytest -q            # about a minute on a laptop; a few minutes on a 2-vCPU CI runner
```

## 2. One dataset, every method

```bash
python -m clusterfi.cli demo --seed 1
```

This draws one development dataset at the base setting (within-cluster correlation 0.7, half the features noise), fits the model, clusters the features and prints, per method, the group rank correlation, the feature rank correlation, the recall of informative groups at k and the noise share at k. Try `--rho 0.9` for tighter clusters or `--noise-frac 0.75` for more noise.

## 3. Use it on your own matrix

```python
import numpy as np
from clusterfi.clustering import ClusterConfig, cluster_features
from clusterfi.model import ModelConfig, fit
from clusterfi import importance as imp

# X_tr, y_tr, X_va, y_va: your training and validation rows (validation later in time
# than training if the data is a time series)
model = fit(X_tr, y_tr, ModelConfig(), seed=0)
cl = cluster_features(X_tr, ClusterConfig())          # no target, training rows only
scores = imp.clustered_mda(model, X_va, y_va, cl.clusters, n_repeats=5, seed=0, n_jobs=4)
for c in np.argsort(-scores)[:10]:
    print(c, round(scores[c], 4), cl.clusters[c])
```

`imp.clustered_mdi` and `imp.clustered_shap` take the same clusters. To feed the model with clusters instead of single features, average each selected cluster's standardised members, as `bench._downstream_cluster_means` does.

## 4. Read the benchmark

`make bench` (a minute or two on a many-core laptop) does two things in order:

1. **Development seeds 1-5**: scores 8 clustering rules by ARI against the true groups and 6 LightGBM settings by validation R^2, and freezes the winners (`results/dev_choices.json`).
2. **Held-out seeds 101-108**, once: a 3 x 3 sweep of correlation strength x noise share, plus drop-column, a second budget and bootstrap stability at the base cell.

Every task writes a record to `results/runs.json`; the summaries and `RESULTS.md` are computed from those records. `make verify` reruns every task and checks it against its record.

## 5. Time the parallel path

```bash
make scale     # writes results/scale.json and re-renders RESULTS.md
```

Expect different numbers on your machine. Efficiency falls as workers are added. I have not profiled why; likely causes: each worker gets a copy of the model and the validation rows, the per-chunk work is small, and on a laptop the extra workers land on slower efficiency cores and share one thermal budget.

## 6. Going bigger

- **More features.** The O(p^2) correlation matrix becomes the cost before the permutations do: at 20,000 features it is 400 million entries, 3.2 GB in float64. Compute it in blocks of standardised columns (`Z.T @ Z / n`), or skip the full matrix and cluster within candidate blocks.
- **More machines.** `permutation_importance`, `mda` and `clustered_mda` take any joblib backend name (`backend=...`), so a Dask or Ray cluster can be plugged in through their joblib integrations. I have not run or benchmarked that here, and CI does not cover it.
