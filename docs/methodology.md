# Methodology

The rules behind every number in the README, and what was decided where.

## Data

`data.py` draws a fresh layout per seed, then independent train (2,000), validation (1,000) and test (4,000) rows from that layout.

- **Latent factors.** Eight independent standard normals: six enter linearly with weights 1.0, 0.85, 0.7, 0.55, 0.4, 0.25; two enter only through their product (weight 1.0) and have no linear correlation with y.
- **Target.** `y = sum_k beta_k f_k + f_7 f_8 + sigma e`, with sigma set so the factors explain 25% of Var(y) in the population (`r2_latent = 0.25`).
- **Copies.** Each factor gets between 2 and 12 copies (drawn per seed, unrelated to importance): `x = sqrt(rho) f + sqrt(1 - rho) e`, so two copies of a factor correlate at rho.
- **Redundant combinations.** Eight features `sqrt(rho) (0.8 f_k + 0.6 f_l) + noise`, pairing each factor with the next.
- **Noise.** Enough noise features to make up the requested share of all features. Half of them come in clusters (2 to 12 copies of a latent noise factor, same rho), half are isolated.
- **Column order** is shuffled, so position carries nothing.

### Ground truth

Truth lives on **groups**: one per informative factor (its copies plus the combinations whose dominant weight, 0.8, is on it), one per noise cluster, one per isolated noise feature. A group's true importance is its share of Var(y): `beta_k^2 / Var(y)` for a linear factor and `(gamma^2 / 2) / Var(y)` for each factor of the interaction. The factors are independent and zero-mean, so these terms are uncorrelated and the split is exact (for the interaction it is the symmetric Shapley split). Noise groups have importance 0. The informative shares sum to 0.25 (`test_group_importance_matches_variance_decomposition`).

Assigning a combination to its dominant factor's group is a judgement call: the 0.6 part also carries information about the other factor.

## Model

One LightGBM regressor for every method: `deterministic=True`, one thread, fixed seed, feature fraction 0.5, L2 1.0, 20 rows per leaf minimum. Leaves and rounds were chosen on development seeds (below).

## Methods

| method | unit | definition |
|---|---|---|
| MDI | feature | total split gain in the model, normalised to sum to 1 |
| MDA | feature | mean drop in validation R^2 over 3 shuffles of the column |
| SHAP | feature | mean absolute TreeSHAP value on validation rows (`pred_contrib=True`) |
| drop-column | feature | validation R^2 of the full model minus that of a model refitted without the feature |
| cMDI | cluster | sum of members' MDI |
| cMDA | cluster | mean drop in validation R^2 over 3 joint shuffles of all members (one row permutation per shuffle) |
| cSHAP | cluster | mean absolute value of the sum of members' SHAP values (SHAP is additive, so this is the cluster's SHAP value) |

Clusters come from the training rows only, without the target: correlation distance `d = sqrt((1 - rho) / 2)`, hierarchical linkage, and the number of clusters chosen per dataset by scanning k = 2 .. p/2 and keeping the best silhouette score.

## Development vs held-out

| | seeds | used for |
|---|---|---|
| development | 1-5 | every choice: generator settings, the clustering rule, the model settings, the selection rule, the metrics |
| held-out | 101-108 | the reported tables, run once with the frozen choices |

Two choices are made by code from development records, so they are reproducible (`results/dev_choices.json`):

- **Clustering rule.** 4 linkages x 2 silhouette scores, scored by adjusted Rand index (ARI) against the true groups over the 9 sweep cells x 5 development seeds. Highest mean ARI wins (single linkage, mean silhouette). Seven of the eight rules scored 0.95 or more, so the choice matters little on this generator.
- **Model settings.** 3 leaf counts x 2 (rounds, learning rate) pairs, scored by validation R^2 at the base cell. The rule is the cheapest setting within one standard error of the best mean: the best mean was 7 leaves / 300 rounds (0.1570 ± 0.0045), and 7 leaves / 100 rounds (0.1548) is within one s.e., so it was chosen. This keeps drop-column and the bootstrap affordable.

Made by hand on development seeds while building, before any held-out run:

- `r2_latent = 0.25` (first tried 0.5, where every method finds the signal easily), copies varying 2 to 12 per factor (first tried a fixed 5; real feature sets have unequal numbers of variants per idea), 3 permutation repeats, k = 8.
- The cluster-means selection variant was added after development runs showed that one representative per cluster loses to the plain top-k picks downstream (the plain picks average several noisy copies of the strongest factors). The variant was defined and frozen before the held-out run.
- The one-standard-error model rule, for run time.

**History of the held-out run.** The first held-out run crashed in the summary step (a wrong keyword argument in the base-cell summary). By then `runs.json`, `dev_choices.json` and `sweep.json` had been written; the console showed only progress lines and the chosen settings, and I did not open those files before the rerun. The fix touched only that call, and the identical, deterministic run was repeated. After the held-out results existed, two things were added that compute nothing new from data: the paired-difference table in `base.json` (a summary of the existing per-seed records) and the build number in the machine info of `scale.json` (`make scale` was rerun for it). No held-out result was used to change a method, a setting or a seed.

## Metrics

All at k = 8, the number of informative groups (known here only because the data is synthetic).

- **Group rank correlation.** Spearman between estimated and true importance over the 8 informative groups. With only 8 points, and one tie in the truth (the two interaction factors share an importance), the statistic is coarse. A method's group score is the sum of its per-feature scores over the **true** group; a cluster's score is split equally among its members first. Summing over the true groups is generous to the plain methods, which in practice would not know the groups.
- **Feature rank correlation.** Spearman over all features between a feature's score and the true importance of its group (noise = 0). Plain methods: the feature's own score. Clustered methods: the cluster's score, undivided, since keeping any member keeps the information. This is the score an analyst reads when pruning, and it is low when credit is diluted across copies.
- **Selection.** Plain: the k highest-scoring features. Clustered: walk clusters from the highest score down and take each cluster's most central member (highest mean absolute correlation with the rest). `MDA+dedupe` ranks by plain MDA but takes at most one feature per found cluster.
- **Recall of informative groups @k.** Share of the 8 informative groups with at least one selected feature.
- **Noise share @k.** Share of the k selected features that belong to a noise group (the false-discovery proportion for noise).
- **Stability.** 4 bootstrap resamples of the training rows per seed; the whole pipeline (model, clustering, importance) is rerun on each. Mean pairwise Spearman of feature scores, mean pairwise Jaccard of the selected features, and of the true groups they cover.
- **Downstream.** Refit the same LightGBM settings on the selected features (training rows), score once on the 4,000 test rows, which no other step touches. `cMDA, cluster means` feeds the top-k clusters as k inputs, each the mean of its standardised members (means and standard deviations from training rows). Baselines: 3 random draws of k features (averaged), all features, and LightGBM on the true factors.

## Standard errors

Each seed is an independent dataset (own layout and rows), so `±` is the sample standard deviation across seeds divided by sqrt(n), with n = 8 held-out seeds (5 development seeds in the development tables). Paired differences use the per-seed differences. Within-seed noise (bootstrap, permutation) is not added separately; it is part of the across-seed spread.

## Leakage controls

| risk | control | where it is tested |
|---|---|---|
| choices fitted on reported data | all choices on seeds 1-5; held-out seeds 101-108 run once | `test_heldout_and_development_seeds_are_disjoint`, `test_library_defaults_are_the_frozen_development_choices` |
| importance computed on training rows | MDA, cMDA, SHAP and drop-column are scored on validation rows; MDI comes from the fitted model | `pipeline.run_methods` |
| downstream score reuses selection data | test rows are used only for the final R^2 | `bench.cell_task` |
| clustering sees the target or test rows | clustering uses training features only | `pipeline.run_methods` |
| standardisation leaks test statistics | cluster means use training means and standard deviations | `bench._downstream_cluster_means` |

The data is i.i.d., so there is no time ordering to respect. On a time series, every split here would have to be by time, with purging and an embargo.

## Reproducibility

`results/runs.json` holds one record per task. `dev_choices.json`, `sweep.json` and `base.json` are pure functions of it, and `RESULTS.md` is rendered from them; `pytest` checks all of this. It also reruns a fixed subset of tasks from scratch (one held-out seed in every cell, including drop-column at the base cell, one stability seed, three development clustering tasks and all development model fits) and compares them with their records to 1e-6. `make verify` (`FULL=1`) reruns every task. I ran the full rerun in a `python:3.11-slim` Linux container with the versions in `constraints.txt`.

## Scale-out

Permutation importance is split into chunks of features and run on a joblib backend (loky processes by default), with LightGBM pinned to one thread per worker. Each (seed, unit, repeat) has its own random generator, so the parallel result equals the serial one bit for bit (`test_parallel_permutation_is_identical_to_serial`, and the `identical_to_serial` column of `scale.json`). The worker pool is started before timing. Speedups are machine-specific and never asserted.
