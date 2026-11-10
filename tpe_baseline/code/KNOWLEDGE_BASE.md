# Knowledge Base: Meta-Features & Online Phase

This document describes the 28 meta-features and the knowledge-base building blocks implemented
in `meta_feature_extraction.py` and `prepare_knowledge_base.py`: what each function does, the
`knowledge_base.json` schema, the full CLI reference for both scripts, and the pset-compatibility
caveat that governs warmstart/recommendation behavior.

## 0. File Overview

### `meta_feature_extraction.py`
Computes the 28 meta-features for a dataset and drives the online phase (warmstart GA search, or
pure meta-learning recommendation) against a knowledge base. Public functions:

| Function | Purpose |
|---|---|
| `extract_features(data)` | Computes all 28 meta-features (MF1-MF28) for a raw data matrix. |
| `calculate_statisticMeasures(distance)` | MF1-MF5: mean/var/std/skew/kurtosis of the distance distribution. |
| `calculate_histogram_measures(distance)` | MF6-MF15: 10-bin histogram of the distance distribution. |
| `calculate_zscoreMeasures(distance)` | MF16-MF19: share of distances per absolute z-score bin. |
| `calculate_landmarkers(data)` | MF20-MF28: MeanShift/DBSCAN/OPTICS x CHI/DBI/SC landmarker scores. |
| `fix_dbi_errors_statistically(d_new, kb)` | Replaces 1e9 DBI error placeholders with a statistically derived penalty before distance comparison. |
| `get_rankedDistances_for_dnew_from_kb(d_new, kb, m)` | Ranks knowledge-base datasets by Euclidean distance over meta-features; returns the `m` closest. |
| `load_knowledge_base(path)` | Loads a consolidated `knowledge_base.json` into `(meta_features_by_dataset, hofs_by_dataset)`. |
| `get_PipleneInstantiations_for_dnew(hofs, rankedDistances, pset=None)` | Builds the warmstart population (one pipeline per ranked dataset), supporting both legacy in-memory hofs and knowledge_base string-pipeline hofs. |
| `get_recommended_pipelines(hofs, rankedDistances, pset, k=10)` | Pure meta-learning: collects up to `k` unique candidate pipelines from the knowledge base, no GA. |
| `evaluate_pipeline_candidates(individuals, pset, data, measure, is_offline, target=None)` | One-shot evaluation (no evolution) of candidate pipelines via `autoclust.evaluatePipeline`, ranked by score. |
| `run_online_phase_for_dataset(dataset_path, kb_metafeatures, kb_hofs, args)` | Runs the full online phase (ranking + `warmstart` GA or `recommend-only` evaluation) for one d_new CSV against an already-loaded knowledge base. Returns a JSON-serializable result dict (see §4.4); raises `ValueError` on validation problems (eg. missing target column for `--eval-measure ari`). Shared by both single-dataset and `--input-folder` batch mode. |
| `resolve_input_csv_paths(input_folder)` | Recursively collects all `*.csv` paths under `input_folder`, sorted — used by `--input-folder` batch mode. |

Running the file directly (`__main__`) drives the online phase end-to-end — see §5 for the CLI.

### `prepare_knowledge_base.py`
One-off consolidation script: turns the per-dataset offline-phase results in `result_ll_kb/*.json`
plus their source CSVs into a single `knowledge_base.json` that `meta_feature_extraction.py` can
load via `--knowledge-base`. Public functions:

| Function | Purpose |
|---|---|
| `build_knowledge_base(kb_results_dir, csv_dir, target_column="target", exclude_dirs=None)` | Resolves each `result_ll_kb/*.json` entry's CSV by filename, skips datasets reserved for online-phase evaluation (see §3.1), computes the remaining ones' meta-features via `extract_features`, and merges them with the existing `hall_of_fame`. Returns the consolidated dict. |
| `collect_excluded_dataset_names(exclude_dirs)` | Globs `*.csv` in each directory of `exclude_dirs` and returns the set of basenames to keep out of the knowledge base. |
| `main()` | Argparse entrypoint; writes the result of `build_knowledge_base` to `--output-path`. |

## 1. Meta-Features (MF1-MF28)

All meta-features are computed by `extract_features(data)` and are based on the pairwise
Euclidean distance matrix of `data` (min-max normalized to `[0, 1]`), except for the landmarkers
(MF20-MF28), which operate directly on the raw data. The full, canonical list of feature names is
in `meta_features.txt`.

### MF1-MF5: Statistical measures of the distance distribution (`calculate_statisticMeasures`)
Mean, variance, standard deviation, skew and kurtosis of the normalized pairwise distances.
Describe the shape and spread of the distance distribution.

### MF6-MF15: Histogram of the distance distribution (`calculate_histogram_measures`)
10 bins (each 10% of `[0, 1]`) over the normalized distances, as relative frequencies. Gives a
coarser, bin-based picture of the distance distribution than the pure moments MF1-5.

### MF16-MF19: Z-score bins (`calculate_zscoreMeasures`)
Share of (unnormalized) pairwise distances with absolute z-score in `[0,1)`, `[1,2)`, `[2,3)`,
`[3,∞)`. Measures how many point pairs are "unusually" far apart (an indicator for
outliers/cluster separation).

### MF20-MF28: Landmarkers (`calculate_landmarkers`)
Three simple, untuned clustering algorithms (MeanShift, DBSCAN, OPTICS) are run directly on
`data`; for each algorithm three CVIs are collected: Calinski-Harabasz (CHI), Davies-Bouldin
(DBI), Silhouette (SC). If n_cluster <= 1 or an exception occurs, the worst-case values defined in
`getErrorVaule` are used instead.

Before comparing the distance between two feature vectors, the DBI columns (MF21/MF24/MF27) are
corrected via `fix_dbi_errors_statistically`: 1e9 error values (from `getErrorVaule`) would
otherwise massively distort the Euclidean distance; they are replaced by
`max(real_dbi) + 2*std(real_dbi)` over all actually computed DBI values in d_new + knowledge_base
— a "large but not absurd" penalty value instead of 1e9. All other features are carried into the
distance comparison unscaled.

## 2. `knowledge_base.json` Schema

Produced by `prepare_knowledge_base.py` from `result_ll_kb/*.json` (offline-phase results,
pipelines stored as strings via `str(individual)`) plus the matching CSVs in
`benchmark_csv/offline_phase/` (matched by the `dataset_name` filename, NOT the `dataset_path`
stored in the JSONs, which points to a remote machine).

```json
{
  "blobs.csv": {
    "n_features": 2,
    "meta_features": {"MF1_mean": 0.42, "...": "...", "MF28_OPTICS_SC": 0.1},
    "hall_of_fame": [
      {"rank": 1, "ari": 0.912, "pipeline": "MeanShift(matrix, bool_False, bool_True, max_it_66, band_0_9)"}
    ],
    "best_ari": 0.912
  }
}
```

Pipelines are deliberately kept as plain strings (no pickled DEAP objects) — otherwise the exact
same `creator.FitnessMulti`/`creator.Individual` classes used at write time would need to be
registered identically at load time, which is fragile across DEAP/Python versions. Instead,
pipeline strings are only parsed against d_new's `pset` at actual point of use (warmstart or
recommendation), via `gp.PrimitiveTree.from_string` (resp. `creator.Individual.from_string`).

### Datasets without an entry
CSVs in `--csv-dir` that have no matching JSON in `--kb-results-dir` are ignored. Result JSONs
without a matching local CSV are skipped with a warning rather than treated as an error.

## 3. `prepare_knowledge_base.py` CLI Reference

```bash
python3 code/prepare_knowledge_base.py \
  --kb-results-dir result_ll_kb \
  --csv-dir benchmark_csv/offline_phase \
  --output-path knowledge_base.json
```

| Flag | Default | Description |
|---|---|---|
| `--kb-results-dir` | `result_ll_kb` | Folder with the per-dataset JSON results of the offline phase. |
| `--csv-dir` | `benchmark_csv/offline_phase` | Folder with the CSV datasets matching the result files (matched by `dataset_name` filename). |
| `--target-column` | `target` | Name of the target column to drop before meta-feature computation. |
| `--output-path` | `knowledge_base.json` | Path of the consolidated knowledge_base JSON file to write. |
| `--exclude-dir` | (repeatable) | Additional folder whose CSV filenames must be excluded from the knowledge base. Adds to, rather than replaces, the defaults below. |
| `--no-default-excludes` | `False` | Disables the built-in default excludes (see §3.1), so only directories explicitly passed via `--exclude-dir` are excluded. |

### 3.1 Excluding held-out online-phase datasets

Datasets reserved as candidate d_new evaluation data — i.e. CSVs in `benchmark_csv/online_phase/`
and `online_phase_prep4clust/` — are excluded from the knowledge base **by default**, even if they
also happen to have a `result_ll_kb` entry. This matters concretely: `dermatology.csv` and
`wdbc.csv` exist both as `result_ll_kb` entries (with real offline-phase GA results) and in
`online_phase_prep4clust/`. Without exclusion, evaluating `online_phase_prep4clust/dermatology.csv`
as d_new against the knowledge base would find the literal same dataset at distance 0 in the
ranking and trivially warmstart from its own best pipeline — defeating the point of held-out
evaluation.

Exclusion is filename-based: a dataset is excluded if its basename matches a `*.csv` file found in
any of the exclude directories, regardless of which folder its own CSV is actually read from via
`--csv-dir`. Use `--exclude-dir <folder>` (repeatable) to exclude further folders, or
`--no-default-excludes` to fall back to the old, fully-inclusive behavior.

## 4. `meta_feature_extraction.py` CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--knowledge-base` | `None` | Path to a `knowledge_base.json` produced by `prepare_knowledge_base.py`. If omitted, the script runs the original `data/*.csv` demo unchanged (see §4.3) and all other flags below are ignored. |
| `--dataset-path` | `None` | Path to a single d_new CSV. Mutually exclusive with `--input-folder`. One of the two is required when `--knowledge-base` is set. |
| `--input-folder` | `None` | Folder of d_new CSVs to evaluate in batch (searched recursively). Mutually exclusive with `--dataset-path`. See §4.4. |
| `--output-dir` | `online_phase_results` | Folder for per-dataset JSON result files. Only used with `--input-folder`. |
| `--target-column` | `target` | Name of the target column in d_new, if present. Needed for `--eval-measure ari`; otherwise optional. |
| `--m` | `50` | Number of nearest knowledge-base datasets to consider for ranking. |
| `--mode` | `warmstart` | `warmstart` or `recommend-only` (see §4.1/§4.2). |
| `--k` | `10` | Number of recommended pipelines to evaluate in `recommend-only` mode. |
| `--eval-measure` | `ari` | `ari`, `silhouette`, `calinski_harabasz`, or `davies_bouldin` — only used in `recommend-only` mode. |
| `--time-limit-online` | `90` | Time budget per GA run (seconds) — only used in `warmstart` mode. |
| `--max-tree-height` | `6` | Maximum height of generated GP pipeline trees. |
| `--output-path` | `None` | Only used with `--dataset-path` (single-dataset mode). If set, additionally writes the result (cluster labels in `warmstart` mode, ranked pipeline table in `recommend-only` mode) to this CSV path. |

### 4.1 Warmstart mode (default)

The GA search continues from the most similar known pipelines as the starting population
(identical to the previous online-phase behavior, except that the starting population now comes
from the knowledge base instead of a freshly, in-memory computed `hofs` dict):

```bash
python3 code/meta_feature_extraction.py \
  --knowledge-base knowledge_base.json \
  --dataset-path benchmark_csv/online_phase/<dataset>.csv \
  --mode warmstart
```

### 4.2 Pure meta-learning mode (`recommend-only`)

No GA evolution: the `--k` most similar pipelines compatible with d_new are pulled from the
knowledge base and evaluated once directly against `--eval-measure`, then printed sorted by
score. Useful for quickly auditing what the meta-learner alone would suggest, or for benchmarking
recommendation quality against `--eval-measure ari` on labeled datasets (requires a
`--target-column` column in `--dataset-path` for that):

```bash
python3 code/meta_feature_extraction.py \
  --knowledge-base knowledge_base.json \
  --dataset-path benchmark_csv/offline_phase/<labeled-dataset>.csv \
  --mode recommend-only --k 10 --eval-measure ari
```

### 4.3 Without `--knowledge-base`: original behavior

If `--knowledge-base` is omitted, `meta_feature_extraction.py` runs exactly as before: it iterates
over `data/*.csv`, computes meta-features and a fresh offline-phase run per dataset in-memory,
picks the first CSV as d_new, and runs the online phase on it. All other flags (`--mode`, `--k`,
`--eval-measure`, `--m`, ...) are ignored in this case.

### 4.4 Batch evaluation (`--input-folder`)

Evaluates every CSV under a folder (searched recursively) as its own d_new, one
`--mode warmstart`/`recommend-only` run each, against the same already-loaded knowledge base:

```bash
python3 code/meta_feature_extraction.py \
  --knowledge-base knowledge_base.json \
  --input-folder online_phase_prep4clust \
  --output-dir online_phase_results \
  --mode recommend-only --eval-measure silhouette
```

`online_phase_prep4clust/` ships as headerless CSVs (raw numeric rows, label as the last column, no
`target` column name) — run `python add_csv_headers.py` (repo root) once before using it as
`--input-folder`/`--dataset-path`; see the top-level `README.md` for details.

Each dataset's result is written to `<output-dir>/<dataset_stem>.json` **immediately after that
dataset finishes** (not batched and written only at the end), using the same per-dataset
JSON-result helpers (`dataset_output_path`, `write_result_to_path`, filename-collision handling)
that `autoclust.py`'s own `--input-folder` offline-phase mode already uses. Two consequences:

- If the run is interrupted (crash, Ctrl-C, kill) partway through, every dataset that already
  finished has its result file safely on disk.
- Re-running the exact same command afterwards **skips (and logs) any dataset whose result file
  already exists**, so the batch run effectively resumes rather than redoing finished work.
- A dataset that raises an error (bad CSV, GA failure, missing target column for
  `--eval-measure ari`, ...) is logged and skipped — no output file is written for it, so it is
  retried (not silently treated as done) on the next run, and the rest of the batch keeps going.

Result file schema, one JSON object per dataset:

```jsonc
// --mode warmstart
{
  "dataset_name": "iris.csv", "dataset_path": "/abs/path/iris.csv", "mode": "warmstart",
  "n_features": 4, "ranked_datasets": [{"dataset": "...", "distance": 0.12}, ...],
  "warmstart_population_size": 42, "labels": [0, 1, 2, 0, ...]
}
// --mode recommend-only
{
  "dataset_name": "iris.csv", "dataset_path": "/abs/path/iris.csv", "mode": "recommend-only",
  "n_features": 4, "ranked_datasets": [...], "eval_measure": "ari", "k": 10,
  "results": [{"rank": 1, "pipeline": "...", "score": 0.83, "length": 2}, ...]
}
```

`warmstart` mode runs 3 full GA searches per dataset (one per CVI), so a batch over many datasets
can take a long time; `recommend-only` mode runs no GA at all and is much faster for sweeping over
a whole folder.

## 5. pset Compatibility (Warmstart/Recommendation)

The `pset` depends on d_new's feature count: PCA terminals are only registered for
`n_components_2` through `n_components_{n_features}` (`createPset`). A pipeline stored in the
knowledge base that references e.g. `n_components_15` cannot be parsed against a d_new with only
10 features (`TypeError` in `gp.PrimitiveTree.from_string`). This is expected behavior: that
pipeline string is skipped (warmstart: next pipeline/dataset; recommendation: next entry), not
treated as an error. In warmstart mode, `runGeneticAlgorithm` fills any resulting shortfall in the
starting population with random individuals anyway.
