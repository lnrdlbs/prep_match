# tpe AutoClust Prototype

Python Version: Python 3.12.13

## Installation

```bash
pip install -r requirements.txt
```

## Download Datasets

```bash
python setupData.py
```

For downloading and transforming the benchmark dataset, see `README_convert_clustering_benchmark.md`.

## Usage: `add_csv_headers.py`

Some dataset folders (e.g. `online_phase_prep4clust/`) ship as headerless CSVs — every row, including the first, is raw numeric data with the class label as the last column. All tooling in this repo (`code/autoclust.py`, `code/prepare_knowledge_base.py`, `code/meta_feature_extraction.py`) assumes a header row with a `target` column, so headerless folders need a one-time fix before use. `add_csv_headers.py` prepends a synthetic header (`f0,f1,...,target`) to every headerless CSV in a folder, in place, and is idempotent — files that already have a header are skipped, so it's safe to re-run.

```bash
python add_csv_headers.py --dry-run
```

`--dry-run` previews which files would change without writing anything — recommended first, since the affected CSVs are gitignored (no `*.csv` files are tracked) and therefore not recoverable via git if something goes wrong. Drop `--dry-run` to apply the changes for real:

```bash
python add_csv_headers.py
```

### Flag Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--folder` | path | `online_phase_prep4clust` | Folder of CSVs to add headers to. |
| `--target-column` | str | `target` | Name to use for the last column's header. |
| `--feature-prefix` | str | `f` | Prefix for the generated feature column names (`f0`, `f1`, ...). |
| `--dry-run` | flag | `False` | Only print which files would be changed, without writing anything. |

## Usage: `code/autoclust.py`

`code/autoclust.py` runs the AutoClust genetic algorithm pipeline search on one labeled dataset or on all CSV datasets in a folder. Running it with no arguments reproduces the default behavior (the bundled `data/wine.csv` dataset, a 20 second time budget, and the default GA settings below):

```bash
python code/autoclust.py
```

Run against one explicit dataset and target column:

```bash
python code/autoclust.py --dataset-path data/iris.csv --target-column target
```

Run against all CSV files in a folder:

```bash
python code/autoclust.py --input-folder data --target-column target
```

Folder mode searches recursively for `*.csv` files, sorts them by path, and runs the offline ARI optimization once per dataset. Use `--dataset-path` when you want to force exactly one explicit file.

Store results in an explicit output folder:

```bash
python code/autoclust.py --input-folder data --output-folder results
```

AutoClust writes one JSON file per dataset as soon as that dataset finishes, so completed results remain available for later datasets. With the command above, output files are named after the dataset, for example `results/iris.json` and `results/wine.json`. If the expected result file already exists, that dataset is skipped and the run continues with the next input dataset. If two datasets would produce the same filename, a numeric suffix such as `_2` is appended.

If a dataset hits the `--max-memory-gb` limit, the `--hard-time-limit`, or raises any other unexpected error, it is skipped (no JSON file written for it) and the run continues with the next dataset rather than aborting the whole batch. A summary at the end of the run lists which datasets were skipped and why.

Each JSON output file contains one dataset result with `dataset_name`, the ranked `hall_of_fame` entries including ARI and pipeline string, plus `best_pipeline` and `best_ari`.

Tune the genetic algorithm's run time and population size:

```bash
python code/autoclust.py --time-limit 60 --population-size 100
```

The offline phase (ARI-based, used by `code/autoclust.py` and the offline runs in `code/meta_feature_extraction.py`) stops early, before `--time-limit` is reached, as soon as a pipeline reaches a perfect ARI score of `1.0`. This only applies to the offline phase: the online phase's CVI runs (silhouette, Davies-Bouldin, Calinski-Harabasz) have no fixed optimum and always run for the full time budget.

`code/autoclust.py` is also imported as a module by `code/meta_feature_extraction.py`. The CLI flags below only affect direct script invocation (`python code/autoclust.py ...`); the underlying functions independently default to the same values when called from other code.

### Flag Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--time-limit` | int | `20` | Time budget for the genetic algorithm run, in seconds. The offline (ARI) run may stop earlier if a perfect ARI score of `1.0` is reached. |
| `--max-tree-height` | int | `6` | Maximum height of generated GP pipeline trees. |
| `--dataset-path` | path | `<script_dir>/../data/wine.csv` | Path to one CSV dataset to load. Cannot be combined with `--input-folder`. |
| `--input-folder` | path | none | Folder containing CSV datasets to run recursively. Cannot be combined with `--dataset-path`. |
| `--target-column` | str | `target` | Name of the target/label column in each dataset CSV. |
| `--population-size` | int | `50` | Number of individuals in the GA population; also the warmstart fill/cap size. |
| `--n-select` | int | `10` | Number of parents selected each generation via NSGA-II. |
| `--offspring-multiplier` | int | `5` | Number of cloned offspring generated per selected parent each generation. |
| `--cxpb` | float | `0.05` | Per-individual crossover probability during offspring generation. |
| `--mutpb` | float | `0.9` | Per-individual mutation probability for unaffected offspring. |
| `--hall-of-fame-size` | int | `5` | Number of ranked Hall of Fame pipelines to store per dataset. |
| `--output-folder` | path | `autoclust_results` | Folder for per-dataset JSON result files. |
| `--max-memory-gb` | float | none | Hard cap on process virtual memory in GB (Linux only, via `RLIMIT_AS`). Pipelines that would exceed it raise `MemoryError` instead of triggering the OS OOM killer; the affected dataset is skipped and the run continues with the next one. No-op with a warning on platforms without `RLIMIT_AS` support (e.g. macOS, Windows). |
| `--hard-time-limit` | int | none | Hard wall-clock cap, in seconds, on a single dataset's entire run (search + serialize + write). Unlike `--time-limit`, which is only checked between generations and individual pipeline evaluations, this is enforced by running the dataset in a subprocess that is forcibly terminated if it overruns — so it can stop a single pipeline evaluation stuck in a long-running native call (e.g. `AgglomerativeClustering`, `SpectralClustering`, `KNNImputer` on a large dataset). The affected dataset is skipped and the run continues with the next one. Disabled by default (runs in-process, no subprocess overhead). Should be set higher than `--time-limit`. |

## Usage: `code/prepare_knowledge_base.py`

Consolidates the per-dataset offline-phase results in `result_ll_kb/*.json` (produced by `code/autoclust.py`'s offline-phase CLI) with their source CSVs in `benchmark_csv/offline_phase/` into a single `knowledge_base.json` file: it adds the 28 meta-features for each dataset alongside the dataset's already-computed, ranked `hall_of_fame` pipelines. This consolidated file is what `code/meta_feature_extraction.py --knowledge-base ...` consumes to drive the online phase without recomputing the offline phase from scratch on every run.

```bash
python code/prepare_knowledge_base.py
```

The default invocation reads `result_ll_kb/` and `benchmark_csv/offline_phase/` and writes `knowledge_base.json` at the repo root. Datasets whose CSV can't be found, or whose meta-feature computation fails, are skipped with a warning rather than aborting the whole run. Datasets reserved for online-phase evaluation (`benchmark_csv/online_phase/`, `online_phase_prep4clust/`) are excluded from the knowledge base by default, even if they also have a `result_ll_kb` entry, to avoid the online phase trivially recognizing a held-out d_new dataset as one it already has offline results for.

### Flag Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--kb-results-dir` | path | `result_ll_kb` | Folder with the per-dataset JSON results of the offline phase. |
| `--csv-dir` | path | `benchmark_csv/offline_phase` | Folder with the CSV datasets matching the result files (matched by the `dataset_name` filename, not the `dataset_path` stored in the JSONs). |
| `--target-column` | str | `target` | Name of the target column to drop before meta-feature computation. |
| `--output-path` | path | `knowledge_base.json` | Path of the consolidated knowledge_base JSON file to write. |
| `--exclude-dir` | path | (repeatable) | Additional folder whose CSV filenames must be excluded from the knowledge base. Adds to, rather than replaces, the default excludes below. |
| `--no-default-excludes` | flag | `False` | Disables the built-in default excludes (`benchmark_csv/online_phase`, `online_phase_prep4clust`), so only directories explicitly passed via `--exclude-dir` are excluded. |

## Usage: `code/meta_feature_extraction.py`

Computes the 28 meta-features for a dataset and drives the online phase. Running it with no arguments reproduces the original demo behavior (iterates over `data/*.csv`, builds an in-memory knowledge base, runs a fresh offline-phase GA per dataset, then runs the online phase on the first dataset):

```bash
python code/meta_feature_extraction.py
```

Run the online phase against a pre-built knowledge base, warmstarting the GA search from the most similar known pipelines (default mode):

```bash
python code/meta_feature_extraction.py --knowledge-base knowledge_base.json --dataset-path benchmark_csv/online_phase/<dataset>.csv
```

Run pure meta-learning recommendation instead — no GA evolution, just evaluate the `k` most similar known pipelines directly against a score:

```bash
python code/meta_feature_extraction.py --knowledge-base knowledge_base.json --dataset-path benchmark_csv/offline_phase/<labeled-dataset>.csv --mode recommend-only --k 10 --eval-measure ari
```

Evaluate every CSV in a folder instead of a single dataset. Each dataset's result is written to `--output-dir` as soon as that dataset finishes (not batched until the end), and datasets whose result file already exists are skipped and logged — so an interrupted batch run can simply be re-run to resume:

```bash
python code/meta_feature_extraction.py --knowledge-base knowledge_base.json --input-folder online_phase_prep4clust --output-dir online_phase_results --mode recommend-only --eval-measure silhouette
```

For details on the 28 meta-features themselves, the `knowledge_base.json` schema, the batch-mode result file schema, and the pset-compatibility caveat that governs which knowledge-base pipelines get skipped during warmstart/recommendation, see [`code/KNOWLEDGE_BASE.md`](code/KNOWLEDGE_BASE.md).

### Flag Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `--knowledge-base` | path | none | Path to a `knowledge_base.json` produced by `prepare_knowledge_base.py`. If omitted, the script runs the original `data/*.csv` demo unchanged and all flags below are ignored. |
| `--dataset-path` | path | none | Path to a single d_new CSV. Mutually exclusive with `--input-folder`. One of the two is required when `--knowledge-base` is set. |
| `--input-folder` | path | none | Folder of d_new CSVs to evaluate in batch (searched recursively). Mutually exclusive with `--dataset-path`. |
| `--output-dir` | path | `online_phase_results` | Folder for per-dataset JSON result files. Only used with `--input-folder`. |
| `--target-column` | str | `target` | Name of the target column in d_new, if present. Needed for `--eval-measure ari`; otherwise optional. |
| `--m` | int | `50` | Number of nearest knowledge-base datasets to consider for ranking. |
| `--mode` | choice | `warmstart` | `warmstart`: GA online phase seeded from knowledge-base pipelines. `recommend-only`: no GA evolution, evaluate up to `--k` knowledge-base pipelines directly. |
| `--k` | int | `10` | Number of recommended pipelines to evaluate in `recommend-only` mode. |
| `--eval-measure` | choice | `ari` | `ari`, `silhouette`, `calinski_harabasz`, or `davies_bouldin` — only used in `recommend-only` mode. `ari` requires `--target-column` to exist in d_new. |
| `--time-limit-online` | int | `90` | Time budget per GA run, in seconds — only used in `warmstart` mode. |
| `--max-tree-height` | int | `6` | Maximum height of generated GP pipeline trees. |
| `--output-path` | path | none | Only used with `--dataset-path` (single-dataset mode). If set, additionally writes the result (cluster labels in `warmstart` mode, ranked pipeline table in `recommend-only` mode) to this CSV path. |
