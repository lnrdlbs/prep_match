# PrepMatch

This is the prototypical implementation of the submitted paper PrepMatch Meta-Learning to Suggest Preprocessing Pipelines for Clustering. In the following we provide an overview of the structure and how to use the parts of PrepMatch. 

## Overview

PrepMatch consists of several scripts covering data generation, optimization, meta-learning, and benchmarking:


- **`create_data.py`**
  - Generates synthetic clustering datasets from a YAML configuration.
  - Produces the synthetic data used for the knowledge base.

- **`mv_optimization.py`**
  - Optimizes preprocessing pipelines specifically for **missing value imputation**.
  - Uses an evolutionary algorithm to select imputation strategies.
  - Outputs imputed datasets

- **`optimization.py`**
  - Core evolutionary optimizer for preprocessing pipelines (outlier removal, scaling, dimensionality reduction).
  - Evaluates pipelines using clustering quality metrics (ARI, NMI, AMI).
  - Produces the results that later form the **knowledge base** for meta-learning.

- **`meta-learner.py`**
  - Application phase of PrepMatch.
  - Uses a precomputed knowledge base to suggest preprocessing pipelines for new datasets.
  - Selects pipelines via nearest-neighbor search in meta-feature space.

- **`ml_random_search.py`**
  - Baseline method for benchmarking.
  - Applies random (or semi-guided) search over preprocessing pipelines.
  - Used to compare meta-learning against non-informed search strategies.

- **`paper_figures/`**
  - Reproduces the figures and tables of the paper's evaluation.
  - Works from precomputed results included in the repository, so no experiments have to be re-run.

## Installation
This project is tested with Python version 3.11.15.

### Decompress Data Files

Files larger than 50 MB (the knowledge base `kb.csv`, several real-world datasets, and the TPOT surrogate model `tpot_baseline/poac_sv7.joblib`) are stored gzip-compressed (`.gz`) to stay within GitHub's file size limits. Decompress them once after cloning, from the repository root:

```
python decompress_data.py
```

The script uses only the Python standard library, restores each original file next to its `.gz` archive, and verifies it against its SHA-256 checksum. Files that are already restored are skipped.

### Create Conda Env (optional but recommended)
```
conda create -n prepmatch python=3.11.9
```

```
conda activate prepmatch
```
### Install Requirements

```
pip install -r requirements.txt
```

## `create_data.py`

This script generates the synthetic data based on a configuration file and writes the results to disk.

### Arguments

| Flag | Long Flag | Type | Required | Default | Description |
|-----|----------|------|----------|---------|-------------|
| `-c` | `--config` | `str` | Yes | — | Path to the configuration file used to define how synthetic data is generated. |
| `-si` | `--start_id` | `int` | No | `0` | Starting ID for data generation. Use this **only** to resume a previously interrupted generation process. |
| `-o` | `--output_folder` | `str` | Yes | — | Directory where the generated output files will be written. |

### Example Usage

```bash
python create_data.py \
  --config configs/config_small.yaml \
  --output_folder syn_data
```

## `mv_optimization.py`

Script for the imputation of missing values. 

### Required Arguments

| Flag | Long Flag | Type | Required | Description |
|-----|----------|------|----------|-------------|
| `-d` | `--data_folder` | `str` | Yes | Path to the folder containing the input datasets. |
| `-o` | `--output_folder` | `str` | Yes | Path to the folder where optimization results will be written. |
| `-l` | `--log_folder` | `str` | Yes | Path to the folder where per-run JSONL log files will be stored. |
| `-io` | `--imputed_output` | `str` | Yes | path for the imputed data. |

### Optional Arguments

| Flag | Long Flag | Type | Default | Description |
|-----|----------|------|---------|-------------|
| `-e` | `--eval` | `str` | `ari` | Evaluation metric used for optimization. Choices: `ari`, `nmi`, `ami`. |
| `-m` | `--mutpb` | `float` | `[0.2]` | Probability that an individual undergoes mutation. |
| `-c` | `--cxpb` | `float` | `[0.5]` | Probability of crossover between two individuals. |
| `-i` | `--indpb` | `float` | `[0.2]` | Probability that a given parameter within an individual is mutated. |
| `-ts` | `--tourn_size` | `int` | `[5]` | Tournament size used during selection. |
| `-g` | `--generations` | `int` | `[10]` | Number of generations per dataset. |
| `-p` | `--population_size` | `int` | `[20]` | Population size per dataset. |
| `-s` | `--seed` | `int` | `[876]` | Random seed for reproducibility. |
| `-t` | `--time-budget` | `int` | `[600]` | Time budget (in seconds) per dataset. Optimization stops when exceeded. |
| `-oe` | `--overwrite_existing` | `int` | `0` | If set to `1`, existing result files are overwritten. Otherwise, configurations already evaluated in the output folder are skipped. |


### Notes

- When multiple values are provided for hyperparameters, the script evaluates **all combinations** using a grid search over evolutionary runs.
- Each run generates:
  - A unique log file (`.jsonl`) in the log folder
  - Result files written to the output folder
- Datasets with `p_missing_values == 0` are skipped automatically.
- Optimization may terminate early if the time budget is reached.

### Example Usage

Short: 
```bash 
python mv_optimization.py \
  --data_folder syn_data \
  --output_folder results_imputed \
  --imputed_output output_imputed \
  --log_folder logs 
```

Long:
```bash
python mv_optimization.py \
  --data_folder syn_data \
  --output_folder results_imputed \
  --imputed_output output_imputed \
  --log_folder logs \
  --eval ari \
  --mutpb 0.1 0.2 \
  --cxpb 0.4 0.6 \
  --generations 20 \
  --population_size 50
```

## `optimization.py`

Script for the main optimization

### Required Arguments

| Flag | Long Flag | Type | Default | Description |
|-----|----------|------|---------|-------------|
| `-d` | `--data_folder` | `str` | — | Path to the folder containing input datasets.|
| `-o` | `--output_folder` | `str` | — | Path to the folder where results are written |
| `-l` | `--log_folder` | `str` | — | Path to the folder where logs are written. |

### Optional Arguments

| Flag | Long Flag | Type | Default | Description |
|-----|----------|------|---------|-------------|
| `-if` | `--imputed_folder` | `str` | `None` | Path to a folder containing already-imputed data used as input when datasets contain missing values. |
| `-e` | `--eval` | `str` | `ari` | Evaluation metric used for optimization. Choices: `ari`, `nmi`, `ami`. |
| `-m` | `--mutpb` | `float` | `[0.2]` | Probability that an individual undergoes mutation. |
| `-c` | `--cxpb` | `float` | `[0.5]` | Probability of crossover between two individuals. |
| `-i` | `--indpb` | `float` | `[0.2]` | Probability that a given parameter within an individual is mutated. |
| `-s` | `--seed` | `int` (list) | `[876]` | Random seed(s) for reproducibility. |
| `-is` | `--init_strategy` | `{'random','dc_based'}` | `dc_based` | Population initialization strategy.  |
| `-t` | `--time_budget` | `int` (list) | `[600]` | Time budget in seconds per dataset.  |
| `-g` | `--generations` | `int` (list) | `[50]` | Maximum generations per dataset|
| `-p` | `--population_size` | `int` (list) | `[20]` | Population size per dataset.|
| `-ts` | `--tourn_size` | `int` (list) | `[5]` | Tournament selection size. |
| `-oe` | `--overwrite_existing` | `int` | `0` | If `1`, overwrite existing results. Otherwise, skip configs already present in the output folder (by run-hash match). |


### Notes

- When multiple values are provided for hyperparameters, the script evaluates **all combinations** using a grid search over evolutionary runs.
- Each run generates:
  - A unique log file (`.jsonl`) in the log folder
  - Result files written to the output folder

### Example Usage

```bash
OMP_NUM_THREADS=5 python optimization.py \
  --data_folder syn_data \
  --output_folder ./results \
  --imputed_folder output_imputed \
  --log_folder ./logs
```
## `meta-learner.py`

Script for using the meta-learning 

### Required Arguments

| Flag | Long Flag | Type | Default | Description |
|-----|----------|------|---------|-------------|
| `-kb` | `--knowledge_base` | `str` | — | Path to the knowledge base. |
| `-o` | `--output_folder` | `str` | — | Output directory where result CSVs are written. |
| `-k` | *(no long flag)* | `int` (list) | — | Number of neighbors to consider (`k`). Accepts multiple values (`nargs='+'`) and runs for each value. |
| `-p` | `--pipelines` | `int` (list) | — | Number of pipelines to evaluate per neighbor. Accepts multiple values and runs for each value. |

### Optional Arguments

| Flag | Long Flag | Type | Default | Description |
|-----|----------|------|---------|-------------|
| `-sd` | `--single_dataset` | `bool` | `False` | If true, evaluate only a single dataset file (requires `--single_dataset_path`). |
| `-d` | `--datasets` | `str` | `None` | Path to a dataset folder. All `.pickle` files in the folder are considered when `--single_dataset` is not set. |
| `-dsp` | `--single_dataset_path` | `str` | `None` | Path to a single dataset file. Used only when `--single_dataset` is true. |
| `-oe` | `--overwrite_existing` | `int` | `0` | If `1`, overwrite existing result files. If `0`, skip runs whose output file already exists. |
| `-mfg` | `--meta_feature_groups` | `{'all','general','statistical','complexity'}` (list) | `['all']` | Which meta-feature groups to use for nearest-neighbor search. Multiple values allowed. |
| `-c` | `--clustering` | `str` | `kmeans` | Clustering algorithm used to evaluate suggested pipelines (ARI before/after). Supported: `kmeans`, `dbscan`, `hdbscan`, `agglomerative`. |

### Notes / Constraints

- Dataset selection:
  - If `--single_dataset` is true, the script reads only `--single_dataset_path`.
  - Otherwise it reads all `.pickle` files in `--datasets`.
- Grid execution:
  - The script iterates over all combinations of `k` (from `-k`) and `p` (from `--pipelines`).
- File skipping:
  - With `--overwrite_existing 0`, the script checks for an existing output file name for the `(dataset, kb, k, p, mf_groups)` combination and skips if present.

### Example Usage



Single dataset
```bash
python meta-learner.py \
  --knowledge_base kb.csv \
  -sd true \
  -dsp ./real_world_data/dermatology.pickle \
  --output_folder ./ml_results \
  -k 1  \
  -p 1  
  ```

All datasets (can run long)
  ```bash
python meta-learner.py \
  --knowledge_base kb.csv \
  -d ./real_world_data \
  --output_folder ./ml_results \
  -k 1  \
  -p 1  
  ```

## `ml_random_search.py`

Random search (used to benchmark the application phase)


### Required Arguments

| Flag | Long Flag | Type | Default | Description |
|-----|----------|------|---------|-------------|
| `-o` | `--output_folder` | `str` | — | Output directory where result CSVs are written. |

### Optional Arguments

| Flag | Long Flag | Type | Default | Description |
|-----|----------|------|---------|-------------|
| `-sd` | `--single_dataset` | `bool` | `False` | If true, evaluate only a single dataset file (requires `--single_dataset_path`). |
| `-d` | `--datasets` | `str` | `None` | Path to a dataset folder. All `.pickle` files in the folder are considered when `--single_dataset` is not set. |
| `-dsp` | `--single_dataset_path` | `str` | `None` | Path to a single dataset file. Used only when `--single_dataset` is true. |
| `-oe` | `--overwrite_existing` | `int` | `0` | If `1`, overwrite existing result files. If `0`, skip runs whose output file already exists. |
| `-tb` | `--time_budget` | `int` | `600` | Time budget in seconds (intended per dataset/seed run). **Note:** the current loop checks against `600` directly in the code path shown, not `time_budget`. |
| `-mt` | `--max_tries` | `int` | `50` | Maximum number of pipelines to try (`max_iterations`). |
| `-pl` | `--pipeline_length` | `tuple(int,int)` | `(1, 3)` | Pipeline length range (min, max) used when sampling pipeline length. |
| `-s` | `--strategy` | `str` | `help` | Pipeline sampling strategy. `help` uses method families depending on missing values; other values fall back to a more “full random” branch. |
| `-nrs` | `--number_seeds` | `int` | `0` | Number of random seeds to try. If `0`, uses a single fixed seed `876`. Otherwise generates `number_seeds` random seeds. |

### Notes / Constraints

- Dataset selection:
  - If `--single_dataset` is true, the script reads only `--single_dataset_path`.
  - Otherwise it reads all `.pickle` files in `--datasets`.
- Repetition:
  - The script loops over `seed` values (either `[876]` or randomly generated) and creates one output CSV per dataset/seed combination.

### Example Usage

Run on a dataset folder:

```bash
python ml_random_search.py \
  --datasets ./real_world_data/dermatology.pickle \ \
  --output_folder ./random_search_results \
  --max_tries 100 \
  --pipeline_length "(1, 4)" \
  --strategy help \
  --time_budget 600
  ```

## Paper Figures

The folder `paper_figures/` reproduces the figures and tables of the paper's evaluation from precomputed results, without re-running any experiments.

| Path | Content |
|------|---------|
| `paper_figures/export/paper_figures.ipynb` | Notebook that creates all figures and tables. |
| `paper_figures/export/data/` | Precomputed results read by the notebook (see below). |
| `paper_figures/export/output/` | Generated figures (`.pdf`) and LaTeX tables (`.tex`). |
| `paper_figures/build_data.py` | Script that was used to extract `export/data/` from the raw experiment outputs (see note below). |

The precomputed results in `paper_figures/export/data/` are:

- `prepmatch_kmeans/`, `prepmatch_dbscan/`, `prepmatch_hdbscan/`, `prepmatch_agglomerative/`: results of `meta-learner.py` (`k = 23`, `p = 5`) on the real-world datasets for each clustering algorithm
- `kb_results.csv`: optimization results that form the knowledge base
- `random_search.csv`: results of `ml_random_search.py`
- `random_baselines.csv`: random pipeline baselines
- `tpe_autoclust.csv`, `tpot_clustering.csv`: results of the TPE-AutoClust and TPOT-Clustering baselines
- `datasets.csv`: number of samples per real-world dataset

### Generate the Figures

```bash
cd paper_figures/export
pip install -r requirements.txt   # not needed if the main requirements.txt is already installed
jupyter nbconvert --to notebook --execute --inplace paper_figures.ipynb
```

The notebook writes these files to `paper_figures/export/output/`:

| File | Content |
|------|---------|
| `all_algorithms_median_bars4.pdf` | Median ARI for four clustering algorithms |
| `k_p_comb3.pdf` | Median ARI for different `k` (neighbors) and `p` (pipelines per neighbor) |
| `baselines_small_5.pdf` | ARI distribution on the synthetic datasets |
| `tab_baselines.tex` | ARI comparison of PrepMatch and the baselines |
| `tab_ppevalgroups.tex` | ARI change per data characteristic |

