# TPOT-Clustering — ARI Evaluation (Reproducibility Package)

Self-contained snapshot to reproduce the surrogate-model-guided TPOT-Clustering
runs and their Adjusted Rand Index (ARI) evaluation.

The `tpotclustering/` package here is a **modified fork**. The original projects:

- TPOT-Clustering (upstream): <https://github.com/Mcamilo/tpot-clustering>
- Original TPOT: <https://github.com/EpistasisLab/tpot>

## Contents

| Path | Description |
|------|-------------|
| `ari_evaluation_example.py` | Runs TPOT-Clustering on one or more headerless CSVs and scores each result with ARI against the ground-truth label column. |
| `tpotclustering/` | Vendored TPOT-Clustering package (the exact modified fork used for these results — **not** the public PyPI/GitHub version). |
| `poac_sv7.joblib` | Pre-trained surrogate model used as the optimization scorer (`SurrogateScorer`). |
| `own_datasets/` | 22 headerless CSV datasets; the last column is the ground-truth label. |
| `requirements-lock.txt` | Exact package versions verified to load the model and run the script. |
| `requirements.txt` | Original loose requirements (reference only — not reproducible). |
| `setup.py` | Packaging metadata for the vendored package. |

## Setup

```bash
# Python 3.11 recommended (author used 3.11.15)
python -m venv .venv && source .venv/bin/activate

# Install the pinned dependency versions (important: the surrogate model is
# pickled and only guaranteed to load under these versions).
pip install -r requirements-lock.txt

# Make the vendored package importable
pip install -e .
```

## Run

Single dataset (quick smoke test):

```bash
python ari_evaluation_example.py own_datasets/iris.csv --max-time-mins 2
```

Reproduce the full evaluation over every dataset in the folder:

```bash
python ari_evaluation_example.py own_datasets \
    --model-path poac_sv7.joblib \
    --max-time-mins 30 \
    --max-eval-time-mins 1.0 \
    --population-size 50 \
    --random-state 42 \
    --results-path ari_evaluation_results.csv
```

One result row per dataset is written to `--results-path` (default
`ari_evaluation_results.csv`), including ARI, internal score, discovered vs.
true cluster counts, runtime, and the optimized pipeline.

### Useful flags

- `--label-index N` — column holding the ground-truth labels (default `-1`, the last column; used only for ARI).
- `--fix-n-clusters` — fix `n_clusters` to the true number of classes for algorithms that accept it.
- `--scaling True` — apply MinMax scaling to numeric features before meta-feature extraction.
- `--export-dir DIR` — export each optimized pipeline as a Python file.

## Reproducibility notes

- **Pinned versions matter.** `poac_sv7.joblib` was pickled with the environment
  in `requirements-lock.txt` (scikit-learn 1.8.0, numpy 2.4.4, pandas 3.0.2,
  joblib 1.5.3). Loading it under other versions may fail or behave differently.
- **Vendored package.** `tpotclustering/` is the author's modified fork; do not
  replace it with a pip-installed TPOT, or results will differ.
- TPOT-Clustering uses genetic programming; `--random-state 42` fixes the seed,
  but exact reproducibility can still vary with `--n-jobs > 1` due to
  parallel evaluation ordering. Use `--n-jobs 1` for strict determinism.

## License

The vendored TPOT-Clustering package is distributed under the GNU Lesser General
Public License v3 (LGPLv3), inherited from upstream. See [`LICENSE`](LICENSE).
