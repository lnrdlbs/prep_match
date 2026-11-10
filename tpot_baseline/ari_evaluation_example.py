import argparse
import pdb
import time
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from pymfe.mfe import MFE
from sklearn.preprocessing import MinMaxScaler


class SurrogateScorer:
    def __init__(self, model, meta_features, cvi=[silhouette_score, davies_bouldin_score, calinski_harabasz_score]) -> None:
        self.model = model
        self.meta_features = meta_features
        self.cvi = cvi

    def __call__(self, estimator, X):
        try:
            warnings.filterwarnings('ignore')
            cluster_labels = estimator.fit_predict(X)
            mf = self.meta_features.copy()
            mf.extend([score(X,cluster_labels) for score in self.cvi])
            surrogate_score = self.model.predict([mf])[0]
            return surrogate_score if len(set(cluster_labels)) > 1 else -float('inf') 
        except Exception as e:
            raise TypeError(f"{e}")

metafeatures_records = []

features = ['attr_conc', 'attr_ent', 'attr_to_inst', 'cohesiveness', 'cor', 'cov',
       'eigenvalues', 'inst_to_attr', 'iq_range', 'kurtosis', 'mad', 'max', 'mean', 'median', 'min',
       'nr_attr', 'nr_cor_attr', 'nr_inst',
       'one_itemset', 'range', 'sd', 'skewness',
       'sparsity', 't2', 't3', 't4', 't_mean', 'two_itemset', 'var', 'wg_dist',
       ]

meta_features = ['attr_conc.mean','attr_conc.sd','attr_ent.mean',
        'attr_ent.sd','attr_to_inst','cohesiveness.mean','cohesiveness.sd',
        'cov.mean',
        'eigenvalues.mean','eigenvalues.sd',
        'inst_to_attr','iq_range.mean','iq_range.sd',
        'mad.mean','mad.sd',
        'median.mean',
        'median.sd',
        'nr_attr','nr_cor_attr','nr_inst','one_itemset.mean',
        'one_itemset.sd',
        'sd.mean','sd.sd'
        ,'sparsity.mean','sparsity.sd','t2','t3','t4','t_mean.mean',
        't_mean.sd','two_itemset.mean','two_itemset.sd','var.mean','var.sd',
        'wg_dist.mean','wg_dist.sd',
        'sil', 'dbs', 'chs', 'ari'
        ]


def _extract_metafeatures(X, meta_features):
        # Handle categorical columns if they exist
        if isinstance(X, pd.DataFrame):
            # Identify categorical columns
            categorical_columns = X.select_dtypes(exclude=['number']).columns
            if len(categorical_columns) > 0:
                # Encode categorical columns
                categorical_data = X[categorical_columns].apply(lambda col: col.astype('category').cat.codes).to_numpy()
                # Combine numeric and categorical data
                X_processed = np.hstack([X.select_dtypes(include=['number']).to_numpy(), categorical_data])
            else:
                # If no categorical data, use X as is
                X_processed = X.to_numpy()
        else:
            # If X is already a NumPy array, assume it's fully processed
            X_processed = X

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
            # Meta-feature extraction
        mfe = MFE(groups="all", features=features)
        mfe.fit(X_processed)
        ft = mfe.extract()

        # Extract only the specified meta-features
        return [value for key, value in zip(ft[0], ft[1]) if key in meta_features]

def load_features_and_labels(csv_path, label_index):
    data = pd.read_csv(csv_path, header=None)

    label_index = label_index if label_index >= 0 else data.shape[1] + label_index
    if label_index < 0 or label_index >= data.shape[1]:
        raise ValueError(f"Label index {label_index} is outside the CSV column range.")

    y_true = data.iloc[:, label_index].to_numpy()
    X = data.drop(columns=data.columns[label_index])

    # TPOT-Clustering works with numeric sklearn-style feature matrices.
    # One-hot encode categorical columns and fill numeric missing values.
    X = pd.get_dummies(X)
    X = X.fillna(X.median(numeric_only=True))

    return X, y_true


def cluster_labels_from_pipeline(pipeline, X):
    if hasattr(pipeline, "fit_predict"):
        return pipeline.fit_predict(X)
    if hasattr(pipeline, "predict"):
        return pipeline.predict(X)

    raise TypeError("The optimized pipeline has neither fit_predict nor predict.")


def find_datasets(input_path):
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.csv"))

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def run_dataset(csv_path: Path, args, TPOTClustering):
    started_at = time.perf_counter()
    export_path = None

    try:
        X, y_true = load_features_and_labels(csv_path, args.label_index)
        dataset_name = csv_path.stem
        # tpot = TPOTClustering(
        #     generations=args.generations,
        #     population_size=args.population_size,
        #     max_time_mins=args.max_time_mins,
        #     max_eval_time_mins=args.max_eval_time_mins,
        #     scoring=args.scoring,
        #     verbosity=args.verbosity,
        #     random_state=args.random_state,
        #     n_jobs=args.n_jobs,
        # )

        # tpot = TPOTClustering(
        #     #generations=5,
        #     generations=args.generations,
        #     population_size=args.population_size,
        #     max_time_mins=1,
        #     max_eval_time_mins=1,
        #     #scoring=args.scoring,
        #     scoring='combined_internal_score',
        #     verbosity=3,
        #     random_state=42,
        # )

        # tpot = TPOTClustering(
        #     generations=args.generations,
        #     population_size=args.population_size,
        #     max_time_mins=args.max_time_mins,
        #     max_eval_time_mins=args.max_eval_time_mins,
        #     random_state=args.random_state,
        #     scoring=args.scoring,
        #     verbosity=args.verbosity,
        #     n_jobs=args.n_jobs,
        # )
        
        tmp_debug_model_path = args.model_path
        model = joblib.load(args.model_path)

        if args.scaling == True: 
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X.select_dtypes(include=['number']))
        #extracted_mf = _extract_metafeatures(X_scaled, meta_features)
        extracted_mf = _extract_metafeatures(X, meta_features)
        metafeatures_records.append(dict(zip(["Dataset"] + meta_features, [dataset_name] + extracted_mf)))
        tpot_kwargs = dict(
            population_size=args.population_size,  # default 50
            verbosity=args.verbosity,
            random_state=args.random_state,
            crossover_rate=0.05,
            mutation_rate=0.9,
            max_time_mins=args.max_time_mins,  # default 30
            max_eval_time_mins=args.max_eval_time_mins,  # default 1.0
            scoring=SurrogateScorer(model=model, meta_features=extracted_mf),
        )
        if args.fix_n_clusters:
            tpot_kwargs['n_clusters'] = len(set(y_true))
        tpot = TPOTClustering(**tpot_kwargs)

        tpot.fit(X)

        y_pred = cluster_labels_from_pipeline(tpot.fitted_pipeline_, X)
        ari = adjusted_rand_score(y_true, y_pred)
        internal_score = tpot.score(X)

        if args.export_dir is not None:
            args.export_dir.mkdir(parents=True, exist_ok=True)
            export_path = args.export_dir / f"{csv_path.stem}_tpot_{args.scoring}.py"
            tpot.export(str(export_path))

        return {
            "dataset": csv_path.name,
            "path": str(csv_path),
            "status": "ok",
            "error": "",
            "rows": X.shape[0],
            "features": X.shape[1],
            "true_clusters": len(set(y_true)),
            "predicted_clusters": len(set(y_pred)),
            "scoring": args.scoring,
            "internal_score": internal_score,
            "adjusted_rand_index": ari,
            "generations": args.generations,
            "population_size": args.population_size,
            "max_time_mins": args.max_time_mins,
            "max_eval_time_mins": args.max_eval_time_mins,
            "random_state": args.random_state,
            "runtime_seconds": time.perf_counter() - started_at,
            "pipeline": str(tpot.fitted_pipeline_),
            "export_path": "" if export_path is None else str(export_path),
        }
    except Exception as exc:
        return {
            "dataset": csv_path.name,
            "path": str(csv_path),
            "status": "failed",
            "error": repr(exc),
            "rows": "",
            "features": "",
            "true_clusters": "",
            "predicted_clusters": "",
            "scoring": args.scoring,
            "internal_score": "",
            "adjusted_rand_index": "",
            "generations": args.generations,
            "population_size": args.population_size,
            "max_time_mins": args.max_time_mins,
            "max_eval_time_mins": args.max_eval_time_mins,
            "random_state": args.random_state,
            "runtime_seconds": time.perf_counter() - started_at,
            "pipeline": "",
            "export_path": "",
        }


def main():
    from tpotclustering import TPOTClustering

    parser = argparse.ArgumentParser(
        description=(
            "Run TPOT-Clustering on one headerless CSV or a folder of headerless "
            "CSVs, then evaluate each result with Adjusted Rand Index."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to one CSV dataset or a folder containing CSV datasets.",
    )
    parser.add_argument(
        "--label-index",
        type=int,
        default=-1,
        help=(
            "Zero-based index of the ground-truth label column used only for ARI "
            "evaluation. Defaults to -1, meaning the last column."
        ),
    )
    parser.add_argument(
        "--scoring",
        default="surrogate_scorer",
        choices=[
            "silhouette_score",
            "davies_bouldin_score",
            "calinski_harabasz_score",
            "intra_cluster_cosine",
            "combined_internal_score",
            "surrogate_scorer"
        ],
        help="Internal unsupervised score TPOT optimizes.",
    )
    parser.add_argument(
        "--max-time-mins",
        type=int,
        default=30,
        help="Total TPOT optimization budget in minutes.",
    )
    parser.add_argument(
        "--max-eval-time-mins",
        type=float,
        default=1.0,
        help="Maximum time allowed for one candidate pipeline.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="Number of candidate pipelines per generation.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help=(
            "Optional number of generations. Leave unset to stop by "
            "--max-time-mins."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible TPOT searches.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=10,
        help="Parallel jobs for evaluating pipelines. Use 1 if your machine struggles.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("ari_evaluation_results.csv"),
        help="CSV file where one result row per dataset is stored.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional folder where optimized pipeline Python files are exported.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=3,
        help="TPOT verbosity level.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="poac_sv7.joblib",
        help="Path to the surrogate model",
    )
    parser.add_argument(
        "--scaling",
        type=bool,
        default=False,
        help="Include static scaling",
    )
    parser.add_argument(
        "--fix-n-clusters",
        action="store_true",
        default=False,
        help=(
            "Fix n_clusters to the true number of clusters (derived from the "
            "ground-truth label column) for all clustering algorithms that "
            "accept it.  When set, n_clusters cannot be mutated during "
            "optimization.  Requires --label-index to point at a valid label "
            "column."
        ),
    )

    args = parser.parse_args()

    datasets = find_datasets(args.input_path)
    if not datasets:
        raise ValueError(f"No CSV files found in: {args.input_path}")

    print(f"Found {len(datasets)} dataset(s).")

    results = []
    for index, csv_path in enumerate(datasets, start=1):
        print(f"\n[{index}/{len(datasets)}] Running {csv_path}")
        result = run_dataset(csv_path, args, TPOTClustering)
        results.append(result)

        if result["status"] == "ok":
            print(f"Internal TPOT score ({args.scoring}): {result['internal_score']}")
            print(f"Adjusted Rand Index: {result['adjusted_rand_index']}")
            print(f"Number of discovered clusters: {result['predicted_clusters']}")
        else:
            print(f"Failed: {result['error']}")

        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(args.results_path, index=False)
        print(f"Saved results to: {args.results_path}")


if __name__ == "__main__":
    main()
