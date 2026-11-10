import os
import sys
import json
import glob
import argparse

import pandas as pd

from meta_feature_extraction import extract_features


def collect_excluded_dataset_names(exclude_dirs) -> set:
    """
    Collects the basenames of all *.csv files found in exclude_dirs. Used to keep datasets that
    are reserved for online-phase evaluation (e.g. held-out benchmark folders) out of the
    knowledge_base, even if they also happen to have a result_ll_kb entry. Directories that don't
    exist are skipped with a warning rather than raising.
    """
    excluded_names = set()

    for exclude_dir in exclude_dirs or []:
        if not os.path.isdir(exclude_dir):
            print(f"Warning: --exclude-dir {exclude_dir} does not exist, ignoring.", file=sys.stderr)
            continue

        for csv_path in glob.glob(os.path.join(exclude_dir, "*.csv")):
            excluded_names.add(os.path.basename(csv_path))

    return excluded_names


def build_knowledge_base(kb_results_dir: str, csv_dir: str, target_column: str = "target", exclude_dirs=None) -> dict:
    """
    Consolidates the offline-phase results from result_ll_kb/*.json with the matching CSVs in
    csv_dir into a single knowledge_base. For each dataset, the meta-features (via
    extract_features) are recomputed and merged with the already existing hall_of_fame.

    :param exclude_dirs: optional list of folders whose CSV filenames must never end up in the
        knowledge_base (eg. datasets held out for online-phase evaluation), to avoid the online
        phase trivially "recognizing" d_new as a dataset it already has offline-phase results for.
    :return: { dataset_name: {"n_features": int, "meta_features": {...}, "hall_of_fame": [...], "best_ari": float} }
    """
    knowledge_base = {}
    excluded_names = collect_excluded_dataset_names(exclude_dirs)
    if excluded_names:
        print(f"Excluding {len(excluded_names)} dataset name(s) found in: {exclude_dirs}")

    result_paths = sorted(glob.glob(os.path.join(kb_results_dir, "*.json")))

    for result_path in result_paths:
        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        dataset_name = result.get("dataset_name", os.path.basename(result_path))

        if dataset_name in excluded_names:
            print(f"Skipping {dataset_name}: reserved for online-phase evaluation (excluded via --exclude-dir).", file=sys.stderr)
            continue

        csv_path = os.path.join(csv_dir, dataset_name)

        if not os.path.isfile(csv_path):
            print(f"Skipping {dataset_name}: no CSV found at {csv_path}.", file=sys.stderr)
            continue

        try:
            df = pd.read_csv(csv_path)
            if target_column in df.columns:
                df = df.drop(columns=[target_column])
            X = df.values

            meta_features = extract_features(X)

            knowledge_base[dataset_name] = {
                "n_features": X.shape[1],
                "meta_features": meta_features,
                "hall_of_fame": result.get("hall_of_fame", []),
                "best_ari": result.get("best_ari"),
            }
        except Exception as e:
            print(f"Skipping {dataset_name}: error during meta-feature computation ({e}).", file=sys.stderr)
            continue

    return knowledge_base


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))

    parser = argparse.ArgumentParser(
        description="Consolidates the offline-phase results (result_ll_kb/*.json) and their "
                     "matching datasets into a single knowledge_base file for the online phase."
    )
    parser.add_argument("--kb-results-dir", type=str, default=os.path.join(repo_root, "result_ll_kb"),
        help="Folder with the per-dataset JSON results of the offline phase (default: result_ll_kb)")
    parser.add_argument("--csv-dir", type=str, default=os.path.join(repo_root, "benchmark_csv", "offline_phase"),
        help="Folder with the CSV datasets matching the result files (default: benchmark_csv/offline_phase)")
    parser.add_argument("--target-column", type=str, default="target",
        help="Name of the target column to drop before meta-feature computation (default: 'target')")
    parser.add_argument("--output-path", type=str, default=os.path.join(repo_root, "knowledge_base.json"),
        help="Path of the consolidated knowledge_base JSON file (default: knowledge_base.json)")
    parser.add_argument("--exclude-dir", type=str, action="append", default=None,
        help="Folder whose CSV filenames must be excluded from the knowledge_base, even if they "
             "have a result_ll_kb entry (eg. datasets held out for online-phase evaluation). "
             "Can be passed multiple times; adds to the defaults below unless --no-default-excludes is set.")
    parser.add_argument("--no-default-excludes", action="store_true", default=False,
        help="Disable the built-in default excludes (benchmark_csv/online_phase, online_phase_prep4clust), "
             "so only directories explicitly passed via --exclude-dir (if any) are excluded.")
    args = parser.parse_args()

    default_exclude_dirs = [] if args.no_default_excludes else [
        os.path.join(repo_root, "benchmark_csv", "online_phase"),
        os.path.join(repo_root, "online_phase_prep4clust"),
    ]
    exclude_dirs = default_exclude_dirs + (args.exclude_dir or [])

    knowledge_base = build_knowledge_base(args.kb_results_dir, args.csv_dir, args.target_column, exclude_dirs)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, indent=2)
        f.write("\n")

    print(f"\nknowledge_base with {len(knowledge_base)} dataset(s) written to: {args.output_path}")


if __name__ == "__main__":
    main()
