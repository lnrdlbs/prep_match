import os
import sys
import glob
import argparse


def csv_has_header(csv_path: str) -> bool:
    """
    A CSV is considered to already have a header if any token in its first row fails to parse
    as a float. Used to make this script idempotent (safe to re-run).
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    for token in first_line.split(","):
        try:
            float(token)
        except ValueError:
            return True

    return False


def add_header(csv_path: str, target_column: str = "target", feature_prefix: str = "f"):
    """
    Prepends a synthetic header row f0,f1,...,f{n-2},target to csv_path, in place.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        rest = f.read()

    n_cols = len(first_line.strip().split(","))
    header = [f"{feature_prefix}{i}" for i in range(n_cols - 1)] + [target_column]

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        f.write(first_line)
        f.write(rest)

    return n_cols


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Adds a synthetic header row (f0,f1,...,target) to headerless CSVs in a "
                     "folder, in place. Idempotent: files that already have a header are skipped."
    )
    parser.add_argument("--folder", type=str, default=os.path.join(current_dir, "online_phase_prep4clust"),
        help="Folder of CSVs to add headers to (default: online_phase_prep4clust)")
    parser.add_argument("--target-column", type=str, default="target",
        help="Name to use for the last column's header (default: 'target')")
    parser.add_argument("--feature-prefix", type=str, default="f",
        help="Prefix for the generated feature column names (default: 'f', giving f0, f1, ...)")
    parser.add_argument("--dry-run", action="store_true", default=False,
        help="Only print which files would be changed, without writing anything")
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.folder, "*.csv")))
    print(f"Found {len(csv_paths)} CSV file(s) in {args.folder}.")

    changed, skipped, failed = [], [], []

    for csv_path in csv_paths:
        try:
            if csv_has_header(csv_path):
                skipped.append(csv_path)
                print(f"Skipping {os.path.basename(csv_path)}: already has a header.")
                continue

            if args.dry_run:
                print(f"Would add header to {os.path.basename(csv_path)}.")
                changed.append(csv_path)
                continue

            n_cols = add_header(csv_path, args.target_column, args.feature_prefix)
            changed.append(csv_path)
            print(f"Added header to {os.path.basename(csv_path)} ({n_cols} columns).")

        except Exception as e:
            failed.append(csv_path)
            print(f"Skipping {os.path.basename(csv_path)}: error ({e}).", file=sys.stderr)

    verb = "Would change" if args.dry_run else "Changed"
    print(f"\n{verb} {len(changed)} file(s), skipped {len(skipped)} (already had a header), failed {len(failed)}.")


if __name__ == "__main__":
    main()
