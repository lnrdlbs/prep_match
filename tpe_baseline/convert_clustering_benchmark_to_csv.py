#!/usr/bin/env python3
"""
Convert deric/clustering-benchmark ARFF datasets to plain CSV files.

Output format:
- header row with generated column names by default
- no index column
- no ARFF comments/descriptions/metadata
- numeric feature columns are passed through
- categorical/string feature columns are one-hot encoded
- class/label column is integer encoded and written as the final column

Usage examples:
  python convert_clustering_benchmark_to_csv.py --download --output clustering_csv
  python convert_clustering_benchmark_to_csv.py --repo /path/to/clustering-benchmark --output clustering_csv
  python convert_clustering_benchmark_to_csv.py --input /path/to/datasets --output clustering_csv --zip
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

GITHUB_ZIP_URL = "https://github.com/deric/clustering-benchmark/archive/refs/heads/master.zip"
DATASET_REL_DIR = Path("src/main/resources/datasets")
DEFAULT_ONLINE_PHASE_LOOKUP = Path("online_phase_data_lookup.json")
MISSING_ONLINE_PHASE_FILE = "missing_online_phase_datasets.txt"

_ATTR_RE = re.compile(r"^@attribute\s+(?P<name>(?:'[^']+'|\"[^\"]+\"|\S+))\s+(?P<type>.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Attribute:
    name: str
    type_spec: str


def _strip_inline_comment(line: str) -> str:
    """Strip ARFF comments beginning with %, unless inside single/double quotes."""
    in_single = False
    in_double = False
    escaped = False
    out = []
    for ch in line:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
            continue
        if ch == "%" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).strip()


def _clean_attr_name(raw_name: str) -> str:
    raw_name = raw_name.strip()
    if len(raw_name) >= 2 and raw_name[0] == raw_name[-1] and raw_name[0] in {"'", '"'}:
        return raw_name[1:-1]
    return raw_name


def _clean_nominal_value(raw_value: str) -> str:
    raw_value = raw_value.strip()
    if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in {"'", '"'}:
        return raw_value[1:-1]
    return raw_value


def _nominal_values(type_spec: str) -> Optional[List[str]]:
    type_spec = type_spec.strip()
    if not (type_spec.startswith("{") and type_spec.endswith("}")):
        return None
    body = type_spec[1:-1].strip()
    if not body:
        return []
    return [_clean_nominal_value(value) for value in next(csv.reader([body], skipinitialspace=True))]


def _is_numeric_attribute(attribute: Attribute) -> bool:
    return attribute.type_spec.strip().lower() in {"numeric", "real", "integer"}


def _is_numeric_value(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _dummy_column_name(index: int) -> str:
    """Return spreadsheet-like names: a, b, ..., z, aa, ab, ..."""
    chars = []
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        chars.append(chr(ord("a") + remainder))
    return "".join(reversed(chars))


def _column_base_name(attribute: Attribute, fallback_index: int) -> str:
    name = attribute.name.strip()
    return name if name else _dummy_column_name(fallback_index)


def _parse_dense_row(line: str) -> List[str]:
    # Most ARFF dense rows are CSV-like, but some UCI-derived files in this
    # collection use whitespace-separated rows. shlex preserves quoted strings.
    if "," not in line:
        return shlex.split(line)
    return [cell.strip() for cell in next(csv.reader([line], skipinitialspace=True))]


def _parse_sparse_row(line: str, n_attributes: int) -> List[str]:
    # Basic sparse ARFF support: {index value, index value, ...}
    row = ["0"] * n_attributes
    body = line.strip()[1:-1].strip()
    if not body:
        return row
    for part in next(csv.reader([body], skipinitialspace=True)):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(None, 1)
        if len(pieces) != 2:
            raise ValueError(f"Malformed sparse ARFF entry: {part!r}")
        idx = int(pieces[0])
        if idx < 0 or idx >= n_attributes:
            raise ValueError(f"Sparse ARFF index {idx} outside 0..{n_attributes - 1}")
        row[idx] = pieces[1].strip()
    return row


def read_arff(path: Path) -> Tuple[List[Attribute], List[List[str]]]:
    """Return (attributes, data_rows) for an ARFF file."""
    attributes: List[Attribute] = []
    rows: List[List[str]] = []
    in_data = False

    # Prefer UTF-8; fall back for older UCI files with non-UTF8 bytes.
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")

    for original_line in text.splitlines():
        line = _strip_inline_comment(original_line)
        if not line:
            continue

        low = line.lower()
        if not in_data:
            if low.startswith("@data"):
                in_data = True
                # Some ARFF files place data after @DATA on the same line.
                remainder = line[5:].strip()
                if remainder:
                    if remainder.startswith("{") and remainder.endswith("}"):
                        rows.append(_parse_sparse_row(remainder, len(attributes)))
                    else:
                        rows.append(_parse_dense_row(remainder))
                continue
            m = _ATTR_RE.match(line)
            if m:
                attributes.append(Attribute(_clean_attr_name(m.group("name")), m.group("type").strip()))
            continue

        if line.startswith("{") and line.endswith("}"):
            row = _parse_sparse_row(line, len(attributes))
        else:
            row = _parse_dense_row(line)
        if row:
            rows.append(row)

    if not attributes:
        raise ValueError("No @ATTRIBUTE declarations found")
    if not rows:
        raise ValueError("No @DATA rows found")
    return attributes, rows


def find_label_column(attributes: Sequence[Attribute]) -> int:
    lower_names = [a.name.strip().lower() for a in attributes]
    preferred = ["class", "label", "labels", "cluster", "clusters", "classlabel", "class_label", "target"]
    for candidate in preferred:
        if candidate in lower_names:
            return lower_names.index(candidate)
    # In this benchmark collection class is normally last. Keep a safe fallback.
    return len(attributes) - 1


def _build_encoding(values: Sequence[str], attribute: Attribute) -> Dict[str, int]:
    declared_values = _nominal_values(attribute.type_spec)
    if declared_values is not None:
        mapping = {value: idx for idx, value in enumerate(declared_values)}
        for value in values:
            if value not in mapping:
                mapping[value] = len(mapping)
        return mapping

    mapping: Dict[str, int] = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


def _validate_rows(attributes: Sequence[Attribute], rows: Sequence[Sequence[str]]) -> None:
    for row in rows:
        if len(row) != len(attributes):
            raise ValueError(
                f"Row has {len(row)} columns but ARFF declares {len(attributes)} attributes: {list(row)[:5]!r}"
            )


def _categorical_feature_mappings(
    attributes: Sequence[Attribute],
    rows: Sequence[Sequence[str]],
    label_idx: int,
) -> Dict[int, Dict[str, int]]:
    mappings: Dict[int, Dict[str, int]] = {}
    for idx, attribute in enumerate(attributes):
        if idx == label_idx:
            continue
        values = [row[idx] for row in rows]
        should_encode = not _is_numeric_attribute(attribute)
        has_nonnumeric = any(not _is_numeric_value(value) for value in values)
        if should_encode or has_nonnumeric:
            mappings[idx] = _build_encoding(values, attribute)
    return mappings


def _generated_column_names(
    categorical_feature_mappings: Dict[int, Dict[str, int]],
    attributes: Sequence[Attribute],
    label_idx: int,
) -> List[str]:
    names: List[str] = []
    for attr_idx, attribute in enumerate(attributes):
        if attr_idx == label_idx:
            continue
        base_name = _column_base_name(attribute, len(names))
        mapping = categorical_feature_mappings.get(attr_idx)
        if mapping is None:
            names.append(base_name)
            continue
        for value, code in mapping.items():
            category_name = str(value).strip() or str(code)
            names.append(f"{base_name}={category_name}")

    names.append("target")
    return names


def write_plain_csv(
    attributes: Sequence[Attribute],
    rows: Sequence[Sequence[str]],
    out_path: Path,
    write_header: bool = True,
) -> int:
    label_idx = find_label_column(attributes)
    _validate_rows(attributes, rows)
    categorical_feature_mappings = _categorical_feature_mappings(attributes, rows, label_idx)
    column_names = _generated_column_names(categorical_feature_mappings, attributes, label_idx)
    label_values = [row[label_idx] for row in rows]
    label_mapping = _build_encoding(label_values, attributes[label_idx])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        if write_header:
            writer.writerow(column_names)
        for row in rows:
            row = list(row)
            features: List[object] = []
            for idx, value in enumerate(row):
                if idx == label_idx:
                    continue
                mapping = categorical_feature_mappings.get(idx)
                if mapping is None:
                    features.append(value)
                    continue
                features.extend(1 if value == category else 0 for category in mapping)
            label = label_mapping[row[label_idx]]
            writer.writerow(features + [label])
            written += 1
    return written


def _normalize_dataset_name(name: str) -> str:
    stem = Path(str(name).strip()).stem.lower()
    return re.sub(r"[^a-z0-9]+", "", stem)


def _extract_dataset_names(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        names: List[str] = []
        for item in value:
            names.extend(_extract_dataset_names(item))
        return names
    if isinstance(value, dict):
        if "datasets" in value:
            return _extract_dataset_names(value["datasets"])
        if "name" in value:
            return _extract_dataset_names(value["name"])
        if "dataset" in value:
            return _extract_dataset_names(value["dataset"])
        return _extract_dataset_names(list(value.values()))
    return []


def load_online_phase_lookup(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        names = _extract_dataset_names(data)
    except json.JSONDecodeError:
        # Accept the existing JSON-like lookup shape:
        # { datasets: [ d1: "iris", d2: "glass" ] }
        names = re.findall(r":\s*['\"]([^'\"]+)['\"]", text)

    deduped: List[str] = []
    seen = set()
    for name in names:
        name = str(name).strip()
        key = _normalize_dataset_name(name)
        if not name or key in seen:
            continue
        deduped.append(name)
        seen.add(key)
    return deduped


def resolve_online_phase_lookup_path(raw_path: Optional[str]) -> Optional[Path]:
    if raw_path is None:
        default_path = Path(__file__).resolve().parent / DEFAULT_ONLINE_PHASE_LOOKUP
        return default_path if default_path.exists() else None
    if raw_path.strip().lower() in {"", "none", "off", "false", "0"}:
        return None
    return Path(raw_path).expanduser().resolve()


def _phase_output_path(
    arff_path: Path,
    input_root: Path,
    output_root: Path,
    online_lookup: Optional[Dict[str, str]],
    matched_online_names: set,
    seen_phase_names: Dict[str, set],
) -> Path:
    dataset_name = arff_path.stem
    normalized_name = _normalize_dataset_name(dataset_name)
    phase_dir = "offline_phase"
    if online_lookup is not None and normalized_name in online_lookup:
        phase_dir = "online_phase"
        matched_online_names.add(normalized_name)

    out_name = arff_path.with_suffix(".csv").name
    if out_name in seen_phase_names[phase_dir]:
        rel = arff_path.relative_to(input_root).with_suffix(".csv")
        out_name = "__".join(rel.parts)
    seen_phase_names[phase_dir].add(out_name)
    return output_root / phase_dir / out_name


def write_missing_online_phase_datasets(
    output_root: Path,
    online_dataset_names: Sequence[str],
    matched_online_names: set,
) -> List[str]:
    missing = [
        name
        for name in online_dataset_names
        if _normalize_dataset_name(name) not in matched_online_names
    ]
    missing_path = output_root / MISSING_ONLINE_PHASE_FILE
    with missing_path.open("w", encoding="utf-8") as f:
        for name in missing:
            f.write(f"{name}\n")
    return missing


def convert_tree(
    input_root: Path,
    output_root: Path,
    flatten: bool = False,
    write_header: bool = True,
    online_dataset_names: Optional[Sequence[str]] = None,
) -> Tuple[List[Tuple[Path, Path, int]], List[str]]:
    arff_files = sorted(input_root.rglob("*.arff"))
    if not arff_files:
        raise FileNotFoundError(f"No .arff files found under {input_root}")

    converted: List[Tuple[Path, Path, int]] = []
    seen_names = set()
    online_lookup = None
    matched_online_names = set()
    seen_phase_names = {"online_phase": set(), "offline_phase": set()}
    if online_dataset_names is not None:
        online_lookup = {_normalize_dataset_name(name): name for name in online_dataset_names}
        (output_root / "online_phase").mkdir(parents=True, exist_ok=True)
        (output_root / "offline_phase").mkdir(parents=True, exist_ok=True)

    for arff_path in arff_files:
        attributes, rows = read_arff(arff_path)
        if online_lookup is not None:
            out_path = _phase_output_path(
                arff_path,
                input_root,
                output_root,
                online_lookup,
                matched_online_names,
                seen_phase_names,
            )
        elif flatten:
            out_name = arff_path.with_suffix(".csv").name
            if out_name in seen_names:
                rel = arff_path.relative_to(input_root).with_suffix(".csv")
                out_name = "__".join(rel.parts)
            seen_names.add(out_name)
            out_path = output_root / out_name
        else:
            out_path = output_root / arff_path.relative_to(input_root).with_suffix(".csv")
        n = write_plain_csv(
            attributes,
            rows,
            out_path,
            write_header=write_header,
        )
        converted.append((arff_path, out_path, n))

    missing_online_phase = []
    if online_dataset_names is not None:
        missing_online_phase = write_missing_online_phase_datasets(
            output_root,
            online_dataset_names,
            matched_online_names,
        )
    return converted, missing_online_phase


def download_repo_zip(work_dir: Path) -> Path:
    zip_path = work_dir / "clustering-benchmark-master.zip"
    print(f"Downloading {GITHUB_ZIP_URL}", file=sys.stderr)
    with urllib.request.urlopen(GITHUB_ZIP_URL) as r, zip_path.open("wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work_dir)
    extracted = work_dir / "clustering-benchmark-master"
    if not extracted.exists():
        raise FileNotFoundError("Downloaded archive did not contain clustering-benchmark-master/")
    return extracted


def make_zip(output_root: Path, zip_path: Optional[Path] = None) -> Path:
    if zip_path is None:
        zip_path = output_root.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_root.rglob("*.csv")):
            zf.write(path, path.relative_to(output_root))
    return zip_path


def resolve_input(args: argparse.Namespace, temp_dir: Path) -> Path:
    provided = [bool(args.repo), bool(args.input), bool(args.download)]
    if sum(provided) != 1:
        raise SystemExit("Choose exactly one of --download, --repo, or --input")

    if args.download:
        repo_root = download_repo_zip(temp_dir)
        return repo_root / DATASET_REL_DIR
    if args.repo:
        repo_root = Path(args.repo).expanduser().resolve()
        dataset_root = repo_root / DATASET_REL_DIR
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
        return dataset_root
    return Path(args.input).expanduser().resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert clustering-benchmark ARFF files to CSV files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", action="store_true", help="Download deric/clustering-benchmark from GitHub first")
    group.add_argument("--repo", help="Path to a local clustering-benchmark repo checkout")
    group.add_argument("--input", help="Path to a directory containing ARFF files")
    parser.add_argument("--output", default="clustering_csv", help="Output directory for CSV files")
    parser.add_argument("--flatten", action="store_true", help="Write all CSVs into one directory instead of preserving subfolders")
    parser.add_argument(
        "--online-phase-lookup",
        default=None,
        help=(
            "JSON file listing datasets for online_phase output. "
            "Defaults to online_phase_data_lookup.json beside this script when present; use 'none' to disable."
        ),
    )
    parser.add_argument("--no-header", action="store_true", help="Do not write the default CSV header row")
    parser.add_argument("--zip", action="store_true", help="Also create a ZIP archive of the CSV output")
    parser.add_argument("--zip-path", help="Path for ZIP archive; defaults to <output>.zip")
    args = parser.parse_args(argv)

    lookup_path = resolve_online_phase_lookup_path(args.online_phase_lookup)
    online_dataset_names = None
    if lookup_path is not None:
        online_dataset_names = load_online_phase_lookup(lookup_path)
        if not online_dataset_names:
            raise ValueError(f"No dataset names found in online phase lookup: {lookup_path}")
        if args.flatten:
            print("Ignoring --flatten because online/offline phase output is enabled.", file=sys.stderr)

    output_root = Path(args.output).expanduser().resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        input_root = resolve_input(args, Path(td))
        converted, missing_online_phase = convert_tree(
            input_root,
            output_root,
            flatten=args.flatten,
            write_header=not args.no_header,
            online_dataset_names=online_dataset_names,
        )

    zip_created = None
    if args.zip:
        zip_created = make_zip(output_root, Path(args.zip_path).expanduser().resolve() if args.zip_path else None)

    print(f"Converted {len(converted)} ARFF files to CSV under {output_root}")
    total_rows = sum(n for _, _, n in converted)
    print(f"Rows written: {total_rows}")
    if online_dataset_names is not None:
        print(f"Online phase lookup: {lookup_path}")
        print(f"Online phase datasets not found: {len(missing_online_phase)}")
        print(f"Missing dataset list written: {output_root / MISSING_ONLINE_PHASE_FILE}")
        for name in missing_online_phase:
            print(f"Missing online phase dataset: {name}", file=sys.stderr)
    if zip_created:
        print(f"ZIP written: {zip_created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
