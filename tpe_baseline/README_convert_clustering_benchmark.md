# Conversion Report: ARFF to CSV

## 1. Scope

This report documents the behavior of [`convert_clustering_benchmark_to_csv.py`](convert_clustering_benchmark_to_csv.py), which converts ARFF datasets from [`deric/clustering-benchmark`](https://github.com/deric/clustering-benchmark) into CSV files with numeric data rows and a header row by default.

The benchmark datasets are expected under:

```text
src/main/resources/datasets
```

Typical dataset groups in this repository:

| Group | Example |
|---|---|
| `artificial/` | `src/main/resources/datasets/artificial/2d-4c.arff` |
| `real-world/` | `src/main/resources/datasets/real-world/iris.arff` |

## 2. Output Contract

Each generated CSV is designed to contain numeric data rows. By default, the first row contains generated column names:

```csv
feature_1,feature_2,...,feature_n,target
0.12,0.34,...,0.56,1
```

| Property | Behavior |
|---|---|
| Header row | Written by default; disable with `--no-header` |
| Row index | Not written |
| ARFF comments | Removed |
| ARFF `@RELATION` metadata | Removed |
| ARFF `@ATTRIBUTE` metadata | Used to generate readable CSV headers |
| Numeric feature columns | Passed through unchanged |
| Categorical feature columns | One-hot encoded |
| String feature columns | One-hot encoded |
| Class/label column | Integer encoded and written as the final CSV column named `target` |
| Column interpretation | Available from the CSV header row |
| Directory layout | Split into `online_phase/` and `offline_phase/` when an online phase lookup is active; otherwise preserved unless `--flatten` is used |

## 3. Encoding Rationale

Feature columns and label columns are encoded differently because they have different statistical roles.

| Column role | Encoding | Reason |
|---|---|---|
| Numeric feature | Passthrough | Numeric distances and magnitudes are already meaningful. |
| Categorical/string feature | One-hot | Avoids creating artificial category order or distance. |
| Class/label | Integer label code | Keeps the target compact as one final column. |

For nominal ARFF attributes, the declared category order is used. This is more deterministic than a generic alphabetical label encoder because the source file already specifies the nominal order.

Example ARFF declaration:

```arff
@ATTRIBUTE class {CYT,NUC,MIT,ME3,ME2,ME1,EXC,VAC,POX,ERL}
```

Resulting class mapping:

| Class | Code |
|---|---:|
| `CYT` | 0 |
| `NUC` | 1 |
| `MIT` | 2 |
| `ME3` | 3 |
| `ME2` | 4 |
| `ME1` | 5 |
| `EXC` | 6 |
| `VAC` | 7 |
| `POX` | 8 |
| `ERL` | 9 |

For free string attributes without a nominal declaration, categories are assigned in first-seen order within that dataset.

## 4. Processing Pipeline

The script performs the following operations for every run:

| Step | Operation | Result |
|---:|---|---|
| 1 | Resolve input mode | Determine whether data comes from `--download`, `--repo`, or `--input`. |
| 2 | Resolve online phase lookup | Load `online_phase_data_lookup.json` if present, unless disabled with `--online-phase-lookup none`. |
| 3 | Prepare output | Remove the existing output directory and recreate it. |
| 4 | Discover ARFF files | Recursively find `*.arff` files below the input root. |
| 5 | Parse ARFF metadata | Read attribute names and type declarations. |
| 6 | Parse ARFF data | Read dense or sparse rows. |
| 7 | Detect label column | Prefer known names such as `class`, `label`, or `target`; otherwise use the last attribute. |
| 8 | Validate rows | Check every row has the declared attribute count. |
| 9 | Encode features | One-hot encode categorical/string features and pass numeric features through. |
| 10 | Encode label | Convert class labels to integer codes. |
| 11 | Choose output phase | Write lookup matches to `online_phase/` and all other datasets to `offline_phase/` when phase lookup is active. |
| 12 | Write CSV | Emit a CSV with a header row by default and numeric data rows. |
| 13 | Write missing lookup report | Write lookup entries without a matching ARFF file to `missing_online_phase_datasets.txt` and log them. |
| 14 | Package output | Optionally create a ZIP archive. |
| 15 | Report summary | Print converted file count, row count, and phase lookup summary. |

## 5. Input Modes

Exactly one input mode is required.

| Argument | Meaning | Input root used |
|---|---|---|
| `--download` | Download upstream `master` ZIP from GitHub. | `clustering-benchmark-master/src/main/resources/datasets` inside a temporary directory |
| `--repo PATH` | Use a local `clustering-benchmark` repository checkout. | `PATH/src/main/resources/datasets` |
| `--input PATH` | Use a directory directly. | `PATH` |

`--download` fetches:

```text
https://github.com/deric/clustering-benchmark/archive/refs/heads/master.zip
```

Use `--repo` for reproducible conversions from a specific local checkout. Use `--input` for tests, subsets, or custom ARFF folders.

## 6. ARFF Parsing

### 6.1 File Decoding

| Attempt | Encoding |
|---:|---|
| 1 | `utf-8-sig` |
| 2 | `latin-1` |
| Fallback | UTF-8 with replacement characters |

### 6.2 Metadata Parsing

| ARFF construct | Converter behavior |
|---|---|
| `% comment` | Removed unless `%` is inside quotes |
| `@RELATION` | Ignored |
| `@ATTRIBUTE` | Parsed for name and type declaration |
| `@DATA` | Starts data-row parsing |
| Blank lines | Ignored |

Attribute names may be unquoted, single-quoted, or double-quoted. Surrounding quotes are removed.

### 6.3 Dense Rows

Comma-separated dense rows are parsed with Python's `csv.reader`:

```arff
5.1,3.5,1.4,0.2,Iris-setosa
```

Whitespace-separated dense rows are parsed with `shlex.split`:

```arff
ADT1_YEAST  0.58  0.61  0.47  0.13  0.50  0.00  0.48  0.22  MIT
```

`shlex.split` preserves quoted values while still allowing whitespace-delimited UCI-style rows.

### 6.4 Sparse Rows

Sparse rows of this form are supported:

```arff
{0 1.0, 4 blue, 8 class_a}
```

Sparse-row behavior:

| Case | Behavior |
|---|---|
| Unspecified values | Filled with `"0"` |
| Index below `0` | Error |
| Index beyond declared attribute count | Error |
| Entry without both index and value | Error |

## 7. Label Detection

The converter searches attribute names case-insensitively in this order:

| Priority | Name |
|---:|---|
| 1 | `class` |
| 2 | `label` |
| 3 | `labels` |
| 4 | `cluster` |
| 5 | `clusters` |
| 6 | `classlabel` |
| 7 | `class_label` |
| 8 | `target` |

If none is present, the final ARFF attribute is treated as the label.

## 8. Feature Encoding

Feature columns are all columns except the detected label column.

| Source column condition | Feature encoding |
|---|---|
| ARFF type is `numeric`, `real`, or `integer`, and observed values are numeric | Passthrough |
| ARFF type is nominal, for example `{red,green,blue}` | One-hot |
| ARFF type is `string` | One-hot |
| Any nonnumeric observed feature value | One-hot |

One-hot example:

| Original `color` value | `color=red` | `color=green` | `color=blue` |
|---|---:|---:|---:|
| `red` | 1 | 0 | 0 |
| `green` | 0 | 1 | 0 |
| `blue` | 0 | 0 | 1 |

The generated CSV includes a header by default, so the row for `green` is written as:

```csv
color=red,color=green,color=blue
0,1,0
```

The meaning of the generated columns is recorded in the CSV header. Use `--no-header` only when the downstream tool requires numeric data rows without a header.

## 8.1 Column Names

Column names are generated for the transformed output columns:

| Output column type | Column name |
|---|---|
| Numeric passthrough feature | Source ARFF attribute name |
| One-hot feature | `attribute=value` |
| Encoded label | Always `target` |

If a usable source feature attribute name is unavailable, the converter falls back to deterministic dummy names such as `a`, `b`, `c`, and continues with spreadsheet-style names after `z`. The encoded label column is always named `target`, regardless of the source ARFF label attribute name.

## 9. Label Encoding

The label column is encoded as a single integer column and appended last.

| Source label type | Label-code order |
|---|---|
| Nominal ARFF declaration | Declared order |
| Undeclared value in nominal column | Appended after declared values |
| No nominal declaration | First-seen order |

Example:

```arff
@ATTRIBUTE class {CYT,NUC,MIT}
```

```csv
0.58,0.61,0.47,2
```

In this example, `2` means `MIT`.

## 10. Output Layout

### 10.1 Phase Layout

When `online_phase_data_lookup.json` is present beside the converter, or when a lookup is passed with `--online-phase-lookup`, the script writes phase folders instead of preserving the benchmark source folders.

Lookup file structure:

```json
{
  "datasets": [
    "iris",
    "glass",
    "aggregation"
  ]
}
```

Input:

```text
src/main/resources/datasets/artificial/2d-4c.arff
src/main/resources/datasets/real-world/iris.arff
```

Output:

```text
clustering_csv/offline_phase/2d-4c.csv
clustering_csv/online_phase/iris.csv
clustering_csv/missing_online_phase_datasets.txt
```

Datasets whose normalized ARFF filename matches an entry in the lookup are written to `online_phase/`. All other discovered ARFF files are written to `offline_phase/`.

`missing_online_phase_datasets.txt` contains lookup entries that were not found in the discovered ARFF files. Each missing dataset is also logged as:

```text
Missing online phase dataset: <dataset name>
```

Dataset matching is case-insensitive and ignores punctuation such as hyphens and underscores, so `2d-25c-no0.arff` matches `2d-25c-no0` in the lookup.

If duplicate output filenames occur inside the same phase folder, the script joins the original relative path components with `__`.

### 10.2 Preserved Directory Layout

Disable phase routing with `--online-phase-lookup none` to preserve the source directory layout:

```text
clustering_csv/artificial/2d-4c.csv
clustering_csv/real-world/iris.csv
```

### 10.3 Flattened Layout

With `--flatten`, all generated files are written directly into the output directory when phase routing is disabled:

```text
clustering_csv/2d-4c.csv
clustering_csv/iris.csv
```

If an online phase lookup is active, the script ignores `--flatten` and writes `online_phase/` and `offline_phase/` instead.

If flattening would create duplicate names, the script joins relative path components with `__`.

Example:

```text
artificial__example.csv
real-world__example.csv
```

## 11. Checks And Failure Conditions

| Stage | Check | Failure behavior |
|---|---|---|
| Input mode | Exactly one of `--download`, `--repo`, or `--input` is supplied | Exits with an argument error |
| Repository input | `PATH/src/main/resources/datasets` exists | Raises `FileNotFoundError` |
| Online phase lookup | Lookup file contains at least one dataset name | Raises `ValueError` |
| File discovery | At least one `.arff` file exists | Raises `FileNotFoundError` |
| ARFF structure | At least one `@ATTRIBUTE` is present | Raises `ValueError` |
| ARFF structure | At least one data row is present | Raises `ValueError` |
| Sparse parsing | Sparse index is within the declared attribute range | Raises `ValueError` |
| Sparse parsing | Sparse entry contains both index and value | Raises `ValueError` |
| Row consistency | Parsed row length equals declared attribute count | Raises `ValueError` with row preview |
| Output | Parent directories are created as needed | Fails only if filesystem write fails |
| Missing online datasets | Lookup entries without matching ARFF files are recorded | Writes `missing_online_phase_datasets.txt` and logs each missing dataset |
| ZIP output | Existing ZIP at target path is replaced | Fails only if filesystem write fails |

Encoding-specific checks:

| Check | Behavior |
|---|---|
| Declared nominal values | Preserve declared order |
| Undeclared observed categories | Append in first-seen order |
| Nonnumeric values in feature columns | Treat the column as categorical so text is not written to CSV |

## 12. Side Effects

The output directory is treated as generated output.

If the selected output path already exists, it is removed before conversion:

```python
shutil.rmtree(output_root)
```

Do not point `--output` at a directory containing files you want to keep.

Temporary files created by `--download` are stored in a temporary directory and removed automatically after conversion.

One-hot encoding may significantly increase column count. For example, `yeast.arff` contains a string identifier feature named `SequenceName`; this becomes one binary column per observed sequence name.

## 13. Command-Line Reference

### 13.1 Required Input Mode

Choose exactly one:

| Argument | Value | Description |
|---|---|---|
| `--download` | none | Download upstream `master` and convert its dataset directory. |
| `--repo` | repository path | Convert a local `clustering-benchmark` checkout. |
| `--input` | directory path | Convert `.arff` files under an arbitrary directory. |

### 13.2 Optional Arguments

| Argument | Value | Default | Description |
|---|---|---|---|
| `--output` | directory path | `clustering_csv` | Output directory for generated files. |
| `--flatten` | none | disabled | Write all generated files directly into the output directory. |
| `--online-phase-lookup` | JSON file path, or `none` | `online_phase_data_lookup.json` if present | Split output into `online_phase/` for lookup datasets and `offline_phase/` for all others. Missing lookup datasets are written to `missing_online_phase_datasets.txt` and logged. |
| `--no-header` | none | disabled | Do not write the default CSV header row. |
| `--zip` | none | disabled | Create a ZIP archive containing generated `.csv` files. |
| `--zip-path` | file path | `<output>.zip` | Explicit ZIP archive path. |

## 14. Examples

Convert the bundled datasets:

```bash
python3 convert_clustering_benchmark_to_csv.py \
  --input src/main/resources/datasets \
  --output clustering_csv
```

Convert only the test folder:

```bash
python3 convert_clustering_benchmark_to_csv.py \
  --input test \
  --output clustering_csv_test
```

Download upstream data and create a ZIP archive:

```bash
python3 convert_clustering_benchmark_to_csv.py \
  --download \
  --output clustering_csv \
  --zip
```

With `online_phase_data_lookup.json` present beside the converter, output is split into:

```text
clustering_csv/online_phase/
clustering_csv/offline_phase/
clustering_csv/missing_online_phase_datasets.txt
```

Disable the phase split:

```bash
python3 convert_clustering_benchmark_to_csv.py \
  --download \
  --output clustering_csv \
  --online-phase-lookup none
```

Convert a local repository checkout:

```bash
git clone https://github.com/deric/clustering-benchmark.git

python3 convert_clustering_benchmark_to_csv.py \
  --repo clustering-benchmark \
  --output clustering_csv \
  --zip
```

Flatten the output:

```bash
python3 convert_clustering_benchmark_to_csv.py \
  --input my_arff_files \
  --output flat_csv \
  --online-phase-lookup none \
  --flatten
```

Skip CSV headers:

```bash
python3 convert_clustering_benchmark_to_csv.py \
  --input src/main/resources/datasets \
  --output clustering_csv \
  --no-header
```

Write the ZIP archive to an explicit path:

```bash
python3 convert_clustering_benchmark_to_csv.py \
  --input src/main/resources/datasets \
  --output clustering_csv \
  --zip \
  --zip-path clustering-benchmark-csv.zip
```

## 15. Expected Summary Output

Default run:

```text
Converted <N> ARFF files to CSV under <output directory>
Rows written: <M>
```

Run with an active online phase lookup:

```text
Converted <N> ARFF files to CSV under <output directory>
Rows written: <M>
Online phase lookup: <lookup path>
Online phase datasets not found: <K>
Missing dataset list written: <output directory>/missing_online_phase_datasets.txt
```

For each missing lookup entry, the script also logs:

```text
Missing online phase dataset: <dataset name>
```

Run with `--zip`:

```text
Converted <N> ARFF files to CSV under <output directory>
Rows written: <M>
ZIP written: <zip path>
```

## 16. Yeast Dataset Case Study

`test/yeast.arff` is a useful regression fixture because it combines two edge cases:

| Edge case | Why it matters |
|---|---|
| Whitespace-delimited dense rows | A comma-only parser reads each row as one field. |
| `SequenceName string` feature | Text must not appear in the generated CSV. |

Relevant ARFF declarations:

```arff
@ATTRIBUTE SequenceName string
@ATTRIBUTE mcg real
@ATTRIBUTE gvh real
@ATTRIBUTE alm real
@ATTRIBUTE mit real
@ATTRIBUTE erl real
@ATTRIBUTE pox real
@ATTRIBUTE vac real
@ATTRIBUTE nuc real
@ATTRIBUTE class {CYT,NUC,MIT,ME3,ME2,ME1,EXC,VAC,POX,ERL}
```

Representative input row:

```arff
ADT1_YEAST  0.58  0.61  0.47  0.13  0.50  0.00  0.48  0.22  MIT
```

Representative conversion behavior:

| Source value | Output behavior |
|---|---|
| `SequenceName = ADT1_YEAST` | One `SequenceName=ADT1_YEAST` indicator is `1`; other sequence indicators are `0`. |
| `mcg = 0.58` | Numeric passthrough |
| `gvh = 0.61` | Numeric passthrough |
| `class = MIT` | Integer label code `2` |

Observed generated shape for `yeast.csv`:

| Rows | Columns | Notes |
|---:|---:|---|
| 1,484 | 1,471 | 1,462 one-hot sequence columns, 8 numeric feature columns, 1 label column |

With `--no-header`, the generated CSV data rows were checked for alphabetic characters; none were found.

## 17. Verification Performed

Syntax check:

```bash
python3 -m py_compile convert_clustering_benchmark_to_csv.py
```

Yeast fixture conversion:

```bash
python3 convert_clustering_benchmark_to_csv.py \
  --input test \
  --output /private/tmp/clustering_csv_test_encoded
```

Observed result:

```text
Converted 1 ARFF files to CSV under /private/tmp/clustering_csv_test_encoded
Rows written: 1484
```

Full bundled dataset conversion:

```bash
python3 convert_clustering_benchmark_to_csv.py \
  --input src/main/resources/datasets \
  --output /private/tmp/clustering_csv_all_encoded
```

Observed result:

```text
Converted 146 ARFF files to CSV under /private/tmp/clustering_csv_all_encoded
Rows written: 557585
```

## 18. Limitations

| Limitation | Consequence |
|---|---|
| Numeric values are not standardized or scaled | Downstream tools must handle scaling if needed. |
| Missing values are not imputed | Missing-value handling remains a downstream responsibility. |
| High-cardinality string features are one-hot encoded | CSV files can become wide. |
| Dense rows must be comma-separated or whitespace-separated | Other dense row syntaxes are out of scope. |
| Sparse support is limited to `{index value, ...}` | More complex ARFF sparse forms may fail. |
| Feature header names are generated from ARFF metadata | Keep the header row when downstream tools need column interpretation. The final label header is always `target`. |

## 19. Reproducibility Notes

Runs using `--repo` or `--input` are reproducible with respect to the local files provided to the script.

Runs using `--download` depend on the upstream GitHub `master` branch at the time the command is executed. For fully reproducible conversion, use a local clone checked out to a specific commit and pass it with `--repo`.
