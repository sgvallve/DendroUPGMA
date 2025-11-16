# 🧩 DendroUPGMA — Hierarchical Clustering and Bootstrap Analysis Tool (Python version)

**Author:** Santi Garcia-Vallvé (QiN-URV)  
**Version:** 2025-11-13  
**Language:** Python 3.14 
**Dependencies:** numpy (required), toytree>3 (optional for PNG/SVG rendering and bootstrap support), argparse, scipy (for test.py)


This program builds **UPGMA** or **WPGMA** dendrograms from:
- a raw data matrix (rows = samples, columns = variables),
- a similarity matrix (values in [0, 1], symmetric),
- a distance matrix (values >= 0, symmetric).

It can compute similarity/distance matrices using several metrics,
optionally normalize the raw data, omit identical rows, and output:
- similarity and/or distance matrices,
- a Newick tree,
- a graphical dendrogram (if toytree is installed),
- the cophenetic correlation coefficient (CCC). It is the Pearson correlation between the original distances and the cophenetic distances derived from the tree. Measures how faithfully the dendrogram preserves the pairwise distances of the original data.
- bootstrap trees and support values (optional). A resampling method that evaluates the stability of clusters by repeatedly generating trees from random subsets of variables. Node support values represent the percentage of replicates in which each cluster appears.
---

## 🔧 How to run DendroUPGMA from the terminal

Download or clone the repository to your computer, and run the program from a terminal:

```
python UPGMAboot.py -in data.tsv -type data -delim '\t' -method 2 -boot_reps 10 -out results
```

This command reads the input file, computes the distance or similarity matrix, performs UPGMA/WPGMA clustering, and generates all output files in the specified folder.

To verify that everything is working correctly, you can run the included test script:
```
python test.py
```
This will execute a small example dataset and check whether the results are as expected.

## 🧬 Input formats supported

📑 **Column delimiter control**

By default, tabular input files are split using the regular expression '[\s,]+' which matches any whitespace or comma. This behaviour can be changed using the argument:

```
-delim <regex>
--delimiter <regex>

```

Examples: 

| Delimiter | Meaning                      |
| --------- | ---------------------------- |
| `'\t'`    | split only on tab characters |
| `','`     | comma-separated values       |
| `';'`     | semicolon-separated values   |
| `'[,;]+'` | comma or semicolon           |
| default   | whitespace or comma          |

Specifying a custom delimiter allows sample names or column names to contain spaces without being split unintentionally.

The delimiter used for parsing is recorded in the run_log.txt file.

The program automatically detects the input file type:

### 1️⃣ Data matrix (`.tsv` or `.csv`)
- Rows = samples, columns = variables.  
- First column: sample identifiers.  
- First row: variable names (optional, but must start with # if present).

```
#A   Var1   Var2   Var3
a1  1.2    3.4    2.8
a2  0.8    2.5    3.0
```

### 2️⃣ FASTA-like numeric format
```
>a1
1 1 1 1 1 1 1 1 1 1
>a2
0 0 0 0 0 0 1 1 0 0
>a3
1 1 1 0 1 1 0 1 0 1
```

### 3️⃣ Similarity matrix
```
#     A      B      C
A   1.000  0.923  0.811
B   0.923  1.000  0.812
C   0.811  0.812  1.000
```

### 4️⃣ Distance matrix
```
#     A      B      C
A   0.000  0.120  0.189
B   0.120  0.000  0.166
C   0.189  0.166  0.000
```

---

### ⚙️ Command-line parameters

The following command-line arguments are supported:
| Parameter                                 | Description                                                                                                     | Default        | Allowed values / Notes                                                                                                                                                                                           |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-in <file>`<br>`--input <file>`          | Input file (data matrix, FASTA-numeric, similarity matrix, or distance matrix).                                 | **Required**   | Any readable text file.                                                                                                                                                                                          |
| `-out <folder>`<br>`--output <folder>`    | Output directory where all files will be written.                                                               | `UPGMA_output` | Any valid directory path (will be created if missing).                                                                                                                                                           |
| `-type <mode>`                            | Specifies the interpretation of the input file.                                                                 | `auto`         | `auto` = automatic detection; `data` = raw data matrix or FASTA-numeric; `similarity` = similarity matrix (must be square, symmetric, diag=1); `distance` = distance matrix (must be square, symmetric, diag=0). |
| `-method <int>`                           | Similarity/distance method (only used when `-type data`).                                                       | `1`            | `1` Pearson; `2` Jaccard; `3` Dice; `4` MSD; `5` RMSD; `6` Euclidean; `7` Manhattan.                                                                                                                             |
| `-linkage <type>`                         | Clustering method.                                                                                              | `upgma`        | `upgma` (unweighted), `wpgma` (weighted).                                                                                                                                                                        |
| `-remove_zero_columns <0/1>`              | Remove columns where all values are zero.                                                                       | `0`            | `1` = enabled, `0` = disabled.                                                                                                                                                                                   |
| `-remove_constant_columns <0/1>`          | Remove columns where all values are identical.                                                                  | `0`            | `1` = enabled, `0` = disabled.                                                                                                                                                                                   |
| `-omit_identical_rows <0/1>`              | Remove duplicate rows (keep first occurrence).                                                                  | `0`            | `1` = enabled, `0` = disabled.                                                                                                                                                                                   |
| `-normalize_data <0/1>`                   | Z-score normalize columns (mean 0, std 1). Only for continuous-distance methods (4–7). Ignored for binary data. | `0`            | `1` = enabled, `0` = disabled.                                                                                                                                                                                   |
| `-scale_d <float>`                        | Scale the final distance matrix by the given factor.                                                            | `1.0`          | Any positive float. Does not affect tree topology, only branch lengths.                                                                                                                                          |
| `-boot_reps <int>`                        | Number of bootstrap replicates (resampling columns with replacement).                                           | `1`            | Integer between 1 and 100. `1` = no bootstrap.                                                                                                                                                                   |
| `-delim <regex>`<br>`--delimiter <regex>` | Regular expression used to split columns in input files (FASTA-numeric or tabular).                             | `[\s,]+`       | Any valid Python regex. Examples: `'\t'`, `','`, `';'`, `'[,;]+'`.                                                                                                                                               |
| `-tree_layout <r/c/d/unr>` | Tree orientation in graphical output: `'r'` = right-oriented dendrogram, `'c'` = circular tree, `unr` = unrooted, `d` = down-oriented | `r`     | `r`, `c`, `unr`,`d` |
| `-h`, `--help`                            | Show help message and exit.                                                                                     | –              | Built-in.                                                                                                                                                                                                        |

## ⚙️ Preprocessing options

All preprocessing options are **off by default**.

**Execution order when multiple options are active:**  
`remove_zero_columns → remove_constant_columns → omit_identical_rows → normalize_data`

---

## 🧮 Methods available for data matrix and FASTA-numeric input

When the input file is a data matrix or a FASTA-like numeric file (-type data), the following similarity and distance methods can be applied:

| Code | Method | Type | Notes |
|------|---------|------|-------|
| 1 | Pearson correlation | Similarity | Scaled to [0–1] |
| 2 | Jaccard similarity | Similarity | Binary data only |
| 3 | Dice similarity | Similarity | Binary data only |
| 4 | Mean Squared Difference (MSD) | Distance | Sensitive to outliers |
| 5 | Root MSD (RMSD) | Distance | √(MSD) |
| 6 | Euclidean distance | Distance | Default for continuous data |
| 7 | Manhattan distance | Distance | Uses absolute differences |

These methods are ignored when the user provides -type similarity or -type distance, in which case the program directly uses the supplied matrix. In these methods the program performs strict validation:

- the input matrix must be square (n × n),
- the matrix must be symmetric,
- the diagonal must be:
  - 1.0 for similarity matrices,
  - 0.0 for distance matrices.

If any of these conditions is not met, execution stops and the problem is reported in the log file. This ensures correctness of user-provided matrices.

---

## 📏 Scaling option

Use `-scale_d <factor>` to scale distances as:

```
d_scaled = (1 − s) × factor
```

Scaling is applied **after normalization** and does **not** affect the topology of the dendrogram.

---


## 🌿 Bootstrap analysis

Use the option `-boot_reps <N>` to perform bootstrap resampling (1–100 replicates). This option requires the toytree library:

| Option | Description |
|---------|--------------|
| `-boot_reps 1` | Build only the main tree (no bootstrap). |
| `-boot_reps N` | Perform *N* bootstrap replicates (resampling columns with replacement). Each replicate generates a new tree. Node support values are computed as the % of replicates that contain the same clade. |

**Bootstrap outputs:**
| File | Description |
|------|--------------|
| `bootstrap_trees.txt` | One Newick tree per line (all bootstrap replicates). |
| `tree_bootstrap.newick` | Main tree annotated with bootstrap support values. |
| `dendrogram_bootstrap_<linkage>.png` | Graphical bootstrap tree (if Toytree installed). |


## 📤 Output files

All generated files are stored in a dedicated folder (default: `UPGMA_output/`):

| File | Description |
|------|--------------|
| `similarity_matrix.tsv` | Similarity matrix (if applicable) |
| `distance_matrix.tsv` | Distance matrix |
| `tree.newick` | Dendrogram in Newick format |
| `dendrogram_<linkage>.png` | Graphical dendrogram (optional) |
| `cophenetic.txt` | the cophenetic correlation coefficient (CCC). Not calculated if the number of leaves is >250 |
| `run_log.txt` | Execution log (timestamp, preprocessing steps, CCC, etc.) |


---

## 🌳 Graphical dendrograms (Toytree)

If the **Toytree** library is installed, the program automatically generates graphical dendrograms in PNG or SVG format.

- Main tree: `dendrogram_<linkage>.png`
- Bootstrap tree (if applicable): `dendrogram_bootstrap_<linkage>.png`
- White background, black branches and labels.
- Bootstrap support values displayed in red if available.

The tree can be drawn using different layouts selected with -tree_layout:

- r (right, default)
- d (down)
- unr (unrooted)
- c (circular)

The size of the tree canvas and the font size of tip labels are automatically scaled according to the number of leaves to improve readability.

If Toytree is not installed, the program still runs normally and prints a warning:

[WARN] Toytree is not installed. No graphical output will be generated.

---

## 🧾 Log file contents

The log file records:
- Timestamp and runtime information  
- Delimiter used for splitting columns  
- Detected input type and matrix dimensions  
- Preprocessing steps applied  
- Normalization/scaling parameters  
- Method and arguments used  
- Paths to generated files  
- Cophenetic correlation coefficient (CCC)  
- Bootstrap summary (if available)

---

## 🧪 Examples

**Example 1:**
```bash
python UPGMAboot.py -input data.tsv -method 1 -delim '\t'
```

**Example 2 – Binary data with preprocessing:**
```bash
python UPGMAboot.py -in binary_data.txt -method 2 -remove_zero_columns 1
```

**Example 3 – Continuous data with normalization and scaling:**
```bash
python UPGMAboot.py -in data.tsv -method 6 -normalize_data 1 -scale_d 10 -out out_folder
```

**Example 4 – 100 bootstrap replicates:**
```bash
python UPGMAboot.py -in data.tsv -out out_folder -method 6 -boot_reps 100 -tree_layout r
```

---

## 🧠 Notes

- Jaccard and Dice require strictly binary (0/1) data.  
- The program automatically detects binary datasets.  
- Normalization and zero-column removal have no effect on uniform data.  
- Output matrices are symmetrical with diagonals = 1.0 (similarity) or 0.0 (distance).  
- Sample and column order are preserved.  
- A comprehensive test suite (`test.py`) validates all numeric operations and preprocessing logic.

---

© 2025 — Santi Garcia-Vallvé, QiN Research Group, Universitat Rovira i Virgili (URV)
