#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UPGMAboot.py
-------------
Builds UPGMA or WPGMA dendrograms from:
  - a raw data matrix (rows = samples, columns = variables),
  - a similarity matrix (values in [0, 1], symmetric),
  - a distance matrix (values >= 0, symmetric).

It can compute similarity/distance matrices using several metrics,
optionally normalize the raw data, omit identical rows, and output:
  - similarity and/or distance matrices,
  - a Newick tree,
  - a graphical dendrogram (Toytree if available, PNG),
  - the cophenetic correlation coefficient (CCC).

Developed by: Santi Garcia-Vallvé (QiN-URV)
Version: 2025-11-02 (Toytree + bootstrap version)
Language: Python 3

Dependencies: numpy (required), toytree + toyplot (optional, for PNG rendering), argparse, scipy (for test.py)
"""

import argparse
import datetime
import os
import sys
import re
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# Toytree + toyplot are recommended for tree rendering. We gracefully degrade if missing.
try:
    import toytree
    import toyplot.png
    import toyplot.svg
    _HAS_TOYTREE = True
except Exception:
    _HAS_TOYTREE = False

# ------------------------------------------------------------
# 1. INPUT READERS
# ------------------------------------------------------------

def read_fasta_numeric(path: str, delimiter: str = r'[\s,]+') -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Read a FASTA-like numeric file:
      >row_label
      1 2 3 ...
      >row_label2
      ...

    Returns
    -------
    row_labels : list of str
    data       : np.ndarray (rows = samples, columns = variables)
    col_labels : list of str (Var1..VarN)
    """
    row_labels: List[str] = []
    rows: List[List[float]] = []
    current_label: Optional[str] = None

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                current_label = line[1:].strip()
                row_labels.append(current_label)
            else:
                parts = re.split(delimiter, line)
                if len(parts) <= 1:
                     raise ValueError("Input line could not be split into multiple fields. "
                            "This suggests that the selected delimiter is incorrect.\n"
                            f"Line read: {line!r}\n"
                            f"Delimiter used: {delimiter!r}")
                values = [float(x) for x in parts if x]
                rows.append(values)

    data = np.array(rows, dtype=float)
    n_cols = data.shape[1]
    col_labels = [f'Var{i+1}' for i in range(n_cols)]
    return row_labels, data, col_labels


def _is_number(x: str) -> bool:
    """Return True if x can be converted to float, else False."""
    try:
        float(x)
        return True
    except ValueError:
        return False


def read_tabular(path: str,  delimiter: str = r'[\s,]+') -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Read a generic tabular file (TSV/CSV/whitespace).
    If the first non-comment line starts with numbers, rows are unnamed and
    default labels (R1, R2, ...) are used. Otherwise, the first column is used
    as row labels and the rest as numeric data.
    """
    rows: List[List[float]] = []
    row_labels: List[str] = []
    col_labels: Optional[List[str]] = None
    first_data_line = True

    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if (not line) or line.startswith('#'):
                continue
            parts = re.split(delimiter, line)
            if len(parts) <= 1:
                  raise ValueError("Input line could not be split into multiple fields. "
                         "This suggests that the selected delimiter is incorrect.\n"
                         f"Line read: {line!r}\n"
                         f"Delimiter used: {delimiter!r}")
            # --- Detect header line ---
            if first_data_line  and not all(_is_number(p) for p in parts[1:]):
            # This is a header
                col_labels = parts[1:]  # Skip first column (row labels)
                first_data_line = False
                continue
            
            first_data_line = False
            row_labels.append(parts[0])
            rows.append([float(x) for x in parts[1:]])
            
    data = np.array(rows, dtype=float)
    if col_labels is None:
        col_labels = [f'Var{i+1}' for i in range(data.shape[1])]
    return row_labels, data, col_labels


def detect_input_type(row_labels: List[str], data: np.ndarray) -> str:
    """
    Try to detect input type based on matrix properties.

    Returns one of: 'data', 'similarity', 'distance'.
    """
    n, m = data.shape
    if n == m:
        if not np.allclose(data, data.T):
            return 'data'
        # Check diagonal values
        diag = np.diag(data)
        if np.allclose(diag, 1.0):
            return 'similarity'
        if np.allclose(diag, 0.0):
            return 'distance'
    return 'data'


# ------------------------------------------------------------
# 2. PREPROCESSING HELPERS
# ------------------------------------------------------------

def remove_zero_columns(data: np.ndarray, col_labels: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Remove columns that are entirely zero.

    Returns
    -------
    filtered_data, filtered_col_labels, removed_col_labels
    """
    col_sums = np.sum(np.abs(data), axis=0)
    keep_mask = col_sums != 0
    removed = [lab for lab, k in zip(col_labels, keep_mask) if not k]
    return data[:, keep_mask], [lab for lab, k in zip(col_labels, keep_mask) if k], removed


def remove_constant_columns(data: np.ndarray, col_labels: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Remove columns that have the same value in all rows.

    Returns
    -------
    filtered_data, filtered_col_labels, removed_col_labels
    """
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    keep_mask = (maxs - mins) != 0
    removed = [lab for lab, k in zip(col_labels, keep_mask) if not k]
    return data[:, keep_mask], [lab for lab, k in zip(col_labels, keep_mask) if k], removed


def omit_identical_rows(
        data: np.ndarray,
        row_labels: List[str],
        input_type: str
    ) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Removes duplicated taxa depending on input type.

    Returns:
        data          -> updated matrix
        row_labels    -> updated labels
        omitted       -> list of labels removed
    """

    n = len(data)
    to_remove = set()

    # === CASE 1: distance matrix → duplicates where distance == 0 ===
    if input_type == "distance":
        for i in range(n):
            for j in range(i + 1, n):
                if data[i, j] == 0:
                    to_remove.add(j)

        if not to_remove:
            return data, row_labels, []

        idx = sorted(to_remove)
        omitted = [row_labels[i] for i in idx]

        data = np.delete(data, idx, axis=0)
        data = np.delete(data, idx, axis=1)
        row_labels = [lab for i, lab in enumerate(row_labels) if i not in to_remove]

        return data, row_labels, omitted

    # === CASE 2: similarity matrix → duplicates where similarity == 1 ===
    if input_type == "similarity":
        for i in range(n):
            for j in range(i + 1, n):
                if data[i, j] == 1:
                    to_remove.add(j)

        if not to_remove:
            return data, row_labels, []

        idx = sorted(to_remove)
        omitted = [row_labels[i] for i in idx]

        data = np.delete(data, idx, axis=0)
        data = np.delete(data, idx, axis=1)
        row_labels = [lab for i, lab in enumerate(row_labels) if i not in to_remove]

        return data, row_labels, omitted

    # === CASE 3: data or fasta → identical rows ===
    for i in range(n):
        for j in range(i + 1, n):
            if np.array_equal(data[i], data[j]):
                to_remove.add(j)

    if not to_remove:
        return data, row_labels, []

    idx = sorted(to_remove)
    omitted = [row_labels[i] for i in idx]

    data = np.delete(data, idx, axis=0)
    row_labels = [lab for i, lab in enumerate(row_labels) if i not in to_remove]

    return data, row_labels, omitted


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Z-score normalization per column (variable).

    Each column is transformed to mean 0 and std 1 (if std == 0, column
    is left unchanged).
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    return (data - mean) / std


def is_binary_matrix(data: np.ndarray) -> bool:
    """
    Check if all entries are 0/1 (within a small numerical tolerance).
    """
    return np.all((data == 0) | (data == 1))


# ------------------------------------------------------------
# 3. SIMILARITY / DISTANCE METHODS
# ------------------------------------------------------------

def pearson_similarity(data: np.ndarray) -> np.ndarray:
    """
    Compute Pearson similarity matrix scaled to [0, 1].

    Pearson r is computed row-wise, then transformed as:
        s = (r + 1) / 2
    """
    n = data.shape[0]
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            x = data[i]
            y = data[j]
            xm = x.mean()
            ym = y.mean()
            num = np.sum((x - xm) * (y - ym))
            den = np.sqrt(np.sum((x - xm) ** 2) * np.sum((y - ym) ** 2))
            if den == 0:
                r = 0.0
            else:
                r = num / den
            s = (r + 1.0) / 2.0
            sim[i, j] = sim[j, i] = s
    return sim


def jaccard_similarity_binary(data: np.ndarray) -> np.ndarray:
    """
    Jaccard similarity for binary data (0/1).

    s_ij = a / (a + b + c), where:
      a = number of positions where both rows are 1
      b = 1/0 mismatches
      c = 0/1 mismatches
    """
    n = data.shape[0]
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            r1 = data[i]
            r2 = data[j]
            a = np.sum((r1 == 1) & (r2 == 1))
            b = np.sum((r1 == 1) & (r2 == 0))
            c = np.sum((r1 == 0) & (r2 == 1))
            denom = a + b + c
            s = 0.0 if denom == 0 else a / denom
            sim[i, j] = sim[j, i] = s
    return sim


def dice_similarity_binary(data: np.ndarray) -> np.ndarray:
    """
    Dice (Sørensen) similarity for binary data.

    s_ij = 2a / (2a + b + c)
    """
    n = data.shape[0]
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            r1 = data[i]
            r2 = data[j]
            a = np.sum((r1 == 1) & (r2 == 1))
            b = np.sum((r1 == 1) & (r2 == 0))
            c = np.sum((r1 == 0) & (r2 == 1))
            denom = 2 * a + b + c
            s = 0.0 if denom == 0 else (2 * a) / denom
            sim[i, j] = sim[j, i] = s
    return sim


def msd_distance(data: np.ndarray) -> np.ndarray:
    """
    Mean squared difference (MSD) distance between rows.
    """
    n = data.shape[0]
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            diff = data[i] - data[j]
            d = np.mean(diff ** 2)
            dist[i, j] = dist[j, i] = d
    return dist


def rmsd_distance(data: np.ndarray) -> np.ndarray:
    """
    Root mean squared difference (RMSD) distance between rows.
    """
    msd = msd_distance(data)
    return np.sqrt(msd)


def euclidean_distance(data: np.ndarray) -> np.ndarray:
    """
    Euclidean distance between rows.
    """
    n = data.shape[0]
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(data[i] - data[j]))
            dist[i, j] = dist[j, i] = d
    return dist


def manhattan_distance(data: np.ndarray) -> np.ndarray:
    """
    Manhattan (city-block) distance between rows.
    """
    n = data.shape[0]
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sum(np.abs(data[i] - data[j])))
            dist[i, j] = dist[j, i] = d
    return dist


# ------------------------------------------------------------
# 4. UPGMA / WPGMA CLUSTERING
# ------------------------------------------------------------

def upgma_or_wpgma(dist_matrix: np.ndarray,
                   labels: List[str],
                   linkage: str = 'upgma') -> Tuple[str, np.ndarray]:
    """
    Perform UPGMA or WPGMA clustering and return:
      - Newick tree string
      - cophenetic distance matrix

    Parameters
    ----------
    dist_matrix : np.ndarray
        Symmetric distance matrix (n x n).
    labels : list of str
        Sample labels.
    linkage : str
        'upgma' (default) or 'wpgma'.

    Returns
    -------
    (tree_newick, cophenetic_matrix)
    """
    n = len(labels)
    clusters = [{'name': labels[i], 'size': 1, 'height': 0.0} for i in range(n)]
    members: Dict[int, List[int]] = {i: [i] for i in range(n)}
    D = dist_matrix.copy().astype(float)
    active = list(range(n))

    coph = np.zeros_like(D)

    while len(active) > 1:
        # Find closest pair of active clusters
        i, j = min(
            ((a, b) for a in active for b in active if a < b),
            key=lambda x: D[x]
        )
        ci, cj = clusters[i], clusters[j]
        h = D[i, j] / 2.0

        # Update cophenetic distances
        for a in members[i]:
            for b in members[j]:
                coph[a, b] = coph[b, a] = D[i, j]

        newick = f"({ci['name']}:{h - ci['height']:.4f},{cj['name']}:{h - cj['height']:.4f})"
        clusters[i] = {'name': newick,
                       'size': ci['size'] + cj['size'],
                       'height': h}
        members[i] = members[i] + members[j]
        del members[j]
        active.remove(j)

        # Update distances
        for k in active:
            if k == i:
                continue
            if linkage == 'upgma':
                d_new = (D[i, k] * ci['size'] + D[j, k] * cj['size']) / (ci['size'] + cj['size'])
            else:
                d_new = (D[i, k] + D[j, k]) / 2.0
            D[i, k] = D[k, i] = d_new

    root = clusters[active[0]]['name'] + ';'
    return root, coph


# ------------------------------------------------------------
# 5. COPHENETIC CORRELATION
# ------------------------------------------------------------

def cophenetic_correlation(orig_dist: np.ndarray, coph_dist: np.ndarray) -> float:
    """
    Compute the Cophenetic Correlation Coefficient (CCC) between
    the original distance matrix and the cophenetic distance matrix.

    Returns
    -------
    float
        Pearson correlation between upper-triangular (i<j) values.
    """
    if orig_dist.shape != coph_dist.shape:
        raise ValueError("orig_dist and coph_dist must have the same shape.")

    n = orig_dist.shape[0]
    if n < 2:
        return 0.0

    # Extract upper triangular (excluding diagonal)
    i_upper, j_upper = np.triu_indices(n, k=1)
    x = orig_dist[i_upper, j_upper]
    y = coph_dist[i_upper, j_upper]

    xm = x.mean()
    ym = y.mean()
    num = np.sum((x - xm) * (y - ym))
    den = np.sqrt(np.sum((x - xm) ** 2) * np.sum((y - ym) ** 2))
    if den == 0:
        return 0.0
    return num / den


# ------------------------------------------------------------
# 5b. BOOTSTRAP TREE GENERATION AND SUPPORT
# ------------------------------------------------------------

def generate_bootstrap_trees(row_labels: List[str],
                             col_labels: List[str],
                             data: np.ndarray,
                             input_type: str,
                             method: int,
                             linkage: str,
                             remove_zero_cols: int,
                             remove_constant_cols: int,
                             omit_identical: int,
                             normalize_flag: int,
                             scale_d: float,
                             boot_reps: int,
                             messages: List[str]) -> List[str]:
    """Generate bootstrap trees by resampling columns (variables) with replacement.

    This function only works when the original input_type is "data". For each
    bootstrap replicate, it resamples columns from the original data matrix with
    replacement, builds a new tree using `run_upgmaboot_core`, and collects the
    resulting Newick strings.
    """
    if boot_reps <= 1:
        return []

    if input_type != "data":
        messages.append("[WARN] Bootstrap requested but input type is not 'data'. Bootstrap will be skipped.")
        return []

    n_rows, n_cols = data.shape
    if n_cols == 0:
        messages.append("[WARN] Bootstrap skipped: no columns available in the input data.")
        return []

    trees: List[str] = []
    for rep in range(boot_reps):
        # Resample columns with replacement
        idx = np.random.randint(0, n_cols, size=n_cols)
        boot_data = data[:, idx]
        boot_cols = [col_labels[i] for i in idx]

        boot_res = run_upgmaboot_core(
            row_labels=row_labels.copy(),
            col_labels=boot_cols,
            data=boot_data,
            input_type="data",
            method=method,
            linkage=linkage,
            remove_zero_cols=remove_zero_cols,
            remove_constant_cols=remove_constant_cols,
            omit_identical=omit_identical,
            normalize_flag=normalize_flag,
            scale_d=scale_d,
        )
        trees.append(boot_res["tree_newick"].strip())

    messages.append(f"[INFO] Generated {boot_reps} bootstrap trees by resampling columns with replacement.")
    return trees


def compute_bootstrap_support(main_newick: str,
                              bootstrap_newicks: List[str],
                              messages: List[str]) -> Tuple[Optional[str], Optional[Any]]:
    """Compute bootstrap support values for internal nodes of the main tree.

    The function uses Toytree (if installed) to parse the reference tree and
    all bootstrap trees. Each internal node is identified by the set of leaf
    names in its clade, and the support is the percentage of bootstrap trees
    in which the same clade appears.
    """
    if not bootstrap_newicks:
        messages.append("[WARN] No bootstrap trees provided. Support values not computed.")
        return None, None

    if not _HAS_TOYTREE:
        messages.append("[WARN] Toytree is not installed. Bootstrap support values cannot be computed.")
        return None, None

    # Parse reference tree
    ref_tree = toytree.tree(main_newick)

    # Map each internal clade (as a frozenset of leaf names) to a count
    clade_counts: Dict[frozenset, int] = {}
    for node in ref_tree:
        if node.is_leaf():
            continue
        clade = frozenset(sorted(node.get_leaf_names()))
        clade_counts[clade] = 0

    if not clade_counts:
        messages.append("[WARN] Reference tree has no internal nodes. No support values assigned.")
        return None, None

    # Count how many times each clade appears across bootstrap trees
    for b_newick in bootstrap_newicks:
        b_tree = toytree.tree(b_newick)
        seen_clades = set()
        for node in b_tree:
            if node.is_leaf():
                continue
            clade = frozenset(sorted(node.get_leaf_names()))
            if clade in clade_counts:
                seen_clades.add(clade)
        for c in seen_clades:
            clade_counts[c] += 1

    # Assign support values (percentage) to internal nodes of the reference tree
    n_boot = float(len(bootstrap_newicks))
    for node in ref_tree:
        if node.is_leaf():
            continue
        clade = frozenset(sorted(node.get_leaf_names()))
        count = clade_counts.get(clade, 0)
        node.support = 100.0 * count / n_boot

    newick_with_support = ref_tree.write(features=["support"])
    messages.append("[INFO] Bootstrap support values assigned to internal nodes (in percent).")
    return newick_with_support, ref_tree


# ============================================================
# Rendering via Toytree (optional)
# ============================================================

def draw_tree_white(newick_str: str,
                    output_file: str = "dendrogram.png",
                    layout: str = "r",
                    node_labels: Optional[str] = None,
                    n_leaves: int = None):
    """Draw a Toytree dendrogram with white background and black edges.

    Parameters
    ----------
    newick_str : str
        Newick string representing the tree to plot.
    output_file : str, optional
        Path to the image file to create (PNG or SVG). If None, the
        function just creates the canvas and does not save it.
    layout : str, optional
        Toytree layout: right', 'r'; 'left', 'l'; 'down', 'd'; 'up', 'unr'; 'unrooted', 'unr'; 'circular', 'c'.
    node_labels : str, optional
        If set (e.g. "support"), Toytree will display that node feature
        (e.g. bootstrap support) on internal nodes.
    n_leaves : int, optional
        Number of leaves in the tree. Must be an integer > 2.
    """
    
    if not _HAS_TOYTREE:
        print("[WARN] Toytree is not installed. No graphical output will be generated.")
        return None
        
    # Layout adjustments
    if layout not in ["c","r","unr","d"]:
       raise ValueError(f"Unsupported tree layout: '{layout}'. Allowed: 'r' (right), 'c' (circular), 'unr' (unrooted), 'd' (down).")
       
    # Validate number of leaves
    if n_leaves is not None:
        if not isinstance(n_leaves, int):
            raise ValueError("Parameter 'n_leaves' must be an integer.")
        if n_leaves <= 2:
            raise ValueError("A dendrogram must contain three leaves (n_leaves > 2).")
    
    # --- Dynamic sizing of labels and canvas ---
    if n_leaves is not None:
        # Vertical size: increases with number of leaves
        height = max(300, n_leaves * 8)

        # Horizontal size: moderate increase if many labels
        width = 600 + min(n_leaves * 2, 400)

        # Font size decreases with number of leaves
        if n_leaves <= 30:
            label_font_size = "12px"
        elif n_leaves <= 80:
            label_font_size = "10px"
        elif n_leaves <= 120:
            label_font_size = "8px"
        else:
            label_font_size = "6px"
    else:
        # Default sizes when number of leaves not known
        height = 500
        width = 700
        label_font_size = "10px"
    
    if layout in ("d", "u"): # down or up
       width, height = height, width
    elif layout in ("c", "unr"): # circular or unrooted
       size = max(width, height)
       height = width = size

    if node_labels == "support":
       style_nodes ={ "fill": "red", "font-size": "9px",  "anchor_shift":-10, "baseline_shift":-10, "text_anchor":"middle", "font_family": "Helvetica"}
    else:
       style_nodes = None
    
    tree = toytree.tree(newick_str)
    
    canvas, axes, mark = tree.draw(
        layout=layout,
        edge_type="c" if layout == "c" else "p",
        node_hover=True,
        node_sizes=0,
        tip_labels_align=True,
        node_labels=node_labels,
        node_labels_style=style_nodes,
        node_colors="black",
        edge_colors="black",
        tip_labels_style={"fill": "black", "font-size":label_font_size},
        width=width,
        height=height,
    )
    canvas.style["background-color"] = "white"

    if output_file:
        # Ensure directory exists
        outdir = os.path.dirname(output_file)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        if output_file.lower().endswith(".png"):
            toyplot.png.render(canvas, output_file, scale=2)
        elif output_file.lower().endswith(".svg"):
            toyplot.svg.render(canvas, output_file, scale=2)
        print(f"[INFO] Toytree dendrogram saved as {output_file}")
    return canvas


# ------------------------------------------------------------
# 6. LOG WRITER
# ------------------------------------------------------------

def write_run_log(outdir: str,
                  args: argparse.Namespace,
                  row_labels: List[str],
                  col_labels: List[str],
                  data: np.ndarray,
                  messages: List[str],
                  error_msg: str = "") -> None:
    """
    Write a detailed run log with parameters, input summary,
    and processing messages (including errors).
    """
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "run_log.txt")

    with open(log_path, "w") as log:
        log.write(f"UPGMAboot run log — {datetime.datetime.now()}\n\n")
        log.write("Command-line arguments:\n")
        for k, v in vars(args).items():
            log.write(f"  {k}: {v}\n")
        log.write("\n")
        log.write(f"\nColumn delimiter used for parsing: {args.delimiter}\n\n")
        log.write(f"Number of rows (samples): {data.shape[0] if data.size else 0}\n")
        log.write(f"Number of columns (variables): {data.shape[1] if data.size else 0}\n\n")

        if messages:
            log.write("Messages:\n")
            for msg in messages:
                log.write(msg + "\n")
            log.write("\n")

        if error_msg:
            log.write("ERROR:\n")
            log.write(error_msg + "\n")


# ------------------------------------------------------------
# 7. CORE PIPELINE (USED ALSO BY test.py)
# ------------------------------------------------------------

def run_upgmaboot_core(row_labels: List[str], col_labels: List[str], data: np.ndarray,
                       input_type: str, method: int, linkage: str,
                       remove_zero_cols: int, remove_constant_cols: int,
                       omit_identical: int, normalize_flag: int, scale_d: float) -> Dict[str, Any]:
    """
    Core function implementing the full UPGMAboot pipeline.

    This is the function that `test.py` uses directly.

    Parameters
    ----------
    row_labels, col_labels : list of str
        Labels for rows (samples) and columns (variables).
    data : np.ndarray
        Input matrix (data, similarity, or distance).
    input_type : {"data", "similarity", "distance"}
        Type of input matrix.
    method : int
        1 Pearson; 2 Jaccard; 3 Dice; 4 MSD; 5 RMSD; 6 Euclidean; 7 Manhattan.
    linkage : {"upgma", "wpgma"}
        Clustering method.
    remove_zero_cols, remove_constant_cols, omit_identical, normalize_flag : int
        Preprocessing flags (0 or 1).
    scale_d : float
        Scaling factor for final distance matrix.

    Returns
    -------
    dict with keys:
        - "labels", "col_labels"
        - "similarity_matrix", "distance_matrix"
        - "tree_newick", "cophenetic_matrix", "cophenetic_ccc"
        - "messages", "omitted_rows", "filtered_data", "filtered_labels", "filtered_cols"
    """
    messages: List[str] = []
    CCC_LINE_THRESHOLD = 252  # If number of leaves > 252, CCC is not computed

    # --- Check and fix duplicated row labels ---
    unique_labels: List[str] = []
    dupes_fixed: List[Tuple[str, str]] = []
    for i, lbl in enumerate(row_labels):
        if lbl in unique_labels:
            new_lbl = f"{lbl}_{i+1}"
            dupes_fixed.append((lbl, new_lbl))
            unique_labels.append(new_lbl)
        else:
            unique_labels.append(lbl)
    if dupes_fixed:
        messages.append("[INFO] Duplicate row labels detected and fixed:")
        row_labels = unique_labels
        for old, new in dupes_fixed:
            messages.append(f"        {old} → {new}")
    else:
        messages.append("[INFO] No duplicate row labels detected.")

    # --- Preprocessing ---
    if remove_zero_cols:
        data, col_labels, removed = remove_zero_columns(data, col_labels)
        if removed:
            messages.append(f"[INFO] Removed {len(removed)} all-zero columns: {', '.join(removed)}")
        else:
            messages.append("[INFO] No all-zero columns found.")

    if remove_constant_cols:
        data, col_labels, removed = remove_constant_columns(data, col_labels)
        if removed:
            messages.append(f"[INFO] Removed {len(removed)} constant columns: {', '.join(removed)}")
        else:
            messages.append("[INFO] No constant columns found.")

    omitted_rows: List[str] = []
    if omit_identical:
        data, row_labels, omitted_rows = omit_identical_rows(data, row_labels, input_type)
        if omitted_rows:
            messages.append(f"[INFO] Removed {len(omitted_rows)} identical rows: {', '.join(omitted_rows)}")
        else:
            messages.append("[INFO] No identical rows found.")

    # --- Binary detection ---
    binary_detected = is_binary_matrix(data)
    if binary_detected:
        messages.append("[INFO] Binary data detected (all values 0 or 1).")
    else:
        messages.append("[INFO] Non-binary (continuous) data detected.")

    similarity_matrix: Optional[np.ndarray] = None
    distance_matrix: Optional[np.ndarray] = None

    # --- Interpret input type and compute matrices ---
    if input_type == 'data':
        if method == 1:
            messages.append("[INFO] Using method 1: Pearson similarity → distance")
            similarity_matrix = pearson_similarity(data)
        elif method == 2:
            messages.append("[INFO] Using method 2: Jaccard (binary only)")
            similarity_matrix = jaccard_similarity_binary(data)
        elif method == 3:
            messages.append("[INFO] Using method 3: Dice (binary only)")
            similarity_matrix = dice_similarity_binary(data)
        elif method in (4, 5, 6, 7):
            if normalize_flag and not binary_detected:
                data = normalize_data(data)
                messages.append("[INFO] Normalizing data using z-score per column (mean=0, std=1).")
                messages.append("[INFO] Data after normalization (z-score per variable).")
                header = "\t".join(col_labels)
                messages.append(f"\t{header}")
                for lbl, row in zip(row_labels, data):
                    row_str = "\t".join(f"{x:.4f}" for x in row)
                    messages.append(f"    {lbl}\t{row_str}")
            elif normalize_flag and binary_detected:
                messages.append("[INFO] Normalization requested but skipped (binary data).")

            if method == 4:
                messages.append("[INFO] Using method 4: MSD distance")
                distance_matrix = msd_distance(data)
            elif method == 5:
                messages.append("[INFO] Using method 5: RMSD distance")
                distance_matrix = rmsd_distance(data)
            elif method == 6:
                messages.append("[INFO] Using method 6: Euclidean distance")
                distance_matrix = euclidean_distance(data)
            elif method == 7:
                messages.append("[INFO] Using method 7: Manhattan distance")
                distance_matrix = manhattan_distance(data)
        else:
            raise ValueError(f"Unknown method code: {method}")

    elif input_type in ('similarity', 'distance'):
        # Copy matrix directly
        if input_type == 'similarity':
            similarity_matrix = data.copy()
            messages.append("[INFO] Input type: similarity matrix (will be converted to distance).")
        else:
            distance_matrix = data.copy()
            messages.append("[INFO] Input type: distance matrix (used as is).")

        # --- Validate square matrix ---
        n, m = data.shape
        if n != m:
            raise ValueError(f"Input declared as '{input_type}' but matrix is not square ({n}x{m}).")

        # --- Validate symmetry ---
        if not np.allclose(data, data.T, atol=1e-8):
            raise ValueError(f"Input declared as '{input_type}' but matrix is not symmetric.")

        # --- Validate diagonal ---
        diag = np.diag(data)
        if input_type == 'similarity':
            if not np.allclose(diag, 1.0, atol=1e-8):
                raise ValueError("Similarity matrix must have diagonal = 1.0")
        else:  # distance
            if not np.allclose(diag, 0.0, atol=1e-8):
                raise ValueError("Distance matrix must have diagonal = 0.0")

    else:
        raise ValueError(f"Unknown input type: {input_type}")

    # --- Convert similarity to distance if needed ---
    if similarity_matrix is not None:
        distance_matrix = 1.0 - similarity_matrix
        messages.append("[INFO] Converted similarity to distance using d = 1 - s.")

    if distance_matrix is None:
        raise ValueError("Distance matrix could not be computed.")

    if scale_d != 1.0:
        distance_matrix = distance_matrix * scale_d
        messages.append(f"[INFO] Distance matrix scaled by factor {scale_d}.")

    # --- Clustering ---
    n_leaves=len(row_labels)
    tree_newick, coph = upgma_or_wpgma(distance_matrix, row_labels, linkage=linkage)
    messages.append(f"[INFO] Tree successfully built using {linkage.upper()}.")
    messages.append(f"[INFO] Number of leaves (samples in final matrix): {n_leaves}")
    messages.append(f"Tree in Newick format:\n{tree_newick}")

    # --- CCC ---
    if len(row_labels) <= CCC_LINE_THRESHOLD:
        ccc = cophenetic_correlation(distance_matrix, coph)
        messages.append(f"[INFO] Cophenetic correlation coefficient (CCC): {ccc:.4f}")
    else:
        ccc = None
        messages.append(f"[INFO] CCC not calculated: number of elements ({len(row_labels)}) "
                        f"exceeds threshold ({CCC_LINE_THRESHOLD}).")

    # Render with Toytree if available (message only; actual drawing is done in main)
    if _HAS_TOYTREE:
        messages.append("[INFO] Toytree is available. Dendrogram PNG files can be generated.")
    else:
        messages.append("[WARN] Toytree is not installed. No dendrogram PNG files will be generated.")

    return {
        "labels": row_labels,
        "col_labels": col_labels,
        "similarity_matrix": similarity_matrix,
        "distance_matrix": distance_matrix,
        "tree_newick": tree_newick,
        "cophenetic_matrix": coph,
        "cophenetic_ccc": ccc,
        "messages": messages,
        "omitted_rows": omitted_rows,
        "filtered_data": data,
        "filtered_labels": row_labels,
        "filtered_cols": col_labels,
        "n_leaves": n_leaves
    }


# ------------------------------------------------------------
# 8. MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='UPGMAboot.py - Python version')
    parser.add_argument('-in', '--input', required=True, help='Input file')
    parser.add_argument('-out', '--output', default='UPGMA_output', help='Output directory')
    parser.add_argument('-type', choices=['auto', 'data', 'similarity', 'distance'], default='auto',
                        help='Input type (auto by default)')
    parser.add_argument('-delim', '--delimiter', default='[\\s,]+',
                    help='Regex or character(s) used to split columns (default: "[\\s,]+")')

    parser.add_argument('-method', type=int, default=1,
                        help='1 Pearson; 2 Jaccard; 3 Dice; 4 MSD; 5 RMSD; 6 Euclidean; 7 Manhattan')
    parser.add_argument('-linkage', choices=['upgma', 'wpgma'], default='upgma', help='Clustering method')
    parser.add_argument('-remove_zero_columns', type=int, default=0, help='1: remove all-zero columns')
    parser.add_argument('-remove_constant_columns', type=int, default=0, help='1: remove constant columns')
    parser.add_argument('-omit_identical_rows', type=int, default=0, help='1: remove identical rows')
    parser.add_argument('-normalize_data', type=int, default=0, help='1: normalize data (distance methods only)')
    parser.add_argument('-scale_d', type=float, default=1.0, help='Scaling factor for final distance matrix')
    parser.add_argument('-tree_layout', '--tree_layout', default='r', choices=['r', 'c','unr','d'], help="Tree layout for drawing: 'r' = right-oriented (default), 'c' = circular.")
    parser.add_argument('-boot_reps', type=int, default=1,
                        help='Number of bootstrap replicates (1–100). 1 = no bootstrap support.')
    args = parser.parse_args()

    if args.boot_reps < 1 or args.boot_reps > 100:
        print('[ERROR] -boot_reps must be between 1 and 100.')
        sys.exit(1)

    # --- Read input ---
    try:
        with open(args.input, 'r') as f:
            first_line = f.readline()
    except Exception as e:
        print(f'[ERROR] Cannot read input file: {e}')
        sys.exit(1)

    try:
        # Detect whether the input is FASTA-like or tabular
        if first_line.lstrip().startswith('>'):
            row_labels, data, col_labels = read_fasta_numeric(args.input, args.delimiter)
            detected_type = 'data'
            messages_input = ['[INFO] Detected FASTA-like numeric input.']
        else:
            row_labels, data, col_labels = read_tabular(args.input, args.delimiter)
            detected_type = detect_input_type(row_labels, data)
            messages_input = [f'[INFO] Detected input type: {detected_type}']
    except Exception as e:
        # If anything fails during input processing, write the log and exit
        write_run_log(args.output, args, [], [], np.zeros((0, 0)), [], error_msg=str(e))
        print(f'[ERROR] {e}')
        sys.exit(1)

    # If user explicitly selected an input type, validate consistency
    if args.type != "auto":
        if args.type != detected_type:
            raise ValueError(
                f"Input type mismatch: you selected '{args.type}', "
                f"but data appears to be '{detected_type}'."
            )

    input_type = args.type if args.type != 'auto' else detected_type

    # --- Core computation ---
    try:
        result = run_upgmaboot_core(
            row_labels=row_labels,
            col_labels=col_labels,
            data=data,
            input_type=input_type,
            method=args.method,
            linkage=args.linkage,
            remove_zero_cols=args.remove_zero_columns,
            remove_constant_cols=args.remove_constant_columns,
            omit_identical=args.omit_identical_rows,
            normalize_flag=args.normalize_data,
            scale_d=args.scale_d,
        )
        result['messages'] = messages_input + result['messages']
        error_msg = ''
    except Exception as e:
        write_run_log(args.output, args, row_labels, col_labels, data, messages_input, error_msg=str(e))
        print(f'[ERROR] {e}')
        sys.exit(1)

    # --- Write outputs ---
    os.makedirs(args.output, exist_ok=True)

    if result['similarity_matrix'] is not None:
        with open(os.path.join(args.output, 'similarity_matrix.tsv'), 'w') as f:
            f.write('\t' + '\t'.join(result['labels']) + '\n')
            for lab, row in zip(result['labels'], result['similarity_matrix']):
                f.write(lab + '\t' + '\t'.join(f'{v:.6f}' for v in row) + '\n')

    with open(os.path.join(args.output, 'distance_matrix.tsv'), 'w') as f:
        f.write('\t' + '\t'.join(result['labels']) + '\n')
        for lab, row in zip(result['labels'], result['distance_matrix']):
            f.write(lab + '\t' + '\t'.join(f'{v:.6f}' for v in row) + '\n')

    with open(os.path.join(args.output, 'tree.newick'), 'w') as f:
        f.write(result['tree_newick'] + '\n')

    # --- Write Cophenetic Correlation Coefficient (CCC) ---
    if "cophenetic_ccc" in result and result["cophenetic_ccc"] is not None:
        ccc = result["cophenetic_ccc"]
        with open(os.path.join(args.output, "cophenetic.txt"), "w") as f:
            f.write(f"Cophenetic Correlation Coefficient (CCC): {ccc:.6f}\n")

    # --- Toytree rendering for main tree ---
    if _HAS_TOYTREE:
        png_filename = f"dendrogram_{args.linkage}.png"
        png_path = os.path.join(args.output, png_filename)
        draw_tree_white(result['tree_newick'],
                        output_file=png_path,
                        layout=args.tree_layout,
                        node_labels=None,
                        n_leaves=result['n_leaves'])
    else:
        print("[WARN] Toytree is not installed. Main dendrogram PNG was not created.")

    # --- Bootstrap analysis (optional) ---
    if args.boot_reps > 1:
        bootstrap_trees = generate_bootstrap_trees(
            row_labels=row_labels,
            col_labels=col_labels,
            data=data,
            input_type=input_type,
            method=args.method,
            linkage=args.linkage,
            remove_zero_cols=args.remove_zero_columns,
            remove_constant_cols=args.remove_constant_columns,
            omit_identical=args.omit_identical_rows,
            normalize_flag=args.normalize_data,
            scale_d=args.scale_d,
            boot_reps=args.boot_reps,
            messages=result['messages'],
        )
        if bootstrap_trees:
            boot_file = os.path.join(args.output, "bootstrap_trees.txt")
            with open(boot_file, "w") as f:
                for t in bootstrap_trees:
                    f.write(t + "\n")
            result['messages'].append(f"[INFO] All {len(bootstrap_trees)} bootstrap trees written to {boot_file}.")

            newick_with_support, boot_tree_obj = compute_bootstrap_support(
                result['tree_newick'],
                bootstrap_trees,
                result['messages'],
            )
            if newick_with_support is not None:
                boot_newick_path = os.path.join(args.output, "tree_bootstrap.newick")
                with open(boot_newick_path, "w") as f:
                    f.write(newick_with_support + "\n")
                result['messages'].append(f"[INFO] Bootstrap tree with support values written to {boot_newick_path}.")

                if _HAS_TOYTREE:
                    boot_png = os.path.join(args.output, f"dendrogram_bootstrap_{args.linkage}.png")
                    draw_tree_white(newick_with_support,
                                    output_file=boot_png,
                                    layout=args.tree_layout,
                                    node_labels="support",
                                    n_leaves=result['n_leaves'])
                    result['messages'].append(f"[INFO] Bootstrap dendrogram PNG created: {boot_png}")
        else:
            result['messages'].append("[WARN] No bootstrap trees were generated.")

    # --- Write final log  ---
    write_run_log(args.output, args, row_labels, col_labels, data, result['messages'], error_msg='')

    print(f'[INFO] All files written in: {args.output}')


if __name__ == '__main__':
    main()

