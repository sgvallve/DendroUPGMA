#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test.py — Validation suite for UPGMAboot.py
Version: 2025-11-02
Author: Santi Garcia-Vallvé (QiN-URV)

This script performs an extensive validation of UPGMAboot.py.
It checks:
  • Continuous data methods (Pearson, MSD, RMSD, Euclidean, Manhattan)
  • Binary data methods (Jaccard, Dice)
  • Consistency of clustering results from data, similarity, and distance inputs
  • Preprocessing options (zero/constant column removal, identical row removal, normalization, scaling)

All tests print detailed (“verbose”) comparisons, including expected vs calculated matrices,
Newick trees, and cophenetic correlation coefficients (CCC).
"""

import numpy as np
import os
import re
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cophenet, to_tree
from scipy.stats import pearsonr
import scipy.stats as stats
from UPGMAboot import run_upgmaboot_core, upgma_or_wpgma, cophenetic_correlation, normalize_data, generate_bootstrap_trees, compute_bootstrap_support, draw_tree_white
import datetime

# ============================================================
# GLOBAL TEST COUNTERS
# ============================================================
test_pass = 0
test_fail = 0


def test_passed(name: str):
    """Register a passed test and print confirmation."""
    global test_pass
    test_pass += 1
    print(f"✅ PASSED: {name}")


def test_failed(name: str):
    """Register a failed test and print warning."""
    global test_fail
    test_fail += 1
    print(f"⚠️ FAILED: {name}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def show_labeled_data(title, data, row_labels, col_labels):
    """Pretty-print a labeled data matrix with row and column headers."""
    print(f"\n{title}")
    header = "       " + " ".join(f"{v:>8}" for v in col_labels)
    print(header)
    for lbl, row in zip(row_labels, data):
        print(f"{lbl:>5} " + " ".join(f"{x:8.4f}" for x in row))


def show_matrix(title, matrix):
    """Print a formatted numerical matrix."""
    print(f"\n{title}")
    if matrix is None:
        print("(None)")
        return
    print(np.array2string(matrix, formatter={'float_kind': lambda x: f'{x:8.4f}'}))


def compare_matrices(name, calculated, expected, tol=0.05, source=None):
    """Compare two matrices element-wise and report absolute differences."""
    diff = np.abs(calculated - expected)
    max_diff = diff.max()
    src = f" ({source})" if source else ""
    print(f"\nExpected {name}{src}:\n", np.round(expected, 4))
    print(f"Calculated {name}:\n", np.round(calculated, 4))
    print(f"Difference (|calc - exp|):\n", np.round(diff, 4))

    if np.allclose(calculated, expected, atol=tol):
        test_passed(f"{name} within tolerance (max diff = {max_diff:.4f})")
    else:
        test_failed(f"{name} differences exceed tolerance (tol={tol})")
        for (i, j), d in np.ndenumerate(diff):
            if d > tol:
                print(f"  ({i+1},{j+1}): expected={expected[i,j]:.4f}, got={calculated[i,j]:.4f}, diff={d:.4f}")
        print(f"  → Maximum difference = {max_diff:.4f}")


def scipy_newick(Z, labels):
    """Convert a SciPy linkage matrix into a Newick tree string."""
    def build(node, parent_dist=0.0):
        if node.is_leaf():
            return labels[node.id] + f":{parent_dist - node.dist:.4f}"
        left = build(node.left, node.dist)
        right = build(node.right, node.dist)
        return f"({left},{right}):{parent_dist - node.dist:.4f}"

    tree = to_tree(Z)
    return build(tree) + ";"


# ============================================================
# TEST 1: CONTINUOUS DATA
# ============================================================

def test_continuous_data():
    print("=== TEST 1: CONTINUOUS DATA (A–D) ===")

    labels = ["A", "B", "C", "D"]
    cols = [f"V{i+1}" for i in range(5)]
    data = np.array([
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [5, 4, 3, 2, 1],
        [2, 2, 3, 4, 5],
    ], dtype=float)
    show_labeled_data("Input observations (rows = A–D, columns = V1–V5):", data, labels, cols)

    # Pearson similarity
    print("\n▶ Pearson similarity test")
    res = run_upgmaboot_core(labels, cols, data, "data", 1, "upgma", 0, 0, 0, 0, 1.0)
    n = data.shape[0]
    pearson_ref = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                pearson_ref[i, j] = 1.0
            else:
                r, _ = pearsonr(data[i], data[j])
                pearson_ref[i, j] = (r + 1) / 2
    compare_matrices("Pearson similarity", res["similarity_matrix"], pearson_ref, tol=0.05, source="SciPy (pearsonr)")

    # MSD and RMSD
    print("\n▶ MSD distance test")
    res = run_upgmaboot_core(labels, cols, data, "data", 4, "upgma", 0, 0, 0, 0, 1.0)
    expected_msd = np.array([
        [0.0, 1.0, 8.0, 0.2],
        [1.0, 0.0, 9.0, 0.8],
        [8.0, 9.0, 0.0, 6.6],
        [0.2, 0.8, 6.6, 0.0],
    ])
    compare_matrices("MSD distance", res["distance_matrix"], expected_msd, tol=0.2, source="manual reference")

    print("\n▶ RMSD distance test")
    res = run_upgmaboot_core(labels, cols, data, "data", 5, "upgma", 0, 0, 0, 0, 1.0)
    expected_rmsd = np.sqrt(expected_msd)
    compare_matrices("RMSD distance", res["distance_matrix"], expected_rmsd, tol=0.2, source="manual reference")

    # Euclidean and Manhattan
    print("\n▶ Euclidean distance test")
    res = run_upgmaboot_core(labels, cols, data, "data", 6, "upgma", 0, 0, 0, 0, 1.0)
    expected_euc = squareform(pdist(data, metric="euclidean"))
    compare_matrices("Euclidean distance", res["distance_matrix"], expected_euc, tol=0.05, source="SciPy (pdist)")

    print("\n▶ Manhattan distance test")
    res = run_upgmaboot_core(labels, cols, data, "data", 7, "upgma", 0, 0, 0, 0, 1.0)
    expected_man = squareform(pdist(data, metric="cityblock"))
    compare_matrices("Manhattan distance", res["distance_matrix"], expected_man, tol=0.05, source="SciPy (pdist)")

    # Euclidean + Newick + CCC
    print("\n▶ Euclidean distance test + Newick tree + Cophenetic correlation coefficient (SciPy reference)")
    res = run_upgmaboot_core(labels, cols, data, "data", 6, "upgma", 0, 0, 0, 0, 1.0)
    dist = res["distance_matrix"]
    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    newick_scipy = scipy_newick(Z, labels)
    ccc_scipy, _ = cophenet(Z, condensed)
    newick_ours, coph = upgma_or_wpgma(dist, labels, linkage="upgma")
    ccc_ours = cophenetic_correlation(dist, coph)
    print("\nUPGMA tree (ours):", newick_ours)
    print("UPGMA tree (SciPy):", newick_scipy)
    diff_ccc = abs(ccc_ours - ccc_scipy)
    print(f"Cophenetic correlation coefficient: ours={ccc_ours:.4f}, SciPy={ccc_scipy:.4f}, Difference={diff_ccc:.4f}")
    if diff_ccc < 0.1:
        test_passed("Cophenetic correlation coefficient")
    else:
        test_failed("Cophenetic correlation coefficient")

# ============================================================
# TEST 2: BINARY DATA
# ============================================================

def test_binary_data():
    print("\n=== TEST 2: BINARY DATA (E,F,G) ===")

    labels = ["E", "F", "G"]
    cols = [f"V{i+1}" for i in range(5)]
    data = np.array([
        [1, 0, 1, 0, 1],
        [1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ], dtype=int)
    show_labeled_data("Input observations (binary):", data, labels, cols)

    for method, code, metric in [("Jaccard", 2, "jaccard"), ("Dice", 3, "dice")]:
        print(f"\n▶ {method} similarity test")
        res = run_upgmaboot_core(labels, cols, data, "data", code, "upgma", 0, 0, 0, 0, 1.0)
        expected = 1 - squareform(pdist(data, metric=metric))
        compare_matrices(f"{method} (binary)", res["similarity_matrix"], expected, tol=0.05, source=f"SciPy ({metric})")


# ============================================================
# TEST 3: CONSISTENCY OF RESULTS (PEARSON)
# ============================================================

def test_consistency_inputs():
    print("\n=== TEST 3: CONSISTENCY OF RESULTS STARTING FROM DATA, SIMILARITY AND DISTANCE INPUTS (PEARSON) ===")

    labels = ["A", "B", "C", "D"]
    cols = [f"V{i+1}" for i in range(4)]
    data = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.5, 2.2, 3.1, 3.9],
        [4.5, 3.5, 2.5, 1.5],
        [1.0, 1.7, 2.3, 3.0],
    ])
    show_labeled_data("Base data (4 observations × 4 variables):", data, labels, cols)

    # Compute similarity and distance with SciPy
    n = len(labels)
    sim_scipy = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = stats.pearsonr(data[i], data[j])
            sim_scipy[i, j] = sim_scipy[j, i] = (r + 1) / 2
    dist_scipy = 1 - sim_scipy

    show_matrix("Pearson similarity matrix (computed with SciPy, scaled 0–1):", sim_scipy)
    show_matrix("Pearson distance matrix (computed with SciPy, 1 - similarity):", dist_scipy)

    # Run UPGMAboot
    res_data = run_upgmaboot_core(labels, cols, data, "data", 1, "upgma", 0, 0, 0, 0, 1.0)
    res_sim = run_upgmaboot_core(labels, labels, sim_scipy, "similarity", 1, "upgma", 0, 0, 0, 0, 1.0)
    res_dist = run_upgmaboot_core(labels, labels, dist_scipy, "distance", 1, "upgma", 0, 0, 0, 0, 1.0)

    show_matrix("Similarity matrix returned by UPGMAboot (from data input, method = Pearson):", res_data["similarity_matrix"])
    show_matrix("Distance matrix returned by UPGMAboot (from data input, method = Pearson):", res_data["distance_matrix"])

    # Compare trees and CCC
    sources = ["DATA", "SIMILARITY", "DISTANCE"]
    results = [res_data, res_sim, res_dist]
    trees, cophs, cccs = [], [], []
    for src, res in zip(sources, results):
        t, coph = upgma_or_wpgma(res["distance_matrix"], labels, linkage="upgma")
        trees.append(t)
        cophs.append(coph)
        ccc = cophenetic_correlation(res["distance_matrix"], coph)
        cccs.append(ccc)
        print(f"[{src}] Newick tree: {t}")
        print(f"[{src}] Cophenetic correlation coefficient: {ccc:.4f}")

    # Pairwise comparisons
    print("\n--- Pairwise correlations between cophenetic matrices ---")
    for i in range(3):
        for j in range(i + 1, 3):
            r = np.corrcoef(cophs[i].flatten(), cophs[j].flatten())[0, 1]
            if r >= 0.9999:
                #print(f"✅ {sources[i]} vs {sources[j]}: r={r:.4f} (identical within precision)")
                test_passed(f"Consistency test {sources[i]} vs {sources[j]}: r={r:.4f} (identical within precision) ")
            elif r >= 0.99:
                #print(f"ℹ️ {sources[i]} vs {sources[j]}: r={r:.4f} (minor differences only)")
                test_passed(f"Consistency test {sources[i]} vs {sources[j]}: r={r:.4f} (minor differences only) ")
            else:
                #print(f"⚠️ {sources[i]} vs {sources[j]}: r={r:.4f} (real differences)")
                test_failed(f"Consistency test {sources[i]} vs {sources[j]}: r={r:.4f} (real differences) ")



# ============================================================
# TEST 4: PREPROCESSING OPTIONS
# ============================================================

def test_preprocessing():
    print("\n=== TEST 4: PREPROCESSING OPTIONS ===")

    def show_before_after(title, data, res, labels, cols):
        filtered_data = res.get("filtered_data", data)
        filtered_labels = res.get("filtered_labels", labels)
        filtered_cols = res.get("filtered_cols", cols)
        show_labeled_data(f"\n{title} – ORIGINAL:", data, labels, cols)
        show_labeled_data(f"{title} – AFTER:", filtered_data, filtered_labels, filtered_cols)
        return filtered_data, filtered_cols, filtered_labels

    # Remove zero columns
    print("\n▶ remove_zero_columns")
    labels, cols = ["R1", "R2"], ["C1", "C2", "C3"]
    data = np.array([[1, 0, 2], [3, 0, 4]], float)
    res = run_upgmaboot_core(labels, cols, data, "data", 6, "upgma", 1, 0, 0, 0, 1.0)
    _, filtered_cols, _ = show_before_after("remove_zero_columns", data, res, labels, cols)
    #print("✅" if "C2" not in filtered_cols else "⚠️", "Zero-only column handling checked.")
    if "C2" not in filtered_cols:
         test_passed("Zero-only column handling checked.")
    else:
         test_failed("Zero-only column handling checked.")

    # Remove constant columns
    print("\n▶ remove_constant_columns")
    data = np.array([[1, 5, 2], [3, 5, 4]], float)
    cols = ["C1", "C2", "C3"]
    res = run_upgmaboot_core(labels, cols, data, "data", 6, "upgma", 0, 1, 0, 0, 1.0)
    _, filtered_cols, _ = show_before_after("remove_constant_columns", data, res, labels, cols)
    #print("✅" if "C2" not in filtered_cols else "⚠️", "Constant column handling checked.")
    if "C2" not in filtered_cols:
         test_passed("Constant column handling checked.")
    else:
         test_failed("Constant column handling checked.")

    # Omit identical rows
    print("\n▶ omit_identical_rows")
    labels = ["R1", "R2", "R3"]
    data = np.array([[1, 2, 3], [1, 2, 3], [2, 3, 4]], float)
    res = run_upgmaboot_core(labels, ["C1", "C2", "C3"], data, "data", 6, "upgma", 0, 0, 1, 0, 1.0)
    filtered_data, _, filtered_labels = show_before_after("omit_identical_rows", data, res, labels, ["C1", "C2", "C3"])
    if len(filtered_labels) < len(labels):
        test_passed("Identical rows removed.")
    else:
        test_failed("Identical rows not removed.")

    # Normalize data
    print("\n▶ normalize_data")
    labels = ["N1", "N2", "N3"]
    cols = ["A", "B", "C"]
    data = np.array([
        [1.0, 5.0, 10.0],
        [2.0, 8.0, 15.0],
        [3.0, 10.0, 20.0],
    ])
    res = run_upgmaboot_core(labels, cols, data, "data", 6, "upgma", 0, 0, 0, 1, 1.0)
    _, _, _ = show_before_after("normalize_data", data, res, labels, cols)
    normalized = res["filtered_data"]
    expected = normalize_data(data)
    if np.allclose(normalized, expected, atol=1e-6):
        test_passed("Normalization applied correctly (z-score per column).")
    else:
        test_failed(f"Normalization differs from expected z-score. Expected {expected}, Got {normalized}")

    # Scale distances
    print("\n▶ scale_d (×10)")
    labels = ["S1", "S2", "S3"]
    cols = ["X", "Y", "Z"]
    data = np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6]], float)
    res_normal = run_upgmaboot_core(labels, cols, data, "data", 6, "upgma", 0, 0, 0, 0, 1.0)
    res_scaled = run_upgmaboot_core(labels, cols, data, "data", 6, "upgma", 0, 0, 0, 0, 10.0)
    d1, d2 = res_normal["distance_matrix"], res_scaled["distance_matrix"]
    show_matrix("Distance matrix – ORIGINAL (scale_d = 1.0):", d1)
    show_matrix("Distance matrix – AFTER (scale_d = 10.0):", d2)
    mask = ~np.isclose(d1, 0.0)
    ratios = d2[mask] / d1[mask]
    if np.allclose(ratios, 10.0, atol=1e-6):
        test_passed("Distance scaling factor applied correctly (×10).")
    else:
        test_failed(f"Distance scaling failed.Ratios (excluding diagonal): {ratios}")

# ============================================================
# TEST 5 - Toytree drawing test
# ============================================================

def run_test_5():
    print("\n=== TEST 5: Toytree drawing functionality check ===")

    try:
        import toytree
        test_passed("Toytree library is installed")
        
        newick_str = "((A,B),(C,D));"
        draw_tree_white(newick_str, output_file='test_toytree.png', layout="r", node_labels=None)
        if os.path.exists('test_toytree.png'):
            test_passed("Toytree dendrogram successfully generated")
        else:
            test_failed("Toytree drawing did not produce the expected image file")
            
    except ImportError:
        test_failed("Toytree library is not installed, but it is optional")

# ============================================================
# TEST 6 - Bootstrap tree consistency and clade frequency validation
# ============================================================

def extract_bootstrap_from_tree(tree_str: str):
    """
    Extract clades (list of taxa) and their bootstrap values from a Newick tree string.
    Uses only Python stdlib (no external libraries).
    Returns a dict mapping each clade (tuple of taxa) to its bootstrap support value.
    """
    import re
    # Remove [&support=...] annotations and branch lengths
    clean = re.sub(r"\[&[^\]]*\]", "", tree_str)
    clean = re.sub(r":[0-9\.Ee\+\-]+", "", clean)

    def parse_subtree(s, idx):
        leaf_names = []
        results = []
        while idx < len(s):
            if s[idx] == '(':
                sub_result, sub_leaves, idx = parse_subtree(s, idx + 1)
                leaf_names.extend(sub_leaves)
                results.extend(sub_result)
            elif s[idx] == ')':
                idx += 1
                bootstrap = ""
                while idx < len(s) and s[idx].isdigit():
                    bootstrap += s[idx]
                    idx += 1
                if bootstrap:
                    results.append((tuple(sorted(leaf_names.copy())), int(bootstrap)))
                return results, leaf_names, idx
            elif s[idx] == ',':
                idx += 1
            elif s[idx] == ';':
                break
            else:
                start = idx
                while idx < len(s) and s[idx] not in ',);':
                    idx += 1
                leaf = s[start:idx]
                if leaf:
                    leaf_names.append(leaf)
        return results, leaf_names, idx

    results, _, _ = parse_subtree(clean, 0)
    return {clade: bs for clade, bs in results}


def extract_clades_from_newick_simple(newick_str):
    """
    Extract clades (sets of leaf labels) from a Newick string.
    - Removes branch lengths, bootstrap values, and [&support=...] tags.
    - Returns only real leaf-based clades (tuples of labels).
    """
    import re

    # 1️⃣ Remove non-informative parts
    s = re.sub(r"\[&[^\]]*\]", "", newick_str)      # remove [&support=...]
    s = re.sub(r":[0-9\.Ee\+\-]+", "", s)           # remove branch lengths
    s = re.sub(r"\)\d+", ")", s)                    # remove )90 bootstrap tags
    s = s.replace(";", "").strip()

    # 2️⃣ Hierarchical parser
    stack = []
    token = ""
    clades = []

    for ch in s:
        if ch.isalnum() or ch in ['_', '-']:
            token += ch
        elif ch == ',':
            if token:
                stack.append([token])
                token = ""
        elif ch == '(':
            stack.append('(')
        elif ch == ')':
            if token:
                stack.append([token])
                token = ""
            group = []
            while stack:
                item = stack.pop()
                if item == '(':
                    break
                if isinstance(item, list):
                    group.extend(item)
            # només clades amb almenys dues fulles
            if len(group) > 1:
                clades.append(tuple(sorted(set(group))))
            # reintroduïm la llista per poder fer nivells superiors
            stack.append(group)
        elif ch == ';':
            break
        else:
            token = ""

    # 3️⃣ Remove duplicates and trivial clades
    unique = []
    seen = set()
    for c in clades:
        c = tuple(sorted(set(c)))
        if len(c) > 1 and c not in seen:
            unique.append(c)
            seen.add(c)
    return unique

def run_test_6():
    print("\n=== TEST 6: Bootstrap tree consistency and clade frequency validation ===")
        
    # --- Example input dataset ---
    labels = [f"a{i}" for i in range(1, 7)]
    cols = [f"Var{i}" for i in range(1, 11)]
    data = np.array([
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
     [1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
     [1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    ])

    print("[INFO] Example input matrix (6 samples x 10 variables):")
    for i, row in enumerate(data):
        print(f"  {labels[i]}: {np.round(row, 2)}")

    # --- Run main clustering (no bootstrap, just main tree) ---
    res = run_upgmaboot_core(
        labels, cols, data,
        input_type="data",
        method=2,
        linkage="upgma",
        remove_zero_cols=0,
        remove_constant_cols=0,
        omit_identical=0,
        normalize_flag=1,
        scale_d=1.0,
    )
    main_tree = res["tree_newick"]
    print(f"\n[INFO] Main UPGMA tree:\n  {main_tree.strip()}")

    # --- Generate bootstrap replicates in-memory ---
    trees = generate_bootstrap_trees(
    row_labels=labels,
    col_labels=cols,
    data=data,
    input_type="data",
    method=2,
    linkage="upgma",
    remove_zero_cols=0,
    remove_constant_cols=0,
    omit_identical=0,
    normalize_flag=1,
    scale_d=1.0,
    boot_reps=10,
    messages=[]
)

    print(f"\n[INFO] Generated {len(trees)} bootstrap trees.")
    for i in trees:
       print(i)

    # --- Compute bootstrap supports using program's own function ---
    tree_with_support, boot_tree_obj = compute_bootstrap_support(main_tree, trees, messages=[])
    print(f"\n[INFO] Tree with bootstrap supports:\n  {tree_with_support.strip()}")

    # --- Count occurrences of each clade across bootstrap trees ---
    clade_counts = {}
    for t in trees:
        for clade in extract_clades_from_newick_simple(t):
            clade_counts[clade] = clade_counts.get(clade, 0) + 1

    print("\n[INFO] Clade frequency summary (from 10 bootstraps):")
    for clade, count in sorted(clade_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {','.join(clade)}: {count}/10")

    # --- Extract supports from the tree generated by compute_bootstrap_support ---
# --- Compare counted clade frequencies vs. bootstrap supports from the generated tree ---
    support_values = extract_bootstrap_from_tree(tree_with_support)

    print("\n[INFO] Comparison between counted frequencies and program bootstrap supports:")
    print("  Clade\t\tFreq(%)\tSupport(%)\tDifference")
    print("  ----------------------------------")

    differences = []
    for clade, count in sorted(clade_counts.items(), key=lambda x: (-x[1], x[0])):
        freq_percent = count * 10  # Ex: 10 replicats bootstrap
        supp = support_values.get(tuple(sorted(clade)), None)
        if supp is not None:
            dif = abs(freq_percent - supp)
            differences.append(dif)
            print(f"{','.join(clade)}\tFreq={freq_percent}\tSupp={supp}\tDifference={dif}")
        else:
            print(f"{','.join(clade)}\tFreq={freq_percent}\tSupp=-\tDifference=-")

    if differences:
        global_difference = sum(differences) / len(differences)
        print(f"\nGlobal mean difference:: {global_difference:.2f}")

        
    if global_difference <15:
        test_passed(f"Bootstrap support values match clade frequencies.")
    else:
        test_failed("Inconsistent bootstrap supports vs clade frequencies.")
# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print(f"\nUPGMAboot Test Suite — Version 2025-11-02")
    print(f"Started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    test_continuous_data()
    test_binary_data()
    test_consistency_inputs()
    test_preprocessing()
    run_test_5()
    try:
        import toytree
        run_test_6()

    except ImportError:
        print("\n=== TEST 6: Bootstrap tree consistency and clade frequency validation ===")
        print("Test Skipped. Toytree library is not installed, but it is optional. The bootstrap tree can not be calculated.")
    
    total = test_pass + test_fail
    print("\n=== TEST SUMMARY ===")
    print(f"Total tests evaluated: {total}")
    print(f"  ✅ Passed: {test_pass}")
    print(f"  ⚠️ Failed: {test_fail}")
    if total > 0:
        success_rate = (test_pass / total) * 100
        print(f"  → Success rate: {success_rate:.1f}%")
    print("\nCompleted at", datetime.datetime.now().strftime('%H:%M:%S'))

