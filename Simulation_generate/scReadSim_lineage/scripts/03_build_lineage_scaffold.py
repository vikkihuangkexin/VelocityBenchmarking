#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from Bio import Phylo

BARCODE_10X_RE = re.compile(r"([ACGTN]{16}-\d+)")
BARCODE_16_RE = re.compile(r"([ACGTN]{16})")


def parse_args():
    p = argparse.ArgumentParser(description="Build a donor lineage scaffold from a Newick tree.")
    p.add_argument("--tree", required=True)
    p.add_argument("--subset-barcodes", required=True)
    p.add_argument("--subset-labels", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--sample", default="sample")
    p.add_argument("--target-major-clades", type=int, default=8)
    p.add_argument("--min-donors-per-label", type=int, default=5)
    return p.parse_args()


def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def read_lines(path: str):
    with open_text(path) as f:
        return [x.rstrip("\n") for x in f if x.rstrip("\n")]


def norm_bc(x: str) -> str:
    s = str(x).strip()
    m = BARCODE_10X_RE.search(s)
    if m:
        return m.group(1)
    m = BARCODE_16_RE.search(s)
    if m:
        return m.group(1)
    return s


def _load_tree(path: str):
    with open_text(path) as f:
        return Phylo.read(f, "newick")


def _clade_leaf_names(clade):
    return [t.name for t in clade.get_terminals() if t.name is not None]


def _split_to_k_groups(tree, k: int):
    groups = [tree.root]
    while len(groups) < k:
        splittable = [g for g in groups if len(g.clades) > 0]
        if not splittable:
            break
        g = max(splittable, key=lambda c: len(c.get_terminals()))
        groups.remove(g)
        groups.extend(g.clades)
    return groups


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    subset_barcodes = read_lines(args.subset_barcodes)
    subset_labels = read_lines(args.subset_labels)
    if len(subset_barcodes) != len(subset_labels):
        raise SystemExit(f"subset-barcodes ({len(subset_barcodes)}) != subset-labels ({len(subset_labels)})")

    subset_df = pd.DataFrame({
        "barcode": subset_barcodes,
        "celllabel": subset_labels,
        "barcode_norm": [norm_bc(x) for x in subset_barcodes],
    })

    tree = _load_tree(args.tree)
    leaves = [t for t in tree.get_terminals() if t.name is not None]
    leaf_df = pd.DataFrame({
        "tree_leaf_raw": [t.name for t in leaves],
        "barcode_norm": [norm_bc(t.name) for t in leaves],
    }).drop_duplicates("barcode_norm", keep="first")

    donor_meta = subset_df.merge(leaf_df, on="barcode_norm", how="inner")
    if donor_meta.empty:
        raise SystemExit("No overlap between selected loom barcodes and tree leaves.")

    # assign major clades by splitting root greedily
    groups = _split_to_k_groups(tree, args.target_major_clades)
    leaf_to_clade = {}
    for i, cl in enumerate(groups):
        for nm in _clade_leaf_names(cl):
            leaf_to_clade[norm_bc(nm)] = f"major_clade_{i}"

    term_by_norm = {norm_bc(t.name): t for t in leaves}
    donor_meta["donor_major_clade"] = donor_meta["barcode_norm"].map(leaf_to_clade).fillna("major_clade_na")
    donor_meta["donor_leaf_id"] = donor_meta["barcode"]
    donor_meta["donor_tree_leaf_raw"] = donor_meta["tree_leaf_raw"]
    donor_meta["donor_tree_depth"] = donor_meta["barcode_norm"].map(lambda b: float(tree.distance(term_by_norm[b])) if b in term_by_norm else None)
    donor_meta["lineage_profile"] = donor_meta.apply(
        lambda r: f"{r['donor_major_clade']}|depth={r['donor_tree_depth']}", axis=1
    )
    donor_meta["donor_clone_id"] = donor_meta["donor_major_clade"].astype(str).map(lambda x: hashlib.md5(x.encode()).hexdigest()[:12])
    donor_meta["donor_n_intBC"] = 0
    donor_meta["donor_total_UMI"] = 0
    donor_meta["donor_total_readCount"] = 0

    # prune tree to donors actually kept
    keep_norm = set(donor_meta["barcode_norm"])
    pruned = _load_tree(args.tree)
    for term in list(pruned.get_terminals()):
        if norm_bc(term.name) not in keep_norm:
            try:
                pruned.prune(term)
            except Exception:
                pass

    donor_meta = donor_meta.sort_values(["celllabel", "barcode"]).reset_index(drop=True)
    donor_meta.to_csv(outdir / f"{args.sample}.donor_metadata.tsv", sep="\t", index=False)
    with open(outdir / f"{args.sample}.truth_tree.pruned.newick.txt", "wt") as f:
        Phylo.write(pruned, f, "newick")

    label_counts = donor_meta["celllabel"].value_counts().sort_index()
    bad_labels = label_counts[label_counts < args.min_donors_per_label]

    summary = {
        "sample": args.sample,
        "n_subset_cells": int(len(subset_df)),
        "n_subset_cells_with_tree": int(len(donor_meta)),
        "subset_tree_overlap_fraction": float(len(donor_meta) / max(1, len(subset_df))),
        "n_tree_leaves_total": int(len(leaves)),
        "n_unique_major_clades": int(donor_meta["donor_major_clade"].nunique()),
        "label_counts_total_subset": subset_df["celllabel"].value_counts().sort_index().to_dict(),
        "label_counts_tree_donors": label_counts.to_dict(),
        "labels_below_min_donors": bad_labels.to_dict(),
        "outputs": {
            "donor_metadata": str(outdir / f"{args.sample}.donor_metadata.tsv"),
            "truth_tree_pruned": str(outdir / f"{args.sample}.truth_tree.pruned.newick.txt"),
        },
    }
    (outdir / f"{args.sample}.tree_scaffold.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not bad_labels.empty:
        raise SystemExit(
            "[ERROR] Some labels have too few tree-resolved donors: " + ", ".join(f"{k}={v}" for k, v in bad_labels.items())
        )


if __name__ == "__main__":
    main()
