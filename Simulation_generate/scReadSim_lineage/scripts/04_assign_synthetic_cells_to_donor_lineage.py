#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Assign synthetic cells to lineage-resolved donors within label.")
    p.add_argument("--donor-metadata", required=True)
    p.add_argument("--synthetic-barcodes", required=True)
    p.add_argument("--synthetic-labels", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--sample", default="sample")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def read_lines(path: str):
    with open(path, "rt") as f:
        return [x.rstrip("\n") for x in f if x.rstrip("\n")]


def norm_label(x: str) -> str:
    x = str(x).strip()
    if len(x) >= 2 and ((x[0] == '"' and x[-1] == '"') or (x[0] == "'" and x[-1] == "'")):
        x = x[1:-1]
    x = x.strip()
    return x


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    donor = pd.read_csv(args.donor_metadata, sep="\t")
    syn_barcodes = read_lines(args.synthetic_barcodes)
    syn_labels = read_lines(args.synthetic_labels)
    if len(syn_barcodes) != len(syn_labels):
        raise SystemExit(f"synthetic-barcodes ({len(syn_barcodes)}) != synthetic-labels ({len(syn_labels)})")

    donor["celllabel_raw"] = donor["celllabel"].astype(str)
    donor["celllabel_norm"] = donor["celllabel_raw"].map(norm_label)

    syn = pd.DataFrame({
        "synthetic_barcode": syn_barcodes,
        "synthetic_celllabel_raw": syn_labels,
    })
    syn["synthetic_celllabel"] = syn["synthetic_celllabel_raw"].map(norm_label)

    donor_pools = {k: v.reset_index(drop=True) for k, v in donor.groupby("celllabel_norm")}
    missing_labels = sorted(set(syn["synthetic_celllabel"]) - set(donor_pools))
    if missing_labels:
        debug = {
            "missing_labels": missing_labels,
            "donor_labels_raw_example": sorted(map(str, donor["celllabel_raw"].drop_duplicates().tolist()))[:30],
            "donor_labels_norm_example": sorted(map(str, donor["celllabel_norm"].drop_duplicates().tolist()))[:30],
            "synthetic_labels_raw_example": sorted(map(str, syn["synthetic_celllabel_raw"].drop_duplicates().tolist()))[:30],
            "synthetic_labels_norm_example": sorted(map(str, syn["synthetic_celllabel"].drop_duplicates().tolist()))[:30],
        }
        (outdir / f"{args.sample}.synthetic_lineage_label_debug.json").write_text(json.dumps(debug, indent=2))
        raise SystemExit(
            "No donor pool for normalized synthetic labels: "
            + ", ".join(missing_labels)
            + f"\nDebug written to {outdir / f'{args.sample}.synthetic_lineage_label_debug.json'}"
        )

    rng = np.random.default_rng(args.seed)
    out_parts = []
    assign_counts = {}
    for label, part in syn.groupby("synthetic_celllabel", sort=True):
        pool = donor_pools[label]
        idx = rng.integers(0, len(pool), size=len(part))
        chosen = pool.iloc[idx].reset_index(drop=True)
        merged = part.reset_index(drop=True).copy()
        merged["donor_original_barcode"] = chosen["barcode"].values
        merged["donor_clone_id"] = chosen["donor_clone_id"].values
        merged["donor_leaf_id"] = chosen["donor_leaf_id"].values
        merged["donor_n_intBC"] = chosen["donor_n_intBC"].values
        merged["donor_total_UMI"] = chosen["donor_total_UMI"].values
        merged["donor_total_readCount"] = chosen["donor_total_readCount"].values
        merged["donor_lineage_profile"] = chosen["lineage_profile"].values
        merged["donor_celllabel"] = chosen["celllabel_raw"].values
        out_parts.append(merged)
        assign_counts[label] = {
            "n_synthetic": int(len(part)),
            "n_donor_pool": int(len(pool)),
            "n_unique_donors_used": int(merged["donor_original_barcode"].nunique()),
        }

    out = pd.concat(out_parts, axis=0).sort_values(["synthetic_celllabel", "synthetic_barcode"]).reset_index(drop=True)
    out_path = outdir / f"{args.sample}.synthetic_cells_with_lineage.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    clone_counts = out["donor_clone_id"].value_counts().rename_axis("donor_clone_id").reset_index(name="n_synthetic_cells")
    clone_counts.to_csv(outdir / f"{args.sample}.synthetic_clone_counts.tsv", sep="\t", index=False)

    summary = {
        "sample": args.sample,
        "n_synthetic_cells": int(len(out)),
        "n_unique_synthetic_labels": int(out["synthetic_celllabel"].nunique()),
        "n_unique_donor_barcodes_used": int(out["donor_original_barcode"].nunique()),
        "n_unique_donor_clones_used": int(out["donor_clone_id"].nunique()),
        "assign_counts_by_label": assign_counts,
        "outputs": {
            "synthetic_cells_with_lineage": str(out_path),
            "synthetic_clone_counts": str(outdir / f"{args.sample}.synthetic_clone_counts.tsv"),
        },
    }
    (outdir / f"{args.sample}.synthetic_lineage_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
