#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create results/reversed_rank/accuracy_rank.csv from accuracy_rank_summary.csv.

This helper does not change the accuracy scoring logic. It only converts the
accuracy pipeline rank summary into the common reversed-rank format required by
the overall-rank scripts.
"""
import argparse
from pathlib import Path
import re
import pandas as pd


def normalize_method(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip())


def canonical_method(x):
    name = normalize_method(x)
    key = re.sub(r"[^a-z0-9]+", "", name.lower())
    if key in {"scrnakinetics", "scrnakinetic"}:
        return "scRNAkinetics"
    if key == "regionvelocity":
        return "Region Velocity"
    if key == "topovelo":
        return "TopoVelo"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank_summary", required=True, help="Path to accuracy_rank_summary.csv")
    ap.add_argument("--output", required=True, help="Output accuracy_rank.csv")
    ap.add_argument("--region_method", default="Region Velocity")
    args = ap.parse_args()

    df = pd.read_csv(args.rank_summary)
    if "method" not in df.columns:
        raise ValueError("Input must contain a method column")
    if "final_accuracy_rank" not in df.columns:
        raise ValueError("Input must contain final_accuracy_rank")

    df = df.copy()
    df["method"] = df["method"].map(canonical_method)
    region = df[df["method"].eq(args.region_method)].copy()
    main_df = df[~df["method"].eq(args.region_method)].copy()
    main_df = main_df.sort_values(["final_accuracy_rank", "method"], ascending=[True, True]).reset_index(drop=True)
    main_df["final_accuracy_rank"] = range(1, len(main_df) + 1)

    if not region.empty:
        r = region.iloc[[0]].copy()
        r["final_accuracy_rank"] = len(main_df) + 1
        main_df = pd.concat([main_df, r], ignore_index=True, sort=False)

    n = len(main_df)
    main_df["reversed_rank"] = n + 1 - main_df["final_accuracy_rank"]

    out_cols = ["method", "final_accuracy_rank", "reversed_rank"]
    for c in ["mean_accuracy_rank", "median_accuracy_rank", "best_accuracy_rank", "worst_accuracy_rank", "mean_accuracy_score", "Directionality", "Consistency"]:
        if c in main_df.columns:
            out_cols.append(c)
    out = main_df[out_cols]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
