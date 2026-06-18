#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute usability rank and reversed rank from overall usability scores.

Input is expected to contain columns:
    Method, Overall
or:
    method, usability_score
Higher usability score is better.
"""
import argparse
from pathlib import Path
import re
import pandas as pd


def canonical_method(x):
    if pd.isna(x):
        return ""
    name = re.sub(r"\s+", " ", str(x).strip())
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
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    method_col = "Method" if "Method" in df.columns else "method"
    if method_col not in df.columns:
        raise ValueError("Cannot find Method/method column")
    score_col = None
    for c in ["Overall", "usability_score", "score"]:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        raise ValueError("Cannot find usability score column: expected Overall/usability_score/score")

    out = pd.DataFrame({
        "method": df[method_col].map(canonical_method),
        "usability_score": pd.to_numeric(df[score_col], errors="coerce"),
    })
    out = out[out["method"].ne("")].copy()
    out = out.sort_values(["usability_score", "method"], ascending=[False, True]).reset_index(drop=True)
    out["final_usability_rank"] = range(1, len(out) + 1)
    out["reversed_rank"] = len(out) + 1 - out["final_usability_rank"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
