#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Scalability rank from Docker_performance_0605.xlsx.

This version is specific to the current Excel structure:

  - Sheet "time": runtime measurements
  - Sheet "memory": memory measurements
  - Sheet "GPU_memory": ignored

Each sheet is expected to be a wide matrix:

  Method | dataset_1 | dataset_2 | ...

Cells with error messages such as "killed", "Stuck", "AssertionError", etc.
are converted to NA and are not used in z-score/rank calculations.

Ranking rule
------------
For each dataset column within each sheet:

  z = (value - mean(value)) / (std(value) + 1e-8)

Runtime and memory are both lower-is-better, so we reverse the z-score:

  scalability_component_score = -z

For each dataset × metric, methods are ranked by this reversed score
(equivalent to raw value ascending). Missing values stay NA.

For each method:

  mean_time_rank   = mean(time ranks across datasets, skip NA)
  mean_memory_rank = mean(memory ranks across datasets, skip NA)
  mean_scalability_rank = mean(all time + memory dataset ranks, skip NA)

Final rank is obtained by sorting mean_scalability_rank ascending.
reversed_rank is computed as:

  reversed_rank = n_methods + 1 - final_scalability_rank

Thus larger reversed_rank means better scalability.

Default output
--------------
Full results:
  PlotData/Results/scalability

Requested integration file:
  PlotData/Results/reversed_rank/scalability_rank.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_INPUT_XLSX = Path("PlotData/scalability/Docker_performance_0605.xlsx")
DEFAULT_OUTPUT_DIR = Path("PlotData/Results/scalability")
DEFAULT_REVERSED_DIR = Path("PlotData/Results/reversed_rank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute scalability rank from time and memory sheets."
    )
    parser.add_argument("--input_xlsx", default=str(DEFAULT_INPUT_XLSX))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reversed_dir", default=str(DEFAULT_REVERSED_DIR))
    parser.add_argument("--method_col", default="Method")
    parser.add_argument("--time_sheet", default="time")
    parser.add_argument("--memory_sheet", default="memory")
    parser.add_argument(
        "--zero_as_na",
        action="store_true",
        help="Treat numeric zero values as NA. Default keeps zero as a valid value.",
    )
    parser.add_argument(
        "--exclude_methods",
        default=None,
        help="Optional comma/semicolon-separated methods to exclude before ranking.",
    )
    return parser.parse_args()


def split_methods(x: str | None) -> set[str]:
    if x is None or str(x).strip() == "":
        return set()
    out = []
    for item in str(x).replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(item)
    return set(out)


def clean_method_name(x) -> str:
    return str(x).strip()


def read_wide_sheet(
    xlsx_path: Path,
    sheet_name: str,
    metric: str,
    method_col: str = "Method",
    zero_as_na: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one wide sheet and convert it to long format.

    Returns
    -------
    long_df:
        method, dataset, metric, raw_value, value, is_numeric, is_missing
    qc_df:
        dataset-level QC counts
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    if method_col not in raw.columns:
        # Fallback: use the first column as method if Method is not found.
        first_col = raw.columns[0]
        raw = raw.rename(columns={first_col: method_col})

    dataset_cols = [c for c in raw.columns if c != method_col]

    # Remove fully empty dataset columns.
    dataset_cols = [c for c in dataset_cols if not raw[c].isna().all()]

    long = raw.melt(
        id_vars=[method_col],
        value_vars=dataset_cols,
        var_name="dataset",
        value_name="raw_value",
    )
    long = long.rename(columns={method_col: "method"})
    long["method"] = long["method"].map(clean_method_name)
    long["dataset"] = long["dataset"].astype(str).str.strip()
    long["metric"] = metric

    # Convert numeric values. Error messages become NA.
    long["value"] = pd.to_numeric(long["raw_value"], errors="coerce")
    if zero_as_na:
        long.loc[long["value"].eq(0), "value"] = np.nan

    long["is_numeric"] = long["value"].notna()
    long["is_missing"] = long["value"].isna()
    long["raw_value_string"] = long["raw_value"].astype(str)

    # Remove blank / nan method labels.
    bad_method = long["method"].isin(["", "nan", "None"])
    long = long.loc[~bad_method].copy()

    qc = (
        long.groupby(["metric", "dataset"], as_index=False)
        .agg(
            n_methods_total=("method", "nunique"),
            n_numeric_values=("value", lambda x: int(x.notna().sum())),
            n_missing_values=("value", lambda x: int(x.isna().sum())),
            n_error_or_text_values=("raw_value", lambda x: int(pd.to_numeric(x, errors="coerce").isna().sum())),
        )
    )
    return long, qc


def add_dataset_zscore_and_rank(long: pd.DataFrame) -> pd.DataFrame:
    """Compute per-dataset z-score, reversed score, and rank.

    Both time and memory are lower-is-better, so score = -z.
    """
    df = long.copy()

    def _zscore(s: pd.Series) -> pd.Series:
        vals = pd.to_numeric(s, errors="coerce")
        mu = vals.mean(skipna=True)
        sd = vals.std(skipna=True)
        if pd.isna(sd):
            sd = 0.0
        return (vals - mu) / (sd + 1e-8)

    df["z_value"] = df.groupby(["metric", "dataset"])["value"].transform(_zscore)
    df["component_score"] = -df["z_value"]

    # Rank within each metric × dataset. Lower raw value is better.
    df["component_rank"] = (
        df.groupby(["metric", "dataset"])["value"]
        .rank(ascending=True, method="average")
    )
    return df


def summarize_method_scores(scored: pd.DataFrame, exclude_methods: set[str]) -> pd.DataFrame:
    df = scored.copy()
    if exclude_methods:
        df = df.loc[~df["method"].isin(exclude_methods)].copy()

    # Summary per method and metric.
    metric_summary = (
        df.groupby(["method", "metric"], as_index=False)
        .agg(
            mean_metric_rank=("component_rank", "mean"),
            median_metric_rank=("component_rank", "median"),
            mean_metric_score=("component_score", "mean"),
            n_valid_metric_values=("value", lambda x: int(x.notna().sum())),
            n_total_metric_values=("dataset", "nunique"),
        )
    )

    # Pivot metric-level summaries.
    rank_wide = metric_summary.pivot(index="method", columns="metric", values="mean_metric_rank")
    score_wide = metric_summary.pivot(index="method", columns="metric", values="mean_metric_score")
    valid_wide = metric_summary.pivot(index="method", columns="metric", values="n_valid_metric_values")
    total_wide = metric_summary.pivot(index="method", columns="metric", values="n_total_metric_values")

    methods = sorted(df["method"].dropna().unique())
    out = pd.DataFrame({"method": methods})

    out["mean_time_rank"] = out["method"].map(rank_wide.get("time", pd.Series(dtype=float)))
    out["mean_memory_rank"] = out["method"].map(rank_wide.get("memory", pd.Series(dtype=float)))
    out["mean_time_score"] = out["method"].map(score_wide.get("time", pd.Series(dtype=float)))
    out["mean_memory_score"] = out["method"].map(score_wide.get("memory", pd.Series(dtype=float)))
    out["n_valid_time_values"] = out["method"].map(valid_wide.get("time", pd.Series(dtype=float)))
    out["n_valid_memory_values"] = out["method"].map(valid_wide.get("memory", pd.Series(dtype=float)))
    out["n_total_time_datasets"] = out["method"].map(total_wide.get("time", pd.Series(dtype=float)))
    out["n_total_memory_datasets"] = out["method"].map(total_wide.get("memory", pd.Series(dtype=float)))

    # Mean across all available dataset-level ranks from both time and memory.
    combined = (
        df.groupby("method", as_index=False)
        .agg(
            mean_scalability_rank=("component_rank", "mean"),
            median_scalability_rank=("component_rank", "median"),
            best_scalability_rank=("component_rank", "min"),
            worst_scalability_rank=("component_rank", "max"),
            mean_scalability_score=("component_score", "mean"),
            n_valid_values=("value", lambda x: int(x.notna().sum())),
            n_total_values=("dataset", "size"),
            n_datasets_with_any_value=("dataset", lambda x: int(x[df.loc[x.index, "value"].notna()].nunique())),
        )
    )
    out = out.merge(combined, on="method", how="left")

    out["time_value_coverage"] = out["n_valid_time_values"] / out["n_total_time_datasets"]
    out["memory_value_coverage"] = out["n_valid_memory_values"] / out["n_total_memory_datasets"]
    out["overall_value_coverage"] = out["n_valid_values"] / out["n_total_values"]

    valid = out.loc[out["mean_scalability_rank"].notna()].copy()
    missing = out.loc[out["mean_scalability_rank"].isna()].copy()

    # Lower mean rank is better; if tied, higher mean score is better.
    valid = valid.sort_values(
        ["mean_scalability_rank", "mean_scalability_score", "method"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    valid["final_scalability_rank"] = np.arange(1, len(valid) + 1, dtype=int)

    if not missing.empty:
        missing = missing.sort_values("method").reset_index(drop=True)
        missing["final_scalability_rank"] = np.arange(
            len(valid) + 1,
            len(valid) + len(missing) + 1,
            dtype=int,
        )

    final = pd.concat([valid, missing], ignore_index=True, sort=False)
    n_methods = len(final)
    final["reversed_rank"] = n_methods + 1 - final["final_scalability_rank"]
    return final, metric_summary


def main() -> None:
    args = parse_args()

    input_xlsx = Path(args.input_xlsx)
    output_dir = Path(args.output_dir)
    reversed_dir = Path(args.reversed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reversed_dir.mkdir(parents=True, exist_ok=True)

    if not input_xlsx.exists():
        raise FileNotFoundError(f"Input Excel file not found: {input_xlsx}")

    exclude_methods = split_methods(args.exclude_methods)

    time_long, time_qc = read_wide_sheet(
        xlsx_path=input_xlsx,
        sheet_name=args.time_sheet,
        metric="time",
        method_col=args.method_col,
        zero_as_na=args.zero_as_na,
    )
    memory_long, memory_qc = read_wide_sheet(
        xlsx_path=input_xlsx,
        sheet_name=args.memory_sheet,
        metric="memory",
        method_col=args.method_col,
        zero_as_na=args.zero_as_na,
    )

    long = pd.concat([time_long, memory_long], ignore_index=True)
    qc_by_dataset = pd.concat([time_qc, memory_qc], ignore_index=True)

    scored = add_dataset_zscore_and_rank(long)
    summary, metric_summary = summarize_method_scores(scored, exclude_methods=exclude_methods)

    # Save full outputs.
    long.to_csv(output_dir / "01_scalability_time_memory_long_raw.csv", index=False)
    scored.to_csv(output_dir / "02_scalability_time_memory_dataset_scores.csv", index=False)
    metric_summary.to_csv(output_dir / "03_scalability_metric_level_summary.csv", index=False)
    summary.to_csv(output_dir / "04_scalability_rank_summary.csv", index=False)
    qc_by_dataset.to_csv(output_dir / "05_scalability_dataset_qc.csv", index=False)

    # Requested reversed-rank output.
    keep_cols = [
        "method",
        "final_scalability_rank",
        "reversed_rank",
        "mean_scalability_rank",
        "median_scalability_rank",
        "best_scalability_rank",
        "worst_scalability_rank",
        "mean_scalability_score",
        "mean_time_rank",
        "mean_memory_rank",
        "mean_time_score",
        "mean_memory_score",
        "n_valid_time_values",
        "n_valid_memory_values",
        "n_valid_values",
        "n_total_values",
        "time_value_coverage",
        "memory_value_coverage",
        "overall_value_coverage",
    ]
    keep_cols = [c for c in keep_cols if c in summary.columns]
    reversed_table = summary[keep_cols].copy()
    reversed_table.to_csv(reversed_dir / "scalability_rank.csv", index=False)

    qc = pd.DataFrame(
        [
            {"item": "input_xlsx", "value": str(input_xlsx)},
            {"item": "time_sheet", "value": args.time_sheet},
            {"item": "memory_sheet", "value": args.memory_sheet},
            {"item": "gpu_memory_included", "value": False},
            {"item": "zero_as_na", "value": bool(args.zero_as_na)},
            {"item": "n_methods", "value": int(summary.shape[0])},
            {"item": "n_time_datasets", "value": int(time_long["dataset"].nunique())},
            {"item": "n_memory_datasets", "value": int(memory_long["dataset"].nunique())},
            {"item": "n_numeric_time_values", "value": int(time_long["value"].notna().sum())},
            {"item": "n_numeric_memory_values", "value": int(memory_long["value"].notna().sum())},
            {"item": "n_missing_or_error_time_values", "value": int(time_long["value"].isna().sum())},
            {"item": "n_missing_or_error_memory_values", "value": int(memory_long["value"].isna().sum())},
            {"item": "exclude_methods", "value": ";".join(sorted(exclude_methods)) if exclude_methods else ""},
            {"item": "requested_output", "value": str(reversed_dir / "scalability_rank.csv")},
        ]
    )
    qc.to_csv(output_dir / "06_scalability_qc_summary.csv", index=False)

    print("Done.")
    print(f"Full results: {output_dir}")
    print(f"Requested reversed-rank file: {reversed_dir / 'scalability_rank.csv'}")
    print("GPU_memory sheet was ignored.")


if __name__ == "__main__":
    main()
