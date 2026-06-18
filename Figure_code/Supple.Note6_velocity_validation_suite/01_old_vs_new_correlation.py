#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis 01: old ranking vs revised ranking.

Purpose
-------
This script compares the previous accuracy ranking against the revised
accuracy ranking. It also quantifies whether the revised ranking
is more concordant with Directionality than with Consistency.

Required inputs
---------------
1. results_dir from the pipeline:
   ../../../PlotData/Results/accuracy

2. An optional old accuracy CSV with at least:
   - method column: Method or method
   - old accuracy column: Accuracy(real+simulate), Accuracy, old_accuracy_score, or user-specified

If no old CSV is provided, the script still computes new-rank correlations with
Directionality and Consistency, but skips old-vs-new comparisons.

Outputs
-------
- 01_new_ranking_family_correlation.csv
- 02_old_vs_new_common_methods.csv, if old CSV is available
- 03_old_vs_new_correlations.csv, if old CSV is available
- validation_summary.md
- figures/*.png/pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from validation_common import (  # noqa: E402
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_EXCLUDE_METHODS,
    ensure_dir,
    read_metric_scores,
    recompute_family_scores,
    compute_dc_weight_sweep,
    add_primary_ranking_columns,
    corr_one,
    normalize_method_name,
    save_csv_readable,
    save_markdown_table,
    setup_matplotlib,
    save_figure,
    write_text,
)


def infer_old_columns(df: pd.DataFrame, method_col: str | None, score_col: str | None):
    if method_col is None:
        candidates = ["Method", "method", "Tool", "tool"]
        method_col = next((c for c in candidates if c in df.columns), None)
    if score_col is None:
        candidates = [
            "Accuracy(real+simulate)",
            "Accuracy(real+simulation)",
            "Accuracy",
            "accuracy",
            "old_accuracy_score",
            "old_accuracy",
        ]
        score_col = next((c for c in candidates if c in df.columns), None)
    if method_col is None or score_col is None:
        raise ValueError(
            "Cannot infer method or old accuracy column. "
            "Use --old_method_col and --old_score_col. "
            f"Available columns: {list(df.columns)}"
        )
    return method_col, score_col


def load_old_accuracy(path: Path, method_col: str | None, score_col: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    method_col, score_col = infer_old_columns(df, method_col, score_col)
    out = df[[method_col, score_col]].copy()
    out.columns = ["method", "old_accuracy_score"]
    out["method"] = out["method"].map(normalize_method_name)
    out["old_accuracy_score"] = pd.to_numeric(out["old_accuracy_score"], errors="coerce")
    out = out.dropna(subset=["method", "old_accuracy_score"])
    out["old_accuracy_rank"] = out["old_accuracy_score"].rank(ascending=False, method="average")
    return out


def plot_rank_scatter(common: pd.DataFrame, outdir: Path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.2, 4.7))
    ax.scatter(common["old_accuracy_rank"], common["D_plus_C_rank"], s=36)
    for _, row in common.iterrows():
        ax.text(row["old_accuracy_rank"] + 0.08, row["D_plus_C_rank"] + 0.08, row["method"], fontsize=6)
    lim_max = np.nanmax([common["old_accuracy_rank"].max(), common["D_plus_C_rank"].max()]) + 1
    ax.plot([0, lim_max], [0, lim_max], linestyle="--", linewidth=1)
    ax.set_xlim(0.5, lim_max)
    ax.set_ylim(lim_max, 0.5)
    ax.invert_xaxis()  # better ranks are closer to the upper-right after y inversion
    ax.set_xlabel("Old accuracy rank")
    ax.set_ylabel("Revised D+C rank")
    ax.set_title("Old vs revised accuracy ranking")
    save_figure(fig, outdir / "fig1_old_vs_new_rank_scatter")


def plot_family_correlation_bars(corr_df: pd.DataFrame, outdir: Path):
    plt = setup_matplotlib()
    d = corr_df[corr_df["method"] == "spearman"].copy()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(d["comparison"], d["rho"])
    ax.axhline(0, linewidth=0.8)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Rank concordance diagnostics")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, outdir / "fig2_rank_family_correlations")


def main():
    parser = argparse.ArgumentParser(description="Compare old and revised rankings.")
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_ANALYSIS_DIR / "01_old_vs_new_correlation"))
    parser.add_argument("--old_accuracy_csv", default=None, help="Optional old ranking CSV.")
    parser.add_argument("--old_method_col", default=None)
    parser.add_argument("--old_score_col", default=None)
    parser.add_argument("--exclude_methods", default=";".join(DEFAULT_EXCLUDE_METHODS))
    parser.add_argument("--include_gt_in_directionality", action="store_true")
    args = parser.parse_args()

    outdir = ensure_dir(args.output_dir)
    figdir = ensure_dir(outdir / "figures")
    exclude_methods = [x for x in args.exclude_methods.split(";") if x]

    metric_df = read_metric_scores(args.results_dir)
    fam = recompute_family_scores(
        metric_df,
        include_gt_in_directionality=args.include_gt_in_directionality,
        exclude_methods=exclude_methods,
    )
    _, dc = compute_dc_weight_sweep(fam)
    dc = add_primary_ranking_columns(dc)

    corr_rows = []
    for x_col, y_col, name in [
        ("D_plus_C_rank", "Directionality_rank", "new rank vs Directionality rank"),
        ("D_plus_C_rank", "Consistency_rank", "new rank vs Consistency rank"),
        ("dc_mean_score", "Directionality", "new score vs Directionality score"),
        ("dc_mean_score", "Consistency", "new score vs Consistency score"),
    ]:
        for method in ["spearman", "kendall", "pearson"]:
            rho, p, n = corr_one(dc[x_col], dc[y_col], method=method)
            corr_rows.append({"comparison": name, "method": method, "rho": rho, "p_value": p, "n_methods": n})
    corr_df = pd.DataFrame(corr_rows)
    save_csv_readable(corr_df, outdir / "01_new_ranking_family_correlation.csv")
    plot_family_correlation_bars(corr_df, figdir)

    summary_lines = []
    summary_lines.append("# Old vs revised ranking validation\n")
    summary_lines.append(f"- Results directory: `{args.results_dir}`")
    summary_lines.append(f"- Excluded methods: {exclude_methods}")
    summary_lines.append(f"- Directionality includes GT: {args.include_gt_in_directionality}\n")
    summary_lines.append("## New ranking vs family ranks\n")
    summary_lines.append(corr_df[corr_df["method"] == "spearman"].to_markdown(index=False))

    if args.old_accuracy_csv:
        old = load_old_accuracy(Path(args.old_accuracy_csv), args.old_method_col, args.old_score_col)
        common = old.merge(dc, on="method", how="inner")
        save_csv_readable(common.sort_values("D_plus_C_rank"), outdir / "02_old_vs_new_common_methods.csv")
        plot_rank_scatter(common, figdir)

        old_corr_rows = []
        pairs = [
            ("old_accuracy_rank", "D_plus_C_rank", "old rank vs revised D+C rank"),
            ("old_accuracy_score", "dc_mean_score", "old score vs revised D+C score"),
            ("old_accuracy_rank", "Directionality_rank", "old rank vs Directionality rank"),
            ("old_accuracy_rank", "Consistency_rank", "old rank vs Consistency rank"),
            ("D_plus_C_rank", "Directionality_rank", "new rank vs Directionality rank"),
            ("D_plus_C_rank", "Consistency_rank", "new rank vs Consistency rank"),
        ]
        for x_col, y_col, name in pairs:
            for method in ["spearman", "kendall", "pearson"]:
                rho, p, n = corr_one(common[x_col], common[y_col], method=method)
                old_corr_rows.append({"comparison": name, "method": method, "rho": rho, "p_value": p, "n_methods": n})
        old_corr = pd.DataFrame(old_corr_rows)
        save_csv_readable(old_corr, outdir / "03_old_vs_new_correlations.csv")
        summary_lines.append("\n## Old-vs-new correlations\n")
        summary_lines.append(old_corr[old_corr["method"] == "spearman"].to_markdown(index=False))
    else:
        summary_lines.append("\nNo old accuracy CSV was provided, so old-vs-new comparison was skipped.\n")
        template = pd.DataFrame({"Method": ["veloVI", "UniTVelo"], "Accuracy(real+simulate)": [16.2946, 12.9911]})
        template.to_csv(outdir / "old_accuracy_template.csv", index=False)

    write_text(outdir / "validation_summary.md", "\n".join(summary_lines))
    print(f"Done. Results written to: {outdir}")


if __name__ == "__main__":
    main()
