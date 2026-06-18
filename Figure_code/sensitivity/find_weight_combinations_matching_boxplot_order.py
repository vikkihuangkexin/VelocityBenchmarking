#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find weight combinations whose ranking is consistent with the boxplot order.

Context
-------
In compute_overall_weight_sensitivity.py, the boxplot method order is:

    summary.sort_values("mean_rank")["method"]

where mean_rank is the mean overall rank across all valid weight combinations.

This script:
1. Reads the sensitivity-analysis outputs.
2. Reconstructs the boxplot order based on 05_sensitivity_summary_by_method.csv.
3. Compares every weight combination's ranking against that boxplot order.
4. Outputs:
   - exact full-order matches, if any
   - exact top-k matches, e.g. top5/top10
   - closest weight combinations by Kendall tau / Spearman correlation
   - per-combination rank comparison tables for the best matches

Default input:
    PlotData/Results/reversed_rank/Results/Sensitivity

Default output:
    PlotData/Results/reversed_rank/Results/Sensitivity/matching_boxplot_order
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


DEFAULT_SENS_DIR = Path("PlotData/Results/reversed_rank/Results/Sensitivity")
DEFAULT_TOPKS = "5,10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find weight combinations matching the boxplot mean-rank order."
    )
    parser.add_argument("--sensitivity_dir", default=str(DEFAULT_SENS_DIR))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--topks",
        default=DEFAULT_TOPKS,
        help="Comma-separated top-k values for exact top-k order checks, e.g. 5,10.",
    )
    parser.add_argument(
        "--n_best",
        type=int,
        default=20,
        help="Number of closest combinations to export in detailed comparison tables.",
    )
    return parser.parse_args()


def parse_topks(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_inputs(sens_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_path = sens_dir / "05_sensitivity_summary_by_method.csv"
    rank_long_path = sens_dir / "03_rank_by_weight_long.csv"
    grid_path = sens_dir / "01_weight_grid.csv"

    for p in [summary_path, rank_long_path, grid_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    summary = pd.read_csv(summary_path)
    rank_long = pd.read_csv(rank_long_path)
    grid = pd.read_csv(grid_path)

    required_summary = {"method", "mean_rank"}
    required_long = {"method", "combo_id", "overall_order_rank"}
    required_grid = {"combo_id", "w_accuracy", "w_scalability", "w_stability", "w_usability"}

    if not required_summary.issubset(summary.columns):
        raise ValueError(f"{summary_path} lacks required columns: {required_summary}")
    if not required_long.issubset(rank_long.columns):
        raise ValueError(f"{rank_long_path} lacks required columns: {required_long}")
    if not required_grid.issubset(grid.columns):
        raise ValueError(f"{grid_path} lacks required columns: {required_grid}")

    return summary, rank_long, grid


def target_order_from_boxplot(summary: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the row order used by the boxplot."""
    # This matches the plotting code:
    # ordered_methods = summary.sort_values("mean_rank")["method"].tolist()
    out = summary.sort_values(["mean_rank", "method"], ascending=[True, True]).reset_index(drop=True)
    out["target_position"] = np.arange(1, len(out) + 1, dtype=int)
    return out[["method", "target_position", "mean_rank", "median_rank", "best_rank", "worst_rank"]]


def compare_one_combo(combo_df: pd.DataFrame, target: pd.DataFrame, topks: list[int]) -> dict:
    d = target[["method", "target_position"]].merge(
        combo_df[["method", "overall_order_rank", "overall_score"]],
        on="method",
        how="inner",
    )

    # Convert the combo's rank to order position. Smaller rank means better.
    d = d.sort_values(["overall_order_rank", "method"], ascending=[True, True]).reset_index(drop=True)
    d["combo_position"] = np.arange(1, len(d) + 1, dtype=int)

    # Re-align by method for correlations/differences.
    aligned = target[["method", "target_position"]].merge(
        d[["method", "combo_position", "overall_order_rank", "overall_score"]],
        on="method",
        how="inner",
    )

    n = len(aligned)
    if n == 0:
        return {}

    # Exact full order check.
    aligned_by_target = aligned.sort_values("target_position")
    combo_order_in_target_space = aligned_by_target["combo_position"].to_numpy()
    exact_full_order = bool(np.array_equal(combo_order_in_target_space, np.arange(1, n + 1)))

    # Correlations.
    ktau, kp = kendalltau(aligned["target_position"], aligned["combo_position"])
    srho, sp = spearmanr(aligned["target_position"], aligned["combo_position"])

    # Rank distance.
    abs_diff = np.abs(aligned["target_position"] - aligned["combo_position"])
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())

    # Count pairwise inversions relative to target order.
    arr = aligned_by_target["combo_position"].to_numpy()
    inversions = 0
    for i in range(n):
        inversions += int(np.sum(arr[i + 1:] < arr[i]))

    row = {
        "n_methods_compared": n,
        "exact_full_order_match": exact_full_order,
        "kendall_tau_vs_boxplot_order": float(ktau) if pd.notna(ktau) else np.nan,
        "kendall_p": float(kp) if pd.notna(kp) else np.nan,
        "spearman_rho_vs_boxplot_order": float(srho) if pd.notna(srho) else np.nan,
        "spearman_p": float(sp) if pd.notna(sp) else np.nan,
        "mean_abs_rank_position_difference": mean_abs_diff,
        "max_abs_rank_position_difference": max_abs_diff,
        "pairwise_inversions": inversions,
    }

    for k in topks:
        target_top = (
            target.sort_values("target_position")
            .head(k)["method"]
            .tolist()
        )
        combo_top = (
            d.sort_values("combo_position")
            .head(k)["method"]
            .tolist()
        )
        row[f"exact_top{k}_order_match"] = bool(target_top == combo_top)
        row[f"same_top{k}_set"] = bool(set(target_top) == set(combo_top))
        row[f"top{k}_overlap_count"] = len(set(target_top).intersection(combo_top))

    return row


def build_combo_diagnostics(rank_long: pd.DataFrame, grid: pd.DataFrame, target: pd.DataFrame, topks: list[int]) -> pd.DataFrame:
    rows = []
    for combo_id, sub in rank_long.groupby("combo_id", sort=False):
        row = compare_one_combo(sub, target, topks)
        if not row:
            continue
        row["combo_id"] = combo_id
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.merge(grid, on="combo_id", how="left")

    # A useful sorting order: exact full matches first, then top-k matches,
    # then closest global rank agreement.
    top_sort_cols = []
    for k in sorted(topks):
        col = f"exact_top{k}_order_match"
        if col in out.columns:
            top_sort_cols.append(col)

    sort_cols = (
        ["exact_full_order_match"]
        + top_sort_cols
        + [
            "kendall_tau_vs_boxplot_order",
            "spearman_rho_vs_boxplot_order",
            "pairwise_inversions",
            "mean_abs_rank_position_difference",
        ]
    )
    ascending = (
        [False]
        + [False] * len(top_sort_cols)
        + [False, False, True, True]
    )

    out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return out


def export_best_comparison_tables(
    rank_long: pd.DataFrame,
    target: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    n_best: int,
):
    compare_dir = output_dir / "best_combo_rank_comparisons"
    compare_dir.mkdir(parents=True, exist_ok=True)

    best_combo_ids = diagnostics.head(n_best)["combo_id"].tolist()

    for idx, combo_id in enumerate(best_combo_ids, start=1):
        sub = rank_long.loc[rank_long["combo_id"] == combo_id].copy()
        comp = target.merge(
            sub[["method", "overall_order_rank", "overall_rank", "overall_score"]],
            on="method",
            how="left",
        )
        comp = comp.sort_values("target_position").reset_index(drop=True)
        comp["position_difference_combo_minus_boxplot"] = (
            comp["overall_order_rank"] - comp["target_position"]
        )

        comp.to_csv(
            compare_dir / f"rank_comparison_{idx:02d}_{combo_id}.csv",
            index=False,
        )


def main() -> None:
    args = parse_args()

    sens_dir = Path(args.sensitivity_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sens_dir / "matching_boxplot_order"
    output_dir.mkdir(parents=True, exist_ok=True)

    topks = parse_topks(args.topks)

    summary, rank_long, grid = load_inputs(sens_dir)
    target = target_order_from_boxplot(summary)

    diagnostics = build_combo_diagnostics(rank_long, grid, target, topks=topks)

    # Exact matches.
    exact_full = diagnostics.loc[diagnostics["exact_full_order_match"]].copy()

    target.to_csv(output_dir / "01_boxplot_target_order_mean_rank.csv", index=False)
    diagnostics.to_csv(output_dir / "02_weight_combinations_rank_agreement.csv", index=False)
    exact_full.to_csv(output_dir / "03_exact_full_order_matches.csv", index=False)

    # Top-k exact matches.
    for k in topks:
        col = f"exact_top{k}_order_match"
        if col in diagnostics.columns:
            diagnostics.loc[diagnostics[col]].to_csv(
                output_dir / f"04_exact_top{k}_order_matches.csv",
                index=False,
            )
        set_col = f"same_top{k}_set"
        if set_col in diagnostics.columns:
            diagnostics.loc[diagnostics[set_col]].to_csv(
                output_dir / f"05_same_top{k}_set_matches.csv",
                index=False,
            )

    export_best_comparison_tables(rank_long, target, diagnostics, output_dir, args.n_best)

    # Summary text.
    lines = []
    lines.append("# Weight combinations matching the boxplot order\n")
    lines.append(f"- Sensitivity directory: `{sens_dir}`")
    lines.append(f"- Output directory: `{output_dir}`")
    lines.append(f"- Number of weight combinations checked: {len(diagnostics)}")
    lines.append("")
    lines.append("## What is the boxplot order?")
    lines.append("")
    lines.append(
        "The row order in `fig1_rank_distribution_boxplot` is sorted by "
        "`mean_rank` from `05_sensitivity_summary_by_method.csv`, i.e. "
        "the average overall rank of each method across all valid weight combinations."
    )
    lines.append("")
    lines.append("Therefore, the boxplot row order is not a single weight setting; it is a mean-rank order across the full grid.")
    lines.append("")
    lines.append("## Exact full-order matches")
    lines.append("")
    lines.append(f"- Exact full-order matches: {len(exact_full)}")
    lines.append("")
    for k in topks:
        col = f"exact_top{k}_order_match"
        set_col = f"same_top{k}_set"
        if col in diagnostics.columns:
            lines.append(f"- Exact top-{k} order matches: {int(diagnostics[col].sum())}")
        if set_col in diagnostics.columns:
            lines.append(f"- Same top-{k} set matches: {int(diagnostics[set_col].sum())}")
    lines.append("")
    lines.append("## Closest weight combinations")
    lines.append("")
    cols = [
        "combo_id",
        "w_accuracy",
        "w_scalability",
        "w_stability",
        "w_usability",
        "kendall_tau_vs_boxplot_order",
        "spearman_rho_vs_boxplot_order",
        "pairwise_inversions",
        "mean_abs_rank_position_difference",
    ]
    cols = [c for c in cols if c in diagnostics.columns]
    lines.append(diagnostics[cols].head(10).to_markdown(index=False))
    lines.append("")
    lines.append("Main table: `02_weight_combinations_rank_agreement.csv`.")

    (output_dir / "README_matching_boxplot_order.md").write_text("\n".join(lines), encoding="utf-8")

    print("Done.")
    print(f"Target order: {output_dir / '01_boxplot_target_order_mean_rank.csv'}")
    print(f"All combo diagnostics: {output_dir / '02_weight_combinations_rank_agreement.csv'}")
    print(f"Exact full matches: {output_dir / '03_exact_full_order_matches.csv'}")
    print(f"README: {output_dir / 'README_matching_boxplot_order.md'}")


if __name__ == "__main__":
    main()
