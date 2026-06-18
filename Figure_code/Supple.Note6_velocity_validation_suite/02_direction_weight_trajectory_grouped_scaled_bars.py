#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis 02 revised: grouped Directionality/Consistency weight trajectory diagnostics.

Purpose
-------
This script improves the original 02_direction_weight_trajectory.py by:

1. Classifying tools into three diagnostic profiles:
   - Directionality-driven:
       Directionality rank is substantially better than Consistency rank.
   - Consistency-driven:
       Consistency rank is substantially better than Directionality rank.
   - Balanced:
       Directionality and Consistency ranks differ less than the threshold.

2. Plotting rank trajectories separately for the three profiles, so the
   trajectory figure is not overloaded with too many overlapping lines.

3. Adding separate Directionality and Consistency score bar plots,
   sorted from high to low, so each family profile can be inspected directly.

Definitions
-----------
directionality_advantage_rank = Consistency_rank - Directionality_rank

Because rank is lower-is-better:
  - positive value: Directionality rank is better than Consistency rank
  - negative value: Consistency rank is better than Directionality rank

Default classification:
  - Directionality-driven: directionality_advantage_rank >= 5
  - Consistency-driven:    directionality_advantage_rank <= -5
  - Balanced:              otherwise

This classification is used only for visualization and diagnostic summaries.
It does NOT affect final accuracy rank calculation.

Expected input
--------------
This script expects the results directory, especially:
  metric_scores.csv

It uses validation_common.py from the same directory as this script.

Outputs
-------
CSV:
  01_rank_by_weight_wide.csv
  02_directionality_advantage_rank_change.csv
  03_profile_assignment.csv
  04_directionality_driven_tools.csv
  05_consistency_driven_tools.csv
  06_balanced_tools.csv
  07_rank_change_correlations.csv

Figures:
  fig1a_rank_trajectory_directionality_driven.png/pdf
  fig1b_rank_trajectory_consistency_driven.png/pdf
  fig1c_rank_trajectory_balanced.png/pdf
  fig2_directionality_advantage_vs_rank_improvement.png/pdf
  fig3a_directionality_score_bar.png/pdf
  fig3b_consistency_score_bar.png/pdf
  fig4_profile_counts.png/pdf

Author: OpenAI ChatGPT
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

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
    parse_weight_grid,
    weight_label,
    corr_one,
    save_csv_readable,
    setup_matplotlib,
    save_figure,
    write_text,
)


PROFILE_ORDER = [
    "Directionality-driven",
    "Balanced",
    "Consistency-driven",
]

# Consistent visual identity across trajectory plots, scatter plots, and bar plots.
# Directionality-driven tools use a cool blue-purple palette.
# Consistency-driven tools use a warm yellow-orange palette.
# Balanced tools use a green/teal palette.
PROFILE_BASE_COLORS = {
    "Directionality-driven": "#4F46E5",   # indigo / blue-purple
    "Balanced": "#059669",                # green / teal
    "Consistency-driven": "#F59E0B",      # amber / orange
    "Unclassified": "#9CA3AF",
}

PROFILE_PALETTES = {
    "Directionality-driven": [
        "#1E3A8A", "#2563EB", "#3B82F6", "#4F46E5",
        "#6366F1", "#7C3AED", "#8B5CF6", "#A78BFA",
    ],
    "Balanced": [
        "#064E3B", "#047857", "#059669", "#10B981",
        "#14B8A6", "#2DD4BF", "#6EE7B7", "#99F6E4",
    ],
    "Consistency-driven": [
        "#92400E", "#B45309", "#D97706", "#F59E0B",
        "#F97316", "#FB923C", "#FCD34D", "#FDE68A",
    ],
    "Unclassified": ["#9CA3AF"],
}



def classify_profiles(summary: pd.DataFrame, rank_gap: float) -> pd.DataFrame:
    """Assign diagnostic profile based on Directionality vs Consistency rank gap."""
    out = summary.copy()
    out["directionality_advantage_rank"] = (
        out["Consistency_rank"] - out["Directionality_rank"]
    )

    def _profile(x: float) -> str:
        if pd.isna(x):
            return "Unclassified"
        if x >= rank_gap:
            return "Directionality-driven"
        if x <= -rank_gap:
            return "Consistency-driven"
        return "Balanced"

    out["dc_profile"] = out["directionality_advantage_rank"].apply(_profile)
    out["abs_family_rank_gap"] = out["directionality_advantage_rank"].abs()
    return out


def add_rank_improvement(summary: pd.DataFrame, first_label: str, last_label: str) -> pd.DataFrame:
    out = summary.copy()
    out["rank_improvement"] = out[first_label] - out[last_label]
    out["abs_rank_change"] = out["rank_improvement"].abs()
    return out


def select_profile_methods(summary: pd.DataFrame, profile: str, max_methods: int | None) -> List[str]:
    """Select methods to plot for a profile.

    If max_methods is None, all methods in the profile are used.
    Otherwise, methods with larger absolute rank change are prioritized, then
    better final consensus rank.
    """
    sub = summary.loc[summary["dc_profile"] == profile].copy()
    if sub.empty:
        return []
    sub = sub.sort_values(
        ["abs_rank_change", "dc_consensus_rank", "method"],
        ascending=[False, True, True],
    )
    if max_methods is not None and max_methods > 0:
        sub = sub.head(max_methods)
    return sub["method"].tolist()


def simple_label_positions(y_values: List[float], min_gap: float = 0.35) -> List[float]:
    """Slightly separate labels with similar y coordinates."""
    if not y_values:
        return []
    indexed = sorted(enumerate(y_values), key=lambda x: x[1])
    adjusted = [None] * len(y_values)
    last = None
    for idx, y in indexed:
        y_adj = float(y)
        if last is not None and y_adj - last < min_gap:
            y_adj = last + min_gap
        adjusted[idx] = y_adj
        last = y_adj
    return adjusted


def plot_profile_trajectory(
    details: pd.DataFrame,
    summary: pd.DataFrame,
    profile: str,
    outdir: Path,
    max_methods_per_panel: int | None = None,
):
    """Plot rank trajectories for one profile with direct labels at line ends."""
    plt = setup_matplotlib()
    methods = select_profile_methods(summary, profile, max_methods_per_panel)
    if not methods:
        return

    d = details.loc[details["method"].isin(methods)].copy()
    d["x"] = d["w_directionality"]

    # Stable method order: better consensus rank first.
    ordered_methods = (
        summary.loc[summary["method"].isin(methods)]
        .sort_values(["dc_consensus_rank", "method"])["method"]
        .tolist()
    )

    palette = PROFILE_PALETTES.get(profile, PROFILE_PALETTES["Unclassified"])
    marker_list = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]
    linestyle = "-"  # keep profile information in color family rather than line style

    fig_h = max(4.2, 0.35 * len(ordered_methods) + 2.8)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))

    end_labels = []
    for i, method in enumerate(ordered_methods):
        sub = d.loc[d["method"] == method].sort_values("x")
        if sub.empty:
            continue
        color = palette[i % len(palette)]
        marker = marker_list[i % len(marker_list)]
        ax.plot(
            sub["x"],
            sub["accuracy_rank"],
            marker=marker,
            linewidth=1.8,
            markersize=5,
            color=color,
            linestyle=linestyle,
        )
        last_x = sub["x"].iloc[-1]
        last_y = sub["accuracy_rank"].iloc[-1]
        end_labels.append((method, last_x, last_y, color))

    # Direct labels at the right end.
    label_y = simple_label_positions([x[2] for x in end_labels], min_gap=0.35)
    for (method, last_x, last_y, color), y_lab in zip(end_labels, label_y):
        ax.plot([last_x, last_x + 0.012], [last_y, y_lab], color=color, linewidth=0.8)
        ax.text(
            last_x + 0.018,
            y_lab,
            method,
            color=color,
            fontsize=9,
            va="center",
            ha="left",
        )

    ax.invert_yaxis()
    ax.set_xlim(d["x"].min() - 0.02, d["x"].max() + 0.16)
    ax.set_xlabel("Directionality weight")
    ax.set_ylabel("Accuracy rank")
    ax.set_title(f"Rank trajectory: {profile}")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    safe = profile.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, outdir / f"fig1_rank_trajectory_{safe}")


def plot_all_profile_trajectories(
    details: pd.DataFrame,
    summary: pd.DataFrame,
    outdir: Path,
    max_methods_per_panel: int | None = None,
):
    for profile in PROFILE_ORDER:
        plot_profile_trajectory(details, summary, profile, outdir, max_methods_per_panel)


def plot_advantage_vs_improvement(df: pd.DataFrame, outdir: Path):
    """Scatter plot showing mechanism check."""
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    profile_color = PROFILE_BASE_COLORS
    for profile, sub in df.groupby("dc_profile"):
        ax.scatter(
            sub["directionality_advantage_rank"],
            sub["rank_improvement"],
            s=50,
            label=profile,
            color=profile_color.get(profile, "#9CA3AF"),
            alpha=0.9,
        )
        for _, row in sub.iterrows():
            ax.text(
                row["directionality_advantage_rank"] + 0.15,
                row["rank_improvement"] + 0.15,
                row["method"],
                fontsize=6.5,
                color=profile_color.get(profile, "#9CA3AF"),
            )

    ax.axhline(0, linewidth=0.8, linestyle="--", color="black", alpha=0.5)
    ax.axvline(0, linewidth=0.8, linestyle="--", color="black", alpha=0.5)
    ax.set_xlabel("Directionality advantage = Consistency rank - Directionality rank")
    ax.set_ylabel("Rank improvement = rank(D min) - rank(D max)")
    ax.set_title("Directionality advantage predicts rank improvement")
    ax.legend(frameon=False, loc="best")
    save_figure(fig, outdir / "fig2_directionality_advantage_vs_rank_improvement")


def plot_family_score_bar(
    df: pd.DataFrame,
    score_col: str,
    rank_col: str,
    title: str,
    out_prefix: Path,
):
    """Plot family scores sorted from high to low.

    The original family score can be negative because it is based on
    z-score-like aggregation. For visualization only, this function min-max
    scales the score to [0, 1]:

        plot_score = (raw_score - min(raw_score)) / (max(raw_score) - min(raw_score) + 1e-8)

    This does NOT change sorting, ranks, or any downstream calculation.
    Methods are still ordered by the original family score.
    """
    plt = setup_matplotlib()

    d = df[["method", score_col, rank_col, "dc_profile"]].copy()
    d = d.loc[d[score_col].notna()].copy()

    # Sort by original score, not scaled score.
    d = d.sort_values(score_col, ascending=False).reset_index(drop=True)

    # Min-max scaling for visualization only.
    raw_score = pd.to_numeric(d[score_col], errors="coerce")
    raw_min = raw_score.min(skipna=True)
    raw_max = raw_score.max(skipna=True)
    d["plot_score"] = (raw_score - raw_min) / (raw_max - raw_min + 1e-8)

    # Profile colors are only used as an annotation cue; the ranking itself is
    # determined solely by the original family score.
    profile_color = PROFILE_BASE_COLORS
    colors = [profile_color.get(x, "#9CA3AF") for x in d["dc_profile"]]

    fig_h = max(5.5, 0.34 * len(d) + 1.8)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))

    y = np.arange(len(d))
    ax.barh(y, d["plot_score"], color=colors, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(d["method"])
    ax.invert_yaxis()
    ax.set_xlabel(f"Min-max scaled {score_col} family score")
    ax.set_title(title + " (scaled for visualization)")
    ax.set_xlim(0, 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    # Annotate scaled score, rank, and raw score for transparency.
    for i, row in d.iterrows():
        label = (
            f"{row['plot_score']:.2f}  "
            f"(rank {row[rank_col]:.0f}, raw {row[score_col]:.2f})"
        )
        ax.text(
            row["plot_score"] + 0.015,
            i,
            label,
            va="center",
            ha="left",
            fontsize=8,
        )

    # Add profile legend.
    handles = []
    labels = []
    for profile in PROFILE_ORDER:
        if (d["dc_profile"] == profile).any():
            handles.append(
                plt.Line2D(
                    [0], [0], marker="s", color="w",
                    markerfacecolor=profile_color[profile], markersize=8,
                )
            )
            labels.append(profile)
    if handles:
        ax.legend(handles, labels, frameon=False, loc="lower right")

    # Save the plotted values for exact reproducibility.
    plot_table = d[
        ["method", "dc_profile", score_col, "plot_score", rank_col]
    ].rename(
        columns={
            score_col: f"raw_{score_col}_score",
            rank_col: f"{score_col}_rank",
        }
    )
    plot_table.to_csv(out_prefix.with_suffix(".plot_values.csv"), index=False)

    save_figure(fig, out_prefix)


def plot_family_score_bars(df: pd.DataFrame, outdir: Path):
    """Plot Directionality and Consistency scores separately."""
    plot_family_score_bar(
        df=df,
        score_col="Directionality",
        rank_col="Directionality_rank",
        title="Directionality family score by method",
        out_prefix=outdir / "fig3a_directionality_score_bar",
    )
    plot_family_score_bar(
        df=df,
        score_col="Consistency",
        rank_col="Consistency_rank",
        title="Consistency family score by method",
        out_prefix=outdir / "fig3b_consistency_score_bar",
    )


def plot_profile_counts(df: pd.DataFrame, outdir: Path):
    plt = setup_matplotlib()
    counts = df["dc_profile"].value_counts().reindex(PROFILE_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    colors = [PROFILE_BASE_COLORS.get(profile, "#9CA3AF") for profile in counts.index]
    ax.bar(counts.index, counts.values, color=colors, alpha=0.9)
    ax.set_ylabel("Number of methods")
    ax.set_title("Diagnostic profile counts")
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, outdir / "fig4_profile_counts")


def main():
    parser = argparse.ArgumentParser(
        description="Grouped trajectory and Directionality/Consistency heatmap diagnostics."
    )
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_ANALYSIS_DIR / "02_direction_weight_trajectory_grouped"))
    parser.add_argument("--exclude_methods", default=";".join(DEFAULT_EXCLUDE_METHODS))
    parser.add_argument("--include_gt_in_directionality", action="store_true")
    parser.add_argument("--weight_grid", default=None, help="Example: 0.5:0.5,0.6:0.4,0.9:0.1")
    parser.add_argument("--profile_rank_gap", type=float, default=5.0)
    parser.add_argument(
        "--max_methods_per_panel",
        type=int,
        default=0,
        help="Max methods per profile panel. 0 means plot all methods in that profile.",
    )
    args = parser.parse_args()

    outdir = ensure_dir(args.output_dir)
    figdir = ensure_dir(outdir / "figures")

    exclude_methods = [x for x in args.exclude_methods.split(";") if x]
    weight_grid = parse_weight_grid(args.weight_grid)
    first_label = weight_label(*weight_grid[0])
    last_label = weight_label(*weight_grid[-1])
    max_methods = None if args.max_methods_per_panel == 0 else args.max_methods_per_panel

    metric_df = read_metric_scores(args.results_dir)
    fam = recompute_family_scores(metric_df, args.include_gt_in_directionality, exclude_methods)
    details, summary = compute_dc_weight_sweep(fam, weight_grid)
    summary = add_primary_ranking_columns(summary)

    rank_wide = details.pivot(index="method", columns="weight_label", values="accuracy_rank").reset_index()
    score_wide = details.pivot(index="method", columns="weight_label", values="accuracy_score").reset_index()

    summary = summary.merge(rank_wide, on="method", how="left")
    summary = summary.merge(score_wide.add_prefix("score_"), left_on="method", right_on="score_method", how="left")
    if "score_method" in summary.columns:
        summary = summary.drop(columns=["score_method"])

    summary = classify_profiles(summary, args.profile_rank_gap)
    summary = add_rank_improvement(summary, first_label, last_label)

    direction_driven = summary[summary["dc_profile"] == "Directionality-driven"].copy()
    consistency_driven = summary[summary["dc_profile"] == "Consistency-driven"].copy()
    balanced = summary[summary["dc_profile"] == "Balanced"].copy()

    corr_rows = []
    for method in ["spearman", "kendall", "pearson"]:
        rho, p, n = corr_one(summary["directionality_advantage_rank"], summary["rank_improvement"], method=method)
        corr_rows.append({
            "comparison": "directionality_advantage_rank vs rank_improvement",
            "method": method,
            "rho": rho,
            "p_value": p,
            "n_methods": n,
        })
    corr_df = pd.DataFrame(corr_rows)

    save_csv_readable(summary.sort_values("dc_consensus_rank"), outdir / "01_rank_by_weight_wide.csv")
    save_csv_readable(
        summary.sort_values("directionality_advantage_rank", ascending=False),
        outdir / "02_directionality_advantage_rank_change.csv",
    )
    save_csv_readable(
        summary[["method", "dc_profile", "Directionality_rank", "Consistency_rank",
                 "directionality_advantage_rank", "rank_improvement",
                 "dc_consensus_rank", "dc_mean_score"]]
        .sort_values(["dc_profile", "dc_consensus_rank", "method"]),
        outdir / "03_profile_assignment.csv",
    )
    save_csv_readable(
        direction_driven.sort_values("rank_improvement", ascending=False),
        outdir / "04_directionality_driven_tools.csv",
    )
    save_csv_readable(
        consistency_driven.sort_values("rank_improvement"),
        outdir / "05_consistency_driven_tools.csv",
    )
    save_csv_readable(
        balanced.sort_values("dc_consensus_rank"),
        outdir / "06_balanced_tools.csv",
    )
    save_csv_readable(corr_df, outdir / "07_rank_change_correlations.csv")

    # Figures
    plot_all_profile_trajectories(details, summary, figdir, max_methods_per_panel=max_methods)
    plot_advantage_vs_improvement(summary, figdir)
    plot_family_score_bars(summary, figdir)
    plot_profile_counts(summary, figdir)

    sp = corr_df[corr_df["method"] == "spearman"].iloc[0]
    counts = summary["dc_profile"].value_counts().reindex(PROFILE_ORDER).fillna(0).astype(int)

    text = f"""# Grouped Directionality/Consistency weight trajectory validation

- Results directory: `{args.results_dir}`
- Excluded methods: {exclude_methods}
- Directionality includes groundtruth_correlation: {args.include_gt_in_directionality}
- Weight grid: {', '.join(weight_label(*w) for w in weight_grid)}
- Profile rank gap: {args.profile_rank_gap}

## Profile definition

`directionality_advantage_rank = Consistency_rank - Directionality_rank`

Because lower rank is better:

- Directionality-driven: `directionality_advantage_rank >= {args.profile_rank_gap}`
- Consistency-driven: `directionality_advantage_rank <= -{args.profile_rank_gap}`
- Balanced: otherwise

## Profile counts

- Directionality-driven: {counts.get('Directionality-driven', 0)}
- Balanced: {counts.get('Balanced', 0)}
- Consistency-driven: {counts.get('Consistency-driven', 0)}

## Mechanism check

`rank_improvement = rank({first_label}) - rank({last_label})`

Spearman correlation between directionality advantage and rank improvement:

- rho = {sp['rho']:.3f}
- p = {sp['p_value']:.3g}
- n = {int(sp['n_methods'])}

A positive correlation means that methods with stronger Directionality relative to Consistency tend to move upward as Directionality weight increases.

## Output figures

- `fig1_rank_trajectory_directionality_driven`
- `fig1_rank_trajectory_balanced`
- `fig1_rank_trajectory_consistency_driven`
- `fig2_directionality_advantage_vs_rank_improvement`
- `fig3a_directionality_score_bar`
- `fig3b_consistency_score_bar`
- `fig4_profile_counts`
"""
    write_text(outdir / "validation_summary.md", text)

    print(f"Done. Results written to: {outdir}")


if __name__ == "__main__":
    main()
