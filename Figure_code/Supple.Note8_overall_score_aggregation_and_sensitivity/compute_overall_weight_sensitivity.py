#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity analysis for overall ranking weights.

This script performs a grid-search sensitivity analysis for the overall score:

    OverallRankScore = w_acc * R_accuracy
                     + w_sca * R_scalability
                     + w_sta * R_stability
                     + w_use * R_usability

where all R_* values are reversed ranks, so larger values are better.

The weight grid is constrained by:

    w_acc + w_sca + w_sta + w_use = 1

Default ranges:
    Accuracy:    0.50-0.70
    Scalability: 0.10-0.20
    Stability:   0.10-0.20
    Usability:   0.05-0.15

Default step:
    0.025

Default input directory:
    PlotData/Results/reversed_rank

Expected input files:
    accuracy_rank.csv
    scalability_rank.csv
    stability_rank.csv
    usability_rank.csv

Default output directory:
    PlotData/Results/reversed_rank/Results/Sensitivity

Default behavior:
    - Harmonize scRNAkinetics name variants.
        - Compute rank for every valid weight combination.
    - Summarize mean / median / best / worst rank across weight combinations.
    - Generate sensitivity plots.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("PlotData/Results/reversed_rank")
DEFAULT_OUTPUT_DIR = Path("PlotData/Results/reversed_rank/Results/Sensitivity")

DEFAULT_FILES = {
    "accuracy": "accuracy_rank.csv",
    "scalability": "scalability_rank.csv",
    "stability": "stability_rank.csv",
    "usability": "usability_rank.csv",
}

DEFAULT_EXCLUDE_METHODS = [
    "Region Velocity",
    "TopoVelo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search sensitivity analysis for overall ranking weights."
    )

    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--accuracy_csv", default=None)
    parser.add_argument("--scalability_csv", default=None)
    parser.add_argument("--stability_csv", default=None)
    parser.add_argument("--usability_csv", default=None)

    parser.add_argument("--method_col", default="method")
    parser.add_argument("--rank_col", default="reversed_rank")

    parser.add_argument("--acc_min", type=float, default=0.50)
    parser.add_argument("--acc_max", type=float, default=0.70)
    parser.add_argument("--sca_min", type=float, default=0.10)
    parser.add_argument("--sca_max", type=float, default=0.20)
    parser.add_argument("--sta_min", type=float, default=0.10)
    parser.add_argument("--sta_max", type=float, default=0.20)
    parser.add_argument("--use_min", type=float, default=0.05)
    parser.add_argument("--use_max", type=float, default=0.15)

    parser.add_argument(
        "--step",
        type=float,
        default=0.025,
        help="Grid step for all four weights.",
    )
    parser.add_argument(
        "--sum_tolerance",
        type=float,
        default=1e-8,
        help="Tolerance for checking weights sum to 1.",
    )

    parser.add_argument("--baseline_acc", type=float, default=0.60)
    parser.add_argument("--baseline_sca", type=float, default=0.15)
    parser.add_argument("--baseline_sta", type=float, default=0.15)
    parser.add_argument("--baseline_use", type=float, default=0.10)

    parser.add_argument(
        "--missing_policy",
        choices=["strict", "renormalize", "zero"],
        default="strict",
        help=(
            "strict: require all four reversed ranks. "
            "renormalize: use available ranks and renormalize available weights. "
            "zero: missing reversed rank contributes 0."
        ),
    )

    parser.add_argument(
        "--exclude_methods",
        default=";".join(DEFAULT_EXCLUDE_METHODS),
        help="Comma/semicolon-separated methods excluded from sensitivity analysis.",
    )

    parser.add_argument(
        "--top_n_plot",
        type=int,
        default=12,
        help="Number of top baseline methods to show in selected trajectory plot.",
    )

    parser.add_argument(
        "--cmap",
        default="viridis_r",
        help="Matplotlib colormap for rank heatmap. Default makes better ranks visually stronger.",
    )

    return parser.parse_args()


# -----------------------------
# Method-name handling
# -----------------------------

def normalize_method_name(x: object) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip())


def method_key(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_method_name(x).lower())


def canonical_method_name(x: object) -> str:
    name = normalize_method_name(x)
    key = method_key(name)

    if key in {"scrnakinetics", "scrnakinetic"}:
        return "scRNAkinetics"
    if key == "regionvelocity":
        return "Region Velocity"
    if key == "topovelo":
        return "TopoVelo"

    return name


def split_method_list(s: Optional[str]) -> list[str]:
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in re.split(r"[;,]", s) if x.strip()]


# -----------------------------
# Inputs
# -----------------------------

def resolve_input_paths(args: argparse.Namespace) -> Dict[str, Path]:
    input_dir = Path(args.input_dir)
    paths = {
        "accuracy": Path(args.accuracy_csv) if args.accuracy_csv else input_dir / DEFAULT_FILES["accuracy"],
        "scalability": Path(args.scalability_csv) if args.scalability_csv else input_dir / DEFAULT_FILES["scalability"],
        "stability": Path(args.stability_csv) if args.stability_csv else input_dir / DEFAULT_FILES["stability"],
        "usability": Path(args.usability_csv) if args.usability_csv else input_dir / DEFAULT_FILES["usability"],
    }

    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} input file not found: {path}")

    return paths


def load_component_table(
    path: Path,
    component: str,
    method_col: str,
    rank_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    if method_col not in df.columns:
        raise ValueError(
            f"`{method_col}` column not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    if rank_col not in df.columns:
        raise ValueError(
            f"`{rank_col}` column not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    raw = pd.DataFrame()
    raw["raw_method"] = df[method_col].map(normalize_method_name)
    raw["method"] = raw["raw_method"].map(canonical_method_name)
    raw[f"R_{component}"] = pd.to_numeric(df[rank_col], errors="coerce")
    raw = raw.loc[raw["method"] != ""].copy()

    dup = (
        raw.groupby("method", as_index=False)
        .agg(
            n_rows=("raw_method", "size"),
            raw_method_names=("raw_method", lambda x: "; ".join(sorted(set(map(str, x))))),
            **{f"R_{component}_values": (f"R_{component}", lambda x: "; ".join(map(str, x.tolist())))},
        )
    )
    dup = dup.loc[dup["n_rows"] > 1].copy()
    dup.insert(0, "component", component)

    collapsed = (
        raw.groupby("method", as_index=False)
        .agg(**{f"R_{component}": (f"R_{component}", "mean")})
    )

    return collapsed, dup


def merge_component_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for _, table in tables.items():
        if merged is None:
            merged = table.copy()
        else:
            merged = merged.merge(table, on="method", how="outer")

    if merged is None:
        raise ValueError("No input tables were loaded.")

    return merged


def apply_exclusion(
    merged: pd.DataFrame,
    exclude_methods: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not exclude_methods:
        return merged.copy(), merged.iloc[0:0].copy()

    exclude_keys = {method_key(canonical_method_name(x)) for x in exclude_methods}
    mask = merged["method"].map(method_key).isin(exclude_keys)

    excluded = merged.loc[mask].copy()
    excluded["excluded_reason"] = "excluded_from_sensitivity"
    kept = merged.loc[~mask].copy()

    return kept, excluded


# -----------------------------
# Weight grid
# -----------------------------

def grid_values(min_v: float, max_v: float, step: float) -> list[float]:
    values = []
    x = min_v
    while x <= max_v + step / 2:
        values.append(round(float(x), 10))
        x += step
    return [v for v in values if v <= max_v + 1e-10]


def generate_weight_grid(args: argparse.Namespace) -> pd.DataFrame:
    acc_vals = grid_values(args.acc_min, args.acc_max, args.step)
    sca_vals = grid_values(args.sca_min, args.sca_max, args.step)
    sta_vals = grid_values(args.sta_min, args.sta_max, args.step)
    use_vals = grid_values(args.use_min, args.use_max, args.step)

    rows = []
    combo_id = 1
    for w_acc in acc_vals:
        for w_sca in sca_vals:
            for w_sta in sta_vals:
                for w_use in use_vals:
                    total = w_acc + w_sca + w_sta + w_use
                    if abs(total - 1.0) <= args.sum_tolerance:
                        rows.append(
                            {
                                "combo_id": f"W{combo_id:04d}",
                                "w_accuracy": w_acc,
                                "w_scalability": w_sca,
                                "w_stability": w_sta,
                                "w_usability": w_use,
                                "weight_sum": total,
                                "weight_label": (
                                    f"A{w_acc:.3f}_Sca{w_sca:.3f}_"
                                    f"Sta{w_sta:.3f}_Use{w_use:.3f}"
                                ),
                            }
                        )
                        combo_id += 1

    grid = pd.DataFrame(rows)
    if grid.empty:
        raise ValueError(
            "No valid weight combinations found. "
            "Check ranges, step size, and sum_tolerance."
        )

    grid = grid.sort_values(
        ["w_accuracy", "w_scalability", "w_stability", "w_usability"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    return grid


def add_baseline_flag(grid: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = grid.copy()
    target = np.array([args.baseline_acc, args.baseline_sca, args.baseline_sta, args.baseline_use], dtype=float)

    weights = out[["w_accuracy", "w_scalability", "w_stability", "w_usability"]].to_numpy(dtype=float)
    dist = np.sqrt(((weights - target) ** 2).sum(axis=1))
    best_idx = int(np.argmin(dist))

    out["is_baseline"] = False
    out.loc[best_idx, "is_baseline"] = True
    out["distance_to_requested_baseline"] = dist

    return out


# -----------------------------
# Rank calculation
# -----------------------------

def compute_score_for_combo(
    merged: pd.DataFrame,
    weights: dict[str, float],
    missing_policy: str,
) -> pd.Series:
    cols = {
        "accuracy": "R_accuracy",
        "scalability": "R_scalability",
        "stability": "R_stability",
        "usability": "R_usability",
    }

    for col in cols.values():
        if col not in merged.columns:
            raise ValueError(f"Required component column not found: {col}")

    if missing_policy == "strict":
        score = (
            weights["accuracy"] * merged["R_accuracy"]
            + weights["scalability"] * merged["R_scalability"]
            + weights["stability"] * merged["R_stability"]
            + weights["usability"] * merged["R_usability"]
        )
        required = merged[list(cols.values())].notna().all(axis=1)
        return score.where(required, np.nan)

    if missing_policy == "renormalize":
        numerator = pd.Series(0.0, index=merged.index, dtype=float)
        denom = pd.Series(0.0, index=merged.index, dtype=float)
        for comp, col in cols.items():
            available = merged[col].notna()
            numerator += weights[comp] * merged[col].fillna(0)
            denom += weights[comp] * available.astype(float)
        return numerator / denom.replace(0, np.nan)

    if missing_policy == "zero":
        return (
            weights["accuracy"] * merged["R_accuracy"].fillna(0)
            + weights["scalability"] * merged["R_scalability"].fillna(0)
            + weights["stability"] * merged["R_stability"].fillna(0)
            + weights["usability"] * merged["R_usability"].fillna(0)
        )

    raise ValueError(f"Unknown missing_policy: {missing_policy}")


def compute_ranks_for_grid(
    merged: pd.DataFrame,
    grid: pd.DataFrame,
    missing_policy: str,
) -> pd.DataFrame:
    rows = []

    for _, w in grid.iterrows():
        weights = {
            "accuracy": float(w["w_accuracy"]),
            "scalability": float(w["w_scalability"]),
            "stability": float(w["w_stability"]),
            "usability": float(w["w_usability"]),
        }
        score = compute_score_for_combo(merged, weights, missing_policy=missing_policy)

        tmp = merged[["method", "R_accuracy", "R_scalability", "R_stability", "R_usability"]].copy()
        tmp["overall_score"] = score
        tmp["overall_rank"] = tmp["overall_score"].rank(
            ascending=False,
            method="average",
            na_option="bottom",
        )

        valid = tmp.loc[tmp["overall_score"].notna()].copy()
        missing = tmp.loc[tmp["overall_score"].isna()].copy()

        valid = valid.sort_values(["overall_score", "method"], ascending=[False, True]).reset_index(drop=True)
        valid["overall_order_rank"] = np.arange(1, len(valid) + 1, dtype=int)

        if not missing.empty:
            missing = missing.sort_values("method").reset_index(drop=True)
            missing["overall_order_rank"] = np.arange(
                len(valid) + 1,
                len(valid) + len(missing) + 1,
                dtype=int,
            )

        tmp2 = pd.concat([valid, missing], ignore_index=True, sort=False)

        for col in [
            "combo_id", "weight_label", "w_accuracy", "w_scalability",
            "w_stability", "w_usability", "is_baseline",
        ]:
            tmp2[col] = w[col]

        rows.append(tmp2)

    out = pd.concat(rows, ignore_index=True, sort=False)
    return out


# -----------------------------
# Summaries
# -----------------------------

def summarize_sensitivity(rank_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = rank_long.loc[rank_long["is_baseline"]].copy()
    baseline_cols = baseline[
        ["method", "overall_rank", "overall_order_rank", "overall_score", "combo_id", "weight_label"]
    ].rename(
        columns={
            "overall_rank": "baseline_overall_rank",
            "overall_order_rank": "baseline_overall_order_rank",
            "overall_score": "baseline_overall_score",
            "combo_id": "baseline_combo_id",
            "weight_label": "baseline_weight_label",
        }
    )

    summary = (
        rank_long.groupby("method", as_index=False)
        .agg(
            mean_rank=("overall_rank", "mean"),
            median_rank=("overall_rank", "median"),
            best_rank=("overall_rank", "min"),
            worst_rank=("overall_rank", "max"),
            sd_rank=("overall_rank", "std"),
            iqr_rank=("overall_rank", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            mean_order_rank=("overall_order_rank", "mean"),
            best_order_rank=("overall_order_rank", "min"),
            worst_order_rank=("overall_order_rank", "max"),
            mean_score=("overall_score", "mean"),
            sd_score=("overall_score", "std"),
            n_weight_combinations=("combo_id", "nunique"),
            top1_count=("overall_order_rank", lambda x: int((x == 1).sum())),
            top3_count=("overall_order_rank", lambda x: int((x <= 3).sum())),
            top5_count=("overall_order_rank", lambda x: int((x <= 5).sum())),
        )
    )

    n_combos = rank_long["combo_id"].nunique()
    summary["top1_fraction"] = summary["top1_count"] / n_combos
    summary["top3_fraction"] = summary["top3_count"] / n_combos
    summary["top5_fraction"] = summary["top5_count"] / n_combos

    summary = summary.merge(baseline_cols, on="method", how="left")
    summary["rank_range"] = summary["worst_rank"] - summary["best_rank"]

    summary = summary.sort_values(
        ["mean_rank", "baseline_overall_order_rank", "method"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    summary["sensitivity_mean_rank_order"] = np.arange(1, len(summary) + 1, dtype=int)

    top_freq = summary[
        [
            "method", "top1_count", "top1_fraction",
            "top3_count", "top3_fraction",
            "top5_count", "top5_fraction",
            "mean_rank", "best_rank", "worst_rank",
        ]
    ].sort_values(["top1_count", "top3_count", "mean_rank"], ascending=[False, False, True])

    extreme_rows = []
    for method, sub in rank_long.groupby("method"):
        valid = sub.loc[sub["overall_rank"].notna()].copy()
        if valid.empty:
            continue
        best = valid.loc[valid["overall_rank"].idxmin()]
        worst = valid.loc[valid["overall_rank"].idxmax()]
        extreme_rows.append(
            {
                "method": method,
                "best_rank": best["overall_rank"],
                "best_combo_id": best["combo_id"],
                "best_weight_label": best["weight_label"],
                "best_w_accuracy": best["w_accuracy"],
                "best_w_scalability": best["w_scalability"],
                "best_w_stability": best["w_stability"],
                "best_w_usability": best["w_usability"],
                "worst_rank": worst["overall_rank"],
                "worst_combo_id": worst["combo_id"],
                "worst_weight_label": worst["weight_label"],
                "worst_w_accuracy": worst["w_accuracy"],
                "worst_w_scalability": worst["w_scalability"],
                "worst_w_stability": worst["w_stability"],
                "worst_w_usability": worst["w_usability"],
                "rank_range": worst["overall_rank"] - best["overall_rank"],
            }
        )

    extremes = pd.DataFrame(extreme_rows).sort_values(["rank_range", "method"], ascending=[False, True])

    return summary, top_freq, extremes


def compute_rank_shift_from_baseline(rank_long: pd.DataFrame) -> pd.DataFrame:
    baseline = rank_long.loc[rank_long["is_baseline"], ["method", "overall_order_rank"]].rename(
        columns={"overall_order_rank": "baseline_order_rank"}
    )

    out = rank_long.merge(baseline, on="method", how="left")
    out["rank_shift_vs_baseline"] = out["baseline_order_rank"] - out["overall_order_rank"]
    return out


# -----------------------------
# Plots
# -----------------------------

def save_figure(fig, path_prefix: Path) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_rank_distribution(summary: pd.DataFrame, rank_long: pd.DataFrame, figdir: Path) -> None:
    ordered_methods = summary.sort_values("mean_rank")["method"].tolist()
    data = [
        rank_long.loc[rank_long["method"] == m, "overall_order_rank"].dropna().to_numpy()
        for m in ordered_methods
    ]

    fig_h = max(5.5, 0.33 * len(ordered_methods) + 1.5)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))

    ax.boxplot(
        data,
        vert=False,
        tick_labels=ordered_methods,
        showfliers=True,
        flierprops={
            "marker": ".",
            "markersize": 3,
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "linestyle": "none",
        },
    )

    ax.invert_yaxis()
    ax.set_xlabel("Overall rank across weight combinations")
    ax.set_title("Rank distribution under weight space sensitivity analysis")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    save_figure(fig, figdir / "fig1_rank_distribution_boxplot")

def plot_rank_heatmap(summary: pd.DataFrame, rank_long: pd.DataFrame, grid: pd.DataFrame, figdir: Path, cmap: str) -> None:
    method_order = summary.sort_values("mean_rank")["method"].tolist()
    combo_order = grid.sort_values(
        ["w_accuracy", "w_scalability", "w_stability", "w_usability"]
    )["combo_id"].tolist()

    mat = (
        rank_long.pivot_table(index="method", columns="combo_id", values="overall_order_rank", aggfunc="first")
        .reindex(index=method_order, columns=combo_order)
    )

    values = mat.to_numpy(dtype=float)

    fig_w = max(9.0, 0.06 * len(combo_order) + 3.0)
    fig_h = max(5.5, 0.33 * len(method_order) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(values, aspect="auto", cmap=cmap)

    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(method_order)

    n_combo = len(combo_order)
    tick_step = max(1, math.ceil(n_combo / 12))
    xticks = list(range(0, n_combo, tick_step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([combo_order[i] for i in xticks], rotation=45, ha="right", fontsize=7)

    ax.set_xlabel("Weight combination")
    ax.set_title("Overall rank for every weight combination")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Overall rank; lower is better")

    save_figure(fig, figdir / "fig2_rank_heatmap_all_weight_combinations")


def plot_top_frequency(top_freq: pd.DataFrame, figdir: Path, top_n: int = 15) -> None:
    d = top_freq.head(top_n).copy()
    fig_h = max(4.8, 0.34 * len(d) + 1.2)
    fig, ax = plt.subplots(figsize=(7.8, fig_h))
    y = np.arange(len(d))
    ax.barh(y, d["top3_fraction"])
    ax.set_yticks(y)
    ax.set_yticklabels(d["method"])
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of weight combinations ranked in top 3")
    ax.set_title("Top-3 frequency under weight sensitivity")
    ax.set_xlim(0, 1)
    save_figure(fig, figdir / "fig3_top3_frequency")


def plot_mean_rank_by_accuracy_weight(rank_long: pd.DataFrame, summary: pd.DataFrame, figdir: Path, top_n: int) -> None:
    top_methods = summary.sort_values("baseline_overall_order_rank").head(top_n)["method"].tolist()
    d = (
        rank_long.loc[rank_long["method"].isin(top_methods)]
        .groupby(["method", "w_accuracy"], as_index=False)
        .agg(mean_rank=("overall_order_rank", "mean"))
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for method, sub in d.groupby("method"):
        sub = sub.sort_values("w_accuracy")
        ax.plot(sub["w_accuracy"], sub["mean_rank"], marker="o", linewidth=1.5, label=method)

    ax.invert_yaxis()
    ax.set_xlabel("Accuracy weight")
    ax.set_ylabel("Mean overall rank across combinations with this accuracy weight")
    ax.set_title("Rank sensitivity summarized by Accuracy weight")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    save_figure(fig, figdir / "fig4_mean_rank_by_accuracy_weight_top_methods")


def plot_rank_range_bar(summary: pd.DataFrame, figdir: Path, top_n: int = 20) -> None:
    d = summary.sort_values("rank_range", ascending=False).head(top_n).copy()
    fig_h = max(4.8, 0.34 * len(d) + 1.2)
    fig, ax = plt.subplots(figsize=(7.8, fig_h))
    y = np.arange(len(d))
    ax.barh(y, d["rank_range"])
    ax.set_yticks(y)
    ax.set_yticklabels(d["method"])
    ax.invert_yaxis()
    ax.set_xlabel("Rank range across weight combinations")
    ax.set_title("Methods most sensitive to weight changes")
    save_figure(fig, figdir / "fig5_rank_range_most_sensitive_methods")


def plot_baseline_vs_mean_rank(summary: pd.DataFrame, figdir: Path) -> None:
    d = summary.dropna(subset=["baseline_overall_order_rank", "mean_rank"]).copy()

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(d["baseline_overall_order_rank"], d["mean_rank"], s=42)
    for _, row in d.iterrows():
        ax.text(
            row["baseline_overall_order_rank"] + 0.08,
            row["mean_rank"] + 0.08,
            row["method"],
            fontsize=6.5,
        )

    max_rank = max(d["baseline_overall_order_rank"].max(), d["mean_rank"].max())
    ax.plot([1, max_rank], [1, max_rank], linestyle="--", linewidth=0.8)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Baseline overall rank")
    ax.set_ylabel("Mean rank across all weight combinations")
    ax.set_title("Baseline rank vs sensitivity mean rank")
    save_figure(fig, figdir / "fig6_baseline_vs_sensitivity_mean_rank")


def make_plots(
    summary: pd.DataFrame,
    top_freq: pd.DataFrame,
    rank_long: pd.DataFrame,
    grid: pd.DataFrame,
    output_dir: Path,
    cmap: str,
    top_n_plot: int,
) -> None:
    figdir = output_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    plot_rank_distribution(summary, rank_long, figdir)
    plot_rank_heatmap(summary, rank_long, grid, figdir, cmap=cmap)
    plot_top_frequency(top_freq, figdir)
    plot_mean_rank_by_accuracy_weight(rank_long, summary, figdir, top_n=top_n_plot)
    plot_rank_range_bar(summary, figdir)
    plot_baseline_vs_mean_rank(summary, figdir)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = resolve_input_paths(args)

    tables = {}
    duplicate_reports = []
    for component, path in input_paths.items():
        table, dup = load_component_table(
            path=path,
            component=component,
            method_col=args.method_col,
            rank_col=args.rank_col,
        )
        tables[component] = table
        if not dup.empty:
            duplicate_reports.append(dup)

    merged_all = merge_component_tables(tables)

    exclude_methods = split_method_list(args.exclude_methods)
    merged, excluded = apply_exclusion(merged_all, exclude_methods=exclude_methods)

    grid = generate_weight_grid(args)
    grid = add_baseline_flag(grid, args)

    rank_long = compute_ranks_for_grid(
        merged=merged,
        grid=grid,
        missing_policy=args.missing_policy,
    )

    summary, top_freq, extremes = summarize_sensitivity(rank_long)
    rank_shift = compute_rank_shift_from_baseline(rank_long)

    grid.to_csv(output_dir / "01_weight_grid.csv", index=False)
    merged.to_csv(output_dir / "02_merged_reversed_ranks_used_for_sensitivity.csv", index=False)
    rank_long.to_csv(output_dir / "03_rank_by_weight_long.csv", index=False)

    rank_wide = (
        rank_long.pivot_table(index="method", columns="combo_id", values="overall_order_rank", aggfunc="first")
        .reset_index()
    )
    rank_wide.to_csv(output_dir / "04_rank_by_weight_wide.csv", index=False)

    summary.to_csv(output_dir / "05_sensitivity_summary_by_method.csv", index=False)
    top_freq.to_csv(output_dir / "06_top_method_frequency.csv", index=False)
    extremes.to_csv(output_dir / "07_extreme_rank_combinations_by_method.csv", index=False)
    rank_shift.to_csv(output_dir / "08_rank_shift_from_baseline.csv", index=False)
    excluded.to_csv(output_dir / "09_excluded_methods_from_sensitivity.csv", index=False)

    if duplicate_reports:
        pd.concat(duplicate_reports, ignore_index=True).to_csv(
            output_dir / "10_duplicate_method_names_collapsed.csv",
            index=False,
        )
    else:
        pd.DataFrame(columns=["component", "method", "n_rows", "raw_method_names"]).to_csv(
            output_dir / "10_duplicate_method_names_collapsed.csv",
            index=False,
        )

    make_plots(
        summary=summary,
        top_freq=top_freq,
        rank_long=rank_long,
        grid=grid,
        output_dir=output_dir,
        cmap=args.cmap,
        top_n_plot=args.top_n_plot,
    )

    qc_rows = []
    for comp, path in input_paths.items():
        qc_rows.append({"section": "input", "item": f"{comp}_csv", "value": str(path)})
        qc_rows.append({"section": "input", "item": f"{comp}_n_methods", "value": len(tables[comp])})

    qc_rows.extend(
        [
            {"section": "config", "item": "missing_policy", "value": args.missing_policy},
            {"section": "config", "item": "exclude_methods", "value": ";".join(exclude_methods)},
            {"section": "config", "item": "step", "value": args.step},
            {"section": "config", "item": "accuracy_range", "value": f"{args.acc_min}-{args.acc_max}"},
            {"section": "config", "item": "scalability_range", "value": f"{args.sca_min}-{args.sca_max}"},
            {"section": "config", "item": "stability_range", "value": f"{args.sta_min}-{args.sta_max}"},
            {"section": "config", "item": "usability_range", "value": f"{args.use_min}-{args.use_max}"},
            {"section": "config", "item": "baseline_weights", "value": f"{args.baseline_acc},{args.baseline_sca},{args.baseline_sta},{args.baseline_use}"},
            {"section": "output", "item": "n_methods_before_exclusion", "value": len(merged_all)},
            {"section": "output", "item": "n_methods_excluded", "value": len(excluded)},
            {"section": "output", "item": "excluded_methods_found", "value": ";".join(excluded["method"].tolist()) if not excluded.empty else ""},
            {"section": "output", "item": "n_methods_used", "value": len(merged)},
            {"section": "output", "item": "n_weight_combinations", "value": len(grid)},
            {"section": "output", "item": "baseline_combo_id", "value": grid.loc[grid["is_baseline"], "combo_id"].iloc[0]},
            {"section": "output", "item": "baseline_weight_label", "value": grid.loc[grid["is_baseline"], "weight_label"].iloc[0]},
        ]
    )
    pd.DataFrame(qc_rows).to_csv(output_dir / "sensitivity_qc_summary.csv", index=False)

    print("Done.")
    print(f"Output directory: {output_dir}")
    print(f"Weight combinations: {len(grid)}")
    print(f"Main summary: {output_dir / '05_sensitivity_summary_by_method.csv'}")
    print(f"Figures: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
