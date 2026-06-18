#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis 04: Pareto front / dominated-tool validation.

Purpose
-------
This script evaluates whether D-only top-ranked tools include methods that are
inferior in the two-dimensional Directionality x Consistency space.

A method is Pareto-dominated if another method has:
    Directionality >= it and Consistency >= it,
with at least one strict improvement.

This analysis is useful because it does not rely on an arbitrary ideal-point
threshold. It asks whether top-ranked tools are defensible under a two-objective
view of velocity quality.

Default validation parameters match the current discussion:
    Directionality excludes groundtruth_correlation
    low-consistency = bottom 30% by Consistency rank
    top-k = 5 and 10

Outputs
-------
- 01_method_scores_ranks_pareto.csv
- 02_topk_profile_pareto_summary.csv
- 03_topk_method_lists_with_pareto.csv
- 04_replacement_analysis.csv
- 05_pareto_scores_all_methods.csv
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
    parse_weight_grid,
    minmax01,
    low_rank_flags,
    save_csv_readable,
    setup_matplotlib,
    save_figure,
    write_text,
)

SCHEME_INFO = {
    "D-only": {"rank_col": "D_only_rank"},
    "D+C consensus": {"rank_col": "D_plus_C_rank"},
    "C-only": {"rank_col": "C_only_rank"},
}


def compute_pareto(scores: pd.DataFrame) -> pd.DataFrame:
    """Compute Pareto-dominated status and Pareto depth using scaled D/C scores."""
    df = scores.copy()
    df["D_scaled"] = minmax01(df["Directionality"])
    df["C_scaled"] = minmax01(df["Consistency"])
    methods = df["method"].tolist()
    D = df["D_scaled"].to_numpy(float)
    C = df["C_scaled"].to_numpy(float)

    dominated = []
    dominators = []
    for i, m in enumerate(methods):
        dom_by = []
        for j, mj in enumerate(methods):
            if i == j:
                continue
            if (D[j] >= D[i] - 1e-12) and (C[j] >= C[i] - 1e-12) and ((D[j] > D[i] + 1e-12) or (C[j] > C[i] + 1e-12)):
                dom_by.append(mj)
        dominated.append(len(dom_by) > 0)
        dominators.append("; ".join(dom_by))
    df["is_pareto_dominated"] = dominated
    df["is_pareto_front"] = ~df["is_pareto_dominated"]
    df["dominators"] = dominators

    # Non-dominated sorting / Pareto depth.
    remaining = set(methods)
    depth = {m: np.nan for m in methods}
    current_depth = 1
    while remaining:
        rem_idx = [methods.index(m) for m in remaining]
        front = []
        for i in rem_idx:
            dominated_by_remaining = False
            for j in rem_idx:
                if i == j:
                    continue
                if (D[j] >= D[i] - 1e-12) and (C[j] >= C[i] - 1e-12) and ((D[j] > D[i] + 1e-12) or (C[j] > C[i] + 1e-12)):
                    dominated_by_remaining = True
                    break
            if not dominated_by_remaining:
                front.append(methods[i])
        if not front:  # safety fallback
            front = list(remaining)
        for m in front:
            depth[m] = current_depth
            remaining.remove(m)
        current_depth += 1
    df["pareto_depth"] = df["method"].map(depth).astype(int)

    # Weighted ideal-point distances for optional auxiliary diagnostics.
    for w_d, w_c in [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]:
        label = f"ideal_distance_D{w_d:g}_C{w_c:g}"
        df[label] = np.sqrt(w_d * (1 - df["D_scaled"]) ** 2 + w_c * (1 - df["C_scaled"]) ** 2)
        df[f"ideal_rank_D{w_d:g}_C{w_c:g}"] = df[label].rank(ascending=True, method="average")
    return df


def topk_profile_pareto(scores: pd.DataFrame, topks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    list_rows = []
    for k in topks:
        for scheme, info in SCHEME_INFO.items():
            rank_col = info["rank_col"]
            top = scores.sort_values(rank_col).head(k).copy()
            summary_rows.append({
                "scheme": scheme,
                "top_k": k,
                "n_methods": len(top),
                "mean_directionality_rank": top["Directionality_rank"].mean(),
                "mean_consistency_rank": top["Consistency_rank"].mean(),
                "n_low_consistency": int(top["low_consistency"].sum()),
                "n_pareto_front": int(top["is_pareto_front"].sum()),
                "n_pareto_dominated": int(top["is_pareto_dominated"].sum()),
                "mean_pareto_depth": top["pareto_depth"].mean(),
                "mean_ideal_distance_D0.6_C0.4": top["ideal_distance_D0.6_C0.4"].mean(),
                "methods": "; ".join(top["method"].tolist()),
            })
            for pos, (_, row) in enumerate(top.iterrows(), start=1):
                list_rows.append({
                    "scheme": scheme,
                    "top_k": k,
                    "position_in_scheme": pos,
                    "method": row["method"],
                    "Directionality_rank": row["Directionality_rank"],
                    "Consistency_rank": row["Consistency_rank"],
                    "D_plus_C_rank": row["D_plus_C_rank"],
                    "low_consistency": row["low_consistency"],
                    "is_pareto_front": row["is_pareto_front"],
                    "is_pareto_dominated": row["is_pareto_dominated"],
                    "pareto_depth": row["pareto_depth"],
                    "dominators": row["dominators"],
                })
    return pd.DataFrame(summary_rows), pd.DataFrame(list_rows)


def replacement_analysis(scores: pd.DataFrame, topks: list[int]) -> pd.DataFrame:
    rows = []
    for k in topks:
        d_top = scores.sort_values("D_only_rank").head(k)
        dc_top = scores.sort_values("D_plus_C_rank").head(k)
        d_set, dc_set = set(d_top["method"]), set(dc_top["method"])
        groups = [
            ("retained_by_both", d_set & dc_set),
            ("D_only_only", d_set - dc_set),
            ("D_plus_C_only", dc_set - d_set),
        ]
        for group, methods in groups:
            sub = scores[scores["method"].isin(methods)].copy()
            for _, row in sub.sort_values("D_plus_C_rank").iterrows():
                rows.append({
                    "top_k": k,
                    "replacement_group": group,
                    "method": row["method"],
                    "D_only_rank": row["D_only_rank"],
                    "D_plus_C_rank": row["D_plus_C_rank"],
                    "C_only_rank": row["C_only_rank"],
                    "Directionality_rank": row["Directionality_rank"],
                    "Consistency_rank": row["Consistency_rank"],
                    "low_consistency": row["low_consistency"],
                    "is_pareto_front": row["is_pareto_front"],
                    "is_pareto_dominated": row["is_pareto_dominated"],
                    "pareto_depth": row["pareto_depth"],
                    "dominators": row["dominators"],
                })
    return pd.DataFrame(rows)


def plot_pareto_counts(profile: pd.DataFrame, outdir: Path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = [f"{r.scheme}\nTop{r.top_k}" for r in profile.itertuples()]
    x = np.arange(len(profile))
    width = 0.35
    ax.bar(x - width/2, profile["n_pareto_front"], width=width, label="Pareto-front")
    ax.bar(x + width/2, profile["n_pareto_dominated"], width=width, label="Dominated")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Number of tools")
    ax.set_title("Pareto-front vs dominated tools among top-k")
    ax.legend(frameon=False)
    save_figure(fig, outdir / "fig1_pareto_front_vs_dominated_counts")


def plot_pareto_depth(profile: pd.DataFrame, outdir: Path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = [f"{r.scheme}\nTop{r.top_k}" for r in profile.itertuples()]
    ax.bar(labels, profile["mean_pareto_depth"])
    ax.set_ylabel("Mean Pareto depth; lower is better")
    ax.set_title("Mean Pareto depth among top-ranked tools")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, outdir / "fig2_mean_pareto_depth_topk")


def plot_pareto_scatter(scores: pd.DataFrame, outdir: Path, top_k: int = 5):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    front = scores[scores["is_pareto_front"]]
    dom = scores[scores["is_pareto_dominated"]]
    ax.scatter(dom["D_scaled"], dom["C_scaled"], s=34, alpha=0.6, label="Dominated")
    ax.scatter(front["D_scaled"], front["C_scaled"], s=52, marker="^", label="Pareto front")

    dtop = set(scores.sort_values("D_only_rank").head(top_k)["method"])
    dctop = set(scores.sort_values("D_plus_C_rank").head(top_k)["method"])
    selected = scores[scores["method"].isin(dtop | dctop)]
    for _, row in selected.iterrows():
        ax.text(row["D_scaled"] + 0.015, row["C_scaled"] + 0.015, row["method"], fontsize=7)

    ax.set_xlabel("Directionality score, scaled")
    ax.set_ylabel("Consistency score, scaled")
    ax.set_title(f"Directionality–Consistency Pareto view, top {top_k} labeled")
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, outdir / f"fig3_pareto_scatter_D_vs_C_top{top_k}")


def plot_replacement_groups(repl: pd.DataFrame, outdir: Path, top_k: int = 5):
    plt = setup_matplotlib()
    sub = repl[repl["top_k"] == top_k].copy()
    order = ["D_only_only", "retained_by_both", "D_plus_C_only"]
    sub["replacement_group"] = pd.Categorical(sub["replacement_group"], categories=order, ordered=True)
    sub = sub.sort_values(["replacement_group", "D_plus_C_rank"])
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.35 * len(sub) + 1.2)))
    y = np.arange(len(sub))
    ax.scatter(sub["Directionality_rank"], y, label="D rank", marker="o")
    ax.scatter(sub["Consistency_rank"], y, label="C rank", marker="s")
    ax.set_yticks(y)
    labels = [f"{r.method} [{r.replacement_group}]" for r in sub.itertuples()]
    ax.set_yticklabels(labels)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Rank; lower is better")
    ax.set_title(f"D-only vs D+C top {top_k}: retained and replaced methods")
    ax.legend(frameon=False)
    save_figure(fig, outdir / f"fig4_replacement_profile_top{top_k}")


def main():
    parser = argparse.ArgumentParser(description="Pareto front / dominated-tool validation for D-only vs D+C ranking.")
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_ANALYSIS_DIR / "04_pareto_dominated_validation"))
    parser.add_argument("--exclude_methods", default=";".join(DEFAULT_EXCLUDE_METHODS))
    parser.add_argument("--include_gt_in_directionality", action="store_true")
    parser.add_argument("--topks", default="5,10")
    parser.add_argument("--low_consistency_fraction", type=float, default=0.30)
    parser.add_argument("--weight_grid", default=None)
    args = parser.parse_args()

    outdir = ensure_dir(args.output_dir)
    figdir = ensure_dir(outdir / "figures")
    exclude_methods = [x for x in args.exclude_methods.split(";") if x]
    topks = [int(x) for x in args.topks.split(",") if x.strip()]
    weight_grid = parse_weight_grid(args.weight_grid)

    metric_df = read_metric_scores(args.results_dir)
    fam = recompute_family_scores(metric_df, args.include_gt_in_directionality, exclude_methods)
    _, dc = compute_dc_weight_sweep(fam, weight_grid)
    scores = add_primary_ranking_columns(dc)
    scores["low_consistency"], c_threshold, n_low_c = low_rank_flags(scores, "Consistency_rank", args.low_consistency_fraction)
    scores = compute_pareto(scores)

    profile, lists = topk_profile_pareto(scores, topks)
    repl = replacement_analysis(scores, topks)

    save_csv_readable(scores.sort_values("D_plus_C_rank"), outdir / "01_method_scores_ranks_pareto.csv")
    save_csv_readable(profile, outdir / "02_topk_profile_pareto_summary.csv")
    save_csv_readable(lists, outdir / "03_topk_method_lists_with_pareto.csv")
    save_csv_readable(repl, outdir / "04_replacement_analysis.csv")
    save_csv_readable(scores.sort_values(["pareto_depth", "D_plus_C_rank"]), outdir / "05_pareto_scores_all_methods.csv")

    plot_pareto_counts(profile, figdir)
    plot_pareto_depth(profile, figdir)
    for k in topks:
        plot_pareto_scatter(scores, figdir, top_k=k)
        plot_replacement_groups(repl, figdir, top_k=k)

    text = f"""# Pareto front / dominated-tool validation

- Results directory: `{args.results_dir}`
- Excluded methods: {exclude_methods}
- Directionality includes groundtruth_correlation: {args.include_gt_in_directionality}
- Low-consistency definition: bottom {args.low_consistency_fraction:.0%} by Consistency rank, threshold rank > {c_threshold}, n_low = {n_low_c}
- Top-k values: {topks}

## Top-k Pareto profile

{profile.to_markdown(index=False)}

## Replacement analysis

{repl.to_markdown(index=False)}

Interpretation: A method is Pareto-dominated if another method has equal-or-better Directionality and Consistency and is strictly better in at least one dimension. Dominated tools are not necessarily useless, but they are hard to justify as top-ranked methods under a two-objective view of velocity-field quality.
"""
    write_text(outdir / "validation_summary.md", text)
    print(f"Done. Results written to: {outdir}")


if __name__ == "__main__":
    main()
