#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis 03: Top-k low-consistency contamination and retention/gain analysis.

Purpose
-------
This script compares three ranking schemes:
    1. D-only:       rank by Directionality only
    2. D+C consensus: average rank across D/C weighting schemes
    3. C-only:       rank by Consistency only

It asks:
    - Does D-only put low-consistency tools into top-k?
    - Does D+C reduce low-consistency contamination while preserving Directionality?
    - How much Directionality is lost, and how much Consistency is gained?

Default validation parameters match the current discussion:
    Directionality excludes groundtruth_correlation
    low-consistency = bottom 30% by Consistency rank
    top-k = 5 and 10

Outputs
-------
- 01_method_scores_and_ranks.csv
- 02_topk_profile_summary.csv
- 03_topk_method_lists.csv
- 04_directionality_retention_consistency_gain.csv
- validation_summary.md
- figures/*.png/pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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


def topk_profile(scores: pd.DataFrame, topks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    list_rows = []
    for k in topks:
        for scheme, info in SCHEME_INFO.items():
            rank_col = info["rank_col"]
            top = scores.sort_values(rank_col, ascending=True).head(k).copy()
            summary_rows.append({
                "scheme": scheme,
                "top_k": k,
                "n_methods": len(top),
                "mean_directionality_rank": top["Directionality_rank"].mean(),
                "mean_consistency_rank": top["Consistency_rank"].mean(),
                "mean_directionality_score": top["Directionality"].mean(),
                "mean_consistency_score": top["Consistency"].mean(),
                "n_low_consistency": int(top["low_consistency"].sum()),
                "n_low_directionality": int(top["low_directionality"].sum()),
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
                    "low_directionality": row["low_directionality"],
                })
    return pd.DataFrame(summary_rows), pd.DataFrame(list_rows)


def retention_gain(scores: pd.DataFrame, topks: list[int]) -> pd.DataFrame:
    rows = []
    for k in topks:
        d_top = scores.sort_values("D_only_rank").head(k).copy()
        dc_top = scores.sort_values("D_plus_C_rank").head(k).copy()
        c_top = scores.sort_values("C_only_rank").head(k).copy()
        d_set = set(d_top["method"])
        dc_set = set(dc_top["method"])
        c_set = set(c_top["method"])
        rows.append({
            "top_k": k,
            "D_only_mean_D_rank": d_top["Directionality_rank"].mean(),
            "D_plus_C_mean_D_rank": dc_top["Directionality_rank"].mean(),
            "directionality_rank_loss_DplusC_minus_Donly": dc_top["Directionality_rank"].mean() - d_top["Directionality_rank"].mean(),
            "D_only_mean_C_rank": d_top["Consistency_rank"].mean(),
            "D_plus_C_mean_C_rank": dc_top["Consistency_rank"].mean(),
            "consistency_rank_gain_Donly_minus_DplusC": d_top["Consistency_rank"].mean() - dc_top["Consistency_rank"].mean(),
            "D_only_low_C": int(d_top["low_consistency"].sum()),
            "D_plus_C_low_C": int(dc_top["low_consistency"].sum()),
            "low_C_reduction_Donly_minus_DplusC": int(d_top["low_consistency"].sum()) - int(dc_top["low_consistency"].sum()),
            "Donly_DC_overlap_n": len(d_set & dc_set),
            "Donly_DC_overlap_fraction": len(d_set & dc_set) / max(k, 1),
            "Donly_only_methods": "; ".join(sorted(d_set - dc_set)),
            "DplusC_only_methods": "; ".join(sorted(dc_set - d_set)),
            "Conly_DC_overlap_n": len(c_set & dc_set),
            "Conly_DC_overlap_fraction": len(c_set & dc_set) / max(k, 1),
        })
    return pd.DataFrame(rows)


def plot_topk_mean_family_ranks(profile: pd.DataFrame, outdir: Path):
    plt = setup_matplotlib()
    topks = sorted(profile["top_k"].unique())
    fig, axes = plt.subplots(1, len(topks), figsize=(6.6 * len(topks), 4.0), sharey=True)
    if len(topks) == 1:
        axes = [axes]
    for ax, k in zip(axes, topks):
        sub = profile[profile["top_k"] == k].copy()
        x = range(len(sub))
        width = 0.35
        ax.bar([i - width/2 for i in x], sub["mean_directionality_rank"], width=width, label="Mean D rank")
        ax.bar([i + width/2 for i in x], sub["mean_consistency_rank"], width=width, label="Mean C rank")
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["scheme"], rotation=25, ha="right")
        ax.invert_yaxis()
        ax.set_title(f"Top {k}")
        ax.set_ylabel("Mean rank; lower is better")
        ax.legend(frameon=False)
    save_figure(fig, outdir / "fig1_topk_mean_family_ranks")


def plot_low_consistency(profile: pd.DataFrame, outdir: Path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    labels = [f"{r.scheme}\nTop{r.top_k}" for r in profile.itertuples()]
    ax.bar(labels, profile["n_low_consistency"])
    ax.set_ylabel("Low-consistency tools in top-k")
    ax.set_title("Low-consistency contamination among top-ranked tools")
    ax.tick_params(axis="x", rotation=35)
    save_figure(fig, outdir / "fig2_low_consistency_contamination")


def plot_retention_gain(rg: pd.DataFrame, outdir: Path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = range(len(rg))
    width = 0.35
    ax.bar([i - width/2 for i in x], rg["directionality_rank_loss_DplusC_minus_Donly"], width=width, label="D rank loss")
    ax.bar([i + width/2 for i in x], rg["consistency_rank_gain_Donly_minus_DplusC"], width=width, label="C rank gain")
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Top {k}" for k in rg["top_k"]])
    ax.set_ylabel("Rank difference")
    ax.set_title("D+C retention/gain relative to D-only")
    ax.legend(frameon=False)
    save_figure(fig, outdir / "fig3_directionality_retention_vs_consistency_gain")


def main():
    parser = argparse.ArgumentParser(description="Top-k low-consistency contamination and D/C retention-gain analysis.")
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_ANALYSIS_DIR / "03_topk_contamination_retention"))
    parser.add_argument("--exclude_methods", default=";".join(DEFAULT_EXCLUDE_METHODS))
    parser.add_argument("--include_gt_in_directionality", action="store_true")
    parser.add_argument("--topks", default="5,10")
    parser.add_argument("--low_consistency_fraction", type=float, default=0.30)
    parser.add_argument("--low_directionality_fraction", type=float, default=0.30)
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
    scores["low_directionality"], d_threshold, n_low_d = low_rank_flags(scores, "Directionality_rank", args.low_directionality_fraction)

    profile, lists = topk_profile(scores, topks)
    rg = retention_gain(scores, topks)

    save_csv_readable(scores.sort_values("D_plus_C_rank"), outdir / "01_method_scores_and_ranks.csv")
    save_csv_readable(profile, outdir / "02_topk_profile_summary.csv")
    save_csv_readable(lists, outdir / "03_topk_method_lists.csv")
    save_csv_readable(rg, outdir / "04_directionality_retention_consistency_gain.csv")

    plot_topk_mean_family_ranks(profile, figdir)
    plot_low_consistency(profile, figdir)
    plot_retention_gain(rg, figdir)

    text = f"""# Top-k contamination and retention/gain validation

- Results directory: `{args.results_dir}`
- Excluded methods: {exclude_methods}
- Directionality includes groundtruth_correlation: {args.include_gt_in_directionality}
- Top-k values: {topks}
- Low-consistency definition: bottom {args.low_consistency_fraction:.0%} by Consistency rank, threshold rank > {c_threshold}, n_low = {n_low_c}
- Low-directionality definition: bottom {args.low_directionality_fraction:.0%} by Directionality rank, threshold rank > {d_threshold}, n_low = {n_low_d}

## Top-k profile summary

{profile.to_markdown(index=False)}

## D+C relative to D-only

{rg.to_markdown(index=False)}

Interpretation: a positive `consistency_rank_gain_Donly_minus_DplusC` means D+C improves consistency relative to D-only. A positive `directionality_rank_loss_DplusC_minus_Donly` means D+C sacrifices some Directionality relative to D-only.
"""
    write_text(outdir / "validation_summary.md", text)
    print(f"Done. Results written to: {outdir}")


if __name__ == "__main__":
    main()
