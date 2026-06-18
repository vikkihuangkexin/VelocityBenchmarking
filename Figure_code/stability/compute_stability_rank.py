#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Stability scores and reversed ranks for RNA velocity tools.

This script combines two stability components:

1. Downsampling stability
   - Input: Downsampling_groundtruth_correlation.csv
   - Cell-subsampling columns and gene-subsampling columns are averaged separately.
   - The two perturbation types are then combined as:
       downsampling_score = 0.5 * cellsub_mean + 0.5 * genesub_mean
     with availability-aware re-normalization if one side is missing.

2. Batch-run stability
   - Input: batchrun.csv
   - Uses the AVG column if present; otherwise averages numeric run columns.
   - Batch-run is treated as "not applicable" for tools without available
     batch-run values unless the user provides an explicit applicable-method list.

Final Stability score:
    z_downsampling and z_batchrun are computed separately.
    stability_score = 0.5 * z_downsampling + 0.5 * z_batchrun
    with weights re-normalized over available components.

Important:
    Batch-run NA values for non-deep-learning / deterministic tools are not
    treated as poor performance. They are excluded from the weighted average.

Default output:
    PlotData/stability/Results/stability_rank_summary.csv
    PlotData/Results/reversed_rank/stability_rank.csv

reversed_rank:
    reversed_rank = n_methods + 1 - final_stability_rank
    Therefore, larger reversed_rank means better stability.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_STABILITY_DIR = Path("PlotData/stability")
DEFAULT_RESULTS_DIR = Path("PlotData/Results/stability")
DEFAULT_REVERSED_DIR = Path("PlotData/Results/reversed_rank")

DOWNSAMPLING_FILENAME = "Downsampling_groundtruth_correlation.csv"
BATCHRUN_FILENAME = "batchrun.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Stability score/rank and reversed rank."
    )
    parser.add_argument(
        "--downsampling_csv",
        default=None,
        help=(
            "Path to Downsampling_groundtruth_correlation.csv. "
            "If omitted, the script searches common Stability/Data locations."
        ),
    )
    parser.add_argument(
        "--batchrun_csv",
        default=None,
        help=(
            "Path to batchrun.csv. "
            "If omitted, the script searches common Stability/Data locations."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory for full Stability output tables.",
    )
    parser.add_argument(
        "--reversed_dir",
        default=str(DEFAULT_REVERSED_DIR),
        help="Directory for reversed-rank output.",
    )
    parser.add_argument(
        "--downsampling_weight",
        type=float,
        default=0.5,
        help="Weight of downsampling component in final stability score.",
    )
    parser.add_argument(
        "--batchrun_weight",
        type=float,
        default=0.5,
        help="Weight of batch-run component in final stability score.",
    )
    parser.add_argument(
        "--keep_zero_values",
        action="store_true",
        help=(
            "Keep raw zero values as real scores. By default, zeros in "
            "downsampling/batchrun numeric columns are treated as missing, "
            "consistent with the previous accuracy pipeline convention."
        ),
    )
    parser.add_argument(
        "--batchrun_applicable_methods",
        default=None,
        help=(
            "Optional comma/semicolon-separated list of methods to mark as "
            "batch-run applicable. If omitted, only methods with available "
            "batch-run AVG are treated as applicable."
        ),
    )
    parser.add_argument(
        "--region_method",
        default="Region Velocity",
        help="Method name to force to the last rank if it has no stability score.",
    )
    parser.add_argument(
        "--make_plots",
        action="store_true",
        help="Generate simple bar plots in the Results/figures directory.",
    )
    return parser.parse_args()


def find_input_file(user_path: Optional[str], filename: str) -> Path:
    """Find an input file from explicit path or common Stability directories."""
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p

    candidates = [
        DEFAULT_STABILITY_DIR / "Data" / filename,
        DEFAULT_STABILITY_DIR / filename,
        DEFAULT_STABILITY_DIR / "Input" / filename,
        Path.cwd() / filename,
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent.parent / "Data" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p

    msg = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find {filename}. Checked:\n{msg}\n"
        f"Please provide --{filename.split('.')[0].lower()}_csv explicitly."
    )


def split_method_list(s: Optional[str]) -> Optional[set[str]]:
    if s is None or str(s).strip() == "":
        return None
    items = re.split(r"[;,]", s)
    return {x.strip() for x in items if x.strip()}


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    mu = vals.mean(skipna=True)
    sd = vals.std(skipna=True)
    return (vals - mu) / ((sd if pd.notna(sd) else 0.0) + 1e-8)


def clean_numeric_values(df: pd.DataFrame, numeric_cols: list[str], zero_as_missing: bool) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    n_zero = 0
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if zero_as_missing:
            zero_mask = out[col].eq(0)
            n_zero += int(zero_mask.sum())
            out.loc[zero_mask, col] = np.nan
    return out, n_zero


def load_downsampling(path: Path, zero_as_missing: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Method" not in df.columns:
        raise ValueError(f"`Method` column not found in {path}")

    numeric_cols = [c for c in df.columns if c != "Method"]
    df, n_zero = clean_numeric_values(df, numeric_cols, zero_as_missing)

    cell_cols = [c for c in numeric_cols if c.lower().startswith("cellsub")]
    gene_cols = [c for c in numeric_cols if c.lower().startswith("genesub")]

    if not cell_cols:
        cell_cols = [c for c in numeric_cols if "cell" in c.lower()]
    if not gene_cols:
        gene_cols = [c for c in numeric_cols if "gene" in c.lower()]

    if not cell_cols or not gene_cols:
        raise ValueError(
            "Could not identify both cellsub and genesub columns in downsampling file."
        )

    out = pd.DataFrame({"method": df["Method"].astype(str).str.strip()})
    out["cell_downsampling_score"] = df[cell_cols].mean(axis=1, skipna=True)
    out["gene_downsampling_score"] = df[gene_cols].mean(axis=1, skipna=True)
    out["cell_downsampling_n_valid"] = df[cell_cols].notna().sum(axis=1)
    out["gene_downsampling_n_valid"] = df[gene_cols].notna().sum(axis=1)
    out["cell_downsampling_n_total"] = len(cell_cols)
    out["gene_downsampling_n_total"] = len(gene_cols)

    # Combine cell and gene perturbation scores with equal weight and
    # availability-aware re-normalization.
    combined = []
    available_weight_sum = []
    for _, row in out.iterrows():
        numerator = 0.0
        denom = 0.0
        if pd.notna(row["cell_downsampling_score"]):
            numerator += 0.5 * row["cell_downsampling_score"]
            denom += 0.5
        if pd.notna(row["gene_downsampling_score"]):
            numerator += 0.5 * row["gene_downsampling_score"]
            denom += 0.5
        combined.append(numerator / denom if denom > 0 else np.nan)
        available_weight_sum.append(denom)

    out["downsampling_score"] = combined
    out["downsampling_available_weight_sum"] = available_weight_sum
    out["downsampling_n_zero_as_missing"] = n_zero
    out["downsampling_rank"] = out["downsampling_score"].rank(ascending=False, method="average")
    return out


def load_batchrun(path: Path, zero_as_missing: bool, applicable_methods: Optional[set[str]]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Method" not in df.columns:
        raise ValueError(f"`Method` column not found in {path}")

    numeric_cols = [c for c in df.columns if c != "Method"]
    df, n_zero = clean_numeric_values(df, numeric_cols, zero_as_missing)

    out = pd.DataFrame({"method": df["Method"].astype(str).str.strip()})

    if "AVG" in df.columns:
        out["batchrun_score"] = df["AVG"]
        run_cols = [c for c in numeric_cols if c != "AVG"]
    else:
        run_cols = numeric_cols
        out["batchrun_score"] = df[run_cols].mean(axis=1, skipna=True)

    out["batchrun_n_valid"] = df[run_cols].notna().sum(axis=1) if run_cols else out["batchrun_score"].notna().astype(int)
    out["batchrun_n_total"] = len(run_cols) if run_cols else 1
    out["batchrun_available"] = out["batchrun_score"].notna()

    if applicable_methods is None:
        # Conservative default: only available batch-run values are applicable.
        # This prevents penalizing non-DL or deterministic tools with NA batch-run.
        out["batchrun_applicable"] = out["batchrun_available"]
    else:
        out["batchrun_applicable"] = out["method"].isin(applicable_methods) | out["batchrun_available"]

    out["batchrun_missing_but_applicable"] = out["batchrun_applicable"] & (~out["batchrun_available"])
    out["batchrun_n_zero_as_missing"] = n_zero
    out["batchrun_rank_available_only"] = out["batchrun_score"].rank(ascending=False, method="average")
    return out


def combine_stability(
    down: pd.DataFrame,
    batch: pd.DataFrame,
    down_w: float,
    batch_w: float,
    region_method: str,
) -> pd.DataFrame:
    methods = sorted(set(down["method"]).union(set(batch["method"])))
    base = pd.DataFrame({"method": methods})

    df = base.merge(down, on="method", how="left")
    df = df.merge(batch, on="method", how="left")

    df["z_downsampling"] = zscore(df["downsampling_score"])
    # z-score batch-run only across available batch-run values.
    df["z_batchrun"] = zscore(df["batchrun_score"])

    final_scores = []
    available_weights = []
    n_components = []
    used_components = []

    for _, row in df.iterrows():
        numerator = 0.0
        denom = 0.0
        n = 0
        used = []

        if pd.notna(row["z_downsampling"]):
            numerator += down_w * row["z_downsampling"]
            denom += down_w
            n += 1
            used.append("downsampling")

        # Batch-run participates only when an actual batch-run score exists.
        # Missing-but-applicable is flagged but not imputed.
        if pd.notna(row["z_batchrun"]):
            numerator += batch_w * row["z_batchrun"]
            denom += batch_w
            n += 1
            used.append("batchrun")

        final_scores.append(numerator / denom if denom > 0 else np.nan)
        available_weights.append(denom)
        n_components.append(n)
        used_components.append(";".join(used))

    df["stability_score"] = final_scores
    df["available_weight_sum"] = available_weights
    df["n_valid_stability_components"] = n_components
    df["used_components"] = used_components

    df["component_coverage"] = df["n_valid_stability_components"] / 2.0
    df["stability_rank_tie_average"] = df["stability_score"].rank(ascending=False, method="average")

    # Deterministic final rank: valid scores first, missing scores last.
    valid = df.loc[df["stability_score"].notna()].copy()
    missing = df.loc[df["stability_score"].isna()].copy()

    valid = valid.sort_values(
        ["stability_score", "downsampling_score", "batchrun_score", "method"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    valid["final_stability_rank"] = np.arange(1, len(valid) + 1, dtype=int)

    # if it has no valid stability score.
    region_norm = region_method.strip().lower()
    if not missing.empty:
        missing["_is_region"] = missing["method"].astype(str).str.strip().str.lower().eq(region_norm)
        missing = missing.sort_values(["_is_region", "method"], ascending=[True, True]).reset_index(drop=True)
        missing["final_stability_rank"] = np.arange(len(valid) + 1, len(valid) + len(missing) + 1, dtype=int)
        missing = missing.drop(columns=["_is_region"])

    out = pd.concat([valid, missing], ignore_index=True, sort=False)
    n_methods = len(out)
    out["reversed_rank"] = n_methods + 1 - out["final_stability_rank"]
    return out


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not available; skipping plots.")
        return

    figdir = output_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    plot_df = df.loc[df["stability_score"].notna()].sort_values("final_stability_rank")

    fig_h = max(5.5, 0.32 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    ax.barh(plot_df["method"], plot_df["stability_score"])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Stability score")
    ax.set_title("Final Stability score by method")
    for _, row in plot_df.iterrows():
        ax.text(
            row["stability_score"],
            row["method"],
            f"  rank {int(row['final_stability_rank'])}",
            va="center",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(figdir / "stability_score_bar.png", dpi=300, bbox_inches="tight")
    fig.savefig(figdir / "stability_score_bar.pdf", bbox_inches="tight")
    plt.close(fig)

    comp = plot_df[["method", "z_downsampling", "z_batchrun"]].copy()
    x = np.arange(len(comp))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(comp)), 5.0))
    ax.bar(x - width / 2, comp["z_downsampling"], width, label="Downsampling")
    ax.bar(x + width / 2, comp["z_batchrun"], width, label="Batch-run")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comp["method"], rotation=60, ha="right")
    ax.set_ylabel("Component z-score")
    ax.set_title("Stability component z-scores")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "stability_component_zscores.png", dpi=300, bbox_inches="tight")
    fig.savefig(figdir / "stability_component_zscores.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    reversed_dir = Path(args.reversed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reversed_dir.mkdir(parents=True, exist_ok=True)

    zero_as_missing = not args.keep_zero_values
    applicable_methods = split_method_list(args.batchrun_applicable_methods)

    down_path = find_input_file(args.downsampling_csv, DOWNSAMPLING_FILENAME)
    batch_path = find_input_file(args.batchrun_csv, BATCHRUN_FILENAME)

    print(f"Downsampling input: {down_path}")
    print(f"Batch-run input:    {batch_path}")
    print(f"Output directory:  {output_dir}")
    print(f"Reversed output:   {reversed_dir / 'stability_rank.csv'}")

    down = load_downsampling(down_path, zero_as_missing=zero_as_missing)
    batch = load_batchrun(batch_path, zero_as_missing=zero_as_missing, applicable_methods=applicable_methods)

    down.to_csv(output_dir / "01_downsampling_component_scores.csv", index=False)
    batch.to_csv(output_dir / "02_batchrun_component_scores.csv", index=False)

    final = combine_stability(
        down=down,
        batch=batch,
        down_w=args.downsampling_weight,
        batch_w=args.batchrun_weight,
        region_method=args.region_method,
    )

    # Main full output.
    final.to_csv(output_dir / "03_stability_rank_summary.csv", index=False)

    # Compact output for plotting / integration.
    compact_cols = [
        "method",
        "final_stability_rank",
        "reversed_rank",
        "stability_score",
        "stability_rank_tie_average",
        "downsampling_score",
        "downsampling_rank",
        "batchrun_score",
        "batchrun_rank_available_only",
        "z_downsampling",
        "z_batchrun",
        "n_valid_stability_components",
        "available_weight_sum",
        "component_coverage",
        "used_components",
        "batchrun_applicable",
        "batchrun_available",
        "batchrun_missing_but_applicable",
    ]
    compact_cols = [c for c in compact_cols if c in final.columns]
    compact = final[compact_cols].copy()
    compact.to_csv(output_dir / "04_stability_rank_for_plot.csv", index=False)

    # Requested reversed-rank file.
    reversed_path = reversed_dir / "stability_rank.csv"
    compact.to_csv(reversed_path, index=False)

    # QC
    qc = pd.DataFrame(
        [
            {"item": "downsampling_csv", "value": str(down_path)},
            {"item": "batchrun_csv", "value": str(batch_path)},
            {"item": "zero_as_missing", "value": zero_as_missing},
            {"item": "downsampling_weight", "value": args.downsampling_weight},
            {"item": "batchrun_weight", "value": args.batchrun_weight},
            {"item": "n_methods", "value": len(final)},
            {"item": "n_methods_with_downsampling", "value": int(final["downsampling_score"].notna().sum())},
            {"item": "n_methods_with_batchrun", "value": int(final["batchrun_score"].notna().sum())},
            {"item": "n_methods_with_final_stability", "value": int(final["stability_score"].notna().sum())},
            {"item": "n_batchrun_missing_but_applicable", "value": int(final.get("batchrun_missing_but_applicable", pd.Series(dtype=bool)).fillna(False).sum())},
        ]
    )
    qc.to_csv(output_dir / "05_stability_qc_summary.csv", index=False)

    if args.make_plots:
        make_plots(final, output_dir)

    print("Done.")
    print(f"Full stability summary: {output_dir / '03_stability_rank_summary.csv'}")
    print(f"Requested reversed-rank file: {reversed_path}")


if __name__ == "__main__":
    main()
