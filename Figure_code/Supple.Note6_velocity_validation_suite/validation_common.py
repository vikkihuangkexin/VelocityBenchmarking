#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for validating the revised RNA velocity accuracy aggregation.

This module is intentionally lightweight and shared by all validation scripts.
It assumes the accuracy pipeline has already generated files in:
    ../../../PlotData/Results/accuracy

Main input used by most analyses:
    metric_scores.csv

Expected core columns in metric_scores.csv:
    method, metric, metric_score

Default family definitions used for validation:
    Directionality without GT: angle_consistency, CBDir, transition_score
    Directionality with GT:    angle_consistency, CBDir, transition_score, groundtruth_correlation
    Consistency:               ICCoh, peak_location

All scores are assumed to be z-score-like values where larger is better.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "PlotData" / "Results" / "accuracy"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "PlotData" / "Results"

DEFAULT_EXCLUDE_METHODS = ["Region Velocity"]

DIRECTIONALITY_METRICS_NO_GT = ["angle_consistency", "CBDir", "transition_score"]
DIRECTIONALITY_METRICS_WITH_GT = [
    "angle_consistency",
    "CBDir",
    "transition_score",
    "groundtruth_correlation",
]
CONSISTENCY_METRICS = ["ICCoh", "peak_location"]

DEFAULT_WEIGHT_GRID = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_metric_scores(results_dir: Path | str) -> pd.DataFrame:
    """Read metric-level scores from the pipeline."""
    results_dir = Path(results_dir)
    path = results_dir / "metric_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find required file: {path}")
    df = pd.read_csv(path)
    required = {"method", "metric", "metric_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def normalize_method_name(s: str) -> str:
    """Normalize method names for robust merges across old/new ranking tables."""
    if pd.isna(s):
        return s
    s = str(s).strip()
    # Common harmonization. Extend here if old tables use slightly different labels.
    replacements = {
        "scVelo stochastic": "scVelo stochastic",
        "scvelo stochastic": "scVelo stochastic",
        "scVelo dynamical": "scVelo dynamical",
        "scvelo dynamical": "scVelo dynamical",
        "Pyro-Velocity": "Pyro-Velocity",
        "PyroVelocity": "Pyro-Velocity",
        "Region Velocity": "Region Velocity",
    }
    return replacements.get(s, s)


def parse_weight_grid(weight_grid: Optional[str] = None) -> List[Tuple[float, float]]:
    """
    Parse a string such as '0.5:0.5,0.6:0.4,0.9:0.1'.
    If None, return DEFAULT_WEIGHT_GRID.
    """
    if weight_grid is None or str(weight_grid).strip() == "":
        return list(DEFAULT_WEIGHT_GRID)
    out = []
    for item in str(weight_grid).split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":")
        w_d = float(left)
        w_c = float(right)
        if abs((w_d + w_c) - 1.0) > 1e-6:
            raise ValueError(f"Weight pair must sum to 1: {item}")
        out.append((w_d, w_c))
    if not out:
        raise ValueError("No valid weight pair found.")
    return out


def weight_label(w_d: float, w_c: float) -> str:
    return f"D{w_d:g}_C{w_c:g}"


def minmax01(x: pd.Series) -> pd.Series:
    """Scale a numeric Series to [0, 1]; constant vectors become 0.5."""
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(np.nan, index=x.index)
    mn, mx = x.min(skipna=True), x.max(skipna=True)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(0.5, index=x.index)
    return (x - mn) / (mx - mn)


def recompute_family_scores(
    metric_df: pd.DataFrame,
    include_gt_in_directionality: bool = False,
    exclude_methods: Sequence[str] = DEFAULT_EXCLUDE_METHODS,
) -> pd.DataFrame:
    """
    Recompute Directionality and Consistency family scores from metric-level scores.

    This is used for validation because we sometimes want Directionality to exclude
    groundtruth_correlation, even if the original pipeline included it.
    """
    df = metric_df.copy()
    df["method"] = df["method"].map(normalize_method_name)
    if exclude_methods:
        df = df[~df["method"].isin(list(exclude_methods))].copy()

    direction_metrics = (
        DIRECTIONALITY_METRICS_WITH_GT if include_gt_in_directionality else DIRECTIONALITY_METRICS_NO_GT
    )
    metric_to_family = {m: "Directionality" for m in direction_metrics}
    metric_to_family.update({m: "Consistency" for m in CONSISTENCY_METRICS})

    df = df[df["metric"].isin(metric_to_family)].copy()
    df["family"] = df["metric"].map(metric_to_family)

    detail = df[["method", "metric", "family", "metric_score"]].copy()

    family_long = (
        detail.groupby(["method", "family"], dropna=False)
        .agg(
            family_score=("metric_score", "mean"),
            n_valid_metrics=("metric_score", "count"),
            metrics_used=("metric", lambda s: ";".join(sorted(map(str, s.dropna().unique())))),
        )
        .reset_index()
    )

    # Expected total metrics per family for coverage.
    expected_counts = {
        "Directionality": len(direction_metrics),
        "Consistency": len(CONSISTENCY_METRICS),
    }
    family_long["n_total_metrics_expected"] = family_long["family"].map(expected_counts)
    family_long["metric_coverage"] = family_long["n_valid_metrics"] / family_long["n_total_metrics_expected"]

    family_wide = family_long.pivot(index="method", columns="family", values="family_score").reset_index()
    # Add ranks; larger score is better, so ascending=False.
    for fam in ["Directionality", "Consistency"]:
        if fam not in family_wide.columns:
            family_wide[fam] = np.nan
        family_wide[f"{fam}_rank"] = family_wide[fam].rank(ascending=False, method="average")
        family_wide[f"{fam}_scaled"] = minmax01(family_wide[fam])

    # Merge metric coverage for convenience.
    cov = family_long.pivot(index="method", columns="family", values="metric_coverage").reset_index()
    cov = cov.rename(columns={"Directionality": "Directionality_metric_coverage", "Consistency": "Consistency_metric_coverage"})
    family_wide = family_wide.merge(cov, on="method", how="left")

    return family_wide


def compute_dc_weight_sweep(
    family_scores: pd.DataFrame,
    weight_grid: Sequence[Tuple[float, float]] = DEFAULT_WEIGHT_GRID,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute D+C weighted scores/ranks for a set of D/C weights.

    Missing Directionality or Consistency is handled by available-weight renormalization.
    In practice, all standard methods should have both families.
    """
    rows = []
    for w_d, w_c in weight_grid:
        tmp = family_scores.copy()
        d = tmp["Directionality"]
        c = tmp["Consistency"]
        numerator = d.fillna(0) * w_d + c.fillna(0) * w_c
        denom = (~d.isna()).astype(float) * w_d + (~c.isna()).astype(float) * w_c
        tmp["accuracy_score"] = numerator / denom.replace(0, np.nan)
        tmp["accuracy_rank"] = tmp["accuracy_score"].rank(ascending=False, method="average")
        tmp["w_directionality"] = w_d
        tmp["w_consistency"] = w_c
        tmp["weight_label"] = weight_label(w_d, w_c)
        rows.append(tmp)

    details = pd.concat(rows, ignore_index=True)
    summary = (
        details.groupby("method", dropna=False)
        .agg(
            dc_mean_rank=("accuracy_rank", "mean"),
            dc_median_rank=("accuracy_rank", "median"),
            dc_best_rank=("accuracy_rank", "min"),
            dc_worst_rank=("accuracy_rank", "max"),
            dc_mean_score=("accuracy_score", "mean"),
            dc_median_score=("accuracy_score", "median"),
            n_weight_settings=("weight_label", "nunique"),
        )
        .reset_index()
    )
    summary["dc_consensus_rank"] = summary["dc_mean_rank"].rank(ascending=True, method="average")

    # Add family scores/ranks.
    cols = [
        "method",
        "Directionality",
        "Consistency",
        "Directionality_rank",
        "Consistency_rank",
        "Directionality_scaled",
        "Consistency_scaled",
        "Directionality_metric_coverage",
        "Consistency_metric_coverage",
    ]
    summary = summary.merge(family_scores[[c for c in cols if c in family_scores.columns]], on="method", how="left")

    # Wide rank table.
    rank_wide = details.pivot(index="method", columns="weight_label", values="accuracy_rank").reset_index()
    rank_wide = summary[["method", "dc_consensus_rank", "dc_mean_rank"]].merge(rank_wide, on="method", how="left")

    return details, summary


def add_primary_ranking_columns(summary: pd.DataFrame) -> pd.DataFrame:
    """Add explicit D-only, C-only, and D+C rank columns."""
    out = summary.copy()
    out["D_only_rank"] = out["Directionality_rank"]
    out["C_only_rank"] = out["Consistency_rank"]
    out["D_plus_C_rank"] = out["dc_consensus_rank"]
    out["directionality_advantage_rank"] = out["Consistency_rank"] - out["Directionality_rank"]
    return out


def low_rank_flags(df: pd.DataFrame, rank_col: str, fraction: float) -> Tuple[pd.Series, float, int]:
    """
    Return boolean flag for bottom fraction by rank.

    Example: n=21, fraction=0.30 => bottom ceil(6.3)=7 methods.
    Low flag means rank > n - n_low.
    """
    n = int(df[rank_col].notna().sum())
    n_low = int(math.ceil(n * fraction))
    threshold = n - n_low
    flag = df[rank_col] > threshold
    return flag, threshold, n_low


def corr_one(x: pd.Series, y: pd.Series, method: str = "spearman") -> Tuple[float, float, int]:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(data)
    if n < 3:
        return np.nan, np.nan, n
    if method == "spearman":
        r, p = spearmanr(data["x"], data["y"])
    elif method == "pearson":
        r, p = pearsonr(data["x"], data["y"])
    elif method == "kendall":
        r, p = kendalltau(data["x"], data["y"])
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    return float(r), float(p), n


def save_markdown_table(df: pd.DataFrame, path: Path | str, title: Optional[str] = None, max_rows: int = 50) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        if title:
            f.write(f"# {title}\n\n")
        if len(df) > max_rows:
            f.write(df.head(max_rows).to_markdown(index=False))
            f.write(f"\n\nShowing first {max_rows} of {len(df)} rows.\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")


def save_csv_readable(df: pd.DataFrame, path: Path | str, float_digits: int = 4) -> None:
    """Save CSV with stable column order and rounded floats for readability."""
    out = df.copy()
    for col in out.select_dtypes(include=["float"]).columns:
        out[col] = out[col].round(float_digits)
    out.to_csv(path, index=False)


def write_text(path: Path | str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def setup_matplotlib():
    """Use a headless backend and compact default font sizes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def save_figure(fig, outbase: Path | str, close: bool = True) -> None:
    """Save both PNG and PDF versions of a matplotlib figure."""
    outbase = Path(outbase)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
