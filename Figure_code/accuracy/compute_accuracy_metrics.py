#!/usr/bin/env python3
"""
Run accuracy aggregation for RNA velocity tools.

This script implements the accuracy aggregation workflow:
  1. Read all real/sim accuracy metric CSV files.
  2. Convert wide tables to a single long table internally.
  3. Treat 0 as missing/NA, except the known true-zero case:
       real / peak_location / TopicVelo / 4_Mm_visual_cortex = 0
  4. Remove historical invalid values for G_32_Mm_embryos from methods other than
       InterVelo, PhyloVelo, VeloVAE, DeepVelo.
  5. Z-score each metric across all available values:
       z = (vals - vals.mean()) / (vals.std() + 1e-8)
  6. Average z-scores across all available datasets for each method x metric.
  7. Aggregate metric scores into family scores:
       Directionality = angle_consistency, CBDir, transition_score, groundtruth_correlation
       Consistency    = ICCoh, peak_location
  8. Sweep Directionality/Consistency weights:
       0.5/0.5, 0.6/0.4, 0.7/0.3, 0.8/0.2, 0.9/0.1
     and compute the final accuracy rank as the mean rank across weight settings.

The script writes only compact outputs needed for checking this test:
  - metric_scores.csv
  - family_detail.csv
  - family_scores_long.csv
  - family_scores_wide.csv
  - accuracy_weight_sweep_details.csv
  - accuracy_rank_summary.csv
  - rank_by_weight_wide.csv
  - score_by_weight_wide.csv
  - qc_summary.csv

Default paths are relative to the repository root:
  input:  PlotData/accuracy
  output: PlotData/Results/accuracy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

DEFAULT_INPUT_DIR = "PlotData/accuracy"
DEFAULT_OUTPUT_DIR = "PlotData/Results/accuracy"

EPS = 1e-8

# Metric family definitions used in the current accuracy aggregation.
METRIC_FAMILY: Dict[str, str] = {
    "angle_consistency": "Directionality",
    "CBDir": "Directionality",
    "transition_score": "Directionality",
    "groundtruth_correlation": "Directionality",
    "ICCoh": "Consistency",
    "peak_location": "Consistency",
}

FAMILIES = ["Directionality", "Consistency"]

# Weight grid for the final Directionality/Consistency rank consensus.
WEIGHT_GRID: List[Tuple[float, float]] = [
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
]

# Known valid true-zero exception.
VALID_ZERO_EXCEPTIONS = [
    {
        "data_type": "real",
        "metric": "peak_location",
        "method": "TopicVelo",
        "dataset_id": "4_Mm_visual_cortex",
    }
]

# 32_Mm_embryos has reliable results only from these methods.
G32_ALLOWED_METHODS = {"InterVelo", "PhyloVelo", "VeloVAE", "DeepVelo"}
G32_DATASET_ID = "G_32_Mm_embryos"


# -----------------------------
# Path and input helpers
# -----------------------------

def resolve_input_dir(input_dir: str | Path) -> Path:
    """Resolve an input directory that may be either accuracy0609 itself or its parent."""
    input_dir = Path(input_dir).expanduser().resolve()

    if (input_dir / "real").is_dir() and (input_dir / "sim").is_dir():
        return input_dir

    nested = input_dir / "accuracy0609"
    if (nested / "real").is_dir() and (nested / "sim").is_dir():
        return nested

    raise FileNotFoundError(
        f"Cannot find real/ and sim/ under {input_dir} or {nested}. "
        "Please pass PlotData/accuracy or its parent directory."
    )


def metric_from_filename(csv_path: Path, data_type: str) -> str:
    """Infer metric name from filenames such as scRNA_CBDir.csv or SIM_ICCoh.csv."""
    stem = csv_path.stem
    if data_type == "real" and stem.startswith("scRNA_"):
        return stem.replace("scRNA_", "", 1)
    if data_type == "sim" and stem.startswith("SIM_"):
        return stem.replace("SIM_", "", 1)
    # Fallback for unexpected but similar names.
    return stem.replace("scRNA_", "", 1).replace("SIM_", "", 1)


def infer_dataset_group(data_type: str, dataset_id: str) -> str:
    """Keep dataset group labels for QC only; used for quality-control summaries."""
    if data_type == "real":
        return "Gold" if str(dataset_id).startswith("G_") else "Other"
    if str(dataset_id).startswith("Bursting-tree"):
        return "Bursting"
    if str(dataset_id).startswith("lineage-tracing"):
        return "Lineage-tracing"
    return "Dyngen"


def read_metric_csv(csv_path: Path, data_type: str) -> pd.DataFrame:
    """Read one wide metric CSV and convert it to long format."""
    metric = metric_from_filename(csv_path, data_type)
    df = pd.read_csv(csv_path)

    if "Method" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a 'Method' column.")

    long_df = df.melt(
        id_vars=["Method"],
        var_name="dataset_id",
        value_name="value_raw",
    )
    long_df = long_df.rename(columns={"Method": "method"})
    long_df["data_type"] = data_type
    long_df["metric"] = metric
    long_df["source_file"] = str(csv_path)
    long_df["dataset_group"] = [
        infer_dataset_group(data_type, d) for d in long_df["dataset_id"]
    ]

    # Convert values to numeric. Non-numeric entries become NA.
    long_df["value_raw"] = pd.to_numeric(long_df["value_raw"], errors="coerce")
    return long_df


def read_all_metrics(input_dir: Path) -> pd.DataFrame:
    """Read all metric CSVs under real/ and sim/."""
    parts: List[pd.DataFrame] = []
    for data_type in ["real", "sim"]:
        folder = input_dir / data_type
        for csv_path in sorted(folder.glob("*.csv")):
            metric = metric_from_filename(csv_path, data_type)
            if metric not in METRIC_FAMILY:
                print(f"[WARN] Skipping unrecognized metric file: {csv_path}")
                continue
            parts.append(read_metric_csv(csv_path, data_type))

    if not parts:
        raise RuntimeError(f"No recognized metric CSVs found under {input_dir}.")

    return pd.concat(parts, ignore_index=True)


# -----------------------------
# Cleaning and z-scoring
# -----------------------------

def is_valid_zero_exception(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for entries where raw zero should be kept as a true value."""
    mask = pd.Series(False, index=df.index)
    for rule in VALID_ZERO_EXCEPTIONS:
        this = pd.Series(True, index=df.index)
        for col, expected in rule.items():
            this &= df[col].eq(expected)
        mask |= this
    return mask


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply missing-value rules and historical G_32_Mm_embryos cleanup."""
    df = df.copy()

    df["value_clean"] = df["value_raw"]
    df["valid_zero_exception"] = is_valid_zero_exception(df)

    # Historical invalid entries for G_32_Mm_embryos: only allowed methods are kept.
    df["legacy_G32_invalid_method"] = (
        df["data_type"].eq("real")
        & df["dataset_id"].eq(G32_DATASET_ID)
        & ~df["method"].isin(G32_ALLOWED_METHODS)
    )
    df.loc[df["legacy_G32_invalid_method"], "value_clean"] = np.nan

    # General rule: raw 0 means missing, except known valid-zero exception.
    df["zero_as_missing"] = (
        df["value_raw"].eq(0)
        & ~df["valid_zero_exception"]
        & ~df["legacy_G32_invalid_method"]
    )
    df.loc[df["zero_as_missing"], "value_clean"] = np.nan

    return df


def zscore_series(vals: pd.Series) -> pd.Series:
    """Classic z-score with NA skipped by pandas mean/std."""
    return (vals - vals.mean()) / (vals.std() + EPS)


def add_metric_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute z-score within each metric across all available values."""
    df = df.copy()
    df["z_score"] = df.groupby("metric")["value_clean"].transform(zscore_series)
    return df


# -----------------------------
# No-data-weight metric scores
# -----------------------------

def compute_metric_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Directly average z-scores across all available datasets for each method x metric.

    This is the key difference from the main pipeline:
      - no Gold/Other/Dyngen/Bursting/Lineage-tracing Spearman weighting
      - no dataset-group weighted adjustment
    """
    metric_scores = (
        df.groupby(["method", "metric"], dropna=False)
        .agg(
            metric_score=("z_score", "mean"),
            n_valid=("z_score", "count"),
            n_total=("z_score", "size"),
            n_real_total=("data_type", lambda x: int((x == "real").sum())),
            n_sim_total=("data_type", lambda x: int((x == "sim").sum())),
        )
        .reset_index()
    )
    metric_scores["coverage"] = metric_scores["n_valid"] / metric_scores["n_total"].replace(0, np.nan)
    metric_scores["family"] = metric_scores["metric"].map(METRIC_FAMILY)

    # Rank each metric score across methods. Larger z-score is better.
    metric_scores["metric_rank"] = metric_scores.groupby("metric")["metric_score"].rank(
        ascending=False,
        method="average",
        na_option="bottom",
    )
    return metric_scores


def complete_method_metric_grid(metric_scores: pd.DataFrame) -> pd.DataFrame:
    """Add explicit NA rows for method x metric combinations absent from the source files."""
    all_methods = sorted(metric_scores["method"].dropna().unique())
    all_metrics = list(METRIC_FAMILY.keys())
    grid = pd.MultiIndex.from_product(
        [all_methods, all_metrics], names=["method", "metric"]
    ).to_frame(index=False)
    out = grid.merge(metric_scores, on=["method", "metric"], how="left", suffixes=("", "_old"))
    out["family"] = out["metric"].map(METRIC_FAMILY)
    for col in ["n_valid", "n_total", "n_real_total", "n_sim_total"]:
        out[col] = out[col].fillna(0).astype(int)
    out["coverage"] = out["coverage"].astype(float)
    return out


# -----------------------------
# Family scores
# -----------------------------

def compute_family_scores(metric_scores_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate method x metric scores into method x family scores."""
    detail = metric_scores_full.copy()
    detail = detail.rename(columns={"metric_score": "mean_z_metric_score"})

    family_scores = (
        detail.groupby(["method", "family"], dropna=False)
        .agg(
            family_score=("mean_z_metric_score", "mean"),
            n_valid_metrics=("mean_z_metric_score", "count"),
            n_total_metrics=("metric", "size"),
            mean_metric_coverage=("coverage", "mean"),
            total_valid_values=("n_valid", "sum"),
            total_possible_values=("n_total", "sum"),
        )
        .reset_index()
    )
    family_scores["metric_coverage"] = (
        family_scores["n_valid_metrics"] / family_scores["n_total_metrics"].replace(0, np.nan)
    )
    family_scores["value_coverage"] = (
        family_scores["total_valid_values"] / family_scores["total_possible_values"].replace(0, np.nan)
    )
    family_scores["family_rank"] = family_scores.groupby("family")["family_score"].rank(
        ascending=False,
        method="average",
        na_option="bottom",
    )

    wide = family_scores.pivot(index="method", columns="family", values="family_score").reset_index()
    rank_wide = family_scores.pivot(index="method", columns="family", values="family_rank").reset_index()
    rank_wide = rank_wide.rename(columns={fam: f"{fam}_rank" for fam in FAMILIES if fam in rank_wide.columns})

    metric_cov_wide = family_scores.pivot(index="method", columns="family", values="metric_coverage").reset_index()
    metric_cov_wide = metric_cov_wide.rename(
        columns={fam: f"{fam}_metric_coverage" for fam in FAMILIES if fam in metric_cov_wide.columns}
    )

    value_cov_wide = family_scores.pivot(index="method", columns="family", values="value_coverage").reset_index()
    value_cov_wide = value_cov_wide.rename(
        columns={fam: f"{fam}_value_coverage" for fam in FAMILIES if fam in value_cov_wide.columns}
    )

    valid_metric_wide = family_scores.pivot(index="method", columns="family", values="n_valid_metrics").reset_index()
    valid_metric_wide = valid_metric_wide.rename(
        columns={fam: f"{fam}_n_valid_metrics" for fam in FAMILIES if fam in valid_metric_wide.columns}
    )

    total_metric_wide = family_scores.pivot(index="method", columns="family", values="n_total_metrics").reset_index()
    total_metric_wide = total_metric_wide.rename(
        columns={fam: f"{fam}_n_total_metrics" for fam in FAMILIES if fam in total_metric_wide.columns}
    )

    # Merge all compact family-level fields.
    family_wide = wide.merge(rank_wide, on="method", how="left")
    family_wide = family_wide.merge(metric_cov_wide, on="method", how="left")
    family_wide = family_wide.merge(value_cov_wide, on="method", how="left")
    family_wide = family_wide.merge(valid_metric_wide, on="method", how="left")
    family_wide = family_wide.merge(total_metric_wide, on="method", how="left")

    return detail, family_scores, family_wide


# -----------------------------
# Accuracy weight sweep
# -----------------------------

def weighted_available_score(row: pd.Series, wD: float, wC: float) -> Tuple[float, float, int, str, str]:
    """
    Compute weighted accuracy score using available family scores only.

    If one family is missing, its weight is skipped and the remaining available
    family weights are renormalized. If both are missing, score is NA.
    """
    vals = {
        "Directionality": row.get("Directionality", np.nan),
        "Consistency": row.get("Consistency", np.nan),
    }
    weights = {"Directionality": wD, "Consistency": wC}

    used = [fam for fam in FAMILIES if pd.notna(vals[fam]) and weights[fam] > 0]
    missing = [fam for fam in FAMILIES if fam not in used]
    available_weight_sum = sum(weights[fam] for fam in used)

    if available_weight_sum <= 0:
        return np.nan, 0.0, 0, "", ";".join(missing)

    score = sum(weights[fam] * vals[fam] for fam in used) / available_weight_sum
    return score, available_weight_sum, len(used), ";".join(used), ";".join(missing)


def compute_accuracy_weight_sweep(family_wide: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute ranks under each D/C weight and summarize final consensus rank."""
    records = []
    for wD, wC in WEIGHT_GRID:
        label = f"D{wD:.1f}_C{wC:.1f}"
        tmp = family_wide.copy()

        computed = tmp.apply(lambda row: weighted_available_score(row, wD, wC), axis=1)
        tmp["accuracy_score"] = [x[0] for x in computed]
        tmp["available_weight_sum"] = [x[1] for x in computed]
        tmp["n_available_families"] = [x[2] for x in computed]
        tmp["used_families"] = [x[3] for x in computed]
        tmp["missing_families"] = [x[4] for x in computed]

        tmp["accuracy_rank"] = tmp["accuracy_score"].rank(
            ascending=False,
            method="average",
            na_option="bottom",
        )
        tmp["weight_label"] = label
        tmp["w_directionality"] = wD
        tmp["w_consistency"] = wC
        records.append(tmp)

    details = pd.concat(records, ignore_index=True)

    summary = (
        details.groupby("method", dropna=False)
        .agg(
            mean_accuracy_rank=("accuracy_rank", "mean"),
            median_accuracy_rank=("accuracy_rank", "median"),
            best_accuracy_rank=("accuracy_rank", "min"),
            worst_accuracy_rank=("accuracy_rank", "max"),
            mean_accuracy_score=("accuracy_score", "mean"),
            median_accuracy_score=("accuracy_score", "median"),
            min_accuracy_score=("accuracy_score", "min"),
            max_accuracy_score=("accuracy_score", "max"),
            n_weight_settings=("weight_label", "nunique"),
            mean_available_weight_sum=("available_weight_sum", "mean"),
            mean_n_available_families=("n_available_families", "mean"),
        )
        .reset_index()
    )

    # Final rank is the rank of mean rank. Smaller is better.
    summary["final_accuracy_rank"] = summary["mean_accuracy_rank"].rank(
        ascending=True,
        method="average",
        na_option="bottom",
    )

    n_methods = summary["method"].nunique()
    if n_methods > 1:
        summary["consensus_score_rank_scaled"] = 1 - (
            summary["mean_accuracy_rank"] - 1
        ) / (n_methods - 1)
    else:
        summary["consensus_score_rank_scaled"] = 1.0
    summary["consensus_score_rank_scaled"] = summary["consensus_score_rank_scaled"].clip(0, 1)

    # Add family scores/ranks/coverage to the final summary.
    family_cols = [
        c
        for c in family_wide.columns
        if c == "method"
        or c in FAMILIES
        or c.endswith("_rank")
        or c.endswith("_metric_coverage")
        or c.endswith("_value_coverage")
        or c.endswith("_n_valid_metrics")
        or c.endswith("_n_total_metrics")
    ]
    summary = summary.merge(family_wide[family_cols], on="method", how="left")
    summary = summary.sort_values(["final_accuracy_rank", "mean_accuracy_rank", "method"]).reset_index(drop=True)

    # Wide rank table by weight.
    rank_wide = details.pivot(index="method", columns="weight_label", values="accuracy_rank").reset_index()
    score_wide = details.pivot(index="method", columns="weight_label", values="accuracy_score").reset_index()

    ordered_weight_cols = [f"D{wD:.1f}_C{wC:.1f}" for wD, wC in WEIGHT_GRID]
    rank_wide = summary[["method", "final_accuracy_rank", "mean_accuracy_rank"]].merge(
        rank_wide[["method"] + [c for c in ordered_weight_cols if c in rank_wide.columns]],
        on="method",
        how="left",
    )
    score_wide = summary[["method", "final_accuracy_rank", "mean_accuracy_rank"]].merge(
        score_wide[["method"] + [c for c in ordered_weight_cols if c in score_wide.columns]],
        on="method",
        how="left",
    )

    return details, summary, rank_wide, score_wide


# -----------------------------
# QC summary
# -----------------------------

def make_qc_summary(df: pd.DataFrame, metric_scores: pd.DataFrame, family_scores: pd.DataFrame) -> pd.DataFrame:
    """Build a compact QC table to verify missingness and coverage."""
    rows = []

    rows.append({"section": "long_table", "item": "rows_total", "value": len(df)})
    rows.append({"section": "long_table", "item": "valid_z_rows", "value": int(df["z_score"].notna().sum())})
    rows.append({"section": "long_table", "item": "zero_as_missing_rows", "value": int(df["zero_as_missing"].sum())})
    rows.append({"section": "long_table", "item": "valid_zero_exception_rows", "value": int(df["valid_zero_exception"].sum())})
    rows.append({"section": "long_table", "item": "legacy_G32_invalid_rows", "value": int(df["legacy_G32_invalid_method"].sum())})

    for metric, sub in df.groupby("metric"):
        rows.append({
            "section": "metric_raw_coverage",
            "item": metric,
            "value": f"{sub['z_score'].notna().sum()}/{len(sub)} ({sub['z_score'].notna().mean():.3f})",
        })

    for metric, sub in metric_scores.groupby("metric"):
        rows.append({
            "section": "method_metric_scores",
            "item": metric,
            "value": f"methods_with_score={sub['metric_score'].notna().sum()}; methods_total={sub['method'].nunique()}",
        })

    for family, sub in family_scores.groupby("family"):
        rows.append({
            "section": "family_scores",
            "item": family,
            "value": f"methods_with_score={sub['family_score'].notna().sum()}; methods_total={sub['method'].nunique()}",
        })

    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------

def run_pipeline(input_dir: str | Path, output_dir: str | Path) -> None:
    input_dir = resolve_input_dir(input_dir)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input directory : {input_dir}")
    print(f"[INFO] Output directory: {output_dir}")

    raw_long = read_all_metrics(input_dir)
    cleaned = clean_values(raw_long)
    zscored = add_metric_zscores(cleaned)

    metric_scores = compute_metric_scores(zscored)
    metric_scores_full = complete_method_metric_grid(metric_scores)

    family_detail, family_scores_long, family_scores_wide = compute_family_scores(metric_scores_full)
    sweep_details, rank_summary, rank_by_weight, score_by_weight = compute_accuracy_weight_sweep(family_scores_wide)
    qc_summary = make_qc_summary(zscored, metric_scores_full, family_scores_long)

    # Compact outputs only.
    metric_scores_full.to_csv(output_dir / "metric_scores.csv", index=False)
    family_detail.to_csv(output_dir / "family_detail.csv", index=False)
    family_scores_long.to_csv(output_dir / "family_scores_long.csv", index=False)
    family_scores_wide.to_csv(output_dir / "family_scores_wide.csv", index=False)
    sweep_details.to_csv(output_dir / "accuracy_weight_sweep_details.csv", index=False)
    rank_summary.to_csv(output_dir / "accuracy_rank_summary.csv", index=False)
    rank_by_weight.to_csv(output_dir / "rank_by_weight_wide.csv", index=False)
    score_by_weight.to_csv(output_dir / "score_by_weight_wide.csv", index=False)
    qc_summary.to_csv(output_dir / "qc_summary.csv", index=False)

    # A short console preview.
    preview_cols = [
        "method",
        "final_accuracy_rank",
        "mean_accuracy_rank",
        "best_accuracy_rank",
        "worst_accuracy_rank",
        "mean_accuracy_score",
        "Directionality",
        "Consistency",
        "Directionality_rank",
        "Consistency_rank",
        "consensus_score_rank_scaled",
    ]
    preview_cols = [c for c in preview_cols if c in rank_summary.columns]
    print("\n[INFO] Final accuracy rank preview:")
    print(rank_summary[preview_cols].head(25).to_string(index=False))
    print(f"\n[INFO] Done. Results written to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-dataset-weight accuracy aggregation test for velocity benchmark metrics."
    )
    parser.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Input accuracy0609 directory. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input_dir, args.output_dir)
