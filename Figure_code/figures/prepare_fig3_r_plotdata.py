#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare current benchmark results into R-compatible Fig3 input tables.

This script does NOT draw figures. It only converts the current data into the
wide CSV format expected by the R plotting script.

Default output:
    PlotData/Fig3_R/PlotData

Main outputs:
    method_order.csv

    Accuracy/real/Consistency_Score.csv
    Accuracy/real/Velocity_Angle.csv
    Accuracy/real/CBDir.csv
    Accuracy/real/Transition_score.csv
    Accuracy/real/ICCoh.csv
    Accuracy/real/Peak_location.csv

    Accuracy/sim/Consistency_Score.csv
    Accuracy/sim/Velocity_Angle.csv
    Accuracy/sim/CBDir.csv
    Accuracy/sim/Transition_score.csv
    Accuracy/sim/ICCoh.csv
    Accuracy/sim/Groundtruth_correlation.csv
    Accuracy/sim/Peak_location.csv

    Scalability/docker_speed_dim_means.csv
    Scalability/docker_memory_dim_means.csv

    Stability/Downsampling.csv

    Usability/Velocity_Usability_Detailed_Subscore.csv

Notes:
    - TopoVelo is not included unless it appears in final order and data.
    - For accuracy matrices, zero values are treated as missing by default,
      because in the current benchmark zeros generally represent failed/missing
      calculations. Use --keep_zero_values to keep zeros.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_FINAL_ORDER = Path("PlotData/Results/overall/final_overall_rank_for_plot.csv")
DEFAULT_OUTDIR = Path("PlotData/Fig3_R/PlotData")

DEFAULT_ACCURACY_RESULTS = Path("PlotData/Results/accuracy")
DEFAULT_ACCURACY_DATA = Path("PlotData/accuracy")

DEFAULT_SCALABILITY_RESULTS = Path("PlotData/Results/scalability")
DEFAULT_SCALABILITY_XLSX = Path("PlotData/scalability/Docker_performance_0605.xlsx")
DEFAULT_HVG_2K5K = Path("PlotData/scalability/HVG_2k5k_Docker_performance_0519.xlsx")

DEFAULT_STABILITY_DOWNSAMPLING = Path("PlotData/Results/stability/01_downsampling_component_scores.csv")

DEFAULT_USABILITY_XLSX = Path("PlotData/usability/Velocity_Usability_0605.xlsx")

ACCURACY_METRICS = {
    "angle_consistency": {
        "title": "Angle",
        "real_patterns": ["scRNA_angle_consistency", "angle_consistency"],
        "sim_patterns": ["SIM_angle_consistency", "angle_consistency"],
    },
    "CBDir": {
        "title": "CBDir",
        "real_patterns": ["scRNA_CBDir", "CBDir"],
        "sim_patterns": ["SIM_CBDir", "CBDir"],
    },
    "transition_score": {
        "title": "Transition",
        "real_patterns": ["scRNA_transition_score", "transition_score"],
        "sim_patterns": ["SIM_transition_score", "transition_score"],
    },
    "groundtruth_correlation": {
        "title": "Ground truth corr.",
        "real_patterns": [],
        "sim_patterns": ["SIM_groundtruth_correlation", "groundtruth_correlation"],
    },
    "ICCoh": {
        "title": "ICCoh",
        "real_patterns": ["scRNA_ICCoh", "ICCoh"],
        "sim_patterns": ["SIM_ICCoh", "ICCoh"],
    },
    "peak_location": {
        "title": "Peak location",
        "real_patterns": ["scRNA_peak_location", "peak_location"],
        "sim_patterns": ["SIM_peak_location", "peak_location"],
    },
}

REAL_METRIC_ORDER = ["angle_consistency", "CBDir", "transition_score", "ICCoh", "peak_location"]
SIM_METRIC_ORDER = ["angle_consistency", "CBDir", "transition_score", "groundtruth_correlation", "ICCoh", "peak_location"]

SCALABILITY_SIZE_LABELS = ["1k×1k", "1k×10k", "1k×20k", "10k×1k", "200k×1k", "2k HVG", "5k HVG"]
MAIN_SIZE_RULES = [
    ("1k×1k", 1000, 1000),
    ("1k×10k", 1000, 10000),
    ("1k×20k", 1000, 20000),
    ("10k×1k", 10000, 1000),
    ("200k×1k", 200000, 1000),
]


def parse_args():
    p = argparse.ArgumentParser(description="Prepare R-compatible Fig3 input tables.")
    p.add_argument("--final_order_csv", default=str(DEFAULT_FINAL_ORDER))
    p.add_argument("--output_dir", default=str(DEFAULT_OUTDIR))

    p.add_argument("--accuracy_results_dir", default=str(DEFAULT_ACCURACY_RESULTS))
    p.add_argument("--accuracy_data_dir", default=str(DEFAULT_ACCURACY_DATA))
    p.add_argument("--keep_zero_values", action="store_true")

    p.add_argument("--scalability_results_dir", default=str(DEFAULT_SCALABILITY_RESULTS))
    p.add_argument("--scalability_xlsx", default=str(DEFAULT_SCALABILITY_XLSX))
    p.add_argument("--hvg_2k5k_xlsx", default=str(DEFAULT_HVG_2K5K))

    p.add_argument("--stability_downsampling_csv", default=str(DEFAULT_STABILITY_DOWNSAMPLING))
    p.add_argument("--usability_xlsx", default=str(DEFAULT_USABILITY_XLSX))
    return p.parse_args()


def normalize_method_name(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip())


def method_key(x):
    return re.sub(r"[^a-z0-9]+", "", normalize_method_name(x).lower())


def canonical_method_name(x):
    name = normalize_method_name(x)
    key = method_key(name)
    if key in {"scrnakinetics", "scrnakinetic"}:
        return "scRNAkinetics"
    if key == "regionvelocity":
        return "Region Velocity"
    if key == "topovelo":
        return "TopoVelo"
    if key == "scvelodynamic":
        return "scVelo dynamical"
    if key == "scvelostochastic":
        return "scVelo stochastic"
    if key == "pyrovelocity":
        return "Pyro-Velocity"
    return name


def safe_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "none", "null"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def read_base_method_order(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "final_overall_rank" in df.columns:
        df = df.sort_values("final_overall_rank")
    if "method" not in df.columns:
        raise ValueError(f"`method` column not found in {path}")
    methods = [canonical_method_name(x) for x in df["method"].tolist()]
    return [m for m in dict.fromkeys(methods) if m not in {"Region Velocity", "TopoVelo"}]


def find_method_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if method_key(c) in {"method", "methods", "tool", "tools"}:
            return c
    return df.columns[0]


def finalize_method_order(base_order: list[str], available_methods: set[str]) -> list[str]:
    out = [m for m in base_order if m in available_methods and m != "Region Velocity"]
    extra = sorted([m for m in available_methods if m not in set(out) and m not in {"Region Velocity", ""}])
    # Keep unexpected extra methods before Region Velocity but after final-rank methods.
    out.extend(extra)
    if "Region Velocity" in available_methods:
        out.append("Region Velocity")
    return out


def write_wide_matrix(df: pd.DataFrame, out_path: Path, method_order: list[str]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        return False
    d = df.copy()
    d["Method"] = d["Method"].map(canonical_method_name)
    d = d[d["Method"].isin(method_order)].copy()
    d["Method"] = pd.Categorical(d["Method"], categories=method_order, ordered=True)
    d = d.sort_values("Method")
    d.to_csv(out_path, index=False)
    return True


def metric_from_filename(path: Path, data_type: str) -> str:
    """Match the original Python Fig3 logic exactly."""
    stem = path.stem
    if data_type == "real":
        return stem.replace("scRNA_", "", 1)
    if data_type == "sim":
        return stem.replace("SIM_", "", 1)
    return stem


def read_raw_accuracy_data(data_dir: Path, keep_zero_values: bool) -> pd.DataFrame:
    """Read accuracy matrices using the same logic as the original Python Fig3 code."""
    parts = []

    for data_type, subdir in [("real", "real"), ("sim", "sim")]:
        folder = Path(data_dir) / subdir
        if not folder.exists():
            continue

        allowed_metrics = set(REAL_METRIC_ORDER if data_type == "real" else SIM_METRIC_ORDER)

        for path in sorted(folder.glob("*.csv")):
            metric = metric_from_filename(path, data_type)
            if metric not in allowed_metrics:
                continue

            df = pd.read_csv(path)

            # Original files usually use Method, but accept the first column as fallback.
            if "Method" in df.columns:
                method_col = "Method"
            else:
                method_col = find_method_column(df)

            long = df.melt(id_vars=[method_col], var_name="dataset_id", value_name="value_raw")
            long = long.rename(columns={method_col: "Method"})
            long["Method"] = long["Method"].map(canonical_method_name)
            long["data_type"] = data_type
            long["metric"] = metric
            long["value_raw"] = long["value_raw"].map(safe_numeric)

            if not keep_zero_values:
                # Same convention as the current benchmark: most zero cells represent failed/missing values.
                true_zero = (
                    (data_type == "real") &
                    (metric == "peak_location") &
                    (long["Method"] == "TopicVelo") &
                    (long["dataset_id"] == "4_Mm_visual_cortex")
                )
                zero_as_missing = (long["value_raw"] == 0) & (~true_zero)
                long.loc[zero_as_missing, "value_raw"] = np.nan

            # Historical cleanup for G_32_Mm_embryos if present.
            allowed_g32 = {"InterVelo", "PhyloVelo", "VeloVAE", "DeepVelo"}
            g32_invalid = (
                (data_type == "real") &
                (long["dataset_id"] == "G_32_Mm_embryos") &
                (~long["Method"].isin(allowed_g32))
            )
            long.loc[g32_invalid, "value_raw"] = np.nan

            parts.append(long)

    if not parts:
        raise FileNotFoundError(f"No raw accuracy metric CSVs were found under {data_dir}/real and {data_dir}/sim")

    return pd.concat(parts, ignore_index=True)


def read_long_from_results(results_dir: Path) -> Optional[pd.DataFrame]:
    """Use long result table only when it matches the original Python Fig3 expectations."""
    candidates = [
        "01_long_zscore.csv",
        "long_zscore.csv",
        "long_zscore.csv",
        "accuracy_long_zscore.csv",
        "accuracy_metric_long_zscores.csv",
    ]

    for name in candidates:
        path = Path(results_dir) / name
        if not path.exists():
            continue

        df = pd.read_csv(path)
        required = {"method", "metric", "data_type"}
        if not required.issubset(df.columns):
            continue

        if "value_raw" not in df.columns:
            if "value_clean" in df.columns:
                df["value_raw"] = df["value_clean"]
            elif "value" in df.columns:
                df["value_raw"] = df["value"]
            elif "metric_value" in df.columns:
                df["value_raw"] = df["metric_value"]
            else:
                continue

        out = df.copy()
        out["Method"] = out["method"].map(canonical_method_name)
        out["metric"] = out["metric"].astype(str)
        out["data_type"] = out["data_type"].astype(str).str.lower()
        out["data_type"] = out["data_type"].replace({"simulated": "sim"})
        out["value_raw"] = out["value_raw"].map(safe_numeric)
        if "dataset_id" not in out.columns:
            if "dataset" in out.columns:
                out["dataset_id"] = out["dataset"]
            else:
                out["dataset_id"] = np.arange(len(out))
        return out[["Method", "data_type", "metric", "dataset_id", "value_raw"]]

    return None


def prepare_accuracy(base_order, outdir, results_dir, data_dir, keep_zero_values):
    """Prepare accuracy matrices for the R plotting script.

    This follows the original Python Fig3 logic:
        - use a long z-score/result table from Accuracy/Results if available;
        - otherwise fallback to raw matrices from Accuracy/Data;
        - only use REAL_METRIC_ORDER and SIM_METRIC_ORDER.
    """
    long_df = read_long_from_results(results_dir)
    source_mode = "Accuracy/Results long table"

    if long_df is None:
        long_df = read_raw_accuracy_data(data_dir, keep_zero_values=keep_zero_values)
        source_mode = "Accuracy/Data raw matrices"

    long_df["Method"] = long_df["Method"].map(canonical_method_name)
    long_df["data_type"] = long_df["data_type"].astype(str).str.lower()
    long_df["data_type"] = long_df["data_type"].replace({"simulated": "sim"})
    long_df["metric"] = long_df["metric"].astype(str)

    available_methods = set(long_df["Method"].dropna().unique().tolist())
    written = []
    missing = []

    for data_type, metric_order in [("real", REAL_METRIC_ORDER), ("sim", SIM_METRIC_ORDER)]:
        for metric in metric_order:
            sub = long_df[(long_df["data_type"] == data_type) & (long_df["metric"] == metric)].copy()
            if sub.empty:
                missing.append({"data_type": data_type, "metric": metric, "reason": "not_present_after_python_logic", "source_mode": source_mode})
                continue

            mat = sub.pivot_table(index="Method", columns="dataset_id", values="value_raw", aggfunc="mean")
            order = finalize_method_order(base_order, set(mat.index))
            mat = mat.reindex(order)
            mat.insert(0, "Method", mat.index)

            out_path = outdir / "Accuracy" / data_type / f"{metric}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mat.to_csv(out_path, index=False)

            written.append({
                "data_type": data_type,
                "metric": metric,
                "source_mode": source_mode,
                "output": str(out_path),
                "n_methods": int(mat["Method"].notna().sum()),
                "n_datasets": int(mat.shape[1] - 1),
            })

    return available_methods, pd.DataFrame(written), pd.DataFrame(missing)




def read_results_long_scalability(results_dir: Path) -> Optional[pd.DataFrame]:
    for name in ["02_scalability_time_memory_dataset_scores.csv", "02_scalability_dataset_level_scores.csv"]:
        p = results_dir / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if {"method", "dataset", "metric", "value"}.issubset(df.columns):
            out = df[["method", "dataset", "metric", "value"]].copy()
            out["method"] = out["method"].map(canonical_method_name)
            out["metric"] = out["metric"].astype(str).str.lower()
            out["value"] = out["value"].map(safe_numeric)
            out["size_label_override"] = None
            return out
        if {"method", "dataset", "time_value", "memory_value"}.issubset(df.columns):
            parts = []
            for metric, col in [("time", "time_value"), ("memory", "memory_value")]:
                tmp = df[["method", "dataset", col]].copy().rename(columns={col: "value"})
                tmp["metric"] = metric
                parts.append(tmp)
            out = pd.concat(parts, ignore_index=True)
            out["method"] = out["method"].map(canonical_method_name)
            out["value"] = out["value"].map(safe_numeric)
            out["size_label_override"] = None
            return out[["method", "dataset", "metric", "value", "size_label_override"]]
    return None


def is_unit_row(x):
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return bool(re.match(r"^unit\s*[:：]", s)) or s in {"unit", "unit seconds", "unit minutes"}


def read_standard_scalability_xlsx(path: Path) -> pd.DataFrame:
    parts = []
    for metric in ["time", "memory"]:
        try:
            raw = pd.read_excel(path, sheet_name=metric, header=None)
        except Exception:
            continue
        header_i = None
        for i in range(min(len(raw), 30)):
            if str(raw.iloc[i, 0]).strip().lower() == "method":
                header_i = i
                break
        if header_i is None:
            continue
        df = pd.read_excel(path, sheet_name=metric, header=header_i)
        method_col = find_method_column(df)
        df = df[df[method_col].notna()].copy()
        df = df[~df[method_col].map(is_unit_row)].copy()
        long = df.melt(id_vars=[method_col], var_name="dataset", value_name="raw_value").rename(columns={method_col: "method"})
        long["method"] = long["method"].map(canonical_method_name)
        long["metric"] = metric
        long["value"] = long["raw_value"].map(safe_numeric)
        long["size_label_override"] = None
        parts.append(long[["method", "dataset", "metric", "value", "size_label_override"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def hvg_block_label(x):
    s = str(x)
    m = re.search(r"(\d+)\s*velocity\s*genes", s, flags=re.I)
    if not m:
        return None
    n = int(m.group(1))
    if n == 2000:
        return "2k HVG"
    if n == 5000:
        return "5k HVG"
    return f"{n} HVG"



def resolve_hvg_2k5k_path(path: Path) -> Optional[Path]:
    """Find the HVG 2k/5k workbook robustly.

    The file has moved across a few project folders. The plotting data prep
    should not silently leave the 2k/5k columns empty just because the default
    path is stale.
    """
    candidates = [
        path,
        Path("PlotData/Fig3/Scalability/2k5k/HVG_2k5k_Docker_performance_0519.xlsx"),
        Path("PlotData/Fig3/Scalability/HVG_2k5k_Docker_performance_0519.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            return p

    # Last resort: bounded recursive search in likely project areas.
    search_roots = [
        Path("PlotData/Fig3"),
    ]
    for root in search_roots:
        if root.exists():
            hits = sorted(root.rglob("HVG_2k5k_Docker_performance_0519.xlsx"))
            if hits:
                return hits[0]
    return None

def read_hvg_2k5k(path: Path) -> pd.DataFrame:
    resolved = resolve_hvg_2k5k_path(path)
    if resolved is None:
        print(f"[WARN] HVG 2k/5k workbook not found. Requested path: {path}")
        return pd.DataFrame(columns=["method", "dataset", "metric", "value", "size_label_override"])
    print(f"[INFO] Using HVG 2k/5k workbook: {resolved}")
    path = resolved
    parts = []
    for metric in ["time", "memory"]:
        try:
            raw = pd.read_excel(path, sheet_name=metric, header=None)
        except Exception:
            continue
        i = 0
        while i < len(raw):
            label = hvg_block_label(raw.iloc[i, 0])
            if label is None:
                i += 1
                continue
            header_i = i + 1
            if header_i >= len(raw):
                break
            headers = raw.iloc[header_i].tolist()
            if str(headers[0]).strip().lower() != "method":
                i += 1
                continue
            rows = []
            j = header_i + 1
            while j < len(raw):
                first = raw.iloc[j, 0]
                first_s = "" if pd.isna(first) else str(first).strip()
                # Do not use startswith("unit"): UniTVelo starts with "Uni".
                if first_s == "" or is_unit_row(first_s) or hvg_block_label(first_s) is not None:
                    break
                rows.append(raw.iloc[j].tolist())
                j += 1
            if rows:
                block = pd.DataFrame(rows, columns=headers)
                method_col = headers[0]
                block = block[block[method_col].notna()].copy()
                block = block[~block[method_col].map(is_unit_row)].copy()
                long = block.melt(id_vars=[method_col], var_name="dataset", value_name="raw_value").rename(columns={method_col: "method"})
                long["method"] = long["method"].map(canonical_method_name)
                long["metric"] = metric
                long["value"] = long["raw_value"].map(safe_numeric)
                long["size_label_override"] = label
                parts.append(long[["method", "dataset", "metric", "value", "size_label_override"]])
            i = max(j, i + 1)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def parse_size(dataset):
    s = str(dataset)
    m_cell = re.search(r"cell[_\- ]*(\d+)", s, flags=re.I)
    m_gene = re.search(r"gene[_\- ]*(\d+)", s, flags=re.I)
    if m_cell and m_gene:
        return int(m_cell.group(1)), int(m_gene.group(1))
    ss = s.lower().replace("×", "x").replace("*", "x")
    m = re.search(r"(\d+)\s*k\s*x\s*(\d+)\s*k", ss)
    if m:
        return int(m.group(1)) * 1000, int(m.group(2)) * 1000
    return None


def assign_size_label(row):
    override = row.get("size_label_override", None)
    if isinstance(override, str) and override.strip():
        return override
    parsed = parse_size(row["dataset"])
    if parsed is None:
        return None
    cell, gene = parsed
    for label, c, g in MAIN_SIZE_RULES:
        if cell == c and gene == g:
            return label
    return None


def prepare_scalability(base_order, outdir, results_dir, xlsx_path, hvg_path):
    d = read_results_long_scalability(results_dir)
    if d is None or d.empty:
        d = read_standard_scalability_xlsx(xlsx_path)
    hvg = read_hvg_2k5k(hvg_path)
    out_scal = outdir / "Scalability"
    out_scal.mkdir(parents=True, exist_ok=True)
    if not hvg.empty:
        hvg.to_csv(out_scal / "hvg_2k5k_raw_long_values.csv", index=False)
        hvg_qc = (
            hvg.groupby(["metric", "size_label_override"], as_index=False)
            .agg(
                n_methods=("method", lambda x: int(pd.Series(x).dropna().nunique())),
                n_numeric_values=("value", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
                n_total_values=("value", "size"),
            )
        )
        hvg_qc.to_csv(out_scal / "hvg_2k5k_parse_qc.csv", index=False)
        d = pd.concat([d, hvg], ignore_index=True, sort=False)

    d["method"] = d["method"].map(canonical_method_name)
    d["metric"] = d["metric"].astype(str).str.lower()
    d["value"] = d["value"].map(safe_numeric)
    d["size_label"] = d.apply(assign_size_label, axis=1)
    d = d[d["metric"].isin(["time", "memory"]) & d["size_label"].isin(SCALABILITY_SIZE_LABELS)].copy()

    # All time values in the current scalability files are seconds before plotting.
    # Convert to minutes for the R heatmap.
    d.loc[d["metric"] == "time", "value"] = d.loc[d["metric"] == "time", "value"] / 60.0

    summary = (
        d.groupby(["method", "metric", "size_label"], as_index=False)
        .agg(mean_value=("value", "mean"), n_valid=("value", lambda x: int(pd.notna(x).sum())), n_total=("value", "size"))
    )

    available = set(summary["method"].dropna())
    order = finalize_method_order(base_order, available)

    out_scal = outdir / "Scalability"
    out_scal.mkdir(parents=True, exist_ok=True)

    for metric, fname in [("time", "docker_speed_dim_means.csv"), ("memory", "docker_memory_dim_means.csv")]:
        mat = (
            summary[summary["metric"] == metric]
            .pivot(index="method", columns="size_label", values="mean_value")
            .reindex(index=order, columns=SCALABILITY_SIZE_LABELS)
        )
        mat.insert(0, "Method", mat.index)
        mat.to_csv(out_scal / fname, index=False)

    summary.to_csv(out_scal / "scalability_selected_size_summary.csv", index=False)
    return available


def prepare_stability(base_order, outdir, path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    method_col = find_method_column(df)
    d = df.copy()
    d["Method"] = d[method_col].map(canonical_method_name)

    candidates = []
    for c in d.columns:
        cl = str(c).lower()
        if c == method_col or c == "Method":
            continue
        if any(x in cl for x in ["cellsub", "genesub", "downsampling"]):
            if pd.to_numeric(d[c], errors="coerce").notna().any():
                candidates.append(c)
    if not candidates:
        numeric = [c for c in d.columns if c not in {method_col, "Method"} and pd.to_numeric(d[c], errors="coerce").notna().any()]
        candidates = numeric[:1]

    out = d[["Method"] + candidates].copy()
    out.columns = ["Method"] + [f"Downsampling_{i+1}" for i in range(len(candidates))]
    available = set(out["Method"].dropna())
    order = finalize_method_order(base_order, available)
    out["Method"] = pd.Categorical(out["Method"], categories=order, ordered=True)
    out = out.sort_values("Method")
    out_path = outdir / "Stability" / "Downsampling.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return available


def find_col(columns, candidates):
    norm = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if key in norm:
            return norm[key]
    return None


def prepare_usability(base_order, outdir, xlsx_path):
    if not xlsx_path.exists():
        return set()
    df = pd.read_excel(xlsx_path, sheet_name="Detailed_Subscore")
    method_col = find_col(df.columns, ["method", "Method", "RNA velocity Platform"])
    category_col = find_col(df.columns, ["category", "subcategory", "Category", "Subcategory"])
    score_col = find_col(df.columns, ["Category score", "category_score", "score"])
    if method_col is None or category_col is None or score_col is None:
        raise ValueError(f"Cannot identify Method/Category/Category score columns in {xlsx_path}. Columns: {list(df.columns)}")

    d = pd.DataFrame({
        "Method": df[method_col].map(canonical_method_name),
        "Metric": df[category_col].astype(str).str.strip(),
        "Score": pd.to_numeric(df[score_col], errors="coerce"),
    })
    # Normalize common category names, keeping the original high-level meaning.
    d["Metric"] = d["Metric"].replace({"Install-friendly": "Installation", "Usage guidance": "Guidance"})

    wide = d.pivot_table(index="Method", columns="Metric", values="Score", aggfunc="mean")
    available = set(wide.index)
    order = finalize_method_order(base_order, available)
    wide = wide.reindex(order)

    preferred = ["Installation", "Guidance", "Maintenance", "Code quality", "Paper support", "Feedback"]
    cols = [c for c in preferred if c in wide.columns] + [c for c in wide.columns if c not in preferred]
    wide = wide[cols]
    wide.insert(0, "Method", wide.index)

    out_path = outdir / "Usability" / "Velocity_Usability_Detailed_Subscore.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, index=False)
    return available


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_order = read_base_method_order(Path(args.final_order_csv))

    all_available = set()

    acc_available, acc_written, acc_missing = prepare_accuracy(
        base_order=base_order,
        outdir=outdir,
        results_dir=Path(args.accuracy_results_dir),
        data_dir=Path(args.accuracy_data_dir),
        keep_zero_values=args.keep_zero_values,
    )
    all_available.update(acc_available)

    all_available.update(prepare_scalability(
        base_order=base_order,
        outdir=outdir,
        results_dir=Path(args.scalability_results_dir),
        xlsx_path=Path(args.scalability_xlsx),
        hvg_path=Path(args.hvg_2k5k_xlsx),
    ))

    all_available.update(prepare_stability(
        base_order=base_order,
        outdir=outdir,
        path=Path(args.stability_downsampling_csv),
    ))

    all_available.update(prepare_usability(
        base_order=base_order,
        outdir=outdir,
        xlsx_path=Path(args.usability_xlsx),
    ))

    method_order = finalize_method_order(base_order, all_available)
    pd.DataFrame({"Method": method_order}).to_csv(outdir / "method_order.csv", index=False)

    acc_written.to_csv(outdir / "accuracy_conversion_manifest.csv", index=False)
    acc_missing.to_csv(outdir / "accuracy_missing_metrics.csv", index=False)

    print("Done.")
    print(f"R-compatible PlotData written to: {outdir}")
    print(f"Method order: {outdir / 'method_order.csv'}")
    print(f"Accuracy manifest: {outdir / 'accuracy_conversion_manifest.csv'}")
    if not acc_missing.empty:
        print(f"Some accuracy metrics were not found. See: {outdir / 'accuracy_missing_metrics.csv'}")


if __name__ == "__main__":
    main()
