#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute overall benchmark ranking from four reversed-rank tables.

Default input directory:
    PlotData/Results/reversed_rank

Expected input files:
    accuracy_rank.csv
    scalability_rank.csv
    stability_rank.csv
    usability_rank.csv

Default output directory:
    PlotData/Results/reversed_rank/Results
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("PlotData/Results/reversed_rank")
DEFAULT_OUTPUT_DIR = Path("PlotData/Results/reversed_rank/Results")

DEFAULT_FILES = {
    "accuracy": "accuracy_rank.csv",
    "scalability": "scalability_rank.csv",
    "stability": "stability_rank.csv",
    "usability": "usability_rank.csv",
}

DEFAULT_WEIGHTS = {
    "accuracy": 0.60,
    "scalability": 0.15,
    "stability": 0.15,
    "usability": 0.10,
}

# Default methods excluded from overall ranking.
# Matching is performed after canonicalization, so case/spacing variants are removed.
DEFAULT_EXCLUDE_METHODS = [
    "Region Velocity",
    "TopoVelo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute overall benchmark rank from four reversed-rank tables."
    )
    parser.add_argument(
        "--input_dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing four reversed-rank CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write overall-ranking outputs.",
    )
    parser.add_argument("--accuracy_csv", default=None)
    parser.add_argument("--scalability_csv", default=None)
    parser.add_argument("--stability_csv", default=None)
    parser.add_argument("--usability_csv", default=None)
    parser.add_argument(
        "--method_col",
        default="method",
        help="Method column name in input CSVs.",
    )
    parser.add_argument(
        "--rank_col",
        default="reversed_rank",
        help="Column containing reversed rank in input CSVs.",
    )
    parser.add_argument(
        "--missing_policy",
        choices=["strict", "renormalize", "zero"],
        default="strict",
        help=(
            "How to compute OverallRank when a component is missing. "
            "strict: require all four components; renormalize: use available "
            "components and re-normalize weights; zero: missing contributes 0."
        ),
    )
    parser.add_argument("--w_accuracy", type=float, default=DEFAULT_WEIGHTS["accuracy"])
    parser.add_argument("--w_scalability", type=float, default=DEFAULT_WEIGHTS["scalability"])
    parser.add_argument("--w_stability", type=float, default=DEFAULT_WEIGHTS["stability"])
    parser.add_argument("--w_usability", type=float, default=DEFAULT_WEIGHTS["usability"])
    parser.add_argument(
        "--exclude_methods",
        default=";".join(DEFAULT_EXCLUDE_METHODS),
        help=(
            "Comma/semicolon-separated methods to exclude from overall ranking. "
            "Default: Region Velocity;TopoVelo"
        ),
    )
    parser.add_argument(
        "--method_alias_csv",
        default=None,
        help=(
            "Optional CSV with columns `from` and `to` for additional method-name "
            "harmonization before merging."
        ),
    )
    return parser.parse_args()


def normalize_method_name(x: object) -> str:
    """Basic method-name cleanup."""
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip())


def method_key(x: object) -> str:
    """Case-/spacing-/punctuation-insensitive key for matching."""
    return re.sub(r"[^a-z0-9]+", "", normalize_method_name(x).lower())


def canonical_method_name(x: object) -> str:
    """Canonicalize known method-name variants."""
    name = normalize_method_name(x)
    key = method_key(name)

    # Fix scRNAkinetics case variants before merging.
    if key in {"scrnakinetics", "scrnakinetic"}:
        return "scRNAkinetics"

    # Fix Region Velocity case/spacing variants.
    if key == "regionvelocity":
        return "Region Velocity"

    # Fix TopoVelo variants.
    if key == "topovelo":
        return "TopoVelo"

    return name


def split_method_list(s: Optional[str]) -> list[str]:
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in re.split(r"[;,]", s) if x.strip()]


def load_alias_map(path: Optional[str]) -> Dict[str, str]:
    """Load optional user-provided alias map."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Method alias file not found: {p}")
    df = pd.read_csv(p)
    if "from" not in df.columns or "to" not in df.columns:
        raise ValueError("Alias CSV must contain columns: from, to")
    out = {}
    for _, row in df.iterrows():
        src = canonical_method_name(row["from"])
        dst = canonical_method_name(row["to"])
        if src and dst:
            out[src] = dst
    return out


def apply_alias(method: str, alias_map: Dict[str, str]) -> str:
    return alias_map.get(method, method)


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
    alias_map: Dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load one reversed-rank table.

    Returns:
        component_table:
            method + R_component after canonicalization and duplicate collapse.
        duplicate_report:
            rows showing method names collapsed after canonicalization.
    """
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
    raw["method"] = raw["method"].map(lambda x: apply_alias(x, alias_map))
    raw[f"R_{component}"] = pd.to_numeric(df[rank_col], errors="coerce")

    raw = raw.loc[raw["method"] != ""].copy()

    # Duplicate report: useful for checking scRNAkinetics / scRNAKinetics collapse.
    dup_report = (
        raw.groupby("method", as_index=False)
        .agg(
            n_rows=("raw_method", "size"),
            raw_method_names=("raw_method", lambda x: "; ".join(sorted(set(map(str, x))))),
            **{f"R_{component}_values": (f"R_{component}", lambda x: "; ".join(map(str, x.tolist())))},
        )
    )
    dup_report = dup_report.loc[dup_report["n_rows"] > 1].copy()
    dup_report.insert(0, "component", component)

    # Collapse duplicate canonical names by averaging reversed ranks.
    # For scRNAkinetics, this makes scRNAkinetics / scRNAKinetics one method.
    out = (
        raw.groupby("method", as_index=False)
        .agg(**{f"R_{component}": (f"R_{component}", "mean")})
    )

    return out, dup_report


def merge_components(component_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for _, df in component_tables.items():
        if merged is None:
            merged = df.copy()
        else:
            merged = merged.merge(df, on="method", how="outer")
    if merged is None:
        raise ValueError("No component tables were loaded.")
    return merged


def apply_exclusion(
    merged: pd.DataFrame,
    exclude_methods: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude methods from overall ranking after canonicalization."""
    if not exclude_methods:
        empty = merged.iloc[0:0].copy()
        return merged.copy(), empty

    exclude_keys = {method_key(canonical_method_name(x)) for x in exclude_methods}
    keys = merged["method"].map(method_key)
    mask = keys.isin(exclude_keys)

    excluded = merged.loc[mask].copy()
    excluded["excluded_reason"] = "excluded_from_overall"
    kept = merged.loc[~mask].copy()
    return kept, excluded


def compute_overall(
    merged: pd.DataFrame,
    weights: Dict[str, float],
    missing_policy: str,
) -> pd.DataFrame:
    df = merged.copy()

    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0):
        weights = {k: v / weight_sum for k, v in weights.items()}

    component_cols = {comp: f"R_{comp}" for comp in weights}

    for col in component_cols.values():
        if col not in df.columns:
            df[col] = np.nan

    df["n_available_components"] = 0
    df["available_weight_sum"] = 0.0
    missing_component_arrays = []

    for comp, col in component_cols.items():
        available = df[col].notna()
        df["n_available_components"] += available.astype(int)
        df["available_weight_sum"] += available.astype(float) * weights[comp]
        missing_component_arrays.append(np.where(available, "", comp))

    miss_df = pd.DataFrame(
        {comp: arr for comp, arr in zip(weights.keys(), missing_component_arrays)}
    )
    df["missing_components"] = miss_df.apply(
        lambda row: ";".join([x for x in row.tolist() if x]),
        axis=1,
    )

    if missing_policy == "strict":
        score = pd.Series(0.0, index=df.index, dtype=float)
        for comp, col in component_cols.items():
            score += weights[comp] * df[col]
        score[df["n_available_components"] < len(weights)] = np.nan

    elif missing_policy == "renormalize":
        numerator = pd.Series(0.0, index=df.index, dtype=float)
        for comp, col in component_cols.items():
            numerator += weights[comp] * df[col].fillna(0)
        denom = df["available_weight_sum"].replace(0, np.nan)
        score = numerator / denom

    elif missing_policy == "zero":
        score = pd.Series(0.0, index=df.index, dtype=float)
        for comp, col in component_cols.items():
            score += weights[comp] * df[col].fillna(0)

    else:
        raise ValueError(f"Unknown missing_policy: {missing_policy}")

    df["OverallRank"] = score

    # Average rank for ties.
    df["overall_rank_tie_average"] = df["OverallRank"].rank(
        ascending=False,
        method="average",
        na_option="bottom",
    )

    # Deterministic integer order.
    valid = df.loc[df["OverallRank"].notna()].copy()
    missing = df.loc[df["OverallRank"].isna()].copy()

    valid = valid.sort_values(["OverallRank", "method"], ascending=[False, True]).reset_index(drop=True)
    valid["final_overall_rank"] = np.arange(1, len(valid) + 1, dtype=int)

    if not missing.empty:
        missing = missing.sort_values(
            ["n_available_components", "method"],
            ascending=[False, True],
        ).reset_index(drop=True)
        missing["final_overall_rank"] = np.arange(
            len(valid) + 1,
            len(valid) + len(missing) + 1,
            dtype=int,
        )

    out = pd.concat([valid, missing], ignore_index=True, sort=False)

    n_methods = len(out)
    out["overall_reversed_rank"] = n_methods + 1 - out["final_overall_rank"]

    for comp, col in component_cols.items():
        out[f"weighted_{comp}_contribution"] = weights[comp] * out[col]

    return out


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = {
        "accuracy": args.w_accuracy,
        "scalability": args.w_scalability,
        "stability": args.w_stability,
        "usability": args.w_usability,
    }

    alias_map = load_alias_map(args.method_alias_csv)
    input_paths = resolve_input_paths(args)

    tables = {}
    duplicate_reports = []
    for comp, path in input_paths.items():
        table, dup = load_component_table(
            path=path,
            component=comp,
            method_col=args.method_col,
            rank_col=args.rank_col,
            alias_map=alias_map,
        )
        tables[comp] = table
        if not dup.empty:
            duplicate_reports.append(dup)

    merged_all = merge_components(tables)

    exclude_methods = split_method_list(args.exclude_methods)
    merged_kept, excluded = apply_exclusion(merged_all, exclude_methods=exclude_methods)

    overall = compute_overall(
        merged_kept,
        weights=weights,
        missing_policy=args.missing_policy,
    )

    leading = [
        "method",
        "OverallRank",
        "final_overall_rank",
        "overall_rank_tie_average",
        "overall_reversed_rank",
        "R_accuracy",
        "R_scalability",
        "R_stability",
        "R_usability",
        "weighted_accuracy_contribution",
        "weighted_scalability_contribution",
        "weighted_stability_contribution",
        "weighted_usability_contribution",
        "n_available_components",
        "available_weight_sum",
        "missing_components",
    ]
    leading = [c for c in leading if c in overall.columns]
    rest = [c for c in overall.columns if c not in leading]
    overall = overall[leading + rest]

    # Save outputs.
    merged_all.to_csv(output_dir / "01_merged_reversed_ranks_before_exclusion.csv", index=False)
    merged_kept.to_csv(output_dir / "02_merged_reversed_ranks_used_for_overall.csv", index=False)
    excluded.to_csv(output_dir / "03_excluded_methods_from_overall.csv", index=False)
    overall.to_csv(output_dir / "overall_rank.csv", index=False)

    compact_cols = [
        "method",
        "OverallRank",
        "final_overall_rank",
        "overall_reversed_rank",
        "R_accuracy",
        "R_scalability",
        "R_stability",
        "R_usability",
        "n_available_components",
        "missing_components",
    ]
    compact_cols = [c for c in compact_cols if c in overall.columns]
    overall[compact_cols].to_csv(output_dir / "overall_rank_for_plot.csv", index=False)

    if duplicate_reports:
        pd.concat(duplicate_reports, ignore_index=True).to_csv(
            output_dir / "04_duplicate_method_names_collapsed.csv",
            index=False,
        )
    else:
        pd.DataFrame(columns=["component", "method", "n_rows", "raw_method_names"]).to_csv(
            output_dir / "04_duplicate_method_names_collapsed.csv",
            index=False,
        )

    qc_rows = []
    for comp, path in input_paths.items():
        qc_rows.append({"section": "input", "item": f"{comp}_csv", "value": str(path)})
        qc_rows.append({"section": "input", "item": f"{comp}_n_methods_after_canonicalization", "value": len(tables[comp])})
        qc_rows.append({"section": "weights", "item": f"w_{comp}", "value": weights[comp]})

    qc_rows.extend([
        {"section": "config", "item": "missing_policy", "value": args.missing_policy},
        {"section": "config", "item": "exclude_methods", "value": ";".join(exclude_methods)},
        {"section": "config", "item": "method_alias_csv", "value": args.method_alias_csv or ""},
        {"section": "output", "item": "n_methods_total_union_before_exclusion", "value": len(merged_all)},
        {"section": "output", "item": "n_methods_excluded", "value": len(excluded)},
        {"section": "output", "item": "excluded_methods_found", "value": ";".join(excluded["method"].tolist()) if not excluded.empty else ""},
        {"section": "output", "item": "n_methods_used_for_overall", "value": len(overall)},
        {"section": "output", "item": "n_methods_complete_4_components", "value": int((overall["n_available_components"] == 4).sum())},
        {"section": "output", "item": "n_methods_with_overall_score", "value": int(overall["OverallRank"].notna().sum())},
    ])
    pd.DataFrame(qc_rows).to_csv(output_dir / "overall_rank_qc_summary.csv", index=False)

    print("Done.")
    print(f"Excluded methods: {', '.join(excluded['method'].tolist()) if not excluded.empty else 'none found'}")
    print(f"Output directory: {output_dir}")
    print(f"Main result: {output_dir / 'overall_rank.csv'}")
    print(f"Duplicate check: {output_dir / '04_duplicate_method_names_collapsed.csv'}")


if __name__ == "__main__":
    main()
