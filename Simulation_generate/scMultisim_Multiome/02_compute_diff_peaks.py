#!/usr/bin/env python3
"""Compute observed and latent differential ATAC peaks before cisTopic."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

from common import (
    bh_adjust,
    build_differentiation_ground_truth,
    read_mtx,
    read_names,
    write_json,
)


def comparison_groups(obs: pd.DataFrame, ground_truth: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    labels = obs["true_cell_type"].astype(str).to_numpy()
    time = obs["true_time"].to_numpy(dtype=float)
    sources = list(map(str, ground_truth["transition_source"]))
    targets = list(map(str, ground_truth["transition_target"]))
    target_set = set(targets)
    root_branches = [x for x in map(str, ground_truth["branch_labels"]) if x not in target_set]
    output: list[tuple[str, np.ndarray, np.ndarray]] = []

    for branch in root_branches:
        idx = np.flatnonzero(labels == branch)
        if len(idx) < 20:
            continue
        ordered = idx[np.argsort(time[idx])]
        n = max(10, int(np.floor(len(ordered) * 0.35)))
        output.append((f"{branch}:early_vs_late", ordered[:n], ordered[-n:]))

    for source, target in zip(sources, targets):
        source_idx = np.flatnonzero(labels == source)
        target_idx = np.flatnonzero(labels == target)
        if len(source_idx) < 10 or len(target_idx) < 10:
            continue
        n_source = max(10, int(np.floor(len(source_idx) * 0.40)))
        n_target = max(10, int(np.floor(len(target_idx) * 0.40)))
        source_late = source_idx[np.argsort(time[source_idx])][-n_source:]
        target_late = target_idx[np.argsort(time[target_idx])][-n_target:]
        output.append((f"{source}_to_{target}", source_late, target_late))

    if not output:
        order = np.argsort(time)
        n = max(10, int(np.floor(len(order) * 0.30)))
        output.append(("global:early_vs_late", order[:n], order[-n:]))
    return output


def observed_stats(
    counts: sparse.csr_matrix,
    region_names: np.ndarray,
    comparisons: list[tuple[str, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    binary = counts.copy()
    binary.data = np.ones(binary.nnz, dtype=np.float32)
    records = []
    for comparison, group_a, group_b in comparisons:
        detected_a = np.asarray(binary[:, group_a].sum(axis=1)).ravel()
        detected_b = np.asarray(binary[:, group_b].sum(axis=1)).ravel()
        rate_a = detected_a / len(group_a)
        rate_b = detected_b / len(group_b)
        delta = rate_b - rate_a
        pooled = (detected_a + detected_b) / (len(group_a) + len(group_b))
        se = np.sqrt(np.maximum(pooled * (1 - pooled) * (1 / len(group_a) + 1 / len(group_b)), 1e-15))
        z = delta / se
        pvalue = 2 * stats.norm.sf(np.abs(z))
        pvalue[(pooled <= 0) | (pooled >= 1)] = 1.0
        padj = bh_adjust(pvalue)

        mean_a = np.asarray(counts[:, group_a].mean(axis=1)).ravel()
        mean_b = np.asarray(counts[:, group_b].mean(axis=1)).ravel()
        log2fc = np.log2((mean_b + 0.01) / (mean_a + 0.01))
        selected = (padj <= 0.10) & (np.abs(delta) >= 0.05) & (np.abs(log2fc) >= 0.5)
        selection_rule = np.full(len(region_names), "fdr_effect", dtype=object)
        minimum = max(10, int(np.ceil(len(region_names) * 0.001)))
        if selected.sum() < minimum:
            score = np.abs(delta) * np.maximum(np.abs(log2fc), 1e-6)
            fallback = np.argsort(score)[-minimum:]
            selected[fallback] = True
            selection_rule[fallback] = "top_effect_fallback"

        records.append(
            pd.DataFrame(
                {
                    "comparison": comparison,
                    "region": region_names,
                    "n_group_a": len(group_a),
                    "n_group_b": len(group_b),
                    "detection_rate_a": rate_a,
                    "detection_rate_b": rate_b,
                    "delta_detection": delta,
                    "mean_count_a": mean_a,
                    "mean_count_b": mean_b,
                    "log2fc": log2fc,
                    "pvalue": pvalue,
                    "padj": padj,
                    "selected": selected,
                    "selection_rule": selection_rule,
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def true_stats(
    latent: sparse.csr_matrix,
    region_names: np.ndarray,
    comparisons: list[tuple[str, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    records = []
    for comparison, group_a, group_b in comparisons:
        mean_a = np.asarray(latent[:, group_a].mean(axis=1)).ravel()
        mean_b = np.asarray(latent[:, group_b].mean(axis=1)).ravel()
        delta = mean_b - mean_a
        log2fc = np.log2((mean_b + 1e-4) / (mean_a + 1e-4))
        threshold = float(np.quantile(np.abs(delta), 0.95))
        selected = (np.abs(delta) >= max(threshold, 1e-8)) & (np.abs(log2fc) >= 0.25)
        if not selected.any():
            selected[np.argsort(np.abs(delta))[-max(10, len(region_names) // 200):]] = True
        records.append(
            pd.DataFrame(
                {
                    "comparison": comparison,
                    "region": region_names,
                    "n_group_a": len(group_a),
                    "n_group_b": len(group_b),
                    "true_mean_a": mean_a,
                    "true_mean_b": mean_b,
                    "true_delta": delta,
                    "true_log2fc": log2fc,
                    "selection_quantile": 0.95,
                    "selected": selected,
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def write_tsv_gz(frame: pd.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt") as handle:
        frame.to_csv(handle, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir).resolve()
    intermediate = outdir / "intermediate"

    region_names = read_names(intermediate / "region_names.tsv")
    cell_names = read_names(intermediate / "cell_names.tsv")
    counts = read_mtx(intermediate / "atac_counts_regions_cells.mtx.gz")
    latent = read_mtx(intermediate / "atac_true_regions_cells.mtx.gz")
    if counts.shape != latent.shape or counts.shape != (len(region_names), len(cell_names)):
        raise RuntimeError("ATAC matrices/names are not aligned")

    obs = pd.read_csv(intermediate / "cell_metadata.tsv.gz", sep="\t", index_col=0)
    obs = obs.loc[cell_names].copy()
    obs["true_cell_type"] = obs["true_cell_type"].astype(str)
    obs["true_time"] = pd.to_numeric(obs["true_time"], errors="raise")
    _, _, ground_truth = build_differentiation_ground_truth(
        obs["true_cell_type"], obs["true_time"].to_numpy()
    )
    comparisons = comparison_groups(obs, ground_truth)
    observed = observed_stats(counts, region_names, comparisons)
    truth = true_stats(latent, region_names, comparisons)

    observed_path = intermediate / "sckinetics_diff_peaks.tsv.gz"
    truth_path = intermediate / "true_differential_peaks.tsv.gz"
    write_tsv_gz(observed, observed_path)
    write_tsv_gz(truth, truth_path)

    peak_meta = pd.read_csv(intermediate / "peak_metadata.tsv.gz", sep="\t")
    selected = observed.loc[observed["selected"]].merge(peak_meta, on="region", how="left")
    bed = selected[
        ["chrom", "chromStart", "chromEnd", "region", "comparison", "log2fc", "padj", "selection_rule"]
    ].copy()
    bed["chromStart"] = np.maximum(0, bed["chromStart"].astype(int) - 1)
    bed.to_csv(
        outdir / "sckinetics_differential_peaks.bed",
        sep="\t",
        index=False,
        header=False,
    )

    summary = {
        "method": "vectorized two-proportion DA test on observed peak detection",
        "comparisons": [x[0] for x in comparisons],
        "n_observed_selected_union": int(observed.loc[observed.selected, "region"].nunique()),
        "n_true_selected_union": int(truth.loc[truth.selected, "region"].nunique()),
        "observed_complete_table": observed_path.name,
        "true_complete_table": truth_path.name,
        "bed": "sckinetics_differential_peaks.bed",
        "fallback_is_explicit_in_column": "selection_rule",
    }
    write_json(intermediate / "diff_peak_summary.json", summary)
    print("Differential peak comparisons:", summary["comparisons"])
    print("Observed selected union:", summary["n_observed_selected_union"])
    print("True selected union:", summary["n_true_selected_union"])
    print("DIFF PEAK STAGE: PASSED")


if __name__ == "__main__":
    main()
