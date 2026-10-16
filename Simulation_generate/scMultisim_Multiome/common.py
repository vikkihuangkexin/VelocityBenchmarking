#!/usr/bin/env python3
"""Shared I/O and topology helpers for the scMultiSim batch pipeline."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


def read_names(path: Path) -> np.ndarray:
    values = pd.read_csv(path, sep="\t", header=None, dtype=str).iloc[:, 0].to_numpy()
    if len(values) != len(set(values)):
        raise RuntimeError(f"Duplicate names in {path}")
    return values.astype(str)


def read_mtx(path: Path, dtype=np.float32) -> sparse.csr_matrix:
    matrix = mmread(path).tocsr().astype(dtype)
    matrix.eliminate_zeros()
    return matrix


def read_dense_f32(path: Path, shape: tuple[int, int]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise RuntimeError(
            f"Binary matrix size mismatch for {path}: {values.size} != {expected}"
        )
    return values.reshape(shape, order="F")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def build_differentiation_ground_truth(
    cell_types: pd.Series, true_time: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decode scMultiSim edge labels into a directed partial order."""
    labels = sorted(cell_types.astype(str).unique())
    branch_nodes: dict[str, tuple[str, str]] = {}
    for label in labels:
        parts = label.split("_")
        if len(parts) >= 2 and parts[0] and parts[1]:
            branch_nodes[label] = (parts[0], parts[1])

    if len(branch_nodes) != len(labels):
        # Defensive fallback for package versions that expose non-edge labels.
        medians = {
            label: float(np.median(true_time[cell_types.astype(str).to_numpy() == label]))
            for label in labels
        }
        ordered = sorted(labels, key=lambda x: (medians[x], x))
        branch_order = {label: i for i, label in enumerate(ordered)}
        transitions = list(zip(ordered[:-1], ordered[1:]))
        lineage_paths = ["->".join(ordered)]
        root_node = ordered[0]
        parent = ["" if i == 0 else ordered[i - 1] for i in range(len(ordered))]
        child = ordered
    else:
        parent_nodes = {x[0] for x in branch_nodes.values()}
        child_nodes = {x[1] for x in branch_nodes.values()}
        roots = sorted(parent_nodes - child_nodes)
        if len(roots) != 1:
            raise RuntimeError(f"Expected one differentiation root; found {roots}")
        root_node = roots[0]
        node_depth = {root_node: 0}
        unresolved = set(labels)
        while unresolved:
            progressed = False
            for label in sorted(unresolved):
                parent_node, child_node = branch_nodes[label]
                if parent_node in node_depth:
                    node_depth[child_node] = node_depth[parent_node] + 1
                    unresolved.remove(label)
                    progressed = True
                    break
            if not progressed:
                raise RuntimeError(f"Disconnected/cyclic branches: {sorted(unresolved)}")
        branch_order = {
            label: int(node_depth[branch_nodes[label][0]]) for label in labels
        }
        transitions = sorted(
            (source, target)
            for source in labels
            for target in labels
            if branch_nodes[source][1] == branch_nodes[target][0]
        )
        children = {
            source: [target for a, target in transitions if a == source]
            for source in labels
        }
        root_branches = [x for x in labels if branch_nodes[x][0] == root_node]
        lineage_paths: list[str] = []

        def visit(branch: str, path: list[str]) -> None:
            current = path + [branch]
            if not children[branch]:
                lineage_paths.append("->".join(current))
            else:
                for child_branch in children[branch]:
                    visit(child_branch, current)

        for branch in root_branches:
            visit(branch, [])
        parent = [branch_nodes[x][0] for x in labels]
        child = [branch_nodes[x][1] for x in labels]

    label_values = cell_types.astype(str).to_numpy()
    cell_order = np.asarray([branch_order[x] for x in label_values], dtype=np.int32)
    branch_pseudotime = np.zeros(len(label_values), dtype=np.float32)
    for label in labels:
        mask = label_values == label
        values = np.asarray(true_time[mask], dtype=np.float64)
        span = np.ptp(values)
        if span > 0:
            branch_pseudotime[mask] = ((values - values.min()) / span).astype(np.float32)

    ground_truth = {
        "ordering_type": "directed_partial_order",
        "root_node": str(root_node),
        "cell_type_key": "true_cell_type",
        "branch_key": "true_branch",
        "order_key": "true_cell_type_order",
        "time_key": "true_time",
        "pseudotime_key": "true_pseudotime",
        "branch_pseudotime_key": "true_branch_pseudotime",
        "branch_labels": np.asarray(labels, dtype=str),
        "branch_parent_node": np.asarray(parent, dtype=str),
        "branch_child_node": np.asarray(child, dtype=str),
        "branch_order": np.asarray([branch_order[x] for x in labels], dtype=np.int32),
        "transition_source": np.asarray([x[0] for x in transitions], dtype=str),
        "transition_target": np.asarray([x[1] for x in transitions], dtype=str),
        "lineage_paths": np.asarray(sorted(lineage_paths), dtype=str),
    }
    return cell_order, branch_pseudotime, ground_truth


def dataframe_to_uns(df: pd.DataFrame) -> dict:
    """Column-oriented, h5ad-safe representation of complete result tables."""
    output: dict[str, np.ndarray] = {}
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            output[column] = series.to_numpy(dtype=bool)
        elif pd.api.types.is_integer_dtype(series):
            output[column] = series.to_numpy(dtype=np.int64)
        elif pd.api.types.is_numeric_dtype(series):
            output[column] = series.to_numpy(dtype=np.float64)
        else:
            output[column] = series.fillna("").astype(str).to_numpy()
    return output


def read_table_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as handle:
        return pd.read_csv(handle, sep="\t")
