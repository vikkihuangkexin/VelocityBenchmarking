#!/usr/bin/env python3
"""Compute transition score from directed paths and velocity graph edges."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parent
MODALITIES = {"rna", "spatial", "atac", "lineage"}


@dataclass(frozen=True)
class Sample:
    data_type: str
    modality: str
    method: str
    dataset_id: str
    run_id: str
    path: Path
    vkey: str
    cluster_key: str
    coord_key: str
    xkey: str


def required(row: dict[str, str], key: str, line: int) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"CSV line {line} needs {key}.")
    return value


def resolve_path(value: str, base: Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else base / candidate


def coordinate_key(basis_name: str, data_type: str, modality: str) -> str:
    basis = basis_name or ("spatial" if modality == "spatial" else
                           "tsne" if modality == "atac" and data_type == "sim" else
                           "dimred" if data_type == "sim" else "umap")
    return basis if basis == "spatial" or basis.startswith("X_") else f"X_{basis}"


def read_manifest(path: Path) -> list[Sample]:
    samples = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = {"method", "id", "path", "vkey", "data_type"}
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError(f"Input CSV needs columns: {', '.join(sorted(fields))}.")
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"CSV line {line} has more values than its header.")
            data_type = required(row, "data_type", line).lower()
            if data_type not in {"real", "sim"}:
                raise ValueError(f"CSV line {line}: data_type must be real or sim.")
            modality = (row.get("modality") or "rna").strip().lower()
            if modality not in MODALITIES:
                raise ValueError(f"CSV line {line}: unsupported modality {modality!r}.")
            cluster_key = (row.get("cluster_key") or "").strip()
            if not cluster_key and not (modality == "atac" and data_type == "sim"):
                raise ValueError(f"CSV line {line} needs cluster_key.")
            method = required(row, "method", line)
            dataset_id = required(row, "id", line)
            run_id = (row.get("run_id") or "").strip() or "single"
            for field, value in (("method", method), ("dataset_id", dataset_id), ("run_id", run_id)):
                if value in {".", ".."} or "/" in value or "\\" in value:
                    raise ValueError(f"CSV line {line}: {field} must be one path component.")
            source = resolve_path(required(row, "path", line), path.parent)
            if source.suffix.lower() != ".h5ad" or not source.is_file():
                raise FileNotFoundError(f"CSV line {line}: H5AD file not found: {source}")
            samples.append(Sample(
                data_type, modality, method, dataset_id, run_id, source,
                required(row, "vkey", line), cluster_key,
                coordinate_key((row.get("basis_name") or "").strip(), data_type, modality),
                (row.get("xkey") or "").strip(),
            ))
    if not samples:
        raise ValueError("Input CSV contains no samples.")
    identities = [(s.data_type, s.modality, s.method, s.dataset_id, s.run_id) for s in samples]
    if len(identities) != len(set(identities)):
        raise ValueError("Input CSV repeats a modality/method/dataset_id/run_id combination.")
    return samples


def read_paths(path: Path) -> dict[str, list[tuple[str, str]]]:
    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    closure: dict[tuple[str, str], bool] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = {"dataset_id", "path_id", "position", "label", "closed"}
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError(f"Path CSV needs columns: {', '.join(sorted(fields))}.")
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"CSV line {line} has more values than its header.")
            key = (required(row, "dataset_id", line), required(row, "path_id", line))
            label = row.get("label") or ""
            if not label.strip():
                raise ValueError(f"Path CSV line {line} has an empty label.")
            position = int(required(row, "position", line))
            if position < 0:
                raise ValueError(f"Path CSV line {line} has a negative position.")
            flag = required(row, "closed", line).lower()
            if flag not in {"true", "false", "1", "0"}:
                raise ValueError(f"Path CSV line {line}: closed must be Boolean.")
            is_closed = flag in {"true", "1"}
            if key in closure and closure[key] != is_closed:
                raise ValueError(f"Path {key} has inconsistent closed values.")
            closure[key] = is_closed
            groups.setdefault(key, []).append((position, label))
    by_dataset: dict[str, list[tuple[str, str]]] = {}
    seen: dict[str, set[tuple[str, str]]] = {}
    for key, entries in groups.items():
        entries.sort()
        if [position for position, _ in entries] != list(range(len(entries))):
            raise ValueError(f"Path {key} needs consecutive positions from 0.")
        labels = [label for _, label in entries]
        if len(labels) < 2 or (closure[key] and labels[0] == labels[-1]):
            raise ValueError(f"Path {key} needs two labels and no repeated closure endpoint.")
        edges = list(zip(labels, labels[1:]))
        if closure[key]:
            edges.append((labels[-1], labels[0]))
        output = by_dataset.setdefault(key[0], [])
        unique = seen.setdefault(key[0], set())
        for edge in edges:
            if edge[0] == edge[1]:
                raise ValueError(f"Path {key} has a self-edge.")
            if edge not in unique:
                output.append(edge)
                unique.add(edge)
    return by_dataset


def coordinates(adata, key: str) -> np.ndarray:
    if key not in adata.obsm:
        raise KeyError(f"Missing obsm[{key!r}].")
    matrix = np.asarray(adata.obsm[key], dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != adata.n_obs or matrix.shape[1] < 2
            or not np.isfinite(matrix).all()):
        raise ValueError(f"obsm[{key!r}] has invalid coordinates.")
    return matrix



def valid_graph(matrix, n_cells: int, *, require_edges: bool = False):
    """Validate a graph and remove stored zeros before enumerating neighbors."""
    if matrix is None or getattr(matrix, "shape", None) != (n_cells, n_cells):
        return None
    try:
        graph = csr_matrix(matrix, copy=True)
        graph.sum_duplicates()
        graph.eliminate_zeros()
        if not np.isfinite(graph.data).all() or (require_edges and graph.nnz == 0):
            return None
    except (TypeError, ValueError):
        return None
    return graph


def connectivities(adata, coord_key: str, n_neighbors: int, *, for_projection: bool = False) -> csr_matrix:
    """Reuse valid input neighbors for every modality; build only missing/invalid graphs.

    Scoring only needs connectivities. scVelo graph construction additionally
    needs nonempty distance rows and standard neighbor keys, even when the
    input stores its matrices under custom keys.
    """
    metadata = adata.uns.get("neighbors", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    graph = None
    for key in dict.fromkeys((metadata.get("connectivities_key", "connectivities"), "connectivities")):
        candidate = valid_graph(adata.obsp.get(key), adata.n_obs, require_edges=True)
        if candidate is not None and not np.any(candidate.data < 0):
            graph = candidate
            break
    distances = None
    if for_projection:
        for key in dict.fromkeys((metadata.get("distances_key", "distances"), "distances")):
            candidate = valid_graph(adata.obsp.get(key), adata.n_obs, require_edges=True)
            if candidate is not None and not np.any(candidate.data < 0) and np.all(np.diff(candidate.indptr) > 0):
                distances = candidate
                break
    if graph is None or (for_projection and distances is None):
        if coord_key not in adata.obsm:
            raise KeyError(f"Missing obsm[{coord_key!r}] for neighbor graph construction.")
        coords = np.asarray(adata.obsm[coord_key])
        if (coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 2
                or not np.isfinite(coords).all()):
            raise ValueError(f"obsm[{coord_key!r}] has invalid coordinates.")
        if adata.n_obs < 3:
            raise ValueError("Building neighbors needs at least three cells.")
        sc.pp.neighbors(adata, use_rep=coord_key, n_neighbors=min(n_neighbors, adata.n_obs - 1))
        metadata = adata.uns["neighbors"]
        graph = valid_graph(adata.obsp[metadata["connectivities_key"]], adata.n_obs, require_edges=True)
        distances = valid_graph(adata.obsp[metadata["distances_key"]], adata.n_obs, require_edges=True)
        if graph is None or (for_projection and (distances is None or np.any(np.diff(distances.indptr) == 0))):
            raise ValueError("Neighbor construction did not produce usable graphs.")
    if for_projection:
        adata.obsp["connectivities"] = graph
        adata.obsp["distances"] = distances
        metadata = dict(metadata)
        metadata.update(connectivities_key="connectivities", distances_key="distances")
        params = metadata.get("params", {})
        params = dict(params) if isinstance(params, dict) else {}
        params.setdefault("n_neighbors", int(np.diff(distances.indptr).max()) + 1)
        metadata["params"] = params
        adata.uns["neighbors"] = metadata
    return graph


def prepare_expression(adata, xkey: str) -> None:
    """Calculate missing moments on existing neighbors without replacing a saved Ms."""
    if xkey:
        if xkey not in adata.layers:
            raise KeyError(f"Missing declared expression layer {xkey!r}.")
        return
    for target, source in (("Ms", "M_s"), ("Mu", "M_u")):
        if target not in adata.layers and source in adata.layers:
            adata.layers[target] = adata.layers[source].copy()
    if "Ms" not in adata.layers and {"spliced", "unspliced"}.issubset(adata.layers):
        scv.pp.moments(adata, n_neighbors=None)
    if "Ms" not in adata.layers and "spliced" not in adata.layers:
        raise KeyError("Building a velocity graph needs Ms, spliced, or an explicit xkey.")


def velocity_graph(adata, sample: Sample, n_neighbors: int) -> csr_matrix:
    key = f"{sample.vkey}_graph"
    graph = valid_graph(adata.uns.get(key), adata.n_obs)
    if graph is None:
        graph = valid_graph(adata.obsp.get(key), adata.n_obs)
    if graph is None:
        if sample.vkey not in adata.layers:
            raise KeyError(f"Missing a valid velocity graph {key!r} and layers[{sample.vkey!r}].")
        connectivities(adata, sample.coord_key, n_neighbors, for_projection=True)
        prepare_expression(adata, sample.xkey)
        for suffix in ("_graph", "_graph_neg"):
            adata.uns.pop(f"{sample.vkey}{suffix}", None)
            adata.obsp.pop(f"{sample.vkey}{suffix}", None)
        options = {"vkey": sample.vkey, "n_jobs": 1, "show_progress_bar": False}
        if sample.xkey:
            options["xkey"] = sample.xkey
        scv.tl.velocity_graph(adata, **options)
        graph = valid_graph(adata.uns.get(key), adata.n_obs)
    if graph is None:
        raise ValueError(f"Velocity graph {key!r} is missing or invalid after construction.")
    return graph


def normalize_spatial_annotation(adata, sample: Sample, cluster_key: str) -> None:
    """Retain the study's shared Neuron annotation for real spatial dataset 69."""
    prefix = re.match(r"^(\d+)", sample.dataset_id)
    if sample.data_type != "real" or sample.modality != "spatial" or prefix is None or prefix.group(1) != "69":
        return
    labels = adata.obs[cluster_key]
    numbered = labels.astype("string").str.fullmatch(r"Neuron_\d+").fillna(False)
    if numbered.any():
        # Preserve missing labels so ICCoh can still count unlabeled cells.
        adata.obs[cluster_key] = labels.astype(object).mask(numbered, "Neuron")


def score_sample(sample: Sample, edges: list[tuple[str, str]], n_neighbors: int):
    adata = sc.read_h5ad(sample.path)
    cluster_key = sample.cluster_key or next(
        (key for key in ("pseudo_celltype", "celltype") if key in adata.obs), "")
    if not cluster_key or cluster_key not in adata.obs:
        raise KeyError(f"Missing obs[{cluster_key or 'pseudo_celltype/celltype'!r}].")
    normalize_spatial_annotation(adata, sample, cluster_key)
    labels = adata.obs[cluster_key].astype(str).to_numpy()
    missing = {label for edge in edges for label in edge} - set(labels)
    if missing:
        raise ValueError(f"Declared path labels are absent from obs[{cluster_key!r}]: {sorted(missing)}.")
    neighbors = connectivities(adata, sample.coord_key, n_neighbors)
    graph = velocity_graph(adata, sample, n_neighbors)
    context = {"data_type": sample.data_type, "modality": sample.modality,
               "method": sample.method, "dataset_id": sample.dataset_id,
               "run_id": sample.run_id}
    cells, by_edge, all_scores = [], [], []
    for source, target in edges:
        eligible = np.flatnonzero(labels == source)
        edge_scores = []
        for index in eligible:
            candidates = neighbors[index].indices
            targets = candidates[labels[candidates] == target]
            score = math.nan
            if len(targets):
                # Absent sparse velocity edges count as zero among these target
                # neighbors; these are graph values, not transition probabilities.
                values = np.asarray(graph[index, targets].toarray()).ravel()
                if np.isfinite(values).all():
                    score = float(values.mean())
                    edge_scores.append(score)
                    all_scores.append(score)
            cells.append({**context, "source": source, "target": target,
                          "cell_id": str(adata.obs_names[index]), "cell_index": int(index),
                          "n_target_neighbors": int(len(targets)), "score": score})
        by_edge.append({**context, "source": source, "target": target,
                        "mean_score": float(np.mean(edge_scores)) if edge_scores else math.nan,
                        "n_cells_scored": len(edge_scores), "n_cells_eligible": len(eligible)})
    total = {**context, "score": float(np.mean(all_scores)) if all_scores else math.nan,
             "n_scores": len(all_scores), "n_edges": len(edges), "n_cells": adata.n_obs,
             "score_definition": "mean_input_velocity_graph_value"}
    return total, by_edge, cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paths-csv", type=Path, help="ID-to-path CSV; relative to the manifest directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-neighbors", type=int, default=30)
    args = parser.parse_args()
    if args.n_neighbors < 2:
        parser.error("--n-neighbors must be at least 2")
    manifest = args.manifest.expanduser().resolve()
    path_table = resolve_path(str(args.paths_csv), manifest.parent) if args.paths_csv else ROOT.parent / "differentiation_paths.csv"
    samples = read_manifest(manifest)
    paths = read_paths(path_table)
    totals, edges, cells = [], [], []
    for sample in samples:
        if sample.dataset_id not in paths:
            raise KeyError(f"No differentiation path for dataset_id {sample.dataset_id!r}.")
        total, by_edge, by_cell = score_sample(sample, paths[sample.dataset_id], args.n_neighbors)
        totals.append(total)
        edges.extend(by_edge)
        cells.extend(by_cell)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(totals).to_csv(args.output_dir / "transition_score.csv", index=False)
    pd.DataFrame(edges).to_csv(args.output_dir / "transition_score_by_edge.csv", index=False)
    pd.DataFrame(cells).to_csv(args.output_dir / "transition_score_cells.csv", index=False)


if __name__ == "__main__":
    main()
