"""Compute in-cluster coherence from a declared velocity layer and cell labels.

A cell score is the mean cosine of its high-dimensional velocity to the
velocities of neighbors with the same annotation. The total pools cell scores.
"""

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
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import cosine_similarity


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
    gene_mask_policy: str


def required(row: dict[str, str], field: str, line: int) -> str:
    value = (row.get(field) or "").strip()
    if not value:
        raise ValueError(f"CSV line {line} needs {field}.")
    return value


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
            policy = (row.get("gene_mask_policy") or
                      ("finite" if modality == "atac" and data_type == "sim"
                       else "not_nan")).strip().lower()
            if policy not in {"not_nan", "finite"}:
                raise ValueError(f"CSV line {line}: gene_mask_policy must be not_nan or finite.")
            source = Path(required(row, "path", line)).expanduser()
            if not source.is_absolute():
                source = path.parent / source
            samples.append(Sample(
                data_type=data_type,
                modality=modality,
                method=required(row, "method", line),
                dataset_id=required(row, "id", line),
                run_id=(row.get("run_id") or "").strip() or "single",
                path=source,
                vkey=required(row, "vkey", line),
                cluster_key=cluster_key,
                coord_key=coordinate_key((row.get("basis_name") or "").strip(), data_type, modality),
                gene_mask_policy=policy,
            ))
    if not samples:
        raise ValueError("Input CSV contains no samples.")
    identities = [(sample.data_type, sample.modality, sample.method, sample.dataset_id, sample.run_id) for sample in samples]
    if len(identities) != len(set(identities)):
        raise ValueError("Input CSV repeats a modality/method/dataset_id/run_id combination.")
    return samples


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


def connectivities(adata, coord_key: str, n_neighbors: int) -> csr_matrix:
    """Reuse valid input neighbors for every modality; build only missing/invalid graphs.

    Only connectivities are needed for the within-cluster comparisons.
    """
    metadata = adata.uns.get("neighbors", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    graph = None
    for key in dict.fromkeys((metadata.get("connectivities_key", "connectivities"), "connectivities")):
        candidate = valid_graph(adata.obsp.get(key), adata.n_obs, require_edges=True)
        if candidate is not None and not np.any(candidate.data < 0):
            graph = candidate
            break
    if graph is None:
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
        if graph is None:
            raise ValueError("Neighbor construction did not produce usable graphs.")
    return graph


def first_row_mask(velocity, policy: str) -> np.ndarray:
    """Preserve the source's first-cell gene mask; do not repair later bad rows."""
    first = velocity[0]
    if issparse(first):
        first = first.toarray()
    values = np.asarray(first).ravel()
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("The declared velocity layer is not numeric.")
    mask = ~np.isnan(values) if policy == "not_nan" else np.isfinite(values)
    if not mask.any():
        raise ValueError("The first velocity row has no usable genes.")
    return mask


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


def score_sample(sample: Sample, n_neighbors: int):
    adata = sc.read_h5ad(sample.path)
    cluster_key = sample.cluster_key or next(
        (key for key in ("pseudo_celltype", "celltype") if key in adata.obs), "")
    if not cluster_key or cluster_key not in adata.obs:
        raise KeyError(f"Missing obs[{cluster_key or 'pseudo_celltype/celltype'!r}].")
    normalize_spatial_annotation(adata, sample, cluster_key)
    if sample.vkey not in adata.layers:
        raise KeyError(f"Missing high-dimensional layers[{sample.vkey!r}].")
    if adata.n_obs == 0:
        raise ValueError("The result contains no cells.")
    graph = connectivities(adata, sample.coord_key, n_neighbors)
    velocity = adata.layers[sample.vkey]
    if velocity.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(f"layers[{sample.vkey!r}] does not match the cells and genes.")
    gene_mask = first_row_mask(velocity, sample.gene_mask_policy)
    velocity = velocity[:, gene_mask]
    label_series = adata.obs[cluster_key]
    labeled = label_series.notna().to_numpy()
    labels = label_series.astype(str).to_numpy()
    cells, clusters, all_scores = [], [], []
    for cluster in pd.unique(labels[labeled]):
        indices = np.flatnonzero(labeled & (labels == cluster))
        cluster_scores = []
        for index in indices:
            neighbors = graph[index].indices
            same = neighbors[labeled[neighbors] & (labels[neighbors] == cluster)]
            score = math.nan
            if len(same):
                source = velocity[index:index + 1]
                targets = velocity[same]
                # Finite zero vectors contribute zero; sklearn rejects NaN/Inf.
                score = float(cosine_similarity(source, targets).mean())
                if math.isfinite(score):
                    cluster_scores.append(score)
                    all_scores.append(score)
            cells.append({
                "method": sample.method, "dataset_id": sample.dataset_id,
                "data_type": sample.data_type,
                "modality": sample.modality, "run_id": sample.run_id,
                "cluster": cluster, "cell_id": str(adata.obs_names[index]),
                "cell_index": int(index), "n_same_cluster_neighbors": int(len(same)),
                "score": score,
            })
        clusters.append({
            "method": sample.method, "dataset_id": sample.dataset_id,
            "data_type": sample.data_type,
            "modality": sample.modality, "run_id": sample.run_id,
            "cluster": cluster,
            "mean_score": float(np.mean(cluster_scores)) if cluster_scores else math.nan,
            "n_cells_scored": len(cluster_scores), "n_cells_eligible": len(indices),
        })
    total = {
        "method": sample.method, "dataset_id": sample.dataset_id,
        "data_type": sample.data_type,
        "modality": sample.modality, "run_id": sample.run_id,
        "score": float(np.mean(all_scores)) if all_scores else math.nan,
        "n_cells_scored": len(all_scores), "n_cells": adata.n_obs,
        "n_unlabeled_cells": int((~labeled).sum()),
        "n_genes_scored": int(gene_mask.sum()),
        "velocity_layer": sample.vkey, "cluster_key": cluster_key,
        "gene_mask_policy": sample.gene_mask_policy,
    }
    return cells, clusters, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-neighbors", type=int, default=30,
                        help="Used only when a neighbor graph is missing or invalid.")
    args = parser.parse_args()
    if args.n_neighbors < 2:
        parser.error("--n-neighbors must be at least 2.")
    samples = read_manifest(args.manifest.expanduser().resolve())
    cells, clusters, totals = [], [], []
    for sample in samples:
        sample_cells, sample_clusters, total = score_sample(sample, args.n_neighbors)
        cells.extend(sample_cells)
        clusters.extend(sample_clusters)
        totals.append(total)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cells).to_csv(output / "iccoh_cells.csv", index=False)
    pd.DataFrame(clusters).to_csv(output / "iccoh_clusters.csv", index=False)
    pd.DataFrame(totals).to_csv(output / "iccoh_totals.csv", index=False)


if __name__ == "__main__":
    main()
