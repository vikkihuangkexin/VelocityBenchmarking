"""Compute cross-boundary direction correctness from explicit input tables.

Each source cell is scored by its mean cosine to target-state neighbors.
The total pools finite source-cell scores across distinct directed edges.
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
import scvelo as scv
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import cosine_similarity


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
    velocity_embedding_key: str
    xkey: str
    invalid_embedding_recompute_ratio: float


def required(row: dict[str, str], key: str, line: int) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"CSV line {line} needs {key}.")
    return value


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def coordinate_key(basis_name: str, data_type: str, modality: str) -> str:
    basis = basis_name or ("spatial" if modality == "spatial" else
                           "tsne" if modality == "atac" and data_type == "sim" else
                           "dimred" if data_type == "sim" else "umap")
    return basis if basis == "spatial" or basis.startswith("X_") else f"X_{basis}"


def ratio(value: str, default: float, field: str, line: int) -> float:
    number = float(value) if value else default
    if not 0 < number <= 1:
        raise ValueError(f"CSV line {line}: {field} must be in (0, 1].")
    return number


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
            coord_key = coordinate_key((row.get("basis_name") or "").strip(), data_type, modality)
            vkey = required(row, "vkey", line)
            samples.append(Sample(
                data_type=data_type,
                modality=modality,
                method=required(row, "method", line),
                dataset_id=required(row, "id", line),
                run_id=(row.get("run_id") or "").strip() or "single",
                path=resolve_path(required(row, "path", line), path.parent),
                vkey=vkey,
                cluster_key=cluster_key,
                coord_key=coord_key,
                velocity_embedding_key=(row.get("velocity_embedding_key") or "").strip()
                    or f"{vkey}_{coord_key.removeprefix('X_')}",
                xkey=(row.get("xkey") or "").strip(),
                invalid_embedding_recompute_ratio=ratio(
                    (row.get("invalid_embedding_recompute_ratio") or "").strip(),
                    0.15, "invalid_embedding_recompute_ratio", line),
            ))
    if not samples:
        raise ValueError("Input CSV contains no samples.")
    identities = [(sample.data_type, sample.modality, sample.method, sample.dataset_id, sample.run_id) for sample in samples]
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
            closed = required(row, "closed", line).lower()
            if closed not in {"true", "false", "1", "0"}:
                raise ValueError(f"Path CSV line {line}: closed must be Boolean.")
            is_closed = closed in {"true", "1"}
            if key in closure and closure[key] != is_closed:
                raise ValueError(f"Path {key} has inconsistent closed values.")
            closure[key] = is_closed
            groups.setdefault(key, []).append((position, label))
    edges_by_id: dict[str, list[tuple[str, str]]] = {}
    seen: dict[str, set[tuple[str, str]]] = {}
    for (dataset_id, path_id), entries in groups.items():
        entries.sort()
        if [position for position, _ in entries] != list(range(len(entries))):
            raise ValueError(f"Path {dataset_id}/{path_id} needs consecutive positions from 0.")
        labels = [label for _, label in entries]
        if len(labels) < 2 or (closure[(dataset_id, path_id)] and labels[0] == labels[-1]):
            raise ValueError(f"Path {dataset_id}/{path_id} needs two labels and no repeated closure endpoint.")
        edges = list(zip(labels, labels[1:]))
        if closure[(dataset_id, path_id)]:
            edges.append((labels[-1], labels[0]))
        result = edges_by_id.setdefault(dataset_id, [])
        unique = seen.setdefault(dataset_id, set())
        for edge in edges:
            if edge[0] == edge[1]:
                raise ValueError(f"Path {dataset_id}/{path_id} has a self-edge.")
            if edge not in unique:
                result.append(edge)
                unique.add(edge)
    return edges_by_id


def coordinates(adata, key: str) -> np.ndarray:
    if key not in adata.obsm:
        raise KeyError(f"Missing obsm[{key!r}].")
    matrix = np.asarray(adata.obsm[key])
    if matrix.ndim != 2 or matrix.shape[0] != adata.n_obs or matrix.shape[1] < 2:
        raise ValueError(f"obsm[{key!r}] has invalid shape.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"obsm[{key!r}] has non-finite coordinates.")
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


def projection_graph(adata, vkey: str, xkey: str, coord_key: str, n_neighbors: int) -> None:
    """Use a validated positive/negative pair belonging to the declared velocity key."""
    keys = (f"{vkey}_graph", f"{vkey}_graph_neg")
    pair = []
    for key in keys:
        graph = valid_graph(adata.obsp.get(key), adata.n_obs)
        if graph is None:
            graph = valid_graph(adata.uns.get(key), adata.n_obs)
        pair.append(graph)
    if any(graph is None for graph in pair):
        if vkey not in adata.layers:
            raise KeyError(f"Rebuilding velocity graphs needs layers[{vkey!r}].")
        connectivities(adata, coord_key, n_neighbors, for_projection=True)
        prepare_expression(adata, xkey)
        # Remove invalid cached copies so scVelo cannot select stale obsp data.
        for key in keys:
            adata.uns.pop(key, None)
            adata.obsp.pop(key, None)
        options = {"vkey": vkey, "n_jobs": 1, "show_progress_bar": False, "approx": False}
        if xkey:
            options["xkey"] = xkey
        scv.tl.velocity_graph(adata, **options)
        pair = [valid_graph(adata.uns.get(key), adata.n_obs) for key in keys]
        if any(graph is None for graph in pair):
            raise ValueError(f"scVelo did not create valid velocity graphs for {vkey!r}.")
    for key, graph in zip(keys, pair):
        adata.uns[key] = graph
        if key in adata.obsp:
            adata.obsp[key] = graph


def nan_velocity_rows(velocity: np.ndarray, shape: tuple[int, int], key: str) -> np.ndarray:
    if velocity.shape != shape:
        raise ValueError(f"obsm[{key!r}] does not match coordinate dimensions.")
    invalid = ~np.isfinite(velocity).all(axis=1)
    nan_rows = np.isnan(velocity).all(axis=1)
    if (invalid & ~nan_rows).any():
        raise ValueError(f"obsm[{key!r}] contains partially non-finite rows.")
    return nan_rows


def velocity_embedding(adata, sample: Sample, n_neighbors: int) -> np.ndarray:
    coords = coordinates(adata, sample.coord_key)
    key = sample.velocity_embedding_key
    if key in adata.obsm:
        velocity = np.asarray(adata.obsm[key])
        nan_rows = nan_velocity_rows(velocity, coords.shape, key)
        if not nan_rows.any():
            return velocity
        if float(nan_rows.mean()) < sample.invalid_embedding_recompute_ratio:
            return velocity
        if sample.vkey not in adata.layers:
            raise ValueError(f"obsm[{key!r}] needs recomputation but layers[{sample.vkey!r}] is absent.")
        del adata.obsm[key]
    if sample.vkey not in adata.layers:
        raise KeyError(f"Missing both obsm[{key!r}] and layers[{sample.vkey!r}].")
    projection_graph(adata, sample.vkey, sample.xkey, sample.coord_key, n_neighbors)
    basis = sample.coord_key.removeprefix("X_")
    coord_alias = f"X_{basis}"
    if coord_alias not in adata.obsm:
        adata.obsm[coord_alias] = coords.copy()
    elif coord_alias != sample.coord_key and not np.allclose(
            coordinates(adata, coord_alias), coords):
        raise ValueError(f"obsm[{coord_alias!r}] differs from declared obsm[{sample.coord_key!r}].")
    # scVelo's projection reads this layer with np.array, which does not
    # convert a SciPy sparse matrix into a numeric cells-by-genes array.
    if issparse(adata.layers[sample.vkey]):
        adata.layers[sample.vkey] = adata.layers[sample.vkey].toarray()
    scv.tl.velocity_embedding(adata, basis=basis, vkey=sample.vkey)
    produced = f"{sample.vkey}_{basis}"
    if produced not in adata.obsm:
        raise KeyError(f"Projection did not create obsm[{produced!r}].")
    velocity = np.asarray(adata.obsm[produced])
    nan_velocity_rows(velocity, coords.shape, produced)
    if key != produced:
        adata.obsm[key] = velocity.copy()
    return velocity


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
    coords = coordinates(adata, sample.coord_key)
    coord_alias = f"X_{sample.coord_key.removeprefix('X_')}"
    if coord_alias != sample.coord_key and coord_alias in adata.obsm \
            and not np.allclose(coordinates(adata, coord_alias), coords):
        raise ValueError(f"obsm[{coord_alias!r}] differs from declared obsm[{sample.coord_key!r}].")
    cluster_key = sample.cluster_key or next(
        (key for key in ("pseudo_celltype", "celltype") if key in adata.obs), "")
    if not cluster_key or cluster_key not in adata.obs:
        raise KeyError(f"Missing obs[{cluster_key or 'pseudo_celltype/celltype'!r}].")
    normalize_spatial_annotation(adata, sample, cluster_key)
    labels = adata.obs[cluster_key].astype(str).to_numpy()
    absent = {label for edge in edges for label in edge} - set(labels)
    if absent:
        raise ValueError(f"Declared path labels are absent from obs[{cluster_key!r}]: {sorted(absent)}.")
    neighbors = connectivities(adata, sample.coord_key, n_neighbors)
    velocity = velocity_embedding(adata, sample, n_neighbors)
    invalid = nan_velocity_rows(velocity, coords.shape, sample.velocity_embedding_key)
    invalid |= ~np.any(velocity != 0, axis=1)
    cells, by_edge, valid = [], [], []
    for source, target in edges:
        source_cells = np.flatnonzero(labels == source)
        edge_scores = []
        for index in source_cells:
            targets = neighbors[index].indices
            targets = targets[labels[targets] == target]
            score = math.nan
            if len(targets) and not invalid[index]:
                displacements = coords[targets] - coords[index]
                # A coincident target contributes cosine zero; source zero/NaN
                # velocity rows were excluded before entering this comparison.
                cosines = cosine_similarity(displacements, velocity[index].reshape(1, -1)).ravel()
                if np.isfinite(cosines).any():
                    score = float(np.nanmean(cosines))
                    edge_scores.append(score)
                    valid.append(score)
            cells.append({
                "method": sample.method, "dataset_id": sample.dataset_id, "data_type": sample.data_type,
                "modality": sample.modality,
                "run_id": sample.run_id,
                "source": source, "target": target, "cell_id": str(adata.obs_names[index]),
                "cell_index": int(index), "n_target_neighbors": int(len(targets)), "score": score,
            })
        by_edge.append({
            "method": sample.method, "dataset_id": sample.dataset_id, "data_type": sample.data_type,
            "modality": sample.modality,
            "run_id": sample.run_id,
            "source": source, "target": target,
            "mean_score": float(np.mean(edge_scores)) if edge_scores else math.nan,
            "n_cells_scored": len(edge_scores), "n_cells_eligible": len(source_cells),
        })
    total = {
        "method": sample.method, "dataset_id": sample.dataset_id, "data_type": sample.data_type,
        "modality": sample.modality,
        "run_id": sample.run_id,
        "score": float(np.mean(valid)) if valid else math.nan, "n_scores": len(valid),
        "n_edges": len(edges), "n_cells": adata.n_obs, "n_invalid_velocity_rows": int(invalid.sum()),
        "coord_key": sample.coord_key, "velocity_embedding_key": sample.velocity_embedding_key,
    }
    return cells, by_edge, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=ROOT.parent / "differentiation_paths.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-neighbors", type=int, default=30,
                        help="Used only when a neighbor graph is missing or invalid.")
    args = parser.parse_args()
    if args.n_neighbors < 2:
        parser.error("--n-neighbors must be at least 2.")
    samples = read_manifest(args.manifest.expanduser().resolve())
    paths = read_paths(args.paths.expanduser().resolve())
    cells, edge_scores, totals = [], [], []
    for sample in samples:
        edges = paths.get(sample.dataset_id)
        if not edges:
            raise KeyError(f"No differentiation path for dataset_id={sample.dataset_id!r}.")
        sample_cells, sample_edges, total = score_sample(sample, edges, args.n_neighbors)
        cells.extend(sample_cells)
        edge_scores.extend(sample_edges)
        totals.append(total)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cells).to_csv(output / "cbdir_cells.csv", index=False)
    pd.DataFrame(edge_scores).to_csv(output / "cbdir_edges.csv", index=False)
    pd.DataFrame(totals).to_csv(output / "cbdir_totals.csv", index=False)


if __name__ == "__main__":
    main()
