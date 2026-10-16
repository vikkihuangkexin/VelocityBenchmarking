"""Score predicted versus true velocity in gene or embedding space.

Each cell score is (cosine + 1) / 2. The dataset score pools valid cells.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy.sparse import csr_matrix, issparse


MODALITIES = {"real", "sim", "atac", "spatial", "lineage"}


def required(row: dict[str, str], key: str, line: int) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"CSV line {line} needs {key}.")
    return value


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = {"modality", "method", "dataset_id", "path", "vkey"}
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError(f"Input CSV needs columns: {', '.join(sorted(fields))}.")
        for line, source in enumerate(reader, 2):
            if None in source:
                raise ValueError(f"CSV line {line} has more values than its header.")
            modality = required(source, "modality", line).lower()
            if modality not in MODALITIES:
                raise ValueError(f"CSV line {line}: unsupported modality {modality!r}.")
            space = (source.get("score_space") or
                     ("embedding" if modality == "atac" else "gene")).strip().lower()
            if space not in {"gene", "embedding"}:
                raise ValueError(f"CSV line {line}: score_space must be gene or embedding.")
            coord_key = (source.get("coord_key") or
                         ("X_tsne" if modality == "atac" else "")).strip()
            if space == "embedding" and not coord_key:
                raise ValueError(f"CSV line {line}: embedding scores need coord_key.")
            reference_name = (source.get("reference_npz") or "").strip()
            if reference_name and space != "embedding":
                raise ValueError(f"CSV line {line}: reference_npz applies only to embedding scores.")
            ratio = float((source.get("min_match_ratio") or "").strip() or 0.95)
            if not 0 < ratio <= 1:
                raise ValueError(f"CSV line {line}: min_match_ratio must be in (0, 1].")
            vkey = required(source, "vkey", line)
            gt_vkey = (source.get("gt_vkey") or "").strip() or "ground_truth_velocity"
            basis = coord_key.removeprefix("X_")
            rows.append({
                "modality": modality, "method": required(source, "method", line),
                "dataset_id": required(source, "dataset_id", line),
                "run_id": (source.get("run_id") or "").strip() or "single",
                "path": resolve_path(required(source, "path", line), path.parent),
                "score_space": space, "vkey": vkey, "gt_vkey": gt_vkey,
                "coord_key": coord_key,
                "pred_embedding_key": (source.get("pred_embedding_key") or "").strip()
                    or f"{vkey}_{basis}",
                "gt_embedding_key": (source.get("gt_embedding_key") or "").strip()
                    or ("true_velocity_tsne" if modality == "atac" and basis == "tsne"
                        else f"{gt_vkey}_{basis}"),
                "xkey": (source.get("xkey") or "").strip(),
                "reference_npz": resolve_path(reference_name, path.parent) if reference_name else None,
                "reference_cell_key": (source.get("reference_cell_key") or "").strip(),
                "reference_gt_key": (source.get("reference_gt_key") or "").strip() or "gt_dimred",
                "reference_basis_key": (source.get("reference_basis_key") or "").strip() or "X_basis",
                "min_match_ratio": ratio,
            })
    if not rows:
        raise ValueError("Input CSV contains no samples.")
    identities = [(r["modality"], r["method"], r["dataset_id"], r["run_id"], r["score_space"])
                  for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Input CSV repeats a modality/method/dataset_id/run_id/score_space combination.")
    return rows


def coordinates(adata, key: str) -> np.ndarray:
    if key not in adata.obsm:
        raise KeyError(f"Missing obsm[{key!r}].")
    values = np.asarray(adata.obsm[key], dtype=float)
    if values.ndim != 2 or values.shape[0] != adata.n_obs or values.shape[1] < 2:
        raise ValueError(f"obsm[{key!r}] has invalid shape.")
    if not np.isfinite(values).all():
        raise ValueError(f"obsm[{key!r}] contains non-finite coordinates.")
    return values


def invalidate_graphs(adata, row: dict) -> None:
    adata.uns.pop("neighbors", None)
    for key in ("connectivities", "distances"):
        adata.obsp.pop(key, None)
    for vkey in (row["vkey"], row["gt_vkey"]):
        for suffix in ("_graph", "_graph_neg"):
            key = f"{vkey}{suffix}"
            adata.uns.pop(key, None)
            adata.obsp.pop(key, None)


def align_reference(adata, row: dict):
    """Align explicit NPZ cells; coordinate fallback requires matching coordinates."""
    with np.load(row["reference_npz"], allow_pickle=True) as payload:
        name_key = "cell_names"
        basis_key = row["reference_basis_key"]
        if name_key not in payload or basis_key not in payload:
            raise KeyError(f"Reference NPZ needs {name_key} and {basis_key}.")
        names = np.asarray(payload[name_key]).astype(str)
        basis = np.asarray(payload[basis_key], dtype=float)
        if basis.ndim != 2 or basis.shape[0] != len(names) or basis.shape[1] < 2:
            raise ValueError("Reference coordinates do not match cell_names.")
        if not np.isfinite(basis).all():
            raise ValueError("Reference coordinates are non-finite.")
        if row["reference_cell_key"]:
            key = row["reference_cell_key"]
            if key not in adata.obs:
                raise KeyError(f"Missing obs[{key!r}].")
            result_names = pd.Index(adata.obs[key].astype(str))
        else:
            result_names = pd.Index(adata.obs_names.astype(str))
        if not result_names.is_unique:
            raise ValueError("Result reference cell identifiers are not unique.")
        best = None
        for key in ("cell_names", "cell_names_unique"):
            if key not in payload:
                continue
            reference_names = pd.Index(np.asarray(payload[key]).astype(str))
            if len(reference_names) != len(names) or not reference_names.is_unique:
                continue
            positions = reference_names.get_indexer(result_names)
            selected = np.flatnonzero(positions >= 0)
            if best is None or len(selected) > len(best[0]):
                best = selected, positions[selected]
        if best is None or len(best[0]) / max(adata.n_obs, 1) < row["min_match_ratio"]:
            if row["coord_key"] not in adata.obsm or adata.n_obs != len(names):
                raise ValueError("Reference cells cannot be aligned by names or equal coordinates.")
            existing = coordinates(adata, row["coord_key"])
            if existing.shape != basis.shape or not np.allclose(existing, basis, rtol=1e-5, atol=1e-8):
                raise ValueError("Reference cells cannot be aligned by names or equal coordinates.")
            best = np.arange(adata.n_obs), np.arange(adata.n_obs)
        selected, positions = best
        if len(selected) < adata.n_obs:
            adata = adata[selected].copy()
            invalidate_graphs(adata, row)
        aligned_basis = basis[positions]
        truth = None
        if row["reference_gt_key"] in payload:
            truth = np.asarray(payload[row["reference_gt_key"]], dtype=float)
            if truth.shape != basis.shape or not np.isfinite(truth).all():
                raise ValueError("Reference true velocity does not match reference coordinates.")
            truth = truth[positions]
    old = coordinates(adata, row["coord_key"]) if row["coord_key"] in adata.obsm else None
    changed = old is None or old.shape != aligned_basis.shape or not np.allclose(
        old, aligned_basis, rtol=1e-5, atol=1e-8)
    if changed:
        if row["vkey"] not in adata.layers:
            raise ValueError("Reference coordinates changed; predicted velocity needs its gene-space layer.")
        adata.obsm.pop(row["pred_embedding_key"], None)
        adata.obsm.pop(row["gt_embedding_key"], None)
        basis_name = row["coord_key"].removeprefix("X_")
        for key in (row["vkey"], row["gt_vkey"]):
            adata.obsm.pop(f"{key}_{basis_name}", None)
        invalidate_graphs(adata, row)
    adata.obsm[row["coord_key"]] = aligned_basis.copy()
    if truth is not None:
        adata.obsm[row["gt_embedding_key"]] = truth
    return adata, len(selected), changed


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


def ensure_embedding(adata, row: dict, vkey: str, output_key: str) -> np.ndarray:
    coords = coordinates(adata, row["coord_key"])
    if output_key in adata.obsm:
        values = np.asarray(adata.obsm[output_key], dtype=float)
        if values.shape != coords.shape:
            raise ValueError(f"obsm[{output_key!r}] does not match declared coordinates.")
        return values
    if vkey not in adata.layers:
        raise KeyError(f"Missing both obsm[{output_key!r}] and layers[{vkey!r}].")
    basis = row["coord_key"].removeprefix("X_")
    alias = f"X_{basis}"
    if alias not in adata.obsm:
        adata.obsm[alias] = coords.copy()
    elif alias != row["coord_key"] and not np.allclose(coordinates(adata, alias), coords):
        raise ValueError(f"obsm[{alias!r}] differs from declared coordinates.")
    projection_graph(adata, vkey, row["xkey"], row["coord_key"], 30)
    # scVelo's projection uses np.array on the velocity layer.
    if issparse(adata.layers[vkey]):
        adata.layers[vkey] = adata.layers[vkey].toarray()
    scv.tl.velocity_embedding(adata, basis=basis, vkey=vkey)
    produced = f"{vkey}_{basis}"
    if produced not in adata.obsm:
        raise KeyError(f"Projection did not create obsm[{produced!r}].")
    values = np.asarray(adata.obsm[produced], dtype=float)
    if values.shape != coords.shape:
        raise ValueError(f"Projected velocity {produced!r} has invalid shape.")
    if output_key != produced:
        adata.obsm[output_key] = values.copy()
    return values


def matrix_norms(values) -> np.ndarray:
    if issparse(values):
        matrix = values.tocsr()
        bad = np.flatnonzero(~np.isfinite(matrix.data))
        for index in np.unique(np.searchsorted(matrix.indptr[1:], bad, side="right")):
            start, end = matrix.indptr[index:index + 2]
            entries = matrix.data[start:end]
            if not np.isnan(entries).all() or len(np.unique(matrix.indices[start:end])) != matrix.shape[1]:
                raise ValueError("Velocity matrix contains partially non-finite rows.")
        return np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    matrix = np.asarray(values, dtype=float)
    invalid = ~np.isfinite(matrix).all(axis=1)
    if (invalid & ~np.isnan(matrix).all(axis=1)).any():
        raise ValueError("Velocity matrix contains partially non-finite rows.")
    return np.linalg.norm(matrix, axis=1)


def matrix_dot(left, right) -> np.ndarray:
    if issparse(left):
        return np.asarray(left.multiply(right).sum(axis=1)).ravel()
    if issparse(right):
        return np.asarray(right.multiply(left).sum(axis=1)).ravel()
    return np.einsum("ij,ij->i", np.asarray(left), np.asarray(right))


def cosine_scores(predicted, truth) -> tuple[np.ndarray, np.ndarray, int, int]:
    if predicted.shape != truth.shape or len(predicted.shape) != 2:
        raise ValueError(f"Predicted and true velocity shapes differ: {predicted.shape}, {truth.shape}.")
    pred_norm = matrix_norms(predicted)
    truth_norm = matrix_norms(truth)
    valid = (pred_norm > 0) & (truth_norm > 0)
    if not valid.any():
        raise ValueError("No cells have nonzero predicted and true velocity.")
    scores = np.full(predicted.shape[0], np.nan)
    cosine = matrix_dot(predicted, truth)[valid] / (pred_norm[valid] * truth_norm[valid])
    scores[valid] = (np.clip(cosine, -1, 1) + 1) / 2
    nan_rows = np.isnan(pred_norm) | np.isnan(truth_norm)
    zero_rows = ((pred_norm == 0) | (truth_norm == 0)) & ~nan_rows
    return scores, valid, int(zero_rows.sum()), int(nan_rows.sum())


def score_sample(row: dict):
    adata = sc.read_h5ad(row["path"])
    input_cells = adata.n_obs
    matched = input_cells
    changed = False
    if row["reference_npz"] is not None:
        adata, matched, changed = align_reference(adata, row)
    if row["score_space"] == "gene":
        if row["vkey"] not in adata.layers or row["gt_vkey"] not in adata.layers:
            raise KeyError("Gene scores need predicted and true cell-by-gene velocity layers.")
        predicted, truth = adata.layers[row["vkey"]], adata.layers[row["gt_vkey"]]
    else:
        predicted = ensure_embedding(adata, row, row["vkey"], row["pred_embedding_key"])
        truth = ensure_embedding(adata, row, row["gt_vkey"], row["gt_embedding_key"])
    scores, valid, n_zero, n_nan = cosine_scores(predicted, truth)
    context = {
        "modality": row["modality"], "method": row["method"],
        "dataset_id": row["dataset_id"], "run_id": row["run_id"],
        "score_space": row["score_space"],
        "reference_mode": "npz" if row["reference_npz"] else "result",
    }
    cells = [{**context, "cell_id": str(adata.obs_names[i]), "cell_index": i,
              "score": float(scores[i]) if valid[i] else np.nan}
             for i in range(adata.n_obs)]
    # Preserve the public column name: mean_cosine stores the mapped [0, 1] score.
    total = {**context, "mean_cosine": float(np.nanmean(scores)),
             "n_cells_input": input_cells, "n_cells_matched": matched,
             "n_cells_valid": int(valid.sum()), "n_cells_zero_velocity": n_zero,
             "n_cells_nan_velocity": n_nan,
             "reference_coordinates_changed": changed,
             "vkey": row["vkey"], "gt_vkey": row["gt_vkey"],
             "coord_key": row["coord_key"] if row["score_space"] == "embedding" else ""}
    return total, cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_manifest(args.manifest.expanduser().resolve())
    totals, cells = [], []
    for row in rows:
        total, cell_rows = score_sample(row)
        totals.append(total)
        cells.extend(cell_rows)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(totals).to_csv(output / "groundtruth_correlation.csv", index=False)
    pd.DataFrame(cells).to_csv(output / "groundtruth_correlation_cells.csv", index=False)


if __name__ == "__main__":
    main()
