"""Spatial velocity and time consistency, with optional excess velocity consistency.

The input H5AD is read without changing its graphs, layers, or annotations.
All scores use a symmetric, binary spatial KNN graph built from the declared
coordinates. ESVC is an exploratory diagnostic and is disabled by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
from scipy.spatial import cKDTree


ESVC_FIELDS = (
    "vc_spatial_global", "vc_null_global", "esvc_global", "global_null_cells",
    "vc_spatial_matched", "vc_null_label_matched", "esvc_label_matched",
    "matched_cells", "local_edges", "matched_edges", "matched_edge_fraction",
    "valid_velocity_cells",
)
BASE_FIELDS = (
    "data_type", "method", "id", "run_id", "path", "vkey", "tkey",
    "basis_name", "cluster_key", "n_cells", "velocity_consistency",
    "time_consistency", "velocity_cells", "time_cells", "status", "error",
)


@dataclass(frozen=True)
class Sample:
    data_type: str
    method: str
    dataset_id: str
    run_id: str
    path: Path
    vkey: str
    tkey: str
    basis_name: str
    cluster_key: str


def read_manifest(path: Path) -> list[Sample]:
    """Read the public metric manifest without study-specific ID restrictions."""
    samples = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"method", "id", "path", "vkey", "data_type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Input CSV needs columns: {', '.join(sorted(required))}.")
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"CSV line {line} has more values than its header.")
            values = {key: (value or "").strip() for key, value in row.items() if key is not None}
            missing = [key for key in required if not values.get(key)]
            if missing:
                raise ValueError(f"CSV line {line} needs {', '.join(sorted(missing))}.")
            data_type = values["data_type"].lower()
            if data_type not in {"real", "sim"}:
                raise ValueError(f"CSV line {line}: data_type must be real or sim.")
            modality = values.get("modality", "spatial").lower() or "spatial"
            if modality != "spatial":
                raise ValueError(f"CSV line {line}: modality must be spatial.")
            source = Path(values["path"]).expanduser()
            if not source.is_absolute():
                source = path.parent / source
            vkey = values["vkey"]
            samples.append(Sample(
                data_type=data_type,
                method=values["method"],
                dataset_id=values["id"],
                run_id=values.get("run_id", "") or "single",
                path=source,
                vkey=vkey,
                tkey=values.get("tkey", "") or f"{vkey}_pseudotime",
                basis_name=values.get("basis_name", "") or "spatial",
                cluster_key=values.get("cluster_key", ""),
            ))
    if not samples:
        raise ValueError("Input CSV contains no samples.")
    identities = [(s.data_type, s.method, s.dataset_id, s.run_id) for s in samples]
    if len(identities) != len(set(identities)):
        raise ValueError("Input CSV repeats a data_type/method/id/run_id combination.")
    return samples


def coordinate_key(basis_name: str) -> str:
    return basis_name if basis_name == "spatial" or basis_name.startswith("X_") else f"X_{basis_name}"


def read_matrix(node: h5py.Dataset | h5py.Group) -> np.ndarray:
    """Read a dense H5AD matrix or a standard CSR/CSC encoded matrix."""
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[()])
    encoding = node.attrs.get("encoding-type")
    if encoding not in {"csr_matrix", "csc_matrix"}:
        raise ValueError(f"Unsupported H5AD matrix encoding at {node.name}: {encoding!r}")
    shape = tuple(int(value) for value in node.attrs["shape"])
    payload = (node["data"][()], node["indices"][()], node["indptr"][()])
    matrix_type = csr_matrix if encoding == "csr_matrix" else csc_matrix
    return matrix_type(payload, shape=shape).toarray()


def read_obs_column(handle: h5py.File, key: str) -> np.ndarray:
    """Read only the requested obs column, including standard nullable encodings."""
    node = handle["obs"][key]
    if isinstance(node, h5py.Dataset):
        values = node.asstr()[()] if node.dtype.kind in {"O", "S", "U"} else node[()]
        return np.asarray(values)
    encoding = node.attrs.get("encoding-type")
    if encoding == "categorical":
        codes = np.asarray(node["codes"][()], dtype=np.int64)
        categories_node = node["categories"]
        categories = (categories_node.asstr()[()] if h5py.check_string_dtype(categories_node.dtype)
                      else categories_node[()])
        values = np.empty(len(codes), dtype=object)
        values[:] = np.nan
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values
    if encoding in {"nullable-integer", "nullable-boolean", "nullable-float"}:
        values = np.asarray(node["values"][()]).astype(object)
        values[np.asarray(node["mask"][()], dtype=bool)] = np.nan
        return values
    raise ValueError(f"Unsupported H5AD obs encoding at {node.name}: {encoding!r}")


def spatial_neighbors(coordinates: np.ndarray, n_neighbors: int = 14) -> csr_matrix:
    """Union of each cell's spatial KNN edges; symmetric, binary, no self edges."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("Spatial coordinates must have at least two columns.")
    if len(coordinates) < 3 or not np.isfinite(coordinates).all():
        raise ValueError("Spatial coordinates need at least three finite cells.")
    n = len(coordinates)
    k = min(n_neighbors, n - 1)
    _, indices = cKDTree(coordinates).query(coordinates, k=min(n, k + 2), workers=1)
    rows = np.repeat(np.arange(n), k)
    cols = np.empty(n * k, dtype=np.int64)
    for index, candidates in enumerate(indices):
        cols[index * k:(index + 1) * k] = candidates[candidates != index][:k]
    graph = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def normalized_velocity(velocity: np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Center and normalize rows for Pearson correlation across velocity genes."""
    values = np.asarray(velocity, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != n_cells or values.shape[1] < 2:
        raise ValueError("Velocity layer must have shape n_cells x n_genes with >=2 genes.")
    valid = np.isfinite(values).all(axis=1)
    unit = values.copy()
    unit[~valid] = 0.0
    unit -= unit.mean(axis=1, keepdims=True)
    scales = np.linalg.norm(unit, axis=1)
    valid &= np.isfinite(scales) & (scales > 0)
    unit[valid] /= scales[valid, None]
    unit[~valid] = 0.0
    return unit, valid


def velocity_consistency(
    velocity: np.ndarray,
    graph: csr_matrix,
    *,
    annotations: np.ndarray | None = None,
    include_esvc: bool = False,
) -> dict[str, float | int]:
    """Mean local Pearson r and, optionally, two exact non-neighbor nulls.

    VC is the mean of each eligible cell's mean neighbor correlation. ESVC
    retains that cell weighting and subtracts the exact mean correlation to
    non-neighbors. The annotation-matched null preserves each local edge's
    target annotation, including edges between different annotations.
    """
    unit, valid = normalized_velocity(velocity, graph.shape[0])
    rows, cols = graph.nonzero()
    edge_valid = valid[rows] & valid[cols]
    rows, cols = rows[edge_valid], cols[edge_valid]
    correlations = np.empty(len(rows), dtype=np.float64)
    for start in range(0, len(rows), 2048):
        sl = slice(start, start + 2048)
        correlations[sl] = np.clip(
            np.einsum("ij,ij->i", unit[rows[sl]], unit[cols[sl]]), -1.0, 1.0
        )
    degree = np.bincount(rows, minlength=len(unit))
    local_sum = np.bincount(rows, weights=correlations, minlength=len(unit))
    used = degree > 0
    local_mean = np.zeros(len(unit), dtype=np.float64)
    local_mean[used] = local_sum[used] / degree[used]
    result: dict[str, float | int] = {
        "velocity_consistency": float(local_mean[used].mean()) if used.any() else math.nan,
        "velocity_cells": int(used.sum()),
    }
    if not include_esvc:
        return result

    result.update({
        "vc_spatial_global": math.nan, "vc_null_global": math.nan,
        "esvc_global": math.nan, "global_null_cells": 0,
        "vc_spatial_matched": math.nan, "vc_null_label_matched": math.nan,
        "esvc_label_matched": math.nan, "matched_cells": 0,
        "local_edges": int(len(rows)), "matched_edges": 0,
        "matched_edge_fraction": 0.0, "valid_velocity_cells": int(valid.sum()),
    })
    if not used.any():
        return result

    # Exclude the anchor and its valid graph neighbors. The local edge sum is
    # subtracted from the all-cell dot product, giving the exact null mean.
    candidate_count = int(valid.sum()) - 1 - degree
    global_used = used & (candidate_count > 0)
    if global_used.any():
        all_dot = unit[global_used] @ unit.sum(axis=0)
        null = (all_dot - 1.0 - local_sum[global_used]) / candidate_count[global_used]
        spatial = local_mean[global_used]
        result.update(
            vc_spatial_global=float(spatial.mean()),
            vc_null_global=float(null.mean()),
            esvc_global=float(np.mean(spatial - null)),
            global_null_cells=int(global_used.sum()),
        )

    if annotations is None:
        return result
    if len(annotations) != len(unit):
        raise ValueError("Annotation count differs from the velocity cell count.")
    labels, _ = pd.factorize(pd.Series(annotations).astype("string").str.strip(), sort=True)
    labels = np.asarray(labels, dtype=np.int64)
    target_labels = labels[cols]
    annotated = target_labels >= 0
    label_count = int(labels.max()) + 1 if (labels >= 0).any() else 0
    if not label_count or not annotated.any():
        return result

    labeled_rows = rows[annotated]
    labeled_corr = correlations[annotated]
    target_labels = target_labels[annotated]
    flat = labeled_rows * label_count + target_labels
    local_count_by_label = np.bincount(flat, minlength=len(unit) * label_count).reshape(len(unit), label_count)
    local_sum_by_label = np.bincount(
        flat, weights=labeled_corr, minlength=len(unit) * label_count
    ).reshape(len(unit), label_count)
    group_count = np.bincount(labels[valid & (labels >= 0)], minlength=label_count)
    matched_count = np.zeros(len(unit), dtype=np.int64)
    matched_spatial_sum = np.zeros(len(unit), dtype=np.float64)
    matched_null_sum = np.zeros(len(unit), dtype=np.float64)
    for label in range(label_count):
        local_count = local_count_by_label[:, label]
        self_in_group = (labels == label) & valid
        eligible_count = group_count[label] - local_count - self_in_group.astype(np.int64)
        anchors = np.flatnonzero((local_count > 0) & (eligible_count > 0))
        if not len(anchors):
            continue
        # For each local i->j edge, replace j with every valid non-neighbor
        # sharing j's annotation. The spatial and null terms retain the same
        # edges and are averaged per anchor before pooling across anchors.
        group_sum = unit[valid & (labels == label)].sum(axis=0)
        anchor_dot = unit[anchors] @ group_sum
        null_mean = (
            anchor_dot - local_sum_by_label[anchors, label]
            - self_in_group[anchors].astype(np.float64)
        ) / eligible_count[anchors]
        matched_count[anchors] += local_count[anchors]
        matched_spatial_sum[anchors] += local_sum_by_label[anchors, label]
        matched_null_sum[anchors] += local_count[anchors] * null_mean
    matched = matched_count > 0
    result["matched_edges"] = int(matched_count.sum())
    result["matched_edge_fraction"] = float(matched_count.sum() / len(rows))
    result["matched_cells"] = int(matched.sum())
    if matched.any():
        spatial = matched_spatial_sum[matched] / matched_count[matched]
        null = matched_null_sum[matched] / matched_count[matched]
        result.update(
            vc_spatial_matched=float(spatial.mean()),
            vc_null_label_matched=float(null.mean()),
            esvc_label_matched=float(np.mean(spatial - null)),
        )
    return result


def time_consistency(times: np.ndarray, graph: csr_matrix) -> tuple[float, int]:
    """Global Moran's I with binary spatial weights and finite times."""
    values = np.asarray(times, dtype=np.float64).ravel()
    if len(values) != graph.shape[0]:
        raise ValueError("Time column length differs from the spatial cell count.")
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return math.nan, int(valid.sum())
    centered = values[valid] - values[valid].mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        return math.nan, int(valid.sum())
    subgraph = graph[valid][:, valid]
    if subgraph.nnz == 0:
        return math.nan, int(valid.sum())
    numerator = float(centered @ (subgraph @ centered))
    return float(len(centered) / subgraph.nnz * numerator / denominator), int(valid.sum())


def score_sample(sample: Sample, include_esvc: bool, n_neighbors: int) -> dict[str, object]:
    row: dict[str, object] = {
        "data_type": sample.data_type, "method": sample.method, "id": sample.dataset_id,
        "run_id": sample.run_id, "path": str(sample.path), "vkey": sample.vkey,
        "tkey": sample.tkey, "basis_name": sample.basis_name,
        "cluster_key": sample.cluster_key, "n_cells": 0,
        "velocity_consistency": math.nan, "time_consistency": math.nan,
        "velocity_cells": 0, "time_cells": 0, "status": "failed", "error": "",
    }
    if include_esvc:
        row.update({key: math.nan if key.startswith(("vc_", "esvc_")) else 0 for key in ESVC_FIELDS})
    if not sample.path.is_file():
        row.update(status="missing", error="Input H5AD not found.")
        return row

    errors = []
    try:
        with h5py.File(sample.path, "r") as handle:
            coordinates = read_matrix(handle["obsm"][coordinate_key(sample.basis_name)])
            graph = spatial_neighbors(coordinates, n_neighbors)
            row["n_cells"] = graph.shape[0]
            try:
                velocity = read_matrix(handle["layers"][sample.vkey])
                annotation = None
                if include_esvc and sample.cluster_key:
                    try:
                        annotation = read_obs_column(handle, sample.cluster_key)
                    except (KeyError, TypeError, ValueError) as exc:
                        errors.append(f"Annotation unavailable: {exc}")
                row.update(velocity_consistency(
                    velocity, graph, annotations=annotation, include_esvc=include_esvc
                ))
                if include_esvc and not sample.cluster_key:
                    errors.append("cluster_key is empty; annotation-matched ESVC unavailable")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"Velocity score unavailable: {exc}")
            try:
                times = read_obs_column(handle, sample.tkey)
                row["time_consistency"], row["time_cells"] = time_consistency(times, graph)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"Time score unavailable: {exc}")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        row["error"] = f"Input or spatial graph unavailable: {exc}"
        return row

    valid_velocity = np.isfinite(row["velocity_consistency"])
    valid_time = np.isfinite(row["time_consistency"])
    valid_esvc = not include_esvc or (
        np.isfinite(row["esvc_global"]) and np.isfinite(row["esvc_label_matched"])
    )
    row["status"] = (
        "ok" if valid_velocity and valid_time and valid_esvc else
        "partial" if valid_velocity or valid_time else "failed"
    )
    if not valid_velocity:
        errors.append("Velocity consistency is undefined")
    if not valid_time:
        errors.append("Time consistency is undefined")
    if include_esvc and not valid_esvc:
        errors.append("One or both ESVC null comparisons are unavailable")
    row["error"] = "; ".join(errors)
    return row


def write_csv(frame: pd.DataFrame, destination: Path) -> None:
    """Replace a completed CSV atomically, including checkpoint files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=f".{destination.stem}_",
        dir=destination.parent, delete=False, newline="", encoding="utf-8",
    ) as stream:
        temporary = Path(stream.name)
        frame.to_csv(stream, index=False, float_format="%.8g", na_rep="NA")
    os.replace(temporary, destination)


def wide_table(frame: pd.DataFrame, samples: list[Sample], data_type: str, field: str) -> pd.DataFrame:
    methods = list(dict.fromkeys(s.method for s in samples if s.data_type == data_type))
    datasets = list(dict.fromkeys(s.dataset_id for s in samples if s.data_type == data_type))
    selected = frame[frame["data_type"] == data_type].copy()
    selected[field] = pd.to_numeric(selected[field], errors="coerce")
    # Keep repeated runs distinct in the long table; a wide cell is their mean.
    wide = selected.groupby(["method", "id"], sort=False)[field].mean().unstack()
    wide = wide.reindex(index=methods, columns=datasets)
    ranking = wide.mean(axis=1, skipna=True).sort_values(ascending=False, kind="stable", na_position="last")
    return wide.loc[ranking.index]


def save_heatmap(wide: pd.DataFrame, field: str, destination: Path) -> None:
    values = wide.to_numpy(dtype=float)
    is_esvc = field.startswith("esvc_")
    display = np.where(np.isfinite(values) & (np.abs(values) < 0.00005), 0.0, values)
    positives = display[np.isfinite(display) & (display > 0)]
    upper = max(0.1, math.ceil(float(positives.max()) * 10) / 10) if is_esvc and len(positives) else 1.0
    cmap = LinearSegmentedColormap.from_list("spatial_blue", ["#f2f7fa", "#a9cfe2", "#4b91bb", "#174f78"])
    cmap.set_under("#a44c0b")
    cmap.set_over("#102e46")
    cmap.set_bad("#d9d9d9")
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(wide.columns) + 2), max(3.5, 0.35 * len(wide.index) + 1.5)))
    mesh = ax.pcolormesh(np.ma.masked_invalid(display), cmap=cmap, norm=Normalize(vmin=0, vmax=upper),
                         edgecolors="white", linewidth=0.25)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(wide.columns)) + 0.5, wide.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(wide.index)) + 0.5, wide.index)
    titles = {
        "velocity_consistency": "Spatial velocity consistency",
        "time_consistency": "Spatial time consistency",
        "esvc_label_matched": "Excess spatial velocity consistency: annotation-matched non-neighbor null",
        "esvc_global": "Excess spatial velocity consistency: all-cell non-neighbor null",
    }
    ax.set_title(titles[field])
    ax.set_xlabel("Dataset ID")
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            value = display[i, j]
            label = (f"{value:.4f}" if is_esvc and abs(value) < 0.01 else f"{value:.3f}") if np.isfinite(value) else "NA"
            rgba = cmap(mesh.norm(value)) if np.isfinite(value) else cmap.get_bad()
            luminance = sum(
                weight * (channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4)
                for weight, channel in zip((0.2126, 0.7152, 0.0722), rgba[:3])
            )
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=6,
                    color="white" if luminance < 0.179 else "black")
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02,
                            extend="min" if np.any(display < 0) else "max" if np.any(display > upper) else "neither")
    colorbar.set_label("Pearson r difference" if is_esvc else "Moran's I" if field == "time_consistency" else "Pearson r")
    fig.tight_layout()
    for suffix in ("svg", "png"):
        fig.savefig(destination.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-file", type=Path, required=True, help="Spatial metric input manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--esvc", action="store_true", help="also compute exploratory excess spatial velocity consistency")
    parser.add_argument("--n-neighbors", type=int, default=14, help="spatial KNN count (default: 14)")
    parser.add_argument("--jobs", type=int, default=2, help="parallel H5AD workers (default: 2)")
    parser.add_argument("--no-plots", action="store_true", help="write CSV files without heatmaps")
    args = parser.parse_args()
    if args.n_neighbors < 1 or args.jobs < 1:
        parser.error("--n-neighbors and --jobs must be positive")
    samples = read_manifest(args.csv_file)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    fields = (*BASE_FIELDS, *ESVC_FIELDS) if args.esvc else BASE_FIELDS
    rows: list[dict[str, object] | None] = [None] * len(samples)
    checkpoint = output / "spatial_consistency_results.csv"
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        pending = {
            pool.submit(score_sample, sample, args.esvc, args.n_neighbors): index
            for index, sample in enumerate(samples)
        }
        for future in as_completed(pending):
            index = pending[future]
            rows[index] = future.result()
            write_csv(pd.DataFrame((row for row in rows if row is not None), columns=fields), checkpoint)
            print(f"[{sum(row is not None for row in rows)}/{len(rows)}] {samples[index].method} "
                  f"{samples[index].dataset_id}: {rows[index]['status']}", flush=True)
    frame = pd.DataFrame(rows, columns=fields)
    write_csv(frame[frame["status"] != "ok"], output / "spatial_consistency_failures.csv")
    scores = ["velocity_consistency", "time_consistency"]
    if args.esvc:
        scores.extend(("esvc_label_matched", "esvc_global"))
    for data_type in dict.fromkeys(sample.data_type for sample in samples):
        for field in scores:
            wide = wide_table(frame, samples, data_type, field)
            filename = output / f"spatial_{field}_{data_type}.csv"
            write_csv(wide.rename_axis("method").reset_index(), filename)
            if not args.no_plots:
                save_heatmap(wide, field, filename.with_suffix(""))
    return int(frame["status"].isin(["failed", "missing"]).any())


if __name__ == "__main__":
    raise SystemExit(main())
