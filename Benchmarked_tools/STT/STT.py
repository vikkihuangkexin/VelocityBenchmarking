"""Run scRNA or spatial STT with https://github.com/sd68515/STT_benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import scanpy as sc
import scvelo as scv
from scipy import sparse

import stt as st


def ensure_spatial_basis(adata) -> None:
    """Provide the X_ coordinate alias required by scVelo's spatial basis."""
    if "X_spatial" not in adata.obsm:
        if "spatial" not in adata.obsm:
            raise KeyError("Missing obsm['spatial'] for scVelo basis='spatial'")
        adata.obsm["X_spatial"] = adata.obsm["spatial"].copy()


def choose_simulated_top_genes(n_vars: int) -> int:
    """Use the gene-count rule from the original scRNA simulation runner."""
    return min((n_vars // 500) * 500, n_vars, 500) if n_vars < 10000 else 2000


def dataset_id(value: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"Dataset ID must be one path component: {value!r}")
    return value


def parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected true or false, got {value!r}")


def read_manifest(path: Path) -> list[dict]:
    required = ("dataset_name", "file_path", "modality", "cluster_key", "dimred_key")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(key not in reader.fieldnames for key in required):
            raise ValueError(f"CSV requires columns: {', '.join(required)}")
        rows, seen = [], set()
        for number, source in enumerate(reader, start=2):
            if None in source:
                raise ValueError(f"CSV row {number} has more values than its header")
            row = {key: (value or "").strip() for key, value in source.items()}
            if any(not row[key] for key in required):
                raise ValueError(f"CSV row {number} is missing a required value")
            row["dataset_id"] = dataset_id(row["dataset_name"])
            row["coordinate_key"] = row["dimred_key"]
            if row["dataset_id"] in seen:
                raise ValueError(f"Duplicate dataset ID: {row['dataset_id']}")
            seen.add(row["dataset_id"])
            if row["modality"] not in {"real", "sim", "spatial-real", "spatial-sim"}:
                raise ValueError(f"CSV row {number}: unsupported modality")
            input_path = Path(row["file_path"])
            row["file_path"] = input_path if input_path.is_absolute() else path.parent / input_path
            rows.append(row)
    if not rows:
        raise ValueError("CSV contains no datasets")
    return rows


def validate_input(adata, name: str, cluster_key: str, coordinate_key: str) -> None:
    if cluster_key not in adata.obs or adata.obs[cluster_key].isna().any():
        raise ValueError(f"{name}: missing or incomplete obs[{cluster_key!r}]")
    if coordinate_key not in adata.obsm:
        raise ValueError(f"{name}: missing obsm[{coordinate_key!r}]")
    if any(layer not in adata.layers for layer in ("spliced", "unspliced")):
        raise ValueError(f"{name}: spliced and unspliced layers are required")
    coordinates = np.asarray(adata.obsm[coordinate_key])
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"{name}: obsm[{coordinate_key!r}] must contain finite 2D coordinates")


def prepare_scrna(adata, cluster_key: str, coordinate_key: str, simulate: bool, args) -> dict:
    if simulate:
        top_genes = choose_simulated_top_genes(adata.n_vars)
        if top_genes < 1:
            raise ValueError("The scRNA simulation gene-count rule selected no genes")
        scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=top_genes)
        if "X_umap" not in adata.obsm:
            adata.obsm["X_umap"] = np.asarray(adata.obsm[coordinate_key]).copy()
    else:
        top_genes = 2000
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=top_genes)
    scv.pp.moments(adata, n_neighbors=args.moments_n_neighbors)
    adata.obs["attractor"] = adata.obs[cluster_key].astype("category")
    return {"simulate": simulate, "n_top_genes": top_genes,
            "min_shared_counts": "none" if simulate else 20,
            "moments_n_neighbors": args.moments_n_neighbors}


def prepare_spatial(adata, name: str, cluster_key: str, coordinate_key: str):
    original_cells = adata.n_obs
    counts = adata.obs[cluster_key].value_counts()
    rare = counts[counts < 10].index
    adata = adata[~adata.obs[cluster_key].isin(rare)].copy()
    if adata.n_obs == 0:
        raise ValueError(f"{name}: rare-class filtering removed every cell")
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category").cat.remove_unused_categories()
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.obs["attractor"] = adata.obs[cluster_key].astype("category")
    adata.obsm["spatial"] = np.asarray(adata.obsm[coordinate_key]).copy()
    ensure_spatial_basis(adata)
    if adata.n_vars > 2000:
        sc.pp.highly_variable_genes(adata, layer="spliced", flavor="seurat_v3", n_top_genes=2000)
        adata = adata[:, adata.var["highly_variable"]].copy()
    scv.pp.filter_and_normalize(adata, min_shared_counts=1)
    n_pcs = min(30, adata.n_vars - 1, adata.n_obs - 1)
    if n_pcs < 2:
        raise ValueError(f"{name}: too few cells or genes for spatial STT PCA")
    scv.pp.neighbors(adata, n_neighbors=min(30, adata.n_obs - 1), n_pcs=n_pcs)
    scv.pp.moments(adata)
    adata.obsm["X_stt_spatial"] = np.asarray(adata.obsm[coordinate_key]).copy()
    sc.pp.neighbors(adata, use_rep="X_stt_spatial", n_neighbors=min(20, adata.n_obs - 1),
                    key_added="spatial")
    graph = adata.obsp["spatial_connectivities"]
    if graph.nnz - np.count_nonzero(graph.diagonal()) == 0:
        raise ValueError(f"{name}: spatial graph has no between-cell edges")
    return adata, {"hvg_flavor": "seurat_v3", "hvg_top_genes": 2000,
                   "min_shared_counts": 1, "spatial_neighbors": 20,
                   "min_cells_per_type": 10,
                   "rare_class_cells_removed": original_cells - adata.n_obs}


def run_one(row: dict, output_dir: Path, args) -> Path:
    name, modality = row["dataset_id"], row["modality"]
    cluster_key, coordinate_key = row["cluster_key"], row["coordinate_key"]
    destination = output_dir / name / "adata_aggr.h5ad"
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    adata = sc.read_h5ad(row["file_path"])
    validate_input(adata, name, cluster_key, coordinate_key)
    if modality.startswith("spatial-"):
        adata, preprocessing = prepare_spatial(adata, name, cluster_key, coordinate_key)
        n_states = int(adata.obs["attractor"].nunique())
        settings = {"n_states": n_states, "n_iter": 15,
                    "weight_connectivities": 0.5,
                    "n_components": max(20, n_states), "n_neighbors": 20,
                    "thresh_ms_gene": args.spatial_thresh_ms_gene,
                    "use_spatial": True, "spa_weight": 0.3,
                    "spa_conn_key": "spatial", "thresh_entropy": 0.1,
                    "monitor_mode": False, "n_jobs": args.spatial_n_jobs}
    else:
        preprocessing = prepare_scrna(adata, cluster_key, coordinate_key, modality == "sim", args)
        n_states = (args.dyn_n_states if args.dyn_n_states is not None
                    else int(adata.obs["attractor"].nunique()))
        settings = {"n_states": n_states, "n_iter": args.dyn_n_iter,
                    "weight_connectivities": args.dyn_weight_connectivities,
                    "n_components": args.dyn_n_components,
                    "n_neighbors": args.dyn_n_neighbors,
                    "thresh_ms_gene": args.dyn_thresh_ms_gene,
                    "use_spatial": args.dyn_use_spatial,
                    "spa_weight": args.dyn_spa_weight,
                    "thresh_entropy": args.dyn_thresh_entropy}
    if n_states < 2:
        raise ValueError(f"{name}: STT requires at least two annotated states")
    result = st.tl.dynamical_iteration(adata, return_aggr_obj=True, **settings)
    if result is None or "vj" not in result.layers or "Ms" not in result.layers:
        raise ValueError(f"{name}: STT did not return vj and Ms layers")
    if result.n_obs != adata.n_obs or not result.obs_names.equals(adata.obs_names):
        raise ValueError(f"{name}: aggregated cell order differs from the input")
    for key in ("vj", "Ms"):
        if result.layers[key].shape != result.shape:
            raise ValueError(f"{name}: {key} layer shape differs from the result")
    velocity = result.layers["vj"]
    values = velocity.data if sparse.issparse(velocity) else np.asarray(velocity)
    if not np.isfinite(values).all() or not np.any(values != 0):
        raise ValueError(f"{name}: vj contains non-finite values or is entirely zero")
    result.obs[cluster_key] = adata.obs[cluster_key].to_numpy()
    result.obsm[coordinate_key] = np.asarray(adata.obsm[coordinate_key]).copy()
    if modality.startswith("spatial-"):
        result.obsm["spatial"] = np.asarray(adata.obsm[coordinate_key]).copy()
        ensure_spatial_basis(result)
        result.obsm.pop("macrostates_fwd_memberships", None)
    elif modality == "sim":
        result.obsm["X_umap"] = np.asarray(adata.obsm["X_umap"]).copy()
    result.uns["stt_run_config"] = {
        "dataset_id": name, "modality": modality,
        "cluster_key": cluster_key, "coordinate_key": coordinate_key,
        "cells_after_filter": adata.n_obs, "genes_after_filter": adata.n_vars,
        **preprocessing, **settings,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="One H5AD input")
    inputs.add_argument("--metadata-file", type=Path, help="CSV with explicit annotation and coordinate keys")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modality", choices=("real", "sim", "spatial-real", "spatial-sim"), help="Required with --input")
    parser.add_argument("--dataset-id", help="Output ID; defaults to the input filename stem")
    parser.add_argument("--cluster-key", help="Existing adata.obs annotation key")
    parser.add_argument("--coordinate-key", help="Existing adata.obsm coordinate key")
    parser.add_argument("--moments-n-neighbors", type=int, default=50)
    parser.add_argument("--dyn-n-states", type=int)
    parser.add_argument("--dyn-n-iter", type=int, default=15)
    parser.add_argument("--dyn-weight-connectivities", type=float, default=0.5)
    parser.add_argument("--dyn-n-components", type=int, default=21)
    parser.add_argument("--dyn-n-neighbors", type=int, default=20)
    parser.add_argument("--dyn-thresh-ms-gene", type=float, default=0.2)
    parser.add_argument("--dyn-use-spatial", type=parse_boolean, default=True)
    parser.add_argument("--dyn-spa-weight", type=float, default=0.3)
    parser.add_argument("--dyn-thresh-entropy", type=float, default=0.1)
    parser.add_argument("--spatial-thresh-ms-gene", type=float, default=0)
    parser.add_argument("--spatial-n-jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.spatial_n_jobs < 1:
        parser.error("--spatial-n-jobs must be positive")
    if args.metadata_file:
        rows = read_manifest(args.metadata_file)
    else:
        if not args.modality or not args.cluster_key or not args.coordinate_key:
            parser.error("--modality, --cluster-key, and --coordinate-key are required with --input")
        rows = [{"dataset_id": dataset_id(args.dataset_id or args.input.stem),
                 "file_path": args.input, "modality": args.modality,
                 "cluster_key": args.cluster_key, "coordinate_key": args.coordinate_key}]
    for row in rows:
        print(run_one(row, args.output_dir, args), flush=True)


if __name__ == "__main__":
    main()
