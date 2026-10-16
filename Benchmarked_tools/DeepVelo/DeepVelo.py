#!/usr/bin/env python3
"""Run DeepVelo with the benchmark settings and the team's Python 3.11 fork.

Install: https://github.com/sd68515/DeepVelo_py311
Input requires spliced/unspliced layers and existing plot coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from copy import deepcopy
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy import sparse


def validate_dataset_id(dataset_id: str) -> str:
    if (not dataset_id or dataset_id in {".", ".."}
            or Path(dataset_id).name != dataset_id or "\\" in dataset_id):
        raise ValueError("Dataset ID must be a nonempty path component")
    return dataset_id


def nonzero_mask(matrix, axis: int) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return np.asarray(matrix.getnnz(axis=axis)).ravel() > 0
    return np.any(np.asarray(matrix) != 0, axis=axis)


def remove_empty(adata, *, cells: bool, genes: bool) -> dict[str, int]:
    counts = {}
    if genes:
        keep = nonzero_mask(adata.layers["spliced"], 0) | nonzero_mask(adata.layers["unspliced"], 0)
        counts["joint_zero_genes"] = int((~keep).sum())
        adata._inplace_subset_var(keep)
    if cells:
        keep = nonzero_mask(adata.layers["spliced"], 1) | nonzero_mask(adata.layers["unspliced"], 1)
        counts["joint_zero_cells"] = int((~keep).sum())
        adata._inplace_subset_obs(keep)
    if not adata.n_obs or not adata.n_vars:
        raise ValueError("No cells or genes remain after removing empty expression profiles")
    return counts


def remove_duplicate_cells(adata) -> int:
    """Deduplicate the joint spliced/unspliced matrix without densifying it."""
    layers = []
    for name in ("spliced", "unspliced"):
        layer = sparse.csr_matrix(adata.layers[name])
        layer.sum_duplicates()
        layer.eliminate_zeros()
        layer.sort_indices()
        layers.append(layer)
    joined = sparse.hstack(layers, format="csr")
    joined.sort_indices()
    keep = np.ones(adata.n_obs, dtype=bool)
    seen: dict[bytes, list[int]] = {}
    for row in range(adata.n_obs):
        start, stop = joined.indptr[row:row + 2]
        digest = hashlib.blake2b(digest_size=20)
        digest.update(np.asarray(joined.indices[start:stop], dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(joined.data[start:stop]).tobytes())
        matches = seen.setdefault(digest.digest(), [])
        if any(
            np.array_equal(joined.indices[start:stop], joined.indices[joined.indptr[other]:joined.indptr[other + 1]])
            and np.array_equal(joined.data[start:stop], joined.data[joined.indptr[other]:joined.indptr[other + 1]])
            for other in matches
        ):
            keep[row] = False
        else:
            matches.append(row)
    removed = int((~keep).sum())
    adata._inplace_subset_obs(keep)
    if not adata.n_obs:
        raise ValueError("No cells remain after duplicate removal")
    return removed


def prepare_adata(adata, kind: str, coordinate_key: str, cluster_key: str) -> dict[str, int]:
    missing = {"spliced", "unspliced"} - set(adata.layers)
    if missing:
        raise ValueError(f"Missing required layers: {sorted(missing)}")
    if cluster_key not in adata.obs:
        raise ValueError(f"Missing adata.obs['{cluster_key}']")
    if coordinate_key not in adata.obsm:
        raise ValueError(f"Missing adata.obsm['{coordinate_key}']")
    coordinates = np.asarray(adata.obsm[coordinate_key])
    if coordinates.ndim != 2 or coordinates.shape[0] != adata.n_obs or coordinates.shape[1] < 2:
        raise ValueError("The coordinate key must contain at least two values per cell")
    if not np.isfinite(coordinates).all():
        raise ValueError("Input coordinates contain non-finite values")
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    original_coordinates = pd.DataFrame(coordinates.copy(), index=adata.obs_names.copy())

    cleanup = remove_empty(adata, cells=True, genes=True)
    cleanup["duplicate_cells_before_filter"] = remove_duplicate_cells(adata)
    adata.X = adata.layers["spliced"].copy()
    if kind == "real":
        scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    elif kind == "sim":
        scv.pp.filter_and_normalize(
            adata, min_counts=None, min_counts_u=None, min_cells=None,
            min_cells_u=None, min_shared_counts=None, min_shared_cells=None,
        )
    else:
        raise ValueError("kind must be 'real' or 'sim'")
    if not adata.n_vars:
        raise ValueError("No genes remain after scVelo filtering")
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars), subset=True)
    cleanup["joint_zero_cells_after_hvg"] = remove_empty(adata, cells=True, genes=False)["joint_zero_cells"]
    keep = nonzero_mask(adata.layers["spliced"], 1)
    cleanup["zero_spliced_cells_after_hvg"] = int((~keep).sum())
    adata._inplace_subset_obs(keep)
    if not adata.n_obs:
        raise ValueError("No cells remain after spliced-count filtering")
    cleanup["duplicate_cells_after_hvg"] = remove_duplicate_cells(adata)
    if adata.n_obs < 30 or adata.n_vars < 30:
        raise ValueError("DeepVelo requires at least 30 cells and 30 genes after preprocessing")
    n_pcs = min(30, adata.n_obs - 1, adata.n_vars - 1)
    n_neighbors = min(30, adata.n_obs - 1)
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    scv.pp.moments(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    adata.obsm[coordinate_key] = original_coordinates.loc[adata.obs_names].to_numpy()
    basis = coordinate_key[2:] if coordinate_key.startswith("X_") else coordinate_key
    adata.obsm[f"X_{basis}"] = np.asarray(adata.obsm[coordinate_key]).copy()
    return cleanup


def sanitize_velocity(adata, maximum_fraction: float = 0.20) -> dict[str, object]:
    """Remove affected cells and genes when at most 20% of elements are invalid."""
    if "velocity" not in adata.layers or adata.layers["velocity"].shape != adata.shape:
        raise ValueError("DeepVelo did not produce a cell-by-gene velocity layer")
    velocity = adata.layers["velocity"]
    bad_cells = np.zeros(adata.n_obs, dtype=bool)
    bad_genes = np.zeros(adata.n_vars, dtype=bool)
    nonfinite_count = 0
    if sparse.issparse(velocity):
        entries = velocity.tocoo(copy=False)
        bad = ~np.isfinite(entries.data)
        nonfinite_count = int(bad.sum())
        bad_cells[entries.row[bad]] = True
        bad_genes[entries.col[bad]] = True
    else:
        values = np.asarray(velocity)
        rows_per_chunk = max(1, 5_000_000 // adata.n_vars)
        for start in range(0, adata.n_obs, rows_per_chunk):
            stop = min(start + rows_per_chunk, adata.n_obs)
            bad = ~np.isfinite(values[start:stop])
            nonfinite_count += int(bad.sum())
            bad_cells[start:stop] = bad.any(axis=1)
            bad_genes |= bad.any(axis=0)
    fraction = nonfinite_count / (adata.n_obs * adata.n_vars)
    if fraction > maximum_fraction:
        raise ValueError(f"Non-finite velocity fraction {fraction:.2%} exceeds {maximum_fraction:.0%}")
    removed_cells, removed_genes = int(bad_cells.sum()), int(bad_genes.sum())
    if removed_cells == adata.n_obs or removed_genes == adata.n_vars:
        raise ValueError("Non-finite velocity filtering would remove all cells or genes")
    if nonfinite_count:
        adata._inplace_subset_obs(~bad_cells)
        adata._inplace_subset_var(~bad_genes)
    return {
        "nonfinite_count": nonfinite_count, "nonfinite_fraction": float(fraction),
        "removed_cells": removed_cells, "removed_genes": removed_genes,
        "n_obs_after": int(adata.n_obs), "n_vars_after": int(adata.n_vars),
    }


def run(input_path: Path, output_dir: Path, kind: str, coordinate_key: str,
        cluster_key: str, dataset_id: str, seed: int = 42, overwrite: bool = False) -> Path:
    import deepvelo as dv
    import torch

    dataset_id = validate_dataset_id(dataset_id)
    output = output_dir / dataset_id / f"{dataset_id}_deepvelo.h5ad"
    if output.exists() and not overwrite:
        print(f"Skipping existing output: {output}")
        return output
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    adata = ad.read_h5ad(input_path)
    cleanup = prepare_adata(adata, kind, coordinate_key, cluster_key)
    from deepvelo.utils.preprocess import autoset_coeff_s
    loss_coeff_s = float(autoset_coeff_s(adata))
    if loss_coeff_s not in {0.5, 0.75, 1.0}:
        raise ValueError(f"Unsupported DeepVelo loss coefficient: {loss_coeff_s}")
    configs = deepcopy(dv.Constants.default_configs)
    configs["name"] = f"DeepVelo_{dataset_id}"
    configs["n_gpu"] = int(torch.cuda.is_available())
    configs["loss"]["args"]["coeff_s"] = loss_coeff_s
    configs["data_loader"]["args"].update(
        shuffle=False, validation_split=0.0, num_workers=0, topC=30, topG=20,
    )
    configs["trainer"].update(
        epochs=100, save_dir=str(output.parent / "saved"),
        save_period=1000, tensorboard=False, verbosity=0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    dv.train(adata, configs)
    for layer in ("cell_specific_beta", "cell_specific_gamma"):
        if layer not in adata.layers or adata.layers[layer].shape != adata.shape:
            raise ValueError(f"DeepVelo did not create a valid '{layer}' layer")
    sanitization = sanitize_velocity(adata)
    basis = coordinate_key[2:] if coordinate_key.startswith("X_") else coordinate_key
    scv.tl.velocity_graph(adata, vkey="velocity")
    scv.tl.velocity_embedding(adata, vkey="velocity", basis=basis)
    scv.tl.velocity_pseudotime(adata, vkey="velocity")
    adata.uns["deepvelo_run"] = {
        "dataset_id": dataset_id, "kind": kind, "coordinate_key": coordinate_key,
        "cluster_key": cluster_key, "seed": int(seed), "epochs": 100,
        "topC": 30, "topG": 20, "shuffle": False,
        "validation_split": 0.0, "loss_coeff_s": loss_coeff_s,
        "cleanup": cleanup, "velocity_sanitization": sanitization,
    }
    adata.write_h5ad(output, compression="lzf")
    print(f"Saved: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="Input H5AD file")
    inputs.add_argument("--metadata-file", type=Path, help="CSV with dataset_name,file_path,modality,cluster_key,dimred_key")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--kind", choices=("real", "sim"))
    parser.add_argument("--coordinate-key", help="Existing adata.obsm key")
    parser.add_argument("--cluster-key", help="Existing adata.obs key")
    parser.add_argument("--dataset-id", help="Output identifier; defaults to the input stem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.metadata_file:
        table = pd.read_csv(args.metadata_file, dtype=str, keep_default_na=False)
        required = ("dataset_name", "file_path", "modality", "cluster_key", "dimred_key")
        if any(column not in table for column in required):
            parser.error(f"Metadata file requires columns: {', '.join(required)}")
        if table.empty:
            parser.error("Metadata file contains no datasets")
        for column in required:
            table[column] = table[column].str.strip()
        if table["dataset_name"].duplicated().any():
            raise ValueError("Metadata dataset IDs must be unique")
        for row in table.to_dict("records"):
            if any(pd.isna(row[key]) or not str(row[key]).strip() for key in required):
                raise ValueError("Metadata rows must supply every required field")
            path = Path(row["file_path"])
            if not path.is_absolute():
                path = args.metadata_file.parent / path
            run(path, args.output_dir, row["modality"], row["dimred_key"],
                row["cluster_key"], row["dataset_name"], args.seed, args.overwrite)
    else:
        if not args.kind or not args.coordinate_key or not args.cluster_key:
            parser.error("--kind, --coordinate-key, and --cluster-key are required with --input")
        run(args.input, args.output_dir, args.kind, args.coordinate_key,
            args.cluster_key, args.dataset_id or args.input.stem, args.seed, args.overwrite)


if __name__ == "__main__":
    main()
