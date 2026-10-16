#!/usr/bin/env python3
"""Independent DeepVelo repetitions from a CSV manifest.

Install the team fork: https://github.com/sd68515/DeepVelo_py311
CSV columns: dataset_name,file_path,modality,cluster_key,dimred_key.
Modality is real or sim. Input paths
may be relative to the CSV. Each repetition saves a separate h5ad file.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--preprocess-seed", type=int, default=2024)
    return parser.parse_args()


def load_manifest(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset_name", "file_path", "modality", "cluster_key", "dimred_key"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        rows, seen = [], set()
        for line, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError(f"CSV line {line} has more values than its header")
            row = {
                "ID": (raw.get("dataset_name") or "").strip(),
                "path": (raw.get("file_path") or "").strip(),
                "kind": (raw.get("modality") or "").strip(),
                "obs": (raw.get("cluster_key") or "").strip(),
                "obsm_vis": (raw.get("dimred_key") or "").strip(),
            }
            ident = row["ID"]
            if not ident or ident in {".", ".."} or Path(ident).name != ident or "\\" in ident:
                raise ValueError(f"Line {line}: ID must be a nonempty path component")
            if ident in seen:
                raise ValueError(f"Line {line}: duplicate ID {ident}")
            if row["kind"] not in {"real", "sim"}:
                raise ValueError(f"Line {line}: kind must be real or sim")
            if not row["path"] or not row["obs"] or not row["obsm_vis"] or row["obsm_vis"] == "X_":
                raise ValueError(f"Line {line}: file_path, cluster_key and dimred_key are required")
            source = Path(row["path"])
            row["path"] = source if source.is_absolute() else path.parent / source
            if not row["path"].is_file():
                raise FileNotFoundError(f"Line {line}: input does not exist: {row['path']}")
            seen.add(ident)
            rows.append(row)
    if not rows:
        raise ValueError("Manifest has no datasets")
    return rows


def seed_all(seed, np, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def in_directory(path):
    before = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


def coordinate_reference(adata, key):
    import numpy as np
    coordinates = np.asarray(adata.obsm[key])
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"{key} must have two finite coordinates per cell")
    return dict(zip(adata.obs_names, coordinates.copy()))


def restore_coordinates(adata, key, reference):
    import numpy as np
    missing = [name for name in adata.obs_names if name not in reference]
    if missing:
        raise ValueError(f"Coordinate alignment failed: {len(missing)} unknown cell names")
    coordinates = np.stack([reference[name] for name in adata.obs_names])
    basis = key.removeprefix("X_")
    adata.obsm[key] = coordinates.copy()
    adata.obsm[f"X_{basis}"] = coordinates.copy()
    return basis


def canonical_csr(matrix):
    from scipy import sparse
    result = matrix.tocsr(copy=True) if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def nonzero_mask(matrix, axis):
    import numpy as np
    return np.asarray(canonical_csr(matrix).getnnz(axis=axis)).ravel() > 0


def remove_joint_duplicates(adata):
    import numpy as np
    from scipy import sparse
    both = sparse.hstack([canonical_csr(adata.layers[key]) for key in ("spliced", "unspliced")], format="csr")
    both.sum_duplicates()
    both.sort_indices()
    keep = np.ones(adata.n_obs, dtype=bool)
    seen = {}
    for row in range(adata.n_obs):
        start, end = both.indptr[row:row + 2]
        digest = hashlib.blake2b(digest_size=20)
        digest.update(np.asarray(both.indices[start:end], dtype=np.int64).tobytes())
        digest.update(np.asarray(both.data[start:end]).tobytes())
        prior = seen.setdefault(digest.digest(), [])
        for other in prior:
            p_start, p_end = both.indptr[other:other + 2]
            if (np.array_equal(both.indices[start:end], both.indices[p_start:p_end])
                    and np.array_equal(both.data[start:end], both.data[p_start:p_end])):
                keep[row] = False
                break
        if keep[row]:
            prior.append(row)
    removed = int((~keep).sum())
    if removed:
        adata._inplace_subset_obs(keep)
    if adata.n_obs == 0:
        raise ValueError("No cells remain after joint duplicate removal")
    return removed


def prepare(row, seed):
    import numpy as np
    import scanpy as sc
    import scvelo as scv
    import torch
    seed_all(seed, np, torch)
    adata = sc.read_h5ad(row["path"])
    for layer in ("spliced", "unspliced"):
        if layer not in adata.layers:
            raise ValueError(f"{row['ID']}: missing layer {layer}")
    if row["obs"] not in adata.obs or row["obsm_vis"] not in adata.obsm:
        raise ValueError(f"{row['ID']}: obs or obsm_vis key is absent")
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    reference = coordinate_reference(adata, row["obsm_vis"])
    cleanup = {}
    for axis, label in ((0, "genes"), (1, "cells")):
        keep = nonzero_mask(adata.layers["spliced"], axis) | nonzero_mask(adata.layers["unspliced"], axis)
        cleanup[f"joint_zero_{label}_before_filter"] = int((~keep).sum())
        if not keep.any():
            raise ValueError(f"No {label} remain after joint zero removal")
        if axis == 0:
            adata._inplace_subset_var(keep)
        else:
            adata._inplace_subset_obs(keep)
    cleanup["duplicate_cells_before_filter"] = remove_joint_duplicates(adata)
    adata.X = adata.layers["spliced"].copy()
    if row["kind"] == "real":
        scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    else:
        scv.pp.filter_and_normalize(adata, min_counts=None, min_counts_u=None,
                                    min_cells=None, min_cells_u=None,
                                    min_shared_counts=None, min_shared_cells=None)
    if adata.n_vars == 0:
        raise ValueError(f"{row['ID']}: no genes remain after scVelo filtering")
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars), subset=True)
    keep = nonzero_mask(adata.layers["spliced"], 1) | nonzero_mask(adata.layers["unspliced"], 1)
    cleanup["joint_zero_cells_after_hvg"] = int((~keep).sum())
    adata._inplace_subset_obs(keep)
    if adata.n_obs == 0:
        raise ValueError(f"{row['ID']}: no cells remain after joint-zero filtering")
    keep = nonzero_mask(adata.layers["spliced"], 1)
    cleanup["zero_spliced_cells_after_hvg"] = int((~keep).sum())
    adata._inplace_subset_obs(keep)
    if adata.n_obs == 0:
        raise ValueError(f"{row['ID']}: no cells remain after zero-spliced filtering")
    cleanup["duplicate_cells_after_hvg"] = remove_joint_duplicates(adata)
    if adata.n_obs < 30 or adata.n_vars < 30:
        raise ValueError("DeepVelo requires at least 30 cells and 30 genes after preprocessing")
    n_pcs = min(30, adata.n_obs - 1, adata.n_vars - 1)
    n_neighbors = min(30, adata.n_obs - 1)
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    scv.pp.moments(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    restore_coordinates(adata, row["obsm_vis"], reference)
    return adata, cleanup, reference


def sanitize_velocity(adata):
    import numpy as np
    from scipy import sparse
    for key in ("velocity", "cell_specific_beta", "cell_specific_gamma"):
        if key not in adata.layers or adata.layers[key].shape != adata.shape:
            raise ValueError(f"DeepVelo output layer {key} is absent or has wrong shape")
    velocity = adata.layers["velocity"]
    bad_cells = np.zeros(adata.n_obs, dtype=bool)
    bad_genes = np.zeros(adata.n_vars, dtype=bool)
    count = 0
    if sparse.issparse(velocity):
        coo = velocity.tocoo(copy=False)
        bad = ~np.isfinite(coo.data)
        count = int(bad.sum())
        bad_cells[coo.row[bad]] = True
        bad_genes[coo.col[bad]] = True
    else:
        values = np.asarray(velocity)
        chunk = max(1, 5_000_000 // adata.n_vars)
        for start in range(0, adata.n_obs, chunk):
            bad = ~np.isfinite(values[start:start + chunk])
            count += int(bad.sum())
            bad_cells[start:start + chunk] = bad.any(axis=1)
            bad_genes |= bad.any(axis=0)
    fraction = count / (adata.n_obs * adata.n_vars)
    if fraction > 0.20:
        raise ValueError(f"Non-finite DeepVelo velocity fraction {fraction:.2%} exceeds 20%")
    removed_cells, removed_genes = int(bad_cells.sum()), int(bad_genes.sum())
    if removed_cells == adata.n_obs or removed_genes == adata.n_vars:
        raise ValueError("Non-finite cleanup would remove every cell or gene")
    if count:
        adata._inplace_subset_obs(~bad_cells)
        adata._inplace_subset_var(~bad_genes)
    return {"nonfinite_count": count, "nonfinite_fraction": fraction,
            "removed_cells": removed_cells, "removed_genes": removed_genes,
            "status": "filtered" if count else "clean"}


def train_one(adata, row, run_index, seed, preprocess_seed, cleanup, output_dir, reference):
    import deepvelo as dv
    import numpy as np
    import scvelo as scv
    import torch
    from deepvelo.utils.preprocess import autoset_coeff_s
    seed_all(seed, np, torch)
    coeff = float(autoset_coeff_s(adata))
    if coeff not in {0.5, 0.75, 1.0}:
        raise ValueError(f"Unsupported data-derived loss coefficient: {coeff}")
    config = deepcopy(dv.Constants.default_configs)
    config["name"] = f"DeepVelo_{row['ID']}_r{run_index}"
    config["n_gpu"] = int(torch.cuda.is_available())
    config["loss"]["args"]["coeff_s"] = coeff
    config["data_loader"]["args"].update(shuffle=False, validation_split=0.0,
                                           num_workers=0, topC=30, topG=20)
    config["trainer"].update(epochs=100, save_dir=str(Path("saved") / config["name"]),
                             save_period=1000, tensorboard=False, verbosity=0)
    # The relative checkpoint directory is resolved from the requested output root.
    with in_directory(output_dir):
        dv.train(adata, config)
    velocity_status = sanitize_velocity(adata)
    basis = restore_coordinates(adata, row["obsm_vis"], reference)
    scv.tl.velocity_graph(adata, vkey="velocity")
    scv.tl.velocity_embedding(adata, vkey="velocity", basis=basis)
    scv.tl.velocity_pseudotime(adata, vkey="velocity")
    adata.uns["deepvelo_batchrun"] = {
        "dataset_id": row["ID"], "kind": row["kind"], "run_index": run_index,
        "preprocess_seed": preprocess_seed, "train_seed": seed,
        "obs_key": row["obs"], "coordinate_key": row["obsm_vis"],
        "projection_key": f"X_{basis}",
        "epochs": 100, "topC": 30, "topG": 20, "shuffle": False,
        "validation_split": 0.0, "loss_coeff_s": coeff,
        **cleanup, **velocity_status,
    }


def main():
    args = parse_args()
    if args.runs < 1 or args.base_seed < 0 or args.base_seed + args.runs > 2**32 - 1:
        raise ValueError("runs must be positive and training seeds must fit uint32")
    if not 0 <= args.preprocess_seed <= 2**32 - 1:
        raise ValueError("preprocess-seed must fit uint32")
    rows = load_manifest(args.input_csv.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for row in rows:
        destination = output_dir / row["kind"] / row["ID"]
        destination.mkdir(parents=True, exist_ok=True)
        pending = []
        for run_index in range(1, args.runs + 1):
            final = destination / f"DeepVelo_{row['ID']}_r{run_index}.h5ad"
            if final.exists():
                print(f"Skipping existing {final.name}")
            else:
                pending.append((run_index, final))
        if not pending:
            continue
        try:
            prepared, cleanup, reference = prepare(row, args.preprocess_seed)
        except Exception as error:
            failed += len(pending)
            print(f"Failed preprocessing {row['ID']}: {type(error).__name__}: {error}")
            continue
        for run_index, final in pending:
            temporary = final.with_suffix(".partial.h5ad")
            try:
                adata = prepared.copy()
                train_one(adata, row, run_index, args.base_seed + run_index,
                          args.preprocess_seed, cleanup, output_dir, reference)
                adata.write_h5ad(temporary, compression="lzf")
                os.replace(temporary, final)
                print(f"Saved {final.name}")
            except Exception as error:
                temporary.unlink(missing_ok=True)
                failed += 1
                print(f"Failed {row['ID']} r{run_index}: {type(error).__name__}: {error}")
    if failed:
        raise SystemExit(f"{failed} DeepVelo runs failed")

if __name__ == "__main__":
    main()
