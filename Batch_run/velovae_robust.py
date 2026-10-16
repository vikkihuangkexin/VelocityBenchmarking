#!/usr/bin/env python3
"""Independent VeloVAE repetitions from a CSV manifest.

CSV columns: dataset_name,file_path,modality,cluster_key,dimred_key.
Modality is real or sim. Input paths
may be relative to the CSV. Each repetition saves a separate h5ad file.
The seed interface is new; historical VeloVAE batch runs did not set seeds.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
import random
from copy import deepcopy
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--preprocess-seed", type=int, default=2024)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch CUDA device")
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


def remove_invalid_simulation_records(adata):
    import numpy as np
    from scipy import sparse
    spliced = canonical_csr(adata.layers["spliced"])
    unspliced = canonical_csr(adata.layers["unspliced"])
    counts = {}
    keep = np.asarray(spliced.getnnz(axis=0) + unspliced.getnnz(axis=0)).ravel() > 0
    counts["joint_zero_genes"] = int((~keep).sum())
    if not keep.any():
        raise ValueError("No genes remain after joint spliced/unspliced cleanup")
    adata._inplace_subset_var(keep)
    spliced, unspliced = spliced[:, keep], unspliced[:, keep]
    keep = np.asarray(spliced.getnnz(axis=1) + unspliced.getnnz(axis=1)).ravel() > 0
    counts["joint_zero_cells"] = int((~keep).sum())
    if not keep.any():
        raise ValueError("No cells remain after joint spliced/unspliced cleanup")
    adata._inplace_subset_obs(keep)
    combined = sparse.hstack((spliced[keep], unspliced[keep]), format="csr")
    combined.sum_duplicates()
    combined.sort_indices()
    keep = np.ones(adata.n_obs, dtype=bool)
    seen = {}
    for row in range(adata.n_obs):
        start, end = combined.indptr[row:row + 2]
        digest = hashlib.blake2b(digest_size=20)
        digest.update(np.asarray(combined.indices[start:end], dtype=np.int64).tobytes())
        digest.update(np.asarray(combined.data[start:end]).tobytes())
        prior = seen.setdefault(digest.digest(), [])
        for other in prior:
            p_start, p_end = combined.indptr[other:other + 2]
            if (np.array_equal(combined.indices[start:end], combined.indices[p_start:p_end])
                    and np.array_equal(combined.data[start:end], combined.data[p_start:p_end])):
                keep[row] = False
                break
        if keep[row]:
            prior.append(row)
    counts["joint_duplicate_cells"] = int((~keep).sum())
    adata._inplace_subset_obs(keep)
    adata.layers["spliced"] = canonical_csr(adata.layers["spliced"])
    adata.layers["unspliced"] = canonical_csr(adata.layers["unspliced"])
    return counts


def remove_degenerate_genes(adata):
    import numpy as np
    from scipy import sparse
    mu, ms = adata.layers["Mu"], adata.layers["Ms"]
    keep = np.ones(adata.n_vars, dtype=bool)
    for index in range(adata.n_vars):
        u = mu[:, index]
        s = ms[:, index]
        u = u.toarray().ravel() if sparse.issparse(u) else np.asarray(u).ravel()
        s = s.toarray().ravel() if sparse.issparse(s) else np.asarray(s).ravel()
        if not np.isfinite(u).all() or not np.isfinite(s).all():
            keep[index] = False
            continue
        positive = (u > 0) & (s > 0)
        if positive.sum() > 3 and (np.ptp(u[positive]) == 0 or np.ptp(s[positive]) == 0):
            keep[index] = False
    if not keep.any():
        raise ValueError("No genes remain for VeloVAE ODE initialization")
    removed = int((~keep).sum())
    adata._inplace_subset_var(keep)
    return removed


def training_parameters(adata, kind):
    if kind == "real":
        n_gene = 500 if adata.n_vars < 1500 else 2000
    elif adata.n_vars < 1500:
        n_gene = 500
    elif adata.n_vars < 10000:
        n_gene = 2000
    elif adata.n_vars < 50000:
        n_gene = 4000
    else:
        n_gene = 5000
    npc = 50 if adata.n_obs >= 15000 else 30
    batch_size = 64 if adata.n_obs < 1500 else 128 if adata.n_obs < 15000 else 256
    return n_gene, npc, {"batch_size": batch_size}


def prepare(row, seed):
    import numpy as np
    import scanpy as sc
    import torch
    import velovae as vv
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
    cleanup = {"joint_zero_genes": 0, "joint_zero_cells": 0, "joint_duplicate_cells": 0}
    if row["kind"] == "real":
        adata.X = adata.layers["spliced"].copy()
        adata.obsm.pop("X_pca", None)
        adata.uns.pop("neighbors", None)
        for key in ("distances", "connectivities"):
            adata.obsp.pop(key, None)
    else:
        cleanup = remove_invalid_simulation_records(adata)
    n_gene, npc, config = training_parameters(adata, row["kind"])
    cleanup["n_gene_requested"] = n_gene
    cleanup["n_pcs"] = npc
    kwargs = {"n_gene": n_gene, "npc": npc}
    if row["kind"] == "real":
        kwargs["compute_umap"] = False
    else:
        kwargs.update(min_shared_counts=0, min_shared_cells=0, enforce=True)
    vv.preprocess(adata, **kwargs)
    cleanup["degenerate_genes"] = remove_degenerate_genes(adata)
    restore_coordinates(adata, row["obsm_vis"], reference)
    return adata, config, cleanup, reference


def resolve_device(requested, torch):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda") and torch.cuda.is_available():
        torch.device(requested)
        return requested
    raise ValueError(f"Requested device is unavailable: {requested}")


def train_one(adata, row, config, cleanup, run_index, seed, preprocess_seed, device, reference):
    import numpy as np
    import scvelo as scv
    import torch
    import velovae as vv
    from scipy import sparse
    seed_all(seed, np, torch)
    config = deepcopy(config)
    basis = row["obsm_vis"].removeprefix("X_")
    vae = vv.VAE(adata, tmax=20, dim_z=5 if row["kind"] == "real" else 4,
                 device=resolve_device(device, torch), reverse_gene_mode=False,
                 full_vb=False, discrete=False, drop_last=True, config=config)
    vae.train(adata, embed=basis, drop_last=True, config=config)
    vae.save_anndata(adata, key="vae")
    restore_coordinates(adata, row["obsm_vis"], reference)
    if "vae_time" not in adata.obs or "vae_velocity" not in adata.layers:
        raise ValueError("VeloVAE did not save time or velocity")
    values = adata.obs["vae_time"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("VeloVAE model time contains non-finite values")
    span = np.ptp(values)
    adata.obs["vae_time_normalized"] = (values - values.min()) / span if span else 0.0
    first = adata.layers["vae_velocity"][0]
    first = first.toarray().ravel() if sparse.issparse(first) else np.asarray(first).ravel()
    gene_subset = adata.var_names[~np.isnan(first)]
    scv.tl.velocity_graph(adata, vkey="vae_velocity", gene_subset=gene_subset)
    scv.tl.velocity_embedding(adata, vkey="vae_velocity", basis=basis)
    scv.tl.velocity_pseudotime(adata, vkey="vae_velocity")
    adata.uns["velovae_batchrun"] = {
        "dataset_id": row["ID"], "kind": row["kind"], "run_index": run_index,
        "preprocess_seed": preprocess_seed, "train_seed": seed,
        "obs_key": row["obs"], "coordinate_key": row["obsm_vis"],
        "projection_key": f"X_{basis}",
        "batch_size": config["batch_size"], **cleanup,
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
            final = destination / f"VeloVAE_{row['ID']}_r{run_index}.h5ad"
            if final.exists():
                print(f"Skipping existing {final.name}")
            else:
                pending.append((run_index, final))
        if not pending:
            continue
        try:
            prepared, config, cleanup, reference = prepare(row, args.preprocess_seed)
        except Exception as error:
            failed += len(pending)
            print(f"Failed preprocessing {row['ID']}: {type(error).__name__}: {error}")
            continue
        for run_index, final in pending:
            temporary = final.with_suffix(".partial.h5ad")
            try:
                adata = prepared.copy()
                train_one(adata, row, config, cleanup, run_index, args.base_seed + run_index,
                          args.preprocess_seed, args.device, reference)
                adata.write_h5ad(temporary, compression="lzf")
                os.replace(temporary, final)
                print(f"Saved {final.name}")
            except Exception as error:
                temporary.unlink(missing_ok=True)
                failed += 1
                print(f"Failed {row['ID']} r{run_index}: {type(error).__name__}: {error}")
    if failed:
        raise SystemExit(f"{failed} VeloVAE runs failed")

if __name__ == "__main__":
    main()
