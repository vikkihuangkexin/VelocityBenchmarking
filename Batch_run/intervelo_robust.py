#!/usr/bin/env python3
"""Independent InterVelo repetitions from a CSV manifest.

Install the team fork: https://github.com/sd68515/InterVelo_py311
CSV columns: dataset_name,file_path,modality,cluster_key,dimred_key.
Optional extra_layers names are comma-separated. preprocess_mode is also accepted.
Paths may be relative to the CSV.
Preprocessing modes: real, sim. Coordinates control projection, not model inputs.
"""
from __future__ import annotations
import argparse
import csv
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

PREPROCESS_MODES = {"real", "sim"}


def input_spec(mode, extra_value):
    layers = [name.strip() for name in extra_value.split(",")] if extra_value else []
    if any(not name for name in layers) or len(layers) != len(set(layers)):
        raise ValueError("extra_layers must contain unique, nonempty layer names")
    if any(name in {"Ms", "Mu"} for name in layers):
        raise ValueError("Ms and Mu are already included in the model input")
    if mode == "atac":  # Legacy shorthand for real data with Mc.
        mode = "real"
        if "Mc" not in layers:
            layers.insert(0, "Mc")
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Unsupported preprocessing mode: {mode}")
    return mode, layers

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260728)
    parser.add_argument("--preprocess-seed", type=int, default=2024)
    return parser.parse_args()

def load_manifest(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "preprocess_mode" in (reader.fieldnames or []) and "modality" in reader.fieldnames:
            raise ValueError("Use either 'modality' or 'preprocess_mode', not both")
        mode_column = "modality" if "modality" in (reader.fieldnames or []) else "preprocess_mode"
        required = {"dataset_name", "file_path", mode_column, "cluster_key", "dimred_key"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        rows, seen = [], set()
        for line, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError(f"CSV line {line} has more values than its header")
            raw_mode = (raw.get(mode_column) or "").strip()
            try:
                mode, extra_layers = input_spec(raw_mode, (raw.get("extra_layers") or "").strip())
            except ValueError as error:
                raise ValueError(f"Line {line}: {error}") from error
            row = {
                "ID": (raw.get("dataset_name") or "").strip(),
                "path": (raw.get("file_path") or "").strip(),
                "preprocess_mode": mode,
                "extra_layers": extra_layers,
                "output_group": raw_mode if mode_column == "modality" else mode,
                "obs": (raw.get("cluster_key") or "").strip(),
                "obsm_vis": (raw.get("dimred_key") or "").strip(),
            }
            ident = row["ID"]
            if not ident or ident in {".", ".."} or Path(ident).name != ident or "\\" in ident:
                raise ValueError(f"Line {line}: ID must be a nonempty path component")
            if ident in seen:
                raise ValueError(f"Line {line}: duplicate ID {ident}")
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

def valid_velocity_layer(adata, key):
    import numpy as np
    from scipy import sparse
    if key not in adata.layers or adata.layers[key].shape != adata.shape:
        return False
    matrix = adata.layers[key]
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    return bool(np.isfinite(values).all() and np.any(values != 0))

def prepare(row, seed):
    import numpy as np
    import scanpy as sc
    import scvelo as scv
    import torch
    from InterVelo.data import preprocess_data
    seed_all(seed, np, torch)
    adata = sc.read_h5ad(row["path"])
    layers = ["spliced", "unspliced", *row["extra_layers"]]
    missing = [key for key in layers if key not in adata.layers]
    if missing:
        raise ValueError(f"{row['ID']}: missing layers {missing}")
    if row["obs"] not in adata.obs or row["obsm_vis"] not in adata.obsm:
        raise ValueError(f"{row['ID']}: obs or obsm_vis key is absent")
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    reference = coordinate_reference(adata, row["obsm_vis"])
    real = row["preprocess_mode"] == "real"
    if real:
        adata.X = adata.layers["spliced"].copy()
    # Optional exact duplicate removal by X densifies sparse input. Avoid
    # enabling it on large datasets without a memory budget:
    # adata = adata[~adata.to_df().duplicated().to_numpy()].copy()
    if row["preprocess_mode"] == "sim":
        scv.pp.filter_and_normalize(adata)
    else:
        scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars), subset=True)
    if row["preprocess_mode"] == "sim":
        keep = np.asarray(adata.layers["spliced"].sum(axis=1)).ravel() != 0
        adata = adata[keep].copy()
    if adata.n_obs < 3 or adata.n_vars < 3:
        raise ValueError(f"{row['ID']}: too few cells or genes after filtering")
    n_pcs = min(30, adata.n_obs - 1, adata.n_vars - 1)
    n_neighbors = min(30, adata.n_obs - 1)
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    model_layers = ["Ms", "Mu", *row["extra_layers"]]
    adata = preprocess_data(adata, layers=model_layers, filter_on_r2=False)
    if row["obs"] not in adata.obs:
        raise ValueError(f"{row['ID']}: preprocessing removed required metadata")
    restore_coordinates(adata, row["obsm_vis"], reference)
    return adata, model_layers, reference

def train_one(adata, layers, row, run_index, train_seed, preprocess_seed, output_dir, reference):
    import numpy as np
    import scvelo as scv
    import torch
    from scipy import sparse
    from InterVelo._utils import autoset_coeff_s, update_dict
    from InterVelo.train import Constants, train
    seed_all(train_seed, np, torch)
    tensors = []
    for key in layers:
        layer = adata.layers[key]
        dense = layer.toarray() if sparse.issparse(layer) else np.asarray(layer)
        tensors.append(torch.from_numpy(np.asarray(dense, dtype=np.float32)))
    input_data = torch.cat(tensors, dim=1)
    method_id = f"InterVelo_{row['ID']}_r{run_index}"
    config = update_dict(deepcopy(Constants.default_configs), {
        "name": method_id, "n_gpu": int(torch.cuda.is_available()),
        "loss_pearson": {"coeff_s": autoset_coeff_s(adata)},
        "arch": {"args": {"n_latent": 20, "pred_unspliced": False}},
        "data_loader": {"args": {"batch_size": min(1024, adata.n_obs),
                                  "num_workers": 0,
                                  "validation_split": 0.1 if adata.n_obs >= 10 else 0.0}},
        "trainer": {"tensorboard": False, "verbosity": 0,
                    "save_dir": str(Path("saved") / method_id)},
    })
    with in_directory(output_dir):
        train(adata, input_data, config)
    basis = restore_coordinates(adata, row["obsm_vis"], reference)
    scv.tl.velocity_graph(adata, vkey="velocity", xkey="Ms")
    scv.tl.velocity_embedding(adata, vkey="velocity", basis=basis)
    truth_status = "not_applicable"
    if row["preprocess_mode"] == "sim":
        truth_status = "skipped_invalid_or_missing"
        if valid_velocity_layer(adata, "ground_truth_velocity"):
            try:
                if "ground_truth_velocity_graph" not in adata.uns:
                    scv.tl.velocity_graph(adata, vkey="ground_truth_velocity", xkey="Ms")
                if f"ground_truth_velocity_{basis}" not in adata.obsm:
                    scv.tl.velocity_embedding(adata, vkey="ground_truth_velocity", basis=basis)
                truth_status = "complete"
            except Exception as error:
                truth_status = f"failed_{type(error).__name__}"
                print(f"Ground-truth projection failed for {row['ID']} r{run_index}: {error}")
    if "pseudotime" in adata.obs:
        values = adata.obs["pseudotime"].to_numpy(dtype=float)
        span = np.ptp(values)
        adata.obs["intervelo_pseudotime_normalized"] = (values - values.min()) / span if span else 0.0
    adata.uns["intervelo_batchrun"] = {
        "dataset_id": row["ID"], "kind": row["preprocess_mode"], "run_index": run_index,
        "preprocess_mode": row["preprocess_mode"], "extra_layers": row["extra_layers"],
        "preprocess_seed": preprocess_seed, "train_seed": train_seed,
        "obs_key": row["obs"], "coordinate_key": row["obsm_vis"],
        "model_layers": ",".join(layers),
        "projection_key": f"X_{basis}",
        "ground_truth_projection": truth_status,
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
        destination = output_dir / row["output_group"] / row["ID"]
        destination.mkdir(parents=True, exist_ok=True)
        pending = []
        for run_index in range(1, args.runs + 1):
            final = destination / f"InterVelo_{row['ID']}_r{run_index}.h5ad"
            if final.exists():
                print(f"Skipping existing {final.name}")
            else:
                pending.append((run_index, final))
        if not pending:
            continue
        try:
            prepared, layers, reference = prepare(row, args.preprocess_seed)
        except Exception as error:
            failed += len(pending)
            print(f"Failed preprocessing {row['ID']}: {type(error).__name__}: {error}")
            continue
        for run_index, final in pending:
            temporary = final.with_suffix(".partial.h5ad")
            try:
                adata = prepared.copy()
                train_one(adata, layers, row, run_index, args.base_seed + run_index,
                          args.preprocess_seed, output_dir, reference)
                adata.write_h5ad(temporary, compression="lzf")
                os.replace(temporary, final)
                print(f"Saved {final.name}")
            except Exception as error:
                temporary.unlink(missing_ok=True)
                failed += 1
                print(f"Failed {row['ID']} r{run_index}: {type(error).__name__}: {error}")
    if failed:
        raise SystemExit(f"{failed} InterVelo runs failed")

if __name__ == "__main__":
    main()
