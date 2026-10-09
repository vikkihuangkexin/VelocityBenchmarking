#!/usr/bin/env python3
"""
Multi-run SvelvetVAE (velvetVAE) robustness driver for RNA velocity prediction.

Requirements:
- velvetVAE must be installed or available in PYTHONPATH
- Input data files should be placed under INPUT_DIR
- Results are written to OUTPUT_DIR

Configuration:
- Set INPUT_DIR and OUTPUT_DIR via environment variables or edit the defaults.
"""

import datetime
import gc
import logging
import os
import random
import sys

import numpy as np
import scanpy as sc
import scvelo as scv
import torch
import velvetvae as vt
from scipy.sparse import issparse
import matplotlib

matplotlib.use("AGG")

# Configuration: edit these paths or set them through environment variables.
INPUT_DIR = os.getenv("INPUT_DIR", "./example")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./example/output/SvelvetVAE")

# Default to GPU 0, but respect an externally pinned CUDA_VISIBLE_DEVICES.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def set_random_seeds():
    """Set random seeds using the current timestamp to vary results per run."""
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def to_dense_float32(layer):
    """Densify a (possibly sparse) layer and cast it to float32."""
    if issparse(layer):
        layer = layer.toarray()
    return np.asarray(layer, dtype=np.float32)


def preprocess(file_path, simulate):
    """Load a dataset and run the scVelo standard preprocessing."""
    print(
        "---------------------------------- preprocess "
        f"{os.path.basename(file_path)} ---------------------------------------------"
    )
    adata = sc.read(file_path)

    if simulate:
        adata.obs_names_make_unique()
        adata.obs["milestone"] = "milestone"
        if "X_dimred" in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_dimred"].copy()

    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    missing_layers = [layer for layer in ("spliced", "unspliced") if layer not in adata.layers]
    if missing_layers:
        raise ValueError(f"Missing required layers: {missing_layers}")

    if simulate:
        n_top_genes = 2000
        if adata.n_vars < n_top_genes:
            candidate = min((adata.n_vars // 500) * 500, adata.n_vars - 1)
            n_top_genes = candidate if candidate > 0 else max(1, adata.n_vars)
        scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=n_top_genes)
    else:
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)

    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    return adata


def main(adata, result_path, n_latent=50, max_epochs=100, freeze_vae_after_epochs=20,
         constrain_vf_after_epochs=20, lr=0.01, knn_neighbors=100):
    """Train SvelvetVAE and store the velocity in adata.layers['velocity']."""
    print("--------------------------------")

    adata.layers["spliced"] = to_dense_float32(adata.layers["spliced"])
    adata.layers["unspliced"] = to_dense_float32(adata.layers["unspliced"])
    if "total" not in adata.layers:
        adata.layers["total"] = adata.layers["spliced"] + adata.layers["unspliced"]
    adata.layers["total"] = to_dense_float32(adata.layers["total"])

    vt.pp.neighborhood(adata, n_neighbors=knn_neighbors)
    vt.ut.set_seed(0)
    vt.md.Svelvet.setup_anndata(
        adata,
        x_layer="total",
        u_layer="unspliced",
        knn_layer="knn_index",
    )

    model = vt.md.Svelvet(
        adata,
        n_latent=n_latent,
        linear_decoder=True,
        neighborhood_space="latent_space",
        gamma_mode="learned",
    )
    model.setup_model(gamma_kwargs={"gamma_min": 0.1, "gamma_max": 1})

    # Fix dtype bugs in velvetVAE: loggamma / ss_gamma are created as float64 in
    # setup_model, and NeighborhoodConstraint.X is created from a float64 numpy
    # array. Both break the forward pass once the constraint becomes active.
    try:
        model.module.loggamma = torch.nn.Parameter(model.module.loggamma.data.float())
        model.module.ss_gamma = model.module.ss_gamma.float()
        model.module.nc.X = model.module.nc.X.float()
        model.module.nc.b = model.module.nc.b.float()
    except Exception as exc:  # pragma: no cover - depends on velvetVAE version
        print(f"Warning: could not cast velvetVAE parameter dtypes: {exc}")

    model.train(
        batch_size=adata.shape[0],
        max_epochs=max_epochs,
        freeze_vae_after_epochs=freeze_vae_after_epochs,
        constrain_vf_after_epochs=constrain_vf_after_epochs,
        lr=lr,
    )

    velocity = model.predict_velocity()
    if issparse(velocity):
        velocity = velocity.toarray()
    adata.layers["velocity"] = np.nan_to_num(velocity, nan=0.0, neginf=0.0, posinf=0.0)
    return True


def main_loop():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"error_log_{timestamp}.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    data_files = [
        {
            "path": os.path.join(INPUT_DIR, "Simulation-data/bifurcating_cell1000_gene500_dataset.h5ad"),
            "simulate": True,
            "id_pre_base": "SvelvetVAE_bifurcating_cell1000_gene500",
        },
        {
            "path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "simulate": False,
            "id_pre_base": "SvelvetVAE_7",
        },
    ]

    n_runs = 5

    for run_idx in range(1, n_runs + 1):
        seed = set_random_seeds()
        print(f"\n[Run {run_idx}/{n_runs}] Seed: {seed}")

        for file_info in data_files:
            file_path = file_info["path"]
            simulate = file_info["simulate"]
            id_pre_base = file_info["id_pre_base"]

            adata = None

            try:
                input_file = os.path.basename(file_path)
                id_pre = f"{id_pre_base}_r{run_idx}"

                print(f"  Processing: {input_file}")

                adata = preprocess(file_path, simulate)

                result_path = os.path.join(OUTPUT_DIR, id_pre)
                os.makedirs(result_path, exist_ok=True)

                adata.write(os.path.join(result_path, "pp.h5ad"))

                success = main(adata, result_path)
                if success:
                    adata.write(os.path.join(result_path, "rc.h5ad"))
                    print(f"  Saved: {id_pre}")
                    logging.info(f"Success: {input_file} run {run_idx}")
                else:
                    print(f"  Failed: {id_pre}")
                    logging.error(f"Failed: {input_file} run {run_idx}")

                gc.collect()

            except Exception as exc:
                logging.error(
                    f"Error processing {file_path} run {run_idx}: {str(exc)}",
                    exc_info=True,
                )


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as exc:
        logging.error(f"Unhandled error: {exc}", exc_info=True)
        sys.exit(1)
