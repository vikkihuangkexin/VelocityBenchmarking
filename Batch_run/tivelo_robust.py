#!/usr/bin/env python3
"""
Multi-run TIVelo analysis script for RNA velocity prediction.

Requirements:
- TIVelo must be installed or available in PYTHONPATH
- Input data files should be located under INPUT_DIR
- Results are written to OUTPUT_DIR

Configuration:
- Set INPUT_DIR and OUTPUT_DIR via environment variables or adjust the defaults below
"""

import datetime
import gc
import logging
import os
import random

import numpy as np
import scanpy as sc
import scvelo as scv
from scipy.sparse.linalg import ArpackNoConvergence
import matplotlib

matplotlib.use("AGG")

from tivelo.main import tivelo

# Configuration: modify these paths or set them via environment variables
INPUT_DIR = os.getenv("INPUT_DIR", "./example")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./example/output/TIVelo")

# Do not override a device selection made by the caller.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("NCCL_DEBUG", "INFO")

# Error messages that indicate an unconnected cluster transition graph.
DISCONNECTED_MESSAGES = ("unconnected sub-graphs",)
SIM_DISCONNECTED_MESSAGES = ("unconnected sub-graphs", "max() arg is an empty sequence")


def set_random_seeds():
    """Set random seeds using the current timestamp so every run differs."""
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()

    random.seed(seed)
    np.random.seed(seed)

    return seed


def preprocess(file_path, simulate, n_top_genes=2000, min_shared_counts=20, n_pcs=30, n_neighbors=30):
    """Load a dataset and run the standard TIVelo preprocessing pipeline."""
    print(f"---------------------------------- preprocess {os.path.basename(file_path)} --------------------------------------------")
    adata = sc.read(file_path)
    adata.obs_names_make_unique()

    if simulate:
        if "X_dimred" not in adata.obsm:
            raise ValueError("Simulated data requires obsm['X_dimred'].")
        adata.obsm["X_umap"] = adata.obsm["X_dimred"]

        top_gene = n_top_genes
        if adata.n_vars < 2000:
            top_gene = (adata.n_vars // 500) * 500
            top_gene = min(top_gene, adata.n_vars - 1)
        effective_shared_counts = None
    else:
        top_gene = n_top_genes
        effective_shared_counts = min_shared_counts

    scv.pp.filter_and_normalize(adata, min_shared_counts=effective_shared_counts, n_top_genes=top_gene)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    print(adata)
    return adata


def run_tivelo(adata, cluster_key, save_dir, simulate, n_epochs=100, batch_size=1024, loss_fun="mse"):
    """Run TIVelo, returning a 'disconnected' marker for unprocessable graphs."""
    try:
        adata = tivelo(
            adata,
            group_key=cluster_key,
            emb_key="X_umap",
            data_name="native",
            save_folder=save_dir,
            show_fig=False,
            filter_genes=True,
            constrain=True,
            loss_fun=loss_fun,
            batch_size=batch_size,
            n_epochs=n_epochs,
            velocity_key="velocity",
            show_DTI=False,
            adjust_DTI=False,
            measure_performance=False,
        )
    except ValueError as exc:
        messages = SIM_DISCONNECTED_MESSAGES if simulate else DISCONNECTED_MESSAGES
        if any(message in str(exc) for message in messages):
            print("  [SKIP] unconnected sub-graphs, TIVelo cannot process this dataset.")
            return "disconnected"
        raise
    except ArpackNoConvergence:
        print("  [SKIP] ARPACK did not converge (disconnected transition matrix).")
        return "disconnected"

    return adata


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"error_log_{timestamp}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_files = [
        {
            "path": os.path.join(INPUT_DIR, "Simulation-data/bifurcating_cell1000_gene500_dataset.h5ad"),
            "simulate": True,
            "cluster_key": "milestone",
            "id_pre_base": "TIVelo_bifurcating_cell1000_gene500",
        },
        {
            "path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "simulate": False,
            "cluster_key": "celltype",
            "id_pre_base": "TIVelo_7",
        },
    ]

    n_runs = 5

    for run_idx in range(1, n_runs + 1):
        # Set a different random seed for each run
        seed = set_random_seeds()
        print(f"\n[Run {run_idx}/{n_runs}] Seed: {seed}")

        for file_info in data_files:
            file_path = file_info["path"]
            simulate = file_info["simulate"]
            cluster_key = file_info["cluster_key"]
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

                adata = run_tivelo(adata, cluster_key, result_path, simulate)

                if isinstance(adata, str):
                    print(f"  Skipped: {id_pre} ({adata})")
                    logging.warning(f"Skipped: {input_file} run {run_idx} ({adata})")
                else:
                    adata.write(os.path.join(result_path, "rc.h5ad"))
                    print(f"  Saved: {id_pre}")
                    logging.info(f"Success: {input_file} run {run_idx}")

                gc.collect()

            except Exception as e:
                logging.error(f"Error processing {file_path} run {run_idx}: {str(e)}", exc_info=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logging.error(f"Unhandled error: {e}", exc_info=True)
