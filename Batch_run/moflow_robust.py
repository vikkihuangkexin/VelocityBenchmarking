#!/usr/bin/env python3
"""
Multi-run MoFlow analysis script for multi-omics RNA velocity prediction.

Requirements:
- MoFlow must be installed or available in PYTHONPATH
- Input data files should be in the INPUT_DIR
- Results will be saved to OUTPUT_DIR

Configuration:
- Set INPUT_DIR and OUTPUT_DIR via environment variables or modify defaults below
"""

import datetime
import gc
import logging
import os
import random

import numpy as np
import scanpy as sc
import scvelo as scv
import matplotlib

matplotlib.use("Agg")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Configuration: Modify these paths or set via environment variables
INPUT_DIR = os.getenv("INPUT_DIR", "./example")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./example/output/MoFlow")

DEFAULT_AUXILIARY_LAYER = "Mc"


def set_random_seeds():
    """Set random seeds using current timestamp to ensure different results each run"""
    # Use microsecond timestamp and process ID to generate a unique seed
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    return seed


def preprocess(file_path, cluster_key="celltype", extra_layers=DEFAULT_AUXILIARY_LAYER, simulate=False):
    print(f"----------------------------------preprocess {os.path.basename(file_path)} ---------------------------------------------")
    adata = sc.read(file_path)

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    if simulate and "X_dimred" in adata.obsm:
        adata.obsm["X_umap"] = np.asarray(adata.obsm["X_dimred"])
    elif "X_umap" not in adata.obsm and "X_tsne" in adata.obsm:
        adata.obsm["X_umap"] = np.asarray(adata.obsm["X_tsne"])

    auxiliary_layer = extra_layers
    if isinstance(auxiliary_layer, (list, tuple)):
        auxiliary_layer = auxiliary_layer[0]
    if auxiliary_layer not in adata.layers:
        raise ValueError(f"Auxiliary layer '{auxiliary_layer}' not found in adata.layers")

    if simulate:
        if cluster_key and cluster_key in adata.obs:
            adata.obs[cluster_key] = "milestone"
        adata.obs["celltype"] = "milestone"
    else:
        if cluster_key not in adata.obs:
            raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")
        adata.obs["celltype"] = adata.obs[cluster_key].astype(str)

    adata.obs["celltype_new"] = adata.obs["celltype"].astype(str)

    missing_layers = [layer for layer in ("spliced", "unspliced") if layer not in adata.layers]
    if missing_layers:
        raise ValueError(f"Missing required layers: {missing_layers}")

    n_top_genes = int(min(2000, adata.n_vars))
    if simulate:
        scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=n_top_genes)
    else:
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=n_top_genes)

    print(adata)
    return adata, auxiliary_layer


def main(adata, result_path, auxiliary_layer=DEFAULT_AUXILIARY_LAYER, n_jobs=10, dataset_name="MoFlow"):
    import MoFlow as mf

    adata_rna = adata.copy()
    adata_atac = adata.copy()
    adata_atac.X = adata_atac.layers[auxiliary_layer].copy()

    model = mf.MOFlow(adata_rna, adata_atac, folder_name=f"MoFlow_velocity_{dataset_name}")

    rc_path = os.path.join(result_path, "rc.h5ad")
    model.velocity(adata_rna, n_jobs=n_jobs, save_path=result_path, file_name="rc.h5ad")

    if os.path.exists(rc_path):
        result = sc.read(rc_path)
    else:
        result = adata_rna

    if "velocity" not in result.layers:
        return False

    result.write(rc_path)
    return True


def main_batch():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"error_log_{timestamp}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_files = [
        {
            "path": os.path.join(INPUT_DIR, "Simulation-data/bifurcating_cell1000_gene500_dataset.h5ad"),
            "simulate": True,
            "id_pre_base": "MoFlow_bifurcating_cell1000_gene500",
            "cluster_key": "celltype"
        },
        {
            "path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "simulate": False,
            "id_pre_base": "MoFlow_7",
            "cluster_key": "celltype"
        }
    ]

    n_runs = 5

    for run_idx in range(1, n_runs + 1):
        # Set different random seed for each run
        seed = set_random_seeds()
        print(f"\n[Run {run_idx}/{n_runs}] Seed: {seed}")

        for file_info in data_files:
            file_path = file_info["path"]
            simulate = file_info["simulate"]
            id_pre_base = file_info["id_pre_base"]
            cluster_key = file_info["cluster_key"]

            adata = None

            try:
                input_file = os.path.basename(file_path)
                id_pre = f"{id_pre_base}_r{run_idx}"

                print(f"  Processing: {input_file}")

                adata, auxiliary_layer = preprocess(
                    file_path,
                    cluster_key=cluster_key,
                    simulate=simulate,
                )

                result_path = os.path.join(OUTPUT_DIR, id_pre)
                os.makedirs(result_path, exist_ok=True)

                adata.write(os.path.join(result_path, "pp.h5ad"))

                success = main(adata, result_path, auxiliary_layer=auxiliary_layer, dataset_name=id_pre)
                if success:
                    print(f"  Saved: {id_pre}")
                    logging.info(f"Success: {input_file} run {run_idx}")
                else:
                    print(f"  Failed: {id_pre}")
                    logging.error(f"Failed: {input_file} run {run_idx}")

                gc.collect()

            except Exception as e:
                logging.error(f"Error processing {file_path} run {run_idx}: {str(e)}", exc_info=True)


if __name__ == "__main__":
    try:
        main_batch()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logging.error(f"Unhandled error: {e}", exc_info=True)
