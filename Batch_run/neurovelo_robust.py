#!/usr/bin/env python3
"""
Multi-run NeuroVelo analysis script for RNA velocity prediction.

Requirements:
- NeuroVelo must be installed or available in PYTHONPATH
- Input data files should be in the INPUT_DIR
- Results will be saved to OUTPUT_DIR

Configuration:
- Set INPUT_DIR and OUTPUT_DIR via environment variables or modify defaults below
"""

import datetime
import gc
import logging
import os
import sys
import random
import numpy as np
import anndata as ann
import scvelo as scv
import scanpy as sc
from neurovelo.train import Trainer
from neurovelo.utils import decode_gene_velocity

scv.settings.verbosity = 0

# Configuration: Modify these paths or set via environment variables
INPUT_DIR = os.getenv('INPUT_DIR', './example')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './example/output/NeuroVelo')

def set_random_seeds():
    """Set random seeds using current timestamp to ensure different results each run"""
    # Use microsecond timestamp and process ID to generate unique seed
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()

    random.seed(seed)
    np.random.seed(seed)

    return seed

def main_single(adata, save_dir, file_id, n_ode_hidden=100, n_vae_hidden=100, n_latent=50, batch_size=100, nepoch=100, simulate=False):
    if simulate:
        scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=adata.n_vars)
    else:
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    sc.pp.log1p(adata, layer='spliced')
    sc.pp.log1p(adata, layer='unspliced')
    scv.pp.neighbors(adata, n_neighbors=30, n_pcs=20)
    adata.obs['sample'] = 0

    model = Trainer(adata, layer='spliced', odesample_obs='sample', n_sample=1, percent=0.8, nepoch=nepoch,
                    n_ode_hidden=n_ode_hidden, n_vae_hidden=n_vae_hidden, n_latent=n_latent, batch_size=batch_size, reconstruct_xt=True)
    model.train()

    os.makedirs(save_dir, exist_ok=True)
    model.save_model(save_dir, file_id)
    model_path = os.path.join(save_dir, f"{file_id}.pth")
    adata.layers['spliced_velocity'] = decode_gene_velocity(adata, model_path, layer='spliced')

    scv.pp.neighbors(adata)
    scv.tl.umap(adata)
    scv.tl.velocity_graph(adata, vkey='spliced_velocity', xkey='spliced')
    scv.tl.velocity_embedding(adata, vkey='spliced_velocity', basis='umap')

    if 'X_dimred' in adata.obsm:
        adata.obsm['X_umap'] = adata.obsm['X_dimred']

    output_h5ad_path = os.path.join(save_dir, f"{file_id}.h5ad")
    adata.write(output_h5ad_path)
    print(f"Analysis completed! Results saved to: {output_h5ad_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"error_log_{timestamp}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_files = [
        {
            "path": os.path.join(INPUT_DIR, "Simulation-data/bifurcating_cell1000_gene500_dataset.h5ad"),
            "simulate": True,
            "id_pre_base": "NeuroVelo_bifurcating_cell1000_gene500"
        },
        {
            "path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "simulate": False,
            "id_pre_base": "NeuroVelo_7"
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

            adata = None

            try:
                input_file = os.path.basename(file_path)
                id_pre = f"{id_pre_base}_r{run_idx}"

                print(f"  Processing: {input_file}")

                adata = ann.read_h5ad(file_path)

                save_dir = os.path.join(OUTPUT_DIR, id_pre)
                file_id = id_pre

                main_single(adata, save_dir, file_id, simulate=simulate)

                print(f"  Saved: {id_pre}.h5ad")
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