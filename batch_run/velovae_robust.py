#!/usr/bin/env python3
"""
Multi-run VeloVAE analysis script for RNA velocity prediction.

Requirements:
- VeloVAE must be installed or available in PYTHONPATH
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
import torch
import scanpy as sc
import scvelo as scv
from scipy.sparse import csr_matrix

# VeloVAE should be installed via pip or available in PYTHONPATH
import velovae as vv

# Configuration: Modify these paths or set via environment variables
INPUT_DIR = os.getenv('INPUT_DIR', './data')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output/VeloVAE')
DEVICE = os.getenv('DEVICE', 'cuda')  # GPU device for training

def set_random_seeds():
    """Set random seeds using current timestamp to ensure different results each run"""
    # Use microsecond timestamp and process ID to generate unique seed
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Note: Setting these to False allows non-deterministic behavior for better randomness
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return seed

def cleanup_resources(adata=None, vae=None):
    """Clean up resources"""
    try:
        if adata is not None:
            del adata
        if vae is not None:
            del vae
        gc.collect()
    except Exception:
        pass

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"error_log_{timestamp}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_files = [
        {
            "path": os.path.join(INPUT_DIR, "bifurcating_cell1000_gene10000_dataset.h5ad"),
            "type": "simulated",
            "clusterkey": "milestone",
            "id_pre_base": "VeloVAE_bifurcating_cell1000_gene10000",
            "dim_z": 4,
            "embed": "dimred"
        },
        {
            "path": os.path.join(INPUT_DIR, "7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "type": "real",
            "clusterkey": "clusters",
            "id_pre_base": "VeloVAE_7",
            "dim_z": 5,
            "embed": "umap"
        }
    ]

    n_runs = 5

    for run_idx in range(1, n_runs + 1):
        # Set different random seed for each run
        seed = set_random_seeds()
        print(f"\n[Run {run_idx}/{n_runs}] Seed: {seed}")

        for file_info in data_files:
            file_path = file_info["path"]
            data_type = file_info["type"]
            clusterkey = file_info["clusterkey"]
            id_pre_base = file_info["id_pre_base"]
            dim_z = file_info["dim_z"]
            embed = file_info["embed"]

            adata = None
            vae = None

            try:
                input_file = os.path.basename(file_path)
                id_pre = f"{id_pre_base}_r{run_idx}"

                print(f"  Processing: {input_file}")

                adata = sc.read(file_path)

                # Parameter configuration
                if adata.n_vars < 1500:
                    n_gene = 500
                elif adata.n_vars < 10000:
                    n_gene = 2000
                elif adata.n_vars < 50000:
                    n_gene = 4000
                else:
                    n_gene = 5000

                npc = 30
                if adata.n_obs < 1500:
                    config = {"batch_size": 64}
                elif adata.n_obs < 15000:
                    config = {"batch_size": 128}
                else:
                    npc = 50
                    config = {"batch_size": 256}

                adata.layers["spliced"] = csr_matrix(adata.layers["spliced"])
                adata.layers["unspliced"] = csr_matrix(adata.layers["unspliced"])
                adata.obs_names_make_unique()
                adata = adata[~adata.to_df().duplicated(), :]

                if data_type == "real":
                    vv.preprocess(adata, n_gene=n_gene, npc=npc)
                else:
                    vv.preprocess(adata, n_gene=n_gene, npc=npc, min_shared_counts=0, min_shared_cells=0)

                vae = vv.VAE(adata, tmax=20, dim_z=dim_z, device=DEVICE)

                vae.train(adata, embed=embed, config=config)

                vae.save_anndata(adata, key='vae')
                adata.obs[clusterkey] = adata.obs[clusterkey].astype(str)
                adata.obs['vae_time'] = adata.obs['vae_time'].astype(float)

                _, _ = vv.post_analysis(adata, id_pre, methods=['VeloVAE'], keys=['vae'], cluster_key=clusterkey)

                # Call velocity_embedding_stream to compute embedding (no plot saving)
                scv.pl.velocity_embedding_stream(
                    adata, vkey='vae_velocity', V=adata.obsm[f'vae_velocity_{embed}'],
                    basis=embed, show=False
                )

                output_file = os.path.join(OUTPUT_DIR, f"{id_pre}.h5ad")
                sc.write(output_file, adata, compression="lzf")

                print(f"  Saved: {id_pre}.h5ad")
                logging.info(f"Success: {input_file} run {run_idx} -> {output_file}")

                cleanup_resources(adata, vae)
                adata = None
                vae = None

            except Exception as e:
                logging.error(f"Error processing {file_path} run {run_idx}: {str(e)}", exc_info=True)

            finally:
                cleanup_resources(adata, vae)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logging.error(f"Unhandled error: {e}", exc_info=True)
    finally:
        cleanup_resources()
