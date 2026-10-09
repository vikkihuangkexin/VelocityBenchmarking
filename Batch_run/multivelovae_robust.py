#!/usr/bin/env python3
"""
Multi-run MultiVeloVAE analysis script for RNA velocity prediction.

Requirements:
- MultiVeloVAE must be installed (see Benchmarked_tools/MultiVeloVAE/MultiVeloVAE.py)
- Input data files should be in the INPUT_DIR
- Results will be saved to OUTPUT_DIR

Configuration:
- Set INPUT_DIR and OUTPUT_DIR via environment variables or modify defaults below
- MultiVeloVAE reads the chromatin modality from `adata.layers['Mc']` (falling back to
  `chromatin`), so the multimodal H5AD must already carry that layer.
"""

import datetime
import gc
import logging
import os
import random
import sys

import numpy as np
import scanpy as sc

# The MultiVeloVAE wrapper lives in the sibling Benchmarked_tools directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Benchmarked_tools", "MultiVeloVAE")
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

import MultiVeloVAE

# Allow the caller to choose a GPU without overriding an existing pinning.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Configuration: Modify these paths or set via environment variables
INPUT_DIR = os.getenv('INPUT_DIR', './example')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './example/output/MultiVeloVAE')


def set_random_seeds():
    """Set random seeds using current timestamp to ensure different results each run"""
    # Use microsecond timestamp and process ID to generate unique seed
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()

    random.seed(seed)
    np.random.seed(seed)

    return seed


def main_single(file_path, save_dir, simulate=False, cluster_key=None, atac_layer="Mc",
                dataset_name=None, batch_size=32, key="vae", seed=2024):
    """Preprocess one dataset, write pp.h5ad, run MultiVeloVAE and write rc.h5ad."""
    os.makedirs(save_dir, exist_ok=True)

    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    cluster_key = cluster_key or "celltype"

    adata = sc.read_h5ad(file_path)
    adata, resolved_atac_layer, active_cluster_key, active_embed = MultiVeloVAE.preprocess_adata(
        adata,
        cluster_key=cluster_key,
        atac_layer=atac_layer,
        embed="umap",
        dimred_key="X_umap",
        simulate=simulate,
    )
    adata.uns['multivelovae_cluster_key'] = active_cluster_key
    adata.write(os.path.join(save_dir, 'pp.h5ad'))

    result_path = MultiVeloVAE.run_multivelovae_analysis(
        input_path=file_path,
        output_dir=save_dir,
        cluster_key=cluster_key,
        dataset_name=dataset_name,
        atac_layer=atac_layer,
        simulate=simulate,
        batch_size=batch_size,
        key=key,
        overwrite=True,
        seed=seed,
    )

    result = sc.read_h5ad(result_path)
    result.write(os.path.join(save_dir, 'rc.h5ad'))
    return result


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
            "atac_layer": "Mc",
            "id_pre_base": "MultiVeloVAE_bifurcating_cell1000_gene500"
        },
        {
            "path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "simulate": False,
            "cluster_key": "celltype",
            "atac_layer": "Mc",
            "id_pre_base": "MultiVeloVAE_7"
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
            cluster_key = file_info["cluster_key"]
            atac_layer = file_info.get("atac_layer", "Mc")
            id_pre_base = file_info["id_pre_base"]

            adata = None

            try:
                input_file = os.path.basename(file_path)
                id_pre = f"{id_pre_base}_r{run_idx}"

                print(f"  Processing: {input_file}")

                save_dir = os.path.join(OUTPUT_DIR, id_pre)
                os.makedirs(save_dir, exist_ok=True)

                adata = main_single(
                    file_path=file_path,
                    save_dir=save_dir,
                    simulate=simulate,
                    cluster_key=cluster_key,
                    atac_layer=atac_layer,
                    dataset_name=id_pre,
                    seed=seed,
                )

                print(f"  Saved: {id_pre}")
                logging.info(f"Success: {input_file} run {run_idx}")

                gc.collect()

            except Exception as e:
                logging.error(f"Error processing {file_path} run {run_idx}: {str(e)}", exc_info=True)
                print(f"  Error: {e}")

            finally:
                del adata
                gc.collect()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logging.error(f"Unhandled error: {e}", exc_info=True)
