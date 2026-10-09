#!/usr/bin/env python3
"""
Multi-run DynaVelo analysis script for RNA velocity prediction.

Requirements:
- DynaVelo must be installed (see Benchmarked_tools/DynaVelo/DynaVelo.py)
- Input data files should be in the INPUT_DIR
- Results will be saved to OUTPUT_DIR

Configuration:
- Set INPUT_DIR and OUTPUT_DIR via environment variables or modify defaults below
- DynaVelo needs a TF motif accessibility matrix (`obsm['chromVAR']`). It is read from
  the same H5AD when present, otherwise from the paired ATAC file listed in
  `data_files` (looked up under ATAC_INPUT_DIR).
"""

import datetime
import gc
import logging
import os
import random
import sys

import numpy as np
import scanpy as sc

# The DynaVelo wrapper lives in the sibling Benchmarked_tools directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Benchmarked_tools", "DynaVelo")
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

import DynaVelo

# Allow the caller to choose a GPU without overriding an existing pinning.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Configuration: Modify these paths or set via environment variables
INPUT_DIR = os.getenv('INPUT_DIR', './example')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './example/output/DynaVelo')
ATAC_INPUT_DIR = os.getenv('ATAC_INPUT_DIR', INPUT_DIR)


def set_random_seeds():
    """Set random seeds using current timestamp to ensure different results each run"""
    # Use microsecond timestamp and process ID to generate unique seed
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()

    random.seed(seed)
    np.random.seed(seed)

    return seed


def main_single(file_path, save_dir, atac_path=None, simulate=False, cluster_key=None,
                dataset_name=None, batch_size=32, max_epoch=200, n_samples=20, seed=2024):
    """Preprocess one dataset, write pp.h5ad, run DynaVelo and write rc.h5ad."""
    os.makedirs(save_dir, exist_ok=True)

    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    cluster_key = cluster_key or "celltype"

    adata = sc.read(file_path)
    adata, active_cluster_key = DynaVelo.preprocess_adata(
        adata,
        cluster_key=cluster_key,
        simulate=simulate,
    )
    adata.uns['dynavelo_cluster_key'] = active_cluster_key
    adata.write(os.path.join(save_dir, 'pp.h5ad'))

    result_path = DynaVelo.run_dynavelo_analysis(
        rna_input=file_path,
        atac_input=atac_path,
        output_dir=save_dir,
        cluster_key=cluster_key,
        dataset_name=dataset_name,
        batch_size=batch_size,
        max_epoch=max_epoch,
        n_samples=n_samples,
        simulate=simulate,
        overwrite=True,
        seed=seed,
    )

    result = sc.read(result_path)
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
            "atac_path": None,
            "simulate": True,
            "cluster_key": "milestone",
            "id_pre_base": "DynaVelo_bifurcating_cell1000_gene500"
        },
        {
            "path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "atac_path": os.path.join(ATAC_INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188_atac.h5ad"),
            "simulate": False,
            "cluster_key": "celltype",
            "id_pre_base": "DynaVelo_7"
        }
    ]

    n_runs = 5

    for run_idx in range(1, n_runs + 1):
        # Set different random seed for each run
        seed = set_random_seeds()
        print(f"\n[Run {run_idx}/{n_runs}] Seed: {seed}")

        for file_info in data_files:
            file_path = file_info["path"]
            atac_path = file_info.get("atac_path")
            simulate = file_info["simulate"]
            cluster_key = file_info["cluster_key"]
            id_pre_base = file_info["id_pre_base"]

            adata = None

            try:
                input_file = os.path.basename(file_path)
                id_pre = f"{id_pre_base}_r{run_idx}"

                print(f"  Processing: {input_file}")

                save_dir = os.path.join(OUTPUT_DIR, id_pre)
                os.makedirs(save_dir, exist_ok=True)

                # Only pass the ATAC companion when it actually exists.
                resolved_atac = atac_path if atac_path and os.path.exists(atac_path) else None

                adata = main_single(
                    file_path=file_path,
                    save_dir=save_dir,
                    atac_path=resolved_atac,
                    simulate=simulate,
                    cluster_key=cluster_key,
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
