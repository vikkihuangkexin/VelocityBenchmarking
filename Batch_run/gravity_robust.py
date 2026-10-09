#!/usr/bin/env python3
"""
Multi-run GRAVITY analysis script for RNA velocity prediction.

Requirements:
- GRAVITY must be installed (see Benchmarked_tools/GRAVITY/GRAVITY.py)
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
import shutil
import sys

import numpy as np
import scanpy as sc

# GRAVITY wrapper lives in the sibling Benchmarked_tools directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Benchmarked_tools", "GRAVITY")
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

import GRAVITY

# Allow the caller to choose a GPU without overriding an existing pinning.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Configuration: Modify these paths or set via environment variables
INPUT_DIR = os.getenv('INPUT_DIR', './example')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './example/output/GRAVITY')


def set_random_seeds():
    """Set random seeds using current timestamp to ensure different results each run"""
    # Use microsecond timestamp and process ID to generate unique seed
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()

    random.seed(seed)
    np.random.seed(seed)

    return seed


def main_single(file_path, save_dir, simulate=False, cluster_key=None, dimred_key='X_umap'):
    """Preprocess one dataset, write pp.h5ad, run GRAVITY and write rc.h5ad."""
    os.makedirs(save_dir, exist_ok=True)

    # The species prior is detected from the original file name (preprocessed
    # files such as pp.h5ad no longer carry the species tag).
    prior = GRAVITY.resolve_prior_network(os.path.basename(file_path), None, simulate)

    adata = sc.read(file_path)
    adata = GRAVITY.preprocess_adata(
        adata,
        dimred_key=dimred_key,
        simulate=simulate,
        min_shared_counts=20,
        n_top_genes=2000,
    )
    pp_path = os.path.join(save_dir, 'pp.h5ad')
    adata.write_h5ad(pp_path)
    del adata
    gc.collect()

    result_path = GRAVITY.run_gravity_analysis(
        input_path=pp_path,
        output_dir=save_dir,
        cluster_key=cluster_key,
        dataset_name='gravity_run',
        dimred_key=dimred_key,
        simulate=simulate,
        prior_network=prior,
        preprocessed=True,
        overwrite=True,
    )

    shutil.copyfile(result_path, os.path.join(save_dir, 'rc.h5ad'))
    return result_path


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
            "id_pre_base": "GRAVITY_bifurcating_cell1000_gene500"
        },
        {
            "path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "simulate": False,
            "cluster_key": "vis_annotation",
            "id_pre_base": "GRAVITY_7"
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
            id_pre_base = file_info["id_pre_base"]

            try:
                input_file = os.path.basename(file_path)
                id_pre = f"{id_pre_base}_r{run_idx}"

                print(f"  Processing: {input_file}")

                save_dir = os.path.join(OUTPUT_DIR, id_pre)

                main_single(
                    file_path,
                    save_dir,
                    simulate=simulate,
                    cluster_key=cluster_key,
                )

                print(f"  Saved: {id_pre}")
                logging.info(f"Success: {input_file} run {run_idx}")

                gc.collect()

            except Exception as e:
                logging.error(f"Error processing {file_path} run {run_idx}: {str(e)}", exc_info=True)

            finally:
                gc.collect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logging.error(f"Unhandled error: {e}", exc_info=True)
