import sdevelo as sv
import scanpy as sc
import os
import argparse
import tensorflow as tf
import numpy as np
import random
import datetime
import logging

# Configuration
INPUT_DIR = os.getenv('INPUT_DIR', './example')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './example/output/SDEvelo')

def set_random_seeds():
    """Set random seeds using current timestamp"""
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return seed

def main_single(data_path, save_dir, celltype_key, gpu, n_epochs, simulate):
    # Configure device visibility
    if gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device_msg = f"cuda:{gpu}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device_msg = "cpu"

    print(f"[INFO] Using device: {device_msg}")

    # Load data
    adata = sc.read_h5ad(data_path)
    print(f"[INFO] Read in {data_path}")

    # If running on simulated data, relax preprocessing and adjust metadata
    if simulate:
        print("[INFO] Simulation mode: relaxing preprocessing and setting cluster labels to 'milestone'")
        adata.obs[celltype_key] = "milestone"
        if "X_dimred" in adata.obsm and "X_umap" not in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_dimred"]
        adata.uns["min_shared_counts"] = None
        if "top_genes" in adata.uns:
            adata.uns["n_top_genes"] = adata.uns.get("top_genes")
        else:
            adata.uns["n_top_genes"] = None

    # Build configuration for SDEvelo
    cfg = sv.Config()
    cfg.cuda_device = gpu
    cfg.vis_type_col = celltype_key
    try:
        cfg.nEpochs = int(n_epochs)
    except Exception:
        pass

    model = sv.SDENN(cfg, adata)

    print(f"[INFO] Start training for {data_path}")
    adata = model.train(cfg.nEpochs)
    print(f"[INFO] Finish training {data_path}")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "adata.h5ad")
    adata.write_h5ad(out_path)
    print(f"[DONE] Saved to {out_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"error_log_{timestamp}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Defined per reference rubost code
    data_files = [
        {
            "path": os.path.join(INPUT_DIR, "Simulation-data/bifurcating_cell1000_gene500_dataset.h5ad"),
            "simulate": True,
            "id_pre_base": "SDEvelo_bifurcating_cell1000_gene500",
            "celltype_key": "cell_type" # Will be overwritten to 'milestone' inside if simulate=True
        },
        {
            "path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "simulate": False,
            "id_pre_base": "SDEvelo_7",
            "celltype_key": "cell_type" 
        }
    ]

    n_runs = 5
    gpu_id = 0 # Default GPU

    for run_idx in range(1, n_runs + 1):
        seed = set_random_seeds()
        print(f"\n[Run {run_idx}/{n_runs}] Seed: {seed}")

        for file_info in data_files:
            file_path = file_info["path"]
            simulate = file_info["simulate"]
            id_pre_base = file_info["id_pre_base"]
            celltype_key = file_info["celltype_key"]

            try:
                input_file = os.path.basename(file_path)
                id_pre = f"{id_pre_base}_r{run_idx}"
                print(f"  Processing: {input_file}")

                save_dir = os.path.join(OUTPUT_DIR, id_pre)
                
                # Default epochs logic from original script
                n_epochs = 200 

                main_single(file_path, save_dir, celltype_key, gpu_id, n_epochs, simulate)

                logging.info(f"Success: {input_file} run {run_idx}")

            except Exception as e:
                logging.error(f"Error processing {file_path} run {run_idx}: {str(e)}", exc_info=True)
                print(f"Error: {e}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logging.error(f"Unhandled error: {e}", exc_info=True)