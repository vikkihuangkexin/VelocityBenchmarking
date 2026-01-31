#!/usr/bin/env python3
"""
Multi-run TFvelo analysis script for RNA velocity prediction.

Requirements:
- TFvelo must be installed or available in PYTHONPATH
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
import TFvelo as TFv
import anndata as ad
import scanpy as sc
import scvelo as scv
import matplotlib
matplotlib.use('AGG')

# Configuration: Modify these paths or set via environment variables
INPUT_DIR = os.getenv('INPUT_DIR', './example')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './example/output/TFvelo')

def set_random_seeds():
    """Set random seeds using current timestamp to ensure different results each run"""
    # Use microsecond timestamp and process ID to generate unique seed
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()

    random.seed(seed)
    np.random.seed(seed)

    return seed

def check_data_type(adata):
    for key in list(adata.var):
        if adata.var[key][0] in ['True', 'False']:
            adata.var[key] = adata.var[key].map({'True': True, 'False': False})
    return

def data_type_tostr(adata, key):
    if key in adata.var.keys():
        if adata.var[key][0] in [True, False]:
            adata.var[key] = adata.var[key].map({True: 'True', False:'False'})
    return

def preprocess(file_path, data_type):
    print(f'----------------------------------preprocess {os.path.basename(file_path)} ---------------------------------------------')
    adata = sc.read(file_path)

    simulate = (data_type == "simulated")
    if simulate:
        adata.obs['clusters'] = 'milestone'
    else:
        # For real data, clusters are already set or will be handled later
        pass

    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    adata.uns['genes_all'] = np.array(adata.var_names)

    if "spliced" in adata.layers:
        adata.layers["total"] = adata.layers["spliced"] + adata.layers["unspliced"]
    elif "new" in adata.layers:
        adata.layers["total"] = np.array(adata.layers["total"].todense())
    else:
        adata.layers["total"] = adata.X
    adata.layers["total_raw"] = adata.layers["total"].copy()
    n_cells, n_genes = adata.X.shape

    sc.pp.filter_genes(adata, min_cells=int(n_cells/50))
    sc.pp.filter_cells(adata, min_genes=int(n_genes/50))

    if simulate:
        TFv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=adata.n_vars, log=True)
    else:
        TFv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, log=True)
    adata.X = adata.layers["total"].copy()

    if not simulate:
        adata.uns['clusters_colors'] = np.array(['red', 'orange', 'yellow', 'green','skyblue', 'blue','purple', 'pink', '#8fbc8f', '#f4a460', '#fdbf6f', '#ff7f00', '#b2df8a', '#1f78b4',
            '#6a3d9a', '#cab2d6'], dtype=object)

    gene_names = []
    for tmp in adata.var_names:
        gene_names.append(tmp.upper())
    adata.var_names = gene_names
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    TFv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    TFv.pp.get_TFs(adata, databases=['ENCODE', 'ChEA'])
    print(adata)
    adata.uns['genes_pp'] = np.array(adata.var_names)
    if 'X_dimred' in adata.obsm:
        adata.obsm['X_umap'] = adata.obsm['X_dimred']
    return adata

def main(adata, result_path, n_jobs=28, max_iter=20, var_names="all", WX_method="lsq_linear", WX_thres=20, max_n_TF=99, n_top_genes=2000, n_time_points=1000, use_raw=0, init_weight_method="correlation"):
    print('--------------------------------')
    n_jobs_max = np.max([int(os.cpu_count()/2), 1])
    if n_jobs >= 1:
        n_jobs = np.min([n_jobs, n_jobs_max])
    else:
        n_jobs = n_jobs_max
    print('n_jobs:', n_jobs)
    flag = TFv.tl.recover_dynamics(adata, n_jobs=n_jobs, max_iter=max_iter, var_names=var_names,
        WX_method=WX_method, WX_thres=WX_thres, max_n_TF=max_n_TF, n_top_genes=n_top_genes,
        fit_scaling=True, use_raw=use_raw, init_weight_method=init_weight_method,
        n_time_points=n_time_points)
    if flag == False:
        return False
    if 'highly_variable_genes' in adata.var.keys():
        data_type_tostr(adata, key='highly_variable_genes')
    return True

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"error_log_{timestamp}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_files = [
        {
            "path": os.path.join(INPUT_DIR, "Simulation-data/bifurcating_cell1000_gene500_dataset.h5ad"),
            "type": "simulated",
            "id_pre_base": "TFvelo_bifurcating_cell1000_gene500"
        },
        {
            "path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "type": "real",
            "id_pre_base": "TFvelo_7"
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
            id_pre_base = file_info["id_pre_base"]

            adata = None

            try:
                input_file = os.path.basename(file_path)
                id_pre = f"{id_pre_base}_r{run_idx}"

                print(f"  Processing: {input_file}")

                adata = preprocess(file_path, data_type)

                result_path = os.path.join(OUTPUT_DIR, f"{id_pre}_")
                os.makedirs(result_path, exist_ok=True)

                adata.write(os.path.join(result_path, 'pp.h5ad'))

                success = main(adata, result_path)
                if success:
                    adata.write(os.path.join(result_path, 'rc.h5ad'))
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
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logging.error(f"Unhandled error: {e}", exc_info=True)