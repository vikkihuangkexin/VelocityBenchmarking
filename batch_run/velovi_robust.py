import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch
import os
from velovi import preprocess_data, VELOVI
from unit import find_cluster_column
import random
import datetime
import logging

GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"

# Configuration: Modify these paths or set via environment variables
INPUT_DIR = os.getenv('INPUT_DIR', './example')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './example/output/VeloVI')

def set_random_seeds():
    """Set random seeds using current timestamp to ensure different results each run"""
    # Use microsecond timestamp and process ID to generate unique seed
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return seed

def main_single(data_dir, save_dir, r, simulate=False):
    set_random_seeds()
    adata = sc.read(data_dir, cache=True)
    data_file = os.path.basename(data_dir)
    ID = os.path.splitext(data_file)[0]
    obsm_key = list(adata.obsm.keys())
    if not any(item.lower().find('x_umap') != -1 for item in obsm_key):
        sc.tl.umap(adata)
    adata.obs_names_make_unique()
    cluster = find_cluster_column(adata)

    if simulate:
        scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=adata.n_vars)
    else:
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30, method='umap')

    adata = preprocess_data(adata)
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train(batch_size=1024)

    latent_time = vae.get_latent_time(n_samples=25)
    velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")

    t = latent_time
    scaling = 20 / t.max(0)

    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_velovi"] = latent_time

    adata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
    adata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
    adata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
    adata.var["fit_t_"] = (
                              torch.nn.functional.softplus(vae.module.switch_time_unconstr)
                                  .detach()
                                  .cpu()
                                  .numpy()
                          ) * scaling
    ss = np.array(scaling)
    adata.layers["fit_t"] = latent_time.values * ss[np.newaxis, :]
    adata.var['fit_scaling'] = 1.0

    scv.tl.velocity_graph(adata)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    scv.pl.velocity_embedding_stream(adata, basis='umap', color=cluster, save=os.path.join(save_dir, 'stream_arrow.pdf'))
    scv.pl.velocity_embedding_grid(adata, basis='umap', color=cluster, save=os.path.join(save_dir, 'grid_arrow.pdf'))
    scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120, save=os.path.join(save_dir, 'full_arrow.pdf'))
    adata.write_h5ad(os.path.join(save_dir, f'{ID}_velo.h5ad'))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"error_log_{timestamp}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_files = [
        {
            "path": os.path.join(INPUT_DIR, "Simulation-data/bifurcating_cell1000_gene500_dataset.h5ad"),
            "simulate": True,
            "id_pre_base": "VeloVI_bifurcating_cell1000_gene500"
        },
        {
            "path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "simulate": False,
            "id_pre_base": "VeloVI_7"
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

            try:
                input_file = os.path.basename(file_path)
                id_pre = f"{id_pre_base}_r{run_idx}"

                print(f"  Processing: {input_file}")

                save_dir = os.path.join(OUTPUT_DIR, id_pre)

                main_single(file_path, save_dir, run_idx, simulate=simulate)

                print(f"  Saved: {id_pre}")
                logging.info(f"Success: {input_file} run {run_idx}")

            except Exception as e:
                logging.error(f"Error processing {file_path} run {run_idx}: {str(e)}", exc_info=True)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logging.error(f"Unhandled error: {e}", exc_info=True)