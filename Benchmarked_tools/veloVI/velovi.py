import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch
import os
import argparse
os.environ.setdefault("NCCL_DEBUG", "INFO")
from velovi import preprocess_data, VELOVI
from unit import find_cluster_column


def main(data_dir, data_file, save_dir, gpu_numbers=None, batch_size=1024, simulate=True):
    if gpu_numbers:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(s) for s in gpu_numbers])

    adata = sc.read(data_dir, cache=True)
    adata.obs_names_make_unique()
    # remove duplicated cells if present
    adata = adata[~adata.to_df().duplicated(), :]
    cluster = find_cluster_column(adata)

    if simulate:
        # Simulation-style preprocessing
        cluster = 'milestone'
        if 'X_dimred' in adata.obsm and 'X_umap' not in adata.obsm:
            adata.obsm['X_umap'] = adata.obsm['X_dimred']
        if adata.n_vars < 10000:
            top_gene = (adata.n_vars // 500) * 500
            top_gene = min(top_gene, adata.n_vars, 500)
            shared_counts = 20
        else:
            top_gene = 2000
            shared_counts = 1
        scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=top_gene)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30, method='umap')
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    else:
        # Real-data preprocessing
        cluster = find_cluster_column(adata)
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30, method='umap')

    adata = preprocess_data(adata)
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train(batch_size=batch_size)

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

    os.makedirs(save_dir, exist_ok=True)
    scv.pl.velocity_embedding_stream(adata, basis='umap', color=cluster, save=f'{save_dir}/stream_arrow.pdf')
    scv.pl.velocity_embedding_grid(adata, basis='umap', color=cluster, save=f'{save_dir}/grid_arrow.pdf')
    scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120, save=f'{save_dir}/full_arrow.pdf')
    adata.write_h5ad(f'{save_dir}/{data_file.split(".")[0]}_velo.h5ad')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run VELOVI for one dataset')
    parser.add_argument('--data_dir', required=True, help='Path to input .h5ad file')
    parser.add_argument('--data_file', required=True, help='Input filename identifier (used for output naming)')
    parser.add_argument('--save_dir', required=True, help='Directory to save outputs')
    parser.add_argument('--gpu_numbers', default='0', help='Comma-separated GPU indices (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training (default: 1024)')
    parser.add_argument('--simulate', dest='simulate', action='store_true', help='Treat input as simulation (default)')
    parser.add_argument('--no-simulate', dest='simulate', action='store_false', help='Treat input as real data')
    parser.set_defaults(simulate=True)
    args = parser.parse_args()

    gpu_list = [int(x) for x in str(args.gpu_numbers).split(',')] if args.gpu_numbers else [0]
    os.makedirs(args.save_dir, exist_ok=True)
    main(args.data_dir, args.data_file, args.save_dir, gpu_numbers=gpu_list, batch_size=args.batch_size, simulate=args.simulate)