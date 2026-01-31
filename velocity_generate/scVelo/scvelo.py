import os.path

import scvelo as scv
import scanpy as sc
import pandas as pd
import argparse
from unit import find_cluster_column


def main(data_dir, data_file, save_dir, simulate=True):
    # Stochastic model flow
    adata = sc.read(data_dir, cache=True)
    adata.obs_names_make_unique()

    if simulate:
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
    else:
        cluster = find_cluster_column(adata)
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)

    if adata.n_vars == 0:
        print(f'{data_file} shape error!')
        return data_dir

    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30, method='umap')
    if not simulate:
        cluster = find_cluster_column(adata)
        if data_file.endswith('time.h5ad'):
            cluster = 'time'

    scv.tl.velocity(adata, mode='stochastic')
    os.makedirs(save_dir, exist_ok=True)
    try:
        scv.tl.velocity_graph(adata)
        scv.pl.velocity_embedding_stream(adata, basis='umap', color=cluster, save=os.path.join(save_dir, 'stream_arrow.pdf'))
        scv.pl.velocity_embedding_grid(adata, basis='umap', color=cluster, save=os.path.join(save_dir, 'grid_arrow.pdf'))
        scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120, save=os.path.join(save_dir, 'full_arrow.pdf'))
        adata.write_h5ad(os.path.join(save_dir, f'{data_file.split(".")[0]}_velo.h5ad'))
    except Exception:
        adata.write_h5ad(os.path.join(save_dir, f'{data_file.split(".")[0]}_velo.h5ad'))

    # Dynamical model flow
    adata = sc.read(data_dir, cache=True)
    adata.obs_names_make_unique()

    if simulate:
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
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30, method='umap')
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    cluster = find_cluster_column(adata)
    scv.tl.recover_dynamics(adata, n_jobs=8)
    scv.tl.velocity(adata, mode='dynamical')
    os.makedirs(save_dir, exist_ok=True)
    try:
        scv.tl.velocity_graph(adata)
        scv.pl.velocity_embedding_stream(adata, basis='umap', color=cluster, save=os.path.join(save_dir, 'stream_arrow_D.pdf'))
        scv.pl.velocity_embedding_grid(adata, basis='umap', color=cluster, save=os.path.join(save_dir, 'grid_arrow_D.pdf'))
        scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120, save=os.path.join(save_dir, 'full_arrow_D.pdf'))
        adata.write_h5ad(os.path.join(save_dir, f'{data_file.split(".")[0]}_velo_D.h5ad'))
    except Exception:
        adata.write_h5ad(os.path.join(save_dir, f'{data_file.split(".")[0]}_velo_D.h5ad'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run scVelo flows (stochastic + dynamical) for a dataset')
    parser.add_argument('--data_dir', required=True, help='Path to input .h5ad file')
    parser.add_argument('--data_file', required=True, help='Filename used for output naming')
    parser.add_argument('--save_dir', required=True, help='Directory to save outputs')
    args = parser.parse_args()
    main(args.data_dir, args.data_file, args.save_dir)

