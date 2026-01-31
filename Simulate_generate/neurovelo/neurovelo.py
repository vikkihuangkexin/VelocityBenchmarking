import anndata as ann
import scvelo as scv
import scanpy as sc
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from neurovelo.train import Trainer
from neurovelo.utils import ModelAnalyzer, latent_data, evaluate, decode_gene_velocity, vector_fields_similarity

scv.settings.verbosity = 0

def main(data_dir, save_dir, n_ode_hidden, n_vae_hidden, n_latent, batch_size, nepoch, simulate=False):
    adata = ann.read_h5ad(data_dir)
    data_file = os.path.basename(data_dir)
    file_id = os.path.splitext(data_file)[0]

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NeuroVelo Single-cell RNA Velocity Analysis Script")
    parser.add_argument("--data_dir",
                        default="./test.h5ad",
                        help="Input h5ad data file path")
    parser.add_argument("--save_dir",
                        default="./test",
                        help="Result saving directory")
    parser.add_argument("--n_ode_hidden", type=int, default=100,
                        help="Number of hidden units in ODE network")
    parser.add_argument("--n_vae_hidden", type=int, default=100,
                        help="Number of hidden units in VAE network")
    parser.add_argument("--n_latent", type=int, default=50,
                        help="Number of latent dimensions")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for training")
    parser.add_argument("--nepoch", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--simulate", action='store_true',
                        help="Whether the data is simulation data")
    args = parser.parse_args()

    main(args.data_dir, args.save_dir, args.n_ode_hidden, args.n_vae_hidden, args.n_latent, args.batch_size, args.nepoch, args.simulate)