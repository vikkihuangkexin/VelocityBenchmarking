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

def main(data_dir, save_dir):
    adata = ann.read_h5ad(data_dir)
    data_file = os.path.basename(data_dir)
    file_id = os.path.splitext(data_file)[0]

    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    sc.pp.log1p(adata, layer='spliced')
    sc.pp.log1p(adata, layer='unspliced')
    scv.pp.neighbors(adata, n_neighbors=30, n_pcs=20)
    adata.obs['sample'] = 0

    model = Trainer(adata, layer='spliced', odesample_obs='sample', n_sample=1, percent=0.8, nepoch=100,
                    n_ode_hidden=100, n_vae_hidden=100, n_latent=50, batch_size=100, reconstruct_xt=True)
    model.train()

    os.makedirs(save_dir, exist_ok=True)
    model.save_model(save_dir, file_id)
    model_path = os.path.join(save_dir, f"{file_id}.pth")
    adata.layers['spliced_velocity'] = decode_gene_velocity(adata, model_path, layer='spliced')

    scv.pp.neighbors(adata)
    scv.tl.umap(adata)
    scv.tl.velocity_graph(adata, vkey='spliced_velocity', xkey='spliced')
    scv.tl.velocity_embedding(adata, vkey='spliced_velocity', basis='umap')

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
    args = parser.parse_args()

    main(args.data_dir, args.save_dir)