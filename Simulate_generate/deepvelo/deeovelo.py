import anndata as ann
import deepvelo as dv
import scvelo as scv
import os
import argparse

scv.settings.verbosity = 0

def main(data_dir, save_dir, simulate=False):
    adata = ann.read_h5ad(data_dir)
    data_file = os.path.basename(data_dir)
    file_id = os.path.splitext(data_file)[0]

    if simulate:
        scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=adata.n_vars)
    else:
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_neighbors=30, n_pcs=30)

    trainer = dv.train(adata, dv.Constants.default_configs)

    scv.tl.umap(adata)
    scv.tl.velocity_graph(adata)

    if 'X_dimred' in adata.obsm:
        adata.obsm['X_umap'] = adata.obsm['X_dimred']

    os.makedirs(save_dir, exist_ok=True)
    output_h5ad_path = os.path.join(save_dir, f"{file_id}_deepvelo.h5ad")

    adata.write(output_h5ad_path)
    print(f"Analysis completed! Results saved to: {output_h5ad_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepVelo Single-cell RNA Velocity Analysis Script")
    parser.add_argument("--data_dir", 
                        default="./test.h5ad",
                        help="Input h5ad data file path")
    parser.add_argument("--save_dir",
                        default="./test",
                        help="Result saving directory")
    parser.add_argument("--simulate", action='store_true',
                        help="Whether the data is simulation data")
    args = parser.parse_args()

    main(args.data_dir, args.save_dir, args.simulate)