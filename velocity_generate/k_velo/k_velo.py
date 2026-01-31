import os
import argparse
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import velocity


def main(data_path, save_dir, celltype_key="cell_type",
         hvgs_n=1000, hvgs_theta=100, minlim=3,
         impute_n_neighbours=30, impute_n_pcs=15,
         simulate=False):

    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()

    # normalize dense/sparse representations consistently
    if isinstance(adata.X, csr_matrix):
        adata.X = adata.X.todense()
    elif isinstance(adata.X, np.ndarray):
        adata.X = csr_matrix(adata.X)

    # ensure spliced/unspliced exist and are in dense form for this pipeline
    for layer in ["spliced", "unspliced"]:
        if layer not in adata.layers:
            raise KeyError(f"Required layer '{layer}' not found in AnnData")
        mat = adata.layers[layer]
        if hasattr(mat, "todense"):
            mat = mat.todense()
        elif isinstance(mat, np.ndarray):
            mat = csr_matrix(mat)
        mat = np.nan_to_num(mat, nan=0.0)
        adata.layers[layer] = mat

    print(f"[INFO] Kvelo preprocess {data_path}", flush=True)

    # Simulation mode: relax gene filtering and set cluster labels
    if simulate:
        print("[INFO] Simulation mode: skipping aggressive gene filtering and setting clusters to 'milestone'")
        adata.obs[celltype_key] = "milestone"
        if "X_dimred" in adata.obsm and "X_umap" not in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_dimred"]
        # if top_genes available, use them; otherwise skip subsetting
        top_genes = adata.uns.get("top_genes", None)
        if top_genes is not None:
            adata = adata[:, top_genes]
    else:
        # High-variance genes selection
        hvgs = velocity.pp.filtering.get_hvgs(adata, no_of_hvgs=hvgs_n, theta=hvgs_theta, layer='spliced')
        adata = adata[:, hvgs]

        # Select genes with high unspliced signal
        us_genes = velocity.pp.filtering.get_high_us_genes(adata, minlim_u=minlim, minlim_s=minlim)
        adata = adata[:, us_genes]

    # Normalise layers
    velocity.pp.normalisation.normalise_layers(adata, mode='combined', norm='L1', total_counts=None)
    for layer in ['spliced', 'unspliced']:
        print(f"{layer} has NaN:", np.isnan(adata.layers[layer]).any())

    # Imputation
    velocity.pp.imputation.impute_counts(adata, n_neighbours=impute_n_neighbours, layer_NN='spliced', n_pcs=impute_n_pcs)
    velocity.pp.imputation.impute_counts(adata, n_neighbours=impute_n_neighbours, layer_NN='unspliced', n_pcs=impute_n_pcs)

    # Fit reaction rate parameters and compute velocity
    velocity.tl.fit.recover_reaction_rate_pars(adata, use_raw=False)
    velocity.tl.fit.get_velocity(adata, use_raw=False, key="fit", normalise=None, scale=True)

    # Ensure save_dir exists and write output
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, os.path.basename(data_path))
    adata.write_h5ad(out_path)
    print(f"[DONE] Saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="k_velo preprocessing and velocity calculation")
    parser.add_argument("--data_path", required=True, help="Path to input .h5ad file")
    parser.add_argument("--save_dir", required=True, help="Directory to save output .h5ad")
    parser.add_argument("--celltype_key", default="cell_type", help="obs column for cell/cluster labels (default: cell_type)")
    parser.add_argument("--hvgs_n", type=int, default=1000, help="Number of HVGs to select (default: 1000)")
    parser.add_argument("--hvgs_theta", type=float, default=100.0, help="Theta parameter for HVG selection")
    parser.add_argument("--minlim", type=int, default=3, help="Minimum counts threshold for high unspliced genes (default: 3)")
    parser.add_argument("--impute_n_neighbours", type=int, default=30, help="Neighbors for imputation (default: 30)")
    parser.add_argument("--impute_n_pcs", type=int, default=15, help="PCs for imputation (default: 15)")
    parser.add_argument("--simulate", action="store_true", help="If set, relax preprocessing (skip hvgs/us gene filtering) and set clusters to 'milestone'")

    args = parser.parse_args()
    main(
        args.data_path,
        args.save_dir,
        celltype_key=args.celltype_key,
        hvgs_n=args.hvgs_n,
        hvgs_theta=args.hvgs_theta,
        minlim=args.minlim,
        impute_n_neighbours=args.impute_n_neighbours,
        impute_n_pcs=args.impute_n_pcs,
        simulate=args.simulate
    )