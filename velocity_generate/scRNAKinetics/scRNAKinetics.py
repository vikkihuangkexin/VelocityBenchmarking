import scanpy as sc
import numpy as np
import scvelo as scv
import cellrank as cr
from cellrank.kernels import CytoTRACEKernel
from rnakinetics import kinetics_inference
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

def main(data_path, save_dir, simulate=False, n_top_genes=200, num_iter=10, n_jobs=64):
    print(f'[INFO] Read in {data_path}', flush=True)
    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # Simulation mode adjustments
    if simulate:
        print("[INFO] Simulation mode: setting cluster labels to 'milestone' and copying X_dimred to X_umap if present")
        adata.obs['milestone'] = "milestone"  # Assuming group_key is 'milestone'
        if "X_dimred" in adata.obsm and "X_umap" not in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_dimred"]
        # Relax preprocessing: use top_genes if available
        if "top_genes" in adata.uns:
            n_top_genes = len(adata.uns["top_genes"])

    print(f'[INFO] SCV preprocess {data_path}', flush=True)
    scv.pp.filter_and_normalize(adata, n_top_genes=n_top_genes)
    scv.pp.moments(adata, n_neighbors=30)
    ctk = CytoTRACEKernel(adata).compute_cytotrace()
    pt = adata.obs['ct_pseudotime']
    adata = kinetics_inference(adata, mode='coarse-grained', pt_key='ct_pseudotime', group_key='milestone', num_iter=num_iter, n_jobs=n_jobs, optimizer='jax')
    adata.write(f'{save_dir}/lores.h5ad')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="scRNAKinetics velocity computation")
    parser.add_argument("--data_path", required=True, help="Path to input .h5ad file")
    parser.add_argument("--save_dir", required=True, help="Directory to save results")
    parser.add_argument("--simulate", action="store_true", help="If set, set clusters to 'milestone' and copy X_dimred to X_umap if present")
    parser.add_argument("--n_top_genes", type=int, default=200, help="Number of top genes to select (default: 200)")
    parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations for kinetics inference (default: 10)")
    parser.add_argument("--n_jobs", type=int, default=64, help="Number of jobs for parallel processing (default: 64)")
    args = parser.parse_args()
    main(args.data_path, args.save_dir, args.simulate, args.n_top_genes, args.num_iter, args.n_jobs)