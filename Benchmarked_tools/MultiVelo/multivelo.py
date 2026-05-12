import os
import scipy
import numpy as np
import pandas as pd
import math
import sys
import Simulate_generate.multivelo.multivelo as mv
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import requests
from dtw import *
import time
import argparse

scv.settings.verbosity = 3
scv.settings.presenter_view = True
scv.set_figure_params('scvelo')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
np.set_printoptions(suppress=True)
mv.settings.VERBOSITY = 0

def main(rna_dir, atac_dir, save_dir, max_iter, n_jobs, n_anchors, simulate=False):
    mv.settings.LOG_FILENAME = "Fig4_" + str(time.time()) + ".txt"
    
    adata_rna = sc.read(rna_dir)
    adata_atac = sc.read(atac_dir)

    adata_rna_scv = adata_rna.copy()
    scv.tl.recover_dynamics(adata_rna_scv)
    scv.tl.velocity(adata_rna_scv, mode="dynamical")
    scv.tl.velocity_graph(adata_rna_scv, n_jobs=1)
    scv.tl.latent_time(adata_rna_scv)
    color_key = 'milestone' if simulate else 'celltype'
    scv.pl.velocity_embedding_stream(adata_rna_scv, basis='umap', color=color_key)

    scv.tl.recover_dynamics(adata_rna)
    scv.tl.velocity(adata_rna, mode="dynamical")
    scv.tl.velocity_graph(adata_rna, n_jobs=1)

    adata_result = mv.recover_dynamics_chrom(adata_rna,
                                            adata_atac,
                                            max_iter=max_iter,
                                            init_mode="invert",
                                            parallel=True,
                                            n_jobs=n_jobs,
                                            save_plot=False,
                                            rna_only=False,
                                            fit=True,
                                            n_anchors=n_anchors,
                                            extra_color_key=color_key
                                            )

    if 'X_dimred' in adata_result.obsm:
        adata_result.obsm['X_umap'] = adata_result.obsm['X_dimred']

    os.makedirs(save_dir, exist_ok=True)
    output_h5ad_path = os.path.join(save_dir, "multivelo_result.h5ad")
    adata_result.write(output_h5ad_path)
    print(f"Analysis completed! Results saved to: {output_h5ad_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MultiVelo Single-cell Multi-omics Velocity Analysis Script")
    parser.add_argument("--rna_dir",
                        default="./adata_postpro.h5ad",
                        help="Input RNA h5ad data file path")
    parser.add_argument("--atac_dir",
                        default="./adata_atac_postpro.h5ad",
                        help="Input ATAC h5ad data file path")
    parser.add_argument("--save_dir",
                        default="./test",
                        help="Result saving directory")
    parser.add_argument("--max_iter", type=int, default=5,
                        help="Maximum iterations for recover_dynamics_chrom")
    parser.add_argument("--n_jobs", type=int, default=15,
                        help="Number of jobs for parallel processing in recover_dynamics_chrom")
    parser.add_argument("--n_anchors", type=int, default=500,
                        help="Number of anchors for recover_dynamics_chrom")
    parser.add_argument("--simulate", action='store_true',
                        help="Whether the data is simulation data")
    args = parser.parse_args()

    main(args.rna_dir, args.atac_dir, args.save_dir, args.max_iter, args.n_jobs, args.n_anchors, args.simulate)