import pandas as pd
import matplotlib.pyplot as plt
import phylovelo as pv
import numpy as np
from scipy.stats import spearmanr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import anndata as ad
import scanpy as sc
import scvelo as scv
import os
import argparse

def main(data_dir, save_dir, celltype_key, umap_key, lineage_info):
    os.makedirs(save_dir, exist_ok=True)
    print(f'[INFO] Read in {data_path}', flush=True)
    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # Simulation mode adjustments
    if simulate:
        print("[INFO] Simulation mode: setting cluster labels to 'milestone' and copying X_dimred to X_umap if present")
        adata.obs[celltype_key] = "milestone"
        if "X_dimred" in adata.obsm and "X_umap" not in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_dimred"]
        # Relax preprocessing: use top_genes if available
        if "top_genes" in adata.uns:
            n_top_genes = len(adata.uns["top_genes"])
        min_count = None  # Relax min_count

    target_lineage = lineage_info
    label_to_id = {}
    next_id = 0

    # Ensure target_lineage is a list of paths (each path is a list of labels)
    if isinstance(target_lineage[0], str):
        target_lineage = [target_lineage]

    for path in target_lineage:
        for label in path:
            if label not in label_to_id:
                label_to_id[label] = next_id
                next_id += 1

    adata.obs['cell_lineage'] = adata.obs[celltype_key].map(label_to_id)
    if adata.shape[1] > 2000:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    print(f'Phylovelo calculation for {celltype_key}', flush=True)
    sd = pv.scData(count=pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names), Xdr=pd.DataFrame(adata.obsm[umap_key], index=adata.obs_names, columns=['UMAP1', 'UMAP2']), cell_generation=np.array(adata.obs[celltype_key]))
    sd.drop_duplicate_genes(target='count')
    sd.normalize_filter(is_normalize=False, is_log=False, min_count=10, target_sum=None)
    sd = pv.velocity_inference(sd, sd.cell_generation, cutoff=0.95, target='count')
    sd = pv.velocity_embedding(sd, target='count', n_neigh=15)
    adata.obsm['phylovelo_velocity'] = sd.velocity_embeded
    sd = pv.calc_phylo_pseudotime(sd, n_neighbors=10, r_sample=0.1)
    adata.obs['latent_time'] = sd.phylo_pseudotime
    print(f'[INFO] Finish calculation for {data_path}', flush=True)

    adata.write_h5ad(f'{save_dir}/adata.h5ad')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phylovelo velocity computation")
    parser.add_argument("--data_path", required=True, help="Path to input .h5ad file")
    parser.add_argument("--save_dir", required=True, help="Directory to save results")
    parser.add_argument("--celltype_key", default="cell_type", help="obs column for cell type (default: cell_type)")
    parser.add_argument("--umap_key", default="umap", help="obsm key for UMAP coordinates (default: umap)")
    parser.add_argument("--simulate", action="store_true", help="If set, set clusters to 'milestone' and copy X_dimred to X_umap if present")

    parser.add_argument(
        "--lineage_path",
        action="append",
        default=None,
        help=(
            "Lineage path(s). "
            "Single path example: --lineage_path s1,s2,s3 ; "
            "Multiple paths example: "
            "--lineage_path sA,sB,sBmid,sC,sEndC "
            "--lineage_path sA,sB,sBmid,sD,sEndD"
        )
    )
    args = parser.parse_args()
    paths = [p.split(",") for p in args.lineage_path]

    if len(paths) == 1:
        lineage_info = paths[0]
    else:
        lineage_info = paths

    main(args.data_path, args.save_dir, args.celltype_key, args.umap_key, lineage_info, args.simulate)