import os
from numpy import save
import pandas as pd
import scanpy as sc
import scvelo as scv
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Iterable, List, Literal, Optional, Tuple
from anndata import AnnData
import traceback
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix
import argparse



def keep_type(adata: AnnData, nodes: Iterable[int], target: str, k_cluster: str):
    """
    Select cells of targeted type

    Args:
        adata (anndata.AnnData):
            Anndata object.
        nodes (Iterable[int]):
            Indexes for cells
        target (str):
            Cluster name.
        k_cluster (str):
            Cluster key in adata.obs dataframe

    Returns:
        list:
             Selected cells.
    """

    return nodes[adata.obs[k_cluster][nodes].values == target]

def inner_cluster_coh(
    adata: AnnData,
    k_cluster: str,
    k_velocity: str,
    gene_mask: Optional[np.ndarray] = None,
    return_raw: bool = False
) -> Tuple[Dict, float]:
    """
    In-Cluster Coherence.

    Measures the average consistency of RNA velocity in each distinct cell type.

    Args:
        adata (anndata.AnnData): AnnData object.
        k_cluster (str): Key to the cluster column in adata.obs DataFrame.
        k_velocity (str): Key to the velocity matrix in adata.layers.
        gene_mask (Optional[np.ndarray], optional): Boolean array to filter out genes. Defaults to None.
        return_raw (bool, optional): Return aggregated or raw scores. Defaults to False.

    Returns:
        Tuple[Dict, float]: 
            - Dict: all_scores indexed by cluster_edges mean scores indexed by cluster_edges.
            - float: Average score over all cells.
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}

    # Get the connectivity matrix
    connectivities = adata.obsp[adata.uns['neighbors']['connectivities_key']]
    
    # Convert to CSR format if it's not already
    if not isinstance(connectivities, csr_matrix):
        connectivities = connectivities.tocsr()

    def get_neighbors(idx):
        return connectivities[idx].indices

    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        sel_indices = np.where(sel)[0]
        
        velocities = adata.layers[k_velocity]
        nan_mask = ~np.isnan(velocities[0]) if gene_mask is None else gene_mask
        velocities = velocities[:, nan_mask]
        
        cat_vels = velocities[sel]
        
        cat_score = []
        for ith, idx in enumerate(sel_indices):
            nbs = get_neighbors(idx)
            same_cat_nodes = keep_type(adata, nbs, cat, k_cluster)
            if len(same_cat_nodes) > 0:
                score = cosine_similarity(cat_vels[[ith]], velocities[same_cat_nodes]).mean()
                cat_score.append(score)
        
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])

def main(data_dir, save_dir, celltype_key, umap_key, vkey):
    adata = sc.read_h5ad(data_dir)
    scv.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity_graph(adata)
    adata.obsm["velocity"] = adata.layers[vkey]
    iccoh = inner_cluster_coh(adata, k_cluster=celltype_key, k_velocity=vkey, gene_mask=None, return_raw=True)
    df_iccoh = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in iccoh.items()]))
    os.makedirs(save_dir, exist_ok=True)
    df_iccoh.to_csv(f'{save_dir}/iccoh.tsv', sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data path select.")
    parser.add_argument("--save_dir", default='/example/result/SDEvelo/...')
    parser.add_argument("--data_dir", default='/example/real-data/...')
    parser.add_argument("--celltype_key", default='cell_type')
    parser.add_argument("--umap_key", default='umap')
    parser.add_argument("--vkey", default='velocity')
    args = parser.parse_args()
    main(args.data_dir, args.save_dir, args.celltype_key, args.umap_key, args.vkey)