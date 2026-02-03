import os
import pandas as pd
import scanpy as sc
import numpy as np
from anndata import AnnData
import scvelo as scv
from tqdm import tqdm
import sys
import re
import glob
import matplotlib
matplotlib.use('Agg') 
from collections import defaultdict
from scipy.sparse import csr_matrix
from typing import Dict, Iterable, List, Literal, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity


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


def cross_boundary_correctness(
    adata: AnnData,
    k_cluster: str,
    k_velocity: str,
    cluster_edges: List[Tuple[str]],
    return_raw: bool = False,
    x_emb: str = "X_umap",
    gene_mask: Optional[np.ndarray] = None
) -> Tuple[Dict, float]:
    """Cross-Boundary Direction Correctness Score (A->B)
    Args:
        adata (:class:`anndata.AnnData`):
            Anndata object.
        k_cluster (str):
            Key to the cluster column in adata.obs.
        k_velocity (str):
            Key to the velocity matrix in adata.obsm.
        cluster_edges (list[tuple[str]]):
            Pairs of clusters has transition direction A->B
        return_raw (bool, optional):
            Return aggregated or raw scores. Defaults to False.
        x_emb (str, optional):
            Key to x embedding for visualization or a count matrix in adata.layers.
            Defaults to "X_umap".
        gene_mask (:class:`numpy.ndarray`, optional):
            Boolean array to filter out non-velocity genes. Defaults to None.
    Returns:
        tuple:
            - dict: all_scores indexed by cluster_edges or mean scores indexed by cluster_edges
            - float: averaged score over all cells
    """
    scores = {}
    all_scores = {}
    x_emb_name = x_emb
    if x_emb in adata.obsm:
        x_emb = adata.obsm[x_emb]
        if x_emb_name == "X_umap":
            v_emb = adata.obsm['{}_umap'.format(k_velocity)]
        else:
            v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
    else:
        x_emb = adata.layers[x_emb]
        v_emb = adata.layers[k_velocity]
        if gene_mask is None:
            gene_mask = ~np.isnan(v_emb[0])
        x_emb = x_emb[:, gene_mask]
        v_emb = v_emb[:, gene_mask]

    # Get the connectivity matrix
    connectivities = adata.obsp[adata.uns['neighbors']['connectivities_key']]
    
    # Convert to CSR format if it's not already
    if not isinstance(connectivities, csr_matrix):
        connectivities = connectivities.tocsr()

    def get_neighbors(idx):
        return connectivities[idx].indices

    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        sel_indices = np.where(sel)[0]
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]
        type_score = []
        for idx, x_pos, x_vel in zip(sel_indices, x_points, x_velocities):
            nbs = get_neighbors(idx)
            nodes = keep_type(adata, nbs, v, k_cluster)
            if len(nodes) == 0:
                continue
            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.nanmean(dir_scores))
        if len(type_score) == 0:
            print(f'Warning: cell type transition pair ({u},{v}) does not exist in the KNN graph. Ignored.')
        else:
            scores[f'{u} -> {v}'] = np.nanmean(type_score)
            all_scores[f'{u} -> {v}'] = type_score
    if return_raw:
        return all_scores
    return scores, np.mean([sc for sc in scores.values()])

def lineage_to_edges(lineage_info):
    """
    Convert lineage_info to edge list:
    [('A','B'), ('B','C'), ...]
    """
    if isinstance(lineage_info, list) and all(isinstance(x, str) for x in lineage_info):
        paths = [lineage_info] #List[List[str]]
    else:
        paths = lineage_info

    edges = []
    for path in paths:
        edges.extend(
            (path[i], path[i + 1])
            for i in range(len(path) - 1)
        )

    return edges

def main(data_dir, save_dir, celltype_key, umap_key, vkey, lineage_info):
    cluster_edges = lineage_to_edges(lineage_info)
    adata = sc.read_h5ad(data_dir)
    adata.obsm['X_umap'] = adata.obsm[umap_key]
    adata.obsm["velocity"] = adata.layers[vkey]
    adata.obsm["velocity"] = np.nan_to_num(adata.layers[vkey], nan=0.0)
    k_cluster = celltype_key

    if adata.obs[k_cluster].dtype == int:
        adata.obs[k_cluster] = adata.obs[k_cluster].astype('str')
        adata.obs[k_cluster] = adata.obs[k_cluster].astype('category')

    scv.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity_graph(adata,show_progress_bar=False, n_neighbors=30, n_jobs=64)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    # scv.pl.velocity_embedding_stream(adata,basis='umap',show=None, arrow_size=1.5)
    scv.tl.velocity_embedding(adata, basis='umap')
    adata.obsm["velocity_umap"] = np.nan_to_num(adata.obsm["velocity_umap"], nan=0.0)
    result = cross_boundary_correctness(
        adata, k_cluster=k_cluster, k_velocity='velocity', cluster_edges=cluster_edges, return_raw=True, x_emb='X_umap', gene_mask=None
    )
    df_cross = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in result.items()]))
    os.makedirs(save_dir, exist_ok=True)

    df_cross.to_csv(f'{save_dir}/cbdir.txt', index=False, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data path select.")
    parser.add_argument("--save_dir", default='.../example/result/SDEvelo/...')
    parser.add_argument("--data_dir", default='.../example/data/...')
    parser.add_argument("--celltype_key", default='cell_type')
    parser.add_argument("--umap_key", default='umap')
    parser.add_argument("--vkey", default='velocity')
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

    main(args.data_dir, args.save_dir, args.celltype_key, args.umap_key, args.vkey, lineage_info)