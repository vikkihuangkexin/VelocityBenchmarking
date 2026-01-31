import stt as st
import scanpy as sc
import scvelo as scv
import numpy as np
import os
import argparse


def _choose_top_genes(n_vars):
    if n_vars < 10000:
        top_gene = (n_vars // 500) * 500
        top_gene = min(top_gene, n_vars, 500)
    else:
        top_gene = 2000
    return top_gene


def main(data_dir, save_dir, celltype_key='cell_type', simulate=False,
         moments_n_neighbors=50,
         dyn_n_states=None,
         dyn_n_iter=15,
         dyn_weight_connectivities=0.5,
         dyn_n_components=21,
         dyn_n_neighbors=20,
         dyn_thresh_ms_gene=0.2,
         dyn_use_spatial=True,
         dyn_spa_weight=0.3,
         dyn_thresh_entropy=0.1):
    """Run STT dynamical iteration on an AnnData object.

    Behavior changes when `simulate=True`:
    - Use relaxed filtering: `min_shared_counts=None` and `n_top_genes=top_gene`.
    - Set cell grouping to `'milestone'` and if `X_dimred` present copy to `X_umap`.
    """

    adata = sc.read_h5ad(data_dir)

    # Preprocessing: filter & normalize depending on data type
    top_gene = _choose_top_genes(adata.n_vars)
    if simulate:
        scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=top_gene)
    else:
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)

    # compute moments (uses n_neighbors parameter)
    scv.pp.moments(adata, n_neighbors=moments_n_neighbors)

    # set grouping / attractor
    if simulate:
        adata.obs['attractor'] = 'milestone'
        if 'X_dimred' in adata.obsm and 'X_umap' not in adata.obsm:
            adata.obsm['X_umap'] = adata.obsm['X_dimred']
    else:
        adata.obs['attractor'] = adata.obs.get(celltype_key, 'unknown')

    # determine number of states if not provided
    n_states = dyn_n_states if dyn_n_states is not None else len(np.unique(adata.obs.get(celltype_key, adata.obs['attractor'])))

    # run dynamical iteration
    adata_aggr = st.tl.dynamical_iteration(
        adata,
        n_states=n_states,
        n_iter=dyn_n_iter,
        return_aggr_obj=True,
        weight_connectivities=dyn_weight_connectivities,
        n_components=dyn_n_components,
        n_neighbors=dyn_n_neighbors,
        thresh_ms_gene=dyn_thresh_ms_gene,
        use_spatial=dyn_use_spatial,
        spa_weight=dyn_spa_weight,
        thresh_entropy=dyn_thresh_entropy,
    )

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'adata_aggr.h5ad')
    adata_aggr.write_h5ad(out_path)
    print(f"Saved aggregated AnnData to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="STT dynamical aggregation runner")
    parser.add_argument('--data_dir', required=True, help='Path to input .h5ad file')
    parser.add_argument('--save_dir', required=True, help='Directory to save outputs')
    parser.add_argument('--celltype_key', default='cell_type', help='Obs column for cell types')
    parser.add_argument('--simulate', action='store_true', help='Treat input as simulation (adjust preprocessing)')

    # moments parameter
    parser.add_argument('--moments_n_neighbors', type=int, default=50, help='n_neighbors for scv.pp.moments')

    # dynamical_iteration parameters (defaults set to previous hardcoded values)
    parser.add_argument('--dyn_n_states', type=int, default=None, help='Number of dynamical states (auto if not set)')
    parser.add_argument('--dyn_n_iter', type=int, default=15, help='Number of iterations for dynamical_iteration')
    parser.add_argument('--dyn_weight_connectivities', type=float, default=0.5, help='weight_connectivities for dynamical_iteration')
    parser.add_argument('--dyn_n_components', type=int, default=21, help='n_components for dynamical_iteration')
    parser.add_argument('--dyn_n_neighbors', type=int, default=20, help='n_neighbors for dynamical_iteration')
    parser.add_argument('--dyn_thresh_ms_gene', type=float, default=0.2, help='thresh_ms_gene for dynamical_iteration')
    parser.add_argument('--dyn_use_spatial', type=lambda x: (str(x).lower() == 'true'), default=True, help='use_spatial flag for dynamical_iteration')
    parser.add_argument('--dyn_spa_weight', type=float, default=0.3, help='spa_weight for dynamical_iteration')
    parser.add_argument('--dyn_thresh_entropy', type=float, default=0.1, help='thresh_entropy for dynamical_iteration')

    args = parser.parse_args()

    main(
        args.data_dir,
        args.save_dir,
        celltype_key=args.celltype_key,
        simulate=args.simulate,
        moments_n_neighbors=args.moments_n_neighbors,
        dyn_n_states=args.dyn_n_states,
        dyn_n_iter=args.dyn_n_iter,
        dyn_weight_connectivities=args.dyn_weight_connectivities,
        dyn_n_components=args.dyn_n_components,
        dyn_n_neighbors=args.dyn_n_neighbors,
        dyn_thresh_ms_gene=args.dyn_thresh_ms_gene,
        dyn_use_spatial=args.dyn_use_spatial,
        dyn_spa_weight=args.dyn_spa_weight,
        dyn_thresh_entropy=args.dyn_thresh_entropy,
    )