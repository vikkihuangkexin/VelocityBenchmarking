import scanpy as sc
import scvelo as scv
import os
import torch
import argparse
import topovelo as tpv


# Default color palette for visualization
PALETTE_30_3 = [
    "#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4",
    "#66c2a5", "#3288bd", "#abdda4", "#e6f598", "#fee08b",
    "#f46d43", "#e7298a", "#a6cee3", "#1f78b4", "#b2df8a",
    "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
    "#cab2d6", "#6a3d9a", "#ffff99", "#b15928", "#8dd3c7",
    "#bc80bd", "#ccebc5", "#ffed6f", "#999999"
]


def main(data_dir, save_dir, celltype_key='cell_type',
         spatial_key='X_spatial',
         prep_n_gene=2000,
         prep_min_count_per_cell=1,
         prep_min_genes_expressed=1,
         prep_compute_umap=False,
         graph_method='KNN',
         graph_radius=30,
         vae_tmax=20,
         vae_dim_z=5,
         vae_hidden_size=(50, 25, 50),
         vae_device='cuda:0',
         vae_graph_decoder=True,
         vae_attention=True,
         lr=2e-4,
         lr_ode=5e-3,
         lr_refine=2e-4,
         post_n_spatial_neighbors=50,
         post_spatial_velocity_graph=True,
         post_compute_metrics=False):
    """Run TopoVelo velocity analysis on a spatial transcriptomics AnnData object.

    Parameters
    ----------
    data_dir : str
        Path to input .h5ad file.
    save_dir : str
        Directory to save all outputs (model, figures, processed AnnData).
    celltype_key : str
        Column name in adata.obs for cell type / cluster labels.
    spatial_key : str
        Key in adata.obsm storing spatial coordinates.
    """

    # ---- Plotting constants ----
    PLOT_FORMATS = ['png', 'pdf']
    PLOT_DPI = 400
    STREAM_DENSITY = 2
    GRID_DENSITY = 0.8

    adata = sc.read_h5ad(data_dir)

    # Ensure spatial coordinates are stored under the expected key
    if spatial_key not in adata.obsm:
        # Try common alternative keys
        for alt_key in ['spatial', 'X_spatial', 'spatial_coords']:
            if alt_key in adata.obsm:
                adata.obsm[spatial_key] = adata.obsm[alt_key]
                break
        else:
            raise KeyError(
                f"spatial_key '{spatial_key}' not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
    else:
        adata.obsm[spatial_key] = adata.obsm[spatial_key]

    os.makedirs(save_dir, exist_ok=True)
    figure_path = os.path.join(save_dir, 'figures')
    model_path = os.path.join(save_dir, 'model')
    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # ---- Preprocessing ----
    tpv.preprocess(
        adata,
        n_gene=prep_n_gene,
        spatial_key=spatial_key,
        min_count_per_cell=prep_min_count_per_cell,
        min_genes_expressed=prep_min_genes_expressed,
        compute_umap=prep_compute_umap
    )

    # ---- Build spatial graph ----
    tpv.build_spatial_graph(
        adata,
        spatial_key=spatial_key,
        graph_key='spatial_graph',
        method=graph_method,
        radius=graph_radius
    )

    # ---- Create and train VAE ----
    vae = tpv.VAE(
        adata,
        tmax=vae_tmax,
        dim_z=vae_dim_z,
        hidden_size=vae_hidden_size,
        device=vae_device,
        graph_decoder=vae_graph_decoder,
        attention=vae_attention
    )

    config = {
        'learning_rate': lr,
        'learning_rate_ode': lr_ode,
        'learning_rate_refine': lr_refine
    }
    vae.train(
        adata,
        adata.obsp['spatial_graph'],
        spatial_key,
        config=config,
        figure_path=figure_path
    )

    # ---- Post-analysis ----
    genes = tpv.sample_genes(adata, 4, 'highly_variable')
    width = 4
    figsize = tpv.compute_figsize(
        adata.obsm[spatial_key], real_aspect_ratio=True, width=width
    )

    cluster_plot_config = {
        'figsize': figsize,
        'real_aspect_ratio': True
    }
    phase_plot_config = {'width': 4}
    gene_plot_config = {'width': 4}
    time_plot_config = {
        'width': 4,
        'color_map': 'viridis',
    }
    stream_plot_config = {
        'width': figsize[0],
        'height': figsize[1],
        'markersize': 100,
        'density': STREAM_DENSITY,
        'linewidth': 2,
    }

    # Save model and processed AnnData
    vae.save_model(model_path, 'encoder', 'decoder')
    vae.save_anndata(adata, 'gat', save_dir, file_name='adata_out.h5ad')

    tpv.post_analysis(
        adata,
        test_id='topovelo',
        methods=['TopoVelo (GAT)'],
        keys=['gat'],
        spatial_graph_key='spatial_graph',
        spatial_velocity_graph=post_spatial_velocity_graph,
        n_spatial_neighbors=post_n_spatial_neighbors,
        spatial_key=spatial_key,
        compute_metrics=post_compute_metrics,
        embed='spatial',
        cluster_plot_config=cluster_plot_config,
        phase_plot_config=phase_plot_config,
        gene_plot_config=gene_plot_config,
        time_plot_config=time_plot_config,
        stream_plot_config=stream_plot_config,
        cluster_key=celltype_key,
        plot_type=[celltype_key]
    )

    # Compute velocity graph and pseudotime
    scv.tl.velocity_graph(adata, vkey='gat_velocity')
    scv.tl.velocity_pseudotime(adata, vkey='gat_velocity')

    # ---- Plotting ----
    dataset_id = os.path.splitext(os.path.basename(data_dir))[0]

    for fmt in PLOT_FORMATS:
        scv.pl.velocity_embedding_stream(
            adata,
            basis='spatial',
            size=100,
            alpha=0.6,
            color=celltype_key,
            legend_fontsize=9,
            legend_loc='right margin',
            fontsize=None,
            dpi=PLOT_DPI,
            arrow_size=0.00001,
            linewidth=0,
            title='TopoVelo',
            palette=PALETTE_30_3,
            vkey='gat_velocity',
            save=os.path.join(save_dir, f'TopoVelo_{dataset_id}_umap.{fmt}')
        )

    for fmt in PLOT_FORMATS:
        scv.pl.velocity_embedding_stream(
            adata,
            basis='spatial',
            size=100,
            alpha=0.6,
            color=celltype_key,
            legend_fontsize=9,
            legend_loc='right margin',
            fontsize=None,
            density=STREAM_DENSITY,
            dpi=PLOT_DPI,
            arrow_size=1,
            linewidth=1,
            palette=PALETTE_30_3,
            vkey='gat_velocity',
            title='TopoVelo',
            save=os.path.join(save_dir, f'TopoVelo_{dataset_id}_stream.{fmt}')
        )

    for fmt in PLOT_FORMATS:
        scv.pl.velocity_embedding_grid(
            adata,
            basis='spatial',
            size=100,
            alpha=0.6,
            color=celltype_key,
            legend_fontsize=9,
            legend_loc='right margin',
            fontsize=None,
            density=GRID_DENSITY,
            dpi=PLOT_DPI,
            arrow_size=1,
            linewidth=0.3,
            palette=PALETTE_30_3,
            vkey='gat_velocity',
            title='TopoVelo',
            save=os.path.join(save_dir, f'TopoVelo_{dataset_id}_grid.{fmt}')
        )

    for fmt in PLOT_FORMATS:
        scv.pl.scatter(
            adata,
            basis='spatial',
            color='gat_velocity_pseudotime',
            cmap='gnuplot',
            size=100,
            dpi=PLOT_DPI,
            figsize=(8, 6),
            title='TopoVelo',
            save=os.path.join(save_dir, f'TopoVelo_{dataset_id}_pseudotime.{fmt}')
        )

    print(f"TopoVelo analysis complete. Outputs saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TopoVelo velocity analysis for spatial transcriptomics'
    )
    parser.add_argument('--data_dir', required=True,
                        help='Path to input .h5ad file')
    parser.add_argument('--save_dir', required=True,
                        help='Directory to save all outputs')
    parser.add_argument('--celltype_key', default='cell_type',
                        help='Column in adata.obs for cell type labels (default: cell_type)')

    # Spatial key
    parser.add_argument('--spatial_key', default='X_spatial',
                        help='Key in adata.obsm for spatial coordinates (default: X_spatial)')

    # Preprocessing parameters
    parser.add_argument('--prep_n_gene', type=int, default=2000,
                        help='Number of highly variable genes to select (default: 2000)')
    parser.add_argument('--prep_min_count_per_cell', type=int, default=1,
                        help='Minimum counts per cell for filtering (default: 1)')
    parser.add_argument('--prep_min_genes_expressed', type=int, default=1,
                        help='Minimum genes expressed per cell for filtering (default: 1)')
    parser.add_argument('--prep_compute_umap', action='store_true',
                        help='Compute UMAP during preprocessing')

    # Spatial graph parameters
    parser.add_argument('--graph_method', default='KNN',
                        choices=['KNN', 'radius'],
                        help='Method to build spatial graph (default: KNN)')
    parser.add_argument('--graph_radius', type=int, default=30,
                        help='Radius for spatial graph when method is radius-based (default: 30)')

    # VAE model parameters
    parser.add_argument('--vae_tmax', type=int, default=20,
                        help='Maximum time for ODE integration (default: 20)')
    parser.add_argument('--vae_dim_z', type=int, default=5,
                        help='Dimension of latent space (default: 5)')
    parser.add_argument('--vae_hidden_size', type=int, nargs='+', default=[50, 25, 50],
                        help='Hidden layer sizes for VAE (default: 50 25 50)')
    parser.add_argument('--vae_device', default='cuda:0',
                        help='Device for training, e.g. cuda:0 or cpu (default: cuda:0)')
    parser.add_argument('--vae_graph_decoder', type=lambda x: str(x).lower() == 'true',
                        default=True,
                        help='Use graph decoder in VAE (default: True)')
    parser.add_argument('--vae_attention', type=lambda x: str(x).lower() == 'true',
                        default=True,
                        help='Use attention mechanism in VAE (default: True)')

    # Learning rate parameters
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate for VAE training (default: 2e-4)')
    parser.add_argument('--lr_ode', type=float, default=5e-3,
                        help='Learning rate for ODE solver (default: 5e-3)')
    parser.add_argument('--lr_refine', type=float, default=2e-4,
                        help='Learning rate for refinement phase (default: 2e-4)')

    # Post-analysis parameters
    parser.add_argument('--post_n_spatial_neighbors', type=int, default=50,
                        help='Number of spatial neighbors for velocity graph (default: 50)')
    parser.add_argument('--post_spatial_velocity_graph',
                        type=lambda x: str(x).lower() == 'true', default=True,
                        help='Compute spatial velocity graph (default: True)')
    parser.add_argument('--post_compute_metrics',
                        type=lambda x: str(x).lower() == 'true', default=False,
                        help='Compute evaluation metrics (default: False)')

    args = parser.parse_args()

    # Convert hidden_size list to tuple for VAE constructor
    vae_hidden_size = tuple(args.vae_hidden_size)

    main(
        args.data_dir,
        args.save_dir,
        celltype_key=args.celltype_key,
        spatial_key=args.spatial_key,
        prep_n_gene=args.prep_n_gene,
        prep_min_count_per_cell=args.prep_min_count_per_cell,
        prep_min_genes_expressed=args.prep_min_genes_expressed,
        prep_compute_umap=args.prep_compute_umap,
        graph_method=args.graph_method,
        graph_radius=args.graph_radius,
        vae_tmax=args.vae_tmax,
        vae_dim_z=args.vae_dim_z,
        vae_hidden_size=vae_hidden_size,
        vae_device=args.vae_device,
        vae_graph_decoder=args.vae_graph_decoder,
        vae_attention=args.vae_attention,
        lr=args.lr,
        lr_ode=args.lr_ode,
        lr_refine=args.lr_refine,
        post_n_spatial_neighbors=args.post_n_spatial_neighbors,
        post_spatial_velocity_graph=args.post_spatial_velocity_graph,
        post_compute_metrics=args.post_compute_metrics,
    )