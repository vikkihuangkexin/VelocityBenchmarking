import Simulate_generate.cell2fate.cell2fate as c2f
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import os
import scvelo as scv
import argparse

scv.settings.verbosity = 0

def main(data_dir, save_dir, max_epochs, batch_size, cells_per_cluster, simulate=False):
    adata = sc.read_h5ad(data_dir)
    data_file = os.path.basename(data_dir)
    file_id = os.path.splitext(data_file)[0]

    clusters_to_remove = []
    cluster_column = 'milestone' if simulate else 'cell_type'
    min_shared_counts = None if simulate else 20
    n_var_genes = adata.n_vars if simulate else 2000
    adata = c2f.utils.get_training_data(
        adata,
        cells_per_cluster=cells_per_cluster,
        cluster_column=cluster_column,
        remove_clusters=clusters_to_remove,
        min_shared_counts=min_shared_counts,
        n_var_genes=n_var_genes
    )

    c2f.Cell2fate_DynamicalModel.setup_anndata(
        adata,
        spliced_label='spliced',
        unspliced_label='unspliced'
    )

    n_modules = c2f.utils.get_max_modules(adata)
    mod = c2f.Cell2fate_DynamicalModel(adata, n_modules=n_modules)

    mod.train(
        max_epochs=max_epochs,
        batch_size=batch_size
    )

    adata = mod.export_posterior(adata)

    velocity_plot_path = os.path.join(
        save_dir,
        f"{file_id}_total_velocity_plots.png"
    )
    mod.compute_and_plot_total_velocity(
        adata,
        plot=False,
        save=velocity_plot_path,
        delete=False
    )

    adata.layers["Velocity"] = adata.layers["Velocity"].cpu().numpy()
    if 'X_dimred' in adata.obsm:
        adata.obsm['X_umap'] = adata.obsm['X_dimred']
    output_h5ad_path = os.path.join(
        save_dir,
        f"{file_id}_cell2fate.h5ad"
    )
    adata.write(output_h5ad_path)

    print(f"Analysis completed! Results saved to: {output_h5ad_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cell2fate Single-cell RNA Velocity Analysis Script"
    )

    parser.add_argument("--data_dir", required=True,
                        help="Input h5ad data file path")
    parser.add_argument("--save_dir", default="./",
                        help="Result saving directory")

    parser.add_argument("--max_epochs", type=int, default=1000,
                        help="Training epochs (1000–3000 recommended)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size (GPU: 512, CPU: 256)")
    parser.add_argument("--cells_per_cluster", type=int, default=100000,
                        help="Number of cells per cluster for training data preparation")
    parser.add_argument("--simulate", action='store_true',
                        help="Whether the data is simulation data")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        cells_per_cluster=args.cells_per_cluster,
        simulate=args.simulate
    )
