import cell2fate as c2f
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import os
import scvelo as scv
import argparse

scv.settings.verbosity = 0

def main(data_dir, save_dir, max_epochs, batch_size):
    adata = sc.read_h5ad(data_dir)
    data_file = os.path.basename(data_dir)
    file_id = os.path.splitext(data_file)[0]

    clusters_to_remove = []
    adata = c2f.utils.get_training_data(
        adata,
        cells_per_cluster=10**5,
        cluster_column='cell_type',
        remove_clusters=clusters_to_remove,
        min_shared_counts=20,
        n_var_genes=2000
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

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size
    )
