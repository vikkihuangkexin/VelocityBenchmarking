import sdevelo as sv
import scanpy as sc
import os
import argparse


def main(data_path, save_dir, celltype_key, gpu, n_epochs, simulate):
    # Configure device visibility
    if gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device_msg = f"cuda:{gpu}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device_msg = "cpu"

    print(f"[INFO] Using device: {device_msg}")

    # Load data
    adata = sc.read_h5ad(data_path)
    print(f"[INFO] Read in {data_path}")

    # If running on simulated data, relax preprocessing and adjust metadata
    if simulate:
        print("[INFO] Simulation mode: relaxing preprocessing and setting cluster labels to 'milestone'")
        # Mark all cells as the same milestone/cluster
        adata.obs[celltype_key] = "milestone"

        # If a low-dimensional embedding was stored under 'X_dimred', copy to 'X_umap'
        if "X_dimred" in adata.obsm and "X_umap" not in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_dimred"]

        # Indicate relaxed preprocessing in adata.uns so downstream code can detect it
        adata.uns["min_shared_counts"] = None
        if "top_genes" in adata.uns:
            adata.uns["n_top_genes"] = adata.uns.get("top_genes")
        else:
            adata.uns["n_top_genes"] = None

    # Build configuration for SDEvelo
    cfg = sv.Config()
    cfg.cuda_device = gpu
    cfg.vis_type_col = celltype_key
    # Allow overriding number of epochs from CLI
    try:
        cfg.nEpochs = int(n_epochs)
    except Exception:
        pass

    model = sv.SDENN(cfg, adata)

    print(f"[INFO] Start training for {data_path}")
    adata = model.train(cfg.nEpochs)
    print(f"[INFO] Finish training {data_path}")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "adata.h5ad")
    adata.write_h5ad(out_path)
    print(f"[DONE] Saved to {out_path}")


if __name__ == "__main__":
    # Instantiate a default config to discover library defaults for help text
    default_cfg = sv.Config()

    parser = argparse.ArgumentParser(description="SDEvelo training")
    parser.add_argument("--data_path", required=True, help="Path to input h5ad file")
    parser.add_argument("--save_dir", required=True, help="Directory to save results")
    parser.add_argument("--celltype_key", default="cell_type", help="obs key for cell type (default: cell_type)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use; set -1 for CPU")
    parser.add_argument("--n_epochs", type=int, default=getattr(default_cfg, "nEpochs", 200), help="Number of training epochs (default from sdevelo.Config)")
    parser.add_argument("--simulate", action="store_true", help="If set, relax preprocessing (min_shared_counts=None) and set cluster labels to 'milestone'")

    args = parser.parse_args()

    main(args.data_path, args.save_dir, args.celltype_key, args.gpu, args.n_epochs, args.simulate)
