import scvelo as scv
scv.settings.verbosity = 0
import velocity_generate.UnitVelo.unitvelo as utv
import tensorflow as tf
import os
import scanpy as sc
import argparse
from unit import find_cluster_column


def main(
    data_dir,
    save_dir,
    gpu="1",
    normalize=False,
    simulate=True,
):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    adata = sc.read(data_dir, cache=True)
    data_file = os.path.basename(data_dir)
    ID = data_file.split(".")[0]
    adata.obs_names_make_unique()

    # normalize gene names that may be inconsistent
    adata.var.index = adata.var.index.str.replace("ENSMU", "ensmu", case=False)

    if simulate:
        # Simulation-style preprocessing
        cluster = 'milestone'
        if 'X_dimred' in adata.obsm and 'X_umap' not in adata.obsm:
            adata.obsm['X_umap'] = adata.obsm['X_dimred']
        if adata.n_vars < 10000:
            top_gene = (adata.n_vars // 500) * 500
            top_gene = min(top_gene, adata.n_vars, 500)
        else:
            top_gene = 2000
        scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=top_gene)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        label = cluster
    else:
        # Real-data preprocessing
        label = find_cluster_column(adata)
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    obsm_key = list(adata.obsm.keys())
    if not any(item.lower().find("x_umap") != -1 for item in obsm_key):
        sc.tl.umap(adata)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    adata.write_h5ad(os.path.join(save_dir, f"{ID}.h5ad"))

    velo_config = utv.config.Configuration()
    velo_config.R2_ADJUST = True
    velo_config.IROOT = None
    velo_config.FIT_OPTION = "1"
    velo_config.GPU = 0

    adata = utv.run_model(os.path.join(save_dir, f"{ID}.h5ad"), label, config_file=velo_config, normalize=normalize)

    scv.tl.velocity_graph(adata)

    out_dir = os.path.join(save_dir, ID)
    os.makedirs(out_dir, exist_ok=True)

    scv.pl.velocity_embedding_stream(adata, basis="umap", color=label, save=os.path.join(out_dir, "stream_arrow.pdf"))
    scv.pl.velocity_embedding_grid(adata, basis="umap", color=label, save=os.path.join(out_dir, "grid_arrow.pdf"))
    scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120, save=os.path.join(out_dir, "full_arrow.pdf"))
    adata.write_h5ad(os.path.join(out_dir, f"{data_file.split('.')[0]}_velo.h5ad"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UnitVelo on an AnnData .h5ad file")
    parser.add_argument("--data_dir", required=True, help="Path to input .h5ad file")
    parser.add_argument("--save_dir", required=True, help="Directory to write outputs")
    parser.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES string (default: '1')")
    parser.add_argument("--normalize", default=False, action="store_true", help="Pass normalize=True to run_model")
    parser.add_argument('--simulate', dest='simulate', action='store_true', help='Treat input as simulation (default)')
    parser.add_argument('--no-simulate', dest='simulate', action='store_false', help='Treat input as real data')
    parser.set_defaults(simulate=True)
    args = parser.parse_args()

    main(
        args.data_dir,
        args.save_dir,
        gpu=args.gpu,
        normalize=args.normalize,
        simulate=args.simulate,
    )
