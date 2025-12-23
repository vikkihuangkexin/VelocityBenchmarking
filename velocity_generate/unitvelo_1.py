import scvelo as scv
scv.settings.verbosity = 0
import unitvelo as utv
import pandas as pd
import tensorflow as tf
import os
import scanpy as sc
from unit import find_cluster_column
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
def main(data_dir, save_dir):
    adata = sc.read(data_dir, cache=True)
    data_file = data_dir.split('/')[-1]
    ID = data_file.split('.')[0]
    adata.obs_names_make_unique()
    label = find_cluster_column(adata)
    adata.var.index = adata.var.index.str.replace('ENSMU', 'ensmu', case=False)
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    obsm_key = list(adata.obsm.keys())
    if not any(item.lower().find('x_umap') != -1 for item in obsm_key):
        sc.tl.umap(adata)
    adata.write_h5ad(os.path.join(save_dir,f'{ID}.h5ad'))
    velo_config = utv.config.Configuration()
    velo_config.R2_ADJUST = True
    velo_config.IROOT = None
    velo_config.FIT_OPTION = '2'
    velo_config.GPU = 0
    nor = True
    adata = utv.run_model(os.path.join(save_dir,f'{ID}.h5ad'), label, config_file=velo_config, normalize=False)
    scv.tl.velocity_graph(adata)
    if not os.path.exists(f'{save_dir}/{ID}'):
        os.makedirs(f'{save_dir}/{ID}')
    scv.pl.velocity_embedding_stream(adata, basis='umap', color=label, save=f'{save_dir}/{ID}/stream_arrow.pdf')
    scv.pl.velocity_embedding_grid(adata, basis='umap', color=label, save=f'{save_dir}/{ID}/grid_arrow.pdf')
    scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120, save=f'{save_dir}/{ID}/full_arrow.pdf')
    adata.write_h5ad(f'{save_dir}/{ID}/{data_file.split(".")[0]}_velo.h5ad')
if __name__ == '__main__':
    save_dir = '.../example/result/unitvelo/...'
    data_dir = '.../example/data/...'
    main(data_dir, save_dir)