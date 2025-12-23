import scvelo as scv
scv.settings.verbosity = 0
import unitvelo as utv
import tensorflow as tf
import os
import scanpy as sc
from unit import find_cluster_column
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
def set_global_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
def main(data_dir, save_dir, r):
    set_global_seed(42+r)
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
        print()
        sc.tl.umap(adata)
    if not os.path.exists(f'{save_dir}/{ID}/{r}'):
        os.makedirs(f'{save_dir}/{ID}/{r}')
    adata.write_h5ad(os.path.join(f'{save_dir}/{ID}/{r}',f'{ID}.h5ad'))
    exp_metrics = {}
    velo_config = utv.config.Configuration()
    velo_config.R2_ADJUST = True
    velo_config.IROOT = None
    velo_config.FIT_OPTION = '2'
    velo_config.GPU = 0
    nor = True
    adata = utv.run_model(os.path.join(f'{save_dir}/{ID}/{r}',f'{ID}.h5ad'), label, config_file=velo_config, normalize=False)
    try:
        scv.tl.velocity_graph(adata)

        scv.pl.velocity_embedding_stream(adata, basis='umap', color=label, save=f'{save_dir}/{ID}/{r}/stream_arrow.pdf')
        scv.pl.velocity_embedding_grid(adata, basis='umap', color=label, save=f'{save_dir}/{ID}/{r}/grid_arrow.pdf')
        scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120, save=f'{save_dir}/{ID}/{r}/full_arrow.pdf')
        adata.write_h5ad(f'{save_dir}/{ID}/{r}/{data_file.split(".")[0]}_velo.h5ad')
    except:

        adata.write_h5ad(f'{save_dir}/{ID}/{r}/{data_file.split(".")[0]}_velo.h5ad')
if __name__ == '__main__':
    save_dir = '.../example/result/unitvelo/...'
    data_dir = '.../example/data/...'
    rep=5
    for r in len(rep):
        print(f"############### 第{r}轮鲁棒性测试############")
        check = main(data_dir, save_dir, r)