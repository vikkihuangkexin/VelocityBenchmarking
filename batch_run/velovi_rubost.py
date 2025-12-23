import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch
import os
from velovi import preprocess_data, VELOVI
from unit import find_cluster_column
import random
GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"
def set_global_seed(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
def main(data_dir, save_dir,r):
    set_global_seed(42+r)
    adata = sc.read(data_dir, cache=True)
    data_file = data_dir.split('/')[-1]
    ID = data_file.split('.')[0]
    obsm_key = list(adata.obsm.keys())
    if not any(item.lower().find('x_umap') != -1 for item in obsm_key):
        sc.tl.umap(adata)
    adata.obs_names_make_unique()
    cluster = find_cluster_column(adata)
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30, method='umap')

    adata = preprocess_data(adata)
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train(batch_size=1024)


    latent_time = vae.get_latent_time(n_samples=25)
    velocities = vae.get_velocity(n_samples=25, velo_statistic="mean")

    t = latent_time
    scaling = 20 / t.max(0)

    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_velovi"] = latent_time

    adata.var["fit_alpha"] = vae.get_rates()["alpha"] / scaling
    adata.var["fit_beta"] = vae.get_rates()["beta"] / scaling
    adata.var["fit_gamma"] = vae.get_rates()["gamma"] / scaling
    adata.var["fit_t_"] = (
                              torch.nn.functional.softplus(vae.module.switch_time_unconstr)
                                  .detach()
                                  .cpu()
                                  .numpy()
                          ) * scaling
    ss=np.array(scaling)
    adata.layers["fit_t"] = latent_time.values * ss[np.newaxis, :]
    adata.var['fit_scaling'] = 1.0

    scv.tl.velocity_graph(adata)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    scv.pl.velocity_embedding_stream(adata, basis='umap', color=cluster, save=f'{save_dir}/stream_arrow.pdf')
    scv.pl.velocity_embedding_grid(adata, basis='umap', color=cluster, save=f'{save_dir}/grid_arrow.pdf')
    scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120, save=f'{save_dir}/full_arrow.pdf')
    adata.write_h5ad(f'{save_dir}/{data_file.split(".")[0]}_velo.h5ad')


if __name__ == '__main__':
    save_dir = '.../example/result/unitvelo/...'
    data_dir = '.../example/data/...'
    rep=5
    for r in len(rep):
        print(f"############### 第{r}轮鲁棒性测试############")
        check = main(data_dir, save_dir, r)