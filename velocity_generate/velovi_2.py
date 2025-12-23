import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch
import os
GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"
from velovi import preprocess_data, VELOVI
from unit import find_cluster_column
import matplotlib.pyplot as plt
import seaborn as sns


def main(data_dir, data_file, save_dir):
    adata = sc.read(data_dir, cache=True)
    adata.obs_names_make_unique()
    if data_file.startswith('48'):
        adata = adata[~adata.to_df().duplicated(), :]
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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    scv.pl.velocity_embedding_stream(adata, basis='umap', color=cluster, save=f'{save_dir}/stream_arrow.pdf')
    scv.pl.velocity_embedding_grid(adata, basis='umap', color=cluster, save=f'{save_dir}/grid_arrow.pdf')
    scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120, save=f'{save_dir}/full_arrow.pdf')
    adata.write_h5ad(f'{save_dir}/{data_file.split(".")[0]}_velo.h5ad')



if __name__ == '__main__':
    datalist = pd.read_csv('/data_d/Velocity/ZY_1/data.csv')
    save_dir = '/data_d/Velocity/ZY/velovi'
    outlist = ['55-new']
    for i in [50,51,52]:
        ID = datalist.iloc[i]['ID']
        if ID in outlist:
            continue
        data_file = datalist.iloc[i]['name']
        data_dir = datalist.iloc[i]['path']
        if os.path.exists(f'{save_dir}/{ID}'):
            continue
        else:
            main(data_dir,data_file,f'{save_dir}/{ID}')