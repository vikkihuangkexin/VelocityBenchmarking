import sys
import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import loompy
import velocyto as vcy
import logging
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
import scvelo as scv
import scanpy as sc
# vcy.read()
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
# %matplotlib inline
plt.rcParams['pdf.fonttype'] = 42
from analysis_1 import VelocytoLoom
from unit import find_cluster_column, vlm_to_adata
import scipy.sparse as sp

# plotting utility functions
def despline():
    ax1 = plt.gca()
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
def minimal_xticks(start, end):
    end_ = np.around(end, -int(np.log10(end)) + 1)
    xlims = np.linspace(start, end_, 5)
    xlims_tx = [""] * len(xlims)
    xlims_tx[0], xlims_tx[-1] = f"{xlims[0]:.0f}", f"{xlims[-1]:.02f}"
    plt.xticks(xlims, xlims_tx)
def minimal_yticks(start, end):
    end_ = np.around(end, -int(np.log10(end)) + 1)
    ylims = np.linspace(start, end_, 5)
    ylims_tx = [""] * len(ylims)
    ylims_tx[0], ylims_tx[-1] = f"{ylims[0]:.0f}", f"{ylims[-1]:.02f}"
    plt.yticks(ylims, ylims_tx)

def main(data_dir,data_file,save_dir):
    adata= sc.read(data_dir, cache=True)
    cluster = find_cluster_column(adata)
    if ID == '55-new':
        cluster = 'cell_type'
    obsm_key = list(adata.obsm.keys())
    # layer_list = list(adata.layers.keys())
    # for layer_name in layer_list:
    #     # 如果是稀疏矩阵，转换为密集矩阵以检查所有值
    #     layer_matrix = adata.layers[layer_name]
    #     if sp.issparse(layer_matrix):
    #         layer_dense = layer_matrix.toarray()
    #     else:
    #         layer_dense = layer_matrix
    #     # 检测 INF 和 NaN
    #     invalid_mask = np.isnan(layer_dense)
    #     if invalid_mask.any():
    #         print(layer_name)
    #         del adata.layers[layer_name]
    if not os.path.exists(fr'/data_d/Velocity/data/loom/{data_file.split(".")[0]}.loom'):
        if any(item.lower().find('x_umap') != -1 for item in obsm_key):
            print(f"Data: {data_file} seems to have been processed using UMAP.")
            if data_file.startswith('48'):
                adata = adata[~adata.to_df().duplicated(), :]
                adata.obs_names_make_unique()
            adata.write_loom(fr'/data_d/Velocity/data/loom/{data_file.split(".")[0]}.loom', write_obsm_varm=True)
        else:
            if os.path.exists(fr'/data_d/Velocity/Umap_backup/{data_file.split(".")[0]}_addUmap.csv'):
                loaded_umap = pd.read_csv(f'/data_d/Velocity/Umap_backup/{data_file.split(".")[0]}_addUmap.csv',index_col=0)
                adata.obsm['X_umap'] = loaded_umap[['UMAP1', 'UMAP2']].values
                if data_file.startswith('48'):
                    adata = adata[~adata.to_df().duplicated(), :]
                    adata.obs_names_make_unique()
                adata.write_loom(fr'/data_d/Velocity/data/loom/{data_file.split(".")[0]}.loom', write_obsm_varm=True)
            else:
                if data_file.startswith('48'):
                    adata = adata[~adata.to_df().duplicated(), :]
                    adata.obs_names_make_unique()
                adata1 = adata.copy()
                scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
                # scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
                sc.pp.pca(adata)
                sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30, method='umap')
                scv.tl.umap(adata)
                adata1.obsm['X_pca'] = adata.obsm['X_pca']
                adata1.obsm['X_umap'] = adata.obsm['X_umap']
                adata1.write_loom(fr'/data_d/Velocity/data/loom/{data_file.split(".")[0]}.loom', write_obsm_varm=True)
    del adata
    # Crate an analysis object
    if any(item.lower().find('ambiguous') != -1 for item in obsm_key):
        vlm = vcy.VelocytoLoom(fr'/data_d/Velocity/data/loom/{data_file.split(".")[0]}.loom')
    else:
        vlm = VelocytoLoom(fr'/data_d/Velocity/data/loom/{data_file.split(".")[0]}.loom')

    # Read column attributes form the loom file and specify colors
    # vlm.ts = np.column_stack([vlm.ca["TSNE1"], vlm.ca["TSNE2"]])
    # colors_dict = {'RadialGlia': np.array([ 0.95,  0.6,  0.1]), 'RadialGlia2': np.array([ 0.85,  0.3,  0.1]), 'ImmAstro': np.array([ 0.8,  0.02,  0.1]),
    #               'GlialProg': np.array([ 0.81,  0.43,  0.72352941]), 'OPC': np.array([ 0.61,  0.13,  0.72352941]), 'nIPC': np.array([ 0.9,  0.8 ,  0.3]),
    #               'Nbl1': np.array([ 0.7,  0.82 ,  0.6]), 'Nbl2': np.array([ 0.448,  0.85490196,  0.95098039]),  'ImmGranule1': np.array([ 0.35,  0.4,  0.82]),
    #               'ImmGranule2': np.array([ 0.23,  0.3,  0.7]), 'Granule': np.array([ 0.05,  0.11,  0.51]), 'CA': np.array([ 0.2,  0.53,  0.71]),
    #                'CA1-Sub': np.array([ 0.1,  0.45,  0.3]), 'CA2-3-4': np.array([ 0.3,  0.35,  0.5])}\
    # vlm.plot_fractions()
    vlm.set_clusters(vlm.ca[cluster])

    vlm.filter_cells(bool_array=vlm.initial_Ucell_size > np.percentile(vlm.initial_Ucell_size, 0.5))

    vlm.score_detection_levels(min_expr_counts=40, min_cells_express=30)
    vlm.filter_genes(by_detection_levels=True)
    if vlm.ra['Gene'].shape[0]<3000:
        top_gene = (vlm.ra['Gene'].shape[0]//500)*500
        top_gene = min(top_gene,vlm.ra['Gene'].shape[0])
    else:
        top_gene=3000
    vlm.score_cv_vs_mean(top_gene, plot=True, max_expr_avg=35)
    vlm.filter_genes(by_cv_vs_mean=True)
    vlm._normalize_S(relative_size=vlm.initial_cell_size,
                     target_size=np.mean(vlm.initial_cell_size))
    vlm._normalize_U(relative_size=vlm.initial_Ucell_size,
                     target_size=np.mean(vlm.initial_Ucell_size))

    vlm.perform_PCA()
    # plt.plot(np.cumsum(vlm.pca.explained_variance_ratio_)[:100])
    n_comps = np.where(np.diff(np.diff(np.cumsum(vlm.pca.explained_variance_ratio_))>0.002))[0][0]
    # plt.axvline(n_comps, c="k")
    n_comps
    k = 500
    if vlm.ca['X_umap'].shape[0]<k*8:
        k = 4
        vlm.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8, b_maxl=k*4, n_jobs=12)
    else:
        vlm.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k * 8, b_maxl=k * 4, n_jobs=12)
    vlm.fit_gammas(limit_gamma=False, fit_offset=False)
    vlm.predict_U()
    vlm.calculate_velocity()
    vlm.calculate_shift(assumption="constant_velocity")
    vlm.extrapolate_cell_at_t(delta_t=1.)

    vlm.umap = vlm.ca['X_umap']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if vlm.ca['X_umap'].shape[0]<=20000:
        vlm.estimate_transition_prob(hidim="Sx_sz", embed="umap", transform="sqrt", psc=1,
                                     n_neighbors=min(2000,len(vlm.cell_size)//10), knn_random=True, sampled_fraction=0.5)
        vlm.calculate_embedding_shift(sigma_corr = 0.05, expression_scaling=True)
        vlm.calculate_grid_arrows(smooth=0.8, steps=(40, 40), n_neighbors=300)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        from matplotlib.lines import Line2D
        plt.figure()
        vlm.plot_grid_arrows(quiver_scale=0.9,
                             scatter_kwargs_dict={"alpha":0.35, "lw":0.35, "edgecolor":"0.4", "s":38, "rasterized":True}, min_mass=24, angles='xy', scale_units='xy',
                             headaxislength=2.75, headlength=5, headwidth=4.8, minlength=1.5,
                             plot_random=False, scale_type="absolute")
        # plt.tight_layout()
        # plt.show()
        plt.savefig(f"{save_dir}/grid_arrows_umap_legend.pdf")
        plt.close()
    vlm.to_hdf5(f'{save_dir}/{data_file.split("_")[0]}.hdf5')
    data_out = vlm_to_adata(vlm,n_comps =n_comps)
    # data_out.ra["Gene"] = data_out.ra["var_names"]
    data_out.write_h5ad(f'{save_dir}/{data_file.split(".")[0]}_velo.h5ad')
    #vcy.load_velocyto_hdf5()

if __name__ == '__main__':
    datalist = pd.read_csv('/data_d/Velocity/ZY_1/data.csv')
    save_dir = '/data_d/Velocity/D/velocyto'
    retrain=[]
    for i in [53]:
        if i in retrain:
            continue
        ID = datalist.iloc[i]['ID']
        data_file = datalist.iloc[i]['name']
        data_dir = datalist.iloc[i]['path']
        if os.path.exists(f'{save_dir}/{ID}'):
            continue
        else:
            print(f'{i}__{data_file}')
            main(data_dir,data_file,f'{save_dir}/{ID}')