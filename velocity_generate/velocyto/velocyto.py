import sys
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import loompy
import velocyto as vcy
import logging
import scvelo as scv
import scanpy as sc
import argparse
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

def main(data_dir, data_file, save_dir, simulate=True):
    adata = sc.read(data_dir, cache=True)
    data_file = os.path.basename(data_file)
    ID = data_file.split('.')[0]
    adata.obs_names_make_unique()
    if simulate:
        cluster = 'milestone'
    else:
        cluster = find_cluster_column(adata)
    obsm_key = list(adata.obsm.keys())
    # choose target loom path based on simulate flag
    if simulate:
        loom_path = fr'/data_d/Velocity/D/simdata/loom/{ID}.loom'
    else:
        loom_path = fr'/data_d/Velocity/data/loom/{ID}.loom'

    if not os.path.exists(loom_path):
        if simulate:
            # simulation: map X_dimred to X_umap and write simulation loom
            if 'X_dimred' in adata.obsm and 'X_umap' not in adata.obsm:
                adata.obsm['X_umap'] = adata.obsm['X_dimred']
            adata.write_loom(loom_path, write_obsm_varm=True)
        else:
            if any(item.lower().find('x_umap') != -1 for item in obsm_key):
                print(f"Data: {data_file} seems to have been processed using UMAP.")          
                adata = adata[~adata.to_df().duplicated(), :]
                adata.obs_names_make_unique()
                adata.write_loom(loom_path, write_obsm_varm=True)
            else:
                backup_csv = fr'/data_d/Velocity/Umap_backup/{ID}_addUmap.csv'
                if os.path.exists(backup_csv):
                    loaded_umap = pd.read_csv(backup_csv, index_col=0)
                    adata.obsm['X_umap'] = loaded_umap[['UMAP1', 'UMAP2']].values
                    if data_file.startswith('48'):
                        adata = adata[~adata.to_df().duplicated(), :]
                        adata.obs_names_make_unique()
                    adata.write_loom(loom_path, write_obsm_varm=True)
                else:
                    if data_file.startswith('48'):
                        adata = adata[~adata.to_df().duplicated(), :]
                        adata.obs_names_make_unique()
                    adata1 = adata.copy()
                    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
                    sc.pp.pca(adata)
                    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30, method='umap')
                    scv.tl.umap(adata)
                    adata1.obsm['X_pca'] = adata.obsm['X_pca']
                    adata1.obsm['X_umap'] = adata.obsm['X_umap']
                    adata1.write_loom(loom_path, write_obsm_varm=True)
    del adata
    # Create an analysis object from the generated loom file
    if any(item.lower().find('ambiguous') != -1 for item in obsm_key):
        vlm = vcy.VelocytoLoom(loom_path)
    else:
        vlm = VelocytoLoom(loom_path)

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
    n_comps = np.where(np.diff(np.diff(np.cumsum(vlm.pca.explained_variance_ratio_))>0.002))[0][0]
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
    parser = argparse.ArgumentParser(description='Run velocyto processing for one dataset')
    parser.add_argument('--data_dir', required=True, help='Path to input .h5ad file')
    parser.add_argument('--data_file', required=True, help='Input filename used for outputs')
    parser.add_argument('--save_dir', required=True, help='Directory to save results')
    parser.add_argument('--simulate', dest='simulate', action='store_true', help='Treat input as simulation (default)')
    parser.add_argument('--no-simulate', dest='simulate', action='store_false', help='Treat input as real data')
    parser.set_defaults(simulate=True)
    args = parser.parse_args()

    out_dir = os.path.join(args.save_dir, args.data_file.split('.')[0])
    os.makedirs(out_dir, exist_ok=True)
    main(args.data_dir, args.data_file, out_dir, simulate=args.simulate)