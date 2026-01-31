import matplotlib.pyplot as plt
import velocyto as vcy
import protaccel.protaccel as pa
import numpy as np
import os
import argparse

def main(protein_dir, loom_dir, save_dir):
    # Create figure directory
    fig_dir = os.path.join(save_dir, "fig")
    os.makedirs(fig_dir, exist_ok=True)

    # Import protein data
    [prot_count_array, prot_cells, adt_names] = pa.import_prot_data(protein_dir)
    print(adt_names)

    # Load VelocytoLoom
    vlm = vcy.VelocytoLoom(loom_dir)

    # Gene filtering and scoring
    vlm.score_cv_vs_mean(3000, plot=True, max_expr_avg=50, min_expr_cells=50)
    vlm.score_detection_levels(min_expr_counts=40, min_cells_express=30)

    # Gene and protein dictionary setup
    gene_dict = {'CD3':['CD3D','CD3E','CD3G'],
                 'CD4':['CD4'],
                 'CD8a':['CD8A'],
                 'CD14':['CD14'],
                 'CD16':['FCGR3A','FCGR3B'],
                 'CD56':['NCAM1'],
                 'CD19':['CD19'],
                 'CD25':['IL2RA'],
                 'CD45RA':['PTPRC'],
                 'CD45RO':['PTPRC'],
                 'PD-1':['PDCD1'],
                 'TIGIT':['TIGIT'],
                 'CD127':['IL7R']}

    InvertDict = lambda d: dict( (v,k) for k in d for v in d[k] )
    prot_dict = InvertDict(gene_dict)
    mrna_targets = list(prot_dict.keys())
    pa.enforce_protein_filter(vlm, mrna_targets, adt_names)
    vlm.filter_genes(by_cv_vs_mean=True, by_detection_levels=True)
    first_char = 0
    last_char = 0

    # Shared cells filter and imputation
    [prot_count_array, shared_cells, prot_cells] = pa.shared_cells_filter(vlm, prot_cells, prot_count_array, first_char, last_char)
    pa.impute(vlm, prot_count_array, k=50, impute_in_prot_space=True, size_norm=False, impute_in_pca_space=False)

    # Cluster identification
    t_cl = [2,0,3,0,1,5,4,0,4,4,5,5,5]
    [cluster_ID, num_clusters] = pa.identify_clusters(vlm, vlm.connectivity,
                                                      correct_tags=True, tag_correction_list=t_cl,
                                                      method_name='ModularityVertexPartition')

    # Color and label setup
    COLORS=np.asarray([[0, 0.4470, 0.7410],
            [0.8500, 0.3250, 0.0980],
            [0.9290, 0.6940, 0.1250],
            [0.4940, 0.1840, 0.5560],
            [0.4660, 0.6740, 0.1880],[240/255,15/255,223/255]])
    cluster_labels = ['CD4+ T','B','Mono.','NK','CD8+ T','Misc.']
    vlm.COLORS = COLORS
    vlm.labels=cluster_labels

    # PCA fitting and visualization
    pa.fit_pcs(vlm,'P_norm','prot_pcs',n_pcs=4)
    pa.visualize_pcs(vlm, [1,2])
    pa.visualize_pcs(vlm, [2,3])

    # Protein marker visualization
    marker_list = ['CD3','CD4','CD8a','CD14','CD16','CD45RA','CD19','CD25','CD127']
    pa.visualize_protein_markers(vlm, protein_markers=marker_list, pc_targets=[1,2], visualize_clusters= True)
    plt.savefig(os.path.join(fig_dir, "tenX_10k_clus.svg"))
    pa.visualize_phase_portraits(vlm, mrna_targets, target='protein', imputed=True, prot_dict=prot_dict)

    # RNA and protein velocity setup
    genes_used_for_prot_velocity = ['CD3D','CD8A','NCAM1','CD14','IL2RA','IL7R','CD19','TIGIT']
    adt_used_for_prot_velocity = ['CD3','CD8a','CD56','CD14','CD25','CD127','CD19','TIGIT']
    target_size_median = [np.median(vlm.S.sum(0)), np.median(vlm.U.sum(0))]
    vlm.normalize(which="both",size=True,target_size=target_size_median)

    # PCA for RNA
    pa.fit_pcs(vlm,'S_norm','pcs',3)
    pa.visualize_pcs(vlm, [1,2], pc_space='pcs')
    pa.visualize_pcs(vlm, [2,3], pc_space='pcs')

    # Gamma fit and extrapolation
    pa.gamma_fit(vlm,'Sx','Ux','rna')
    pa.extrapolate(vlm,vel_type='rna')
    pa.gamma_fit(vlm,'Px','Sx','protein',genes_used_for_prot_velocity, adt_used_for_prot_velocity)
    pa.extrapolate(vlm,vel_type='protein')

    # RNA phase portrait visualization
    n_rna_velo_fit_viz = 24
    np.random.seed(33)
    print(str(len(vlm.rna_velo_gene_ind))+' spliced/unspliced gene pairs have diagonal phase portraits by R2. Displaying '
          +str(n_rna_velo_fit_viz)+' random phase portraits:')
    genes_for_rna_velo_fit=np.random.choice(vlm.rna_velo_gene_ind,n_rna_velo_fit_viz,replace=False)
    rna_vel_genes=vlm.ra['Gene'][genes_for_rna_velo_fit]
    pa.visualize_phase_portraits(vlm, rna_vel_genes, target='mrna', plot_fit=True)
    plt.savefig(os.path.join(fig_dir, "tenX_10k_rnaphase.svg"))
    pa.visualize_phase_portraits(vlm, mrna_targets, target='protein', imputed=True, prot_dict=prot_dict, plot_fit=True)
    plt.savefig(os.path.join(fig_dir, "tenX_10k_protphase.svg"))

    # Embedding KNN and delta calculation (PCA space)
    pa.identify_embedding_knn(vlm,'pcs',[1,2])
    pa.calculate_embedding_delta(vlm,'Sx','delta_S','delta_S_in_S_pca','rna_velo_gene_ind')
    pa.visualize_velocity_projection(vlm, 'delta_S_in_S_pca')
    pa.calculate_embedding_delta(vlm,'Px','delta_P','delta_P_in_S_pca','prot_velo_prot_ind')
    pa.visualize_velocity_projection(vlm, 'delta_P_in_S_pca')
    pa.cluster_specific_plot(vlm, 'delta_S_in_S_pca',draw_cells=True)
    pa.cluster_specific_plot(vlm, 'delta_P_in_S_pca',draw_cells=True)

    # Grid arrows (PCA space)
    pa.initialize_grid_embedding(vlm)
    uv_multiplier=1
    pa.calculate_grid_arrows(vlm,'delta_S_in_S_pca', '_rna', min_mass=1,uv_multiplier=uv_multiplier)
    pa.calculate_grid_arrows(vlm,'delta_P_in_S_pca', '_prot', min_mass=1,uv_multiplier=uv_multiplier)
    r_rnav = [231/255,36/255,20/255]
    b_protv = [38/255,55/255,213/255]
    plt.figure(figsize=(15,15))
    arr_scale=1
    pa.plot_grid_arrows(vlm,'UV_rna',plot_cells=False, arr_col=r_rnav,arr_scale=arr_scale)
    pa.plot_grid_arrows(vlm,'UV_prot',plot_cells=True, arr_col=b_protv, color_cells_by_cluster=True, pivot='tip',
                        arr_scale=arr_scale,cell_alpha=0.2)
    plt.figure(figsize=(15,15))
    pa.plot_bezier(vlm, plot_cells=True, color_cells_by_cluster=True,cell_alpha=0.2)

    # tSNE fitting and visualization
    pa.fit_pcs(vlm,'S_norm','pcs',25)
    pa.fit_tsne(vlm, 'pcs', 'tsne', 25)
    pa.identify_embedding_knn(vlm,'tsne',[0,1])
    pa.calculate_embedding_delta(vlm,'Sx','delta_S','delta_S_in_S_tsne','rna_velo_gene_ind')
    pa.visualize_velocity_projection(vlm, 'delta_S_in_S_tsne')
    pa.calculate_embedding_delta(vlm,'Px','delta_P','delta_P_in_S_tsne','prot_velo_prot_ind')
    pa.visualize_velocity_projection(vlm, 'delta_P_in_S_tsne')
    pa.cluster_specific_plot(vlm, 'delta_S_in_S_tsne',draw_cells=True)
    plt.savefig(os.path.join(fig_dir, "tenX_10k_rnav.svg"))
    pa.cluster_specific_plot(vlm, 'delta_P_in_S_tsne',draw_cells=True)
    plt.savefig(os.path.join(fig_dir, "tenX_10k_protv.svg"))

    if hasattr(vlm,'mass_filter'):
        delattr(vlm,'mass_filter')

    # Grid arrows (tSNE space)
    pa.initialize_grid_embedding(vlm, n_neighbors=20)
    uv_multiplier=10
    pa.calculate_grid_arrows(vlm,'delta_S_in_S_tsne', '_rna', min_mass=0.5,uv_multiplier=uv_multiplier)
    pa.calculate_grid_arrows(vlm,'delta_P_in_S_tsne', '_prot', min_mass=0.5,uv_multiplier=uv_multiplier)
    r_rnav = [231/255,36/255,20/255]
    b_protv = [38/255,55/255,213/255]
    plt.figure(figsize=(15,15))
    pa.plot_grid_arrows(vlm,'UV_rna',plot_cells=False, arr_col=r_rnav,arr_scale=1)
    pa.plot_grid_arrows(vlm,'UV_prot',plot_cells=True, arr_col=b_protv, color_cells_by_cluster=True,
                        pivot='tip',cell_alpha=0.2,arr_scale=1,write_labels=True)
    plt.savefig(os.path.join(fig_dir, "tenX_10k_comb.svg"))

    # Color and parameter setup for additional plots
    r_rnav = [231/255,36/255,20/255]
    b_protv = [38/255,55/255,213/255]
    black_protv = [0/255,0/255,0/255]
    arr_scale = 1
    cell_alpha = 0.2

    # Plot 1: RNA arrows only
    plt.figure(figsize=(15,15))
    pa.plot_bezier(
        vlm,
        plot_cells=True,
        color_cells_by_cluster=True,
        cell_alpha=cell_alpha,
        write_labels=True
    )
    pa.plot_grid_arrows(vlm,'UV_rna', plot_cells=False, arr_col=r_rnav, arr_scale=arr_scale)
    plt.savefig(os.path.join(fig_dir, "tenX_10k_bez_rna.svg"))
    plt.close()
    print("Plot 1 (RNA arrows only) saved: tenX_10k_bez_rna.svg")

    # Plot 2: Protein arrows only
    plt.figure(figsize=(15,15))
    pa.plot_bezier(
        vlm,
        plot_cells=True,
        color_cells_by_cluster=True,
        cell_alpha=cell_alpha,
        write_labels=True
    )
    pa.plot_grid_arrows(vlm,'UV_prot', plot_cells=False, arr_col=b_protv, arr_scale=arr_scale, pivot='tip')
    plt.savefig(os.path.join(fig_dir, "tenX_10k_bez_prot.svg"))
    plt.close()
    print("Plot 2 (Protein arrows only) saved: tenX_10k_bez_prot.svg")

    # Plot 3: Combined black arrows
    plt.figure(figsize=(15,15))
    pa.plot_bezier(
        vlm,
        plot_cells=True,
        color_cells_by_cluster=True,
        cell_alpha=cell_alpha,
        write_labels=True
    )
    vlm.UV_combined = (vlm.UV_rna + vlm.UV_prot) / 2
    pa.plot_grid_arrows(
        vlm,
        'UV_combined',
        plot_cells=False,
        arr_col=black_protv,
        arr_scale=arr_scale*1.2,
        pivot='tip'
    )
    plt.savefig(os.path.join(fig_dir, "tenX_10k_bez_combined.svg"))
    plt.close()
    print("Plot 3 (Combined black arrows) saved: tenX_10k_bez_combined.svg")

    print("\nAll 3 additional plots generated successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ProtAccel Single-cell Protein and RNA Velocity Analysis Script")
    parser.add_argument("--protein_dir",
                        default="./protein_matrix.csv",
                        help="Input protein matrix CSV file path")
    parser.add_argument("--loom_dir",
                        default="./test.loom",
                        help="Input Velocyto loom file path")
    parser.add_argument("--save_dir",
                        default="./",
                        help="Root directory for saving results and figures")
    args = parser.parse_args()

    main(args.protein_dir, args.loom_dir, args.save_dir)