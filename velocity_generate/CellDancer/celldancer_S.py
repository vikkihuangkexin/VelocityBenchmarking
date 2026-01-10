import os
import sys
import glob
import pandas as pd
import math
import matplotlib.pyplot as plt
import celldancer as cd
import celldancer.cdplt as cdplt
from celldancer.cdplt import colormap
import celldancer.utilities as cdutil
import scipy
import numpy as np
import tqdm
import scanpy as sc
import scvelo as scv
# import dynamo as dyn
from unit import find_cluster_column
def adata_to_df_with_embed(adata,
                           us_para=['unspliced', 'spliced'],
                           cell_type_para='cell_type',
                           embed_para='X_umap',
                           save_path='cell_type_u_s_sample_df.csv',
                           gene_list=None):
    def adata_to_raw_one_gene(data, us_para, gene):
        data2 = data[:, data.var.index.isin([gene])].copy()
        n = len(data2)
        u0 = data2.layers[us_para[0]][:, 0].copy().astype(np.float32)
        u0 = scipy.sparse.csr_matrix.todense(u0).tolist()
        # u0 = u0.tolist()
        s0 = data2.layers[us_para[1]][:, 0].copy().astype(np.float32)
        s0 = scipy.sparse.csr_matrix.todense(s0).tolist()
        # s0 = s0.tolist()
        raw_data = pd.DataFrame({'gene_name': [gene] * n, 'unsplice': u0, 'splice': s0})
        raw_data.splice = [i[0] for i in raw_data.splice]
        raw_data.unsplice = [i[0] for i in raw_data.unsplice]
        # raw_data.splice = [i for i in raw_data.splice]
        # raw_data.unsplice = [i for i in raw_data.unsplice]
        return (raw_data)

    if gene_list is None: gene_list = list(adata.var.index)

    dfs = []
    for gene in gene_list:
        global combined
        data_onegene = adata_to_raw_one_gene(adata, us_para=us_para, gene=gene)
        data_onegene.sort_index(inplace=True)
        dfs.append(data_onegene)

    combined = pd.concat(dfs).reset_index(drop=True)

    # cell info
    gene_num = len(gene_list)
    cellID = pd.DataFrame({'cellID': adata.obs.index})
    celltype_meta = adata.obs[cell_type_para].reset_index(drop=True)
    celltype = pd.DataFrame({'clusters': celltype_meta})  #
    embed_map = pd.DataFrame({'embedding1': adata.obsm[embed_para][:, 0], 'embedding2': adata.obsm[embed_para][:, 1]})
    # embed_info_df = pd.concat([embed_info]*gene_num)
    embed_info = pd.concat([cellID, celltype, embed_map], axis=1)
    embed_raw = pd.concat([embed_info] * gene_num)
    embed_raw = embed_raw.reset_index(drop=True)

    raw_data = pd.concat([combined, embed_raw], axis=1)

    return (raw_data)


def main(data_dir,data_file,save_dir):

    adata = scv.read(data_dir, cache=True)
    if adata.shape[0]>=50000:
        n_jobs=3
    else:
        n_jobs=8
    cluster = find_cluster_column(adata)
    if ID == '55-new':
        cluster = 'cell_type'
    elif ID=='46':
        cluster = 'lineage_cat'
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    # # sc.tl.umap(adata)
    obsm_key = list(adata.obsm.keys())
    #查看是否有外置umap的csv文件
    if any(item.lower().find('x_umap') != -1 for item in obsm_key):
        print(f"Data: {data_file} seems to have been processed using UMAP.")
    else:
        if os.path.exists(fr'/data_d/Velocity/Umap_backup/{data_file.split(".")[0]}_addUmap.csv'):
            loaded_umap = pd.read_csv(f'/data_d/Velocity/Umap_backup/{data_file.split(".")[0]}_addUmap.csv',
                                      index_col=0)
            adata.obsm['X_umap'] = loaded_umap[['UMAP1', 'UMAP2']].values
        else:
            sc.tl.umap(adata)
    adata.obs_names_make_unique()
    print('########## Processing Anndata to Dataframe ##########')
    cell_type_u_s = adata_to_df_with_embed(adata,\
                                  us_para=['unspliced', 'spliced'],\
                                  cell_type_para=cluster,\
                                  embed_para='X_umap') ##
    gene_list=list(set(cell_type_u_s.gene_name))
    del adata
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('########## Compute Cell Velocity in each gene ##########')
    loss_df, cellDancer_df=cd.velocity(cell_type_u_s,
                                       gene_list=gene_list,
                                       permutation_ratio=0.5,
                                       n_jobs=n_jobs,
                                       save_path=save_dir)
    print('########## Compute Cell Velocity ##########')
    cellDancer_df=cd.compute_cell_velocity(cellDancer_df=cellDancer_df)
    print('########## Saving ##########')
    cellDancer_df.to_csv(os.path.join(save_dir,'cell_velo.csv'))
    adata_from_dancer = cdutil.to_dynamo(cellDancer_df)
    adata_from_dancer.write_h5ad(f'{save_dir}/{data_file.split(".")[0]}_velo.h5ad')
    # plot cell velocity
    # color_library = [
    #     "#D2EBC8", "#3C77AF", "#7DBFA7", "#AECDE1", "#EE934E",
    #     "#D1352B", "#9B5B33", "#F5CFE4", "#B383B9", "#8FA4AE",
    #     "#FCED82", "#F5D2A8", "#BBDD78",
    #     "#FFB5E8", "#A8D1FF", "#FFCCF9", "#B28DFF", "#97E3FF",
    #     "#6EB5FF", "#85E3C0", "#FFABAB", "#D4FFC3", "#809FFF",
    #     "#FF9ED2", "#FFC9A7", "#C4FAF8", "#FFDA9E", "#C5A3FF",
    #     "#FFA08E", "#DCD3FF", "#FFEBB9", "#B5EAD7", "#E7C7FF",
    #     "#A5D8A7", "#FED4C4", "#B0E0E6", "#FFD8B1", "#C7CEEA",
    #     "#FDD2B3", "#B4E4C6", "#FDDEBD", "#D3BBDD", "#FFC3D8",
    #     "#A4E4B5", "#FFE4E1", "#B5C7E3", "#E6B0AA", "#D1EBD2",
    #     "#F0C2D7", "#C2E0F4", "#ECD5E3", "#D7E8FA", "#F4D03F",
    #     "#58D68D", "#EB984E", "#5DADE2", "#EC7063", "#52BE80",
    #     "#F1948A", "#48C9B0", "#AF7AC5", "#F7DC6F", "#76D7C4"
    # ]
    # color_map = {}
    # celltype = list(set(cellDancer_df['clusters']))
    # for i in range(len(celltype)):
    #     color_map[celltype[i]]=color_library[i]
    # fig, ax = plt.subplots(figsize=(15,15))
    # im = cdplt.scatter_cell(ax,cellDancer_df,
    #                         colors=color_map,
    #                         alpha=0.3,
    #                         s=10,
    #                         velocity=True,
    #                         legend='on',
    #                         min_mass=2,
    #                         arrow_grid=(30,30))
    # ax.axis('off')
    # plt.savefig(os.path.join(save_dir,'grid_arrows.pdf'))


if __name__ == '__main__':
    datalist = pd.read_csv('/data_d/Velocity/ZY_1/data.csv')
    save_dir = '/data_d/Velocity/ZY_1/result/cd'
    outlist = [20,21,22,23,24,26,32,33,34,43,45,46,47,48,53]#,40
    for i in [40]:
        ID = datalist.iloc[i]['ID']
        # if int(ID) in outlist:
        #     continue
        # ID = datalist.iloc[i]['ID']
        data_file = datalist.iloc[i]['name']
        data_dir = datalist.iloc[i]['path']
        if os.path.exists(f'{save_dir}/{ID}'):
            continue
        if ID in outlist:
            continue
        else:
            print(f'{i}___{data_file}')
            main(data_dir,data_file,f'{save_dir}/{ID}')