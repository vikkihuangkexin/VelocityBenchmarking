import os.path

import scvelo as scv
import scanpy as sc
import pandas as pd
from unit import find_cluster_column

def main(data_dir,data_file,save_dir):
    adata= sc.read(data_dir, cache=True)
    adata.obs_names_make_unique()
    cluster = 'milestone'
    adata.obsm['X_umap'] = adata.obsm['X_dimred']

    # scv.pl.proportions(adata, groupby='celltype_full')

    # obsm_key = list(adata.obsm.keys())
    # if any(item.lower().find('x_umap') != -1 for item in obsm_key):
    #     print()
    # else:
    #     if os.path.exists(fr'/data_d/Velocity/Umap_backup/{data_file.split(".")[0]}_addUmap.csv'):
    #         loaded_umap = pd.read_csv(f'/data_d/Velocity/Umap_backup/{data_file.split(".")[0]}_addUmap.csv',index_col=0)
    #         loaded_umap_reindexed = loaded_umap.reindex(adata.obs.index)
    #         adata.obsm['X_umap'] = loaded_umap[['UMAP1', 'UMAP2']].values
    #     else:
    #         scv.tl.umap(adata)
    # if data_file.startswith('48'):
    #     adata = adata[~adata.to_df().duplicated(), :]
    #     adata.obs_names_make_unique()
    if adata.n_vars < 10000:
        top_gene = (adata.n_vars // 500) * 500
        top_gene = min(top_gene, adata.n_vars,500)
        shared_counts = 20
    else:
        top_gene = 2000
        shared_counts=1
    scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=top_gene)
    if adata.n_vars==0:
        print(f'{ID} shape error!')
        return data_dir
    # adata = adata[~adata.to_df().duplicated(), :]
    # sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30,method='umap')
    # scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    cluster = find_cluster_column(adata)
    if data_file.endswith('time.h5ad'):
        cluster='time'
    # adata1 = adata.copy()
    scv.tl.velocity(adata, mode='stochastic')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        scv.tl.velocity_graph(adata)
        scv.pl.velocity_embedding_stream(adata, basis='umap',color=cluster,save = f'{save_dir}/stream_arrow.pdf')
        scv.pl.velocity_embedding_grid(adata, basis='umap', color=cluster, save=f'{save_dir}/grid_arrow.pdf')
        scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120, save = f'{save_dir}/full_arrow.pdf')
        adata.write_h5ad(f'{save_dir}/{data_file.split(".")[0]}_velo.h5ad')
    except:
        adata.write_h5ad(f'{save_dir}/{data_file.split(".")[0]}_velo.h5ad')


    adata= sc.read(data_dir, cache=True)
    adata.obs_names_make_unique()
    # scv.pl.proportions(adata, groupby='celltype_full')
    cluster = 'milestone'
    adata.obsm['X_umap'] = adata.obsm['X_dimred']
    obsm_key = list(adata.obsm.keys())
    # if any(item.lower().find('x_umap') != -1 for item in obsm_key):
    #     print()
    # else:
    #     if os.path.exists(fr'/data_d/Velocity/Umap_backup/{data_file.split(".")[0]}_addUmap.csv'):
    #         loaded_umap = pd.read_csv(f'/data_d/Velocity/Umap_backup/{data_file.split(".")[0]}_addUmap.csv',index_col=0)
    #         adata.obsm['X_umap'] = loaded_umap[['UMAP1', 'UMAP2']].values
    #     else:
    #         scv.tl.umap(adata)
    # if data_file.startswith('48'):
    #     adata = adata[~adata.to_df().duplicated(), :]
    #     adata.obs_names_make_unique()
    if adata.n_vars < 10000:
        top_gene = (adata.n_vars // 500) * 500
        top_gene = min(top_gene, adata.n_vars, 500)
        shared_counts = 20
    else:
        top_gene = 2000
        shared_counts = 1
    scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=top_gene)
    # adata = adata[~adata.to_df().duplicated(), :]
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30,method='umap')
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    cluster = find_cluster_column(adata)
    scv.tl.recover_dynamics(adata,n_jobs=8)
    scv.tl.velocity(adata, mode='dynamical')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        scv.tl.velocity_graph(adata)
        scv.pl.velocity_embedding_stream(adata, basis='umap',color=cluster,save = f'{save_dir}/stream_arrow_D.pdf')
        scv.pl.velocity_embedding_grid(adata, basis='umap', color=cluster, save=f'{save_dir}/grid_arrow_D.pdf')
        scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120, save = f'{save_dir}/full_arrow_D.pdf')
        adata.write_h5ad(f'{save_dir}/{data_file.split(".")[0]}_velo_D.h5ad')
    except:
        adata.write_h5ad(f'{save_dir}/{data_file.split(".")[0]}_velo_D.h5ad')
    # print()
if __name__ == '__main__':
    datalist = pd.read_csv('/data_d/Velocity/data/simdata_local/sim_data_0924.csv')
    save_dir = '/data_d/Velocity/ZY/simdata/dis/scvelo'
    outlist = ['disconnected_cell1000_gene10000']#7, 15,31
    for i in range(len(datalist)):
        ID = datalist.iloc[i]['ID']
        if ID not in outlist:
            continue
        # if ID.startswith('dis'):
        #     continue
        data_file = datalist.iloc[i]['name']
        data_dir = datalist.iloc[i]['path']
        import re
        cell_match = re.search(r'cell(\d+)', data_file)
        gene_match = re.search(r'gene(\d+)', data_file)
        cell_num = int(cell_match.group(1)) if cell_match else None
        gene_num = int(gene_match.group(1)) if gene_match else None
        # if cell_num==1000 or gene_num==1000:
        #     continue

        # if ID=='35':
        #     continue
        if os.path.exists(f'{save_dir}/{ID}'):
            continue
        else:
            print(f"{i}____{data_file}")
            check = main(data_dir,data_file,f'{save_dir}/{ID}')
            if check:
                os.makedirs(os.path.join(save_dir,f'Nvars_error_{data_file}'))

