import celldancer as cd
import pandas as pd
import anndata as ann
import scvelo as scv
import os
import scanpy as sc
from pathlib import Path
from unit import find_cluster_column,velocity_confidence_plot

def find_h5ad(path,ID,type):
    path = Path(path)
    return [str(file) for file in path.rglob(f'{ID}_*{type}') if file.is_file()]

def main(data_dir, figure_dir,method):
    adata = ann.read_h5ad(data_dir)
    data_file = data_dir.split('/')[-1]
    ID = data_file.split('.')[0]
    label = find_cluster_column(adata)
    adata = adata[~adata.to_df().duplicated(), :]
    if method == 'velocyto':
        adata.layers["velocity"] = adata.layers["velocity_S"].toarray()
        adata.layers["Ms"] = adata.layers["M_s"].toarray()
        adata.layers["Mu"] = adata.layers["M_u"].toarray()
        sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30, method='umap')
        scv.tl.velocity_graph(adata, n_jobs=8)
        scv.tl.velocity_pseudotime(adata)
    elif method == 'cellDancer':
        cell_df_path='...../cell_velo.csv'
        cellDancer_df = pd.read_csv(cell_df_path)
        dt = 0.05
        t_total = {dt: int(10 / dt)}
        n_repeats = 10
        print(f'##### estimate cellDancer pseudotime #####')
        cellDancer_df = cd.pseudo_time(cellDancer_df=cellDancer_df,
                                       grid=(30, 30),
                                       dt=dt,
                                       t_total=t_total[dt],
                                       n_repeats=n_repeats,
                                       speed_up=(100, 100),
                                       n_paths=3,
                                       psrng_seeds_diffusion=[i for i in range(n_repeats)],
                                       n_jobs=8)
        gene_name = cellDancer_df.gene_name[0]
        one_gene_idx = cellDancer_df.gene_name == gene_name
        data = cellDancer_df[one_gene_idx][['cellID', 'pseudotime']].dropna()
        data.index = data['cellID']
        data = data.reindex(list(adata.obs_names))
        adata.obs['velocity_pseudotime'] = data['pseudotime'].to_numpy()
        adata.obsm["X_umap"] = adata.obsm["X_cdr"]
        adata.obsm["velocity_umap"] = adata.obsm["velocity_cdr"]
        adata.layers["velocity"] = adata.layers["velocity_S"]
        adata.layers["Ms"] = adata.layers["M_s"]
        adata.layers["Mu"] = adata.layers["M_u"]
        sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30, method='umap')
        scv.tl.velocity_graph(adata, n_jobs=8)
        scv.tl.velocity_pseudotime(adata)
    else:
        sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30, method='umap')
        scv.tl.velocity_graph(adata, n_jobs=8)
        scv.tl.velocity_pseudotime(adata)

    palette_30_3 = [
        "#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4",
        "#66c2a5", "#3288bd", "#abdda4", "#e6f598", "#fee08b",
        "#f46d43", "#e7298a", "#a6cee3", "#1f78b4", "#b2df8a",
        "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
        "#cab2d6", "#6a3d9a", "#ffff99", "#b15928", "#8dd3c7",
        "#bc80bd", "#ccebc5", "#ffed6f", "#999999"
    ]
    print(f'##### Plotting {data_file} #####')
    ####cluster
    for fmt in ['png', 'pdf']:
        save_dir = f'{figure_dir}/png' if fmt == 'png' else f'{figure_dir}/pdf'
        scv.pl.velocity_embedding_stream(
            adata,
            basis="umap",
            size=100,
            alpha=0.6,
            color=label,
            legend_fontsize=9,
            legend_loc='right margin',
            fontsize=None,
            dpi=400,
            arrow_size=0.00001,
            linewidth=0,
            title=method,
            palette=palette_30_3,
            save=os.path.join(save_dir, f"{method}_{ID}_umap.{fmt}")
        )


    ####stream

    for fmt in ['png', 'pdf']:
        save_path = os.path.join(f'{figure_dir}/png' if fmt == 'png' else f'{figure_dir}/pdf', f"{method}_{ID}_stream.{fmt}")
        scv.pl.velocity_embedding_stream(
            adata,
            basis="umap",
            size=100,
            alpha=0.6,
            color=label,
            legend_fontsize=9,
            legend_loc='right margin',
            fontsize=None,
            density=2,
            dpi=400,
            arrow_size=1,
            linewidth=1,
            palette=palette_30_3,
            title=method,    ####根据自己的工具设定
            save=save_path
        )

    ####grid
    for fmt in ['png', 'pdf']:
        save_path = os.path.join(f'{figure_dir}/png' if fmt == 'png' else f'{figure_dir}/pdf', f"{method}_{ID}_grid.{fmt}")
        scv.pl.velocity_embedding_grid(
            adata,
            basis="umap",
            size=100,
            alpha = 0.6,
            color=label,
            legend_fontsize=9,
            legend_loc='right margin',
            fontsize=None,
            density=0.8,
            dpi=400,
            arrow_size=1,
            linewidth=0.3,
            palette=palette_30_3,
            title=method,
            save=save_path
        )

    ####velocity_pseudotime

    for fmt in ['png', 'pdf']:
        save_path = os.path.join(f'{figure_dir}/png' if fmt == 'png' else f'{figure_dir}/pdf', f"{method}_{ID}_pseudotime.{fmt}")
        scv.pl.scatter(
            adata,
            color='velocity_pseudotime',
            cmap='gnuplot',
            size=100,
            dpi=400,
            figsize=(8, 6),
            title=method,
            save=save_path
        )

if __name__ == '__main__':
    method='...'
    save_dir = f'.../example/result/{method}/...'
    figure_dir = f'.../example/figure/{method}'
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    main(save_dir,figure_dir,method)





