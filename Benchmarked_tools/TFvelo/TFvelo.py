import pandas as pd    
import TFvelo as TFv
import anndata as ad
import numpy as np
import scanpy as sc
import scvelo as scv
import matplotlib
matplotlib.use('AGG')

import os, sys

def check_data_type(adata):
    for key in list(adata.var):
        if adata.var[key][0] in ['True', 'False']:
            adata.var[key] = adata.var[key].map({'True': True, 'False': False})
    return          

def data_type_tostr(adata, key):
    if key in adata.var.keys():
        if adata.var[key][0] in [True, False]:
            adata.var[key] = adata.var[key].map({True: 'True', False:'False'})
    return          


def preprocess(args):
    print('----------------------------------preprocess',args.dataset_name,'---------------------------------------------')
    if args.dataset_name == 'pancreas':
        adata = scv.datasets.pancreas()
    elif args.dataset_name == 'gastrulation_erythroid':
        adata = scv.datasets.gastrulation_erythroid()   
        if args.simulate:
            adata.obs['clusters'] = 'milestone'
        else:
            adata.uns['clusters_colors'] = adata.uns['celltype_colors'].copy()
            adata.obs['clusters'] = adata.obs['celltype'].copy()
    elif args.dataset_name == '10x_mouse_brain':
        adata = ad.read_h5ad("data/10x_mouse_brain/adata_rna.h5ad") 
        if args.simulate:
            adata.obs['clusters'] = 'milestone'
        else:
            adata.uns['clusters_colors'] = np.array(['red', 'orange', 'yellow', 'green','skyblue', 'blue','purple', 'pink', '#8fbc8f', '#f4a460', '#fdbf6f', '#ff7f00', '#b2df8a', '#1f78b4',
                '#6a3d9a', '#cab2d6'], dtype=object) 
            adata.obs['clusters'] = adata.obs['celltype'].copy()
    else:
        raise ValueError(f"Unknown dataset_name: {args.dataset_name}")   

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    adata
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    adata.uns['genes_all'] = np.array(adata.var_names)

    if "spliced" in adata.layers:
        #adata.layers["total"] = adata.layers["spliced"].todense() + adata.layers["unspliced"].todense()
        adata.layers["total"] = adata.layers["spliced"] + adata.layers["unspliced"]

    elif "new" in adata.layers:
        adata.layers["total"] = np.array(adata.layers["total"].todense())
    else:
        adata.layers["total"] = adata.X
    adata.layers["total_raw"] = adata.layers["total"].copy()
    n_cells, n_genes = adata.X.shape
    
    sc.pp.filter_genes(adata, min_cells=int(n_cells/50))
    sc.pp.filter_cells(adata, min_genes=int(n_genes/50))
    #sc.pp.filter_genes(adata, min_cells=10)
    #sc.pp.filter_cells(adata, min_genes=200)
    
    TFv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, log=True) #include the following steps
    adata.X = adata.layers["total"].copy()
    


    if not args.simulate and not args.dataset_name in ['10x_mouse_brain']:
        adata.uns['clusters_colors'] = np.array(['red', 'orange', 'yellow', 'green','skyblue', 'blue','purple', 'pink', '#8fbc8f', '#f4a460', '#fdbf6f', '#ff7f00', '#b2df8a', '#1f78b4',
            '#6a3d9a', '#cab2d6'], dtype=object)

    gene_names = []
    for tmp in adata.var_names:
        gene_names.append(tmp.upper())
    adata.var_names = gene_names
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    TFv.pp.moments(adata, n_pcs=30, n_neighbors=args.n_neighbors)

    TFv.pp.get_TFs(adata, databases=args.TF_databases)
    print(adata)
    adata.uns['genes_pp'] = np.array(adata.var_names)
    if 'X_dimred' in adata.obsm:
        adata.obsm['X_umap'] = adata.obsm['X_dimred']
    adata.write(args.result_path + 'pp.h5ad')



def main(args):
    print('--------------------------------')
    adata = ad.read_h5ad(args.result_path + 'pp.h5ad')

    n_jobs_max = np.max([int(os.cpu_count()/2), 1])
    if args.n_jobs >= 1:
        n_jobs = np.min([args.n_jobs, n_jobs_max])
    else:
        n_jobs = n_jobs_max
    print('n_jobs:', n_jobs)
    flag = TFv.tl.recover_dynamics(adata, n_jobs=n_jobs, max_iter=args.max_iter, var_names=args.var_names,
        WX_method = args.WX_method, WX_thres=args.WX_thres, max_n_TF=args.max_n_TF, n_top_genes=args.n_top_genes,
        fit_scaling=True, use_raw=args.use_raw, init_weight_method=args.init_weight_method, 
        n_time_points=args.n_time_points) 
    if flag==False:
        return adata, False
    if 'highly_variable_genes' in adata.var.keys():
        data_type_tostr(adata, key='highly_variable_genes')

    adata.layers['velocity'] = adata.layers['velo_hat'] / np.expand_dims(adata.var['fit_scaling_y'], 0)
    scv.tl.velocity_graph(adata, xkey='M_total', n_jobs=2)
    valid_genes = ~adata.var['fit_scaling_y'].isna()
    adata = adata[:, valid_genes].copy()
    adata.write(args.result_path + 'rc.h5ad')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset_name', type=str, default="pancreas", help='pancreas, gastrulation_erythroid, 10x_mouse_brain, hesc1') 
    parser.add_argument( '--n_jobs', type=int, default=28, help='number of cpus to use')
    parser.add_argument( '--var_names', type=str, default="all", help='all, highly_variable_genes')
    parser.add_argument( '--init_weight_method', type=str, default= "correlation", help='use correlation to initialize the weights')
    parser.add_argument( '--WX_method', type=str, default= "lsq_linear", help='LS, LASSO, Ridge, constant, LS_constrant, lsq_linear')
    parser.add_argument( '--n_neighbors', type=int, default=30, help='number of neighbors')
    parser.add_argument( '--WX_thres', type=int, default=20, help='the threshold for weights')
    parser.add_argument( '--n_top_genes', type=int, default=2000, help='n_top_genes')
    parser.add_argument( '--TF_databases', nargs='+', default='ENCODE ChEA', help='knockTF ChEA ENCODE')
    parser.add_argument( '--max_n_TF', type=int, default=99, help='max number of TFs')
    parser.add_argument( '--max_iter', type=int, default=20, help='max number of iteration in EM')
    parser.add_argument( '--n_time_points', type=int, default=1000, help='use_raw')
    parser.add_argument( '--save_name', type=str, default='_demo', help='save_name')
    parser.add_argument( '--use_raw', type=int, default=0, help='use_raw')
    parser.add_argument( '--basis', type=str, default='umap', help='umap')
    parser.add_argument('--simulate', action='store_true', help='Whether the data is simulation data')

    args = parser.parse_args() 
    args.result_path = 'TFvelo_'+ args.dataset_name + args.save_name+ '/'
    print('********************************************************************************************************')
    print('********************************************************************************************************')  
    print(args)
    preprocess(args)  
    main(args) 
  
