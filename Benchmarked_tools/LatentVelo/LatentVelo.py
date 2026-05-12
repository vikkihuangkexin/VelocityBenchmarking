import os
import numpy as np
import scanpy as sc
import latentvelo as ltv
import scvelo as scv
import torch as th
import scipy
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import argparse
import scipy as scp


def anvi_clean_recipe(adata, spliced_key = 'spliced', unspliced_key = 'unspliced', batch_key = None, root_cells=None, terminal_cells=None,
                          normalize_library=True, n_top_genes = 2000, n_neighbors=30, smooth = True, umap=False, log=True, celltype_key='celltype', r2_adjust=True, share_normalization=False, center=False, 
                      bknn=False, retain_genes = None):
    """
    Clean and setup data for celltype annotated version of LatentVelo
    """
    if normalize_library:
        spliced_library_sizes = adata.layers[spliced_key].sum(1)
        unspliced_library_sizes = adata.layers[unspliced_key].sum(1)
        if len(spliced_library_sizes.shape) == 1:
            spliced_library_sizes = spliced_library_sizes[:,None]
        if len(unspliced_library_sizes.shape) == 1:
            unspliced_library_sizes = unspliced_library_sizes[:,None]
        if share_normalization:
            library_size = spliced_library_sizes + unspliced_library_sizes
        if share_normalization:
            spliced_median_library_sizes = np.median(np.array(library_size)[:,0])
            unspliced_median_library_sizes = np.median(np.array(library_size)[:,0])
        else:
            spliced_median_library_sizes = np.median(np.array(spliced_library_sizes)[:,0])
            unspliced_median_library_sizes = np.median(np.array(unspliced_library_sizes)[:,0])
        spliced_all_size_factors = spliced_library_sizes/spliced_median_library_sizes
        unspliced_all_size_factors = unspliced_library_sizes/unspliced_median_library_sizes
        adata.layers[spliced_key] = adata.layers[spliced_key]/spliced_all_size_factors
        adata.layers[unspliced_key] = adata.layers[unspliced_key]/unspliced_all_size_factors

        adata.obs['spliced_size_factor'] = spliced_library_sizes #spliced_all_size_factors
        adata.obs['unspliced_size_factor'] = unspliced_library_sizes #unspliced_all_size_factors
    adata.X = scp.sparse.csr_matrix(adata.layers[spliced_key].copy())
    if n_top_genes != None:
        scv.pp.filter_genes_dispersion(adata, n_top_genes = n_top_genes, subset=False)
        if retain_genes == None and 'highly_variable' in adata.var.columns.values:
            adata = adata[:, adata.var.highly_variable==True]
            print('Choosing top '+str(n_top_genes) + ' genes')
        elif retain_genes != None and 'highly_variable' in adata.var.columns.values:
            print('retaining specific genes')
            adata = adata[:, (adata.var.highly_variable==True) | (adata.var.index.isin(retain_genes))]
        else:
            print('using all genes')
    if scp.sparse.issparse(adata.layers[spliced_key]):
        adata.layers[spliced_key] = adata.layers[spliced_key].todense()
        adata.layers[unspliced_key] = adata.layers[unspliced_key].todense()
    else:
        adata.layers[spliced_key] = scp.sparse.csr_matrix(adata.layers[spliced_key]).todense()
        adata.layers[unspliced_key] = scp.sparse.csr_matrix(adata.layers[unspliced_key]).todense()
    # include raw counts
    adata.layers['spliced_counts'] = np.array(adata.layers[spliced_key])
    adata.layers['unspliced_counts'] = np.array(adata.layers[unspliced_key])
    adata.X = scp.sparse.csr_matrix(adata.layers[spliced_key].copy())
    adata.layers['mask_spliced'] = np.array((adata.layers[spliced_key] > 0) + (adata.layers[unspliced_key] > 0))*1 #
    adata.layers['mask_unspliced'] = np.array((adata.layers[unspliced_key] > 0) + (adata.layers[spliced_key] > 0))*1 # + 
    if log:
        scv.pp.log1p(adata)
    # sc.pp.pca(adata)
    adata.layers['spliced'] = adata.layers[spliced_key]
    adata.layers['unspliced'] = adata.layers[unspliced_key]
    if bknn:
        import scanpy.external as sce
        sce.pp.bbknn(adata, batch_key=batch_key, local_connectivity=6)
    else:
        scv.pp.neighbors(adata, n_pcs=30, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    adata.obsp['adj'] = adata.obsp['connectivities']
    ltv.velocity_genes.compute_velocity_genes(adata, n_top_genes=n_top_genes,r2_adjust=r2_adjust)
    if umap:
        print('computing UMAP')
        sc.tl.umap(adata)
    if smooth:
        adata.uns['scale_spliced'] = 4*(1+np.std(adata.layers['Ms'], axis=0)[None])
        adata.uns['scale_unspliced'] = 4*(1+np.std(adata.layers['Mu'], axis=0)[None])
        adata.layers['spliced_raw'] = np.array(adata.layers['spliced'])
        adata.layers['unspliced_raw'] = np.array(adata.layers['unspliced'])
        if center:
            adata.uns['mean_spliced'] = np.mean(adata.layers['Ms'], axis=0)[None]
            adata.uns['mean_unspliced'] = np.mean(adata.layers['Mu'], axis=0)[None]
            adata.layers['spliced'] = np.array((adata.layers['Ms'] - adata.uns['mean_spliced'])/adata.uns['scale_spliced'])
            adata.layers[spliced_key] = np.array((adata.layers['Ms'] - adata.uns['mean_spliced'])/adata.uns['scale_spliced'])
            adata.layers['unspliced'] = np.array((adata.layers['Mu'] - adata.uns['mean_unspliced'])/adata.uns['scale_unspliced'])
            adata.layers[unspliced_key] = np.array((adata.layers['Mu'] - adata.uns['mean_unspliced'])/adata.uns['scale_unspliced'])
        else:
            adata.layers['spliced'] = np.array(adata.layers['Ms']/adata.uns['scale_spliced'])
            adata.layers[spliced_key] = np.array(adata.layers['Ms']/adata.uns['scale_spliced'])
            adata.layers['unspliced'] = np.array(adata.layers['Mu']/adata.uns['scale_unspliced'])
            adata.layers[unspliced_key] = np.array(adata.layers['Mu']/adata.uns['scale_unspliced'])
    else:
        adata.uns['scale_spliced'] = 4*(1+np.std(adata.layers[spliced_key], axis=0)[None])
        adata.uns['scale_unspliced'] = 4*(1+np.std(adata.layers[unspliced_key], axis=0)[None])
        adata.layers['spliced'] = adata.layers[spliced_key]/adata.uns['scale_spliced']
        adata.layers[spliced_key] = adata.layers[spliced_key]/adata.uns['scale_spliced']
        adata.layers['unspliced'] = adata.layers[unspliced_key]/adata.uns['scale_unspliced']
        adata.layers[unspliced_key] = adata.layers[unspliced_key]/adata.uns['scale_unspliced']
    # use label encoder
    if batch_key != None:
        label_encoder = LabelEncoder()
        batch_id = label_encoder.fit_transform(adata.obs[batch_key])
        adata.obs['batch_id'] = batch_id

        onehotbatch = OneHotEncoder().fit_transform(batch_id[:,None])
        adata.obsm['batch_onehot'] = onehotbatch
        
    else:
        batch_key = 'batch_id'
        adata.obs[batch_key] = 0
        label_encoder = LabelEncoder()
        batch_id = label_encoder.fit_transform(adata.obs[batch_key])
        adata.obs['batch_id'] = batch_id

        onehotbatch = OneHotEncoder().fit_transform(batch_id[:,None])
        adata.obsm['batch_onehot'] = onehotbatch
    
    label_encoder = LabelEncoder()
    celltype = label_encoder.fit_transform(adata.obs[celltype_key])
    adata.obs['celltype'] = celltype
    
    onehotcelltype = OneHotEncoder().fit_transform(celltype[:,None])
    adata.obsm['celltype'] = onehotcelltype
    

    if celltype_key != None:
        label_encoder = LabelEncoder()
        celltype = label_encoder.fit_transform(adata.obs[celltype_key])
        adata.obs['celltype_id'] = celltype
    else:
        adata.obs['celltype_id'] = 0

    
    if root_cells == 'precalced':
        print('using precalced root cells')
    elif celltype_key != None and root_cells != None:
        adata.obs['root'] = 0
        adata.obs['root'][adata.obs[celltype_key] == root_cells] = 1
    else:
        adata.obs['root'] = 0

    
    if terminal_cells == 'precalced':
        print('using precalced terminal cells')
    elif celltype_key != None and terminal_cells != None:
        adata.obs['terminal'] = 0
        if type(terminal_cells) == list:
            for c in terminal_cells:
                adata.obs['terminal'][adata.obs[celltype_key] == c] = 1
        else:
            adata.obs['terminal'][adata.obs[celltype_key] == terminal_cells] = 1
    else:
        adata.obs['terminal'] = 0
    
    return adata


def check_and_convert_sparse(adata):
    """
    This function checks whether the matrices in the AnnData object are sparse,
    and if so, converts them to dense numpy arrays. It also checks for key existence before conversion.
    """
    def convert_if_sparse(matrix):
        if isinstance(matrix, scipy.sparse.coo_matrix):
            matrix = matrix.tocsr()
        if isinstance(matrix, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
            return matrix.toarray()
        return matrix

    # Check and convert layers
    for key in ['spliced_counts', 'unspliced_counts', 'spliced', 'unspliced', 'mask_spliced', 'mask_unspliced']:
        if key in adata.layers:
            adata.layers[key] = convert_if_sparse(adata.layers[key])

    # Check and convert obs
    for key in ['spliced_size_factor', 'unspliced_size_factor', 'root', 'batch_id', 'celltype_id']:
        if key in adata.obs:
            adata.obs[key] = convert_if_sparse(adata.obs[key])

    # Check and convert obsm
    for key in ['batch_onehot', 'celltype']:
        if key in adata.obsm:
            adata.obsm[key] = convert_if_sparse(adata.obsm[key])

    return adata

def main(data_path, save_dir, celltype_key, simulate=False,
         min_shared_counts=5, n_top_genes=200, latent_dim=40, encoder_hidden=45,
         zr_dim=2, h_dim=3, batch_size=1000, epochs=50, grad_clip=100, random_seed=521):
    adata = sc.read_h5ad(data_path)
    adata.obs_names_make_unique()
    print(f'[INFO] Read in {data_path}', flush=True)

    # Simulation mode adjustments
    if simulate:
        print("[INFO] Simulation mode: relaxing preprocessing and setting cluster labels to 'milestone'")
        # Set cluster labels to 'milestone'
        adata.obs[celltype_key] = "milestone"
        # Copy X_dimred to X_umap if present
        if "X_dimred" in adata.obsm and "X_umap" not in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_dimred"]
        # Relax preprocessing: set min_shared_counts to None, use top_genes if available
        min_shared_counts = None
        if "top_genes" in adata.uns:
            n_top_genes = len(adata.uns["top_genes"])
        else:
            n_top_genes = None

    new_working_dir = f'{save_dir}/latentvelo/'
    name = 'velocity'
    os.makedirs(new_working_dir, exist_ok=True)
    os.chdir(new_working_dir)
    th.manual_seed(random_seed)
    if min_shared_counts is not None:
        scv.pp.filter_genes(adata, min_shared_counts=min_shared_counts)
    if issparse(adata.X):
        adata.X = adata.X.toarray()
        adata.layers['spliced'] = adata.layers['spliced'].toarray()
        adata.layers['unspliced'] = adata.layers['unspliced'].toarray()
    else:
        pass
    print(f'cal for {data_dir}', flush=True)
    adata = check_and_convert_sparse(adata)

    # adata = ltv.utils.anvi_clean_recipe(adata, n_top_genes=300, celltype_key='milestone', log=False)
    adata = anvi_clean_recipe(adata, n_top_genes=n_top_genes, celltype_key=celltype_key, log=False)
    model = ltv.models.AnnotVAE(observed=adata.shape[1], latent_dim=latent_dim, encoder_hidden=encoder_hidden, zr_dim=zr_dim, h_dim=h_dim,
                          celltypes=len(adata.obs[celltype_key].unique())) # 20, 25
    adata = check_and_convert_sparse(adata)
    epochs, val_ae, val_traj = ltv.train_anvi(model, adata, batch_size=batch_size,
                                          epochs=epochs, name=name, grad_clip=grad_clip, random_seed=random_seed)
    latent_adata = ltv.output_results(model, adata)
    latent_adata.write_h5ad(f'{save_dir}/latent_adata.h5ad')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LatentVelo training and velocity computation")
    parser.add_argument("--data_path", required=True, help="Path to input .h5ad file")
    parser.add_argument("--save_dir", required=True, help="Directory to save results")
    parser.add_argument("--celltype_key", default="cell_type", help="obs column for cell type (default: cell_type)")
    parser.add_argument("--simulate", action="store_true", help="If set, relax preprocessing (min_shared_counts=None), set clusters to 'milestone', and copy X_dimred to X_umap if present")
    parser.add_argument("--min_shared_counts", type=int, default=5, help="Minimum shared counts for gene filtering (default: 5)")
    parser.add_argument("--n_top_genes", type=int, default=200, help="Number of top genes to select (default: 200)")
    parser.add_argument("--latent_dim", type=int, default=40, help="Latent dimension for the model (default: 40)")
    parser.add_argument("--encoder_hidden", type=int, default=45, help="Encoder hidden dimension (default: 45)")
    parser.add_argument("--zr_dim", type=int, default=2, help="ZR dimension (default: 2)")
    parser.add_argument("--h_dim", type=int, default=3, help="H dimension (default: 3)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for training (default: 1000)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--grad_clip", type=float, default=100.0, help="Gradient clipping value (default: 100)")
    parser.add_argument("--random_seed", type=int, default=521, help="Random seed for reproducibility (default: 521)")

    args = parser.parse_args()
    main(
        args.data_path, args.save_dir, args.celltype_key, args.simulate,
        args.min_shared_counts, args.n_top_genes, args.latent_dim, args.encoder_hidden,
        args.zr_dim, args.h_dim, args.batch_size, args.epochs, args.grad_clip, args.random_seed
    )