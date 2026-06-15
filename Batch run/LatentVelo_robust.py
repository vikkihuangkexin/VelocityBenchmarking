import os
import numpy as np
import scanpy as sc
import latentvelo as ltv
import scvelo as scv
import torch as th
import scipy
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import scipy as scp
import random
import datetime
import logging

# Configuration
INPUT_DIR = os.getenv('INPUT_DIR', './example')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './example/output/LatentVelo')

def set_random_seeds():
    """Set random seeds using current timestamp"""
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    return seed

# --- Helper Functions from Original Script (Preserved) ---
def anvi_clean_recipe(adata, spliced_key = 'spliced', unspliced_key = 'unspliced', batch_key = None, root_cells=None, terminal_cells=None,
                          normalize_library=True, n_top_genes = 2000, n_neighbors=30, smooth = True, umap=False, log=True, celltype_key='celltype', r2_adjust=True, share_normalization=False, center=False, 
                      bknn=False, retain_genes = None):
    # ... (Keep original logic to ensure consistency) ...
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

        adata.obs['spliced_size_factor'] = spliced_library_sizes 
        adata.obs['unspliced_size_factor'] = unspliced_library_sizes 
    adata.X = scp.sparse.csr_matrix(adata.layers[spliced_key].copy())
    if n_top_genes != None:
        scv.pp.filter_genes_dispersion(adata, n_top_genes = n_top_genes, subset=False)
        if retain_genes == None and 'highly_variable' in adata.var.columns.values:
            adata = adata[:, adata.var.highly_variable==True]
        elif retain_genes != None and 'highly_variable' in adata.var.columns.values:
            adata = adata[:, (adata.var.highly_variable==True) | (adata.var.index.isin(retain_genes))]
    if scp.sparse.issparse(adata.layers[spliced_key]):
        adata.layers[spliced_key] = adata.layers[spliced_key].todense()
        adata.layers[unspliced_key] = adata.layers[unspliced_key].todense()
    else:
        adata.layers[spliced_key] = scp.sparse.csr_matrix(adata.layers[spliced_key]).todense()
        adata.layers[unspliced_key] = scp.sparse.csr_matrix(adata.layers[unspliced_key]).todense()
    adata.layers['spliced_counts'] = np.array(adata.layers[spliced_key])
    adata.layers['unspliced_counts'] = np.array(adata.layers[unspliced_key])
    adata.X = scp.sparse.csr_matrix(adata.layers[spliced_key].copy())
    adata.layers['mask_spliced'] = np.array((adata.layers[spliced_key] > 0) + (adata.layers[unspliced_key] > 0))*1 
    adata.layers['mask_unspliced'] = np.array((adata.layers[unspliced_key] > 0) + (adata.layers[spliced_key] > 0))*1 
    if log:
        scv.pp.log1p(adata)
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
        pass
    elif celltype_key != None and root_cells != None:
        adata.obs['root'] = 0
        adata.obs['root'][adata.obs[celltype_key] == root_cells] = 1
    else:
        adata.obs['root'] = 0
    if terminal_cells == 'precalced':
        pass
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
    def convert_if_sparse(matrix):
        if isinstance(matrix, scipy.sparse.coo_matrix):
            matrix = matrix.tocsr()
        if isinstance(matrix, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
            return matrix.toarray()
        return matrix
    for key in ['spliced_counts', 'unspliced_counts', 'spliced', 'unspliced', 'mask_spliced', 'mask_unspliced']:
        if key in adata.layers:
            adata.layers[key] = convert_if_sparse(adata.layers[key])
    for key in ['spliced_size_factor', 'unspliced_size_factor', 'root', 'batch_id', 'celltype_id']:
        if key in adata.obs:
            adata.obs[key] = convert_if_sparse(adata.obs[key])
    for key in ['batch_onehot', 'celltype']:
        if key in adata.obsm:
            adata.obsm[key] = convert_if_sparse(adata.obsm[key])
    return adata
# --------------------------------------------------

def main_single(data_path, save_dir, celltype_key, simulate=False,
         min_shared_counts=5, n_top_genes=200, latent_dim=40, encoder_hidden=45,
         zr_dim=2, h_dim=3, batch_size=1000, epochs=50, grad_clip=100, random_seed=521):
    
    adata = sc.read_h5ad(data_path)
    adata.obs_names_make_unique()
    print(f'[INFO] Read in {data_path}', flush=True)

    if simulate:
        print("[INFO] Simulation mode: relaxing preprocessing and setting cluster labels to 'milestone'")
        adata.obs[celltype_key] = "milestone"
        if "X_dimred" in adata.obsm and "X_umap" not in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_dimred"]
        min_shared_counts = None
        if "top_genes" in adata.uns:
            n_top_genes = len(adata.uns["top_genes"])
        else:
            n_top_genes = None

    new_working_dir = f'{save_dir}/latentvelo/'
    name = 'velocity'
    os.makedirs(new_working_dir, exist_ok=True)
    
    # Save current CWD to restore later if needed, though robust calling structure handles absolute paths
    original_cwd = os.getcwd()
    os.chdir(new_working_dir)
    
    # Apply seed (Passed from set_random_seeds)
    th.manual_seed(random_seed)
    
    if min_shared_counts is not None:
        scv.pp.filter_genes(adata, min_shared_counts=min_shared_counts)
    if issparse(adata.X):
        adata.X = adata.X.toarray()
        adata.layers['spliced'] = adata.layers['spliced'].toarray()
        adata.layers['unspliced'] = adata.layers['unspliced'].toarray()
    
    adata = check_and_convert_sparse(adata)
    adata = anvi_clean_recipe(adata, n_top_genes=n_top_genes, celltype_key=celltype_key, log=False)
    
    model = ltv.models.AnnotVAE(observed=adata.shape[1], latent_dim=latent_dim, encoder_hidden=encoder_hidden, zr_dim=zr_dim, h_dim=h_dim,
                          celltypes=len(adata.obs[celltype_key].unique()))
    
    adata = check_and_convert_sparse(adata)
    epochs, val_ae, val_traj = ltv.train_anvi(model, adata, batch_size=batch_size,
                                          epochs=epochs, name=name, grad_clip=grad_clip, random_seed=random_seed)
    latent_adata = ltv.output_results(model, adata)
    latent_adata.write_h5ad(f'{save_dir}/latent_adata.h5ad')
    
    # Restore CWD
    os.chdir(original_cwd)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"error_log_{timestamp}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_files = [
        {
            "path": os.path.join(INPUT_DIR, "Simulation-data/bifurcating_cell1000_gene500_dataset.h5ad"),
            "simulate": True,
            "id_pre_base": "LatentVelo_bifurcating_cell1000_gene500",
            "celltype_key": "cell_type"
        },
        {
            "path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "simulate": False,
            "id_pre_base": "LatentVelo_7",
            "celltype_key": "cell_type"
        }
    ]

    n_runs = 5

    for run_idx in range(1, n_runs + 1):
        seed = set_random_seeds()
        print(f"\n[Run {run_idx}/{n_runs}] Seed: {seed}")

        for file_info in data_files:
            file_path = file_info["path"]
            simulate = file_info["simulate"]
            id_pre_base = file_info["id_pre_base"]
            celltype_key = file_info["celltype_key"]

            try:
                input_file = os.path.basename(file_path)
                id_pre = f"{id_pre_base}_r{run_idx}"
                print(f"  Processing: {input_file}")

                save_dir = os.path.join(OUTPUT_DIR, id_pre)

                main_single(
                    data_path=file_path, 
                    save_dir=save_dir, 
                    celltype_key=celltype_key, 
                    simulate=simulate,
                    random_seed=seed,
                    epochs=50 # Default settings
                )

                logging.info(f"Success: {input_file} run {run_idx}")

            except Exception as e:
                logging.error(f"Error processing {file_path} run {run_idx}: {str(e)}", exc_info=True)
                print(f"Error: {e}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logging.error(f"Unhandled error: {e}", exc_info=True)