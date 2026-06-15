import yaml, os, glob, re
import pandas as pd
import scanpy as sc
import pyrovelocity.utils
import pyrovelocity.workflows.main_workflow
from pyrovelocity.utils import print_config_tree
from pyrovelocity.utils import print_docstring
from pyrovelocity.workflows.main_workflow import download_data
from pyrovelocity.workflows.main_workflow import postprocess_data
from pyrovelocity.workflows.main_workflow import preprocess_data
from pyrovelocity.workflows.main_workflow import summarize_data
from pyrovelocity.workflows.main_workflow import train_model
from pyrovelocity.interfaces import PreprocessDataInterface, PyroVelocityTrainInterface
from pyrovelocity.workflows.main_configuration import PostprocessConfiguration
from pyrovelocity.io.compressedpickle import CompressedPickle
import scvelo as scv
import torch
import numpy as np
import random
import datetime
import logging
import tensorflow as tf # For seed consistency if needed

# Configuration
INPUT_DIR = os.getenv('INPUT_DIR', './example')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './example/output/PyroVelocity')

def bool_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:bool', 'True' if data else 'False')
yaml.add_representer(bool, bool_representer)

# Original base config template
base_data_template = {
    'download_dataset': {
        'data_set_name': 'velo',
        'data_external_path': 'None', # Dynamic
        'data_url': 'None',
        'n_obs': 'None',
        'n_vars': 'None'
    },
    'preprocess_data': {
        'data_set_name': 'velo',
        'adata': 'None', # Dynamic
        'data_processed_path': 'None', # Dynamic
        'overwrite': True,
        'n_top_genes': 2000,
        'min_shared_counts': 5,
        'process_cytotrace': True,
        'use_obs_subset': True,
        'n_obs_subset': 300,
        'use_vars_subset': True,
        'n_vars_subset': 200,
        'count_threshold': 0,
        'n_pcs': 30,
        'n_neighbors': 30,
        'default_velocity_mode': 'dynamical',
        'vector_field_basis': 'umap',
        'cell_state': 'cell_type'
    },
    'training_configuration_1': {
        'adata': 'None', # Dynamic
        'data_set_name': 'velo',
        'model_identifier': 'model1',
        'guide_type': 'auto_t0_constraint',
        'model_type': 'auto',
        'batch_size': -1,
        'use_gpu': 'auto',
        'likelihood': 'Poisson',
        'num_samples': 30,
        'log_every': 100,
        'patient_improve': 0.0001,
        'patient_init': 45,
        'seed': 99, # Dynamic
        'learning_rate': 0.01,
        'max_epochs': 300,
        'include_prior': True,
        'library_size': True,
        'offset': False,
        'input_type': 'raw',
        'cell_specific_kinetics': 'None',
        'kinetics_num': 2,
        'force': True
    },
    'postprocess_configuration': {
        'number_posterior_samples': 4
    }
}

_original_torch_load = torch.load
def torch_load_patch(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_patch
os.environ["FLYTE_SDK_TYPE_CHECK"] = "False"

def set_random_seeds():
    """Set random seeds using current timestamp"""
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

def generate_yaml(data, data_path, save_dir, celltype_key, umap_key, seed, simulate=False, n_top_genes=2000, min_shared_counts=5, max_epochs=300, batch_size=-1):
    import copy
    current_data = copy.deepcopy(data)
    
    os.makedirs(save_dir, exist_ok=True)
    current_data['download_dataset']['data_external_path'] = save_dir
    current_data['preprocess_data']['adata'] = data_path
    current_data['preprocess_data']['data_processed_path'] = f'{save_dir}/processed'
    current_data['preprocess_data']['process_cytotrace'] = False
    current_data['preprocess_data']['use_obs_subset'] = False
    current_data['training_configuration_1']['adata'] = data_path
    current_data['preprocess_data']['random_seed'] = seed # Set seed
    current_data['training_configuration_1']['seed'] = seed # Set seed for training
    current_data['preprocess_data']['selected_genes'] = ""
    current_data['preprocess_data']['vector_field_basis'] = umap_key
    current_data['preprocess_data']['cell_state'] = celltype_key
    current_data['preprocess_data']['reports_processed_path'] = f'{save_dir}/reports/processed'

    # Simulation mode adjustments
    if simulate:
        print("[INFO] Simulation mode: relaxing preprocessing and setting cluster labels to 'milestone'")
        adata = sc.read_h5ad(data_path)
        adata.obs[celltype_key] = "milestone"
        if "X_dimred" in adata.obsm and "X_umap" not in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_dimred"]
        min_shared_counts = None
        if "top_genes" in adata.uns:
            n_top_genes = len(adata.uns["top_genes"])
        
        temp_path = f'{save_dir}/temp_adata.h5ad'
        adata.write_h5ad(temp_path)
        current_data['preprocess_data']['adata'] = temp_path
        current_data['training_configuration_1']['adata'] = temp_path

    current_data['preprocess_data']['n_top_genes'] = n_top_genes
    current_data['preprocess_data']['min_shared_counts'] = min_shared_counts
    current_data['training_configuration_1']['max_epochs'] = max_epochs
    current_data['training_configuration_1']['batch_size'] = batch_size

    config_path = f'{save_dir}/config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(current_data, f, default_flow_style=False)
    
    return config_path

def main_single(data_template, data_path, save_dir, celltype_key, umap_key, seed, simulate, n_top_genes=2000, min_shared_counts=5, max_epochs=300, batch_size=-1):
    config_path = generate_yaml(data_template, data_path, save_dir, celltype_key, umap_key, seed, simulate, n_top_genes, min_shared_counts, max_epochs, batch_size)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Switch to save dir for pyro output
    original_cwd = os.getcwd()
    os.chdir(save_dir)
    
    try:
        preprocess_args = PreprocessDataInterface(**config['preprocess_data'])
        processed_data = preprocess_data(
          data=preprocess_args.adata,
          preprocess_data_args=preprocess_args,
        )
        train_args = PyroVelocityTrainInterface(**config['training_configuration_1'])
        model_output = train_model(
          preprocess_outputs=processed_data,
          train_model_configuration=train_args,
        )
        post_args = PostprocessConfiguration(**config['postprocess_configuration'])
        postprocess_data(
          preprocess_data_args=preprocess_args,
          training_outputs=model_output,
          postprocess_configuration=post_args,
        )
    finally:
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
            "id_pre_base": "PyroVelocity_bifurcating_cell1000_gene500",
            "celltype_key": "cell_type",
            "umap_key": "umap"
        },
        {
            "path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "simulate": False,
            "id_pre_base": "PyroVelocity_7",
            "celltype_key": "cell_type",
            "umap_key": "umap"
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
            umap_key = file_info.get("umap_key", "umap")

            try:
                input_file = os.path.basename(file_path)
                id_pre = f"{id_pre_base}_r{run_idx}"
                print(f"  Processing: {input_file}")

                save_dir = os.path.join(OUTPUT_DIR, id_pre)

                main_single(
                    data_template=base_data_template,
                    data_path=file_path, 
                    save_dir=save_dir, 
                    celltype_key=celltype_key, 
                    umap_key=umap_key,
                    seed=seed,
                    simulate=simulate
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