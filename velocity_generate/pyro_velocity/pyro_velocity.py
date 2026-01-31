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
import argparse


def bool_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:bool', 'True' if data else 'False')
yaml.add_representer(bool, bool_representer)

data = {
    'download_dataset': {
        'data_set_name': 'velo',
        'data_external_path': '/data/twang15/velo/pyrovelo/results/9_mouse_Adult_testis_with_celltype',
        'data_url': 'None',
        'n_obs': 'None',
        'n_vars': 'None'
    },
    'preprocess_data': {
        'data_set_name': 'velo',
        'adata': '/data/twang15/velo/data/9_mouse_Adult_testis_with_celltype/9_mouse_Adult_testis_GSE109033_250105.h5ad',
        'data_processed_path': '/data/twang15/velo/pyrovelo/results/9_mouse_Adult_testis_with_celltype/processed',
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
        'adata': '/data/twang15/velo/data/9_mouse_Adult_testis_with_celltype/9_mouse_Adult_testis_GSE109033_250105.h5ad',
        # 'save_dir': '/data/twang15/velo/pyrovelo/results/9_mouse_Adult_testis_with_celltype',
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
        # 'seed': 99,
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
    'training_configuration_2': {
        'adata': '/data/twang15/velo/data/9_mouse_Adult_testis_with_celltype/9_mouse_Adult_testis_GSE109033_250105.h5ad',
        'data_set_name': 'velo',
        'model_identifier': 'model2',
        'guide_type': 'auto',
        'model_type': 'auto',
        'batch_size': -1,
        'use_gpu': 'auto',
        'likelihood': 'Poisson',
        'num_samples': 30,
        'log_every': 100,
        'patient_improve': 0.0001,
        'patient_init': 45,
        'seed': 99,
        'learning_rate': 0.01,
        'max_epochs': 300,
        'include_prior': True,
        'library_size': True,
        'offset': True,
        'input_type': 'raw',
        'cell_specific_kinetics': 'None',
        'kinetics_num': 2,
        'force': False
    },
    'postprocess_configuration': {
        'number_posterior_samples': 4
    },
    'training_resources_requests': {
        'cpu': '64',
        'mem': '50Gi',
        'gpu': '1',
        'ephemeral_storage': '50Gi'
    },
    'training_resources_limits': {
        'cpu': '64',
        'mem': '50Gi',
        'gpu': '1',
        'ephemeral_storage': '50Gi'
    },
    'postprocessing_resources_requests': {
        'cpu': '64',
        'mem': '50Gi',
        'gpu': '0',
        'ephemeral_storage': '50Gi'
    },
    'postprocessing_resources_limits': {
        'cpu': '64',
        'mem': '50Gi',
        'gpu': '0',
        'ephemeral_storage': '50Gi'
    },
    'summarizing_resources_requests': {
        'cpu': '64',
        'mem': '30Gi',
        'gpu': '0',
        'ephemeral_storage': '50Gi'
    },
    'summarizing_resources_limits': {
        'cpu': '64',
        'mem': '50Gi',
        'gpu': '0',
        'ephemeral_storage': '50Gi'
    },
    'upload_results': False
}

_original_torch_load = torch.load

def torch_load_patch(*args, **kwargs):
    # 强制 weights_only=False
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = torch_load_patch

os.environ["FLYTE_SDK_TYPE_CHECK"] = "False"

def generate_yaml(data, data_path, save_dir, celltype_key, umap_key, simulate=False, n_top_genes=2000, min_shared_counts=5, max_epochs=300, batch_size=-1):
    os.makedirs(save_dir, exist_ok=True)
    data['download_dataset']['data_external_path'] = save_dir
    data['preprocess_data']['adata'] = data_path
    data['preprocess_data']['data_processed_path'] = f'{save_dir}/processed'
    data['preprocess_data']['process_cytotrace'] = False
    data['preprocess_data']['use_obs_subset'] = False
    data['training_configuration_1']['adata'] = data_path
    data['preprocess_data']['random_seed'] = 42
    data['preprocess_data']['selected_genes'] = ""
    data['preprocess_data']['vector_field_basis'] = umap_key
    data['preprocess_data']['cell_state'] = celltype_key
    data['preprocess_data']['reports_processed_path'] = f'{save_dir}/reports/processed'

    # Simulation mode adjustments
    if simulate:
        print("[INFO] Simulation mode: relaxing preprocessing and setting cluster labels to 'milestone'")
        # Load adata to modify
        adata = sc.read_h5ad(data_path)
        adata.obs[celltype_key] = "milestone"
        if "X_dimred" in adata.obsm and "X_umap" not in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_dimred"]
        # Relax preprocessing
        min_shared_counts = None
        if "top_genes" in adata.uns:
            n_top_genes = len(adata.uns["top_genes"])
        # Save modified adata temporarily
        temp_path = f'{save_dir}/temp_adata.h5ad'
        adata.write_h5ad(temp_path)
        data['preprocess_data']['adata'] = temp_path
        data['training_configuration_1']['adata'] = temp_path

    data['preprocess_data']['n_top_genes'] = n_top_genes
    data['preprocess_data']['min_shared_counts'] = min_shared_counts
    data['training_configuration_1']['max_epochs'] = max_epochs
    data['training_configuration_1']['batch_size'] = batch_size

    with open(f'{save_dir}/config.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def main(data, data_path, save_dir, celltype_key, umap_key, simulate=False, n_top_genes=2000, min_shared_counts=5, max_epochs=300, batch_size=-1):
    generate_yaml(data, data_path, save_dir, celltype_key, umap_key, simulate, n_top_genes, min_shared_counts, max_epochs, batch_size)
    with open(f'{save_dir}/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    data = config['preprocess_data']['adata']
    os.chdir(save_dir)
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
    postprocessing_outputs = postprocess_data(
      preprocess_data_args=preprocess_args,
      training_outputs=model_output,
      postprocess_configuration=post_args,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyroVelocity training and velocity computation")
    parser.add_argument("--data_path", required=True, help="Path to input .h5ad file")
    parser.add_argument("--save_dir", required=True, help="Directory to save results")
    parser.add_argument("--celltype_key", default="cell_type", help="obs column for cell type (default: cell_type)")
    parser.add_argument("--umap_key", default="umap", help="obsm key for UMAP coordinates (default: umap)")
    parser.add_argument("--simulate", action="store_true", help="If set, relax preprocessing (min_shared_counts=None), set clusters to 'milestone', and copy X_dimred to X_umap if present")
    parser.add_argument("--n_top_genes", type=int, default=2000, help="Number of top genes to select (default: 2000)")
    parser.add_argument("--min_shared_counts", type=int, default=5, help="Minimum shared counts for gene filtering (default: 5)")
    parser.add_argument("--max_epochs", type=int, default=300, help="Maximum training epochs (default: 300)")
    parser.add_argument("--batch_size", type=int, default=-1, help="Batch size for training (default: -1, auto)")
    args = parser.parse_args()
    main(data, args.data_path, args.save_dir, args.celltype_key, args.umap_key, args.simulate, args.n_top_genes, args.min_shared_counts, args.max_epochs, args.batch_size)