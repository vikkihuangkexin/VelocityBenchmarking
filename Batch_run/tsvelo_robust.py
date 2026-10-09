#!/usr/bin/env python3
"""
Multi-run TSvelo analysis script for RNA velocity prediction.

Requirements:
- TSvelo must be installed (see Benchmarked_tools/TSvelo/README.md) or available
  on PYTHONPATH.
- Input data files should be in the INPUT_DIR.
- Results will be saved to OUTPUT_DIR.

The script dispatches each input to the real-data wrapper (TSvelo.py) or the
simulated-data wrapper (TSvelo_sim.py) according to the ``simulate`` flag, then
saves ``pp.h5ad`` (preprocessed) and ``rc.h5ad`` (velocity) per run.

Configuration:
- Set INPUT_DIR and OUTPUT_DIR via environment variables or modify the defaults below.
- TSVELO_DB_DIR can be set to the folder that holds the ENCODE/ and ChEA/ TF databases.
"""

import datetime
import gc
import importlib.util
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import anndata as ad

# Configuration: Modify these paths or set via environment variables
INPUT_DIR = os.getenv('INPUT_DIR', './example')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './example/output/TSvelo')
TSVELO_DB_DIR = os.getenv('TSVELO_DB_DIR', '/opt/TSvelo')

# Wrappers live next to this repository's Benchmarked_tools/ folder.
BENCHMARKED_DIR = Path(__file__).resolve().parent.parent / 'Benchmarked_tools' / 'TSvelo'


def set_random_seeds():
    """Set random seeds using current timestamp to ensure different results each run."""
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2 ** 32) + os.getpid()

    random.seed(seed)
    np.random.seed(seed)

    return seed


def load_wrapper(module_name):
    """Load a sibling wrapper module without polluting package imports."""
    module_path = BENCHMARKED_DIR / f'{module_name}.py'
    if not module_path.exists():
        raise FileNotFoundError(f'Wrapper not found: {module_path}')
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_once(file_info, save_dir, seed):
    """Run one (dataset, run) pair and write pp.h5ad + rc.h5ad."""
    file_path = file_info['path']
    simulate = file_info['simulate']
    cluster_key = file_info.get('cluster_key', 'vis_annotation' if simulate else 'clusters')

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if simulate:
        wrapper = load_wrapper('TSvelo_sim')
        result = wrapper.run_tsvelo_sim_analysis(
            input_path=file_path,
            output_dir=save_dir,
            cluster_key=cluster_key,
            dataset_name=file_info['id_pre_base'],
            seed=seed,
            overwrite=True,
        )
    else:
        wrapper = load_wrapper('TSvelo')
        result = wrapper.run_tsvelo_analysis(
            input_path=file_path,
            output_dir=save_dir,
            cluster_key=cluster_key,
            dataset_name=file_info['id_pre_base'],
            tsvelo_db_dir=TSVELO_DB_DIR,
            seed=seed,
            overwrite=True,
        )

    result = Path(result)
    dataset_dir = result.parent

    # rc.h5ad = final velocity result; pp.h5ad = preprocessed AnnData.
    rc_path = save_dir / 'rc.h5ad'
    pp_path = save_dir / 'pp.h5ad'

    if pp_path.exists():
        pass
    elif (dataset_dir / 'pp.h5ad').exists():
        shutil.copyfile(dataset_dir / 'pp.h5ad', pp_path)
    else:
        ad.read_h5ad(result).write(pp_path)

    shutil.copyfile(result, rc_path)
    return pp_path, rc_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(OUTPUT_DIR, f'error_log_{timestamp}.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    data_files = [
        {
            'path': os.path.join(INPUT_DIR, 'Simulation-data/bifurcating_cell1000_gene500_dataset.h5ad'),
            'simulate': True,
            'id_pre_base': 'TSvelo_bifurcating_cell1000_gene500',
            'cluster_key': 'vis_annotation',
        },
        {
            'path': os.path.join(INPUT_DIR, 'Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad'),
            'simulate': False,
            'id_pre_base': 'TSvelo_7',
            'cluster_key': 'clusters',
        },
    ]

    n_runs = 5

    for run_idx in range(1, n_runs + 1):
        # Set different random seed for each run
        seed = set_random_seeds()
        print(f'\n[Run {run_idx}/{n_runs}] Seed: {seed}')

        for file_info in data_files:
            file_path = file_info['path']
            id_pre_base = file_info['id_pre_base']

            try:
                input_file = os.path.basename(file_path)
                id_pre = f'{id_pre_base}_r{run_idx}'
                save_dir = os.path.join(OUTPUT_DIR, id_pre)

                print(f'  Processing: {input_file}')

                pp_path, rc_path = run_once(file_info, save_dir, seed)

                print(f'  Saved: {rc_path}')
                logging.info(f'Success: {input_file} run {run_idx}')

                gc.collect()

            except Exception as e:
                logging.error(f'Error processing {file_path} run {run_idx}: {str(e)}', exc_info=True)
                gc.collect()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted')
    except Exception as e:
        logging.error(f'Unhandled error: {e}', exc_info=True)
