#!/usr/bin/env python3
"""
Multi-run InterVelo robustness driver for RNA velocity prediction.

Requirements:
- InterVelo must be installed (clone https://github.com/sd68515/InterVelo_py311.git
  and install with `pip install --no-deps`) or otherwise available in PYTHONPATH
- Input data files should be placed under INPUT_DIR
- Results are written to OUTPUT_DIR

Configuration:
- Set INPUT_DIR and OUTPUT_DIR via environment variables or edit the defaults.
"""

import datetime
import gc
import logging
import os
import random
import sys

import numpy as np
import scanpy as sc
import scvelo as scv
import torch
from scipy import sparse

from InterVelo._utils import autoset_coeff_s, update_dict
from InterVelo.data import preprocess_data
from InterVelo.train import Constants, train

import matplotlib

matplotlib.use("AGG")

# Default to GPU 0, but respect an externally pinned CUDA_VISIBLE_DEVICES.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Configuration: edit these paths or set them through environment variables.
INPUT_DIR = os.getenv("INPUT_DIR", "./example")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./example/output/InterVelo")


def set_random_seeds():
    """Set random seeds using the current timestamp to vary results per run."""
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def to_dense_float32(layer):
    """Densify a (possibly sparse) layer and cast it to float32."""
    if sparse.issparse(layer):
        layer = layer.toarray()
    return np.asarray(layer, dtype=np.float32)


def preprocess(file_path, simulate, extra_layers=None):
    """Load a dataset and prepare the `Ms`/`Mu` moments used by InterVelo.

    When `extra_layers` is given (for example `["Mc"]` or `["Ma"]`), those
    cell-aligned layers are carried through `preprocess_data` so that the model
    input becomes `[Ms, Mu, O]` for multi-omic datasets.
    """
    print(
        "---------------------------------- preprocess "
        f"{os.path.basename(file_path)} ---------------------------------------------"
    )
    adata = sc.read(file_path)

    if simulate:
        adata.obs_names_make_unique()
        adata.obs["milestone"] = "milestone"
        if "X_dimred" in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_dimred"].copy()

    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    missing_layers = [layer for layer in ("spliced", "unspliced") if layer not in adata.layers]
    if missing_layers:
        raise ValueError(f"Missing required layers: {missing_layers}")

    extra_layers = list(extra_layers or [])
    missing_extra = [layer for layer in extra_layers if layer not in adata.layers]
    if missing_extra:
        raise ValueError(f"Missing extra layers: {missing_extra}")

    if simulate:
        scv.pp.filter_and_normalize(
            adata,
            min_shared_counts=None,
            n_top_genes=min(2000, adata.n_vars),
        )
    else:
        scv.pp.filter_and_normalize(adata, min_shared_counts=20)

    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars), subset=True)

    n_pcs = max(1, min(30, adata.n_obs - 1, adata.n_vars - 1))
    n_neighbors = max(1, min(30, adata.n_obs - 1))
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)

    adata = preprocess_data(adata, layers=["Ms", "Mu", *extra_layers], filter_on_r2=False)

    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata)

    return adata


def main(adata, result_path, extra_layers=None, n_latent=20):
    """Train InterVelo and store the velocity in adata.layers['velocity']."""
    print("--------------------------------")

    input_layers = ["Ms", "Mu", *(extra_layers or [])]
    tensors = [torch.from_numpy(to_dense_float32(adata.layers[layer])) for layer in input_layers]
    inputdata = torch.cat(tensors, dim=1)

    configs = {
        "name": "InterVelo_robust",
        "n_gpu": 1 if torch.cuda.is_available() else 0,
        "loss_pearson": {"coeff_s": autoset_coeff_s(adata)},
        "arch": {"args": {"n_latent": n_latent, "pred_unspliced": False}},
        "data_loader": {
            "args": {
                "batch_size": max(1, min(1024, adata.n_obs)),
                "num_workers": 0,
                "validation_split": 0.1 if adata.n_obs >= 10 else 0.0,
            }
        },
        "trainer": {
            "tensorboard": False,
            "verbosity": 0,
            "save_dir": os.path.join(result_path, "saved"),
        },
    }
    configs = update_dict(Constants.default_configs, configs)

    train(adata, inputdata, configs)

    if "velocity_graph" not in adata.uns:
        scv.tl.velocity_graph(adata)
    scv.tl.velocity_embedding(adata, basis="umap", vkey="velocity")

    return True


def main_loop():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"error_log_{timestamp}.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    data_files = [
        {
            "path": os.path.join(INPUT_DIR, "Simulation-data/bifurcating_cell1000_gene500_dataset.h5ad"),
            "simulate": True,
            "id_pre_base": "InterVelo_bifurcating_cell1000_gene500",
            # Add auxiliary layers (for example ["Mc"]) for multi-omic datasets.
            "extra_layers": [],
        },
        {
            "path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "simulate": False,
            "id_pre_base": "InterVelo_7",
            "extra_layers": [],
        },
    ]

    n_runs = 5

    for run_idx in range(1, n_runs + 1):
        seed = set_random_seeds()
        print(f"\n[Run {run_idx}/{n_runs}] Seed: {seed}")

        for file_info in data_files:
            file_path = file_info["path"]
            simulate = file_info["simulate"]
            id_pre_base = file_info["id_pre_base"]
            extra_layers = file_info.get("extra_layers", [])

            adata = None

            try:
                input_file = os.path.basename(file_path)
                id_pre = f"{id_pre_base}_r{run_idx}"

                print(f"  Processing: {input_file}")

                adata = preprocess(file_path, simulate, extra_layers=extra_layers)

                result_path = os.path.join(OUTPUT_DIR, id_pre)
                os.makedirs(result_path, exist_ok=True)

                adata.write(os.path.join(result_path, "pp.h5ad"))

                success = main(adata, result_path, extra_layers=extra_layers)
                if success:
                    adata.write(os.path.join(result_path, "rc.h5ad"))
                    print(f"  Saved: {id_pre}")
                    logging.info(f"Success: {input_file} run {run_idx}")
                else:
                    print(f"  Failed: {id_pre}")
                    logging.error(f"Failed: {input_file} run {run_idx}")

                gc.collect()

            except Exception as exc:
                logging.error(
                    f"Error processing {file_path} run {run_idx}: {str(exc)}",
                    exc_info=True,
                )


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as exc:
        logging.error(f"Unhandled error: {exc}", exc_info=True)
        sys.exit(1)
