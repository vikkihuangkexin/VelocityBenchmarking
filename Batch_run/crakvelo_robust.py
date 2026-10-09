#!/usr/bin/env python3
"""
Multi-run CRAK-Velo analysis script for chromatin-informed RNA velocity prediction.

Requirements:
- The upstream CRAK-Velo source tree (crak-velo/main.py) must be available
- Input data files should be in the INPUT_DIR
- Results will be saved to OUTPUT_DIR

Configuration:
- Set INPUT_DIR and OUTPUT_DIR via environment variables or modify defaults below
"""

import datetime
import gc
import importlib.util
import json
import logging
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scanpy as sc
import matplotlib

matplotlib.use("Agg")

# Configuration: Modify these paths or set via environment variables
INPUT_DIR = os.getenv("INPUT_DIR", "./example")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./example/output/CRAK-Velo")

CRAK_MAIN = os.getenv("CRAK_VELO_MAIN", "/opt/CRAK-Velo/crak-velo/main.py")
WINDOW = 10000

DEFAULT_LOGGER_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(message)s"},
        "datetime": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "datetime",
            "filename": "info.log",
            "maxBytes": 99999999,
            "backupCount": 20,
            "encoding": "utf8",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "info_file_handler"]},
}


def load_upstream_module(crak_main):
    """Import the upstream ``main.py`` as a uniquely named module."""
    crak_main = Path(crak_main).resolve()
    crak_dir = str(crak_main.parent)
    added = False
    if crak_dir not in sys.path:
        sys.path.insert(0, crak_dir)
        added = True

    try:
        spec = importlib.util.spec_from_file_location("crak_velo_upstream_main", str(crak_main))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the upstream module from {crak_main}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, crak_dir, added
    except Exception:
        if added and crak_dir in sys.path:
            sys.path.remove(crak_dir)
        raise


def run_upstream_model(config_path, work_dir):
    """
    Invoke the upstream CRAK-Velo pipeline.

    The upstream entry point is imported and called directly; if the import
    fails, ``main.py`` is executed in a subprocess as a fallback.
    """
    crak_main = Path(CRAK_MAIN).resolve()
    if not crak_main.is_file():
        raise FileNotFoundError(
            f"Upstream main.py not found at {crak_main}. Set CRAK_VELO_MAIN to the clone location."
        )

    original_cwd = os.getcwd()
    module = None
    crak_dir = None
    added = False
    try:
        module, crak_dir, added = load_upstream_module(crak_main)
    except Exception as exc:
        print(f"  Direct import of the upstream entry point failed ({exc}); falling back to subprocess.")
        module = None

    try:
        os.chdir(work_dir)
        if module is not None:
            args = SimpleNamespace(config=str(config_path), run_id=None, window=WINDOW)
            config_parser = module.ConfigParser.from_args(args)
            module.run_model(config_parser)
        else:
            cmd = [sys.executable, str(crak_main), "--config", str(config_path), "--w", str(WINDOW)]
            subprocess.run(cmd, cwd=str(work_dir), check=True)
    finally:
        os.chdir(original_cwd)
        if added and crak_dir and crak_dir in sys.path:
            sys.path.remove(crak_dir)


def set_random_seeds():
    """Set random seeds using current timestamp to ensure different results each run"""
    # Use microsecond timestamp and process ID to generate a unique seed
    seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32) + os.getpid()

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    return seed


def slugify(value):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return slug or "dataset"


def preprocess(file_path, cluster_key="cell_type", simulate=False, basis="tsne", work_dir=None):
    print(f"----------------------------------preprocess {os.path.basename(file_path)} ---------------------------------------------")
    adata = sc.read(file_path)

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    if cluster_key not in adata.obs and not simulate:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

    if simulate:
        adata.obs[cluster_key] = "milestone"
        if basis not in adata.obsm and "X_dimred" in adata.obsm:
            adata.obsm[basis] = np.asarray(adata.obsm["X_dimred"]).copy()

    missing_layers = [layer for layer in ("spliced", "unspliced") if layer not in adata.layers]
    if missing_layers:
        raise ValueError(f"Missing required layers: {missing_layers}")

    print(adata)

    if simulate and work_dir is not None:
        os.makedirs(work_dir, exist_ok=True)
        prepared_path = os.path.join(work_dir, "rna_prepared.h5ad")
        adata.write(prepared_path)
        return adata, prepared_path

    return adata, str(file_path)


def main(
    adata,
    rna_path,
    atac_path,
    config_template,
    result_path,
    dataset_name="CRAK-Velo",
    cluster_key="cell_type",
    basis="tsne",
):
    crak_save_dir = os.path.join(result_path, "crak_output")
    os.makedirs(crak_save_dir, exist_ok=True)

    with open(config_template, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    logger_config = Path(CRAK_MAIN).resolve().parent / "config" / "config_logger.json"
    if not logger_config.is_file():
        logger_config = Path(crak_save_dir) / "config_logger.json"
        with logger_config.open("w", encoding="utf-8") as handle:
            json.dump(DEFAULT_LOGGER_CONFIG, handle, indent=4)

    config["name"] = slugify(dataset_name)
    config["logger_config_path"] = str(logger_config)
    config["adata_path"] = str(rna_path)
    config["adata_atac_path"] = str(atac_path)
    config["save_dir"] = str(crak_save_dir)
    config["cluster_name"] = cluster_key
    config.setdefault("system", {})
    config["system"]["seed"] = int(datetime.datetime.now().timestamp()) % (2**31)
    config.setdefault("preprocessing", {})
    config["preprocessing"]["basis"] = basis
    config["preprocessing"]["window"] = WINDOW
    config.setdefault("base_trainer", {})
    config["base_trainer"]["save_dir"] = str(crak_save_dir)

    runtime_config = os.path.join(crak_save_dir, "config.json")
    with open(runtime_config, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)

    run_upstream_model(runtime_config, result_path)

    candidates = sorted(
        Path(crak_save_dir).glob("checkpoints/**/adata_rna_fit.h5ad"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return False

    result = sc.read(candidates[-1])
    if "velocity" not in result.layers:
        return False

    result.write(os.path.join(result_path, "rc.h5ad"))
    return True


def main_batch():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"error_log_{timestamp}.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_files = [
        {
            "rna_path": os.path.join(INPUT_DIR, "Simulation-data/bifurcating_cell1000_gene500_dataset.h5ad"),
            "atac_path": os.path.join(INPUT_DIR, "Simulation-data/bifurcating_cell1000_gene500_atac.h5ad"),
            "config": os.path.join(INPUT_DIR, "config/config_simdata_01.json"),
            "simulate": True,
            "basis": "umap",
            "id_pre_base": "CRAK-Velo_bifurcating_cell1000_gene500",
            "cluster_key": "cell_type"
        },
        {
            "rna_path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188.h5ad"),
            "atac_path": os.path.join(INPUT_DIR, "Real-data/7_mouse_PancreaticE15.5_GSE132188_atac.h5ad"),
            "config": os.path.join(INPUT_DIR, "config/config_main_HSPC.json"),
            "simulate": False,
            "basis": "tsne",
            "id_pre_base": "CRAK-Velo_7",
            "cluster_key": "cell_type"
        }
    ]

    n_runs = 5

    for run_idx in range(1, n_runs + 1):
        # Set different random seed for each run
        seed = set_random_seeds()
        print(f"\n[Run {run_idx}/{n_runs}] Seed: {seed}")

        for file_info in data_files:
            rna_path = file_info["rna_path"]
            atac_path = file_info["atac_path"]
            simulate = file_info["simulate"]
            basis = file_info["basis"]
            id_pre_base = file_info["id_pre_base"]
            cluster_key = file_info["cluster_key"]

            adata = None

            try:
                input_file = os.path.basename(rna_path)
                id_pre = f"{id_pre_base}_r{run_idx}"

                print(f"  Processing: {input_file}")

                result_path = os.path.join(OUTPUT_DIR, id_pre)
                os.makedirs(result_path, exist_ok=True)

                adata, prepared_rna_path = preprocess(
                    rna_path,
                    cluster_key=cluster_key,
                    simulate=simulate,
                    basis=basis,
                    work_dir=os.path.join(result_path, "prepared"),
                )

                adata.write(os.path.join(result_path, "pp.h5ad"))

                success = main(
                    adata,
                    prepared_rna_path,
                    atac_path,
                    file_info["config"],
                    result_path,
                    dataset_name=id_pre,
                    cluster_key=cluster_key,
                    basis=basis,
                )
                if success:
                    print(f"  Saved: {id_pre}")
                    logging.info(f"Success: {input_file} run {run_idx}")
                else:
                    print(f"  Failed: {id_pre}")
                    logging.error(f"Failed: {input_file} run {run_idx}")

                gc.collect()

            except Exception as e:
                logging.error(f"Error processing {rna_path} run {run_idx}: {str(e)}", exc_info=True)


if __name__ == "__main__":
    try:
        main_batch()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logging.error(f"Unhandled error: {e}", exc_info=True)
