#!/usr/bin/env python3
"""
CRAK-Velo velocity analysis pipeline for VelocityBenchmarking.

CRAK-Velo is a semi-mechanistic model that incorporates chromatin accessibility
data into the estimation of RNA velocities. It takes a chromatin accessibility
matrix in addition to spliced/unspliced RNA counts.

The upstream project is a script repository (``crak-velo/main.py``) driven by
JSON config files. This wrapper therefore:

  * builds a runtime config from a base template (``--config``) plus the CLI
    overrides (input paths, output directory, cluster key, seed, window),
  * calls the upstream entry point ``run_model`` directly when the source tree
    can be imported, and falls back to running ``main.py`` in a subprocess,
  * copies the fitted result to ``<output-dir>/<dataset-name>/<stem>.h5ad`` with
    the velocity exported in ``layers['velocity']``.

Installation:
    pip install scanpy==1.8.2 scvelo==0.2.5 anndata==0.8.0 umap-learn==0.5.3 \\
        numba==0.55.2 llvmlite==0.38.1 scipy==1.7.3 unitvelo \\
        matplotlib==3.5.3 pandas==1.5.3 numpy==1.22.4
    conda install -c conda-forge -c bioconda pybedtools
    git clone https://github.com/StatBiomed/CRAK-Velo.git /opt/CRAK-Velo

Usage:
    python CRAK-Velo.py --input rna.h5ad --atac-input atac.h5ad --config config.json \\
        --output-dir ./output --cluster-key celltype
    python CRAK-Velo.py --input rna.h5ad --atac-input atac.h5ad --config config.json \\
        --output-dir ./output --cluster-key milestone --simulate
    python CRAK-Velo.py --metadata-file datasets.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc

DEFAULT_CRAK_MAIN_CANDIDATES = (
    "/opt/CRAK-Velo/crak-velo/main.py",
    "./crak-velo/main.py",
    "./main.py",
)

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


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def detect_separator(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix in {".tsv", ".txt"}:
        return "\t"

    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    return "\t" if "\t" in first_line else ","


def derive_output_stem(input_path: Path) -> str:
    stem = input_path.stem
    if stem.endswith("_dataset"):
        return stem[:-8]
    return stem


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return slug or "dataset"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def cleanup_resources() -> None:
    gc.collect()


def setup_logger(output_dir: Path) -> logging.Logger:
    """Create a compact English logger that writes under ``<output-dir>/log_file``."""
    log_dir = output_dir / "log_file"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("CRAK-Velo")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_dir / "crakvelo_run.log", mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def resolve_crak_main(crak_main: Optional[str | Path] = None) -> Path:
    """Resolve the upstream ``main.py`` from the CLI flag, environment or defaults."""
    candidates = [crak_main, os.environ.get("CRAK_VELO_MAIN"), *DEFAULT_CRAK_MAIN_CANDIDATES]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "Could not locate the upstream CRAK-Velo main.py. Pass --crak-main explicitly "
        "(the clone lives at /opt/CRAK-Velo/crak-velo/main.py in the container) or set CRAK_VELO_MAIN."
    )


def load_upstream_module(crak_main: Path):
    """Import the upstream ``main.py`` as a uniquely named module."""
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


def run_upstream_model(crak_main: Path, config_path: Path, window: Optional[int], work_dir: Path) -> None:
    """
    Invoke the upstream CRAK-Velo pipeline.

    The upstream entry point is imported and called directly; if the import
    fails, ``main.py`` is executed in a subprocess as a fallback.
    """
    original_cwd = os.getcwd()
    module = None
    crak_dir = None
    added = False
    try:
        module, crak_dir, added = load_upstream_module(crak_main)
    except Exception as exc:
        print(f"  Direct import of the upstream entry point failed ({exc}); falling back to subprocess.")
        module = None
        if added and crak_dir and crak_dir in sys.path:
            sys.path.remove(crak_dir)

    try:
        os.chdir(work_dir)
        if module is not None:
            args = SimpleNamespace(config=str(config_path), run_id=None, window=window)
            config_parser = module.ConfigParser.from_args(args)
            module.run_model(config_parser)
        else:
            cmd = [sys.executable, str(crak_main), "--config", str(config_path)]
            if window is not None:
                cmd += ["--w", str(window)]
            subprocess.run(cmd, cwd=str(work_dir), check=True)
    finally:
        os.chdir(original_cwd)
        if added and crak_dir and crak_dir in sys.path:
            sys.path.remove(crak_dir)


def build_runtime_config(
    base_config_path: Path,
    rna_path: Path,
    atac_path: Path,
    save_dir: Path,
    dataset_name: str,
    cluster_key: str,
    crak_main: Path,
    basis: Optional[str],
    window: Optional[int],
    seed: int,
) -> Path:
    """Write a runtime config derived from the base template plus CLI overrides."""
    with base_config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    save_dir.mkdir(parents=True, exist_ok=True)

    logger_config = Path(crak_main).parent / "config" / "config_logger.json"
    if not logger_config.is_file():
        logger_config = save_dir / "config_logger.json"
        with logger_config.open("w", encoding="utf-8") as handle:
            json.dump(DEFAULT_LOGGER_CONFIG, handle, indent=4)

    config["name"] = slugify(dataset_name)
    config["logger_config_path"] = str(logger_config)
    config["adata_path"] = str(rna_path)
    config["adata_atac_path"] = str(atac_path)
    config["save_dir"] = str(save_dir)
    config["cluster_name"] = cluster_key

    config.setdefault("system", {})
    config["system"]["seed"] = int(seed)

    config.setdefault("preprocessing", {})
    if basis:
        config["preprocessing"]["basis"] = basis
    if window is not None:
        config["preprocessing"]["window"] = int(window)

    config.setdefault("base_trainer", {})
    config["base_trainer"]["save_dir"] = str(save_dir)

    runtime_config_path = save_dir / "config.json"
    with runtime_config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)

    return runtime_config_path


def prepare_rna_input(
    input_path: Path,
    cluster_key: str,
    simulate: bool,
    basis: Optional[str],
    work_dir: Path,
) -> Path:
    """
    Validate/adapt the RNA input for CRAK-Velo.

    The simulated branch writes a prepared copy where the cluster label is set to
    ``milestone`` and ``obsm['X_dimred']`` is mirrored into the configured basis.
    """
    if not simulate:
        return input_path

    work_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read(input_path)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    adata.obs[cluster_key] = "milestone"

    if basis and basis not in adata.obsm and "X_dimred" in adata.obsm:
        adata.obsm[basis] = np.asarray(adata.obsm["X_dimred"]).copy()

    prepared_path = work_dir / f"{derive_output_stem(input_path)}_prepared.h5ad"
    adata.write(prepared_path)
    del adata
    gc.collect()
    return prepared_path


def find_fit_output(save_dir: Path, stem: str) -> Path:
    """Locate the RNA fit written by the upstream pipeline and copy it into place."""
    candidates = sorted(save_dir.glob("checkpoints/**/adata_rna_fit.h5ad"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"CRAK-Velo did not produce an RNA fit under {save_dir / 'checkpoints'}")

    target = save_dir / f"{stem}.h5ad"
    shutil.copyfile(candidates[-1], target)
    return target


def run_crakvelo_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    cluster_key: str,
    config_path: Optional[str | Path] = None,
    atac_input: Optional[str | Path] = None,
    dataset_name: Optional[str] = None,
    crak_main: Optional[str | Path] = None,
    basis: Optional[str] = None,
    window: Optional[int] = None,
    simulate: bool = False,
    overwrite: bool = False,
    seed: int = 2024,
) -> Path:
    """Run CRAK-Velo for a single RNA/ATAC dataset pair."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not config_path:
        raise ValueError("--config is required in single-file mode (use one of the JSON templates shipped with CRAK-Velo).")
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config template not found: {config_path}")
    if not atac_input:
        raise ValueError("--atac-input is required in single-file mode (CRAK-Velo needs a chromatin accessibility matrix).")
    atac_path = Path(atac_input)
    if not atac_path.exists():
        raise FileNotFoundError(f"ATAC input file not found: {atac_path}")

    resolved_main = resolve_crak_main(crak_main)

    if dataset_name is None:
        dataset_name = derive_output_stem(input_path)

    dataset_output_dir = output_dir / str(dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    output_stem = derive_output_stem(input_path)
    output_h5ad = dataset_output_dir / f"{output_stem}.h5ad"
    if output_h5ad.exists() and not overwrite:
        print(f"Skipping existing output: {output_h5ad}")
        return output_h5ad

    seed_everything(seed)

    crak_save_dir = dataset_output_dir / "crak_output"
    if overwrite and crak_save_dir.exists():
        shutil.rmtree(crak_save_dir)
    crak_save_dir.mkdir(parents=True, exist_ok=True)

    adata = None
    try:
        print(f"\nProcessing: {input_path.name}")
        logger.info("Starting CRAK-Velo for %s (seed=%s)", input_path, seed)

        rna_path = prepare_rna_input(
            input_path,
            cluster_key=cluster_key,
            simulate=simulate,
            basis=basis,
            work_dir=dataset_output_dir / "prepared",
        )

        runtime_config = build_runtime_config(
            base_config_path=config_path,
            rna_path=rna_path,
            atac_path=atac_path,
            save_dir=crak_save_dir,
            dataset_name=dataset_name,
            cluster_key=cluster_key,
            crak_main=resolved_main,
            basis=basis,
            window=window,
            seed=seed,
        )

        print(f"  Running upstream CRAK-Velo from {resolved_main}...")
        run_upstream_model(resolved_main, runtime_config, window, dataset_output_dir)

        print("  Collecting the fitted result...")
        fit_h5ad = find_fit_output(crak_save_dir, output_stem)

        adata = sc.read(fit_h5ad)
        if "velocity" not in adata.layers:
            raise ValueError("CRAK-Velo finished without producing layers['velocity'].")

        adata.uns["crakvelo_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(input_path.resolve()),
            "atac_input_path": str(atac_path.resolve()),
            "config_path": str(config_path.resolve()),
            "cluster_key": cluster_key,
            "simulate": bool(simulate),
            "window": int(window) if window is not None else -1,
            "output_path": str(output_h5ad.resolve()),
        }

        adata.write(output_h5ad)
        shutil.copyfile(output_h5ad, dataset_output_dir / "rc.h5ad")
        logger.info("Finished CRAK-Velo for %s -> %s", input_path, output_h5ad)
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    except Exception as exc:
        logger.exception("CRAK-Velo failed for %s: %s", input_path, exc)
        raise
    finally:
        del adata
        cleanup_resources()


def load_metadata_file(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path, sep=detect_separator(metadata_path))

    required_columns = ["dataset_name", "file_path", "cluster_key"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "atac_file_path" not in df.columns:
        df["atac_file_path"] = None
    if "config" not in df.columns:
        df["config"] = None
    if "basis" not in df.columns:
        df["basis"] = None
    if "window" not in df.columns:
        df["window"] = np.nan
    if "simulate" not in df.columns:
        df["simulate"] = False

    df["dataset_name"] = df["dataset_name"].astype(str)
    df["file_path"] = df["file_path"].astype(str)
    df["cluster_key"] = df["cluster_key"].astype(str)
    df["simulate"] = df["simulate"].map(parse_bool)
    return df


def run_batch_crakvelo(
    metadata_file: str | Path,
    output_dir: str | Path,
    config_path: Optional[str | Path] = None,
    crak_main: Optional[str | Path] = None,
    basis: Optional[str] = None,
    window: Optional[int] = None,
    overwrite: bool = False,
    seed: int = 2024,
) -> list[Path]:
    """Process every dataset listed in a metadata CSV/TSV file."""
    metadata_file = Path(metadata_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_file(metadata_file)
    outputs: list[Path] = []
    print(f"Batch mode: {len(metadata_df)} datasets")

    for _, row in metadata_df.iterrows():
        file_path = Path(row["file_path"])
        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        row_window = row["window"]
        row_window = window if (isinstance(row_window, float) and np.isnan(row_window)) else int(row_window)

        try:
            output_path = run_crakvelo_analysis(
                input_path=file_path,
                output_dir=output_dir,
                cluster_key=row["cluster_key"],
                config_path=row["config"] or config_path,
                atac_input=row["atac_file_path"],
                dataset_name=row["dataset_name"],
                crak_main=crak_main,
                basis=row["basis"] or basis,
                window=row_window,
                simulate=bool(row["simulate"]),
                overwrite=overwrite,
                seed=seed,
            )
            outputs.append(output_path)
        except Exception as exc:
            print(f"Failed: {row['dataset_name']}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CRAK-Velo chromatin-informed RNA velocity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="RNA H5AD file for single-file mode")
    input_group.add_argument("--metadata-file", help="Metadata CSV/TSV file for batch processing")

    parser.add_argument("--atac-input", default=None, help="ATAC H5AD file for single-file mode")
    parser.add_argument(
        "--config",
        default=None,
        help="Base CRAK-Velo JSON config template (one of the files shipped in crak-velo/config)",
    )
    parser.add_argument(
        "--crak-main",
        default=None,
        help="Path to the upstream crak-velo/main.py (defaults to /opt/CRAK-Velo/crak-velo/main.py or CRAK_VELO_MAIN)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", default=None, help="Dataset folder name for single-file mode")
    parser.add_argument(
        "--cluster-key",
        default="celltype",
        help="Column name in adata.obs holding the cluster/cell-type labels",
    )
    parser.add_argument(
        "--basis",
        default=None,
        help="Embedding basis used by the upstream model (overrides preprocessing.basis in the config, e.g. tsne or umap)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10000,
        help="Window length used to intersect ATAC regions with genes",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Simulated-data branch: label cells 'milestone' and mirror obsm['X_dimred'] into the configured basis",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_crakvelo(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            config_path=args.config,
            crak_main=args.crak_main,
            basis=args.basis,
            window=args.window,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    if not args.cluster_key:
        parser.error("--cluster-key is required in single-file mode")

    return run_crakvelo_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        config_path=args.config,
        atac_input=args.atac_input,
        dataset_name=args.dataset_name,
        crak_main=args.crak_main,
        basis=args.basis,
        window=args.window,
        simulate=args.simulate,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
