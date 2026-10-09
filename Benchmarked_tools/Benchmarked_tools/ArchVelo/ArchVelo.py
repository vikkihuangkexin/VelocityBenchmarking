#!/usr/bin/env python3
"""
ArchVelo velocity analysis pipeline for VelocityBenchmarking.

ArchVelo infers RNA velocity from paired scRNA-seq and scATAC-seq profiles. It
extracts a set of shared archetypal chromatin accessibility programs and models
their regulatory influence on transcription, producing a velocity field that is
decomposed across archetypes.

The wrapper keeps the multi-omic pipeline of the original driver:
  1. build an RNA object and a chromatin object from the same AnnData
     (chromatin is read from an auxiliary layer such as ``Mc``),
  2. run archetypal analysis on the chromatin matrix,
  3. fit the multi-omic chromatin model and the ArchVelo regression model,
  4. export the velocity into ``layers['velocity']`` (native ArchVelo keys are
     preserved).

Installation:
    pip install git+https://github.com/pritykinlab/ArchVelo.git
    # ArchVelo's own dependency list pins the required MultiVelo fork
    # (github.com/MariaAvdeeva/MultiVelo-for-ArchVelo), which pip resolves
    # automatically. A CUDA-enabled PyTorch build (torch==2.3.1) is pulled in
    # by the ArchVelo requirements.

Usage:
    python ArchVelo.py --input multiome.h5ad --output-dir ./output --cluster-key celltype
    python ArchVelo.py --input sim.h5ad --output-dir ./output --cluster-key milestone --simulate
    python ArchVelo.py --input multiome.h5ad --output-dir ./output --cluster-key celltype \
        --dimred-key X_umap --extra-layers Mc
    python ArchVelo.py --metadata-file datasets.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
import gc
import importlib
import logging
import os
import random
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy import sparse


@lru_cache(maxsize=1)
def load_archvelo_api():
    """
    Load the installed ArchVelo package without being shadowed by this file.

    This module is named ``ArchVelo.py`` while the package is also called
    ``ArchVelo``; the script directory is therefore temporarily removed from
    ``sys.path`` so that ``import ArchVelo`` resolves to the installed package.
    """
    script_dir = str(Path(__file__).resolve().parent)
    removed_path = False
    if script_dir in sys.path:
        sys.path.remove(script_dir)
        removed_path = True

    local_module = None
    restore_local_module = __name__ == "ArchVelo" and "ArchVelo" in sys.modules
    if restore_local_module:
        local_module = sys.modules.pop("ArchVelo")

    try:
        archvelo = importlib.import_module("ArchVelo")
    finally:
        if restore_local_module and local_module is not None:
            sys.modules["ArchVelo"] = local_module
        if removed_path:
            sys.path.insert(0, script_dir)

    return archvelo


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


def parse_layer_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value).replace(";", ",").split(",")
    return [str(layer).strip() for layer in values if str(layer).strip()]


def cleanup_resources() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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


def setup_logger(output_dir: Path) -> logging.Logger:
    """Create a compact English logger that writes under ``<output-dir>/log_file``."""
    log_dir = output_dir / "log_file"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ArchVelo")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_dir / "archvelo_run.log", mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def to_dense_float32(layer) -> np.ndarray:
    if sparse.issparse(layer):
        return layer.toarray().astype(np.float32, copy=False)
    return np.asarray(layer, dtype=np.float32)


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


def resolve_dimred(adata, dimred_key: str) -> None:
    """Make sure ``obsm['X_umap']`` exists for the plots.

    Mirrors the original ``ArchVelo_sim.py`` line
    ``adata.obsm['X_umap'] = adata.obsm['X_tsne'].copy()``: when the requested key is
    missing but ``X_tsne`` is present, the latter is remapped to ``X_umap``. Computing a
    UMAP is kept only as a last-resort fallback.
    """
    if dimred_key != "X_umap" and dimred_key in adata.obsm:
        adata.obsm["X_umap"] = np.asarray(adata.obsm[dimred_key]).copy()
    elif "X_umap" in adata.obsm:
        return
    elif "X_tsne" in adata.obsm:
        adata.obsm["X_umap"] = np.asarray(adata.obsm["X_tsne"]).copy()
    elif "X_dimred" in adata.obsm:
        adata.obsm["X_umap"] = np.asarray(adata.obsm["X_dimred"]).copy()
    else:
        print(f"  Computing UMAP because none of '{dimred_key}', 'X_tsne' or 'X_umap' was found...")
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)


def preprocess_adata(
    adata,
    cluster_key: str,
    dimred_key: str = "X_umap",
    extra_layers: Optional[list[str] | str] = None,
    simulate: bool = False,
):
    """
    Prepare the paired RNA/chromatin objects required by ArchVelo.

    Returns ``(adata_rna, adata_atac, atac_layer)`` where ``adata_atac`` carries
    the chromatin accessibility matrix in both ``.X`` and the layer used by
    ArchVelo (``Mc``).
    """
    extra_layers = parse_layer_list(extra_layers)
    atac_layer = extra_layers[0] if extra_layers else "Mc"

    missing = [layer for layer in ("spliced", "unspliced") if layer not in adata.layers]
    if missing:
        raise ValueError(f"Missing required RNA layers: {missing}")
    if atac_layer not in adata.layers:
        raise ValueError(
            f"Missing chromatin accessibility layer '{atac_layer}'. "
            "Pass it through --extra-layers (for example --extra-layers Mc)."
        )

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    if cluster_key not in adata.obs.columns:
        if simulate:
            adata.obs[cluster_key] = "milestone"
        else:
            raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")
    if simulate:
        adata.obs[cluster_key] = "milestone"
    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)

    resolve_dimred(adata, dimred_key)

    adata_rna = adata.copy()
    adata_rna.obs["celltype_new"] = adata_rna.obs[cluster_key].copy()

    adata_atac = adata.copy()
    if atac_layer != "Mc":
        adata_atac.layers["Mc"] = adata_atac.layers[atac_layer].copy()
    adata_atac.X = to_dense_float32(adata_atac.layers["Mc"])
    adata_atac.layers["pearson"] = adata_atac.X.copy()

    return adata_rna, adata_atac, atac_layer


def ensure_moments(adata_rna, n_pcs: int, n_neighbors: int) -> None:
    """Compute the ``Ms``/``Mu`` moment layers when they are not already present."""
    if "Ms" in adata_rna.layers and "Mu" in adata_rna.layers:
        return
    scv.pp.moments(adata_rna, n_pcs=n_pcs, n_neighbors=n_neighbors)


def run_archvelo_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    cluster_key: str,
    dataset_name: Optional[str] = None,
    dimred_key: str = "X_umap",
    extra_layers: Optional[list[str] | str] = None,
    simulate: bool = False,
    num_comps: int = 10,
    n_jobs: int = 10,
    n_neighbors: int = 50,
    n_pcs: int = 30,
    overwrite: bool = False,
    seed: int = 2024,
) -> Path:
    """Run the full ArchVelo pipeline for a single multi-omic dataset."""
    av = load_archvelo_api()

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    extra_layers = parse_layer_list(extra_layers)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if dataset_name is None:
        dataset_name = derive_output_stem(input_path)

    dataset_output_dir = output_dir / str(dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    output_stem = derive_output_stem(input_path)
    # Native ArchVelo file name, preserved from code/ArchVelo_sim.py.
    output_h5ad = dataset_output_dir / f"{output_stem}_ArchVelo.h5ad"

    if output_h5ad.exists() and not overwrite:
        print(f"Skipping existing output: {output_h5ad}")
        return output_h5ad

    seed_everything(seed)

    adata = None
    adata_rna = None
    adata_atac = None
    avel = None

    try:
        print(f"\nProcessing: {input_path.name}")
        logger.info("Starting ArchVelo for %s (seed=%s)", input_path, seed)

        adata = sc.read(input_path)
        adata_rna, adata_atac, atac_layer = preprocess_adata(
            adata,
            cluster_key=cluster_key,
            dimred_key=dimred_key,
            extra_layers=extra_layers,
            simulate=simulate,
        )

        arch_dir = dataset_output_dir / "archetypes"
        model_outdir = dataset_output_dir / "modeling_results"
        arch_dir.mkdir(parents=True, exist_ok=True)
        model_outdir.mkdir(parents=True, exist_ok=True)

        print(f"  Applying archetypal analysis on '{atac_layer}' (k={num_comps})...")
        av.apply_AA_no_test(adata_atac, k=num_comps, outdir=str(arch_dir))

        cell_on_comps = arch_dir / f"cell_on_peaks_{num_comps}_comps.csv"
        peak_on_comps = arch_dir / f"peak_on_peaks_{num_comps}_comps.csv"
        XC_raw = pd.read_csv(cell_on_comps, index_col=[0])
        gene_weights = pd.read_csv(peak_on_comps, index_col=[0])

        smooth_arch = sc.AnnData(
            X=XC_raw.values,
            obs=pd.DataFrame(index=XC_raw.index),
            var=pd.DataFrame(index=XC_raw.columns),
        )
        smooth_arch.layers["Mc"] = XC_raw.values.copy()
        smooth_arch.layers["spliced"] = XC_raw.values.copy()

        # ensure_moments() is an addition relative to ArchVelo_sim.py, which assumes the
        # Ms/Mu moment layers already exist.
        sc.pp.neighbors(adata_rna, n_neighbors=30, n_pcs=30)
        ensure_moments(adata_rna, n_pcs=30, n_neighbors=30)

        print("  Fitting the ArchVelo model...")
        avel = av.apply_ArchVelo_full(
            adata_rna,
            adata_atac,
            smooth_arch,
            gene_weights,
            str(model_outdir) + os.sep,
            n_jobs=n_jobs,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
        )

        avel.layers["velocity"] = np.asarray(avel.layers["velo_s"], dtype=np.float32).copy()

        if "velocity" not in avel.layers:
            raise RuntimeError("ArchVelo did not produce layers['velocity'].")
        avel.uns["archvelo_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(input_path.resolve()),
            "cluster_key": cluster_key,
            "dimred_key": dimred_key,
            "extra_layers": list(extra_layers),
            "simulate": bool(simulate),
            "num_comps": int(num_comps),
            "output_path": str(output_h5ad.resolve()),
        }

        avel.write(output_h5ad)
        shutil.copyfile(output_h5ad, dataset_output_dir / "rc.h5ad")
        logger.info("Finished ArchVelo for %s -> %s", input_path, output_h5ad)
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    except Exception as exc:
        logger.exception("ArchVelo failed for %s: %s", input_path, exc)
        raise
    finally:
        del adata
        del adata_rna
        del adata_atac
        del avel
        cleanup_resources()


def load_metadata_file(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path, sep=detect_separator(metadata_path))

    required_columns = ["dataset_name", "file_path", "cluster_key"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "dimred_key" not in df.columns:
        df["dimred_key"] = "X_umap"
    if "extra_layers" not in df.columns:
        df["extra_layers"] = "Mc"
    if "simulate" not in df.columns:
        df["simulate"] = False
    if "num_comps" not in df.columns:
        df["num_comps"] = np.nan

    df["dataset_name"] = df["dataset_name"].astype(str)
    df["file_path"] = df["file_path"].astype(str)
    df["cluster_key"] = df["cluster_key"].astype(str)
    df["dimred_key"] = df["dimred_key"].astype(str)
    df["extra_layers"] = df["extra_layers"].map(parse_layer_list)
    df["simulate"] = df["simulate"].map(parse_bool)

    return df


def run_batch_archvelo(
    metadata_file: str | Path,
    output_dir: str | Path,
    num_comps: int = 10,
    n_jobs: int = 10,
    n_neighbors: int = 50,
    n_pcs: int = 30,
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

        row_num_comps = row["num_comps"]
        if isinstance(row_num_comps, float) and np.isnan(row_num_comps):
            row_num_comps = num_comps
        else:
            row_num_comps = int(row_num_comps)

        try:
            output_path = run_archvelo_analysis(
                input_path=file_path,
                output_dir=output_dir,
                cluster_key=row["cluster_key"],
                dataset_name=row["dataset_name"],
                dimred_key=row["dimred_key"],
                extra_layers=row["extra_layers"],
                simulate=bool(row["simulate"]),
                num_comps=row_num_comps,
                n_jobs=n_jobs,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs,
                overwrite=overwrite,
                seed=seed,
            )
            outputs.append(output_path)
        except Exception as exc:
            print(f"Failed: {row['dataset_name']}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ArchVelo multi-omic (scRNA + scATAC) velocity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input H5AD file containing RNA layers and an auxiliary chromatin layer")
    input_group.add_argument("--metadata-file", help="Metadata CSV/TSV file for batch processing")

    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", default=None, help="Dataset folder name for single-file mode")
    parser.add_argument(
        "--cluster-key",
        default="celltype",
        help="Column name in adata.obs used for labels and visualization",
    )
    parser.add_argument(
        "--dimred-key",
        default="X_umap",
        help="Dimensionality reduction key in adata.obsm; use X_umap for real data and X_dimred for simulated data",
    )
    parser.add_argument(
        "--extra-layers",
        default="Mc",
        help="Comma-separated adata.layers keys; the first entry is the chromatin accessibility layer used by ArchVelo",
    )
    parser.add_argument(
        "--num-comps",
        type=int,
        default=10,
        help="Number of archetypes extracted from the chromatin accessibility matrix",
    )
    parser.add_argument("--n-jobs", type=int, default=10, help="Number of parallel jobs for model fitting")
    parser.add_argument("--n-neighbors", type=int, default=50, help="Number of neighbors for ArchVelo model fitting")
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of principal components for ArchVelo model fitting")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Treat the input as simulated data: set the cluster label to 'milestone' and mirror the requested embedding",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_archvelo(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            num_comps=args.num_comps,
            n_jobs=args.n_jobs,
            n_neighbors=args.n_neighbors,
            n_pcs=args.n_pcs,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    return run_archvelo_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        dimred_key=args.dimred_key,
        extra_layers=args.extra_layers,
        simulate=args.simulate,
        num_comps=args.num_comps,
        n_jobs=args.n_jobs,
        n_neighbors=args.n_neighbors,
        n_pcs=args.n_pcs,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
