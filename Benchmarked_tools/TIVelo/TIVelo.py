#!/usr/bin/env python3
"""
TIVelo velocity analysis pipeline for VelocityBenchmarking.

Real and simulated datasets are processed by the same script and switched with
the --simulate flag:

* real data: standard scVelo filtering/normalization with 2000 top genes,
  PCA, a 30-NN graph, and moment estimation.
* simulated data: uses the pre-computed `obsm['X_dimred']` embedding, applies a
  relaxed gene filter, and labels the cells with the `milestone` cluster.

Datasets whose cluster transition graph is disconnected cannot be processed by
TIVelo; those runs return a `'disconnected'` marker instead of raising.

Installation:
    git clone https://github.com/cuhklinlab/TIVelo.git
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
    pip install scanpy python-igraph leidenalg scvelo==0.3.1 igraph louvain pybind11 optax==0.2.3
    pip install TIVelo

Usage:
    python TIVelo.py --input data.h5ad --output-dir ./output --cluster-key celltype
    python TIVelo.py --input sim.h5ad --output-dir ./output --cluster-key milestone --simulate
    python TIVelo.py --metadata-file datasets.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy.sparse.linalg import ArpackNoConvergence

# Error messages that indicate an unconnected cluster transition graph.
DISCONNECTED_MESSAGES = ("unconnected sub-graphs",)
SIM_DISCONNECTED_MESSAGES = ("unconnected sub-graphs", "max() arg is an empty sequence")


@lru_cache(maxsize=1)
def load_tivelo_api():
    """
    Load the installed `tivelo` package without being shadowed by this file.

    On case-insensitive file systems the sibling `TIVelo.py` can be picked up as
    the `tivelo` module, so the script directory is temporarily removed from
    `sys.path` while importing the real package.
    """
    script_dir = str(Path(__file__).resolve().parent)
    removed_path = False
    if script_dir in sys.path:
        sys.path.remove(script_dir)
        removed_path = True

    local_module = None
    restore_local_module = __name__ == "TIVelo" and "TIVelo" in sys.modules
    if restore_local_module:
        local_module = sys.modules.pop("TIVelo")

    try:
        main_module = importlib.import_module("tivelo.main")
    finally:
        if restore_local_module and local_module is not None:
            sys.modules["TIVelo"] = local_module
        if removed_path:
            sys.path.insert(0, script_dir)

    return main_module.tivelo


def seed_everything(seed: int) -> None:
    """Seed the random number generators used by the pipeline."""
    np.random.seed(seed)


def cleanup_resources() -> None:
    """Release memory held by the last dataset before moving on."""
    gc.collect()
    plt.close("all")


def derive_output_stem(input_path: Path) -> str:
    """Strip the benchmark `_dataset` suffix from an input file stem."""
    stem = input_path.stem
    if stem.endswith("_dataset"):
        return stem[:-8]
    return stem


def parse_bool(value) -> bool:
    """Convert common textual/boolean values into a real bool."""
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
    """Infer the column separator of a metadata table from its extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix in {".tsv", ".txt"}:
        return "\t"

    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    return "\t" if "\t" in first_line else ","


def load_metadata_file(metadata_path: Path) -> pd.DataFrame:
    """Read and validate the batch metadata table."""
    df = pd.read_csv(metadata_path, sep=detect_separator(metadata_path))

    required_columns = ["dataset_name", "file_path", "cluster_key"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "dimred_key" not in df.columns:
        df["dimred_key"] = "X_umap"
    if "simulate" not in df.columns:
        df["simulate"] = False

    df["dataset_name"] = df["dataset_name"].astype(str)
    df["file_path"] = df["file_path"].astype(str)
    df["cluster_key"] = df["cluster_key"].astype(str)
    df["dimred_key"] = df["dimred_key"].astype(str)
    df["simulate"] = df["simulate"].map(parse_bool)

    return df


def preprocess_adata(
    adata,
    cluster_key: str,
    dimred_key: str = "X_umap",
    simulate: bool = False,
    n_top_genes: int = 2000,
    min_shared_counts: int = 20,
    n_pcs: int = 30,
    n_neighbors: int = 30,
):
    """
    Prepare an AnnData object for TIVelo.

    The simulated branch relies on the pre-computed embedding and a relaxed gene
    filter, while the real branch uses the standard scVelo filtering recipe.
    """
    if simulate:
        adata.obs_names_make_unique()
        if "X_dimred" not in adata.obsm:
            raise ValueError("Simulated data requires obsm['X_dimred'].")
        adata.obsm["X_umap"] = adata.obsm["X_dimred"]

        effective_top_genes = n_top_genes
        if adata.n_vars < 2000:
            effective_top_genes = (adata.n_vars // 500) * 500
            effective_top_genes = min(effective_top_genes, adata.n_vars - 1)

        # `tivelo_sim.py` computes a local `shared_counts = 1` but never uses it; the
        # actual call is `filter_and_normalize(..., min_shared_counts=None, ...)`, so the
        # shared-count filter is disabled for simulated data.
        effective_shared_counts = None

        if cluster_key not in adata.obs.columns:
            adata.obs[cluster_key] = "milestone"
    else:
        effective_top_genes = n_top_genes
        effective_shared_counts = min_shared_counts

    print(
        f"  Filtering and normalizing (min_shared_counts={effective_shared_counts}, "
        f"n_top_genes={effective_top_genes})..."
    )
    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=effective_shared_counts,
        n_top_genes=effective_top_genes,
    )
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    return adata


def run_tivelo_model(adata, cluster_key: str, emb_key: str, save_folder: str, data_name: str, simulate: bool,
                     n_epochs: int, batch_size: int, loss_fun: str, filter_genes: bool, constrain: bool,
                     velocity_key: str, show_dti: bool, adjust_dti: bool, measure_performance: bool, n_jobs: int):
    """
    Invoke the upstream TIVelo entry point.

    Returns the annotated AnnData on success or the string `'disconnected'` when
    the cluster transition graph cannot be processed.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    tivelo = load_tivelo_api()

    try:
        adata = tivelo(
            adata,
            group_key=cluster_key,
            emb_key=emb_key,
            data_name=data_name,
            save_folder=save_folder,
            show_fig=False,
            filter_genes=filter_genes,
            constrain=constrain,
            loss_fun=loss_fun,
            batch_size=batch_size,
            n_epochs=n_epochs,
            velocity_key=velocity_key,
            show_DTI=show_dti,
            adjust_DTI=adjust_dti,
            measure_performance=measure_performance,
            njobs=n_jobs,
        )
    except ValueError as exc:
        messages = SIM_DISCONNECTED_MESSAGES if simulate else DISCONNECTED_MESSAGES
        if any(message in str(exc) for message in messages):
            print("  [SKIP] unconnected sub-graphs, TIVelo cannot process this dataset.")
            return "disconnected"
        raise
    except ArpackNoConvergence:
        print("  [SKIP] ARPACK did not converge (disconnected transition matrix).")
        return "disconnected"

    return adata


def run_tivelo_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    cluster_key: str,
    dataset_name: Optional[str] = None,
    dimred_key: str = "X_umap",
    simulate: bool = False,
    n_top_genes: int = 2000,
    min_shared_counts: int = 20,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    n_epochs: int = 100,
    batch_size: int = 1024,
    loss_fun: str = "mse",
    filter_genes: bool = True,
    constrain: bool = True,
    velocity_key: str = "velocity",
    show_dti: bool = False,
    adjust_dti: bool = False,
    measure_performance: bool = False,
    n_jobs: int = -1,
    seed: int = 2024,
    overwrite: bool = False,
) -> Optional[Path]:
    """Run TIVelo on a single dataset and write the annotated H5AD."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if cluster_key is None:
        cluster_key = "milestone" if simulate else "vis_annotation"

    if dataset_name is None:
        dataset_name = derive_output_stem(input_path)

    dataset_output_dir = output_dir / str(dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = derive_output_stem(input_path)
    output_h5ad = dataset_output_dir / f"{output_stem}.h5ad"
    native_output_dir = dataset_output_dir / "native"

    if output_h5ad.exists() and not overwrite:
        print(f"Skipping existing output: {output_h5ad}")
        return output_h5ad

    seed_everything(seed)

    adata = None
    try:
        print(f"\nProcessing: {input_path.name}")
        adata = sc.read(input_path)

        if simulate and cluster_key not in adata.obs.columns:
            adata.obs[cluster_key] = "milestone"
        if cluster_key not in adata.obs.columns:
            raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

        print("  Preprocessing...")
        adata = preprocess_adata(
            adata=adata,
            cluster_key=cluster_key,
            dimred_key=dimred_key,
            simulate=simulate,
            n_top_genes=n_top_genes,
            min_shared_counts=min_shared_counts,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
        )

        print("  Running TIVelo...")
        emb_key = dimred_key if dimred_key in adata.obsm else "X_umap"
        result = run_tivelo_model(
            adata=adata,
            cluster_key=cluster_key,
            emb_key=emb_key,
            save_folder=str(native_output_dir),
            data_name=output_stem,
            simulate=simulate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            loss_fun=loss_fun,
            filter_genes=filter_genes,
            constrain=constrain,
            velocity_key=velocity_key,
            show_dti=show_dti,
            adjust_dti=adjust_dti,
            measure_performance=measure_performance,
            n_jobs=n_jobs,
        )

        if isinstance(result, str):
            return None

        adata = result
        adata.uns["tivelo_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(input_path.resolve()),
            "cluster_key": cluster_key,
            "dimred_key": dimred_key,
            "simulate": bool(simulate),
            "velocity_key": velocity_key,
            "output_path": str(output_h5ad.resolve()),
        }

        if velocity_key not in adata.layers:
            raise RuntimeError(f"TIVelo did not produce layers['{velocity_key}'].")

        adata.write(output_h5ad, compression="lzf")
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata
        cleanup_resources()


def run_batch_tivelo(
    metadata_file: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    seed: int = 2024,
) -> list[Path]:
    """Run TIVelo for every dataset listed in a metadata table."""
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

        try:
            output_path = run_tivelo_analysis(
                input_path=file_path,
                output_dir=output_dir,
                cluster_key=row["cluster_key"],
                dataset_name=row["dataset_name"],
                dimred_key=row["dimred_key"],
                simulate=bool(row["simulate"]),
                overwrite=overwrite,
                seed=seed,
            )
            if output_path is not None:
                outputs.append(output_path)
        except Exception as exc:
            print(f"Failed: {row['dataset_name']}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TIVelo velocity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input H5AD file")
    input_group.add_argument("--metadata-file", help="Metadata CSV/TSV file for batch processing")

    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", default=None, help="Dataset folder name for single-file mode")
    parser.add_argument(
        "--cluster-key",
        default=None,
        help="Column name in adata.obs used for labels; required for real data and defaults to 'milestone' with --simulate",
    )
    parser.add_argument(
        "--dimred-key",
        default="X_umap",
        help="Embedding key in adata.obsm; X_umap for real data, X_dimred for simulated data",
    )
    parser.add_argument("--simulate", action="store_true", default=False, help="Use the simulated-data pipeline")
    parser.add_argument("--n-top-genes", type=int, default=2000, help="Number of highly variable genes")
    parser.add_argument(
        "--min-shared-counts",
        type=int,
        default=20,
        help="Minimum shared counts for the real-data gene filter",
    )
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of principal components for the neighbor graph")
    parser.add_argument("--n-neighbors", type=int, default=30, help="Number of neighbors for the neighbor graph")
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--loss-fun", default="mse", help="Training loss function")
    parser.add_argument(
        "--no-filter-genes",
        dest="filter_genes",
        action="store_false",
        default=True,
        help="Disable the internal gene filter",
    )
    parser.add_argument(
        "--no-constrain",
        dest="constrain",
        action="store_false",
        default=True,
        help="Disable the monotonicity constraint",
    )
    parser.add_argument("--velocity-key", default="velocity", help="Key used to store the velocity layer")
    parser.add_argument("--show-dti", action="store_true", default=False, help="Plot the directed transition graph")
    parser.add_argument("--adjust-dti", action="store_true", default=False, help="Adjust the directed transition graph")
    parser.add_argument(
        "--measure-performance",
        action="store_true",
        default=False,
        help="Compute the internal TIVelo performance metrics",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs, -1 uses all available cores")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_tivelo(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    if not args.cluster_key:
        parser.error("--cluster-key is required in single-file mode")

    return run_tivelo_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        dimred_key=args.dimred_key,
        simulate=args.simulate,
        n_top_genes=args.n_top_genes,
        min_shared_counts=args.min_shared_counts,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        loss_fun=args.loss_fun,
        filter_genes=args.filter_genes,
        constrain=args.constrain,
        velocity_key=args.velocity_key,
        show_dti=args.show_dti,
        adjust_dti=args.adjust_dti,
        measure_performance=args.measure_performance,
        n_jobs=args.n_jobs,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
