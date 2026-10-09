#!/usr/bin/env python3
"""
DeepKINET velocity analysis pipeline for VelocityBenchmarking.

DeepKINET is a deep generative model that estimates mRNA splicing and
degradation kinetics at the single-cell level. It requires the *raw integer*
spliced/unspliced counts as input, so this wrapper restores the raw count
layers after the scVelo gene filtering step.

Installation:
    git clone https://github.com/3254c/DeepKINET.git
    cd DeepKINET
    pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    pip install -e .
    pip install anndata==0.10.3 scanpy==1.9.6 scvelo==0.2.5
    pip install pandas==1.5.3 numpy==1.23.5 einops==0.7.0
    pip install leidenalg==0.10.1 umap-learn==0.5.5
    pip install matplotlib==3.7.1 seaborn==0.12.2

Usage:
    python DeepKINET.py --input data.h5ad --output-dir ./output --cluster-key celltype
    python DeepKINET.py --input sim.h5ad --output-dir ./output --cluster-key milestone --simulate
    python DeepKINET.py --metadata-file datasets.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
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

CHECKPOINT_NAME = ".deepkinet_opt.pt"


def seed_everything(seed: int) -> None:
    """Seed NumPy and PyTorch (including CUDA) for reproducible runs."""
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@lru_cache(maxsize=1)
def load_deepkinet_api():
    """Import the DeepKINET ``workflow`` and ``utils`` entry-point modules.

    Upstream DeepKINET uses *flat* intra-package imports (``import utils`` and
    ``import exp`` inside ``workflow.py``/``utils.py``), so the package
    directory itself has to be available on ``sys.path`` rather than importing
    ``deepkinet.workflow`` as a dotted submodule. Instead of appending a
    hard-coded source checkout path, we locate the installed ``deepkinet``
    package through the import system and add its directory.
    """
    spec = importlib.util.find_spec("deepkinet")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "DeepKINET is not installed. Install it with "
            "`pip install -e .` from https://github.com/3254c/DeepKINET.git"
        )

    package_dir = str(Path(next(iter(spec.submodule_search_locations))))
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

    workflow = importlib.import_module("workflow")
    utils = importlib.import_module("utils")
    return workflow, utils


def cleanup_resources() -> None:
    gc.collect()
    plt.close("all")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _to_dense(layer) -> np.ndarray:
    """Convert an AnnData layer to a dense ``float64`` NumPy array."""
    return (layer.toarray() if hasattr(layer, "toarray") else np.asarray(layer)).astype(np.float64)


def check_counts_int(adata, layers=("spliced", "unspliced")) -> bool:
    """Check whether the selected layers hold integer counts.

    Returns ``True`` when every value is (numerically) an integer, which is the
    form DeepKINET expects. Returns ``False`` when a layer contains non-integer
    values, which would trigger DeepKINET's ``ValueError`` during
    ``define_exp`` (see ``input_checks`` upstream).
    """
    for key in layers:
        if key not in adata.layers:
            continue
        arr = _to_dense(adata.layers[key])
        if not np.allclose(arr, np.round(arr), atol=1e-6):
            return False
    return True


def restore_int_counts(adata, layers=("spliced", "unspliced")) -> bool:
    """Restore non-integer spliced/unspliced layers to per-cell integer counts.

    DeepKINET models the raw count-generating process, so its input layers must
    be integer counts. Some benchmark inputs store normalized (non-integer)
    spliced/unspliced values, which makes DeepKINET raise a ``ValueError``. This
    helper rescales each cell (row) by its smallest non-zero value: that value
    is taken as the per-cell normalization factor of "1 count", so dividing by
    it and rounding recovers the original integer counts.

    Returns ``True`` when the layers are integer afterwards (either already were,
    or were successfully restored) and ``False`` when a layer still cannot be
    restored to integers. Already-integer layers are simply cast to ``float32``
    (this also avoids ``uint16`` layers that cannot be converted to a tensor).
    """
    ok = True
    for key in layers:
        if key not in adata.layers:
            continue
        arr = _to_dense(adata.layers[key])
        if np.allclose(arr, np.round(arr), atol=1e-6):
            adata.layers[key] = arr.astype(np.float32)
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            row_base = np.where(arr > 0, arr, np.inf).min(axis=1)  # minimum non-zero per cell
            base = np.where(row_base[:, None] > 0, row_base[:, None], 1.0)
            restored = np.where(row_base[:, None] > 0, np.round(arr / base), 0.0)
        if np.allclose(restored, np.round(restored), atol=1e-6):
            adata.layers[key] = restored.astype(np.float32)
        else:
            print(f"    [WARN] layer '{key}' is non-integer and cannot be restored per cell")
            ok = False
    return ok


def resolve_cluster_key(cluster_key: Optional[str], simulate: bool) -> str:
    """Resolve the clustering column; simulated data uses the ``milestone`` label."""
    if cluster_key:
        return cluster_key
    if simulate:
        return "milestone"
    raise ValueError("--cluster-key is required for real data (or use --simulate)")


def preprocess_adata(
    adata,
    input_path,
    cluster_key: Optional[str] = None,
    dimred_key: str = "X_umap",
    simulate: bool = False,
    n_top_genes: int = 2000,
    min_shared_counts: int = 20,
    n_pcs: int = 30,
    n_neighbors: int = 30,
):
    """Filter/normalize an AnnData and restore the raw integer count layers.

    Real data:   ``filter_and_normalize(min_shared_counts=20, n_top_genes=2000)``.
    Simulated data: copy ``X_dimred`` into ``X_umap``, ``min_shared_counts=None``
    and a dynamic ``n_top_genes`` when the dataset has fewer than 2000 genes.

    After filtering, only the *raw integer* ``spliced``/``unspliced`` layers of
    the retained genes are copied back from the original file, because DeepKINET
    operates on raw counts (see ``check_counts_int``/``restore_int_counts``).

    Returns the processed AnnData and the resolved cluster key.
    """
    input_path = Path(input_path)
    cluster_key = resolve_cluster_key(cluster_key, simulate)

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    if simulate:
        if "X_dimred" in adata.obsm and dimred_key not in adata.obsm:
            adata.obsm[dimred_key] = adata.obsm["X_dimred"].copy()
        if dimred_key != "X_umap" and dimred_key in adata.obsm and "X_umap" not in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm[dimred_key].copy()
        dynamic_top_genes = n_top_genes if adata.n_vars >= n_top_genes else adata.n_vars
        scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=dynamic_top_genes)
    else:
        scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, n_top_genes=n_top_genes)

    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors, method="umap")

    # Re-read the raw file and copy only the filtered genes' raw count layers.
    full_adata = sc.read(input_path)
    raw_spliced = full_adata[:, adata.var_names].layers["spliced"]
    raw_unspliced = full_adata[:, adata.var_names].layers["unspliced"]
    adata.layers["spliced"] = raw_spliced
    adata.layers["unspliced"] = raw_unspliced
    del full_adata

    if not check_counts_int(adata):
        print(f"    [INFO] {input_path.name}: non-integer counts detected, restoring raw counts...")
        if not restore_int_counts(adata):
            raise ValueError(
                f"{input_path.name}: spliced/unspliced counts could not be restored to integers, "
                "and DeepKINET requires integer count data."
            )
    else:
        for key in ("spliced", "unspliced"):
            if key in adata.layers:
                adata.layers[key] = _to_dense(adata.layers[key]).astype(np.float32)

    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        sc.tl.umap(adata)

    return adata, cluster_key


def estimate_kinetics_for_adata(
    adata,
    work_dir,
    cluster_key,
    epochs: int = 2000,
    checkpoint: str = CHECKPOINT_NAME,
):
    """Run DeepKINET kinetics estimation and save the latent velocity figure.

    ``workflow.estimate_kinetics`` writes its checkpoint to the current working
    directory, so the process is temporarily moved into the per-dataset output
    directory and restored afterwards.
    """
    workflow, utils = load_deepkinet_api()

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    original_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        adata, _ = workflow.estimate_kinetics(adata, epoch=epochs, checkpoint=checkpoint)
    finally:
        os.chdir(original_cwd)

    # Upstream writes `layers['DeepKINET_velocity']`; the benchmark reads
    # `layers['velocity']`, so the native key is aliased (and kept as well).
    if "velocity" not in adata.layers:
        if "DeepKINET_velocity" not in adata.layers:
            raise RuntimeError(
                "DeepKINET did not produce layers['DeepKINET_velocity'], "
                "so layers['velocity'] cannot be exported."
            )
        adata.layers["velocity"] = np.asarray(adata.layers["DeepKINET_velocity"])

    try:
        figure_path = work_dir / "latent_velocity.png"
        utils.embedding_func(adata, cluster_key, save_path=str(figure_path))
        plt.close("all")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"    [WARN] latent velocity figure failed: {exc}")

    return adata


def derive_output_stem(input_path: Path) -> str:
    stem = input_path.stem
    if stem.endswith("_dataset"):
        return stem[:-8]
    return stem


def run_deepkinet_analysis(
    input_path,
    output_dir,
    cluster_key: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dimred_key: str = "X_umap",
    simulate: bool = False,
    n_top_genes: int = 2000,
    min_shared_counts: int = 20,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    epochs: int = 2000,
    seed: int = 2024,
    overwrite: bool = False,
) -> Path:
    """Run DeepKINET on a single dataset and write ``<stem>_velo.h5ad``."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if dataset_name is None:
        dataset_name = derive_output_stem(input_path)

    dataset_output_dir = output_dir / str(dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    output_stem = derive_output_stem(input_path)
    output_h5ad = dataset_output_dir / f"{output_stem}_velo.h5ad"
    if output_h5ad.exists() and not overwrite:
        print(f"Skipping existing output: {output_h5ad}")
        return output_h5ad

    seed_everything(seed)

    adata = None
    print(f"\nProcessing: {input_path.name}")
    try:
        adata = sc.read(input_path)
        adata, cluster_key = preprocess_adata(
            adata=adata,
            input_path=input_path,
            cluster_key=cluster_key,
            dimred_key=dimred_key,
            simulate=simulate,
            n_top_genes=n_top_genes,
            min_shared_counts=min_shared_counts,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
        )

        adata = estimate_kinetics_for_adata(
            adata=adata,
            work_dir=dataset_output_dir,
            cluster_key=cluster_key,
            epochs=epochs,
        )

        adata.uns["deepkinet_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(input_path.resolve()),
            "cluster_key": cluster_key,
            "simulate": bool(simulate),
            "epochs": int(epochs),
            "output_path": str(output_h5ad.resolve()),
        }

        if "velocity" not in adata.layers:
            raise RuntimeError("DeepKINET did not produce layers['velocity'].")

        adata.write_h5ad(output_h5ad)
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata
        cleanup_resources()


def detect_separator(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix in {".tsv", ".txt"}:
        return "\t"
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    return "\t" if "\t" in first_line else ","


def load_metadata_file(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path, sep=detect_separator(metadata_path))

    required_columns = ["dataset_name", "file_path"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "cluster_key" not in df.columns:
        df["cluster_key"] = ""
    if "dimred_key" not in df.columns:
        df["dimred_key"] = "X_umap"
    if "simulate" not in df.columns:
        df["simulate"] = False

    df["dataset_name"] = df["dataset_name"].astype(str)
    df["file_path"] = df["file_path"].astype(str)
    df["cluster_key"] = df["cluster_key"].fillna("").astype(str)
    df["dimred_key"] = df["dimred_key"].astype(str)
    df["simulate"] = df["simulate"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "t", "yes", "y"}
    )
    return df


def run_batch_deepkinet(
    metadata_file,
    output_dir,
    epochs: int = 2000,
    seed: int = 2024,
    overwrite: bool = False,
) -> list:
    """Run DeepKINET over every dataset listed in a metadata CSV/TSV file."""
    metadata_file = Path(metadata_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_file(metadata_file)
    outputs = []
    print(f"Batch mode: {len(metadata_df)} datasets")

    for _, row in metadata_df.iterrows():
        file_path = Path(row["file_path"])
        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue
        try:
            output_path = run_deepkinet_analysis(
                input_path=file_path,
                output_dir=output_dir,
                cluster_key=row["cluster_key"] or None,
                dataset_name=row["dataset_name"],
                dimred_key=row["dimred_key"],
                simulate=bool(row["simulate"]),
                epochs=epochs,
                seed=seed,
                overwrite=overwrite,
            )
            outputs.append(output_path)
        except Exception as exc:
            print(f"Failed: {row['dataset_name']}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepKINET velocity analysis",
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
        help="Column name in adata.obs used for labels; defaults to 'milestone' with --simulate",
    )
    parser.add_argument(
        "--dimred-key",
        default="X_umap",
        help="Dimensionality reduction key in adata.obsm; use X_umap for real data and X_dimred for simulated data",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Use simulated-data preprocessing (milestone label, dynamic n_top_genes, min_shared_counts=None)",
    )
    parser.add_argument("--n-top-genes", type=int, default=2000, help="Number of highly variable genes for real data")
    parser.add_argument(
        "--min-shared-counts",
        type=int,
        default=20,
        help="Minimum shared counts for real-data gene filtering",
    )
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of principal components for moments")
    parser.add_argument("--n-neighbors", type=int, default=30, help="Number of neighbors for moments")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs for each kinetics stage")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_deepkinet(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            epochs=args.epochs,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    if not args.cluster_key and not args.simulate:
        parser.error("--cluster-key is required in single-file mode unless --simulate is used")

    return run_deepkinet_analysis(
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
        epochs=args.epochs,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
