#!/usr/bin/env python3
"""
MultiVelo multi-omics velocity analysis pipeline for VelocityBenchmarking.

MultiVelo estimates chromatin-informed RNA velocity from paired RNA and
chromatin accessibility data using ``mv.recover_dynamics_chrom``. The RNA
modality supplies spliced and unspliced counts and the ATAC modality supplies
a gene activity layer (``chromatin`` by default, ``Mc`` as a fallback).

Installation:
    pip install multivelo
    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

Usage:
    python MultiVelo.py --rna_dir rna.h5ad --atac_dir atac.h5ad --save_dir results --cluster-key celltype
    python MultiVelo.py --rna_dir rna.h5ad --atac_dir atac.h5ad --save_dir results --cluster-key milestone --simulate
    python MultiVelo.py --metadata_file datasets.csv --save_dir results
"""

from __future__ import annotations

import argparse
import gc
import shutil
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

DEFAULT_ATAC_LAYER = "chromatin"
ATAC_LAYER_FALLBACKS = ("chromatin", "Mc")


@lru_cache(maxsize=1)
def load_multivelo_api():
    """
    Load the installed MultiVelo package without being shadowed by this file.

    On case-insensitive filesystems ``import multivelo`` could resolve to this
    ``MultiVelo.py`` script, so the script directory is temporarily removed
    while importing the real package.
    """
    script_dir = str(Path(__file__).resolve().parent)
    removed_path = False
    if script_dir in sys.path:
        sys.path.remove(script_dir)
        removed_path = True

    local_module = None
    restore_local_module = __name__ == "multivelo" and "multivelo" in sys.modules
    if restore_local_module:
        local_module = sys.modules.pop("multivelo")

    try:
        import multivelo as mv
    finally:
        if restore_local_module and local_module is not None:
            sys.modules["multivelo"] = local_module
        if removed_path:
            sys.path.insert(0, script_dir)

    return mv


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


def cleanup_resources() -> None:
    gc.collect()
    plt.close("all")


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


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

    required_columns = ["dataset_name", "rna_path", "atac_path", "cluster_key"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "simulate" not in df.columns:
        df["simulate"] = False
    if "atac_layer" not in df.columns:
        df["atac_layer"] = DEFAULT_ATAC_LAYER
    if "max_iter" not in df.columns:
        df["max_iter"] = 5
    if "n_jobs" not in df.columns:
        df["n_jobs"] = 1
    if "n_anchors" not in df.columns:
        df["n_anchors"] = 500

    df["dataset_name"] = df["dataset_name"].astype(str)
    df["rna_path"] = df["rna_path"].astype(str)
    df["atac_path"] = df["atac_path"].astype(str)
    df["cluster_key"] = df["cluster_key"].astype(str)
    df["atac_layer"] = df["atac_layer"].astype(str)
    df["simulate"] = df["simulate"].map(parse_bool)
    df["max_iter"] = df["max_iter"].astype(int)
    df["n_jobs"] = df["n_jobs"].astype(int)
    df["n_anchors"] = df["n_anchors"].astype(int)

    return df


def derive_output_stem(input_path: Path) -> str:
    stem = input_path.stem
    if stem.endswith("_dataset"):
        return stem[:-8]
    return stem


def resolve_atac_layer(adata, atac_layer: str) -> str:
    candidates = [atac_layer] + [name for name in ATAC_LAYER_FALLBACKS if name != atac_layer]
    for candidate in candidates:
        if candidate in adata.layers:
            return candidate
    available = sorted(adata.layers.keys())
    raise ValueError(
        f"No ATAC gene activity layer found; tried {candidates}. Available layers: {available}"
    )


def preprocess_adata(
    adata_rna,
    cluster_key: Optional[str],
    atac_layer: str = DEFAULT_ATAC_LAYER,
    simulate: bool = False,
    min_shared_counts: int = 10,
    n_top_genes: Optional[int] = None,
):
    adata_rna.obs_names_make_unique()
    adata_rna.var_names_make_unique()

    if simulate and "X_dimred" in adata_rna.obsm:
        adata_rna.obsm["X_umap"] = np.asarray(adata_rna.obsm["X_dimred"])
    elif "X_umap" not in adata_rna.obsm and "X_tsne" in adata_rna.obsm:
        adata_rna.obsm["X_umap"] = np.asarray(adata_rna.obsm["X_tsne"])

    if simulate:
        if cluster_key and cluster_key in adata_rna.obs:
            adata_rna.obs[cluster_key] = "milestone"
    elif cluster_key is not None and cluster_key not in adata_rna.obs:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

    resolved_atac_layer = resolve_atac_layer(adata_rna, atac_layer)

    missing_layers = [layer for layer in ("spliced", "unspliced") if layer not in adata_rna.layers]
    if missing_layers:
        raise ValueError(f"Missing required layers: {missing_layers}")

    adata_rna.X = adata_rna.layers["spliced"].copy()

    filter_kwargs = {}
    if n_top_genes is not None:
        filter_kwargs["n_top_genes"] = int(min(n_top_genes, adata_rna.n_vars))
    if simulate:
        scv.pp.filter_and_normalize(adata_rna, min_shared_counts=None, **filter_kwargs)
    else:
        scv.pp.filter_and_normalize(
            adata_rna, min_shared_counts=min_shared_counts, **filter_kwargs
        )

    adata_atac = adata_rna.copy()
    adata_atac.X = adata_atac.layers[resolved_atac_layer].copy()

    n_neighbors = min(30, adata_rna.n_obs - 1)
    n_pcs = min(30, adata_rna.n_obs - 1, adata_rna.n_vars - 1)
    if n_neighbors < 2 or n_pcs < 2:
        raise ValueError("Insufficient cells or genes remain after filtering to run MultiVelo.")
    scv.pp.moments(adata_rna, n_pcs=n_pcs, n_neighbors=n_neighbors)

    return adata_rna, adata_atac, resolved_atac_layer


def run_multivelo_analysis(
    rna_path,
    atac_path,
    save_dir,
    cluster_key: Optional[str] = None,
    dataset_name: Optional[str] = None,
    atac_layer: str = DEFAULT_ATAC_LAYER,
    max_iter: int = 5,
    n_jobs: int = 1,
    n_anchors: int = 500,
    init_mode: str = "invert",
    simulate: bool = False,
    min_shared_counts: int = 10,
    n_top_genes: Optional[int] = None,
    overwrite: bool = False,
    seed: int = 2024,
) -> Path:
    mv = load_multivelo_api()
    mv.settings.VERBOSITY = 0

    rna_path = Path(rna_path)
    save_dir = Path(save_dir)
    if not rna_path.exists():
        raise FileNotFoundError(f"Input RNA file not found: {rna_path}")
    if atac_path is not None and not Path(atac_path).exists():
        raise FileNotFoundError(f"Input ATAC file not found: {atac_path}")

    if dataset_name is None:
        dataset_name = derive_output_stem(rna_path)

    dataset_output_dir = save_dir / str(dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    output_stem = derive_output_stem(rna_path)
    # Native MultiVelo file name, preserved from code/MultiVelo_sim.py.
    output_h5ad = dataset_output_dir / f"{output_stem}_MultiVelo.h5ad"

    if output_h5ad.exists() and not overwrite:
        print(f"Skipping existing output: {output_h5ad}")
        return output_h5ad

    seed_everything(seed)

    color_key = "milestone" if simulate else (cluster_key or "celltype")

    adata_rna = None
    adata_atac = None
    try:
        print(f"\nProcessing: {rna_path.name}")
        adata_rna = sc.read(rna_path)

        print("  Preprocessing...")
        adata_rna, adata_atac, resolved_atac_layer = preprocess_adata(
            adata_rna=adata_rna,
            cluster_key=cluster_key,
            atac_layer=atac_layer,
            simulate=simulate,
            min_shared_counts=min_shared_counts,
            n_top_genes=n_top_genes,
        )

        print("  Running MultiVelo recover_dynamics_chrom...")
        kwargs = {}
        if color_key in adata_rna.obs:
            kwargs["extra_color_key"] = color_key

        result = mv.recover_dynamics_chrom(
            adata_rna,
            adata_atac,
            max_iter=max_iter,
            init_mode=init_mode,
            parallel=None,
            n_jobs=n_jobs,
            save_plot=False,
            rna_only=False,
            fit=True,
            n_anchors=n_anchors,
            **kwargs,
        )

        if "X_dimred" in result.obsm:
            result.obsm["X_umap"] = np.asarray(result.obsm["X_dimred"])

        if "velocity" not in result.layers:
            raise RuntimeError("MultiVelo did not produce layers['velocity']")

        result.uns["multivelo_run"] = {
            "dataset_name": str(dataset_name),
            "rna_path": str(rna_path.resolve()),
            "atac_path": None if atac_path is None else str(Path(atac_path).resolve()),
            "cluster_key": None if cluster_key is None else str(cluster_key),
            "atac_layer": str(resolved_atac_layer),
            "simulate": bool(simulate),
            "output_path": str(output_h5ad.resolve()),
        }

        result.write(output_h5ad, compression="lzf")
        shutil.copyfile(output_h5ad, dataset_output_dir / "rc.h5ad")
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata_rna
        del adata_atac
        cleanup_resources()


def run_batch_multivelo(
    metadata_file,
    save_dir,
    overwrite: bool = False,
    seed: int = 2024,
) -> list:
    metadata_file = Path(metadata_file)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_file(metadata_file)
    outputs = []
    print(f"Batch mode: {len(metadata_df)} datasets")

    for _, row in metadata_df.iterrows():
        rna_path = Path(row["rna_path"])
        atac_path = Path(row["atac_path"])
        if not rna_path.exists() or not atac_path.exists():
            print(f"Skipping missing files: {rna_path} / {atac_path}")
            continue

        try:
            output_path = run_multivelo_analysis(
                rna_path=rna_path,
                atac_path=atac_path,
                save_dir=save_dir,
                cluster_key=row["cluster_key"],
                dataset_name=row["dataset_name"],
                atac_layer=row["atac_layer"],
                max_iter=int(row["max_iter"]),
                n_jobs=int(row["n_jobs"]),
                n_anchors=int(row["n_anchors"]),
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
        description="MultiVelo multi-omics velocity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--rna_dir", default="./adata_postpro.h5ad", help="Input RNA h5ad data file path")
    parser.add_argument("--atac_dir", default="./adata_atac_postpro.h5ad", help="Input ATAC h5ad data file path")
    parser.add_argument("--save_dir", default="./test", help="Result saving directory")
    parser.add_argument("--metadata_file", default=None, help="Metadata CSV/TSV file for batch processing")
    parser.add_argument("--dataset_name", default=None, help="Dataset folder name for single-file mode")
    parser.add_argument(
        "--cluster-key",
        default=None,
        help="Column name in adata.obs used as the color key; use milestone for simulated data",
    )
    parser.add_argument(
        "--atac_layer",
        default=DEFAULT_ATAC_LAYER,
        help="adata.layers key holding the ATAC gene activity matrix; falls back to 'Mc' when unset",
    )
    parser.add_argument("--max_iter", type=int, default=5, help="Maximum iterations for recover_dynamics_chrom")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs for parallel processing in recover_dynamics_chrom")
    parser.add_argument("--n_anchors", type=int, default=500, help="Number of anchors for recover_dynamics_chrom")
    parser.add_argument("--init_mode", default="invert", help="Initialization mode passed to recover_dynamics_chrom")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Simulated-data branch: use the milestone label, X_dimred, and relaxed filters",
    )
    parser.add_argument(
        "--min_shared_counts",
        type=int,
        default=10,
        help="Minimum shared counts used by filter_and_normalize in the real-data branch",
    )
    parser.add_argument(
        "--n_top_genes",
        type=int,
        default=None,
        help="Number of highly variable genes retained; when unset the kwarg is omitted",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_multivelo(
            metadata_file=args.metadata_file,
            save_dir=args.save_dir,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    return run_multivelo_analysis(
        rna_path=args.rna_dir,
        atac_path=args.atac_dir,
        save_dir=args.save_dir,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        atac_layer=args.atac_layer,
        max_iter=args.max_iter,
        n_jobs=args.n_jobs,
        n_anchors=args.n_anchors,
        init_mode=args.init_mode,
        simulate=args.simulate,
        min_shared_counts=args.min_shared_counts,
        n_top_genes=args.n_top_genes,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
