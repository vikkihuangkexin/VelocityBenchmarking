#!/usr/bin/env python3
"""
GRAVITY velocity analysis pipeline for VelocityBenchmarking.

GRAVITY predicts RNA velocity and regulatory rewiring by joint deep learning of
cell-state transitions and gene-level kinetics. The pipeline consumes a
cellDancer-style long-format count table and runs a two-stage optimization
(cell-wise trajectory recovery, then gene-wise kinetic refinement).

Installation:
    git clone https://github.com/CSUBioGroup/GRAVITY.git
    cd GRAVITY
    pip install -e .

Usage:
    python GRAVITY.py --input data.h5ad --output-dir ./output --cluster-key celltype
    python GRAVITY.py --input sim.h5ad --output-dir ./output --cluster-key milestone --simulate
    python GRAVITY.py --metadata-file datasets.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Optional

# --- Compatibility patch: scVelo 0.2.5 relies on matplotlib.cbook.mplDeprecation,
#     which was removed in matplotlib >= 3.7. Add it before importing scVelo
#     (and therefore before importing GRAVITY, which imports scVelo internally).
import matplotlib

matplotlib.use("Agg")
import matplotlib.cbook as _cbook

if not hasattr(_cbook, "mplDeprecation"):
    _cbook.mplDeprecation = DeprecationWarning

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

from gravity_result_to_h5ad import convert as convert_result_to_h5ad

def seed_everything(seed: int) -> None:
    """Seed NumPy and PyTorch (including CUDA) for reproducible runs."""
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_gravity_api():
    """Import the installed GRAVITY entry points.

    Returns ``(PipelineConfig, run_pipeline, export_intermediate_from_h5ad,
    adata_to_df_with_embed, compute_cell_velocity_)``.
    """
    from gravity import PipelineConfig, run_pipeline
    from gravity.data.preprocessing import adata_to_df_with_embed, export_intermediate_from_h5ad
    from gravity.velocity import compute_cell_velocity_

    return (
        PipelineConfig,
        run_pipeline,
        export_intermediate_from_h5ad,
        adata_to_df_with_embed,
        compute_cell_velocity_,
    )


def cleanup_resources() -> None:
    gc.collect()
    plt.close("all")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def detect_species(filename: str) -> str:
    """Detect the species from a file name. ``Mm`` -> mouse, ``Hs`` -> human."""
    low = filename.lower()
    if "mm" in low:
        return "mouse"
    if "hs" in low:
        return "human"
    raise ValueError(f"Cannot detect species from file name '{filename}' (expected 'Mm' or 'Hs')")


def resolve_prior_network(input_name: str, prior_network: Optional[str], simulate: bool) -> Optional[str]:
    """Resolve the prior TF-target network path.

    Simulated data carries no species information, so no prior network is used.
    For real data the species is detected from the file name and the matching
    ``prior_data/nichenet_<species>.zip`` archive shipped with GRAVITY is used
    (located relative to the installed package instead of a hard-coded path).
    """
    if simulate:
        return None
    if prior_network:
        return prior_network

    species = detect_species(input_name)
    import gravity

    package_dir = Path(gravity.__file__).resolve().parent
    candidate = package_dir.parent / "prior_data" / f"nichenet_{species}.zip"
    if candidate.exists():
        return str(candidate)
    print(f"[WARN] prior network not found for species '{species}': {candidate}")
    return None


def dynamic_top_genes(n_vars: int, n_top_genes: int = 2000) -> int:
    """Pick the number of highly variable genes for simulated data.

    Small simulated datasets are downsampled to a multiple of 500 (capped at the
    dataset size and at 500 genes); larger datasets use ``n_top_genes``.
    """
    if n_vars < 10000:
        top_gene = (n_vars // 500) * 500
        return min(top_gene, n_vars, 500)
    return n_top_genes


def preprocess_adata(
    adata,
    dimred_key: str = "X_umap",
    simulate: bool = False,
    n_top_genes: int = 2000,
    min_shared_counts: int = 20,
    n_pcs: int = 30,
    n_neighbors: int = 30,
):
    """Preprocess AnnData for GRAVITY and compute the ``Mu``/``Ms`` moments.

    Real data: ``filter_and_normalize(min_shared_counts=20, n_top_genes=2000)``
    followed by ``scv.pp.moments(n_pcs=30, n_neighbors=30)``.

    Simulated data: the 2D simulated embedding (``X_dimred``) is copied into
    ``dimred_key``, ``min_shared_counts=None`` is used and the gene count is
    reduced dynamically for small datasets.

    Either way the result contains the layers (``Mu``/``Ms``) and embedding that
    GRAVITY exports as its long-format count table.
    """
    adata.obs_names_make_unique()

    if simulate and "X_dimred" in adata.obsm:
        adata.obsm[dimred_key] = adata.obsm["X_dimred"].copy()

    if simulate:
        top_gene = dynamic_top_genes(adata.n_vars, n_top_genes=n_top_genes)
        scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=top_gene)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors, method="umap")
        scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    else:
        scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, n_top_genes=n_top_genes)
        scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)

    if dimred_key not in adata.obsm and "X_umap" not in adata.obsm:
        sc.tl.umap(adata)
        if dimred_key != "X_umap":
            adata.obsm[dimred_key] = adata.obsm["X_umap"].copy()

    return adata


def resolve_cluster_key(cluster_key: Optional[str], simulate: bool) -> str:
    """Resolve the clustering column; simulated data uses the ``milestone`` label."""
    if cluster_key:
        return cluster_key
    if simulate:
        return "milestone"
    raise ValueError("--cluster-key is required for real data (or use --simulate)")


def export_long_table(
    input_path: Path,
    csv_path: Path,
    cluster_key: Optional[str],
    dimred_key: str,
    simulate: bool,
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    min_shared_counts: int = 20,
    overwrite: bool = False,
    preprocessed: bool = False,
) -> Path:
    """Write the GRAVITY long-format count table (``cell_type_u_s.csv``).

    Real data: delegated to GRAVITY's ``export_intermediate_from_h5ad`` (which
    performs filter/normalize + moments on the h5ad).
    Simulated data: preprocessed in-memory and exported with
    ``adata_to_df_with_embed`` (``min_shared_counts=None``, dynamic gene count).
    Already-preprocessed input (``preprocessed=True``): the h5ad is exported
    directly with ``adata_to_df_with_embed`` without re-normalizing.
    """
    _, _, export_intermediate_from_h5ad, adata_to_df_with_embed, _ = load_gravity_api()
    cluster_key = resolve_cluster_key(cluster_key, simulate)

    if preprocessed:
        adata = sc.read(input_path)
        adata_to_df_with_embed(
            adata,
            us_para=["Mu", "Ms"],
            cell_type_para=cluster_key,
            embed_para=dimred_key,
            save_path=str(csv_path),
        )
        del adata
        return csv_path

    if not simulate:
        export_intermediate_from_h5ad(
            input_h5ad=str(input_path),
            output_csv=str(csv_path),
            embed_key=dimred_key,
            celltype_key=cluster_key,
            min_shared_counts=min_shared_counts,
            n_top_genes=n_top_genes,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            overwrite=overwrite,
        )
        return csv_path

    adata = sc.read(input_path)
    adata = preprocess_adata(
        adata,
        dimred_key=dimred_key,
        simulate=True,
        n_top_genes=n_top_genes,
        min_shared_counts=min_shared_counts,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
    )
    adata_to_df_with_embed(
        adata,
        us_para=["Mu", "Ms"],
        cell_type_para=cluster_key,
        embed_para=dimred_key,
        save_path=str(csv_path),
    )
    del adata
    return csv_path


def derive_output_stem(input_path) -> str:
    input_path = Path(input_path)
    stem = input_path.stem
    if stem.endswith("_dataset"):
        return stem[:-8]
    return stem


def run_gravity_analysis(
    input_path,
    output_dir,
    cluster_key: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dimred_key: str = "X_umap",
    simulate: bool = False,
    prior_network: Optional[str] = None,
    n_top_genes: int = 2000,
    min_shared_counts: int = 20,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    stage1_epochs: int = 6,
    stage2_epochs: int = 6,
    stage1_lr: float = 1e-6,
    stage2_lr: float = 1e-4,
    batch_size: int = 128,
    num_workers: int = 8,
    seed: int = 2024,
    overwrite: bool = False,
    preprocessed: bool = False,
) -> Path:
    """Run GRAVITY on a single dataset and write ``<stem>.h5ad``.

    Set ``preprocessed=True`` when ``input_path`` already contains the
    filter/normalize + moments result (for example a ``pp.h5ad`` produced by the
    robustness driver); then the long table is exported without re-normalizing.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if dataset_name is None:
        dataset_name = derive_output_stem(input_path)

    dataset_output_dir = output_dir / str(dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    output_stem = derive_output_stem(input_path)
    output_h5ad = dataset_output_dir / f"{output_stem}.h5ad"
    if output_h5ad.exists() and not overwrite:
        print(f"Skipping existing output: {output_h5ad}")
        return output_h5ad

    seed_everything(seed)
    PipelineConfig, run_pipeline, _, _, compute_cell_velocity_ = load_gravity_api()

    if preprocessed and not simulate and not prior_network:
        # A preprocessed file (e.g. pp.h5ad) carries no species in its name, so the
        # prior network must be supplied explicitly for preprocessed real data.
        prior = None
        print("  [WARN] preprocessed real data without --prior-network: running without a prior network")
    else:
        prior = resolve_prior_network(input_path.name, prior_network, simulate)
    print(f"\nProcessing: {input_path.name}")
    print(f"  Species prior network: {prior}")

    csv_path = dataset_output_dir / "cell_type_u_s.csv"
    export_long_table(
        input_path=input_path,
        csv_path=csv_path,
        cluster_key=cluster_key,
        dimred_key=dimred_key,
        simulate=simulate,
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        min_shared_counts=min_shared_counts,
        overwrite=overwrite,
        preprocessed=preprocessed,
    )

    cfg = PipelineConfig(
        raw_counts=str(csv_path),
        workdir=str(dataset_output_dir),
        prior_network=prior,
        accelerator="gpu",
        devices=1,
        batch_size=batch_size,
        num_workers=num_workers,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        stage1_lr=stage1_lr,
        stage2_lr=stage2_lr,
    )
    outputs = run_pipeline(cfg)
    print(f"  stage2.csv: {outputs['stage2_csv']}")

    stage2_df = pd.read_csv(outputs["stage2_csv"])
    try:
        result_df, _ = compute_cell_velocity_(stage2_df)
    except Exception as exc:
        print(f"  [WARN] cell velocity computation failed, using stage2 output: {exc}")
        result_df = stage2_df
    if "loss" not in result_df.columns:
        result_df["loss"] = 0.0

    result_csv = dataset_output_dir / "gravity_result.csv"
    result_df.to_csv(result_csv, index=False)
    print(f"  result CSV: {result_csv}")

    adata = convert_result_to_h5ad(result_csv)
    adata.uns["gravity_run"] = {
        "dataset_name": str(dataset_name),
        "input_path": str(input_path.resolve()),
        "cluster_key": resolve_cluster_key(cluster_key, simulate),
        "dimred_key": dimred_key,
        "simulate": bool(simulate),
        "prior_network": str(prior) if prior else "",
        "output_path": str(output_h5ad.resolve()),
    }

    if "velocity" not in adata.layers:
        raise RuntimeError("GRAVITY did not produce layers['velocity'].")

    adata.write_h5ad(output_h5ad)
    print(f"  Done: {output_h5ad}")

    del adata, result_df, stage2_df
    cleanup_resources()
    return output_h5ad


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
    if "prior_network" not in df.columns:
        df["prior_network"] = ""

    df["dataset_name"] = df["dataset_name"].astype(str)
    df["file_path"] = df["file_path"].astype(str)
    df["cluster_key"] = df["cluster_key"].fillna("").astype(str)
    df["dimred_key"] = df["dimred_key"].astype(str)
    df["prior_network"] = df["prior_network"].fillna("").astype(str)
    df["simulate"] = df["simulate"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "t", "yes", "y"}
    )
    return df


def run_batch_gravity(
    metadata_file,
    output_dir,
    stage1_epochs: int = 6,
    stage2_epochs: int = 6,
    seed: int = 2024,
    overwrite: bool = False,
) -> list:
    """Run GRAVITY over every dataset listed in a metadata CSV/TSV file."""
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
            output_path = run_gravity_analysis(
                input_path=file_path,
                output_dir=output_dir,
                cluster_key=row["cluster_key"] or None,
                dataset_name=row["dataset_name"],
                dimred_key=row["dimred_key"],
                simulate=bool(row["simulate"]),
                prior_network=row["prior_network"] or None,
                stage1_epochs=stage1_epochs,
                stage2_epochs=stage2_epochs,
                seed=seed,
                overwrite=overwrite,
            )
            outputs.append(output_path)
        except Exception as exc:
            print(f"Failed: {row['dataset_name']}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GRAVITY velocity analysis",
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
        help="Column name in adata.obs used as cell-type labels; defaults to 'milestone' with --simulate",
    )
    parser.add_argument(
        "--dimred-key",
        default="X_umap",
        help="Embedding key in adata.obsm; use X_umap for real data and X_dimred for simulated data",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Use simulated-data preprocessing (dynamic gene count, min_shared_counts=None, no prior network)",
    )
    parser.add_argument(
        "--prior-network",
        default=None,
        help="Path to the prior TF-target network archive; detected from the file name for real data",
    )
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        default=False,
        help="Input already contains filter/normalize + moments (e.g. a pp.h5ad), so it is not re-normalized",
    )
    parser.add_argument("--n-top-genes", type=int, default=2000, help="Number of highly variable genes")
    parser.add_argument(
        "--min-shared-counts",
        type=int,
        default=20,
        help="Minimum shared counts for real-data gene filtering",
    )
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of principal components")
    parser.add_argument("--n-neighbors", type=int, default=30, help="Number of neighbors")
    parser.add_argument("--stage1-epochs", type=int, default=6, help="Cell-wise stage epochs")
    parser.add_argument("--stage2-epochs", type=int, default=6, help="Gene-wise stage epochs")
    parser.add_argument("--stage1-lr", type=float, default=1e-6, help="Cell-wise stage learning rate")
    parser.add_argument("--stage2-lr", type=float, default=1e-4, help="Gene-wise stage learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_gravity(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    if not args.cluster_key and not args.simulate:
        parser.error("--cluster-key is required in single-file mode unless --simulate is used")

    return run_gravity_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        dimred_key=args.dimred_key,
        simulate=args.simulate,
        prior_network=args.prior_network,
        n_top_genes=args.n_top_genes,
        min_shared_counts=args.min_shared_counts,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        overwrite=args.overwrite,
        preprocessed=args.preprocessed,
    )


if __name__ == "__main__":
    main()
