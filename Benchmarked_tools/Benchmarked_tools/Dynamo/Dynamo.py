#!/usr/bin/env python3
"""
Dynamo velocity analysis pipeline for VelocityBenchmarking.

Real and simulated datasets are processed by the same script and switched with
the --simulate flag:

* real data: reads the H5AD, reuses an existing UMAP (or computes one), and runs
  the Dynamo Monocle recipe followed by stochastic dynamics.
* simulated data: uses the pre-computed `obsm['X_dimred']` embedding, relaxes the
  gene filter, and labels the cells with the `milestone` cluster.

Installation:
    pip install dynamo-release scanpy scvelo pyarrow simplejson GPUtil pynvml

Usage:
    python Dynamo.py --input data.h5ad --output-dir ./output --cluster-key celltype
    python Dynamo.py --input sim.h5ad --output-dir ./output --cluster-key milestone --simulate
    python Dynamo.py --metadata-file datasets.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
import gc
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
import dynamo as dyn

def seed_everything(seed: int) -> None:
    """Seed the common random number generators used by the pipeline."""
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
    n_pcs: int = 30,
    n_neighbors: int = 30,
    num_dim: int = 30,
    shared_count: int = 20,
):
    """
    Prepare an AnnData object for Dynamo.

    Annotation and embedding handling is branch specific, while the
    `recipe_monocle` normalization uses the branch-specific gene filter.
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
        effective_shared_count = 1

        if cluster_key not in adata.obs.columns:
            adata.obs[cluster_key] = "milestone"
    else:
        has_umap = any("x_umap" in key.lower() for key in adata.obsm.keys())
        if has_umap:
            print("  X_umap already exists")
        else:
            print("  Computing UMAP because no X_umap-like embedding was found...")
            sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
            sc.tl.umap(adata)

        effective_top_genes = n_top_genes
        effective_shared_count = shared_count

    print(
        f"  Running recipe_monocle (n_top_genes={effective_top_genes}, "
        f"shared_count={effective_shared_count}, num_dim={num_dim})..."
    )
    dyn.pp.recipe_monocle(
        adata,
        n_top_genes=effective_top_genes,
        fg_kwargs={"shared_count": effective_shared_count},
        num_dim=num_dim,
    )

    return adata


def compute_dynamics(adata, model: str = "stochastic", fallback_model: str = "deterministic", n_pca_components: int = 30):
    """
    Run Dynamo dynamics and cell velocities.

    `dyn.tl.reduceDimension` overwrites `obsm['X_umap']`, so the existing UMAP is
    backed up and restored to keep the pre-computed embedding intact.
    """
    try:
        dyn.tl.dynamics(adata, model=model)
    except np.linalg.LinAlgError:
        print(
            f"  SVD did not converge with model='{model}'; "
            f"retrying with model='{fallback_model}'."
        )
        dyn.tl.dynamics(adata, model=fallback_model)

    umap_backup = adata.obsm.get("X_umap", None)
    dyn.tl.reduceDimension(adata, n_pca_components=n_pca_components)
    if umap_backup is not None:
        adata.obsm["X_umap"] = umap_backup

    dyn.tl.cell_velocities(adata, method="pearson", other_kernels_dict={"transform": "sqrt"})
    return adata


def copy_dynamo_to_scvelo(adata):
    """Mirror Dynamo velocity keys onto the scVelo naming convention."""
    if "velocity_S" in adata.layers:
        adata.layers["velocity"] = adata.layers["velocity_S"].copy()

    if "velocity_umap" not in adata.obsm and "velocity_S" in adata.layers:
        try:
            from scvelo.tools.velocity_embedding import velocity_embedding

            velocity_embedding(adata, basis="umap")
        except Exception:
            pass

    return adata


def plot_results(adata, dataset_label: str, plot_dir: Path, cluster_key: str) -> None:
    """Save the Dynamo velocity plots produced by the original driver.

    Mirrors the ``dyn.pl`` block of ``dynamo_1.py``: streamline, grid and cell-wise
    vector plots are written as PDF files next to the result. Plotting failures never
    discard a completed fit.
    """
    plot_dir = plot_dir / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        dyn.pl.streamline_plot(
            adata, color=cluster_key, basis="umap", show_legend="on data",
            show_arrowed_spines=True, save_show_or_return="save",
            save_kwargs={"prefix": "stream_arrow", "ext": "pdf", "path": str(plot_dir)},
        )
        dyn.pl.grid_vectors(
            adata, color=cluster_key, basis="umap", show_legend="on data",
            save_show_or_return="save",
            save_kwargs={"prefix": "grid_arrow", "ext": "pdf", "path": str(plot_dir)},
        )
        dyn.pl.cell_wise_vectors(
            adata, color=cluster_key, basis="umap", show_legend="on data",
            quiver_length=6, quiver_size=6, save_show_or_return="save",
            save_kwargs={"prefix": "full_arrow", "ext": "pdf", "path": str(plot_dir)},
        )
    except Exception as exc:  # plotting must never discard a completed fit
        print(f"  Visualization skipped: {exc}")
    finally:
        plt.close("all")


def run_dynamo_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    cluster_key: str,
    dataset_name: Optional[str] = None,
    dimred_key: str = "X_umap",
    simulate: bool = False,
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    num_dim: int = 30,
    n_pca_components: int = 30,
    shared_count: int = 20,
    model: str = "stochastic",
    fallback_model: str = "deterministic",
    seed: int = 2024,
    overwrite: bool = False,
) -> Path:
    """Run Dynamo on a single dataset and write the annotated H5AD."""
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
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            num_dim=num_dim,
            shared_count=shared_count,
        )

        print(f"  Computing dynamics (model='{model}')...")
        compute_dynamics(
            adata,
            model=model,
            fallback_model=fallback_model,
            n_pca_components=n_pca_components,
        )

        print("  Copying Dynamo velocities to scVelo keys...")
        copy_dynamo_to_scvelo(adata)

        print("  Generating plots...")
        plot_results(adata, output_stem, dataset_output_dir, cluster_key)

        adata.uns["dynamo_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(input_path.resolve()),
            "cluster_key": cluster_key,
            "dimred_key": dimred_key,
            "simulate": bool(simulate),
            "model": model,
            "output_path": str(output_h5ad.resolve()),
        }

        if "velocity" not in adata.layers:
            raise RuntimeError("Dynamo did not produce layers['velocity'].")

        adata.write(output_h5ad, compression="lzf")
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata
        cleanup_resources()


def run_batch_dynamo(
    metadata_file: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    seed: int = 2024,
) -> list[Path]:
    """Run Dynamo for every dataset listed in a metadata table."""
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
            output_path = run_dynamo_analysis(
                input_path=file_path,
                output_dir=output_dir,
                cluster_key=row["cluster_key"],
                dataset_name=row["dataset_name"],
                dimred_key=row["dimred_key"],
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
        description="Dynamo velocity analysis",
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
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of principal components for the neighbor graph")
    parser.add_argument("--n-neighbors", type=int, default=30, help="Number of neighbors for the neighbor graph")
    parser.add_argument("--num-dim", type=int, default=30, help="Number of dimensions for recipe_monocle")
    parser.add_argument(
        "--n-pca-components",
        type=int,
        default=30,
        help="Number of PCA components for reduceDimension",
    )
    parser.add_argument(
        "--shared-count",
        type=int,
        default=20,
        help="Minimum shared counts for the real-data gene filter",
    )
    parser.add_argument("--model", default="stochastic", help="Dynamo dynamics model")
    parser.add_argument(
        "--fallback-model",
        default="deterministic",
        help="Dynamo dynamics model used when the primary model fails",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_dynamo(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    if not args.cluster_key:
        parser.error("--cluster-key is required in single-file mode")

    return run_dynamo_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        dimred_key=args.dimred_key,
        simulate=args.simulate,
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        num_dim=args.num_dim,
        n_pca_components=args.n_pca_components,
        shared_count=args.shared_count,
        model=args.model,
        fallback_model=args.fallback_model,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
