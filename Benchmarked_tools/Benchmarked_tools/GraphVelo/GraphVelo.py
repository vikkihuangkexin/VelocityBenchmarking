#!/usr/bin/env python3
"""
GraphVelo RNA velocity analysis pipeline for VelocityBenchmarking.

GraphVelo learns a gene-gene velocity graph on top of an scVelo dynamical
velocity estimate, using the MACk score to select the most informative genes.

Installation:
    pip install graphvelo scvelo==0.2.5 GPUtil pynvml numpy==1.23.5 pygam

Usage:
    python GraphVelo.py --input data.h5ad --output-dir ./output --cluster-key celltype
    python GraphVelo.py --input sim.h5ad --output-dir ./output --cluster-key milestone --simulate
    python GraphVelo.py --metadata-file datasets.csv --output-dir ./output
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
import scvelo as scv

import graphvelo.graph_velocity as gv_module
from graphvelo.graph_velocity import GraphVelo
from graphvelo.utils import adj_to_knn, mack_score

PALETTE = [
    "#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4",
    "#66c2a5", "#3288bd", "#abdda4", "#e6f598", "#fee08b",
    "#f46d43", "#e7298a", "#a6cee3", "#1f78b4", "#b2df8a",
    "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
    "#cab2d6", "#6a3d9a", "#ffff99", "#b15928", "#8dd3c7",
    "#bc80bd", "#ccebc5", "#ffed6f", "#999999",
    "#8B0000", "#006400", "#FF69B4", "#00CED1", "#FFD700",
]


def seed_everything(seed: int) -> None:
    """Seed the random number generators used by the pipeline."""
    np.random.seed(seed)


def cleanup_resources() -> None:
    """Release memory held by the previous dataset."""
    gc.collect()
    plt.close("all")


def parse_bool(value) -> bool:
    """Convert common textual/numeric boolean representations to bool."""
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
    """Infer the column separator of a metadata table."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix in {".tsv", ".txt"}:
        return "\t"
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    return "\t" if "\t" in first_line else ","


def load_metadata_file(metadata_path: Path) -> pd.DataFrame:
    """Load and validate a batch metadata table."""
    df = pd.read_csv(metadata_path, sep=detect_separator(metadata_path))

    required_columns = ["dataset_name", "file_path"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "cluster_key" not in df.columns:
        df["cluster_key"] = "milestone"
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


def derive_output_stem(input_path: Path) -> str:
    """Return the per-dataset output stem used by the benchmark."""
    stem = input_path.stem
    if stem.endswith("_dataset"):
        return stem[:-8]
    return stem


def patch_estimate_dt() -> None:
    """
    Patch GraphVelo's _estimate_dt so that zero-norm velocity cells get a zero
    time step instead of producing NaNs through division by zero.
    """
    def _safe_estimate_dt(X, V, nbrs_idx):
        dt = np.zeros(X.shape[0])
        for i, ind in enumerate(nbrs_idx):
            delta_i = X[ind] - X[i]
            density = np.mean(np.linalg.norm(delta_i, axis=1))
            v_norm = np.linalg.norm(V[i])
            dt[i] = 0.0 if v_norm == 0 else np.median(density / v_norm)
        return dt.reshape(-1, 1)

    gv_module._estimate_dt = _safe_estimate_dt


def resolve_n_top_genes(n_vars: int, n_top_genes: int, simulate: bool) -> int:
    """Apply the simulated-data dynamic n_top_genes rule."""
    if simulate and n_vars < n_top_genes:
        candidate = min((n_vars // 500) * 500, n_vars - 1)
        n_top_genes = candidate if candidate > 0 else max(1, n_vars)
    return int(max(1, n_top_genes))


def preprocess_adata(
    adata,
    simulate: bool = False,
    n_top_genes: int = 2000,
    min_shared_counts: int = 20,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    n_jobs: int = 8,
    cluster_key: str = "vis_annotation",
):
    """
    Run the scVelo dynamical pipeline that GraphVelo builds on.

    Real data uses min_shared_counts=20 / n_top_genes=2000. Simulated data
    derives n_top_genes from the number of genes when fewer than 2000 genes are
    present; the shared-count threshold stays at 20, which is the value used by
    both the real entry point and the simulated block of the original driver.
    """
    if simulate:
        adata.obs_names_make_unique()
        if "X_dimred" in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_dimred"].copy()
        n_top_genes = resolve_n_top_genes(adata.n_vars, n_top_genes, simulate)

    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=min_shared_counts,
        n_top_genes=n_top_genes,
    )

    if adata.n_vars == 0:
        raise ValueError("No genes left after filtering; the dataset is too small.")

    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors, random_state=0)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    scv.tl.recover_dynamics(adata, n_jobs=n_jobs)
    scv.tl.velocity(adata, mode="dynamical")
    scv.tl.latent_time(adata)
    scv.tl.velocity_graph(adata)

    if simulate and cluster_key not in adata.obs.columns:
        adata.obs[cluster_key] = "milestone"

    return adata


def ensure_knn_indices(adata) -> None:
    """Make sure uns['neighbors']['indices'] exists and matches the cell count."""
    neighbors = adata.uns.get("neighbors", {})
    indices = neighbors.get("indices") if "indices" in neighbors else None
    if indices is None or np.asarray(indices).shape[0] != adata.n_obs:
        indices, _ = adj_to_knn(adata.obsp["connectivities"])
        adata.uns["neighbors"]["indices"] = indices


def select_mack_genes(adata, n_mack_genes: int, n_jobs: int) -> list:
    """Compute the MACk score and return the top genes used for the graph."""
    if "mack_score" not in adata.var:
        try:
            tkey = "latent_time" if "latent_time" in adata.obs else "velocity_pseudotime"
            mack_score(adata, ekey="Ms", vkey="velocity", tkey=tkey, n_jobs=n_jobs)
        except Exception as exc:
            print(f"  Warning: MACk score failed ({exc}); using all genes")
            adata.var["mack_score"] = 0.0

    num_genes = min(n_mack_genes, adata.n_vars)
    return adata.var["mack_score"].sort_values(ascending=False)[:num_genes].index.to_list()


def project_velocity(adata, gv) -> None:
    """Project the GraphVelo velocity onto the expression layers and embeddings."""
    adata.layers["velocity_gvs"] = gv.project_velocity(adata.layers["Ms"])
    adata.layers["velocity_gvu"] = gv.project_velocity(adata.layers["Mu"])
    if "X_pca" in adata.obsm:
        adata.obsm["gv_pca"] = gv.project_velocity(adata.obsm["X_pca"])
    if "X_umap" in adata.obsm:
        adata.obsm["gv_umap"] = gv.project_velocity(adata.obsm["X_umap"])


def plot_results(adata, plot_dir: Path, cluster_key: str, basis: str, title: str = "GraphVelo") -> None:
    """Write stream/grid plots for the original and GraphVelo-corrected velocity."""
    if basis not in adata.obsm or cluster_key not in adata.obs:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)

    try:
        scv.pl.velocity_embedding_stream(
            adata,
            basis=basis,
            color=cluster_key,
            palette=PALETTE,
            legend_loc="right margin",
            title=title,
            show=False,
        )
        plt.savefig(plot_dir / f"{title}_{basis}_stream.png", dpi=300, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:  # pragma: no cover - plotting must never be fatal
        print(f"  Warning: velocity stream plot skipped: {exc}")
        plt.close("all")

    if "gv_umap" not in adata.obsm:
        return
    try:
        fig_kwargs = {
            "color": cluster_key,
            "X": adata.obsm["X_umap"],
            "V": adata.obsm["gv_umap"],
            "legend_loc": "right",
            "title": "",
            "figsize": (6, 5),
            "show": False,
        }
        scv.pl.velocity_embedding_stream(adata, **fig_kwargs)
        plt.savefig(plot_dir / f"{title}_{basis}_stream_gv.png", dpi=300, bbox_inches="tight")
        plt.close("all")
        scv.pl.velocity_embedding_grid(adata, **fig_kwargs)
        plt.savefig(plot_dir / f"{title}_{basis}_grid_gv.png", dpi=300, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:  # pragma: no cover - plotting must never be fatal
        print(f"  Warning: GraphVelo plot skipped: {exc}")
        plt.close("all")


def run_graphvelo_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    cluster_key: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dimred_key: str = "X_umap",
    simulate: bool = False,
    n_top_genes: int = 2000,
    min_shared_counts: int = 20,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    n_jobs: int = 8,
    n_mack_genes: int = 200,
    overwrite: bool = False,
    seed: int = 2024,
) -> Path:
    """Run GraphVelo on a single h5ad dataset and write the velocity result."""
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
    patch_estimate_dt()

    adata = None
    try:
        print(f"\nProcessing: {input_path.name}")
        adata = sc.read(input_path)

        missing_layers = [layer for layer in ("spliced", "unspliced") if layer not in adata.layers]
        if missing_layers:
            raise ValueError(f"Missing required layers: {missing_layers}")

        if simulate and cluster_key not in adata.obs.columns:
            adata.obs[cluster_key] = "milestone"
        if cluster_key not in adata.obs.columns:
            raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

        print("  Preprocessing (scVelo dynamical)...")
        adata = preprocess_adata(
            adata,
            simulate=simulate,
            n_top_genes=n_top_genes,
            min_shared_counts=min_shared_counts,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
            cluster_key=cluster_key,
        )

        print("  Fixing the kNN indices...")
        ensure_knn_indices(adata)

        print("  Selecting MACk genes...")
        mac_genes = select_mack_genes(adata, n_mack_genes=n_mack_genes, n_jobs=n_jobs)

        print("  Training GraphVelo...")
        gv = GraphVelo(adata, gene_subset=mac_genes)
        gv.train()
        gv.write_to_adata(adata)

        print("  Projecting velocity...")
        project_velocity(adata, gv)

        basis = dimred_key[2:] if dimred_key.startswith("X_") else dimred_key
        plot_results(adata, dataset_output_dir / "plot", cluster_key, basis)

        adata.uns["graphvelo_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(input_path.resolve()),
            "cluster_key": cluster_key,
            "dimred_key": dimred_key,
            "simulate": bool(simulate),
            "n_mack_genes": int(n_mack_genes),
            "seed": int(seed),
            "output_path": str(output_h5ad.resolve()),
        }

        if "velocity" not in adata.layers:
            raise RuntimeError("GraphVelo did not produce layers['velocity'].")

        adata.write(output_h5ad)
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata
        cleanup_resources()


def run_batch_graphvelo(
    metadata_file: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    seed: int = 2024,
) -> list[Path]:
    """Run GraphVelo for every dataset listed in a metadata table."""
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
            output_path = run_graphvelo_analysis(
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
        description="GraphVelo RNA velocity analysis",
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
        help="Dimensionality reduction key in adata.obsm; use X_dimred for simulated data",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Use the simulated-data preprocessing branch (relaxed filters, milestone labels)",
    )
    parser.add_argument("--n-top-genes", type=int, default=2000, help="Number of highly variable genes")
    parser.add_argument("--min-shared-counts", type=int, default=20, help="Minimum shared counts used for real-data filtering")
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of principal components used for the neighbour graph")
    parser.add_argument("--n-neighbors", type=int, default=30, help="Neighbours used for the scanpy neighbour graph")
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of parallel jobs for scVelo and the MACk score")
    parser.add_argument("--n-mack-genes", type=int, default=200, help="Number of top MACk genes used by GraphVelo")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_graphvelo(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    if not args.cluster_key and not args.simulate:
        parser.error("--cluster-key is required in single-file mode")

    return run_graphvelo_analysis(
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
        n_jobs=args.n_jobs,
        n_mack_genes=args.n_mack_genes,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
