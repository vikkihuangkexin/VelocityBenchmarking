#!/usr/bin/env python3
"""
SvelvetVAE (velvetVAE) velocity analysis pipeline for VelocityBenchmarking.

SvelvetVAE is a variational-autoencoder based RNA velocity method. It models
the spliced/unspliced counts with a latent-space neighbourhood constraint and a
learned kinetic parameter gamma.

Installation:
    pip install "pip<24.1"
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118
    pip install "jax[cuda11_local]==0.4.13" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install chex==0.1.7 flax==0.6.11
    pip install scvi-tools==0.19.0 scanpy==1.9.8 scvelo==0.2.5 anndata==0.8.0
    git clone https://github.com/rorymaizels/velvetVAE.git /opt/velvetVAE
    pip install --no-deps -e /opt/velvetVAE

Usage:
    python SvelvetVAE.py --input data.h5ad --output-dir ./output --cluster-key celltype
    python SvelvetVAE.py --input sim.h5ad --output-dir ./output --cluster-key milestone --simulate
    python SvelvetVAE.py --metadata-file datasets.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
import gc
import random
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
import torch
import velvetvae as vt
from scipy.sparse import issparse

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_resources() -> None:
    """Release memory held by the previous dataset."""
    gc.collect()
    plt.close("all")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def dense_float32(layer) -> np.ndarray:
    """Densify a (possibly sparse) count matrix and cast it to float32."""
    if issparse(layer):
        layer = layer.toarray()
    return np.asarray(layer, dtype=np.float32)


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
    cluster_key: str = "vis_annotation",
):
    """
    Run the scVelo standard preprocessing used by SvelvetVAE.

    Real data uses min_shared_counts=20 / n_top_genes=2000. Simulated data
    relaxes the shared-count filter and derives n_top_genes from the number of
    genes when fewer than 2000 genes are present.
    """
    if simulate:
        adata.obs_names_make_unique()
        if "X_dimred" in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_dimred"].copy()
        n_top_genes = resolve_n_top_genes(adata.n_vars, n_top_genes, simulate)
        min_shared_counts = None

    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=min_shared_counts,
        n_top_genes=n_top_genes,
    )

    if adata.n_vars == 0:
        raise ValueError("No genes left after filtering; the dataset is too small.")

    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    if simulate and cluster_key not in adata.obs.columns:
        adata.obs[cluster_key] = "milestone"

    return adata


def prepare_velocity_layers(adata) -> None:
    """
    Densify the spliced/unspliced layers, build the total layer and cast all
    three to float32.

    velvetVAE builds float64 tensors from the float64 numpy arrays produced by
    scVelo, which causes "Found dtype Float but expected Double" errors during
    training. Casting the source layers up front fixes every such leak.
    """
    adata.layers["spliced"] = dense_float32(adata.layers["spliced"])
    adata.layers["unspliced"] = dense_float32(adata.layers["unspliced"])
    if "total" not in adata.layers:
        adata.layers["total"] = adata.layers["spliced"] + adata.layers["unspliced"]
    adata.layers["total"] = dense_float32(adata.layers["total"])


def cast_model_dtypes(model) -> None:
    """
    Force the velvetVAE parameters that are created as float64 into float32.

    loggamma/ss_gamma are built in setup_model and the neighbourhood-constraint
    weights are built in NeighborhoodConstraint.__init__; both default to
    float64 and break the forward pass once the constraint is active.
    """
    try:
        module = model.module
        module.loggamma = torch.nn.Parameter(module.loggamma.data.float())
        module.ss_gamma = module.ss_gamma.float()
        module.nc.X = module.nc.X.float()
        module.nc.b = module.nc.b.float()
    except Exception as exc:  # pragma: no cover - depends on velvetVAE version
        print(f"  Warning: could not cast velvetVAE parameter dtypes: {exc}")


def plot_results(adata, plot_dir: Path, cluster_key: str, basis: str, title: str = "SvelvetVAE") -> None:
    """Write a velocity stream plot when the embedding is available."""
    if basis not in adata.obsm or cluster_key not in adata.obs:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        if "velocity_graph" not in adata.uns:
            scv.tl.velocity_graph(adata)
        adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)
        scv.pl.velocity_embedding_stream(
            adata,
            basis=basis,
            vkey="velocity",
            color=cluster_key,
            palette=PALETTE,
            legend_loc="right margin",
            title=title,
            show=False,
        )
        plt.savefig(plot_dir / f"{title}_{basis}_stream.png", dpi=300, bbox_inches="tight")
    except Exception as exc:  # pragma: no cover - plotting must never be fatal
        print(f"  Warning: velocity stream plot skipped: {exc}")
    finally:
        plt.close("all")


def run_svelvetvae_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    cluster_key: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dimred_key: str = "X_umap",
    simulate: bool = False,
    n_latent: int = 50,
    max_epochs: int = 100,
    freeze_vae_after_epochs: int = 20,
    constrain_vf_after_epochs: int = 20,
    lr: float = 0.01,
    knn_neighbors: int = 100,
    n_top_genes: int = 2000,
    min_shared_counts: int = 20,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    linear_decoder: bool = True,
    neighborhood_space: str = "latent_space",
    gamma_mode: str = "learned",
    gamma_min: float = 0.1,
    gamma_max: float = 1.0,
    overwrite: bool = False,
    save_plots: bool = False,
    seed: int = 2024,
) -> Path:
    """Run SvelvetVAE on a single h5ad dataset and write the velocity result."""
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

        missing_layers = [layer for layer in ("spliced", "unspliced") if layer not in adata.layers]
        if missing_layers:
            raise ValueError(f"Missing required layers: {missing_layers}")

        if simulate and cluster_key not in adata.obs.columns:
            adata.obs[cluster_key] = "milestone"
        if cluster_key not in adata.obs.columns:
            raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

        print("  Preprocessing...")
        adata = preprocess_adata(
            adata,
            simulate=simulate,
            n_top_genes=n_top_genes,
            min_shared_counts=min_shared_counts,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            cluster_key=cluster_key,
        )

        print("  Preparing velocity layers...")
        prepare_velocity_layers(adata)

        print("  Building the neighbourhood graph...")
        vt.pp.neighborhood(adata, n_neighbors=knn_neighbors)
        vt.ut.set_seed(seed)
        vt.md.Svelvet.setup_anndata(
            adata,
            x_layer="total",
            u_layer="unspliced",
            knn_layer="knn_index",
        )

        print("  Training SvelvetVAE...")
        model = vt.md.Svelvet(
            adata,
            n_latent=n_latent,
            linear_decoder=linear_decoder,
            neighborhood_space=neighborhood_space,
            gamma_mode=gamma_mode,
        )
        model.setup_model(gamma_kwargs={"gamma_min": gamma_min, "gamma_max": gamma_max})
        cast_model_dtypes(model)
        model.train(
            batch_size=adata.shape[0],
            max_epochs=max_epochs,
            freeze_vae_after_epochs=freeze_vae_after_epochs,
            constrain_vf_after_epochs=constrain_vf_after_epochs,
            lr=lr,
        )

        print("  Predicting velocity...")
        velocity = model.predict_velocity()
        if issparse(velocity):
            velocity = velocity.toarray()
        adata.layers["velocity"] = np.nan_to_num(velocity, nan=0.0, neginf=0.0, posinf=0.0)

        basis = dimred_key[2:] if dimred_key.startswith("X_") else dimred_key
        # Addition relative to code/svelvet.py, which had this plotting block commented
        # out: the stream plot (and the velocity_graph it needs) is opt-in via --save-plots.
        if save_plots:
            plot_results(adata, dataset_output_dir / "plot", cluster_key, basis)

        adata.uns["svelvetvae_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(input_path.resolve()),
            "cluster_key": cluster_key,
            "dimred_key": dimred_key,
            "simulate": bool(simulate),
            "n_latent": int(n_latent),
            "max_epochs": int(max_epochs),
            "lr": float(lr),
            "seed": int(seed),
            "output_path": str(output_h5ad.resolve()),
        }

        if "velocity" not in adata.layers:
            raise RuntimeError("SvelvetVAE did not produce layers['velocity'].")

        adata.write(output_h5ad)
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata
        cleanup_resources()


def run_batch_svelvetvae(
    metadata_file: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    save_plots: bool = False,
    seed: int = 2024,
) -> list[Path]:
    """Run SvelvetVAE for every dataset listed in a metadata table."""
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
            output_path = run_svelvetvae_analysis(
                input_path=file_path,
                output_dir=output_dir,
                cluster_key=row["cluster_key"],
                dataset_name=row["dataset_name"],
                dimred_key=row["dimred_key"],
                simulate=bool(row["simulate"]),
                overwrite=overwrite,
                save_plots=save_plots,
                seed=seed,
            )
            outputs.append(output_path)
        except Exception as exc:
            print(f"Failed: {row['dataset_name']}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SvelvetVAE velocity analysis",
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
    parser.add_argument("--n-latent", type=int, default=50, help="Latent space dimension")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--freeze-vae-after-epochs", type=int, default=20, help="Epoch after which the VAE is frozen")
    parser.add_argument("--constrain-vf-after-epochs", type=int, default=20, help="Epoch after which the velocity field is constrained")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--knn-neighbors", type=int, default=100, help="Neighbours for the velvetVAE neighbourhood graph")
    parser.add_argument("--n-top-genes", type=int, default=2000, help="Number of highly variable genes")
    parser.add_argument("--min-shared-counts", type=int, default=20, help="Minimum shared counts used for real-data filtering")
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of principal components used for the neighbour graph")
    parser.add_argument("--n-neighbors", type=int, default=30, help="Neighbours used for the scanpy neighbour graph")
    parser.add_argument("--no-linear-decoder", action="store_true", default=False, help="Disable the linear decoder")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=False,
        help="Also write the velocity stream plot; disabled by default (commented out in code/svelvet.py)",
    )
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_svelvetvae(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            save_plots=args.save_plots,
            seed=args.seed,
        )

    if not args.cluster_key and not args.simulate:
        parser.error("--cluster-key is required in single-file mode")

    return run_svelvetvae_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        dimred_key=args.dimred_key,
        simulate=args.simulate,
        n_latent=args.n_latent,
        max_epochs=args.max_epochs,
        freeze_vae_after_epochs=args.freeze_vae_after_epochs,
        constrain_vf_after_epochs=args.constrain_vf_after_epochs,
        lr=args.lr,
        knn_neighbors=args.knn_neighbors,
        n_top_genes=args.n_top_genes,
        min_shared_counts=args.min_shared_counts,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        linear_decoder=not args.no_linear_decoder,
        overwrite=args.overwrite,
        save_plots=args.save_plots,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
