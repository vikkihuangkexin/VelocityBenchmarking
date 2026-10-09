#!/usr/bin/env python3
"""
MultiVeloVAE velocity analysis pipeline for VelocityBenchmarking.

MultiVeloVAE is a probabilistic framework for RNA velocity inference from
multi-lineage, multi-omic and multi-sample single-cell data. It models gene
expression and chromatin accessibility on a shared time scale through a variational
autoencoder with a mechanistic ODE model.

Real and simulated datasets are processed by the same script and switched with the
--simulate flag:

* real data: reads the H5AD, runs the standard scVelo preprocessing so the Mu/Ms
  layers become kNN-smoothed continuous values, then trains MultiVeloVAE.
* simulated data: uses the pre-computed `obsm['X_dimred']` embedding, relaxes the
  gene filter, and labels the cells with the `milestone` cluster.

The chromatin modality is read from `adata.layers['Mc']` (selected with --atac-layer,
with `chromatin` as a fallback), following the upstream multi-omic layout.

Installation:
    pip install multivelovae==0.1.0
    pip install torch==2.0.0
    pip install anndata==0.8.0 scanpy==1.9.3 scvelo==0.2.5 numpy==1.23.5

Usage:
    python MultiVeloVAE.py --input data.h5ad --output-dir ./output --cluster-key celltype
    python MultiVeloVAE.py --input sim.h5ad --output-dir ./output --cluster-key milestone --simulate
    python MultiVeloVAE.py --metadata-file datasets.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
import gc
import shutil
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

def seed_everything(seed: int) -> None:
    """Seed the random number generators used by MultiVeloVAE."""
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_resources() -> None:
    """Release memory held by the last dataset before moving on."""
    gc.collect()
    plt.close("all")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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

    required_columns = ["dataset_name", "file_path"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "cluster_key" not in df.columns:
        df["cluster_key"] = "leiden"
    if "atac_layer" not in df.columns:
        df["atac_layer"] = "Mc"
    if "embed" not in df.columns:
        df["embed"] = "tsne"
    if "simulate" not in df.columns:
        df["simulate"] = False

    df["dataset_name"] = df["dataset_name"].astype(str)
    df["file_path"] = df["file_path"].astype(str)
    df["cluster_key"] = df["cluster_key"].astype(str)
    df["atac_layer"] = df["atac_layer"].astype(str)
    df["embed"] = df["embed"].astype(str)
    df["simulate"] = df["simulate"].map(parse_bool)

    return df


def resolve_atac_layer(adata, atac_layer: str) -> str:
    """Return an available chromatin-activity layer name."""
    if atac_layer in adata.layers:
        return atac_layer
    if "chromatin" in adata.layers:
        print(f"  Layer '{atac_layer}' not found; using 'chromatin' instead.")
        return "chromatin"

    raise ValueError(
        f"Missing chromatin-activity layer: neither '{atac_layer}' nor 'chromatin' is present in adata.layers."
    )


def resolve_cluster_key(adata, cluster_key: str, simulate: bool) -> str:
    """Return a usable ``obs`` column for plotting and cell labels."""
    if cluster_key in adata.obs.columns:
        return cluster_key

    for fallback in ("celltype", "cell_type", "milestone", "clusters", "leiden"):
        if fallback in adata.obs.columns:
            print(f"  Column '{cluster_key}' not found; using '{fallback}' instead.")
            return fallback

    label = "milestone" if simulate else "cluster"
    print(f"  No cluster column found; writing a constant '{label}' label.")
    adata.obs[label] = label
    return label


def resolve_embedding(adata, embed: str, dimred_key: str, simulate: bool) -> str:
    """Make sure the embedding MultiVeloVAE wants exists in ``obsm``."""
    if f"X_{embed}" in adata.obsm:
        return embed

    if simulate and "X_dimred" in adata.obsm:
        adata.obsm["X_umap"] = adata.obsm["X_dimred"].copy()
        if embed != "umap":
            adata.obsm[f"X_{embed}"] = adata.obsm["X_dimred"].copy()
        return embed

    if dimred_key in adata.obsm:
        adata.obsm[f"X_{embed}"] = np.asarray(adata.obsm[dimred_key])
        return embed

    print(f"  Embedding 'X_{embed}' not found; computing UMAP.")
    sc.tl.pca(adata, n_comps=min(30, max(1, min(adata.n_obs - 1, adata.n_vars - 1))))
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    if embed != "umap":
        adata.obsm[f"X_{embed}"] = adata.obsm["X_umap"].copy()
    return embed


def determine_preprocessing_params(adata) -> tuple[int, int]:
    """Choose safe PCA / neighbourhood sizes for small datasets."""
    n_pcs = max(1, min(30, adata.n_obs - 1, adata.n_vars - 1))
    n_neighbors = max(1, min(30, adata.n_obs - 1))
    return n_pcs, n_neighbors


def preprocess_adata(
    adata,
    cluster_key: str,
    atac_layer: str,
    embed: str,
    dimred_key: str,
    simulate: bool,
    n_top_genes: int = 2000,
    min_shared_counts: Optional[int] = None,
):
    """Run the upstream preprocessing, including the crucial ``scv.pp.moments`` step.

    ``VAEChrom`` reads ``Mc``/``Mu``/``Ms`` without normalizing them. With raw integer
    counts some genes can be constant across the selected cells, which makes the gene
    covariance exactly zero and raises
    ``numpy.linalg.LinAlgError: 1-th leading minor of the array is not positive definite``
    during model initialization. ``scv.pp.moments`` turns the moment layers into
    kNN-smoothed continuous values, which removes those degenerate genes.
    """
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    resolved_atac_layer = resolve_atac_layer(adata, atac_layer)

    if simulate:
        if "X_dimred" not in adata.obsm:
            raise ValueError("Simulated data requires obsm['X_dimred'].")
        adata.obsm["X_umap"] = adata.obsm["X_dimred"].copy()
        min_shared_counts = None
        if adata.n_vars < 2000:
            n_top_genes = max(1, min((adata.n_vars // 500) * 500, adata.n_vars - 1))

    scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, n_top_genes=n_top_genes)

    n_pcs, n_neighbors = determine_preprocessing_params(adata)
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors, method="umap")
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)

    active_cluster_key = resolve_cluster_key(adata, cluster_key, simulate)
    active_embed = resolve_embedding(adata, embed, dimred_key, simulate)

    return adata, resolved_atac_layer, active_cluster_key, active_embed


def write_velocity_layer(adata, key: str):
    """Expose the MultiVeloVAE velocity as ``layers['velocity']``.

    ``save_anndata`` stores the fitted velocity under the model key, i.e.
    ``layers[f'{key}_velocity']``. The benchmark reads ``layers['velocity']``, so the
    native key is copied over (and kept as well).
    """
    native_key = f"{key}_velocity"
    if native_key not in adata.layers:
        candidates = [k for k in adata.layers if k.endswith("_velocity")]
        if not candidates:
            raise RuntimeError(
                f"MultiVeloVAE did not produce layers['{native_key}'], so layers['velocity'] cannot be exported."
            )
        native_key = candidates[0]
        print(f"  Using '{native_key}' as the native velocity layer.")

    adata.layers["velocity"] = np.asarray(adata.layers[native_key])
    return adata


def run_multivelovae_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    cluster_key: str = "leiden",
    dataset_name: Optional[str] = None,
    atac_layer: str = "Mc",
    embed: str = "tsne",
    dimred_key: str = "X_umap",
    batch_size: int = 32,
    n_top_genes: int = 2000,
    min_shared_counts: Optional[int] = None,
    simulate: bool = False,
    key: str = "vae",
    device: str = "cuda:0",
    overwrite: bool = False,
    seed: int = 2022,
) -> Path:
    """Train MultiVeloVAE on one dataset and export the predicted velocity."""
    import torch

    import multivelovae as vv

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if dataset_name is None:
        dataset_name = derive_output_stem(input_path)

    dataset_output_dir = output_dir / str(dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Native MultiVeloVAE file name, preserved from code/MultiVeloVAE_simdata.py.
    output_h5ad = dataset_output_dir / f"{dataset_name}_MultiVeloVAE.h5ad"
    model_dir = dataset_output_dir / "model"
    figure_dir = dataset_output_dir / "figures"

    if output_h5ad.exists() and not overwrite:
        print(f"Skipping existing output: {output_h5ad}")
        return output_h5ad

    seed_everything(seed)

    adata = None
    adata_atac = None

    try:
        print(f"\nProcessing: {input_path.name}")
        adata = sc.read_h5ad(input_path)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        print("  Preprocessing...")
        adata, resolved_atac_layer, active_cluster_key, active_embed = preprocess_adata(
            adata,
            cluster_key=cluster_key,
            atac_layer=atac_layer,
            embed=embed,
            dimred_key=dimred_key,
            simulate=simulate,
            n_top_genes=n_top_genes,
            min_shared_counts=min_shared_counts,
        )

        adata_atac = adata.copy()
        adata_atac.X = adata_atac.layers[resolved_atac_layer]

        model_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        print("  Training MultiVeloVAE...")
        model = vv.VAEChrom(
            adata,
            adata_atac,
            device=device,
            plot_init=False,
            gene_plot=[],
            cluster_key=active_cluster_key,
            figure_path=str(figure_dir),
            embed=active_embed,
            key=key,
        )
        model.config["batch_size"] = batch_size
        model.train(plot=False, gene_plot=[], figure_path=str(figure_dir), embed=active_embed)

        print("  Saving model and results...")
        model.save_model(str(model_dir))
        model.save_anndata(str(dataset_output_dir), file_name=output_h5ad.name)

        adata = sc.read_h5ad(output_h5ad)
        adata = write_velocity_layer(adata, key)
        adata.uns["multivelovae_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(input_path.resolve()),
            "cluster_key": active_cluster_key,
            "atac_layer": resolved_atac_layer,
            "embed": active_embed,
            "dimred_key": dimred_key,
            "simulate": bool(simulate),
            "batch_size": int(batch_size),
            "model_key": str(key),
            "seed": int(seed),
            "output_path": str(output_h5ad.resolve()),
        }
        adata.write_h5ad(output_h5ad, compression="lzf")
        shutil.copyfile(output_h5ad, dataset_output_dir / "rc.h5ad")
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata
        del adata_atac
        cleanup_resources()


def run_batch_multivelovae(
    metadata_file: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    seed: int = 2024,
) -> list[Path]:
    """Run MultiVeloVAE over every entry of a metadata table."""
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
            output_path = run_multivelovae_analysis(
                input_path=file_path,
                output_dir=output_dir,
                cluster_key=row["cluster_key"],
                dataset_name=row["dataset_name"],
                atac_layer=row["atac_layer"],
                embed=row["embed"],
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
        description="MultiVeloVAE multi-omic velocity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input H5AD file")
    input_group.add_argument("--metadata-file", help="Metadata CSV/TSV file for batch processing")

    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", default=None, help="Dataset folder name in single-file mode")
    parser.add_argument(
        "--cluster-key",
        default="leiden",
        help="Column name in adata.obs used for labels and plots; use milestone for simulated data",
    )
    parser.add_argument(
        "--atac-layer",
        default="Mc",
        help="adata.layers key holding the chromatin activity matrix; falls back to 'chromatin'",
    )
    parser.add_argument("--embed", default="tsne", help="Embedding name in adata.obsm, stored as X_<embed>")
    parser.add_argument("--dimred-key", default="X_umap", help="Fallback embedding key in adata.obsm")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--n-top-genes", type=int, default=2000, help="Number of highly variable genes")
    parser.add_argument("--min-shared-counts", type=int, default=None, help="Minimum shared counts filter")
    parser.add_argument("--key", default="vae", help="Model key used for the output layers, e.g. vae_velocity")
    parser.add_argument("--device", default="cuda:0", help="Torch device, for example cuda:0 or cpu")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Simulated-data branch: uses obsm['X_dimred'], relaxes the filter and labels cells milestone",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2022, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_multivelovae(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    return run_multivelovae_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        atac_layer=args.atac_layer,
        embed=args.embed,
        dimred_key=args.dimred_key,
        batch_size=args.batch_size,
        n_top_genes=args.n_top_genes,
        min_shared_counts=args.min_shared_counts,
        simulate=args.simulate,
        key=args.key,
        device=args.device,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
