#!/usr/bin/env python3
"""
InterVelo velocity analysis pipeline for VelocityBenchmarking.

Installation:
    pip install git+https://github.com/sd68515/InterVelo_py311.git
    pip install "numpy==1.26.4" "numba==0.59.1"
    # Install a GPU-enabled PyTorch build according to your system.
    # The example workflow was validated with torch 2.5.1 + CUDA 12.1.

Usage:
    python InterVelo.py --input data.h5ad --output-dir ./output --modality real --cluster-key celltype --dimred-key X_umap
    python InterVelo.py --input sim.h5ad --output-dir ./output --modality sim --cluster-key milestone --dimred-key X_dimred
    python InterVelo.py --input multiome.h5ad --output-dir ./output --modality atac --cluster-key celltype --dimred-key X_umap
    python InterVelo.py --metadata-file datasets.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
import gc
import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy import sparse

if TYPE_CHECKING:
    import torch

PALETTE = [
    "#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4",
    "#66c2a5", "#3288bd", "#abdda4", "#e6f598", "#fee08b",
    "#f46d43", "#e7298a", "#a6cee3", "#1f78b4", "#b2df8a",
    "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
    "#cab2d6", "#6a3d9a", "#ffff99", "#b15928", "#8dd3c7",
    "#bc80bd", "#ccebc5", "#ffed6f", "#999999",
    "#8B0000", "#006400", "#FF69B4", "#00CED1", "#FFD700",
]


@lru_cache(maxsize=1)
def load_intervelo_api():
    """
    Load the installed InterVelo package without being shadowed by this file.
    """
    script_dir = str(Path(__file__).resolve().parent)
    removed_path = False
    if script_dir in sys.path:
        sys.path.remove(script_dir)
        removed_path = True

    local_module = None
    restore_local_module = __name__ == "InterVelo" and "InterVelo" in sys.modules
    if restore_local_module:
        local_module = sys.modules.pop("InterVelo")

    try:
        train_module = importlib.import_module("InterVelo.train")
        data_module = importlib.import_module("InterVelo.data")
        utils_module = importlib.import_module("InterVelo._utils")
    finally:
        if restore_local_module and local_module is not None:
            sys.modules["InterVelo"] = local_module
        if removed_path:
            sys.path.insert(0, script_dir)

    return (
        train_module.train,
        train_module.Constants,
        data_module.preprocess_data,
        utils_module.update_dict,
        utils_module.autoset_coeff_s,
    )


MODALITIES = ("real", "sim", "atac")


def validate_dataset_name(name: str) -> str:
    if not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
        raise ValueError("Dataset name must be a nonempty path component")
    return name


def cleanup_resources():
    import torch

    gc.collect()
    plt.close("all")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_everything(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    df = pd.read_csv(metadata_path, sep=detect_separator(metadata_path), dtype=str,
                     keep_default_na=False)

    required_columns = ["dataset_name", "file_path", "modality", "cluster_key", "dimred_key"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Metadata file contains no datasets")

    for column in required_columns:
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Manifest column '{column}' must contain a value in every row")
        df[column] = df[column].astype(str).str.strip()
    df["dataset_name"] = df["dataset_name"].map(validate_dataset_name)
    if df["dataset_name"].duplicated().any():
        raise ValueError("Manifest dataset names must be unique")
    invalid = sorted(set(df["modality"]) - set(MODALITIES))
    if invalid:
        raise ValueError(f"Unsupported modalities: {invalid}; choose from {MODALITIES}")
    df["file_path"] = df["file_path"].map(
        lambda path: str((metadata_path.parent / path).resolve())
        if not Path(path).is_absolute() else path
    )

    return df


def ensure_required_layers(adata, modality: str) -> None:
    missing_layers = [layer for layer in ("spliced", "unspliced") if layer not in adata.layers]
    if missing_layers:
        raise ValueError(f"Missing required layers: {missing_layers}")

    if modality == "atac" and "Mc" not in adata.layers:
        raise ValueError("ATAC input requires adata.layers['Mc']")


def determine_preprocessing_params(adata) -> tuple[int, int, int]:
    n_obs = int(adata.n_obs)
    n_vars = int(adata.n_vars)
    n_pcs = max(1, min(30, n_obs - 1, n_vars - 1))
    n_neighbors = max(1, min(30, n_obs - 1))
    batch_size = max(1, min(1024, n_obs))
    return n_pcs, n_neighbors, batch_size


def validate_dimred(adata, dimred_key: str) -> str:
    basis_name = dimred_key[2:] if dimred_key.startswith("X_") else dimred_key

    if dimred_key not in adata.obsm:
        raise ValueError(f"Requested coordinate key '{dimred_key}' is missing from adata.obsm")

    embedding = np.asarray(adata.obsm[dimred_key])
    if embedding.ndim != 2 or embedding.shape[1] < 2:
        raise ValueError(f"Embedding '{dimred_key}' must be a 2D matrix with at least two columns.")
    if embedding.shape[0] != adata.n_obs or not np.isfinite(embedding).all():
        raise ValueError(f"Embedding '{dimred_key}' must have one finite row per cell")

    # scVelo addresses all projection coordinates through X_<basis>.
    if dimred_key != f"X_{basis_name}":
        adata.obsm[f"X_{basis_name}"] = embedding.copy()

    return basis_name


def preprocess_adata(adata, dimred_key: str, modality: str):
    _, _, preprocess_data, _, _ = load_intervelo_api()

    if modality not in MODALITIES:
        raise ValueError(f"Unsupported modality '{modality}'")
    ensure_required_layers(adata, modality)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    basis_name = validate_dimred(adata, dimred_key)
    initial_coordinates = pd.DataFrame(
        np.asarray(adata.obsm[dimred_key]).copy(), index=adata.obs_names.copy()
    )

    if modality == "real":
        adata.X = adata.layers["spliced"].copy()
    # Optional X-based deduplication: adata = adata[~adata.to_df().duplicated()].copy()
    # Converting X to a dense DataFrame can exhaust memory on large datasets.

    n_pcs, n_neighbors, batch_size = determine_preprocessing_params(adata)

    if modality == "sim":
        scv.pp.filter_and_normalize(adata)
    else:
        scv.pp.filter_and_normalize(adata, min_shared_counts=20)

    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars), subset=True)
    if modality == "real":
        spliced_sum = np.asarray(adata.layers["spliced"].sum(axis=1)).ravel()
        adata = adata[spliced_sum != 0].copy()
        if adata.n_obs == 0:
            raise ValueError("Real-data filtering removed every cell with zero spliced counts")
    n_pcs, n_neighbors, batch_size = determine_preprocessing_params(adata)
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    layers = ["Ms", "Mu", "Mc"] if modality == "atac" else ["Ms", "Mu"]
    adata = preprocess_data(adata, layers=layers, filter_on_r2=False)
    missing_cells = adata.obs_names.difference(initial_coordinates.index)
    if len(missing_cells):
        raise ValueError("Preprocessing produced cells absent from the input coordinates")
    adata.obsm[dimred_key] = initial_coordinates.loc[adata.obs_names].to_numpy()
    validate_dimred(adata, dimred_key)

    return adata, basis_name, batch_size


def to_dense_float32(layer) -> np.ndarray:
    if sparse.issparse(layer):
        return layer.toarray().astype(np.float32, copy=False)
    return np.asarray(layer, dtype=np.float32)


def build_inputdata(adata, modality: str) -> torch.Tensor:
    import torch

    input_layers = ["Ms", "Mu", "Mc"] if modality == "atac" else ["Ms", "Mu"]
    tensors = [torch.from_numpy(to_dense_float32(adata.layers[layer])) for layer in input_layers]
    return torch.cat(tensors, dim=1)


def build_configs(adata, method_name: str, saved_dir: Path, batch_size: int):
    import torch

    _, Constants, _, update_dict, autoset_coeff_s = load_intervelo_api()

    configs = {
        "name": method_name,
        "n_gpu": 1 if torch.cuda.is_available() else 0,
        "loss_pearson": {
            "coeff_s": autoset_coeff_s(adata),
        },
        "arch": {
            "args": {
                "n_latent": 20,
                "pred_unspliced": False,
            }
        },
        "data_loader": {
            "args": {
                "batch_size": batch_size,
                "num_workers": 0,
                "validation_split": 0.1 if adata.n_obs >= 10 else 0.0,
            }
        },
        "trainer": {
            "tensorboard": False,
            "verbosity": 0,
            "save_dir": str(saved_dir),
        },
    }
    return update_dict(Constants.default_configs, configs)


def ensure_velocity_embedding(adata, basis_name: str) -> str:
    scv.tl.velocity_graph(adata, vkey="velocity")
    scv.tl.velocity_embedding(adata, basis=basis_name, vkey="velocity")
    return f"velocity_{basis_name}"


def add_normalized_pseudotime(adata, output_key: str = "intervelo_pseudotime_normalized") -> None:
    pseudotime = adata.obs["pseudotime"].to_numpy(dtype=float)
    denominator = np.max(pseudotime) - np.min(pseudotime)
    if denominator == 0:
        adata.obs[output_key] = 0.0
    else:
        adata.obs[output_key] = (pseudotime - np.min(pseudotime)) / denominator


def save_current_figure(save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close("all")


def get_plot_formats(save_pdf: bool) -> list[str]:
    return ["png", "pdf"] if save_pdf else ["png"]


def plot_results(
    adata,
    dataset_label: str,
    plot_dir: Path,
    basis_name: str,
    velocity_embed_key: str,
    cluster_key: str,
    save_pdf: bool,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)
    add_normalized_pseudotime(adata)

    for fmt in get_plot_formats(save_pdf):
        scatter_path = plot_dir / f"{dataset_label}_{basis_name}.{fmt}"
        scv.pl.scatter(
            adata,
            basis=basis_name,
            vkey="velocity",
            color=cluster_key,
            palette=PALETTE,
            size=100,
            alpha=0.6,
            legend_loc="right margin",
            legend_fontsize=9,
            title="InterVelo",
            show=False,
        )
        save_current_figure(scatter_path)

    for fmt in get_plot_formats(save_pdf):
        stream_path = plot_dir / f"{dataset_label}_stream.{fmt}"
        scv.pl.velocity_embedding_stream(
            adata,
            basis=basis_name,
            vkey="velocity",
            V=adata.obsm[velocity_embed_key],
            color=cluster_key,
            palette=PALETTE,
            size=100,
            alpha=0.6,
            legend_fontsize=9,
            legend_loc="right margin",
            density=2,
            arrow_size=1,
            linewidth=1,
            title="InterVelo",
            show=False,
        )
        save_current_figure(stream_path)

    for fmt in get_plot_formats(save_pdf):
        grid_path = plot_dir / f"{dataset_label}_grid.{fmt}"
        scv.pl.velocity_embedding_grid(
            adata,
            basis=basis_name,
            vkey="velocity",
            V=adata.obsm[velocity_embed_key],
            color=cluster_key,
            palette=PALETTE,
            size=100,
            alpha=0.6,
            legend_fontsize=9,
            legend_loc="right margin",
            density=0.8,
            arrow_size=1,
            linewidth=0.3,
            title="InterVelo",
            show=False,
        )
        save_current_figure(grid_path)

    for fmt in get_plot_formats(save_pdf):
        pseudotime_path = plot_dir / f"{dataset_label}_pseudotime.{fmt}"
        scv.pl.scatter(
            adata,
            basis=basis_name,
            color="intervelo_pseudotime_normalized",
            cmap="gnuplot",
            size=100,
            colorbar=True,
            title="InterVelo",
            show=False,
        )
        save_current_figure(pseudotime_path)


def derive_output_stem(input_path: Path) -> str:
    stem = input_path.stem
    if stem.endswith("_dataset"):
        return stem[:-8]
    return stem


def run_intervelo_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    cluster_key: str,
    modality: str,
    dataset_name: Optional[str] = None,
    dimred_key: str = "X_umap",
    save_pdf: bool = False,
    overwrite: bool = False,
    seed: int = 2024,
) -> Path:
    train, _, _, _, _ = load_intervelo_api()

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if modality not in MODALITIES:
        raise ValueError(f"Unsupported modality '{modality}'; choose from {MODALITIES}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if dataset_name is None:
        dataset_name = derive_output_stem(input_path)
    dataset_name = validate_dataset_name(str(dataset_name))

    dataset_output_dir = output_dir / str(dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    output_stem = derive_output_stem(input_path)
    output_h5ad = dataset_output_dir / f"{output_stem}_plot.h5ad"
    plot_dir = dataset_output_dir / "plot"
    saved_dir = dataset_output_dir / "saved"
    method_name = f"InterVelo_{dataset_name}"

    if output_h5ad.exists() and not overwrite:
        print(f"Skipping existing output: {output_h5ad}")
        return output_h5ad

    seed_everything(seed)

    adata = None
    inputdata = None

    try:
        print(f"\nProcessing: {input_path.name}")
        adata = sc.read(input_path)

        if cluster_key not in adata.obs.columns:
            raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

        print("  Preprocessing...")
        adata, basis_name, batch_size = preprocess_adata(
            adata=adata,
            dimred_key=dimred_key,
            modality=modality,
        )

        print("  Building model input...")
        inputdata = build_inputdata(adata, modality=modality)
        configs = build_configs(
            adata=adata,
            method_name=method_name,
            saved_dir=saved_dir,
            batch_size=batch_size,
        )

        print("  Training InterVelo...")
        coordinates = np.asarray(adata.obsm[dimred_key]).copy()
        train(adata, inputdata, configs)
        adata.obsm[dimred_key] = coordinates
        validate_dimred(adata, dimred_key)

        print("  Computing velocity embedding...")
        velocity_embed_key = ensure_velocity_embedding(adata, basis_name)

        print("  Generating plots...")
        plot_results(
            adata=adata,
            dataset_label=output_stem,
            plot_dir=plot_dir,
            basis_name=basis_name,
            velocity_embed_key=velocity_embed_key,
            cluster_key=cluster_key,
            save_pdf=save_pdf,
        )

        adata.uns["intervelo_run"] = {
            "dataset_name": str(dataset_name),
            "cluster_key": cluster_key,
            "dimred_key": dimred_key,
            "modality": modality,
            "input_layers": ["Ms", "Mu", "Mc"] if modality == "atac" else ["Ms", "Mu"],
            "seed": int(seed),
        }

        adata.write(output_h5ad, compression="lzf")
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata
        del inputdata
        cleanup_resources()


def run_batch_intervelo(
    metadata_file: str | Path,
    output_dir: str | Path,
    save_pdf: bool = False,
    overwrite: bool = False,
    seed: int = 2024,
) -> list[Path]:
    metadata_file = Path(metadata_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_file(metadata_file)
    outputs: list[Path] = []
    print(f"Batch mode: {len(metadata_df)} datasets")

    for _, row in metadata_df.iterrows():
        file_path = Path(row["file_path"])
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        output_path = run_intervelo_analysis(
            input_path=file_path,
            output_dir=output_dir,
            cluster_key=row["cluster_key"],
            modality=row["modality"],
            dataset_name=row["dataset_name"],
            dimred_key=row["dimred_key"],
            save_pdf=save_pdf,
            overwrite=overwrite,
            seed=seed,
        )
        outputs.append(output_path)

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="InterVelo velocity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input H5AD file")
    input_group.add_argument("--metadata-file", help="Metadata CSV/TSV file for batch processing")

    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", default=None, help="Dataset folder name for single-file mode")
    parser.add_argument("--modality", choices=MODALITIES, help="Input modality for single-file mode")
    parser.add_argument(
        "--cluster-key",
        default=None,
        help="Column name in adata.obs used for labels and visualization in single-file mode",
    )
    parser.add_argument(
        "--dimred-key",
        help="Existing coordinate key in adata.obsm for single-file mode",
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        default=False,
        help="Also save PDF figures in addition to the default PNG figures",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_intervelo(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            save_pdf=args.save_pdf,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    if not args.cluster_key or not args.modality or not args.dimred_key:
        parser.error("--cluster-key, --modality, and --dimred-key are required in single-file mode")

    return run_intervelo_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        modality=args.modality,
        dataset_name=args.dataset_name,
        dimred_key=args.dimred_key,
        save_pdf=args.save_pdf,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
