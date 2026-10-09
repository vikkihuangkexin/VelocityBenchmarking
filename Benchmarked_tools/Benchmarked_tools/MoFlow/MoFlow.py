#!/usr/bin/env python3
"""
MoFlow multi-omics velocity analysis pipeline for VelocityBenchmarking.

MoFlow couples the RNA modality with an auxiliary omic layer (for example
chromatin accessibility gene activity) to estimate RNA velocity. The RNA
modality is passed as one AnnData object and the auxiliary omic matrix is
passed as a second AnnData object whose expression matrix is set to the
auxiliary layer.

Installation:
    pip install moflow

Usage:
    python MoFlow.py --input data.h5ad --output-dir ./output --cluster-key celltype
    python MoFlow.py --input sim.h5ad --output-dir ./output --cluster-key milestone --simulate
    python MoFlow.py --input multiome.h5ad --output-dir ./output --cluster-key celltype --extra-layers Mc
    python MoFlow.py --metadata-file datasets.csv --output-dir ./output
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

DEFAULT_AUXILIARY_LAYER = "Mc"
DEFAULT_DIMARGS = "X_tsne:X_umap"


@lru_cache(maxsize=1)
def load_moflow_api():
    """
    Load the installed MoFlow package without being shadowed by this file.

    When this script is executed directly its own directory is prepended to
    ``sys.path``, so ``import MoFlow`` would resolve to this file instead of the
    installed package. The script directory is temporarily removed to import
    the real package.
    """
    script_dir = str(Path(__file__).resolve().parent)
    removed_path = False
    if script_dir in sys.path:
        sys.path.remove(script_dir)
        removed_path = True

    local_module = None
    restore_local_module = __name__ == "MoFlow" and "MoFlow" in sys.modules
    if restore_local_module:
        local_module = sys.modules.pop("MoFlow")

    try:
        import MoFlow as mf
    finally:
        if restore_local_module and local_module is not None:
            sys.modules["MoFlow"] = local_module
        if removed_path:
            sys.path.insert(0, script_dir)

    return mf


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


def parse_layer_list(value) -> list:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value).replace(";", ",").split(",")
    return [str(layer).strip() for layer in values if str(layer).strip()]


def parse_dimargs(value) -> list:
    """Parse ``src:dst`` embedding remaps, defaulting ``dst`` to ``X_umap``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value).replace(";", ",").split(",")

    remaps = []
    for entry in values:
        entry = str(entry).strip()
        if not entry:
            continue
        if ":" in entry:
            source, target = entry.split(":", 1)
        else:
            source, target = entry, "X_umap"
        source, target = source.strip(), target.strip()
        if source and target:
            remaps.append((source, target))
    return remaps


def cleanup_resources() -> None:
    gc.collect()
    plt.close("all")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
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

    required_columns = ["dataset_name", "file_path", "cluster_key"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "dimred_key" not in df.columns:
        df["dimred_key"] = "X_umap"
    if "simulate" not in df.columns:
        df["simulate"] = False
    if "extra_layers" not in df.columns:
        df["extra_layers"] = DEFAULT_AUXILIARY_LAYER
    if "dimargs" not in df.columns:
        df["dimargs"] = DEFAULT_DIMARGS

    df["dataset_name"] = df["dataset_name"].astype(str)
    df["file_path"] = df["file_path"].astype(str)
    df["cluster_key"] = df["cluster_key"].astype(str)
    df["dimred_key"] = df["dimred_key"].astype(str)
    df["simulate"] = df["simulate"].map(parse_bool)
    df["extra_layers"] = df["extra_layers"].map(parse_layer_list)
    df["dimargs"] = df["dimargs"].map(parse_dimargs)

    return df


def derive_output_stem(input_path: Path) -> str:
    stem = input_path.stem
    if stem.endswith("_dataset"):
        return stem[:-8]
    return stem


def apply_dimargs(adata, dimargs) -> None:
    """Expose source embeddings under the requested target keys."""
    for source, target in dimargs:
        if source in adata.obsm:
            adata.obsm[target] = np.asarray(adata.obsm[source])


def ensure_dimred(adata, dimred_key: str) -> None:
    if dimred_key in adata.obsm:
        return
    print(f"  Computing UMAP because '{dimred_key}' was not found...")
    sc.tl.umap(adata)
    if dimred_key != "X_umap":
        adata.obsm[dimred_key] = adata.obsm["X_umap"].copy()


def resolve_auxiliary_layer(adata, extra_layers) -> str:
    extra_layers = parse_layer_list(extra_layers)
    if not extra_layers:
        extra_layers = [DEFAULT_AUXILIARY_LAYER]

    if len(extra_layers) > 1:
        print(
            f"  Warning: MoFlow consumes a single auxiliary omic matrix; "
            f"using '{extra_layers[0]}' and ignoring {extra_layers[1:]}."
        )

    layer = extra_layers[0]
    if layer not in adata.layers:
        available = sorted(adata.layers.keys())
        raise ValueError(
            f"Auxiliary layer '{layer}' not found in adata.layers. Available layers: {available}"
        )
    return layer


def preprocess_adata(
    adata,
    cluster_key: Optional[str],
    extra_layers,
    dimargs,
    simulate: bool,
    dimred_key: str = "X_umap",
    min_shared_counts: int = 20,
    n_top_genes: int = 2000,
    preprocess_input: bool = False,
):
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    apply_dimargs(adata, dimargs)
    if simulate and "X_dimred" in adata.obsm:
        adata.obsm["X_umap"] = np.asarray(adata.obsm["X_dimred"])
    ensure_dimred(adata, dimred_key)

    auxiliary_layer = resolve_auxiliary_layer(adata, extra_layers)

    if simulate:
        if cluster_key and cluster_key in adata.obs:
            adata.obs[cluster_key] = "milestone"
        adata.obs["celltype"] = "milestone"
    else:
        if cluster_key is not None and cluster_key not in adata.obs:
            raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")
        fallback_key = cluster_key if cluster_key in adata.obs else "celltype"
        if fallback_key not in adata.obs:
            raise ValueError("No cluster labels available in adata.obs")
        adata.obs["celltype"] = adata.obs[fallback_key].astype(str)

    adata.obs["celltype_new"] = adata.obs["celltype"].astype(str)

    missing_layers = [layer for layer in ("spliced", "unspliced") if layer not in adata.layers]
    if missing_layers:
        raise ValueError(f"Missing required layers: {missing_layers}")

    # Opt-in only: the original code/Moflow_sim.py never calls filter_and_normalize, so
    # by default the input is handed to the model exactly as read.
    if preprocess_input:
        resolved_n_top_genes = int(min(n_top_genes, adata.n_vars))
        if simulate:
            scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=resolved_n_top_genes)
        else:
            scv.pp.filter_and_normalize(
                adata, min_shared_counts=min_shared_counts, n_top_genes=resolved_n_top_genes
            )

    return adata, auxiliary_layer


def build_moflow_model(mf, adata, auxiliary_layer: str, dataset_name: str, device: Optional[str]):
    adata_rna = adata.copy()
    adata_atac = adata.copy()
    adata_atac.X = adata_atac.layers[auxiliary_layer].copy()

    kwargs = {}
    if device:
        kwargs["device"] = device

    model = mf.MOFlow(adata_rna, adata_atac, folder_name=f"MoFlow_velocity_{dataset_name}", **kwargs)
    return model, adata_rna


def run_moflow_analysis(
    input_path,
    output_dir,
    cluster_key: Optional[str],
    dataset_name: Optional[str] = None,
    dimred_key: str = "X_umap",
    extra_layers=DEFAULT_AUXILIARY_LAYER,
    dimargs=DEFAULT_DIMARGS,
    simulate: bool = False,
    n_jobs: int = 10,
    device: Optional[str] = None,
    min_shared_counts: int = 20,
    n_top_genes: int = 2000,
    preprocess_input: bool = False,
    overwrite: bool = False,
    seed: int = 2024,
) -> Path:
    mf = load_moflow_api()

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if dataset_name is None:
        dataset_name = derive_output_stem(input_path)

    dataset_output_dir = output_dir / str(dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Native MoFlow file name, preserved from code/Moflow_sim.py (`{ID}_moflow.h5ad`).
    output_h5ad = dataset_output_dir / f"{dataset_name}_moflow.h5ad"

    if output_h5ad.exists() and not overwrite:
        print(f"Skipping existing output: {output_h5ad}")
        return output_h5ad

    seed_everything(seed)

    adata = None
    try:
        print(f"\nProcessing: {input_path.name}")
        adata = sc.read(input_path)

        print("  Preprocessing...")
        adata, auxiliary_layer = preprocess_adata(
            adata=adata,
            cluster_key=cluster_key,
            extra_layers=extra_layers,
            dimargs=parse_dimargs(dimargs),
            simulate=simulate,
            dimred_key=dimred_key,
            min_shared_counts=min_shared_counts,
            n_top_genes=n_top_genes,
            preprocess_input=preprocess_input,
        )

        print("  Building MoFlow model...")
        model, adata_rna = build_moflow_model(
            mf=mf,
            adata=adata,
            auxiliary_layer=auxiliary_layer,
            dataset_name=str(dataset_name),
            device=device,
        )

        print("  Running MoFlow velocity estimation...")
        model.velocity(
            adata_rna,
            n_jobs=n_jobs,
            save_path=str(dataset_output_dir),
            file_name=output_h5ad.name,
        )

        if output_h5ad.exists():
            result = sc.read(output_h5ad)
        else:
            result = adata_rna

        if "velocity" not in result.layers:
            raise RuntimeError("MoFlow did not produce layers['velocity']")

        result.uns["moflow_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(input_path.resolve()),
            "cluster_key": None if cluster_key is None else str(cluster_key),
            "dimred_key": str(dimred_key),
            "auxiliary_layer": str(auxiliary_layer),
            "simulate": bool(simulate),
            "output_path": str(output_h5ad.resolve()),
        }

        result.write(output_h5ad, compression="lzf")
        shutil.copyfile(output_h5ad, dataset_output_dir / "rc.h5ad")
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata
        cleanup_resources()


def run_batch_moflow(
    metadata_file,
    output_dir,
    overwrite: bool = False,
    seed: int = 2024,
) -> list:
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
            output_path = run_moflow_analysis(
                input_path=file_path,
                output_dir=output_dir,
                cluster_key=row["cluster_key"],
                dataset_name=row["dataset_name"],
                dimred_key=row["dimred_key"],
                extra_layers=row["extra_layers"],
                dimargs=row["dimargs"],
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
        description="MoFlow multi-omics velocity analysis",
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
        help="Column name in adata.obs used for labels in single-file mode",
    )
    parser.add_argument(
        "--dimred-key",
        default="X_umap",
        help="Dimensionality reduction key in adata.obsm; use X_umap for real data and X_dimred for simulated data",
    )
    parser.add_argument(
        "--extra-layers",
        default=DEFAULT_AUXILIARY_LAYER,
        help=(
            "Comma-separated adata.layers keys for the auxiliary omic matrix; "
            "MoFlow consumes a single matrix, so only the first entry is used"
        ),
    )
    parser.add_argument(
        "--dimargs",
        default=DEFAULT_DIMARGS,
        help=(
            "Comma-separated embedding remaps applied before training, each given as "
            "<source> or <source>:<target>; <target> defaults to X_umap"
        ),
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Simulated-data branch: use the milestone label, X_dimred, and relaxed filters",
    )
    parser.add_argument("--n-jobs", type=int, default=10, help="Number of parallel jobs used by MoFlow")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device passed to the MoFlow model, for example cuda or cpu; default uses the model default",
    )
    parser.add_argument(
        "--preprocess-input",
        action="store_true",
        default=False,
        help=(
            "Opt-in: run scv.pp.filter_and_normalize before MoFlow. Disabled by default so "
            "the input is passed to the model exactly as in code/Moflow_sim.py"
        ),
    )
    parser.add_argument(
        "--min-shared-counts",
        type=int,
        default=20,
        help="Only used with --preprocess-input: minimum shared counts for filter_and_normalize",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=2000,
        help="Only used with --preprocess-input: number of highly variable genes retained",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_moflow(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    if not args.cluster_key:
        parser.error("--cluster-key is required in single-file mode")

    return run_moflow_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        dimred_key=args.dimred_key,
        extra_layers=args.extra_layers,
        dimargs=args.dimargs,
        simulate=args.simulate,
        n_jobs=args.n_jobs,
        device=args.device,
        min_shared_counts=args.min_shared_counts,
        n_top_genes=args.n_top_genes,
        preprocess_input=args.preprocess_input,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
