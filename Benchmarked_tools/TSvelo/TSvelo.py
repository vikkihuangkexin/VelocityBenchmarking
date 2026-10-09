#!/usr/bin/env python3
"""
TSvelo single-dataset and batch RNA velocity pipeline for VelocityBenchmarking.

TSvelo jointly models splicing, transcription and a transcription-factor
regulatory network to infer RNA velocity from spliced/unspliced counts.

Installation:
    pip install pandas==2.0.3 anndata==0.9.2 scanpy==1.9.8 numpy==1.24.4 scipy==1.10.1
    pip install numba==0.58.1 matplotlib==3.7.5 scvelo==0.3.2 torch==2.4.1
    pip install torchdiffeq==0.2.4 typing_extensions mygene==3.2.2 leidenalg==0.10.2 pygam==0.9.1
    # The TSvelo package itself is installed from the upstream repository, e.g.
    # pip install git+https://github.com/lijc0804/TSvelo.git
    # ENCODE / ChEA TF databases (ENCODE/processed, ChEA/ChEA_2016.txt) must be
    # unpacked next to the TSvelo sources; point --tsvelo-db-dir at that folder.

Usage:
    python TSvelo.py --input data.h5ad --output-dir ./output --cluster-key celltype
    python TSvelo.py --input data.h5ad --output-dir ./output --dataset-name pancreas --cluster-key celltype
    python TSvelo.py --metadata-file datasets.csv --output-dir ./output

Note:
    Simulated-data processing is genuinely different (an a-priori GRN is injected
    instead of the ENCODE/ChEA TF-database lookup), so it lives in the separate
    TSvelo_sim.py script.
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvelo as scv

# Default location of the TSvelo sources / TF databases inside the container.
# It can be overridden with --tsvelo-db-dir or the TSVELO_DB_DIR env variable.
DEFAULT_TSVELO_DB_DIR = os.environ.get("TSVELO_DB_DIR", "/opt/TSvelo")

_TSVELO_API = {}


def seed_everything(seed: int) -> None:
    """Seed numpy, Python ``random`` and (if available) PyTorch."""
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


def configure_cuda(cuda: Optional[int]) -> Optional[str]:
    """Expose ``--cuda`` through ``CUDA_VISIBLE_DEVICES`` without overriding the caller."""
    if cuda is None:
        return os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cuda))
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def load_tsvelo_api():
    """Import the installed TSvelo package lazily (after CUDA env setup)."""
    if not _TSVELO_API:
        from TSvelo.TSvelo_model import (
            init_US,
            init_W,
            init_t,
            make_loss_mask,
            run,
            to_adata,
        )
        from TSvelo.TSvelo_pp_utils import get_TFs, select_gene
        from TSvelo.TSvelo_branch import init_branch
        from TSvelo.TSvelo_concate import concate
        from TSvelo.TSvelo_utils import scv_analysis

        _TSVELO_API.update(
            init_US=init_US,
            init_W=init_W,
            init_t=init_t,
            make_loss_mask=make_loss_mask,
            run=run,
            to_adata=to_adata,
            get_TFs=get_TFs,
            select_gene=select_gene,
            init_branch=init_branch,
            concate=concate,
            scv_analysis=scv_analysis,
        )
    return _TSVELO_API


def cleanup_resources() -> None:
    gc.collect()
    plt.close("all")


def _detect_separator(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix in {".tsv", ".txt"}:
        return "\t"
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    return "\t" if "\t" in first_line else ","


def load_metadata_file(metadata_path: Path) -> pd.DataFrame:
    """Read the batch manifest and validate the required columns."""
    df = pd.read_csv(metadata_path, sep=_detect_separator(metadata_path))
    required = ["dataset_name", "file_path", "cluster_key"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "save_name" not in df.columns:
        df["save_name"] = ""

    df["dataset_name"] = df["dataset_name"].astype(str)
    df["file_path"] = df["file_path"].astype(str)
    df["cluster_key"] = df["cluster_key"].astype(str)
    df["save_name"] = df["save_name"].fillna("").astype(str)
    return df


def preprocess_adata(
    adata,
    cluster_key: str = "vis_annotation",
    n_top_genes: int = 2000,
    n_neighbors: int = 30,
    n_jobs: int = -1,
    n_selected_genes: int = 100,
    tf_databases: Sequence[str] = ("ENCODE", "ChEA"),
    tsvelo_db_dir: str = DEFAULT_TSVELO_DB_DIR,
    umap: bool = True,
):
    """Normalized TSvelo preprocessing for an in-memory AnnData object.

    Replaces the hard-coded dataset loading of the original ``TSvelo_pp.preprocess``
    with an explicit ``cluster_key`` coming from the input file, while keeping the
    same filtering, TF-database lookup and velocity-gene selection steps.
    """
    api = load_tsvelo_api()

    if cluster_key not in adata.obs.columns:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

    adata.obs["clusters"] = pd.Categorical(adata.obs[cluster_key].astype(str))

    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=n_top_genes)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    # Addition relative to code/tsvelo.py: compute UMAP only when it is absent.
    if umap and "X_umap" not in adata.obsm:
        sc.tl.umap(adata)

    # Match the ENCODE/ChEA TF database naming convention (upper-case symbols).
    adata.var_names = [str(g).upper() for g in adata.var_names]
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # get_TFs opens ENCODE/ and ChEA/ relative to the TSvelo source directory.
    previous_cwd = os.getcwd()
    try:
        os.chdir(tsvelo_db_dir)
        adata = api["get_TFs"](adata, list(tf_databases))
    finally:
        os.chdir(previous_cwd)

    adata = api["select_gene"](
        adata, n_selected_genes, n_neighbors=n_neighbors, n_jobs=n_jobs
    )
    return adata


def _build_runtime_args(save_folder: Path, dataset_name: str, runtime) -> SimpleNamespace:
    """Build the attribute bag consumed by the installed TSvelo functions."""
    return SimpleNamespace(
        save_folder=str(save_folder) + "/",
        dataset_name=dataset_name,
        n_jobs=runtime.n_jobs,
        n_neighbors=runtime.n_neighbors,
        n_top_genes=runtime.n_top_genes,
        n_selected_genes=runtime.n_selected_genes,
        TF_databases=list(runtime.tf_databases),
        N_steps=runtime.n_steps,
        cuda=runtime.cuda if runtime.cuda is not None else 0,
        N_EPOCH=runtime.n_epoch,
        num_epochs=runtime.num_epochs,
        min_decrease=runtime.min_decrease,
        n_genes2show=runtime.n_genes2show,
    )


def _run_branch(args: SimpleNamespace, branch_index: int) -> None:
    """Train TSvelo on one lineage branch (mirrors ``TSvelo_run.main``)."""
    api = load_tsvelo_api()

    adata = ad.read_h5ad(args.save_folder + "l" + str(branch_index) + "_pp.h5ad")
    adata, W_ini, W_0_mask, n_selected_genes = api["init_W"](adata)
    Y, U, S = api["init_US"](adata)
    loss_mask = api["make_loss_mask"](adata, Y, U, S)

    figure_folder = args.save_folder + "figures_l" + str(branch_index) + "/"
    adata, t_steps = api["init_t"](args, adata, figure_folder=figure_folder)
    adata, best_results = api["run"](
        args, adata, W_ini, W_0_mask, n_selected_genes, Y, U, S, loss_mask, t_steps,
        figure_folder=figure_folder,
    )
    _, best_epoch_loss, U, S, W, W_bias, BETA, GAMMA, t_steps, Y_pre = best_results
    print(f"Best results at EPOCH {best_epoch_loss:.6f}: branch {branch_index}")

    adata = api["to_adata"](
        adata, U, S, W, W_bias, BETA, GAMMA, t_steps, Y_pre,
        n_selected_genes, loss_mask,
    )
    adata = api["scv_analysis"](adata, figure_folder=figure_folder)
    adata.write(args.save_folder + "l" + str(branch_index) + "_TSvelo.h5ad")


def derive_output_stem(input_path: Path) -> str:
    """Strip the benchmark `_dataset` suffix from an input file stem."""
    stem = input_path.stem
    if stem.endswith("_dataset"):
        return stem[:-8]
    return stem


def run_tsvelo_analysis(
    input_path,
    output_dir,
    cluster_key: str,
    dataset_name: Optional[str] = None,
    save_name: str = "",
    n_jobs: int = -1,
    n_neighbors: int = 30,
    n_top_genes: int = 2000,
    n_selected_genes: int = 100,
    tf_databases: Sequence[str] = ("ENCODE", "ChEA"),
    n_steps: int = 500,
    n_epoch: int = 10,
    num_epochs: int = 100,
    min_decrease: float = 0.0,
    n_genes2show: int = 0,
    cuda: Optional[int] = None,
    tsvelo_db_dir: str = DEFAULT_TSVELO_DB_DIR,
    seed: int = 2024,
    overwrite: bool = False,
) -> Path:
    """Run the full TSvelo pipeline for a single h5ad dataset."""
    api = load_tsvelo_api()

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if dataset_name is None:
        dataset_name = input_path.stem
    dataset_name = f"{dataset_name}{save_name}"

    work_dir = output_dir / dataset_name
    work_dir.mkdir(parents=True, exist_ok=True)
    output_h5ad = work_dir / f"{derive_output_stem(input_path)}.h5ad"

    if output_h5ad.exists() and not overwrite:
        print(f"Skipping existing output: {output_h5ad}")
        return output_h5ad

    seed_everything(seed)
    configure_cuda(cuda)

    runtime = SimpleNamespace(
        n_jobs=n_jobs,
        n_neighbors=n_neighbors,
        n_top_genes=n_top_genes,
        n_selected_genes=n_selected_genes,
        tf_databases=tf_databases,
        n_steps=n_steps,
        cuda=cuda,
        n_epoch=n_epoch,
        num_epochs=num_epochs,
        min_decrease=min_decrease,
        n_genes2show=n_genes2show,
    )
    args = _build_runtime_args(work_dir, dataset_name, runtime)

    adata = None
    try:
        print(f"\nProcessing: {input_path.name}")
        adata = sc.read(input_path)
        adata = preprocess_adata(
            adata,
            cluster_key=cluster_key,
            n_top_genes=n_top_genes,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
            n_selected_genes=n_selected_genes,
            tf_databases=tf_databases,
            tsvelo_db_dir=tsvelo_db_dir,
        )
        adata.write(work_dir / "pp.h5ad")

        print("  Initializing lineages...")
        api["init_branch"](args)

        branch_files = sorted(glob.glob(os.path.join(str(work_dir), "*_pp.h5ad")))
        branch_files = [f for f in branch_files if os.path.basename(f).startswith("l")]
        if not branch_files:
            raise RuntimeError("Lineage initialization produced no branch files")

        print(f"  Training {len(branch_files)} lineage branch(es)...")
        for branch_index in range(len(branch_files)):
            _run_branch(args, branch_index)

        if len(branch_files) > 1:
            api["concate"](args)
        else:
            os.replace(work_dir / "l0_TSvelo.h5ad", work_dir / "TSvelo.h5ad")
            figures_src = work_dir / "figures_l0"
            if figures_src.exists():
                os.replace(figures_src, work_dir / "figures")

        merged = work_dir / "TSvelo.h5ad"
        if not merged.exists():
            raise RuntimeError("TSvelo did not produce a merged TSvelo.h5ad")

        # Copy to the benchmark output convention: <output-dir>/<dataset>/<stem>.h5ad
        result = ad.read_h5ad(merged)
        result.uns["tsvelo_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(input_path.resolve()),
            "cluster_key": cluster_key,
            "n_lineages": int(len(branch_files)),
            "output_path": str(output_h5ad.resolve()),
        }

        if "velocity" not in result.layers:
            raise RuntimeError("TSvelo did not produce layers['velocity'].")

        result.write(output_h5ad, compression="lzf")
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata
        cleanup_resources()


def run_batch_tsvelo(
    metadata_file,
    output_dir,
    default_cluster_key: Optional[str] = None,
    save_name: str = "",
    n_jobs: int = -1,
    n_neighbors: int = 30,
    n_top_genes: int = 2000,
    n_selected_genes: int = 100,
    tf_databases: Sequence[str] = ("ENCODE", "ChEA"),
    n_steps: int = 500,
    n_epoch: int = 10,
    num_epochs: int = 100,
    min_decrease: float = 0.0,
    n_genes2show: int = 0,
    cuda: Optional[int] = None,
    tsvelo_db_dir: str = DEFAULT_TSVELO_DB_DIR,
    seed: int = 2024,
    overwrite: bool = False,
) -> List[Path]:
    """Run TSvelo over every row of a metadata manifest."""
    metadata_file = Path(metadata_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_file(metadata_file)
    outputs: List[Path] = []
    print(f"Batch mode: {len(metadata_df)} datasets")

    for _, row in metadata_df.iterrows():
        file_path = Path(row["file_path"])
        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        cluster_key = row["cluster_key"] or default_cluster_key
        if not cluster_key:
            print(f"Skipping {row['dataset_name']}: no cluster key")
            continue

        try:
            outputs.append(
                run_tsvelo_analysis(
                    input_path=file_path,
                    output_dir=output_dir,
                    cluster_key=cluster_key,
                    dataset_name=row["dataset_name"],
                    save_name=row["save_name"],
                    n_jobs=n_jobs,
                    n_neighbors=n_neighbors,
                    n_top_genes=n_top_genes,
                    n_selected_genes=n_selected_genes,
                    tf_databases=tf_databases,
                    n_steps=n_steps,
                    n_epoch=n_epoch,
                    num_epochs=num_epochs,
                    min_decrease=min_decrease,
                    n_genes2show=n_genes2show,
                    cuda=cuda,
                    tsvelo_db_dir=tsvelo_db_dir,
                    seed=seed,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep batch runs going
            print(f"Failed: {row['dataset_name']}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TSvelo RNA velocity (real data)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input H5AD file")
    input_group.add_argument("--metadata-file", help="Metadata CSV/TSV file for batch processing")

    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--dataset-name", "--dataset_name", dest="dataset_name", default=None,
        help="Dataset folder name",
    )
    parser.add_argument(
        "--cluster-key", "--cluster_key", dest="cluster_key", default=None,
        help="Column name in adata.obs used for cell-type labels",
    )
    parser.add_argument("--save-name", "--save_name", dest="save_name", default="",
                        help="Suffix appended to the dataset folder name")
    parser.add_argument("--preprocess", action="store_true", default=False,
                        help="Force re-preprocessing (equivalent to --overwrite for the pp step)")
    parser.add_argument("--n-jobs", "--n_jobs", dest="n_jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--n-neighbors", "--n_neighbors", dest="n_neighbors", type=int, default=30,
                        help="Number of neighbors for the KNN graph")
    parser.add_argument("--n-top-genes", "--n_top_genes", dest="n_top_genes", type=int, default=2000,
                        help="Number of highly variable genes")
    parser.add_argument("--n-selected-genes", "--n_selected_genes", dest="n_selected_genes", type=int, default=100,
                        help="Number of selected velocity genes")
    parser.add_argument("--TF-databases", "--TF_databases", dest="tf_databases", nargs="+",
                        default=["ENCODE", "ChEA"], help="TF databases to use, e.g. ENCODE ChEA")
    parser.add_argument("--N-steps", "--N_steps", dest="n_steps", type=int, default=500,
                        help="Number of time steps")
    parser.add_argument("--N-EPOCH", "--N_EPOCH", dest="n_epoch", type=int, default=10,
                        help="Maximum number of EM epochs")
    parser.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int, default=100,
                        help="Maximum number of neural-ODE epochs")
    parser.add_argument("--min-decrease", "--min_decrease", dest="min_decrease", type=float, default=0.0,
                        help="Minimum decrease for early stopping")
    parser.add_argument("--n-genes2show", "--n_genes2show", dest="n_genes2show", type=int, default=0,
                        help="Number of genes to plot during training")
    parser.add_argument("--cuda", type=int, default=None, help="CUDA device id; exposed via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--tsvelo-db-dir", dest="tsvelo_db_dir", default=DEFAULT_TSVELO_DB_DIR,
                        help="Directory containing the ENCODE/ and ChEA/ TF databases")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    overwrite = bool(args.overwrite or args.preprocess)

    if args.metadata_file:
        return run_batch_tsvelo(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            default_cluster_key=args.cluster_key,
            save_name=args.save_name,
            n_jobs=args.n_jobs,
            n_neighbors=args.n_neighbors,
            n_top_genes=args.n_top_genes,
            n_selected_genes=args.n_selected_genes,
            tf_databases=args.tf_databases,
            n_steps=args.n_steps,
            n_epoch=args.n_epoch,
            num_epochs=args.num_epochs,
            min_decrease=args.min_decrease,
            n_genes2show=args.n_genes2show,
            cuda=args.cuda,
            tsvelo_db_dir=args.tsvelo_db_dir,
            seed=args.seed,
            overwrite=overwrite,
        )

    if not args.cluster_key:
        parser.error("--cluster-key is required in single-file mode")

    return run_tsvelo_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        save_name=args.save_name,
        n_jobs=args.n_jobs,
        n_neighbors=args.n_neighbors,
        n_top_genes=args.n_top_genes,
        n_selected_genes=args.n_selected_genes,
        tf_databases=args.tf_databases,
        n_steps=args.n_steps,
        n_epoch=args.n_epoch,
        num_epochs=args.num_epochs,
        min_decrease=args.min_decrease,
        n_genes2show=args.n_genes2show,
        cuda=args.cuda,
        tsvelo_db_dir=args.tsvelo_db_dir,
        seed=args.seed,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()
