#!/usr/bin/env python3
"""
TSvelo simulated-data RNA velocity pipeline for VelocityBenchmarking.

Simulated datasets (scmultisim "Bursting-tree" / dyngen) do not have a real
ENCODE/ChEA TF-database lookup available; instead the a-priori regulatory network
that was used to generate the data is injected directly into the AnnData object.
For that reason this simulated-data branch is kept as a **separate** script from
TSvelo.py (real data).

Installation:
    pip install pandas==2.0.3 anndata==0.9.2 scanpy==1.9.8 numpy==1.24.4 scipy==1.10.1
    pip install numba==0.58.1 matplotlib==3.7.5 scvelo==0.3.2 torch==2.4.1
    pip install torchdiffeq==0.2.4 typing_extensions leidenalg==0.10.2 pygam==0.9.1
    # The TSvelo package itself is installed from the upstream repository, e.g.
    # pip install git+https://github.com/lijc0804/TSvelo.git

Usage:
    python TSvelo_sim.py --input sim.h5ad --output-dir ./output --cluster-key vis_annotation
    python TSvelo_sim.py --input sim.h5ad --output-dir ./output \
        --grn-100-csv GRN_params_100.csv --grn-1139-csv GRN_params_1139.csv
"""

from __future__ import annotations

import argparse
import gc
import os
import re
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
import scanpy as sc
import scvelo as scv

# Sensible defaults resolve relative to this script so it can be launched anywhere.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GRN_100_CSV = str(SCRIPT_DIR / "GRN_params_100.csv")
DEFAULT_GRN_1139_CSV = str(SCRIPT_DIR / "GRN_params_1139.csv")
DEFAULT_DYNGEN_FEATURE_NETWORK_CSV = str(
    SCRIPT_DIR / "dyngen_bifurcating_cell10000_gene500_feature_network.csv"
)

_TSVELO_API = {}


# ---------------------------------------------------------------------------
# GRN-prior utilities (inlined from the original grn_utils.py)
# ---------------------------------------------------------------------------
def detect_sim_type(adata) -> tuple:
    """Detect the simulation engine: 'scmultisim' (numeric gene ids) or 'dyngen'."""
    pure = [g for g in adata.var_names if re.fullmatch(r"\d+", str(g))]
    if pure:
        return "scmultisim", len(pure)
    return "dyngen", 0


def choose_grn_csv(n_regulatory_genes: int, grn_100_csv: str, grn_1139_csv: str) -> str:
    """Pick the scmultisim GRN csv according to the number of regulatory genes."""
    if n_regulatory_genes <= 500:
        return grn_100_csv
    return grn_1139_csv


def load_grn(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def grn_regulatory_genes(grn_df: pd.DataFrame) -> List[str]:
    ids = set()
    for column in ("regulated.gene", "regulator.gene", "target", "regulator"):
        if column in grn_df.columns:
            ids.update(grn_df[column].astype(int))
    return sorted([str(i) for i in ids])


def detect_tf_from_names(adata) -> List[str]:
    """Infer TF gene names from dyngen-style ``*_TF1`` names."""
    return sorted([str(g) for g in adata.var_names if re.search(r"_TF\d*$", str(g))])


def load_dyngen_feature_network(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def inject_grn(adata, grn_df: pd.DataFrame, effect_scale: float = 1.0):
    """Inject a long-format scmultisim GRN, replicating ``get_TFs`` output."""
    gene_names = list(adata.var_names)
    gene_index = {g: i for i, g in enumerate(gene_names)}
    n_gene = adata.shape[1]

    columns = list(grn_df.columns)
    if "regulated.gene" in columns and "regulator.gene" in columns:
        tgt_col, reg_col, eff_col = "regulated.gene", "regulator.gene", "regulator.effect"
    else:
        tgt_col, reg_col, eff_col = "target", "regulator", "effect"

    edges = {}
    all_tfs = set()
    for _, row in grn_df.iterrows():
        target = str(int(row[tgt_col]))
        regulator = str(int(row[reg_col]))
        effect = float(row[eff_col])
        edges.setdefault(target, []).append((regulator, effect))
        all_tfs.add(regulator)

    all_tfs = sorted(all_tfs, key=lambda x: gene_index.get(x, n_gene))
    max_n_tf = max((len(v) for v in edges.values()), default=0)
    max_n_tf = max(max_n_tf, 1)

    adata.varm["TFs"] = np.full([n_gene, max_n_tf], "blank", dtype=object)
    adata.varm["TFs_id"] = np.full([n_gene, max_n_tf], -1, dtype=int)
    adata.varm["TFs_times"] = np.full([n_gene, max_n_tf], 0, dtype=int)
    adata.varm["TFs_correlation"] = np.full([n_gene, max_n_tf], 0.0, dtype=float)
    adata.var["n_TFs"] = np.zeros(n_gene, dtype=int)

    n_skipped = 0
    for target, regulators in edges.items():
        if target not in gene_index:
            n_skipped += len(regulators)
            continue
        target_idx = gene_index[target]
        adata.var["n_TFs"][target_idx] = len(regulators)
        for j, (regulator, effect) in enumerate(regulators):
            if regulator not in gene_index:
                n_skipped += 1
                continue
            regulator_idx = gene_index[regulator]
            adata.varm["TFs"][target_idx, j] = regulator
            adata.varm["TFs_id"][target_idx, j] = regulator_idx
            adata.varm["TFs_times"][target_idx, j] = 1
            adata.varm["TFs_correlation"][target_idx, j] = effect * effect_scale

    adata.uns["all_TFs"] = all_tfs
    print(
        f"[inject_grn] injected {int(np.sum(adata.var['n_TFs']))} edges, "
        f"{len(all_tfs)} TFs, max_n_TF={max_n_tf}, skipped={n_skipped}"
    )
    return adata


def inject_grn_from_correlation(adata, tf_list: Sequence[str], top_k: int = 8, layer: str = "Ms"):
    """Fallback GRN for dyngen data: keep the top-|spearman| TFs per gene."""
    from scipy import stats

    gene_names = list(adata.var_names)
    gene_index = {g: i for i, g in enumerate(gene_names)}
    n_gene = adata.shape[1]

    tf_list = [str(t).upper() for t in tf_list]
    tf_list = sorted({t for t in tf_list if t in gene_index}, key=lambda x: gene_index[x])
    if not tf_list:
        raise ValueError("No TF in tf_list is present in adata.var_names.")

    expr = np.asarray(adata.layers[layer], dtype=float)
    k = min(top_k, len(tf_list))

    adata.varm["TFs"] = np.full([n_gene, k], "blank", dtype=object)
    adata.varm["TFs_id"] = np.full([n_gene, k], -1, dtype=int)
    adata.varm["TFs_times"] = np.full([n_gene, k], 0, dtype=int)
    adata.varm["TFs_correlation"] = np.full([n_gene, k], 0.0, dtype=float)
    adata.var["n_TFs"] = np.zeros(n_gene, dtype=int)

    n_edges = 0
    for gene_id, gene in enumerate(gene_names):
        target_expr = expr[:, gene_id]
        scored = []
        for tf in tf_list:
            if tf == gene:
                continue
            tf_id = gene_index[tf]
            tf_expr = expr[:, tf_id]
            flag = (tf_expr > 0.1) & (target_expr > 0.1)
            if flag.sum() < 2:
                corr = 0.0
            else:
                corr, _ = stats.spearmanr(target_expr[flag], tf_expr[flag])
                if np.isnan(corr):
                    corr = 0.0
            scored.append((tf, tf_id, corr))

        scored.sort(key=lambda x: -abs(x[2]))
        top = scored[:k]
        adata.var["n_TFs"][gene_id] = len(top)
        for j, (tf, tf_id, corr) in enumerate(top):
            adata.varm["TFs"][gene_id, j] = tf
            adata.varm["TFs_id"][gene_id, j] = tf_id
            adata.varm["TFs_times"][gene_id, j] = 1
            adata.varm["TFs_correlation"][gene_id, j] = corr
            n_edges += 1

    adata.uns["all_TFs"] = tf_list
    print(f"[inject_grn_from_correlation] {len(tf_list)} TFs, {n_edges} edges "
          f"(top_k={k}, layer={layer})")
    return adata


def inject_grn_from_edges(adata, edges_df: pd.DataFrame, weight_mode: str = "effect_strength"):
    """Inject an explicit dyngen ``feature_network`` edge list into AnnData."""
    gene_names = list(adata.var_names)
    gene_index = {g: i for i, g in enumerate(gene_names)}
    n_gene = adata.shape[1]

    reg_col = "regulator" if "regulator" in edges_df.columns else "from"
    tgt_col = "target" if "target" in edges_df.columns else "to"
    has_strength = "strength" in edges_df.columns

    edges = {}
    all_tfs = set()
    for _, row in edges_df.iterrows():
        regulator = str(row[reg_col]).upper()
        target = str(row[tgt_col]).upper()
        if regulator == target:
            continue
        effect = float(row["effect"])
        if weight_mode == "effect_strength" and has_strength:
            weight = effect * float(row["strength"])
        else:
            weight = effect
        edges.setdefault(target, []).append((regulator, weight))
        all_tfs.add(regulator)

    all_tfs = sorted(all_tfs, key=lambda x: gene_index.get(x, n_gene))
    max_n_tf = max((len(v) for v in edges.values()), default=0)
    max_n_tf = max(max_n_tf, 1)

    adata.varm["TFs"] = np.full([n_gene, max_n_tf], "blank", dtype=object)
    adata.varm["TFs_id"] = np.full([n_gene, max_n_tf], -1, dtype=int)
    adata.varm["TFs_times"] = np.full([n_gene, max_n_tf], 0, dtype=int)
    adata.varm["TFs_correlation"] = np.full([n_gene, max_n_tf], 0.0, dtype=float)
    adata.var["n_TFs"] = np.zeros(n_gene, dtype=int)

    n_injected = 0
    for target, regulators in edges.items():
        if target not in gene_index:
            continue
        target_idx = gene_index[target]
        adata.var["n_TFs"][target_idx] = len(regulators)
        for j, (regulator, weight) in enumerate(regulators):
            if regulator not in gene_index:
                continue
            regulator_idx = gene_index[regulator]
            adata.varm["TFs"][target_idx, j] = regulator
            adata.varm["TFs_id"][target_idx, j] = regulator_idx
            adata.varm["TFs_times"][target_idx, j] = 1
            adata.varm["TFs_correlation"][target_idx, j] = weight
            n_injected += 1

    adata.uns["all_TFs"] = all_tfs
    print(f"[inject_grn_from_edges] injected {n_injected} edges, {len(all_tfs)} regulators")
    return adata


# ---------------------------------------------------------------------------
# TSvelo package access
# ---------------------------------------------------------------------------
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


def configure_cuda(cuda: Optional[int]) -> Optional[str]:
    if cuda is None:
        return os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cuda))
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def load_tsvelo_api():
    if not _TSVELO_API:
        from TSvelo.TSvelo_model import (
            init_US,
            init_W,
            init_t,
            make_loss_mask,
            run,
            to_adata,
        )
        from TSvelo.TSvelo_pp_utils import select_gene
        from TSvelo.TSvelo_branch import init_t as branch_init_t

        _TSVELO_API.update(
            init_US=init_US,
            init_W=init_W,
            init_t=init_t,
            make_loss_mask=make_loss_mask,
            run=run,
            to_adata=to_adata,
            select_gene=select_gene,
            branch_init_t=branch_init_t,
        )
    return _TSVELO_API


def cleanup_resources() -> None:
    gc.collect()
    plt.close("all")


def preprocess_adata(
    adata,
    cluster_key: str = "vis_annotation",
    n_neighbors: int = 30,
    n_jobs: int = -1,
    n_selected_genes: int = 100,
    grn_100_csv: str = DEFAULT_GRN_100_CSV,
    grn_1139_csv: str = DEFAULT_GRN_1139_CSV,
    dyngen_feature_network_csv: str = DEFAULT_DYNGEN_FEATURE_NETWORK_CSV,
    dyngen_correlation_top_k: int = 10,
    sim_type: Optional[str] = None,
):
    """Normalized TSvelo preprocessing for simulated data with a GRN prior."""
    api = load_tsvelo_api()

    adata.obs_names_make_unique()
    if adata.n_vars == 0:
        raise ValueError("Input AnnData has zero genes")

    if "X_dimred" in adata.obsm:
        adata.obsm["X_umap"] = adata.obsm["X_dimred"]

    # Detect the simulation engine and pick / build the appropriate GRN prior.
    detected_type, n_reg = detect_sim_type(adata)
    sim_type = sim_type or detected_type
    tf_list = None
    edges_df = None

    if sim_type == "scmultisim":
        grn_df = load_grn(choose_grn_csv(n_reg, grn_100_csv, grn_1139_csv))
        retain_genes = grn_regulatory_genes(grn_df)
        print(f"scmultisim detected: {n_reg} regulatory genes, GRN edges={len(grn_df)}")
    else:
        tf_list = detect_tf_from_names(adata)
        retain_genes = tf_list
        if dyngen_feature_network_csv and os.path.exists(dyngen_feature_network_csv):
            edges_df = load_dyngen_feature_network(dyngen_feature_network_csv)
            reg_col = "regulator" if "regulator" in edges_df.columns else "from"
            retain_genes = sorted({str(r) for r in edges_df[reg_col]})
        print(f"dyngen detected: {len(tf_list)} TFs")

    # HVG selection; keep the priors' genes so filtering does not drop TFs.
    if adata.n_vars < 2000:
        top_gene = min((adata.n_vars // 500) * 500, adata.n_vars - 1)
    else:
        top_gene = 2000
    top_gene = max(1, int(top_gene))

    retain_genes = [g for g in retain_genes if g in list(adata.var_names)]
    scv.pp.filter_and_normalize(
        adata, min_shared_counts=None, n_top_genes=top_gene, retain_genes=retain_genes
    )
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    # Match the TF naming convention used by init_W / get_colors.
    adata.var_names = [str(g).upper() for g in adata.var_names]
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.obs["clusters"] = pd.Categorical(adata.obs[cluster_key].astype(str))

    if sim_type == "scmultisim":
        adata = inject_grn(adata, grn_df)
    elif edges_df is not None:
        adata = inject_grn_from_edges(adata, edges_df)
    else:
        adata = inject_grn_from_correlation(
            adata, tf_list, top_k=dyngen_correlation_top_k
        )

    adata = api["select_gene"](
        adata, n_selected_genes, n_neighbors=n_neighbors, n_jobs=n_jobs
    )
    return adata


def _build_runtime_args(save_folder: Path, runtime) -> SimpleNamespace:
    return SimpleNamespace(
        save_folder=str(save_folder) + "/",
        dataset_name="simulated",
        n_jobs=runtime.n_jobs,
        n_neighbors=runtime.n_neighbors,
        N_steps=runtime.n_steps,
        cuda=runtime.cuda if runtime.cuda is not None else 0,
        N_EPOCH=runtime.n_epoch,
        num_epochs=runtime.num_epochs,
        min_decrease=runtime.min_decrease,
        n_genes2show=runtime.n_genes2show,
    )


def derive_output_stem(input_path: Path) -> str:
    """Strip the benchmark `_dataset` suffix from an input file stem."""
    stem = input_path.stem
    if stem.endswith("_dataset"):
        return stem[:-8]
    return stem


def run_tsvelo_sim_analysis(
    input_path,
    output_dir,
    cluster_key: str = "vis_annotation",
    dataset_name: Optional[str] = None,
    n_jobs: int = -1,
    n_neighbors: int = 30,
    n_selected_genes: int = 100,
    n_steps: int = 500,
    n_epoch: int = 10,
    num_epochs: int = 100,
    min_decrease: float = 0.0,
    n_genes2show: int = 0,
    cuda: Optional[int] = None,
    grn_100_csv: str = DEFAULT_GRN_100_CSV,
    grn_1139_csv: str = DEFAULT_GRN_1139_CSV,
    dyngen_feature_network_csv: str = DEFAULT_DYNGEN_FEATURE_NETWORK_CSV,
    dyngen_correlation_top_k: int = 10,
    sim_type: Optional[str] = None,
    save_pdf: bool = False,
    seed: int = 2024,
    overwrite: bool = False,
) -> Path:
    """Run the full TSvelo pipeline for one simulated h5ad dataset."""
    api = load_tsvelo_api()

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    dataset_name = dataset_name or input_path.stem
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
        n_steps=n_steps,
        cuda=cuda,
        n_epoch=n_epoch,
        num_epochs=num_epochs,
        min_decrease=min_decrease,
        n_genes2show=n_genes2show,
    )
    args = _build_runtime_args(work_dir, runtime)

    adata = None
    try:
        print(f"\nProcessing: {input_path.name}")
        adata = sc.read(input_path)
        adata = preprocess_adata(
            adata,
            cluster_key=cluster_key,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
            n_selected_genes=n_selected_genes,
            grn_100_csv=grn_100_csv,
            grn_1139_csv=grn_1139_csv,
            dyngen_feature_network_csv=dyngen_feature_network_csv,
            dyngen_correlation_top_k=dyngen_correlation_top_k,
            sim_type=sim_type,
        )
        adata.write(work_dir / "pp.h5ad")

        # Compute the diffusion-pseudotime initialization required by init_t.
        start_cluster = list(adata.obs["clusters"].cat.categories)[0]
        adata = api["branch_init_t"](args, adata, "clusters", start_cluster)

        adata, W_ini, W_0_mask, n_selected_genes = api["init_W"](adata)
        Y, U, S = api["init_US"](adata)
        loss_mask = api["make_loss_mask"](adata, Y, U, S)

        figure_folder = work_dir / "figures"
        figure_folder.mkdir(parents=True, exist_ok=True)
        adata, t_steps = api["init_t"](args, adata, figure_folder=str(figure_folder))
        adata, best_results = api["run"](
            args, adata, W_ini, W_0_mask, n_selected_genes, Y, U, S, loss_mask, t_steps,
            figure_folder=str(figure_folder),
        )
        _, best_epoch_loss, U, S, W, W_bias, BETA, GAMMA, t_steps, Y_pre = best_results
        print(f"Best results at EPOCH {best_epoch_loss:.6f}")

        adata = api["to_adata"](
            adata, U, S, W, W_bias, BETA, GAMMA, t_steps, Y_pre,
            n_selected_genes, loss_mask,
        )

        # Velocity = beta * U(t) - gamma * S(t) on the selected genes.
        selected_genes_mask = adata.var["selected_genes"]
        adata_selected = adata[:, selected_genes_mask]
        U_t = adata_selected.uns["U_t"][:, selected_genes_mask][adata_selected.obs["t_steps"]]
        S_t = adata_selected.uns["S_t"][:, selected_genes_mask][adata_selected.obs["t_steps"]]
        velocity = np.full(adata.shape, np.nan)
        velocity[:, selected_genes_mask] = (
            U_t * np.tile(np.array(adata_selected.var["beta"]), (adata_selected.shape[0], 1))
            - S_t * np.tile(np.array(adata_selected.var["gamma"]), (adata_selected.shape[0], 1))
        )
        adata.layers["velocity"] = velocity

        adata.uns["tsvelo_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(input_path.resolve()),
            "cluster_key": cluster_key,
            "sim_type": str(sim_type or "auto"),
            "output_path": str(output_h5ad.resolve()),
        }

        if "velocity" not in adata.layers:
            raise RuntimeError("TSvelo did not produce layers['velocity'].")

        adata.write(output_h5ad, compression="lzf")
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata
        cleanup_resources()


def run_batch_tsvelo_sim(
    metadata_file,
    output_dir,
    cluster_key: str = "vis_annotation",
    n_jobs: int = -1,
    n_neighbors: int = 30,
    n_selected_genes: int = 100,
    n_steps: int = 500,
    n_epoch: int = 10,
    num_epochs: int = 100,
    min_decrease: float = 0.0,
    n_genes2show: int = 0,
    cuda: Optional[int] = None,
    grn_100_csv: str = DEFAULT_GRN_100_CSV,
    grn_1139_csv: str = DEFAULT_GRN_1139_CSV,
    dyngen_feature_network_csv: str = DEFAULT_DYNGEN_FEATURE_NETWORK_CSV,
    dyngen_correlation_top_k: int = 10,
    sim_type: Optional[str] = None,
    save_pdf: bool = False,
    seed: int = 2024,
    overwrite: bool = False,
) -> List[Path]:
    """Run the simulated TSvelo pipeline over a metadata manifest."""
    metadata_file = Path(metadata_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(metadata_file)
    outputs: List[Path] = []
    print(f"Batch mode: {len(metadata_df)} simulated datasets")

    for _, row in metadata_df.iterrows():
        file_path = Path(str(row["file_path"]))
        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue
        try:
            outputs.append(
                run_tsvelo_sim_analysis(
                    input_path=file_path,
                    output_dir=output_dir,
                    cluster_key=str(row.get("cluster_key", cluster_key)),
                    dataset_name=str(row.get("dataset_name", file_path.stem)),
                    n_jobs=n_jobs,
                    n_neighbors=n_neighbors,
                    n_selected_genes=n_selected_genes,
                    n_steps=n_steps,
                    n_epoch=n_epoch,
                    num_epochs=num_epochs,
                    min_decrease=min_decrease,
                    n_genes2show=n_genes2show,
                    cuda=cuda,
                    grn_100_csv=grn_100_csv,
                    grn_1139_csv=grn_1139_csv,
                    dyngen_feature_network_csv=dyngen_feature_network_csv,
                    dyngen_correlation_top_k=dyngen_correlation_top_k,
                    sim_type=sim_type,
                    save_pdf=save_pdf,
                    seed=seed,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep batch runs going
            print(f"Failed: {file_path.name}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TSvelo RNA velocity (simulated data with a GRN prior)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input H5AD file")
    input_group.add_argument("--metadata-file", help="Metadata CSV/TSV file for batch processing")

    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", default=None,
                        help="Dataset folder name")
    parser.add_argument("--cluster-key", "--cluster_key", dest="cluster_key", default="vis_annotation",
                        help="Column name in adata.obs used for cell-type labels")
    parser.add_argument("--sim-type", dest="sim_type", choices=["scmultisim", "dyngen", "auto"],
                        default="auto", help="Simulation engine; 'auto' detects it from gene names")
    parser.add_argument("--n-jobs", "--n_jobs", dest="n_jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--n-neighbors", "--n_neighbors", dest="n_neighbors", type=int, default=30,
                        help="Number of neighbors for the KNN graph")
    parser.add_argument("--n-selected-genes", "--n_selected_genes", dest="n_selected_genes", type=int, default=100,
                        help="Number of selected velocity genes")
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
    parser.add_argument("--grn-100-csv", dest="grn_100_csv", default=DEFAULT_GRN_100_CSV,
                        help="scmultisim GRN csv for small networks (~100 regulators)")
    parser.add_argument("--grn-1139-csv", dest="grn_1139_csv", default=DEFAULT_GRN_1139_CSV,
                        help="scmultisim GRN csv for large networks (~1100 regulators)")
    parser.add_argument("--dyngen-feature-network-csv", dest="dyngen_feature_network_csv",
                        default=DEFAULT_DYNGEN_FEATURE_NETWORK_CSV,
                        help="dyngen model feature_network csv (true GRN edges)")
    parser.add_argument("--dyngen-correlation-top-k", dest="dyngen_correlation_top_k", type=int, default=10,
                        help="Top-k correlated TFs kept per gene when rebuilding a dyngen GRN")
    parser.add_argument("--save-pdf", action="store_true", default=False,
                        help="Also save PDF figures in addition to PNG")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    sim_type = None if args.sim_type == "auto" else args.sim_type

    if args.metadata_file:
        return run_batch_tsvelo_sim(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            cluster_key=args.cluster_key,
            n_jobs=args.n_jobs,
            n_neighbors=args.n_neighbors,
            n_selected_genes=args.n_selected_genes,
            n_steps=args.n_steps,
            n_epoch=args.n_epoch,
            num_epochs=args.num_epochs,
            min_decrease=args.min_decrease,
            n_genes2show=args.n_genes2show,
            cuda=args.cuda,
            grn_100_csv=args.grn_100_csv,
            grn_1139_csv=args.grn_1139_csv,
            dyngen_feature_network_csv=args.dyngen_feature_network_csv,
            dyngen_correlation_top_k=args.dyngen_correlation_top_k,
            sim_type=sim_type,
            save_pdf=args.save_pdf,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    return run_tsvelo_sim_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        n_jobs=args.n_jobs,
        n_neighbors=args.n_neighbors,
        n_selected_genes=args.n_selected_genes,
        n_steps=args.n_steps,
        n_epoch=args.n_epoch,
        num_epochs=args.num_epochs,
        min_decrease=args.min_decrease,
        n_genes2show=args.n_genes2show,
        cuda=args.cuda,
        grn_100_csv=args.grn_100_csv,
        grn_1139_csv=args.grn_1139_csv,
        dyngen_feature_network_csv=args.dyngen_feature_network_csv,
        dyngen_correlation_top_k=args.dyngen_correlation_top_k,
        sim_type=sim_type,
        save_pdf=args.save_pdf,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
