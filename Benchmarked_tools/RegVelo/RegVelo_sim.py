#!/usr/bin/env python3
"""
RegVelo RNA velocity pipeline for VelocityBenchmarking (simulated data).

Simulated datasets have no real species database, so the pySCENIC prior used for
real data is replaced by an a-priori GRN (converted into the 0/1 regulator-target
matrix that RegVelo's ``set_prior_grn`` expects). The GRN source depends on the
simulation engine:

* scmultisim datasets use the known regulatory network shipped as a long-format
  GRN csv (``GRN_params_100.csv`` / ``GRN_params_1139.csv``).
* dyngen datasets have no prior file in use, so the GRN is always rebuilt from
  Spearman correlation between the known TFs and every gene. If a reliable
  simulated GRN for a dyngen dataset is available (such as a dumped dyngen
  ``model$feature_network`` edge list), the commented block in
  :func:`load_prior_grn` and the matching commented definitions in
  ``grn_utils.py`` show how to use those true edges instead.

Because the GRN handling is genuinely different, this simulated branch is kept as
a **separate** script from RegVelo.py (real data). pySCENIC is therefore required
exclusively by the real-data workflow.

Installation:
    pip install regvelo scanpy scvelo GPUtil pynvml psutil pandas numpy scipy scikit-learn
    # A GPU-enabled PyTorch build is installed automatically as a dependency of regvelo.

Usage:
    python RegVelo_sim.py --input sim.h5ad --output-dir ./output --cluster-key vis_annotation
    python RegVelo_sim.py --input sim.h5ad --output-dir ./output \
        --grn-100-csv GRN_params_100.csv --grn-1139-csv GRN_params_1139.csv
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grn_utils import (  # noqa: E402 - imported after the sys.path setup above
    build_regvelo_grn_from_correlation,
    build_regvelo_grn_from_scmultisim,
    choose_grn_csv,
    detect_sim_type,
    detect_tf_from_names,
    grn_regulatory_genes,
    load_grn,
)

# The optional dyngen edge-list branch in load_prior_grn() (see the commented
# block below) additionally needs "build_regvelo_grn" from grn_utils and, if the
# dataset-to-csv map is enabled, "DYNGEN_MODEL_GRN" / "load_dyngen_feature_network".
# Add those names to the import above when you enable that branch.

DEFAULT_GRN_100_CSV = str(SCRIPT_DIR / "GRN_params_100.csv")
DEFAULT_GRN_1139_CSV = str(SCRIPT_DIR / "GRN_params_1139.csv")

_REGVELO_API = {}


# ---------------------------------------------------------------------------
# Environment / package helpers
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


def pick_free_gpu() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            universal_newlines=True,
        )
        best_idx, best_free = 0, -1
        for line in out.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            free = int(parts[1])
            if free > best_free:
                best_free = free
                best_idx = int(parts[0])
        return best_idx
    except Exception as exc:  # noqa: BLE001
        print(f"[device] nvidia-smi query failed, using GPU 0: {exc}", file=sys.stderr)
        return 0


def configure_cuda(cuda: Optional[int]) -> Optional[str]:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    if cuda is None:
        cuda = pick_free_gpu()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cuda))
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def load_regvelo_api():
    if not _REGVELO_API:
        import torch
        import regvelo as rgv
        from regvelo import REGVELOVI

        _REGVELO_API.update(torch=torch, rgv=rgv, REGVELOVI=REGVELOVI)
    return _REGVELO_API


def cleanup_resources() -> None:
    gc.collect()
    plt.close("all")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def load_prior_grn(
    adata,
    grn_100_csv: str,
    grn_1139_csv: str,
    data_file_name: Optional[str] = None,
    sim_type: Optional[str] = None,
) -> tuple:
    """Return ``(grn_matrix_or_None, retain_genes, tf_list_or_None, sim_type)``.

    scmultisim datasets use the matching a-priori GRN csv. dyngen datasets have no
    prior file in use, so ``grn_matrix`` stays ``None`` and the caller rebuilds the
    GRN from Spearman correlation after ``moments``. See the commented block below
    for how to switch a dyngen dataset to a reliable dumped GRN edge list.

    ``data_file_name`` is the input h5ad file name; it identifies the dataset and
    is used by the optional dyngen edge-list lookup.
    """
    detected_type, n_reg = detect_sim_type(adata)
    sim_type = sim_type or detected_type
    tf_list = None

    if sim_type == "scmultisim":
        grn_df = load_grn(choose_grn_csv(n_reg, grn_100_csv, grn_1139_csv))
        grn_matrix = build_regvelo_grn_from_scmultisim(grn_df)
        retain_genes = grn_regulatory_genes(grn_df)
        print(f"scmultisim detected: {n_reg} regulatory genes, GRN edges={len(grn_df)}")
    else:
        # dyngen: no prior file is used, the GRN is always rebuilt by Spearman
        # correlation on the known ``*_TF1`` genes.
        tf_list = detect_tf_from_names(adata)
        grn_matrix = None
        retain_genes = tf_list
        print(f"dyngen detected ({data_file_name or 'input'}): {len(tf_list)} TFs; "
              f"GRN will be rebuilt by Spearman correlation")

        # ------------------------------------------------------------------
        # Optional: use a reliable dyngen GRN instead of Spearman correlation.
        # ------------------------------------------------------------------
        # If a trustworthy simulated GRN is available for this dyngen dataset --
        # for example the dataset's dyngen model ``model$feature_network`` edge
        # list dumped to csv, such as the shipped
        # ``dyngen_bifurcating_cell10000_gene500_feature_network.csv`` -- uncomment
        # both this block and the matching definitions in grn_utils.py and add
        # "build_regvelo_grn" (+ "DYNGEN_MODEL_GRN" /
        # "load_dyngen_feature_network") to the grn_utils import at the top of this
        # file. The true edges then replace the correlation-based GRN; datasets
        # without a reliable GRN keep the Spearman fallback.
        #
        # The lookup key is the dataset name with the benchmark ``_dataset`` suffix
        # stripped, matching DYNGEN_MODEL_GRN and derive_output_stem().
        #
        # dyngen_key = Path(data_file_name or "").stem
        # if dyngen_key.endswith("_dataset"):
        #     dyngen_key = dyngen_key[:-8]
        # dyngen_grn_csv = DYNGEN_MODEL_GRN.get(dyngen_key)
        # if dyngen_grn_csv and os.path.exists(dyngen_grn_csv):
        #     edges_df = load_dyngen_feature_network(dyngen_grn_csv)
        #     grn_matrix = build_regvelo_grn(edges_df)
        #     retain_genes = sorted(
        #         {str(r) for r in edges_df["regulator"]}
        #         | {str(t) for t in edges_df["target"]}
        #     )
        #     print(f"dyngen: using the dumped GRN edge list {dyngen_grn_csv}")

    return grn_matrix, retain_genes, tf_list, sim_type


def preprocess_adata(
    adata,
    cluster_key: str = "vis_annotation",
    data_file_name: Optional[str] = None,
    grn_100_csv: str = DEFAULT_GRN_100_CSV,
    grn_1139_csv: str = DEFAULT_GRN_1139_CSV,
    dyngen_correlation_top_k: int = 10,
    sim_type: Optional[str] = None,
    n_pcs: int = 30,
    n_neighbors: int = 30,
):
    """Normalized RegVelo preprocessing for simulated data with a GRN prior."""
    api = load_regvelo_api()
    rgv = api["rgv"]

    adata.obs_names_make_unique()
    if adata.n_vars == 0:
        raise ValueError("Input AnnData has zero genes")

    if cluster_key in adata.obs.columns:
        adata.obs["clusters"] = adata.obs[cluster_key].astype(str)
    if "X_dimred" in adata.obsm:
        adata.obsm["X_umap"] = adata.obsm["X_dimred"]

    grn_matrix, retain_genes, tf_list, sim_type = load_prior_grn(
        adata, grn_100_csv, grn_1139_csv, data_file_name, sim_type
    )

    if adata.n_vars < 2000:
        top_gene = min((adata.n_vars // 500) * 500, adata.n_vars - 1)
    else:
        top_gene = 2000
    top_gene = max(1, int(top_gene))

    retain_genes = [g for g in retain_genes if g in list(adata.var_names)]
    scv.pp.filter_and_normalize(adata, min_shared_counts=None, retain_genes=retain_genes)
    sc.pp.log1p(adata)
    for layer in ("spliced", "unspliced"):
        if layer in adata.layers:
            adata.layers[layer] = np.log1p(adata.layers[layer])
    sc.pp.highly_variable_genes(adata, n_top_genes=top_gene, subset=False)
    keep = adata.var["highly_variable"] | adata.var_names.isin(retain_genes)
    adata = adata[:, keep].copy()
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    if grn_matrix is None:
        grn_matrix = build_regvelo_grn_from_correlation(adata, tf_list, top_k=dyngen_correlation_top_k)

    # set_prior_grn expects rows=targets, columns=regulators -> transpose.
    adata = rgv.pp.set_prior_grn(adata, grn_matrix.T)

    scv.tl.velocity(adata, mode="deterministic")
    scv.tl.velocity_genes(adata)
    velocity_genes = adata.var_names[adata.var["velocity_genes"]].tolist()
    tf_mask = adata.var_names[adata.uns["skeleton"].sum(1) != 0]
    var_mask = np.union1d(tf_mask, velocity_genes)
    adata = adata[:, var_mask].copy()
    adata.uns["skeleton"] = adata.uns["skeleton"].loc[
        adata.var_names.tolist(), adata.var_names.tolist()
    ]

    if sim_type != "scmultisim":
        adata = rgv.pp.filter_genes(adata)
    adata = rgv.pp.preprocess_data(adata, filter_on_r2=False)
    adata.var["velocity_genes"] = adata.var_names.isin(velocity_genes)
    adata.var["TF"] = adata.var_names.isin(tf_mask)
    return adata


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def train_regvelo(
    adata,
    max_epochs: int = 500,
    batch_size: Optional[int] = None,
    train_size: float = 0.8,
    soft_constraint: bool = True,
    lam: float = 1.0,
    lam2: float = 0.0,
):
    """Build and train a RegVelo model, returning it (velocity written to adata)."""
    api = load_regvelo_api()
    torch = api["torch"]
    rgv = api["rgv"]
    REGVELOVI = api["REGVELOVI"]

    weights = adata.uns["skeleton"].copy()
    weights = torch.tensor(np.array(weights).astype(np.float32)).int()
    weights = weights.T  # regulators x targets

    tf_list = adata.var_names[adata.var["TF"]]

    REGVELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")

    vae = REGVELOVI(
        adata,
        W=weights,
        regulators=tf_list,
        soft_constraint=soft_constraint,
        lam=lam,
        lam2=lam2,
    )

    if batch_size is None and adata.n_obs > 10000:
        batch_size = 512
        print(f"[INFO] {adata.n_obs} cells > 10000; mini-batch training (batch_size={batch_size})")

    vae.train(
        max_epochs=max_epochs,
        lr=1e-2,
        weight_decay=1e-5,
        eps=1e-16,
        train_size=train_size,
        batch_size=batch_size,
        validation_size=None,
        early_stopping=True,
        gradient_clip_val=10,
        optimizer="AdamW",
    )

    rgv.tl.set_output(adata, vae, n_samples=30, batch_size=adata.n_obs)
    return vae


def derive_output_stem(input_path: Path) -> str:
    """Strip the benchmark `_dataset` suffix from an input file stem."""
    stem = input_path.stem
    if stem.endswith("_dataset"):
        return stem[:-8]
    return stem


def run_regvelo_sim_analysis(
    input_path,
    output_dir,
    cluster_key: str = "vis_annotation",
    dataset_name: Optional[str] = None,
    grn_100_csv: str = DEFAULT_GRN_100_CSV,
    grn_1139_csv: str = DEFAULT_GRN_1139_CSV,
    dyngen_correlation_top_k: int = 10,
    sim_type: Optional[str] = None,
    max_epochs: int = 500,
    batch_size: Optional[int] = None,
    train_size: float = 0.8,
    soft_constraint: bool = True,
    lam: float = 1.0,
    lam2: float = 0.0,
    save_model: bool = True,
    cuda: Optional[int] = None,
    seed: int = 2024,
    overwrite: bool = False,
) -> Path:
    """Run the full RegVelo pipeline for one simulated h5ad dataset."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    dataset_name = dataset_name or input_path.stem
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_h5ad = dataset_dir / f"{derive_output_stem(input_path)}.h5ad"

    if output_h5ad.exists() and not overwrite:
        print(f"Skipping existing output: {output_h5ad}")
        return output_h5ad

    seed_everything(seed)
    configure_cuda(cuda)

    adata = None
    try:
        print(f"\nProcessing: {input_path.name}")
        adata = sc.read(input_path)
        adata = preprocess_adata(
            adata,
            cluster_key=cluster_key,
            data_file_name=input_path.name,
            grn_100_csv=grn_100_csv,
            grn_1139_csv=grn_1139_csv,
            dyngen_correlation_top_k=dyngen_correlation_top_k,
            sim_type=sim_type,
        )
        adata.write(dataset_dir / "pp.h5ad")

        print("  Training RegVelo...")
        vae = train_regvelo(
            adata,
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_size=train_size,
            soft_constraint=soft_constraint,
            lam=lam,
            lam2=lam2,
        )

        if save_model:
            model_path = dataset_dir / "regvelo_model"
            vae.save(str(model_path), overwrite=True)
            print(f"[INFO] RegVelo model saved: {model_path}")

        if "velocity" not in adata.layers:
            raise RuntimeError("RegVelo did not write layers['velocity'].")

        try:
            sc.tl.pca(adata, layer="Ms")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] PCA on Ms skipped: {exc}")

        adata.uns["regvelo_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(input_path.resolve()),
            "cluster_key": cluster_key,
            "sim_type": str(sim_type or "auto"),
            "output_path": str(output_h5ad.resolve()),
        }
        adata.write(output_h5ad, compression="lzf")
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata
        cleanup_resources()


def run_batch_regvelo_sim(
    metadata_file,
    output_dir,
    cluster_key: str = "vis_annotation",
    grn_100_csv: str = DEFAULT_GRN_100_CSV,
    grn_1139_csv: str = DEFAULT_GRN_1139_CSV,
    dyngen_correlation_top_k: int = 10,
    sim_type: Optional[str] = None,
    max_epochs: int = 500,
    batch_size: Optional[int] = None,
    cuda: Optional[int] = None,
    seed: int = 2024,
    overwrite: bool = False,
) -> List[Path]:
    """Run the simulated RegVelo pipeline over a metadata manifest."""
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
                run_regvelo_sim_analysis(
                    input_path=file_path,
                    output_dir=output_dir,
                    cluster_key=str(row.get("cluster_key", cluster_key)),
                    dataset_name=str(row.get("dataset_name", file_path.stem)),
                    grn_100_csv=grn_100_csv,
                    grn_1139_csv=grn_1139_csv,
                    dyngen_correlation_top_k=dyngen_correlation_top_k,
                    sim_type=sim_type,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    cuda=cuda,
                    seed=seed,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep batch runs going
            print(f"Failed: {file_path.name}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RegVelo RNA velocity (simulated data with a GRN prior)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input H5AD file")
    input_group.add_argument("--metadata-file", help="Metadata CSV file for batch processing")

    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", dest="dataset_name", default=None, help="Dataset folder name")
    parser.add_argument("--cluster-key", dest="cluster_key", default="vis_annotation",
                        help="Column name in adata.obs used for cell-type labels")
    parser.add_argument("--sim-type", dest="sim_type", choices=["scmultisim", "dyngen", "auto"],
                        default="auto", help="Simulation engine; 'auto' detects it from gene names")
    parser.add_argument("--grn-100-csv", dest="grn_100_csv", default=DEFAULT_GRN_100_CSV,
                        help="scmultisim GRN csv for small networks (~100 regulators)")
    parser.add_argument("--grn-1139-csv", dest="grn_1139_csv", default=DEFAULT_GRN_1139_CSV,
                        help="scmultisim GRN csv for large networks (~1100 regulators)")
    parser.add_argument("--dyngen-correlation-top-k", dest="dyngen_correlation_top_k", type=int, default=10,
                        help="Top-k correlated TFs kept per gene when rebuilding a dyngen GRN by Spearman correlation")
    parser.add_argument("--max-epochs", dest="max_epochs", type=int, default=500,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                        help="Mini-batch size (None = all cells; large datasets suggest 512)")
    parser.add_argument("--train-size", dest="train_size", type=float, default=0.8,
                        help="Fraction of cells used for training")
    parser.add_argument("--soft-constraint", dest="soft_constraint", action="store_true", default=True,
                        help="Use the soft regulatory constraint")
    parser.add_argument("--no-soft-constraint", dest="soft_constraint", action="store_false",
                        help="Disable the soft regulatory constraint")
    parser.add_argument("--lam", dest="lam", type=float, default=1.0, help="First regularization weight")
    parser.add_argument("--lam2", dest="lam2", type=float, default=0.0, help="Second regularization weight")
    parser.add_argument("--no-save-model", dest="save_model", action="store_false", default=True,
                        help="Do not save the trained RegVelo model")
    parser.add_argument("--cuda", type=int, default=None,
                        help="CUDA device id; when omitted the freest GPU is auto-selected")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    sim_type = None if args.sim_type == "auto" else args.sim_type

    if args.metadata_file:
        return run_batch_regvelo_sim(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            cluster_key=args.cluster_key,
            grn_100_csv=args.grn_100_csv,
            grn_1139_csv=args.grn_1139_csv,
            dyngen_correlation_top_k=args.dyngen_correlation_top_k,
            sim_type=sim_type,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            cuda=args.cuda,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    return run_regvelo_sim_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        grn_100_csv=args.grn_100_csv,
        grn_1139_csv=args.grn_1139_csv,
        dyngen_correlation_top_k=args.dyngen_correlation_top_k,
        sim_type=sim_type,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        train_size=args.train_size,
        soft_constraint=args.soft_constraint,
        lam=args.lam,
        lam2=args.lam2,
        save_model=args.save_model,
        cuda=args.cuda,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
