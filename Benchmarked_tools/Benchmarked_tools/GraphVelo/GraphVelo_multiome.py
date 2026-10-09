#!/usr/bin/env python3
"""
GraphVelo multi-omics (RNA + ATAC) velocity analysis pipeline.

This is the multi-modal companion of GraphVelo.py. It builds an LSI embedding
from the ATAC peaks, fuses the RNA and ATAC neighbourhoods with a weighted
nearest-neighbour graph, derives a gene-activity layer Mc, runs the scVelo
dynamical pipeline on the fused graph and finally trains GraphVelo.

Installation:
    pip install graphvelo scvelo==0.2.5 GPUtil pynvml numpy==1.23.5 pygam

Usage:
    python GraphVelo_multiome.py --rna-input rna.h5ad --atac-input atac.h5ad \
        --output-dir ./output --cluster-key celltype
    python GraphVelo_multiome.py --metadata-file multiome.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from graphvelo.graph_velocity import GraphVelo
from graphvelo.mo import gen_wnn
from graphvelo.utils import adj_to_knn, mack_score


def seed_everything(seed: int) -> None:
    """Seed the random number generators used by the pipeline."""
    np.random.seed(seed)


def cleanup_resources() -> None:
    """Release memory held by the previous dataset."""
    gc.collect()


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

    required_columns = ["dataset_name", "rna_path", "atac_path"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if "cluster_key" not in df.columns:
        df["cluster_key"] = "celltype"
    if "dimred_key" not in df.columns:
        df["dimred_key"] = "X_umap"

    df["dataset_name"] = df["dataset_name"].astype(str)
    df["rna_path"] = df["rna_path"].astype(str)
    df["atac_path"] = df["atac_path"].astype(str)
    df["cluster_key"] = df["cluster_key"].astype(str)
    df["dimred_key"] = df["dimred_key"].astype(str)
    return df


def derive_output_stem(input_path: Path) -> str:
    """Return the per-dataset output stem used by the benchmark."""
    stem = input_path.stem
    if stem.endswith("_dataset"):
        return stem[:-8]
    return stem


def run_lsi(adata_atac, n_components: int = 50, drop_first: bool = True):
    """Compute a TF-IDF / TruncatedSVD LSI embedding for the ATAC peaks."""
    X = adata_atac.X.copy()

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    X_tf = normalize(X, norm="l1", axis=1)

    n_cells = X.shape[0]
    peak_cells = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log1p(n_cells / (1 + peak_cells))
    X_tfidf = X_tf.multiply(idf)

    # Optional scaling commonly used for ATAC TF-IDF.
    X_tfidf = X_tfidf * 1e4
    X_tfidf.data = np.log1p(X_tfidf.data)

    n_svd = n_components + 1 if drop_first else n_components
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    X_lsi = svd.fit_transform(X_tfidf)

    if drop_first:
        X_lsi = X_lsi[:, 1:n_components + 1]

    adata_atac.obsm["X_lsi"] = X_lsi
    return adata_atac


def prepare_fused_adata(adata, adata_atac_peak, adata_atac, n_lsi_components: int, wnn_k: int):
    """Build the RNA+ATAC fused AnnData with the WNN graph and gene-activity layer."""
    adata_atac_peak = run_lsi(adata_atac_peak, n_components=n_lsi_components)
    adata.obsm["X_lsi"] = adata_atac_peak.obsm["X_lsi"].copy()

    adata = gen_wnn(adata, copy=True, k=wnn_k)
    adata.uns["neighbors"] = adata.uns["WNN"].copy()
    adata.obsp["connectivities"] = adata.obsp["WNN"].copy()
    adata.obsp["distances"] = adata.obsp["WNN_distance"].copy()
    adata.layers["Mc"] = adata.obsp["WNN"] @ adata_atac.X
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
    adata.layers["velocity_gv"] = gv.project_velocity(adata.layers["Ms"])
    adata.layers["velocity_c"] = gv.project_velocity(adata.layers["Mc"])
    if "X_pca" in adata.obsm:
        adata.obsm["gv_pca"] = gv.project_velocity(adata.obsm["X_pca"])
    if "X_tsne" in adata.obsm:
        adata.obsm["gv_tsne"] = gv.project_velocity(adata.obsm["X_tsne"])


def run_graphvelo_multiome_analysis(
    rna_input: str | Path,
    atac_input: str | Path,
    output_dir: str | Path,
    cluster_key: str = "celltype",
    dataset_name: Optional[str] = None,
    dimred_key: str = "X_umap",
    n_lsi_components: int = 50,
    wnn_k: int = 50,
    n_jobs: int = 8,
    n_mack_genes: int = 200,
    overwrite: bool = False,
    seed: int = 2024,
) -> Path:
    """Run the GraphVelo multi-omics pipeline and write the velocity result."""
    rna_input = Path(rna_input)
    atac_input = Path(atac_input)
    output_dir = Path(output_dir)
    if not rna_input.exists():
        raise FileNotFoundError(f"RNA input file not found: {rna_input}")
    if not atac_input.exists():
        raise FileNotFoundError(f"ATAC input file not found: {atac_input}")

    if dataset_name is None:
        dataset_name = derive_output_stem(rna_input)

    dataset_output_dir = output_dir / str(dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    output_stem = derive_output_stem(rna_input)
    # Multi-omic results keep the GraphVelo native file name.
    output_h5ad = dataset_output_dir / f"{output_stem}_graphvelo.h5ad"

    if output_h5ad.exists() and not overwrite:
        print(f"Skipping existing output: {output_h5ad}")
        return output_h5ad

    seed_everything(seed)

    adata = None
    try:
        print(f"\nProcessing: {rna_input.name} + {atac_input.name}")
        adata = sc.read(rna_input)
        adata_atac_peak = sc.read(atac_input)

        if "Mc" not in adata.layers:
            raise ValueError("Missing required gene-activity layer 'Mc' in the RNA input")

        adata_atac = adata.copy()
        adata_atac.X = adata.layers["Mc"]
        if cluster_key not in adata.obs.columns:
            raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

        print("  Building the fused RNA+ATAC graph...")
        adata = prepare_fused_adata(
            adata,
            adata_atac_peak,
            adata_atac,
            n_lsi_components=n_lsi_components,
            wnn_k=wnn_k,
        )

        print("  Preprocessing (scVelo dynamical)...")
        scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
        scv.tl.recover_dynamics(adata, n_jobs=n_jobs)
        scv.tl.velocity(adata)
        scv.tl.velocity_graph(adata, n_jobs=n_jobs)
        scv.tl.latent_time(adata)

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

        adata.uns["graphvelo_multiome_run"] = {
            "dataset_name": str(dataset_name),
            "rna_input": str(rna_input.resolve()),
            "atac_input": str(atac_input.resolve()),
            "cluster_key": cluster_key,
            "dimred_key": dimred_key,
            "n_lsi_components": int(n_lsi_components),
            "wnn_k": int(wnn_k),
            "n_mack_genes": int(n_mack_genes),
            "seed": int(seed),
            "output_path": str(output_h5ad.resolve()),
        }

        if "velocity" not in adata.layers:
            raise RuntimeError("GraphVelo did not produce layers['velocity'].")

        adata.write(output_h5ad)
        shutil.copyfile(output_h5ad, dataset_output_dir / "rc.h5ad")
        print(f"  Done: {output_h5ad}")
        return output_h5ad
    finally:
        del adata
        cleanup_resources()


def run_batch_graphvelo_multiome(
    metadata_file: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    seed: int = 2024,
) -> list[Path]:
    """Run the multi-omics pipeline for every dataset listed in a metadata table."""
    metadata_file = Path(metadata_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_file(metadata_file)
    outputs: list[Path] = []
    print(f"Batch mode: {len(metadata_df)} datasets")

    for _, row in metadata_df.iterrows():
        rna_path = Path(row["rna_path"])
        atac_path = Path(row["atac_path"])
        if not rna_path.exists() or not atac_path.exists():
            print(f"Skipping missing inputs: {rna_path} / {atac_path}")
            continue
        try:
            output_path = run_graphvelo_multiome_analysis(
                rna_input=rna_path,
                atac_input=atac_path,
                output_dir=output_dir,
                cluster_key=row["cluster_key"],
                dataset_name=row["dataset_name"],
                dimred_key=row["dimred_key"],
                overwrite=overwrite,
                seed=seed,
            )
            outputs.append(output_path)
        except Exception as exc:
            print(f"Failed: {row['dataset_name']}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GraphVelo multi-omics (RNA + ATAC) velocity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--rna-input", help="Input RNA H5AD file (must contain the Mc gene-activity layer)")
    input_group.add_argument("--metadata-file", help="Metadata CSV/TSV file for batch processing")

    parser.add_argument("--atac-input", default=None, help="Input ATAC peak H5AD file (single-file mode)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", default=None, help="Dataset folder name for single-file mode")
    parser.add_argument("--cluster-key", default="celltype", help="Column name in adata.obs used for labels")
    parser.add_argument("--dimred-key", default="X_umap", help="Dimensionality reduction key in adata.obsm")
    parser.add_argument("--n-lsi-components", type=int, default=50, help="Number of LSI components fitted on the ATAC peaks")
    parser.add_argument("--wnn-k", type=int, default=50, help="Neighbours used to build the weighted nearest-neighbour graph")
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
        return run_batch_graphvelo_multiome(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    if not args.atac_input:
        parser.error("--atac-input is required when --rna-input is used")

    return run_graphvelo_multiome_analysis(
        rna_input=args.rna_input,
        atac_input=args.atac_input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        dimred_key=args.dimred_key,
        n_lsi_components=args.n_lsi_components,
        wnn_k=args.wnn_k,
        n_jobs=args.n_jobs,
        n_mack_genes=args.n_mack_genes,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
