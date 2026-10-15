#!/usr/bin/env python3
"""
veloAgent spatial RNA velocity analysis pipeline for VelocityBenchmarking.

veloAgent couples a cell-state VAE, a protein-interaction-constrained gene network
and an agent-based model to produce a spatial RNA velocity. It needs the
spliced/unspliced layers, the spatial coordinates and the protein interaction
tables shipped with the upstream repository.

The pipeline is a statement-for-statement port of the original driver
(``velocity_spatial_code/veloagent_run.py``): the same layer / ``obsm`` / ``uns``
keys and the same local variable names. Only the argument parsing, the metadata
batch loop and the ``uns`` run record are wrapper plumbing.

Installation:
    See the repository Dockerfile, which installs the upstream veloAgent source
    plus the pinned scientific stack:
        git clone https://github.com/mcgilldinglab/veloAgent.git /opt/veloAgent
    The protein interaction tables are read from the checkout through
    ``--protein-data-dir`` (or the ``VELOAGENT_PROTEIN_DIR`` environment variable).

Usage:
    python veloAgent.py --input spatial.h5ad --output-dir ./output --spatial-key spatial
    python veloAgent.py --metadata-file datasets.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# The upstream protein interaction tables live in the veloAgent checkout; the
# location can be overridden without editing the script.
DEFAULT_PROTEIN_DATA_DIR = os.environ.get("VELOAGENT_PROTEIN_DIR", "/opt/veloAgent")

DEFAULT_SPLICED_LAYER = "spliced"
DEFAULT_UNSPLICED_LAYER = "unspliced"

# ``veloagent.CellModel.step`` writes the agent-based velocity directly into
# ``adata.layers['velocity']`` (together with ``velocity_u``), so 'velocity' is the
# native key and no aliasing is required.
NATIVE_VELOCITY_KEY = "velocity"


@lru_cache(maxsize=1)
def load_veloagent_api():
    """Import the installed ``veloagent`` package without local shadowing.

    When this script is executed directly its own directory is prepended to
    ``sys.path``, so the script directory is temporarily removed while importing
    the real package.
    """
    script_dir = str(Path(__file__).resolve().parent)
    removed_path = False
    if script_dir in sys.path:
        sys.path.remove(script_dir)
        removed_path = True

    local_module = None
    restore_local_module = __name__ == "veloAgent" and "veloAgent" in sys.modules
    if restore_local_module:
        local_module = sys.modules.pop("veloAgent")

    try:
        veloagent = importlib.import_module("veloagent")
    finally:
        if restore_local_module and local_module is not None:
            sys.modules["veloAgent"] = local_module
        if removed_path:
            sys.path.insert(0, script_dir)

    return veloagent


def seed_everything(seed: int) -> None:
    """Seed the random number generators used by veloAgent."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_resources() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        df["cluster_key"] = ""
    if "spatial_key" not in df.columns:
        df["spatial_key"] = "spatial"

    df["dataset_name"] = df["dataset_name"].astype(str)
    df["file_path"] = df["file_path"].astype(str)
    df["cluster_key"] = df["cluster_key"].fillna("").astype(str)
    df["spatial_key"] = df["spatial_key"].fillna("spatial").astype(str)

    return df


def preprocess_adata(
    adata,
    spatial_key: str = "spatial",
    spliced_layer: str = DEFAULT_SPLICED_LAYER,
    unspliced_layer: str = DEFAULT_UNSPLICED_LAYER,
    cluster_key: Optional[str] = None,
):
    """Standardize the layer and coordinate keys required by veloAgent.

    The checks, their order and the key names are the ones of the original
    ``run_one``.
    """
    if cluster_key and cluster_key not in adata.obs:
        raise KeyError(f"Missing cell type column: {cluster_key}")
    if spatial_key not in adata.obsm:
        raise KeyError(f"Missing spatial coordinates: adata.obsm[{spatial_key!r}]")
    for layer in (spliced_layer, unspliced_layer):
        if layer not in adata.layers:
            raise KeyError(f"Missing layer: {layer}")
    if spatial_key != "spatial":
        adata.obsm["spatial"] = adata.obsm[spatial_key].copy()
    for source, target in ((spliced_layer, "spliced"), (unspliced_layer, "unspliced")):
        if source != target:
            adata.layers[target] = adata.layers[source].copy()
    return adata


def run_veloagent_analysis(
    input_path,
    output_dir,
    spatial_key: str = "spatial",
    spliced_layer: str = DEFAULT_SPLICED_LAYER,
    unspliced_layer: str = DEFAULT_UNSPLICED_LAYER,
    cluster_key: Optional[str] = None,
    dataset_name: Optional[str] = None,
    protein_data_dir: str = DEFAULT_PROTEIN_DATA_DIR,
    species: str = "human",
    device: str = "cuda:0",
    z_dim: int = 5,
    vae_lr: float = 1e-2,
    genenet_lr: float = 1e-4,
    batch_size: int = 256,
    patience: int = 25,
    vae_epochs: int = 1000,
    gg_epochs: int = 500,
    nbr_epochs: int = 50,
    abm_steps: int = 100,
    overwrite: bool = False,
    seed: int = 144,
) -> Path:
    """Run the full veloAgent pipeline for a single spatial dataset.

    ``root``, ``folder`` and ``out`` are the ones the original ``run_all`` loop
    builds around ``run_one``::

        name = path.stem
        folder = root / name
        out = folder / f"{name}.h5ad"
    """
    veloagent = load_veloagent_api()

    path = Path(input_path)
    root = Path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    name = path.stem
    if dataset_name is None:
        dataset_name = name
    folder = root / str(dataset_name)
    out = folder / f"{name}.h5ad"

    if out.exists() and not overwrite:
        print(f"Skipping existing output: {out}")
        return out
    folder.mkdir(parents=True, exist_ok=True)

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    seed_everything(seed)

    import scanpy as sc

    adata = None
    try:
        adata = sc.read_h5ad(path)
        adata = preprocess_adata(
            adata,
            spatial_key=spatial_key,
            spliced_layer=spliced_layer,
            unspliced_layer=unspliced_layer,
            cluster_key=cluster_key,
        )

        device = torch.device(device)

        veloagent.preprocess(adata)
        vae, optimizer, loss_fn = veloagent.get_vae(adata, z_dim=z_dim, lr=vae_lr)
        vae.to(device)
        global_nb_indices = adata.uns["neighbors"]["indices"]
        best_model = veloagent.train_vae(adata, vae, optimizer, loss_fn, global_nb_indices, patience=patience, num_epochs=vae_epochs, batch=batch_size, device=device)
        vae.load_state_dict(best_model)
        veloagent.get_embedding(adata, vae, device)
        protein_names_path, protein_alias_path, gene_conn_path = veloagent.load_protein_paths(species=species, base=protein_data_dir)
        conn_mat = veloagent.create_con_mat(adata, adata.n_vars, protein_names_path, protein_alias_path, gene_conn_path, varname="index", confidence=False, conf_threshold=400)
        genenet = veloagent.GeneNet(in_dim=adata.obsm["cell_embed"][0].shape[0], gene_dim=adata.n_vars, conn=conn_mat)
        optimizer = torch.optim.Adam(genenet.parameters(), lr=genenet_lr, weight_decay=1e-5)
        genenet.to(device)
        veloagent.train_gg(num_epochs=gg_epochs, data=adata, embed_basis="cell_embed", genenet=genenet, device=device, optimizer=optimizer, patience=patience, num_nbrs=30, dt=0.5, batch=batch_size)
        veloagent.train_nbr(num_epochs=nbr_epochs, data=adata, embed_basis="cell_embed", genenet=genenet, device=device, optimizer=optimizer, num_nbrs=30, dt=0.5, batch=batch_size, tau_nbr=0.2)
        abm = veloagent.CellModel(adata, steps=abm_steps, tau=2, nbr_radius=30, sig_ratio=0.7)
        abm.step(show_progress=True)

        # Benchmark contract: the agent-based step already writes the native
        # layers['velocity'], so only the presence of that layer is verified.
        if NATIVE_VELOCITY_KEY not in adata.layers:
            raise RuntimeError(f"veloAgent did not produce layers['{NATIVE_VELOCITY_KEY}'].")

        adata.uns["veloagent_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(path.resolve()),
            "spatial_key": spatial_key,
            "cluster_key": cluster_key or "",
            "spliced_layer": spliced_layer,
            "unspliced_layer": unspliced_layer,
            "species": species,
            "device": str(device),
            "z_dim": z_dim,
            "batch_size": batch_size,
            "vae_epochs": vae_epochs,
            "gg_epochs": gg_epochs,
            "nbr_epochs": nbr_epochs,
            "abm_steps": abm_steps,
            "native_velocity_key": NATIVE_VELOCITY_KEY,
            "output_path": str(out.resolve()),
        }

        adata.write_h5ad(out)
        return out
    finally:
        del adata
        cleanup_resources()


def run_batch_veloagent(
    metadata_file,
    output_dir,
    spatial_key: str = "spatial",
    spliced_layer: str = DEFAULT_SPLICED_LAYER,
    unspliced_layer: str = DEFAULT_UNSPLICED_LAYER,
    protein_data_dir: str = DEFAULT_PROTEIN_DATA_DIR,
    species: str = "human",
    device: str = "cuda:0",
    z_dim: int = 5,
    vae_lr: float = 1e-2,
    genenet_lr: float = 1e-4,
    batch_size: int = 256,
    patience: int = 25,
    vae_epochs: int = 1000,
    gg_epochs: int = 500,
    nbr_epochs: int = 50,
    abm_steps: int = 100,
    overwrite: bool = False,
    seed: int = 144,
) -> list[Path]:
    """Run veloAgent for every dataset listed in a metadata table."""
    metadata_file = Path(metadata_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_file(metadata_file)
    outputs: list[Path] = []
    print(f"Batch mode: {len(metadata_df)} datasets")

    for _, row in metadata_df.iterrows():
        file_path = Path(str(row["file_path"]))
        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        try:
            outputs.append(
                run_veloagent_analysis(
                    input_path=file_path,
                    output_dir=output_dir,
                    spatial_key=str(row["spatial_key"]) or spatial_key,
                    spliced_layer=spliced_layer,
                    unspliced_layer=unspliced_layer,
                    cluster_key=str(row["cluster_key"]) or None,
                    dataset_name=str(row["dataset_name"]),
                    protein_data_dir=protein_data_dir,
                    species=species,
                    device=device,
                    z_dim=z_dim,
                    vae_lr=vae_lr,
                    genenet_lr=genenet_lr,
                    batch_size=batch_size,
                    patience=patience,
                    vae_epochs=vae_epochs,
                    gg_epochs=gg_epochs,
                    nbr_epochs=nbr_epochs,
                    abm_steps=abm_steps,
                    overwrite=overwrite,
                    seed=seed,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep batch runs going
            print(f"Failed: {row['dataset_name']}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="veloAgent spatial RNA velocity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input H5AD file (single-file mode)")
    input_group.add_argument("--metadata-file", help="Metadata CSV/TSV file for batch processing")

    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", dest="dataset_name", default=None, help="Dataset folder name")
    parser.add_argument(
        "--spatial-key",
        dest="spatial_key",
        default="spatial",
        help="Spatial coordinates key in adata.obsm",
    )
    parser.add_argument(
        "--cluster-key",
        dest="cluster_key",
        default=None,
        help="Column name in adata.obs used for cell labels (optional for veloAgent)",
    )
    parser.add_argument("--spliced-layer", dest="spliced_layer", default=DEFAULT_SPLICED_LAYER,
                        help="Spliced counts layer")
    parser.add_argument("--unspliced-layer", dest="unspliced_layer", default=DEFAULT_UNSPLICED_LAYER,
                        help="Unspliced counts layer")
    parser.add_argument(
        "--protein-data-dir",
        dest="protein_data_dir",
        default=DEFAULT_PROTEIN_DATA_DIR,
        help="Directory containing the veloAgent protein interaction files",
    )
    parser.add_argument("--species", default="human", help="Species used to pick the protein interaction tables")
    parser.add_argument("--device", default="cuda:0", help="Torch device, for example cuda:0 or cpu")
    parser.add_argument("--z-dim", dest="z_dim", type=int, default=5, help="Latent dimension of the VAE")
    parser.add_argument("--vae-lr", dest="vae_lr", type=float, default=1e-2, help="VAE learning rate")
    parser.add_argument("--genenet-lr", dest="genenet_lr", type=float, default=1e-4, help="GeneNet learning rate")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--patience", type=int, default=25, help="Early-stopping patience")
    parser.add_argument("--vae-epochs", dest="vae_epochs", type=int, default=1000, help="VAE training epochs")
    parser.add_argument("--gg-epochs", dest="gg_epochs", type=int, default=500, help="Gene-gene training epochs")
    parser.add_argument("--nbr-epochs", dest="nbr_epochs", type=int, default=50, help="Neighbour training epochs")
    parser.add_argument("--abm-steps", dest="abm_steps", type=int, default=100, help="Agent-based model steps")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=144, help="Random seed (upstream default)")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_veloagent(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            spatial_key=args.spatial_key,
            spliced_layer=args.spliced_layer,
            unspliced_layer=args.unspliced_layer,
            protein_data_dir=args.protein_data_dir,
            species=args.species,
            device=args.device,
            z_dim=args.z_dim,
            vae_lr=args.vae_lr,
            genenet_lr=args.genenet_lr,
            batch_size=args.batch_size,
            patience=args.patience,
            vae_epochs=args.vae_epochs,
            gg_epochs=args.gg_epochs,
            nbr_epochs=args.nbr_epochs,
            abm_steps=args.abm_steps,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    return run_veloagent_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        spatial_key=args.spatial_key,
        spliced_layer=args.spliced_layer,
        unspliced_layer=args.unspliced_layer,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        protein_data_dir=args.protein_data_dir,
        species=args.species,
        device=args.device,
        z_dim=args.z_dim,
        vae_lr=args.vae_lr,
        genenet_lr=args.genenet_lr,
        batch_size=args.batch_size,
        patience=args.patience,
        vae_epochs=args.vae_epochs,
        gg_epochs=args.gg_epochs,
        nbr_epochs=args.nbr_epochs,
        abm_steps=args.abm_steps,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
