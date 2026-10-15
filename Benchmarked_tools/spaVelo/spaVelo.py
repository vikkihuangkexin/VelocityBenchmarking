#!/usr/bin/env python3
"""
spaVelo spatial RNA velocity analysis pipeline for VelocityBenchmarking.

spaVelo extends the VeloVAE / VAE framework with a spatial Gaussian process prior:
it denoises spliced/unspliced counts with a spatial GP-driven variational
autoencoder and exports a spatial RNA velocity for every cell. The model needs
both the expression layers and the spatial coordinates.

The pipeline is a statement-for-statement port of the original driver
(``velocity_spatial_code/spavelo_run.py``): the same ``Config`` class and field
values, the same layer / ``obsm`` keys and the same local variable names. Only the
argument parsing, the metadata batch loop and the ``uns`` run record are wrapper
plumbing.

Installation:
    See the repository Dockerfile, which installs the upstream spaVelo source
    (script-based, no packaging) plus the pinned scientific stack:
        git clone https://github.com/1062638515/spavelo.git /opt/spavelo
    spaVelo has no setup.py, so the checkout is imported through ``--repo-dir``
    (or the ``SPAVELO_REPO_DIR`` environment variable).

Usage:
    python spaVelo.py --input spatial.h5ad --output-dir ./output --spatial-key spatial
    python spaVelo.py --metadata-file datasets.csv --output-dir ./output
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# The upstream checkout is a script-only repository; it is located through this
# environment variable so the container default can be overridden without editing
# the script.
DEFAULT_REPO_DIR = os.environ.get("SPAVELO_REPO_DIR", "/opt/spavelo")

DEFAULT_SPLICED_LAYER = "spliced"
DEFAULT_UNSPLICED_LAYER = "unspliced"

# Native spaVelo key for the inferred velocity (spaVeloDenoise.get_velocity).
NATIVE_VELOCITY_KEY = "pred_velocity"


class Config:
    """Configuration container of the original spaVelo driver (values unchanged)."""

    def __init__(self):
        self.data_file = None
        self.spliced_layer = "spliced"
        self.unspliced_layer = "unspliced"
        self.batch_size = 256
        self.maxiter = 5000
        self.train_size = 0.95
        self.patience = 100
        self.lr = 1e-3
        self.weight_decay = 1e-6
        self.dropoutE = 0.0
        self.dropoutD = 0.0
        self.encoder_layers = [256, 128]
        self.GP_dim = 2
        self.Normal_dim = 8
        self.decoder_layers = [128, 256]
        self.dynamicVAE = True
        self.init_beta = 4
        self.min_beta = 1
        self.max_beta = 25
        self.KL_loss = 0.025
        self.KL_weight = 15.0
        self.penalty_scale = 10.0
        self.n_samples = 1
        self.fix_inducing_points = True
        self.grid_inducing_points = True
        self.inducing_point_steps = 15
        self.inducing_point_nums = None
        self.fixed_gp_params = False
        self.loc_range = 20.0
        self.w_latent_loss = 1.0
        self.kernel_scale = 20.0
        self.gene_selection = "spatial"
        self.use_raw = False
        self.no_rho = False
        self.save_dir = None
        self.model_file = "model.pt"
        self.device = "cuda:0"


@lru_cache(maxsize=8)
def load_spavelo_api(repo_dir: str):
    """Import the upstream spaVelo modules from the given checkout directory.

    The repository is script-based, so both the repository root (``spaVelo.py``,
    ``spaVeloDenoise.py``) and its ``experiments/`` folder are added to ``sys.path``
    before the modules are imported.
    """
    repo_dir = str(Path(repo_dir).expanduser().resolve())
    if not Path(repo_dir).is_dir():
        raise FileNotFoundError(f"spaVelo repository directory not found: {repo_dir}")

    for path in (repo_dir, str(Path(repo_dir) / "experiments")):
        if path not in sys.path:
            sys.path.append(path)

    denoise_module = importlib.import_module("spaVeloDenoise")
    preprocess_module = importlib.import_module("preprocess_real")
    return denoise_module.SPAVELODenoise, preprocess_module.preprocess_data_real, preprocess_module.get_init_variables


def seed_everything(seed: int) -> None:
    """Seed the random number generators used by spaVelo."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    cfg: Config,
    spatial_key: str = "spatial",
    spliced_layer: str = DEFAULT_SPLICED_LAYER,
    unspliced_layer: str = DEFAULT_UNSPLICED_LAYER,
    cluster_key: Optional[str] = None,
):
    """Prepare the spatial AnnData and build the GP inducing points.

    The statement sequence, the keys and the ``cfg`` fields are the ones of the
    original ``run_one``: log-normalized highly variable genes, min-max scaled
    coordinates in ``obsm['spatial']`` and the inducing points of the spatial GP.
    """
    import scanpy as sc

    if cluster_key and cluster_key not in adata.obs:
        raise KeyError(f"Missing cell type column: {cluster_key}")
    if spatial_key not in adata.obsm:
        raise KeyError(f"Missing spatial coordinates: {spatial_key}")
    for source, target in ((spliced_layer, "spliced"), (unspliced_layer, "unspliced")):
        if source not in adata.layers:
            raise KeyError(f"Missing layer: {source}")
        if source != target:
            adata.layers[target] = adata.layers[source].copy()
    if spatial_key != "spatial":
        adata.obsm["spatial"] = adata.obsm[spatial_key].copy()
    adata.X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()
    adata.obsm["spatial_raw"] = adata.obsm["spatial"].astype(np.float64)
    scaler = MinMaxScaler()
    adata.obsm["spatial"] = scaler.fit_transform(adata.obsm["spatial"]) * cfg.loc_range

    if cfg.grid_inducing_points:
        eps = 1e-5
        initial_inducing_points = np.mgrid[0:(1+eps):(1./cfg.inducing_point_steps), 0:(1+eps):(1./cfg.inducing_point_steps)].reshape(2, -1).T * cfg.loc_range
        print(initial_inducing_points.shape)
    else:
        loc = adata.obsm["spatial"]
        loc_kmeans = KMeans(n_clusters=cfg.inducing_point_nums, n_init=100).fit(loc)
        np.savetxt(f"{cfg.save_dir}/location_centroids.txt", loc_kmeans.cluster_centers_, delimiter=",")
        np.savetxt(f"{cfg.save_dir}/location_kmeans_labels.txt", loc_kmeans.labels_, delimiter=",", fmt="%i")
        initial_inducing_points = loc_kmeans.cluster_centers_

    return adata, initial_inducing_points


def run_spavelo_analysis(
    input_path,
    output_dir,
    spatial_key: str = "spatial",
    spliced_layer: str = DEFAULT_SPLICED_LAYER,
    unspliced_layer: str = DEFAULT_UNSPLICED_LAYER,
    cluster_key: Optional[str] = None,
    dataset_name: Optional[str] = None,
    repo_dir: str = DEFAULT_REPO_DIR,
    device: str = "cuda:0",
    batch_size: int = 256,
    maxiter: int = 5000,
    patience: int = 100,
    overwrite: bool = False,
    seed: int = 144,
) -> Path:
    """Run the full spaVelo pipeline for a single spatial dataset.

    ``root``, ``folder`` and ``out`` are the ones the original ``run_all`` loop
    builds around ``run_one``::

        name = path.stem
        folder = root / name
        out = folder / f"{name}.h5ad"
    """
    SPAVELODenoise, preprocess_data_real, get_init_variables = load_spavelo_api(repo_dir)

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

    import scanpy as sc

    cfg = Config()
    cfg.data_file = str(path)
    cfg.save_dir = str(folder)
    seed_everything(seed)
    cfg.device = device
    cfg.spliced_layer = "spliced"
    cfg.unspliced_layer = "unspliced"
    cfg.batch_size = batch_size
    cfg.maxiter = maxiter
    cfg.patience = patience
    os.makedirs(cfg.save_dir, exist_ok=True)
    cfg.model_file = str(folder / "model.pt")
    pd.Series(vars(cfg)).to_csv(f"{cfg.save_dir}/args.csv")

    adata = None
    model = None
    try:
        adata = sc.read_h5ad(cfg.data_file)
        adata, initial_inducing_points = preprocess_adata(
            adata,
            cfg,
            spatial_key=spatial_key,
            spliced_layer=spliced_layer,
            unspliced_layer=unspliced_layer,
            cluster_key=cluster_key,
        )

        (
            adata,
            us_upper,
            ms_upper,
            alpha_unconstr,
            gamma_unconstr,
            steady_beta_unconstr,
            steady_gamma_unconstr,
            alpha_1_unconstr,
            lambda_alpha_unconstr,
            prior_prob,
            gamma_r2,
        ) = get_init_variables(
            adata,
            spliced_layer=cfg.spliced_layer,
            unspliced_layer=cfg.unspliced_layer,
            gamma_init_data=False,
            dynamo_init=True,
        )
        adata.layers["prior_prob"] = prior_prob.argmax(axis=2)

        model = SPAVELODenoise(
            n_genes=adata.n_vars,
            true_time_switch=None,
            t_max=20,
            penalty_scale=cfg.penalty_scale,
            log_variational=False,
            GP_dim=cfg.GP_dim,
            Normal_dim=cfg.Normal_dim,
            encoder_layers=cfg.encoder_layers,
            decoder_layers=cfg.decoder_layers,
            encoder_dropout=cfg.dropoutE,
            decoder_dropout=cfg.dropoutD,
            fixed_inducing_points=cfg.fix_inducing_points,
            initial_inducing_points=initial_inducing_points,
            fixed_gp_params=cfg.fixed_gp_params,
            kernel_scale=cfg.kernel_scale,
            N_train=adata.n_obs,
            KL_loss=cfg.KL_loss,
            dynamicVAE=cfg.dynamicVAE,
            init_beta=cfg.init_beta,
            min_beta=cfg.min_beta,
            max_beta=cfg.max_beta,
            KL_weight=cfg.KL_weight,
            gamma_unconstr_init=gamma_unconstr,
            alpha_unconstr_init=alpha_unconstr,
            alpha_1_unconstr_init=alpha_1_unconstr,
            lambda_alpha_unconstr_init=lambda_alpha_unconstr,
            switch_spliced=ms_upper,
            switch_unspliced=us_upper,
            linear_decoder=False,
            time_dep_transcription_rate=False,
            dirichlet_concentration=0.25,
            dtype=torch.float64,
            device=cfg.device,
            w_latent_loss=cfg.w_latent_loss,
            use_raw=cfg.use_raw,
            adata=adata,
            spliced_layer=cfg.spliced_layer,
            unspliced_layer=cfg.unspliced_layer,
            use_rho=(not cfg.no_rho),
        )

        model.train_model(
            adata,
            prior_prob=prior_prob,
            spatial_layer="spatial",
            spliced_layer=cfg.spliced_layer,
            unspliced_layer=cfg.unspliced_layer,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            n_samples=cfg.n_samples,
            train_size=cfg.train_size,
            maxiter=cfg.maxiter,
            patience=cfg.patience,
            save_model=True,
            model_weights=cfg.model_file,
            log_dir=cfg.save_dir,
            gamma_r2=gamma_r2,
        )

        adata.layers["pred_velocity"] = model.get_velocity(
            adata,
            spatial_layer="spatial",
            spliced_layer=cfg.spliced_layer,
            unspliced_layer=cfg.unspliced_layer,
            batch_size=cfg.batch_size,
            n_samples=25,
        )

        # Benchmark contract: the native spaVelo key is kept and the same values are
        # exposed as layers['velocity'] so the benchmark can read the result.
        if NATIVE_VELOCITY_KEY not in adata.layers:
            raise RuntimeError(f"spaVelo did not produce layers['{NATIVE_VELOCITY_KEY}'].")
        adata.layers["velocity"] = np.asarray(adata.layers[NATIVE_VELOCITY_KEY])

        adata.uns["spavelo_run"] = {
            "dataset_name": str(dataset_name),
            "input_path": str(path.resolve()),
            "spatial_key": spatial_key,
            "cluster_key": cluster_key or "",
            "spliced_layer": cfg.spliced_layer,
            "unspliced_layer": cfg.unspliced_layer,
            "device": cfg.device,
            "batch_size": int(cfg.batch_size),
            "maxiter": int(cfg.maxiter),
            "patience": int(cfg.patience),
            "native_velocity_key": NATIVE_VELOCITY_KEY,
            "output_path": str(out.resolve()),
        }

        print(f"Saving results to {out}")
        adata.write_h5ad(str(out))
        return out
    finally:
        del adata
        del model
        cleanup_resources()


def run_batch_spavelo(
    metadata_file,
    output_dir,
    spatial_key: str = "spatial",
    spliced_layer: str = DEFAULT_SPLICED_LAYER,
    unspliced_layer: str = DEFAULT_UNSPLICED_LAYER,
    repo_dir: str = DEFAULT_REPO_DIR,
    device: str = "cuda:0",
    batch_size: int = 256,
    maxiter: int = 5000,
    patience: int = 100,
    overwrite: bool = False,
    seed: int = 144,
) -> list[Path]:
    """Run spaVelo for every dataset listed in a metadata table."""
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
                run_spavelo_analysis(
                    input_path=file_path,
                    output_dir=output_dir,
                    spatial_key=str(row["spatial_key"]) or spatial_key,
                    spliced_layer=spliced_layer,
                    unspliced_layer=unspliced_layer,
                    cluster_key=str(row["cluster_key"]) or None,
                    dataset_name=str(row["dataset_name"]),
                    repo_dir=repo_dir,
                    device=device,
                    batch_size=batch_size,
                    maxiter=maxiter,
                    patience=patience,
                    overwrite=overwrite,
                    seed=seed,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep batch runs going
            print(f"Failed: {row['dataset_name']}: {exc}")

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="spaVelo spatial RNA velocity analysis",
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
        help="Column name in adata.obs used for cell labels (optional for spaVelo)",
    )
    parser.add_argument("--spliced-layer", dest="spliced_layer", default=DEFAULT_SPLICED_LAYER,
                        help="Spliced counts layer")
    parser.add_argument("--unspliced-layer", dest="unspliced_layer", default=DEFAULT_UNSPLICED_LAYER,
                        help="Unspliced counts layer")
    parser.add_argument(
        "--repo-dir",
        dest="repo_dir",
        default=DEFAULT_REPO_DIR,
        help="spaVelo checkout containing spaVeloDenoise.py and experiments/",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device, for example cuda:0 or cpu")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--maxiter", type=int, default=5000, help="Maximum training iterations")
    parser.add_argument("--patience", type=int, default=100, help="Early-stopping patience")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=144, help="Random seed (upstream default)")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if args.metadata_file:
        return run_batch_spavelo(
            metadata_file=args.metadata_file,
            output_dir=args.output_dir,
            spatial_key=args.spatial_key,
            spliced_layer=args.spliced_layer,
            unspliced_layer=args.unspliced_layer,
            repo_dir=args.repo_dir,
            device=args.device,
            batch_size=args.batch_size,
            maxiter=args.maxiter,
            patience=args.patience,
            overwrite=args.overwrite,
            seed=args.seed,
        )

    return run_spavelo_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        spatial_key=args.spatial_key,
        spliced_layer=args.spliced_layer,
        unspliced_layer=args.unspliced_layer,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        repo_dir=args.repo_dir,
        device=args.device,
        batch_size=args.batch_size,
        maxiter=args.maxiter,
        patience=args.patience,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
