"""Run spatial spaVelo with https://github.com/sd68515/spavelo_benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys
import time

import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
import torch

# spaVeloDenoise imports spaVelo; resolve it from the upstream checkout.
_import_path = sys.path[:]
try:
    sys.path = [entry for entry in sys.path
                if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
    from preprocess_real import get_init_variables, preprocess_data_real
    from spaVeloDenoise import SPAVELODenoise
finally:
    sys.path = _import_path


def ensure_spatial_basis(adata) -> None:
    """Provide the X_ coordinate alias required by scVelo's spatial basis."""
    if "X_spatial" not in adata.obsm:
        if "spatial" not in adata.obsm:
            raise KeyError("Missing obsm['spatial'] for scVelo basis='spatial'")
        adata.obsm["X_spatial"] = adata.obsm["spatial"].copy()


def validate_dataset_id(value: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"Dataset ID must be one path component: {value!r}")
    return value


def read_manifest(path: Path) -> list[dict]:
    required = ("dataset_name", "file_path", "modality", "cluster_key")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(key not in reader.fieldnames for key in required):
            raise ValueError(f"CSV requires columns: {', '.join(required)}")
        rows, seen = [], set()
        for number, source in enumerate(reader, start=2):
            if None in source:
                raise ValueError(f"CSV row {number} has more values than its header")
            row = {key: (value or "").strip() for key, value in source.items()}
            if any(not row[key] for key in required):
                raise ValueError(f"CSV row {number} is missing a required value")
            if row["modality"] != "spatial":
                raise ValueError(f"CSV row {number}: modality must be spatial")
            row["dataset_id"] = validate_dataset_id(row["dataset_name"])
            row["coordinate_key"] = row.get("dimred_key") or "spatial"
            if row["dataset_id"] in seen:
                raise ValueError(f"Duplicate dataset ID: {row['dataset_id']}")
            seen.add(row["dataset_id"])
            input_path = Path(row["file_path"])
            row["file_path"] = input_path if input_path.is_absolute() else path.parent / input_path
            rows.append(row)
    if not rows:
        raise ValueError("CSV contains no datasets")
    return rows


def finite(name: str, matrix) -> None:
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


def seed_training(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(threads)


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return requested
    if not torch.cuda.is_available():
        raise ValueError(f"CUDA is unavailable: {requested}")
    device = torch.device(requested)
    if device.type != "cuda" or (device.index is not None and device.index >= torch.cuda.device_count()):
        raise ValueError(f"Unavailable CUDA device: {requested}")
    return requested


def sanitize_columns(adata) -> None:
    for axis in ("obs", "var"):
        frame = getattr(adata, axis)
        mapping = {key: str(key).replace("/", "_") for key in frame.columns if "/" in str(key)}
        if mapping:
            final = [mapping.get(key, key) for key in frame.columns]
            if len(set(final)) != len(final):
                raise ValueError(f"{axis} column sanitization would create duplicate names")
            frame.rename(columns=mapping, inplace=True)


def run_one(row: dict, output_dir: Path, args, device: str) -> Path:
    name = row["dataset_id"]
    run_dir = output_dir / name
    destination = run_dir / "adata_out.h5ad"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already contains files: {run_dir}")
    adata = sc.read_h5ad(row["file_path"])
    cluster_key, coordinate_key = row["cluster_key"], row["coordinate_key"]
    if cluster_key not in adata.obs or adata.obs[cluster_key].isna().any():
        raise ValueError(f"{name}: missing or incomplete obs[{cluster_key!r}]")
    if coordinate_key not in adata.obsm:
        raise ValueError(f"{name}: missing obsm[{coordinate_key!r}]")
    if any(key not in adata.layers for key in ("spliced", "unspliced")):
        raise ValueError(f"{name}: spliced and unspliced layers are required")
    cells = adata.obs_names.copy()
    original_coords = np.asarray(adata.obsm[coordinate_key]).copy()
    if original_coords.shape != (adata.n_obs, 2):
        raise ValueError(f"{name}: coordinates must have shape (n_cells, 2)")
    finite("coordinates", original_coords)
    for key in ("spliced", "unspliced"):
        finite(key, adata.layers[key])
    original_umap = np.asarray(adata.obsm["X_umap"]).copy() if "X_umap" in adata.obsm else None
    if original_umap is not None and original_umap.shape[0] != adata.n_obs:
        raise ValueError(f"{name}: X_umap cell count does not match input")
    seed_training(args.seed, args.threads)
    run_dir.mkdir(parents=True, exist_ok=True)

    adata.obsm["spatial"] = original_coords.copy()
    ensure_spatial_basis(adata)
    scaled = MinMaxScaler().fit_transform(original_coords) * 20.0
    adata.obsm["spatial_model"] = scaled
    eps = 1e-5
    inducing = np.mgrid[0:(1 + eps):(1 / 15), 0:(1 + eps):(1 / 15)].reshape(2, -1).T * 20.0
    adata = preprocess_data_real(
        adata, spatial_layer="spatial_model", spliced_layer="spliced",
        unspliced_layer="unspliced", dtype="float64", log_transform=True,
        min_max_scale=True, filter_on_r2=False, filter_on_ss=False,
        gene_selection="spatial", use_raw=False, smooth_pca=False,
        compute_umap=False, moran_jobs=args.threads)
    if adata.n_vars == 0 or not adata.obs_names.equals(cells):
        raise ValueError(f"{name}: preprocessing removed all genes or changed cell order")
    if original_umap is None:
        adata.obsm.pop("X_umap", None)
    else:
        adata.obsm["X_umap"] = original_umap

    (adata, us_upper, ms_upper, alpha, gamma, steady_beta,
     steady_gamma, alpha_1, lambda_alpha, prior_prob, gamma_r2) = get_init_variables(
        adata, spliced_layer="spliced", unspliced_layer="unspliced",
        gamma_init_data=False, dynamo_init=True)
    if adata.n_vars == 0 or not adata.obs_names.equals(cells):
        raise ValueError(f"{name}: initialization removed all genes or changed cell order")
    ensure_spatial_basis(adata)
    model = SPAVELODenoise(
        n_genes=adata.n_vars, true_time_switch=None, t_max=20,
        penalty_scale=10.0, log_variational=False, GP_dim=2, Normal_dim=8,
        encoder_layers=[256, 128], decoder_layers=[128, 256],
        encoder_dropout=0.0, decoder_dropout=0.0, fixed_inducing_points=True,
        initial_inducing_points=inducing, fixed_gp_params=False,
        kernel_scale=20.0, N_train=adata.n_obs, KL_loss=0.025,
        dynamicVAE=True, init_beta=4, min_beta=1, max_beta=25,
        KL_weight=15.0, gamma_unconstr_init=gamma,
        alpha_unconstr_init=alpha, alpha_1_unconstr_init=alpha_1,
        lambda_alpha_unconstr_init=lambda_alpha, switch_spliced=ms_upper,
        switch_unspliced=us_upper, linear_decoder=False,
        time_dep_transcription_rate=False, dirichlet_concentration=0.25,
        dtype=torch.float64, device=device, w_latent_loss=1.0, use_raw=False,
        adata=adata, spliced_layer="spliced", unspliced_layer="unspliced",
        use_rho=True)
    checkpoint = run_dir / "model.pt"
    started = time.monotonic()

    def report_progress(epoch: int, total: int) -> None:
        if epoch == 1 or epoch % args.progress_every == 0:
            hours = (time.monotonic() - started) / 3600
            print(f"{name}: epoch {epoch}/{total}, elapsed {hours:.2f} h", flush=True)

    model.train_model(
        adata, prior_prob=prior_prob, spatial_layer="spatial_model",
        spliced_layer="spliced", unspliced_layer="unspliced",
        lr=1e-3, weight_decay=1e-6, batch_size=args.batch_size, n_samples=1,
        train_size=0.95, maxiter=args.maxiter, patience=args.patience,
        save_model=True, model_weights=str(checkpoint), log_dir=str(run_dir),
        gamma_r2=gamma_r2, verbose=False,
        progress_callback=report_progress, recorder_save_every=50)
    if not checkpoint.is_file():
        raise RuntimeError(f"{name}: training did not write model.pt")
    adata.layers["pred_velocity"] = model.get_velocity(
        adata, spatial_layer="spatial_model", spliced_layer="spliced",
        unspliced_layer="unspliced", batch_size=args.batch_size, n_samples=25)
    if "Ms" not in adata.layers or adata.layers["pred_velocity"].shape != adata.shape:
        raise ValueError(f"{name}: missing Ms or invalid pred_velocity dimensions")
    finite("Ms", adata.layers["Ms"])
    finite("pred_velocity", adata.layers["pred_velocity"])
    adata.obsm[coordinate_key] = original_coords
    adata.obsm["spatial"] = original_coords.copy()
    ensure_spatial_basis(adata)
    del adata.obsm["spatial_model"]
    if original_umap is not None:
        adata.obsm["X_umap"] = original_umap
    sanitize_columns(adata)
    adata.uns["spavelo_run"] = {
        "dataset_id": name, "cluster_key": cluster_key.replace("/", "_"),
        "coordinate_key": coordinate_key, "seed": args.seed,
        "device": device, "threads": args.threads,
        "gene_selection": "spatial", "external_hvg": False,
        "selected_genes": adata.n_vars, "maxiter": args.maxiter,
        "patience": args.patience, "batch_size": args.batch_size,
        "velocity_samples": 25, "moran_jobs": args.threads,
        "compute_umap": False, "recorder_save_every": 50,
    }
    adata.write_h5ad(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="One H5AD input")
    inputs.add_argument("--metadata-file", type=Path, help="CSV with explicit annotation and coordinate keys")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", help="Output ID; defaults to the input filename stem")
    parser.add_argument("--cluster-key", help="Existing adata.obs annotation key")
    parser.add_argument("--coordinate-key", default="spatial", help="Existing adata.obsm coordinate key (default: spatial)")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<index>")
    parser.add_argument("--seed", type=int, default=144)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.maxiter, args.patience, args.batch_size,
           args.progress_every) < 1:
        parser.error("Training counts must be positive")
    rows = read_manifest(args.metadata_file) if args.metadata_file else None
    if rows is None:
        if not args.cluster_key:
            parser.error("--cluster-key is required with --input")
        rows = [{"dataset_id": validate_dataset_id(args.dataset_id or args.input.stem),
                 "file_path": args.input, "cluster_key": args.cluster_key,
                 "coordinate_key": args.coordinate_key}]
    device = resolve_device(args.device)
    for row in rows:
        print(run_one(row, args.output_dir, args, device), flush=True)


if __name__ == "__main__":
    main()
