"""Run spatial STEER with https://github.com/sd68515/STEER_benchmark."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import random
import sys

import numpy as np
import scanpy as sc
import scvelo as scv
from scipy import sparse
import torch

# Direct execution must import the installed package rather than this file.
_import_path = sys.path[:]
try:
    sys.path = [entry for entry in sys.path
                if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
    import steer
    from steer.prior.prior import PriorInferenceManager
finally:
    sys.path = _import_path

SEED = 618


def ensure_spatial_basis(adata):
    """Provide the X_ coordinate alias required by scVelo's spatial basis."""
    if "X_spatial" not in adata.obsm:
        if "spatial" not in adata.obsm:
            raise KeyError("Missing obsm['spatial'] for scVelo basis='spatial'")
        adata.obsm["X_spatial"] = adata.obsm["spatial"].copy()


def dataset_id(value):
    value = value.strip()
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"Invalid dataset ID: {value!r}")
    return value


def read_rows(args):
    if args.metadata_file:
        with args.metadata_file.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"dataset_name", "file_path", "modality", "cluster_key"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"CSV needs {', '.join(sorted(required))}")
            rows = []
            for source in reader:
                if None in source:
                    raise ValueError("CSV row has more values than its header")
                row = {key: (value or "").strip() for key, value in source.items()}
                if any(not row.get(key) for key in required):
                    raise ValueError("CSV row has an empty required field")
                path = Path(row["file_path"]).expanduser()
                path = path if path.is_absolute() else args.metadata_file.parent / path
                if row["modality"] != "spatial":
                    raise ValueError("STEER CSV modality must be spatial")
                rows.append((dataset_id(row["dataset_name"]), path, row["cluster_key"], row.get("dimred_key") or "spatial"))
    else:
        if not args.annotation_key:
            raise ValueError("--annotation-key is required for one input")
        rows = [(dataset_id(args.dataset_id or args.input.stem), args.input,
                 args.annotation_key, args.coordinate_key)]
    if not rows or len({row[0] for row in rows}) != len(rows):
        raise ValueError("Dataset list is empty or has duplicate IDs")
    return rows


def setup_seed(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(threads)


def device_from_arg(value):
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count()):
            raise ValueError(f"CUDA device is unavailable: {value}")
    elif device.type != "cpu":
        raise ValueError("--device must be cpu, cuda, or cuda:<index>")
    return device


def prepare_input(path, annotation_key, coordinate_key):
    adata = sc.read_h5ad(path)
    if annotation_key not in adata.obs or adata.obs[annotation_key].isna().any():
        raise ValueError(f"Missing or incomplete obs[{annotation_key!r}]")
    if coordinate_key not in adata.obsm:
        raise KeyError(f"Missing obsm[{coordinate_key!r}]")
    coords = np.asarray(adata.obsm[coordinate_key], dtype=np.float64)
    if coords.shape != (adata.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates must be a finite cells-by-2 matrix")
    for key in ("spliced", "unspliced"):
        if key not in adata.layers:
            raise KeyError(f"Missing layers[{key!r}]")
        matrix = adata.layers[key]
        values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        if not np.isfinite(values).all():
            raise ValueError(f"layers[{key!r}] contains non-finite values")
        # Upstream STEER expects sparse count layers in its dataset loader.
        adata.layers[key] = (
            matrix.tocsr().astype(np.float64) if sparse.issparse(matrix)
            else sparse.csr_matrix(matrix, dtype=np.float64)
        )
    duplicates = int(adata.var_names.duplicated().sum())
    adata.var["_source_var_name"] = adata.var_names.astype(str).to_numpy(copy=True)
    if duplicates:
        adata.var_names_make_unique(join="-")
    adata.var_names = adata.var_names.astype(str)
    adata.obsm["spatial"] = coords.copy()
    ensure_spatial_basis(adata)
    return adata, duplicates


def neighbor_count(adata, requested):
    count = min(requested, adata.n_obs - 1)
    if count < 2:
        raise ValueError("STEER needs at least three cells")
    return count


def spatial_preprocess(adata):
    pcs = min(30, adata.n_obs - 1, adata.n_vars - 1)
    if pcs < 2:
        raise ValueError("STEER needs at least three cells and genes after filtering")
    return steer.preprocess_anndata_spatial(
        adata, npc=pcs, NUM_AD_NEIGH=neighbor_count(adata, 30),
        spatial_neighbors=neighbor_count(adata, 8),
        SMOOTH_NEIGH=neighbor_count(adata, 50), moments_adj=True,
        combine_mode="union", neighbor_metric="cosine",
        spatial_key="spatial", use_us=True,
    )


def save_plots(result, annotation_key, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_spatial_basis(result)
    path.mkdir(parents=True, exist_ok=True)
    scv.tl.velocity_embedding(result, basis="spatial", vkey="pred_vs_norm")
    scv.pl.scatter(result, basis="spatial", color=annotation_key, show=False)
    plt.savefig(path / "annotation.png", dpi=300, bbox_inches="tight")
    plt.close()
    scv.pl.velocity_embedding_stream(result, basis="spatial", vkey="pred_vs_norm",
                                     color=annotation_key, show=False)
    plt.savefig(path / "velocity_stream.png", dpi=300, bbox_inches="tight")
    plt.close()
    scv.pl.scatter(result, basis="spatial", color="Pred Time", show=False)
    plt.savefig(path / "pred_time.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_one(row, args, device):
    name, input_path, annotation_key, coordinate_key = row
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    destination = args.output_dir / f"{name}.h5ad"
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    work = args.output_dir / name / "training"
    work.mkdir(parents=True, exist_ok=True)
    setup_seed(SEED, args.threads)

    adata, duplicates = prepare_input(input_path, annotation_key, coordinate_key)
    original_shape = (adata.n_obs, adata.n_vars)
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    frame, adjacency, adata = spatial_preprocess(adata)
    estimate = steer.estimate_expert_number(
        adata, candidate_range=range(2, 11), used_obsm="X_pca_combined",
        random_seed=SEED, plot=True, save_path=str(work),
    )
    experts = int(estimate["recommended_expert"])
    if experts not in range(2, 11):
        raise ValueError("Expert estimator returned a value outside 2-10")
    fine_method = "hierarchical" if experts < 5 else "none"
    direction_base = "fine_cluster" if experts < 5 else "expert"

    dataset = steer.preload_datasets_all_genes_anndata(df=frame, MODEL_MODE="pretrain", adata=adata)
    pyg = steer.create_pyg_data(dataset, adjacency, normalize=True)
    stage1 = steer.model_training_share_neighbor_adata(
        device=device, device2=device, pyg_data=pyg, MODEL_MODE="pretrain",
        adata=adata, NUM_LOSS_NEIGH=neighbor_count(adata, 30),
        corr_mode="u", max_n_cluster=experts, path=str(work),
    )
    if not args.skip_mclust_refine:
        stage1 = steer.mclust_R(stage1, num_cluster=experts,
                                used_obsm="X_pre_embed", random_seed=SEED)
    prior = PriorInferenceManager(stage1, frame, str(work), seed=SEED, n_jobs=args.threads)
    prior.task1_define_fine_clusters(method=fine_method, target_size=300)
    prior.task2_filter_genes(based_on="expert", keep_ngene=1000, use_filter=True)
    prior.task3_calc_convexity(based_on=direction_base, n_jobs=args.threads)
    prior_adata, _, fine_clusters = prior.finalize_for_training()

    fresh, fresh_duplicates = prepare_input(input_path, annotation_key, coordinate_key)
    if fresh_duplicates != duplicates:
        raise ValueError("Duplicate gene handling changed between stages")
    scv.pp.filter_and_normalize(fresh, min_shared_counts=20, n_top_genes=2000)
    if not np.array_equal(prior_adata.obs_names, fresh.obs_names) or not np.array_equal(prior_adata.var_names, fresh.var_names):
        raise ValueError("Cells or genes changed between STEER training stages")
    fresh.layers["pred_cell_type"] = prior_adata.layers["pred_cell_type"]
    fresh.obsm["X_pre_embed"] = prior_adata.obsm["X_pre_embed"]
    fresh.obs["pred_cluster"] = prior_adata.obs["pred_cluster"].astype(int)
    mask = np.asarray(prior_adata.var["is_velocity_gene"], dtype=bool)
    if not mask.any():
        raise ValueError("STEER selected no velocity genes")
    selected = fresh[:, mask].copy()
    fine_frame, fine_adjacency, selected = spatial_preprocess(selected)
    fine_dataset = steer.preload_datasets_all_genes_anndata(
        df=fine_frame, MODEL_MODE="whole", adata=selected,
    )
    fine_pyg = steer.create_pyg_data(fine_dataset, fine_adjacency, normalize=True)
    fine_pyg.fine_clus_vec = torch.tensor(fine_clusters, dtype=torch.long, device=device)
    result = steer.model_training_share_neighbor_adata(
        device=device, device2=device, pyg_data=fine_pyg, MODEL_MODE="whole",
        adata=selected, NUM_LOSS_NEIGH=neighbor_count(selected, 30),
        max_n_cluster=experts, corr_mode="u", path=str(work),
    )
    if "X_refine_embed_t" not in result.obsm:
        raise KeyError("STEER output lacks X_refine_embed_t")
    sc.pp.neighbors(result, use_rep="X_refine_embed_t", n_neighbors=neighbor_count(result, 30))
    steer.velocity_graph(result, vkey="pred_vs_norm", xkey="model_Ms", n_jobs=args.threads)
    for key in ("pred_vs_norm", "model_Ms"):
        if key not in result.layers:
            raise KeyError(f"STEER output lacks layers[{key!r}]")
    if "Pred Time" not in result.obs:
        raise KeyError("STEER output lacks obs['Pred Time']")
    if annotation_key not in result.obs:
        raise KeyError(f"STEER output lacks obs[{annotation_key!r}]")
    velocity = result.layers["pred_vs_norm"]
    values = velocity.data if sparse.issparse(velocity) else np.asarray(velocity)
    if not np.isfinite(values).all() or not np.isfinite(result.obs["Pred Time"].to_numpy(dtype=float)).all():
        raise ValueError("STEER output has non-finite velocity or time")
    if not np.array_equal(result.obs_names, selected.obs_names):
        raise ValueError("STEER output changed cell order")
    if "spatial" not in result.obsm:
        raise KeyError("STEER output lacks obsm['spatial']")
    ensure_spatial_basis(result)
    if coordinate_key != "spatial":
        result.obsm[coordinate_key] = np.asarray(result.obsm["spatial"]).copy()
    for frame in (result.obs, result.var):
        names = [str(key).replace("/", "_") for key in frame.columns]
        if len(set(names)) != len(names):
            raise ValueError("Column sanitization would create duplicate names")
        frame.columns = names
    result.uns["steer_run"] = {
        "dataset_id": name, "annotation_key": annotation_key.replace("/", "_"),
        "coordinate_key": coordinate_key, "seed": SEED,
        "device": str(device), "threads": args.threads,
        "experts": experts, "fine_method": fine_method,
        "direction_base": direction_base, "mclust_refine": not args.skip_mclust_refine,
        "velocity_gene_count": int(mask.sum()),
        "original_shape": list(original_shape), "output_shape": [result.n_obs, result.n_vars],
        "duplicate_gene_names_made_unique": duplicates,
        "graph_combine_mode": "union", "neighbor_metric": "cosine",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(destination)
    if args.plot_dir:
        save_plots(result, annotation_key.replace("/", "_"), args.plot_dir / name)
    print(f"{name}: {result.n_obs} cells, {result.n_vars} genes, {experts} experts -> {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path)
    inputs.add_argument("--metadata-file", type=Path)
    parser.add_argument("--dataset-id")
    parser.add_argument("--annotation-key")
    parser.add_argument("--coordinate-key", default="spatial")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--skip-mclust-refine", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    device = device_from_arg(args.device)
    for row in read_rows(args):
        run_one(row, args, device)


if __name__ == "__main__":
    main()
