"""Run scRNA or spatial TopoVelo with https://github.com/sd68515/TopoVelo_benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import scanpy as sc
import scvelo as scv
import torch

# Direct execution must import the installed package rather than this file.
_import_path = sys.path[:]
try:
    sys.path = [entry for entry in sys.path
                if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
    import topovelo as tpv
finally:
    sys.path = _import_path


PALETTE_30_3 = [
    "#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4",
    "#66c2a5", "#3288bd", "#abdda4", "#e6f598", "#fee08b",
    "#f46d43", "#e7298a", "#a6cee3", "#1f78b4", "#b2df8a",
    "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00",
    "#cab2d6", "#6a3d9a", "#ffff99", "#b15928", "#8dd3c7",
    "#bc80bd", "#ccebc5", "#ffed6f", "#999999",
]


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


def parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected true or false, got {value!r}")


def read_manifest(path: Path) -> list[dict]:
    required = ("dataset_name", "file_path", "modality", "cluster_key", "dimred_key")
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
            row["dataset_id"] = validate_dataset_id(row["dataset_name"])
            row["coordinate_key"] = row["dimred_key"]
            if row["dataset_id"] in seen:
                raise ValueError(f"Duplicate dataset ID: {row['dataset_id']}")
            seen.add(row["dataset_id"])
            if row["modality"] not in {"scrna", "spatial"}:
                raise ValueError(f"CSV row {number}: modality must be scrna or spatial")
            input_path = Path(row["file_path"])
            row["file_path"] = input_path if input_path.is_absolute() else path.parent / input_path
            rows.append(row)
    if not rows:
        raise ValueError("CSV contains no datasets")
    return rows


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if not torch.cuda.is_available():
            return "cpu"
        free = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
        return f"cuda:{int(np.argmax(free))}"
    if requested == "cpu":
        return requested
    if not requested.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError(f"Unavailable training device: {requested}")
    device = torch.device(requested)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"Unavailable training device: {requested}")
    return requested


def normalize_gat_time(values) -> np.ndarray | None:
    """Min-max scale finite model time, or report an indistinguishable range."""
    time = np.asarray(values, dtype=np.float64)
    if time.ndim != 1 or time.size < 2 or not np.isfinite(time).all():
        raise ValueError("gat_time must be a finite vector with at least two cells")
    lower, upper = float(time.min()), float(time.max())
    if np.isclose(lower, upper):
        return None
    return (time - lower) / (upper - lower)


def store_model_time(adata) -> bool:
    if "gat_time" not in adata.obs:
        raise ValueError("TopoVelo did not produce obs['gat_time']")
    normalized = normalize_gat_time(adata.obs["gat_time"].to_numpy())
    if normalized is None:
        if "gat_time_normalized" in adata.obs:
            del adata.obs["gat_time_normalized"]
        return False
    adata.obs["gat_time_normalized"] = normalized
    return True


def compute_scvelo_pseudotime(adata, n_jobs: int | None = None) -> None:
    for layer in ("gat_velocity", "Ms"):
        if layer not in adata.layers:
            raise ValueError(f"TopoVelo did not produce layers[{layer!r}]")
    if n_jobs is None:
        scv.tl.velocity_graph(adata, vkey="gat_velocity")
    else:
        scv.tl.velocity_graph(adata, vkey="gat_velocity", xkey="Ms",
                              n_jobs=n_jobs, backend="threading")
    if "gat_velocity_pseudotime" in adata.obs:
        del adata.obs["gat_velocity_pseudotime"]
    scv.tl.velocity_pseudotime(adata, vkey="gat_velocity")
    key = "gat_velocity_pseudotime"
    if key not in adata.obs:
        raise ValueError(f"scVelo did not produce obs[{key!r}]")
    values = np.asarray(adata.obs[key], dtype=float)
    if values.shape != (adata.n_obs,) or not np.isfinite(values).all() or np.ptp(values) <= 0:
        raise ValueError("scVelo pseudotime is non-finite or constant")


def spatial_hidden_size(n_genes: int) -> tuple[int, int, int]:
    if n_genes <= 300:
        width = 50
    elif n_genes <= 800:
        width = 100
    elif n_genes <= 1500:
        width = 200
    else:
        width = 300
    return width, width // 2, width


def spatial_batch_size(n_cells: int, override: int | None) -> int:
    if override is not None:
        if override < 1:
            raise ValueError("Spatial batch size must be positive")
        return override
    return 128 if n_cells < 2000 else 256 if n_cells <= 5000 else 512


def validate_input(adata, name: str, cluster_key: str, coordinate_key: str) -> np.ndarray:
    if cluster_key not in adata.obs or adata.obs[cluster_key].isna().any():
        raise ValueError(f"{name}: missing or incomplete obs[{cluster_key!r}]")
    if coordinate_key not in adata.obsm:
        raise ValueError(f"{name}: missing obsm[{coordinate_key!r}]")
    if any(layer not in adata.layers for layer in ("spliced", "unspliced")):
        raise ValueError(f"{name}: spliced and unspliced layers are required")
    coordinates = np.asarray(adata.obsm[coordinate_key])
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"{name}: obsm[{coordinate_key!r}] must contain finite 2D coordinates")
    return coordinates.copy()


def plot_scrna(adata, run_dir: Path, name: str, cluster_key: str) -> None:
    ensure_spatial_basis(adata)
    for fmt in ("png", "pdf"):
        common = dict(basis="spatial", size=100, alpha=0.6, color=cluster_key,
                      legend_fontsize=9, legend_loc="right margin", fontsize=None,
                      dpi=400, palette=PALETTE_30_3, vkey="gat_velocity", title="TopoVelo")
        scv.pl.velocity_embedding_stream(
            adata, **common, arrow_size=0.00001, linewidth=0,
            save=str(run_dir / f"TopoVelo_{name}_umap.{fmt}"))
        scv.pl.velocity_embedding_stream(
            adata, **common, density=2, arrow_size=1, linewidth=1,
            save=str(run_dir / f"TopoVelo_{name}_stream.{fmt}"))
        scv.pl.velocity_embedding_grid(
            adata, **common, density=0.8, arrow_size=1, linewidth=0.3,
            save=str(run_dir / f"TopoVelo_{name}_grid.{fmt}"))
        scv.pl.scatter(
            adata, basis="spatial", color="gat_velocity_pseudotime",
            cmap="gnuplot", size=100, dpi=400, figsize=(8, 6), title="TopoVelo",
            save=str(run_dir / f"TopoVelo_{name}_pseudotime.{fmt}"))


def run_scrna(adata, run_dir: Path, name: str, cluster_key: str, args, device: str) -> dict:
    ensure_spatial_basis(adata)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    tpv.preprocess(
        adata, n_gene=args.prep_n_gene, spatial_key="X_spatial",
        min_count_per_cell=args.prep_min_count_per_cell,
        min_genes_expressed=args.prep_min_genes_expressed,
        compute_umap=args.prep_compute_umap)
    tpv.build_spatial_graph(
        adata, spatial_key="X_spatial", graph_key="spatial_graph",
        method=args.graph_method, radius=args.graph_radius)
    vae = tpv.VAE(
        adata, tmax=args.vae_tmax, dim_z=args.vae_dim_z,
        hidden_size=tuple(args.vae_hidden_size), device=device,
        graph_decoder=args.vae_graph_decoder, attention=args.vae_attention)
    training = {"learning_rate": args.lr, "learning_rate_ode": args.lr_ode,
                "learning_rate_refine": args.lr_refine}
    vae.train(adata, adata.obsp["spatial_graph"], "X_spatial",
              config=training, figure_path=str(figure_dir))
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    vae.save_model(str(model_dir), "encoder", "decoder")
    vae.save_anndata(adata, "gat", str(run_dir), file_name="adata_out.h5ad")
    ensure_spatial_basis(adata)
    figsize = tpv.compute_figsize(adata.obsm["X_spatial"], real_aspect_ratio=True, width=4)
    tpv.post_analysis(
        adata, test_id="topovelo", methods=["TopoVelo (GAT)"], keys=["gat"],
        spatial_graph_key="spatial_graph",
        spatial_velocity_graph=args.post_spatial_velocity_graph,
        n_spatial_neighbors=args.post_n_spatial_neighbors,
        spatial_key="X_spatial", compute_metrics=args.post_compute_metrics,
        embed="spatial",
        cluster_plot_config={"figsize": figsize, "real_aspect_ratio": True},
        phase_plot_config={"width": 4}, gene_plot_config={"width": 4},
        time_plot_config={"width": 4, "color_map": "viridis"},
        stream_plot_config={"width": figsize[0], "height": figsize[1],
                            "markersize": 100, "density": 2, "linewidth": 2},
        cluster_key=cluster_key, plot_type=[cluster_key])
    store_model_time(adata)
    compute_scvelo_pseudotime(adata)
    plot_scrna(adata, run_dir, name, cluster_key)
    return {"n_gene_requested": args.prep_n_gene, "graph_method": args.graph_method,
            "graph_radius": args.graph_radius, "hidden_size": list(args.vae_hidden_size),
            "batch_size": "TopoVelo default", **training,
            "post_analysis": True, "post_n_spatial_neighbors": args.post_n_spatial_neighbors,
            "post_spatial_velocity_graph": args.post_spatial_velocity_graph,
            "post_compute_metrics": args.post_compute_metrics,
            "pseudotime_source": "scvelo"}


def run_spatial(adata, run_dir: Path, cluster_key: str, args, device: str) -> dict:
    ensure_spatial_basis(adata)
    original_cells = adata.n_obs
    tpv.preprocess(
        adata, n_gene=min(2000, adata.n_vars), cluster_key=cluster_key,
        spatial_key="X_spatial", min_count_per_cell=1,
        min_genes_expressed=1, compute_umap=False)
    if adata.n_obs != original_cells:
        raise ValueError(f"TopoVelo preprocessing removed {original_cells - adata.n_obs} cells")
    if adata.n_vars <= 5:
        raise ValueError("Too few genes remain for spatial TopoVelo")
    neighbors = min(16, adata.n_obs - 1)
    if neighbors < 1:
        raise ValueError("Spatial TopoVelo requires at least two cells")
    tpv.build_spatial_graph(
        adata, spatial_key="X_spatial", graph_key="spatial_graph",
        method="KNN", n_neighbors=neighbors)
    graph = adata.obsp["spatial_graph"].tocsr().copy()
    graph.setdiag(0)
    graph.eliminate_zeros()
    if graph.shape != (adata.n_obs, adata.n_obs) or graph.nnz == 0 or np.any(np.diff(graph.indptr) == 0):
        raise ValueError("Spatial graph has a cell without a nonself neighbor")
    sizes = spatial_hidden_size(adata.n_vars)
    batch_size = spatial_batch_size(adata.n_obs, args.spatial_batch_size)
    vae = tpv.VAE(
        adata, tmax=20, dim_z=5, hidden_size=sizes, device=device,
        graph_decoder=True, attention=True)
    batches_per_epoch = max(1, (len(vae.train_idx) + batch_size - 1) // batch_size)
    training = {"learning_rate": 2e-4, "learning_rate_ode": 5e-3,
                "learning_rate_refine": 2e-4, "batch_size": batch_size,
                "test_iter": 2 * batches_per_epoch}
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    ensure_spatial_basis(adata)
    vae.train(
        adata, adata.obsp["spatial_graph"], "X_spatial", config=training,
        cluster_key=cluster_key, embed="spatial", plot=False,
        figure_path=str(model_dir))
    vae.save_model(str(model_dir), "encoder", "decoder")
    vae.save_anndata(adata, "gat", str(run_dir), file_name="adata_out.h5ad")
    if store_model_time(adata):
        if "gat_velocity_pseudotime" in adata.obs:
            del adata.obs["gat_velocity_pseudotime"]
        source = "gat_time"
    else:
        compute_scvelo_pseudotime(adata, n_jobs=args.spatial_n_jobs)
        source = "scvelo"
    return {"n_gene_max": 2000, "n_gene_retained": adata.n_vars,
            "graph_method": "KNN", "graph_neighbors": neighbors,
            "hidden_size": list(sizes), "dim_z": 5,
            "post_analysis": False, "pseudotime_source": source, **training}


def run_one(row: dict, output_dir: Path, args, device: str) -> Path:
    name, modality = row["dataset_id"], row["modality"]
    run_dir = output_dir / name
    destination = run_dir / "adata_out.h5ad"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already contains files: {run_dir}")
    adata = sc.read_h5ad(row["file_path"])
    coordinate_key, cluster_key = row["coordinate_key"], row["cluster_key"]
    coordinates = validate_input(adata, name, cluster_key, coordinate_key)
    adata.obsm["X_spatial"] = coordinates.copy()
    adata.obsm["spatial"] = coordinates.copy()
    run_dir.mkdir(parents=True, exist_ok=True)
    if modality == "scrna":
        details = run_scrna(adata, run_dir, name, cluster_key, args, device)
    else:
        details = run_spatial(adata, run_dir, cluster_key, args, device)
    if "gat_velocity" not in adata.layers or adata.layers["gat_velocity"].shape != adata.shape:
        raise ValueError("TopoVelo did not produce an aligned gat_velocity layer")
    adata.uns["topovelo_run"] = {
        "dataset_id": name, "modality": modality, "cluster_key": cluster_key,
        "coordinate_key": coordinate_key, "device": device, **details}
    # save_anndata writes an intermediate result; save again after all analysis.
    adata.write_h5ad(destination, compression="lzf")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="One H5AD input")
    inputs.add_argument("--metadata-file", type=Path, help="CSV with explicit annotation and coordinate keys")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modality", choices=("scrna", "spatial"), help="Required with --input")
    parser.add_argument("--dataset-id", help="Output ID; defaults to the input filename stem")
    parser.add_argument("--cluster-key", help="Existing adata.obs annotation key")
    parser.add_argument("--coordinate-key", help="Existing adata.obsm coordinate key")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--spatial-batch-size", type=int)
    parser.add_argument("--spatial-n-jobs", type=int, default=2)
    parser.add_argument("--prep-n-gene", type=int, default=2000)
    parser.add_argument("--prep-min-count-per-cell", type=int, default=1)
    parser.add_argument("--prep-min-genes-expressed", type=int, default=1)
    parser.add_argument("--prep-compute-umap", action="store_true")
    parser.add_argument("--graph-method", choices=("KNN", "radius"), default="KNN")
    parser.add_argument("--graph-radius", type=int, default=30)
    parser.add_argument("--vae-tmax", type=int, default=20)
    parser.add_argument("--vae-dim-z", type=int, default=5)
    parser.add_argument("--vae-hidden-size", type=int, nargs="+", default=[50, 25, 50])
    parser.add_argument("--vae-graph-decoder", type=parse_boolean, default=True)
    parser.add_argument("--vae-attention", type=parse_boolean, default=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-ode", type=float, default=5e-3)
    parser.add_argument("--lr-refine", type=float, default=2e-4)
    parser.add_argument("--post-n-spatial-neighbors", type=int, default=50)
    parser.add_argument("--post-spatial-velocity-graph", type=parse_boolean, default=True)
    parser.add_argument("--post-compute-metrics", type=parse_boolean, default=False)
    args = parser.parse_args()
    if args.spatial_n_jobs < 1:
        parser.error("--spatial-n-jobs must be positive")
    if args.metadata_file:
        rows = read_manifest(args.metadata_file)
    else:
        if not args.modality or not args.cluster_key or not args.coordinate_key:
            parser.error("--modality, --cluster-key, and --coordinate-key are required with --input")
        rows = [{"dataset_id": validate_dataset_id(args.dataset_id or args.input.stem),
                 "file_path": args.input, "modality": args.modality,
                 "cluster_key": args.cluster_key, "coordinate_key": args.coordinate_key}]
    device = resolve_device(args.device)
    for row in rows:
        print(run_one(row, args.output_dir, args, device), flush=True)


if __name__ == "__main__":
    main()
