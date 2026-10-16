"""Run PhyloVelo on scRNA or lineage inputs with the benchmark fork.

The fork provides a forward 2D projection and an embedding-derived graph.
PhyloVelo estimates one slope per gene, not a cell-by-gene velocity matrix.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

# Direct execution must import the installed package rather than this file.
_import_path = sys.path[:]
try:
    sys.path = [entry for entry in sys.path
                if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
    import phylovelo as pv
    from phylovelo.benchmark import add_embedding_velocity_graph
finally:
    sys.path = _import_path


def h5_array(node):
    if h5py.check_string_dtype(node.dtype) is not None:
        return np.asarray(node.asstr()[()])
    return np.asarray(node[()])


def h5_column(node):
    if isinstance(node, h5py.Dataset):
        return h5_array(node)
    encoding = node.attrs.get("encoding-type")
    if encoding == "categorical":
        return pd.Categorical.from_codes(
            np.asarray(node["codes"][()], dtype=int), h5_array(node["categories"]),
            ordered=bool(node.attrs.get("ordered", False)),
        )
    if encoding == "nullable-integer":
        return pd.arrays.IntegerArray(h5_array(node["values"]), h5_array(node["mask"]))
    if encoding == "nullable-boolean":
        return pd.arrays.BooleanArray(h5_array(node["values"]), h5_array(node["mask"]))
    raise ValueError(f"Unsupported H5AD column at {node.name}: {encoding!r}")


def h5_frame(node):
    if node.attrs.get("encoding-type") != "dataframe":
        raise ValueError(f"Expected an H5AD dataframe: {node.name}")
    index_key = node.attrs["_index"]
    if isinstance(index_key, bytes):
        index_key = index_key.decode("utf-8")
    columns = [key.decode("utf-8") if isinstance(key, bytes) else str(key)
               for key in node.attrs["column-order"]]
    return pd.DataFrame(
        {key: h5_column(node[key]) for key in columns},
        index=pd.Index(h5_column(node[index_key]).astype(str)),
    )


def h5_matrix(node):
    if isinstance(node, h5py.Dataset):
        return h5_array(node)
    encoding = node.attrs.get("encoding-type")
    if encoding not in ("csr_matrix", "csc_matrix"):
        raise ValueError(f"Unsupported H5AD matrix: {node.name}")
    matrix_type = sparse.csr_matrix if encoding == "csr_matrix" else sparse.csc_matrix
    parts = node["data"][()], node["indices"][()], node["indptr"][()]
    return matrix_type(parts, shape=tuple(node.attrs["shape"])).tocsr()


def read_input(row):
    """Read required fields directly, including modern H5AD with legacy AnnData."""
    with h5py.File(row["file_path"], "r") as handle:
        obs, var = h5_frame(handle["obs"]), h5_frame(handle["var"])
        layer = row["count_layer"]
        if layer == "X":
            counts = h5_matrix(handle["X"])
        else:
            if layer not in handle["layers"]:
                raise KeyError(f"Missing layers[{layer!r}]")
            counts = h5_matrix(handle["layers"][layer])
        obsm = {}
        keys = {row["coordinate_key"], "X_pca", row["clone_key"]} - {""}
        for key in keys:
            if key in handle["obsm"]:
                obsm[key] = h5_matrix(handle["obsm"][key])
    if row["coordinate_key"] not in obsm:
        raise KeyError(f"Missing obsm[{row['coordinate_key']!r}]")
    if row["clock_mode"] == "clone_size" and row["clone_key"] not in obsm:
        raise KeyError(f"Missing obsm[{row['clone_key']!r}]")
    if counts.shape != (len(obs), len(var)):
        raise ValueError("Count matrix shape disagrees with obs and var")
    if not obs.index.is_unique or not var.index.is_unique:
        raise ValueError("Cell and gene names must be unique")
    values = counts.data if sparse.issparse(counts) else np.asarray(counts).ravel()
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError("PhyloVelo needs finite nonnegative integer counts")
    positive = np.asarray(counts.sum(axis=1)).ravel() > 0
    if not positive.any():
        raise ValueError("All cells have zero count library size")
    removed = int((~positive).sum())
    if removed:
        obs = obs.iloc[positive].copy()
        counts = counts[positive].copy()
        obsm = {key: matrix[positive].copy() for key, matrix in obsm.items()}
    return obs, var, obsm, counts, removed


def path_clock(labels, paths):
    if not paths:
        raise ValueError("Path clock needs --lineage-path or CSV lineage_paths")
    positions = {}
    for path in paths:
        steps = [part.strip() for part in path.split(",")]
        if len(steps) < 2 or any(not part for part in steps):
            raise ValueError("Each lineage path needs at least two comma-separated labels")
        for position, label in enumerate(steps):
            if label in positions and positions[label] != position:
                raise ValueError(f"Conflicting path positions for {label!r}")
            positions[label] = position
    if labels.isna().any():
        raise ValueError("Annotation contains missing values")
    clock = labels.astype(str).map(positions)
    if clock.isna().any():
        raise ValueError(f"Unmapped labels: {sorted(set(labels[clock.isna()].astype(str)))}")
    return clock.to_numpy(dtype=float)


def get_clock(obs, obsm, row):
    train = np.ones(len(obs), dtype=bool)
    mode = row["clock_mode"]
    if mode == "path":
        key = row["annotation_key"]
        if key not in obs:
            raise KeyError(f"Missing obs[{key!r}]")
        clock = path_clock(obs[key], row["lineage_paths"])
        description = "annotation positions in supplied lineage paths"
    elif mode == "obs":
        key = row["clock_key"]
        if key not in obs:
            raise KeyError(f"Missing obs[{key!r}]")
        clock = pd.to_numeric(obs[key], errors="raise").to_numpy(dtype=float)
        description = f"obs[{key!r}]"
    else:
        key = row["clone_key"]
        clones = obsm[key]
        clones = sparse.csr_matrix(clones, copy=True)
        clones.sum_duplicates()
        clones.eliminate_zeros()
        if not np.all(clones.data == 1) or not np.all(np.diff(clones.indptr) == 1):
            raise ValueError("Clone-size clock needs exactly one one-hot membership per cell")
        sizes = np.bincount(clones.indices, minlength=clones.shape[1])[clones.indices]
        train = (sizes >= row["clone_min"]) & (sizes <= row["clone_max"])
        if train.sum() < 100 or np.unique(sizes[train]).size < 3:
            raise ValueError("Too few training cells or clone sizes")
        clock = np.round(np.log(sizes), 1)
        description = f"round(log(obsm[{key!r}] clone size), 1)"
    if not np.isfinite(clock).all() or np.unique(clock[train]).size < 2:
        raise ValueError("Training clock needs at least two finite values")
    return clock, train, description


def select_genes(counts, names, train, n_genes):
    if not 2 <= n_genes <= len(names):
        raise ValueError("--genes must be between 2 and the input gene count")
    temporary = ad.AnnData(X=counts[train].copy(), var=pd.DataFrame(index=names.copy()))
    sc.pp.filter_genes(temporary, min_counts=20)
    if temporary.n_vars < n_genes:
        raise ValueError(f"Only {temporary.n_vars} genes pass the 20-count filter")
    sc.pp.normalize_total(temporary, target_sum=1e4)
    sc.pp.log1p(temporary)
    sc.pp.highly_variable_genes(temporary, n_top_genes=n_genes, flavor="cell_ranger")
    scores = temporary.var["dispersions_norm"].to_numpy(dtype=float)
    if np.isfinite(scores).sum() < n_genes:
        raise ValueError("Too few finite HVG dispersion scores")
    ranked = np.argsort(-np.where(np.isfinite(scores), scores, -np.inf), kind="stable")
    selected = np.asarray(names.isin(temporary.var_names[ranked[:n_genes]]))
    if selected.sum() != n_genes:
        raise RuntimeError("HVG selection returned the wrong number of genes")
    return selected


def make_output(obs, var, obsm, counts, selected, clock, train, row):
    raw = counts[:, selected].copy()
    output = ad.AnnData(X=raw.copy(), obs=obs.copy(), var=var.iloc[np.flatnonzero(selected)].copy())
    output.layers["raw_count"] = raw
    for key in (row["coordinate_key"], "X_pca"):
        if key in obsm:
            values = np.asarray(obsm[key], dtype=float)
            if values.shape[0] != output.n_obs or not np.isfinite(values).all():
                raise ValueError(f"Invalid coordinates in obsm[{key!r}]")
            output.obsm[key] = values.copy()
    coords = output.obsm[row["coordinate_key"]]
    if coords.shape != (output.n_obs, 2):
        raise ValueError("PhyloVelo requires 2D coordinates")
    output.obsm["X_umap"] = coords.copy()
    output.obs["phylovelo_clock"] = clock
    output.obs["phylovelo_training_cell"] = train
    sc.pp.normalize_total(output, target_sum=1e4)
    sc.pp.log1p(output)
    return output


def infer(output, clock, train, row):
    raw = output.layers["raw_count"]
    dense = raw.toarray() if sparse.issparse(raw) else np.asarray(raw)
    counts = pd.DataFrame(dense.astype(np.float32, copy=False),
                          index=output.obs_names, columns=output.var_names)
    coords = pd.DataFrame(output.obsm["X_umap"], index=output.obs_names,
                          columns=["UMAP1", "UMAP2"])
    data = pv.scData(count=counts, Xdr=coords, cell_names=list(output.obs_names), cell_generation=clock)
    fit = pv.scData(count=counts.iloc[np.flatnonzero(train)],
                    cell_names=list(output.obs_names[train]), cell_generation=clock[train])
    pv.velocity_inference(fit, time=clock[train], cutoff=row["cutoff"], target="count")
    gene_velocity = np.asarray(fit.velocity, dtype=float)
    if gene_velocity.shape != (output.n_vars,) or not np.isfinite(gene_velocity).all():
        raise ValueError("Invalid gene-wise PhyloVelo slope")
    if not fit.megs or not np.any(gene_velocity):
        raise ValueError("No usable monotonically expressed genes")
    data.velocity, data.megs = gene_velocity, fit.megs
    pv.velocity_embedding(data, target="count", n_neigh=row["embedding_neighbors"], direction="backward")
    backward = np.asarray(data.velocity_embeded, dtype=float).copy()
    if backward.shape != (output.n_obs, 2) or not np.isfinite(backward).all():
        raise ValueError("Invalid backward projection")
    pv.calc_phylo_pseudotime(data, n_neighbors=row["pseudotime_neighbors"],
                             r_sample=row["pseudotime_sample"], method="graph",
                             random_state=row["seed"])
    pseudotime = np.asarray(data.phylo_pseudotime, dtype=float)
    if pseudotime.shape != (output.n_obs,) or not np.isfinite(pseudotime).all() or np.ptp(pseudotime) == 0:
        raise ValueError("Invalid or constant PhyloVelo pseudotime")
    pv.velocity_embedding(data, target="count", n_neigh=row["embedding_neighbors"], direction="forward")
    forward = np.asarray(data.velocity_embeded, dtype=float).copy()
    if forward.shape != (output.n_obs, 2) or not np.isfinite(forward).all():
        raise ValueError("Invalid forward projection")
    if not np.any(np.linalg.norm(forward, axis=1) > 0):
        raise ValueError("All forward projected velocities are zero")
    output.var["phylovelo_gene_velocity"] = gene_velocity.astype(np.float32)
    output.var["phylovelo_meg"] = output.var_names.isin(fit.megs)
    for key, table in (("pval", fit.pvals), ("qval", fit.qvals)):
        output.var[f"phylovelo_{key}"] = table.iloc[0].reindex(output.var_names).to_numpy(dtype=float)
    output.obsm["phylovelo_velocity"] = forward.astype(np.float32)
    output.obsm["phylovelo_velocity_umap"] = output.obsm["phylovelo_velocity"].copy()
    output.obsm["phylovelo_velocity_backward"] = backward.astype(np.float32)
    output.obsm["phylovelo_velocity_backward_umap"] = output.obsm["phylovelo_velocity_backward"].copy()
    add_embedding_velocity_graph(output, vkey="phylovelo_velocity",
                                 velocity_key="phylovelo_velocity_umap",
                                 n_neighbors=row["graph_neighbors"])
    output.obs["phylovelo_pseudotime"] = pseudotime.astype(np.float32)
    output.obs["phylovelo_pseudotime_normalized"] = (
        (pseudotime - pseudotime.min()) / np.ptp(pseudotime)
    ).astype(np.float32)
    return len(fit.megs)


def plot_result(output, row, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scvelo as scv

    directory.mkdir(parents=True, exist_ok=True)
    key = row["annotation_key"]
    if key not in output.obs:
        raise KeyError(f"Missing obs[{key!r}] for plotting")
    velocity = output.obsm["phylovelo_velocity_backward_umap"]
    plots = (
        ("embedding", lambda: scv.pl.scatter(output, basis="umap", color=key, show=False)),
        ("stream", lambda: scv.pl.velocity_embedding_stream(
            output, basis="umap", vkey="phylovelo_velocity_backward", V=velocity,
            color=key, show=False)),
        ("grid", lambda: scv.pl.velocity_embedding_grid(
            output, basis="umap", vkey="phylovelo_velocity_backward", V=velocity,
            color=key, show=False)),
        ("pseudotime", lambda: scv.pl.scatter(
            output, basis="umap", color="phylovelo_pseudotime_normalized", show=False)),
    )
    for name, draw in plots:
        draw()
        plt.savefig(directory / f"{name}.png", dpi=400, bbox_inches="tight")
        plt.close()


def make_row(source, base, args):
    source = {key: (value or "").strip() for key, value in source.items()}
    file_name = source.get("file_path") or str(args.input or "")
    if not file_name:
        raise ValueError("Each dataset needs a file_path")
    path = Path(file_name).expanduser()
    path = path if path.is_absolute() else base / path
    name = source.get("dataset_name") or args.dataset_id or path.stem
    if not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(f"Invalid dataset ID: {name!r}")
    modality = (source.get("modality") or args.modality or "").lower()
    if modality not in {"scrna", "lineage"}:
        raise ValueError(f"{name}: modality must be scrna or lineage")
    annotation = source.get("cluster_key") or args.annotation_key
    coordinate = source.get("dimred_key") or args.coordinate_key
    if not annotation or not coordinate:
        raise ValueError(f"{name}: annotation_key and coordinate_key are required")
    paths = [part.strip() for part in source.get("lineage_paths", "").split("|") if part.strip()]
    paths = paths or args.lineage_path or []
    clock_key = source.get("clock_key") or args.clock_key or ""
    clone_key = source.get("clone_key") or args.clone_key or ""
    mode = (source.get("clock_mode") or args.clock_mode or ("obs" if clock_key else "path")).lower()
    if mode not in {"path", "obs", "clone_size"}:
        raise ValueError(f"{name}: invalid clock_mode")
    if mode == "path" and modality != "scrna":
        raise ValueError(f"{name}: lineage mode requires a numeric or clone-size clock")
    if mode == "path" and not paths:
        raise ValueError(f"{name}: path clock needs lineage_paths")
    if mode == "obs" and not clock_key:
        raise ValueError(f"{name}: obs clock needs clock_key")
    if mode == "clone_size" and not clone_key:
        raise ValueError(f"{name}: clone-size clock needs clone_key")
    row = {
        "dataset_id": name, "file_path": path, "modality": modality,
        "annotation_key": annotation, "coordinate_key": coordinate,
        "count_layer": source.get("count_layer") or args.count_layer or "X",
        "clock_mode": mode, "clock_key": clock_key, "clone_key": clone_key,
        "lineage_paths": paths, "genes": int(source.get("genes") or args.genes),
        "cutoff": float(source.get("cutoff") or args.cutoff),
        "embedding_neighbors": int(source.get("embedding_neighbors") or args.embedding_neighbors),
        "pseudotime_neighbors": int(source.get("pseudotime_neighbors") or args.pseudotime_neighbors or
                                    (10 if modality == "scrna" else 30)),
        "pseudotime_sample": float(source.get("pseudotime_sample") or args.pseudotime_sample),
        "graph_neighbors": int(source.get("graph_neighbors") or args.graph_neighbors),
        "clone_min": int(source.get("clone_min") or args.clone_min),
        "clone_max": int(source.get("clone_max") or args.clone_max),
        "seed": int(source.get("seed") or args.seed),
    }
    if not 0 < row["cutoff"] < 1 or not 0 < row["pseudotime_sample"] <= 1:
        raise ValueError(f"{name}: cutoff or pseudotime sample is out of range")
    if min(row["embedding_neighbors"], row["pseudotime_neighbors"]) < 2 or row["graph_neighbors"] < 1:
        raise ValueError(f"{name}: neighbor count is out of range")
    if row["clone_min"] < 1 or row["clone_max"] < row["clone_min"]:
        raise ValueError(f"{name}: clone-size bounds are invalid")
    return row


def input_rows(args):
    if args.metadata_file:
        with args.metadata_file.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"dataset_name", "file_path", "modality", "cluster_key", "dimred_key"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"CSV needs {', '.join(sorted(required))}")
            rows = [make_row(source, args.metadata_file.parent, args) for source in reader]
    else:
        rows = [make_row({}, Path.cwd(), args)]
    if not rows or len({row["dataset_id"] for row in rows}) != len(rows):
        raise ValueError("Dataset list is empty or has duplicate IDs")
    return rows


def run_one(row, output_dir, plot_dir, overwrite):
    name = row["dataset_id"]
    destination = output_dir / f"{name}.h5ad"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    np.random.seed(row["seed"])
    obs, var, obsm, counts, removed = read_input(row)
    if row["annotation_key"] not in obs:
        raise KeyError(f"Missing obs[{row['annotation_key']!r}]")
    clock, train, description = get_clock(obs, obsm, row)
    selected = select_genes(counts, var.index, train, row["genes"])
    output = make_output(obs, var, obsm, counts, selected, clock, train, row)
    n_megs = infer(output, clock, train, row)
    output.uns["phylovelo_run"] = {
        "dataset_id": name, "modality": row["modality"],
        "count_source": row["count_layer"], "clock_source": description,
        "clock_mode": row["clock_mode"], "lineage_paths": row["lineage_paths"],
        "zero_count_cells_removed": removed, "n_training_cells": int(train.sum()),
        "n_hvg": int(output.n_vars), "n_megs": n_megs, "cutoff": row["cutoff"],
        "embedding_neighbors": row["embedding_neighbors"],
        "pseudotime_neighbors": row["pseudotime_neighbors"],
        "pseudotime_sample": row["pseudotime_sample"],
        "graph_neighbors": row["graph_neighbors"], "seed": row["seed"],
        "velocity_kind": "gene-wise slope in var; forward/backward 2D projections in obsm",
        "graph_kind": "benchmark-derived forward 2D cosine graph",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(destination, compression="lzf")
    if plot_dir is not None:
        plot_result(output, row, plot_dir / name)
    print(f"{name}: {output.n_obs} cells, {output.n_vars} genes, {n_megs} MEGs -> {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input", type=Path)
    sources.add_argument("--metadata-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, help="Optional PNG output directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset-id")
    parser.add_argument("--modality", choices=("scrna", "lineage"))
    parser.add_argument("--annotation-key")
    parser.add_argument("--coordinate-key")
    parser.add_argument("--count-layer", help="Integer counts in X or this layer")
    parser.add_argument("--clock-mode", choices=("path", "obs", "clone_size"))
    parser.add_argument("--clock-key", help="Numeric obs column")
    parser.add_argument("--clone-key", help="One-hot clone membership in obsm")
    parser.add_argument("--lineage-path", action="append", help="Comma-separated path labels; repeat for branches")
    parser.add_argument("--genes", type=int, default=1000)
    parser.add_argument("--cutoff", type=float, default=0.95)
    parser.add_argument("--embedding-neighbors", type=int, default=15)
    parser.add_argument("--pseudotime-neighbors", type=int)
    parser.add_argument("--pseudotime-sample", type=float, default=0.1)
    parser.add_argument("--graph-neighbors", type=int, default=30)
    parser.add_argument("--clone-min", type=int, default=11)
    parser.add_argument("--clone-max", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for row in input_rows(args):
        run_one(row, args.output_dir, args.plot_dir, args.overwrite)


if __name__ == "__main__":
    main()
