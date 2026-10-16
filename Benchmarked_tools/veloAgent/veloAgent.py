"""Run the veloAgent benchmark fork with explicit spatial and STRING inputs.

Source: https://github.com/sd68515/veloAgent_benchmark
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import torch

# Direct execution must import the installed package rather than this file.
_import_path = sys.path[:]
try:
    sys.path = [entry for entry in sys.path
                if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
    import veloagent
finally:
    sys.path = _import_path


TARGET_NEIGHBORS = 17.5
SEED = 618


def ensure_spatial_basis(adata):
    """Provide the X_ coordinate alias required by scVelo's spatial basis."""
    if "X_spatial" not in adata.obsm:
        if "spatial" not in adata.obsm:
            raise KeyError("Missing obsm['spatial'] for scVelo basis='spatial'")
        adata.obsm["X_spatial"] = adata.obsm["spatial"].copy()


def resolve(value, base):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def read_rows(args):
    required = ("dataset_name", "file_path", "modality", "cluster_key",
                "protein_info", "protein_aliases", "protein_links")
    if args.metadata_file:
        with args.metadata_file.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not set(required).issubset(reader.fieldnames or []):
                raise ValueError(f"CSV needs {', '.join(required)}")
            sources = list(reader)
        base = args.metadata_file.parent
    else:
        sources = [{
            "dataset_name": args.dataset_id or args.input.stem,
            "file_path": str(args.input), "modality": "spatial", "cluster_key": args.annotation_key,
            "dimred_key": args.coordinate_key or "spatial", "protein_info": str(args.protein_info or ""),
            "protein_aliases": str(args.protein_aliases or ""),
            "protein_links": str(args.protein_links or ""),
            "homolog_key": args.homolog_key or "",
            "radius_reference": str(args.radius_reference or ""),
            "nbr_radius": str(args.nbr_radius) if args.nbr_radius is not None else "",
        }]
        base = Path.cwd()
    rows = []
    for source in sources:
        if None in source:
            raise ValueError("CSV row has more values than its header")
        source = {key: (value or "").strip() for key, value in source.items()}
        if any(not source.get(key) for key in required):
            raise ValueError("Dataset row has an empty required field")
        if source["modality"] != "spatial":
            raise ValueError("veloAgent CSV modality must be spatial")
        name = source["dataset_name"]
        if name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError(f"Invalid dataset ID: {name!r}")
        radius = source.get("nbr_radius") or (str(args.nbr_radius) if args.nbr_radius is not None else "")
        reference = source.get("radius_reference") or str(args.radius_reference or "")
        if radius and reference:
            raise ValueError(f"{name}: choose either nbr_radius or radius_reference")
        radius = float(radius) if radius else None
        if radius is not None and (not math.isfinite(radius) or radius <= 0):
            raise ValueError(f"{name}: nbr_radius must be positive and finite")
        row = {
            "dataset_id": name,
            "file_path": resolve(source["file_path"], base),
            "annotation_key": source["cluster_key"],
            "coordinate_key": source.get("dimred_key") or "spatial",
            "protein_info": resolve(source["protein_info"], base),
            "protein_aliases": resolve(source["protein_aliases"], base),
            "protein_links": resolve(source["protein_links"], base),
            "homolog_key": source.get("homolog_key") or args.homolog_key or "",
            "radius_reference": resolve(reference, base) if reference else None,
            "nbr_radius": radius,
        }
        rows.append(row)
    if not rows or len({row["dataset_id"] for row in rows}) != len(rows):
        raise ValueError("Dataset list is empty or has duplicate IDs")
    return rows


def seed_training(threads):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(threads)


def select_device(value):
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count()):
            raise ValueError(f"CUDA device is unavailable: {value}")
    elif device.type != "cpu":
        raise ValueError("--device must be cpu, cuda, or cuda:<index>")
    return device


def coordinates(adata, key):
    if {"x_raw", "y_raw"}.issubset(adata.obs.columns):
        values = adata.obs[["x_raw", "y_raw"]].to_numpy(dtype=float)
        if values.shape != (adata.n_obs, 2) or not np.isfinite(values).all():
            raise ValueError("obs['x_raw'] and obs['y_raw'] must contain finite coordinates")
        return values
    if key not in adata.obsm:
        raise KeyError(f"Missing obsm[{key!r}]")
    values = np.asarray(adata.obsm[key], dtype=float)
    if values.shape != (adata.n_obs, 2) or not np.isfinite(values).all():
        raise ValueError(f"obsm[{key!r}] must contain finite 2D coordinates")
    return values


def choose_radius(coords, target=TARGET_NEIGHBORS):
    count = len(coords)
    if count < 2 or count - 1 < target:
        raise ValueError("Too few cells for the target spatial neighbor count")
    tree = cKDTree(coords)

    def average(radius):
        return (tree.count_neighbors(tree, radius) - count) / count

    if average(0.0) >= target:
        raise ValueError("Duplicate coordinates exceed target neighbors at zero radius")
    kth = min(count, math.ceil(target) + 1)
    high = float(np.median(tree.query(coords, k=kth)[0][:, -1]))
    if high <= 0:
        raise ValueError("Cannot find a positive spatial radius")
    low = 0.0
    while average(high) < target:
        low, high = high, 2 * high
        if not math.isfinite(high):
            raise ValueError("Spatial radius search exceeded finite range")
    for _ in range(36):
        middle = (low + high) / 2
        if middle in {low, high}:
            break
        if average(middle) < target:
            low = middle
        else:
            high = middle
    return min((low, high), key=lambda value: (abs(average(value) - target), value))


def radius_for(row, coords):
    if row["nbr_radius"] is not None:
        radius, mode = row["nbr_radius"], "explicit"
    elif row["radius_reference"] is not None:
        reference = sc.read_h5ad(row["radius_reference"], backed="r")
        try:
            radius = choose_radius(coordinates(reference, row["coordinate_key"]))
        finally:
            reference.file.close()
        mode = "reference_tissue"
    else:
        radius, mode = choose_radius(coords), "dataset_auto"
    counts = np.asarray(cKDTree(coords).query_ball_point(coords, radius, return_length=True)) - 1
    stats = {
        "radius": float(radius), "mode": mode,
        "mean_neighbors": float(counts.mean()),
        "median_neighbors": float(np.median(counts)),
        "zero_neighbor_cells": int(np.count_nonzero(counts == 0)),
        "target_mean_neighbors": TARGET_NEIGHBORS,
    }
    return float(radius), stats


def gene_queries(adata, homolog_key):
    original = adata.var_names.astype(str).to_numpy(copy=True)
    adata.var["_source_var_name"] = original
    if homolog_key:
        if homolog_key not in adata.var:
            raise KeyError(f"Missing var[{homolog_key!r}]")
        queries = adata.var[homolog_key].astype(str).str.extract(
            r"(?:^|\|)\s*([^|]+?)\[hs\]", expand=False,
        ).str.strip()
        if queries.isna().any():
            raise ValueError(f"var[{homolog_key!r}] has genes without a [hs] homolog")
    else:
        queries = adata.var_names.astype(str).str.replace(r"\.\d+$", "", regex=True)
    adata.var["_string_query"] = queries.to_numpy()
    duplicates = int(adata.var_names.duplicated().sum())
    if duplicates:
        adata.var_names_make_unique(join="-")
    adata.var_names = adata.var_names.astype(str)
    return duplicates


def map_proteins(queries, info_path, alias_path):
    values = [str(value).strip() for value in queries]
    wanted = {value for value in values if value and value.lower() != "nan"}
    wanted_upper = defaultdict(set)
    for value in wanted:
        wanted_upper[value.upper()].add(value)
    preferred = defaultdict(set)
    with info_path.open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            fields = line.rstrip("\n").split("\t", 2)
            if len(fields) >= 2:
                protein, symbol = fields[0], fields[1].strip()
                matches = ({symbol} if symbol in wanted else set()) | wanted_upper.get(symbol.upper(), set())
                for query in matches:
                    preferred[query].add(protein)
    unresolved = wanted - set(preferred)
    unresolved_upper = defaultdict(set)
    for value in unresolved:
        unresolved_upper[value.upper()].add(value)
    aliases = defaultdict(set)
    if unresolved:
        with alias_path.open(encoding="utf-8") as handle:
            next(handle, None)
            for line in handle:
                fields = line.rstrip("\n").split("\t", 2)
                if len(fields) >= 2:
                    protein, alias = fields[0], fields[1].strip()
                    matches = ({alias} if alias in unresolved else set()) | unresolved_upper.get(alias.upper(), set())
                    for query in matches:
                        aliases[query].add(protein)
    chosen = {query: min(proteins) for query in wanted
              if (proteins := preferred.get(query) or aliases.get(query))}
    return [chosen.get(query) for query in values]


def string_prior(queries, row, threshold):
    """Build a symmetric binary prior from STRING's undirected interactions."""
    proteins = map_proteins(queries, row["protein_info"], row["protein_aliases"])
    keep = np.asarray([protein is not None for protein in proteins])
    mapped = [protein for protein in proteins if protein is not None]
    if len(mapped) < 2:
        raise ValueError("Fewer than two genes mapped to STRING proteins")
    protein_indices = defaultdict(list)
    for index, protein in enumerate(mapped):
        protein_indices[protein].append(index)
    matrix = np.zeros((len(mapped), len(mapped)), dtype=np.uint8)
    with row["protein_links"].open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            fields = line.split()
            if len(fields) < 3:
                continue
            left, right = fields[:2]
            if left not in protein_indices or right not in protein_indices:
                continue
            if threshold is not None and int(fields[2]) < threshold:
                continue
            left_indices = protein_indices[left]
            right_indices = protein_indices[right]
            matrix[np.ix_(left_indices, right_indices)] = 1
            matrix[np.ix_(right_indices, left_indices)] = 1
    isolated = np.flatnonzero(np.all(matrix == 0, axis=1))
    for index in isolated:
        matrix[index, :] = 1
        matrix[:, index] = 1
    np.fill_diagonal(matrix, 1)
    stats = {
        "mapped_genes": len(mapped), "unmapped_genes_removed": int((~keep).sum()),
        "unique_proteins": len(protein_indices), "isolated_rows_connected": len(isolated),
        "density": float(matrix.mean()),
        "confidence_threshold": "disabled" if threshold is None else threshold,
    }
    return matrix, stats, keep


def neighbor_indices(adata, representation=None, requested=30):
    key = representation or "X_pca"
    if key not in adata.obsm:
        raise KeyError(f"Missing obsm[{key!r}]")
    count = min(requested, adata.n_obs)
    if count < 2:
        raise ValueError("At least two cells are needed for neighbors")
    values = np.asarray(adata.obsm[key])
    indices = NearestNeighbors(n_neighbors=count).fit(values).kneighbors(values, return_distance=False)
    adata.uns.setdefault("neighbors", {})["indices"] = indices
    return indices


def embedding_from_vae(adata, vae, device):
    indices = neighbor_indices(adata)
    sources = np.repeat(np.arange(adata.n_obs), indices.shape[1] - 1)
    targets = indices[:, 1:].reshape(-1)
    edges = torch.as_tensor(np.vstack((sources, targets)), dtype=torch.long, device=device)
    def as_tensor(layer):
        values = layer.toarray() if issparse(layer) else np.asarray(layer)
        return torch.as_tensor(values, dtype=torch.float32, device=device)
    vae.eval()
    with torch.no_grad():
        *_, latent = vae(as_tensor(adata.layers["unspliced"]),
                         as_tensor(adata.layers["spliced"]), edges)
    adata.obsm["cell_embed"] = latent.detach().cpu().numpy()
    sc.pp.neighbors(adata, n_neighbors=min(30, adata.n_obs - 1), use_rep="cell_embed")
    neighbor_indices(adata, "cell_embed")


def run_one(row, args, device):
    name = row["dataset_id"]
    destination = args.output_dir / f"{name}.h5ad"
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    for key in ("file_path", "protein_info", "protein_aliases", "protein_links"):
        if not row[key].is_file():
            raise FileNotFoundError(row[key])
    seed_training(args.threads)
    adata = sc.read_h5ad(row["file_path"])
    annotation = row["annotation_key"]
    if annotation not in adata.obs or adata.obs[annotation].isna().any():
        raise ValueError(f"Missing or incomplete obs[{annotation!r}]")
    coords = coordinates(adata, row["coordinate_key"])
    adata.obsm["spatial"] = coords.copy()
    ensure_spatial_basis(adata)
    adata.obs["x_raw"], adata.obs["y_raw"] = coords[:, 0], coords[:, 1]
    duplicates = gene_queries(adata, row["homolog_key"])
    original_shape = (adata.n_obs, adata.n_vars)
    veloagent.preprocess(adata, num_genes=2000, min_count=20, log_norm=True)
    prior, prior_stats, keep = string_prior(adata.var["_string_query"].astype(str), row,
                                            args.confidence_threshold)
    adata = adata[:, keep].copy()
    if prior.shape != (adata.n_vars, adata.n_vars):
        raise ValueError("STRING prior does not match the selected genes")
    veloagent.preprocess(adata, num_genes=adata.n_vars, min_count=0, log_norm=False)
    ensure_spatial_basis(adata)
    radius, radius_stats = radius_for(row, coordinates(adata, "spatial"))

    neighbors = neighbor_indices(adata)
    vae, optimizer, loss = veloagent.get_vae(adata, z_dim=5, lr=1e-2)
    vae.to(device)
    best = veloagent.train_vae(adata, vae, optimizer, loss, neighbors,
                               patience=25, num_epochs=1000, batch=256, device=device)
    if best is None:
        raise ValueError("veloAgent VAE returned no best model")
    vae.load_state_dict(best)
    embedding_from_vae(adata, vae, device)
    genenet = veloagent.GeneNet(in_dim=adata.obsm["cell_embed"].shape[1],
                                gene_dim=adata.n_vars, conn=prior)
    genenet.to(device)
    optimizer = torch.optim.Adam(genenet.parameters(), lr=1e-4, weight_decay=1e-5)
    veloagent.train_gg(num_epochs=500, data=adata, embed_basis="cell_embed",
                       genenet=genenet, device=device, optimizer=optimizer,
                       patience=25, num_nbrs=min(30, adata.n_obs), dt=0.5, batch=256)
    veloagent.train_nbr(num_epochs=50, data=adata, embed_basis="cell_embed",
                        genenet=genenet, device=device, optimizer=optimizer,
                        num_nbrs=min(30, adata.n_obs), dt=0.5, batch=256, tau_nbr=0.2)
    model = veloagent.CellModel(adata, steps=100, tau=2, nbr_radius=radius, sig_ratio=0.7)
    model.step(show_progress=True)
    if "velocity" not in adata.layers:
        raise KeyError("veloAgent output lacks layers['velocity']")
    velocity = adata.layers["velocity"]
    values = velocity.data if issparse(velocity) else np.asarray(velocity)
    if not np.isfinite(values).all():
        raise ValueError("veloAgent output velocity contains non-finite values")
    for frame in (adata.obs, adata.var):
        names = [str(key).replace("/", "_") for key in frame.columns]
        if len(set(names)) != len(names):
            raise ValueError("Column sanitization would duplicate names")
        frame.columns = names
    adata.uns["veloagent_run"] = {
        "dataset_id": name, "annotation_key": annotation.replace("/", "_"),
        "coordinate_key": row["coordinate_key"], "seed": SEED,
        "device": str(device), "threads": args.threads,
        "homolog_key": row["homolog_key"],
        "string_prior": prior_stats, "string_unmapped_policy": "drop",
        "string_isolated_policy": "full_connect",
        "radius": radius_stats, "original_shape": list(original_shape),
        "output_shape": [adata.n_obs, adata.n_vars],
        "duplicate_gene_names_made_unique": duplicates,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    ensure_spatial_basis(adata)
    adata.write_h5ad(destination)
    print(f"{name}: {adata.n_obs} cells, {adata.n_vars} genes, radius {radius:.6g} -> {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input", type=Path)
    sources.add_argument("--metadata-file", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-id")
    parser.add_argument("--annotation-key")
    parser.add_argument("--coordinate-key", default="spatial")
    parser.add_argument("--protein-info", type=Path)
    parser.add_argument("--protein-aliases", type=Path)
    parser.add_argument("--protein-links", type=Path)
    parser.add_argument("--homolog-key", help="var column with [hs] human homolog symbols")
    parser.add_argument("--radius-reference", type=Path, help="Parent tissue H5AD for a shared radius")
    parser.add_argument("--nbr-radius", type=float)
    parser.add_argument("--confidence-threshold", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or (args.confidence_threshold is not None and args.confidence_threshold < 0):
        parser.error("--threads must be positive and --confidence-threshold nonnegative")
    device = select_device(args.device)
    for row in read_rows(args):
        run_one(row, args, device)


if __name__ == "__main__":
    main()
