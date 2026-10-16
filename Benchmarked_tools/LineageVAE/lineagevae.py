#!/usr/bin/env python3
"""Train LineageVAE on explicit lineage count, clock, and clone inputs.

The output contains the original cells and the selected modeling genes.  Its
``velocity`` layer is the forward, one-day change in log-normalized decoded
expression: decoder(z_today) minus decoder(z_yesterday).  It is derived from
LineageVAE's backward latent transition, not from splicing kinetics.

No synthetic historical cells are written to the benchmark object.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from modules import LineageVAE


def drop_zero_expression_cells(adata, layer):
    if layer not in adata.layers:
        raise KeyError(f"Missing layers[{layer!r}]")
    totals = np.asarray(adata.layers[layer].sum(axis=1)).ravel()
    if not np.isfinite(totals).all() or np.any(totals < 0):
        raise ValueError("Count layer contains invalid cell totals")
    keep = totals > 0
    if not keep.any():
        raise ValueError("All cells have zero counts")
    removed = int((~keep).sum())
    return (adata[keep].copy() if removed else adata), removed


def add_embedding_velocity_graph(adata, *, vkey, velocity_key, n_neighbors=30):
    """Store benchmark-derived 2D cosine graphs separately from the gene graph."""
    positions = np.asarray(adata.obsm["X_umap"], dtype=np.float64)
    velocity = np.asarray(adata.obsm[velocity_key], dtype=np.float64)
    if positions.shape != velocity.shape or positions.shape != (adata.n_obs, 2):
        raise ValueError("UMAP coordinates and embedded velocity must be cells x 2")
    if not np.isfinite(positions).all() or not np.isfinite(velocity).all() or adata.n_obs < 2:
        raise ValueError("Invalid UMAP positions or embedded velocities")
    k = min(n_neighbors + 1, adata.n_obs)
    indices = np.asarray(cKDTree(positions).query(positions, k=k)[1]).reshape(adata.n_obs, k)
    rows = np.repeat(np.arange(adata.n_obs), k)
    columns = indices.ravel()
    keep = rows != columns
    rows, columns = rows[keep], columns[keep]
    displacement = positions[columns] - positions[rows]
    numerator = np.einsum("ij,ij->i", velocity[rows], displacement)
    denominator = np.linalg.norm(velocity[rows], axis=1) * np.linalg.norm(displacement, axis=1)
    cosine = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    shape = (adata.n_obs, adata.n_obs)
    positive = csr_matrix((np.clip(cosine, 0, 1), (rows, columns)), shape=shape)
    negative = csr_matrix((np.clip(cosine, -1, 0), (rows, columns)), shape=shape)
    positive.eliminate_zeros()
    negative.eliminate_zeros()
    adata.uns[f"{vkey}_graph"] = positive
    adata.uns[f"{vkey}_graph_neg"] = negative
    adata.uns[f"{vkey}_embedding_graph_params"] = {
        "basis": "X_umap", "velocity_key": velocity_key,
        "n_neighbors": int(min(n_neighbors, adata.n_obs - 1)),
        "graph_kind": "benchmark-derived 2D cosine graph",
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input", type=Path)
    sources.add_argument("--metadata-file", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-id")
    parser.add_argument("--annotation-key")
    parser.add_argument("--coordinate-key")
    parser.add_argument("--pca-key", help="Optional existing PCA representation")
    parser.add_argument("--count-layer")
    parser.add_argument("--time-key")
    parser.add_argument("--clone-key", help="Clone IDs in obs")
    parser.add_argument("--clone-matrix-key", help="One-hot clone membership in obsm")
    parser.add_argument("--required-day", type=int)
    parser.add_argument("--min-distinct-days", type=int, default=1)
    parser.add_argument("--genes", type=int, default=1000, help="Number of highly variable genes")
    parser.add_argument("--clone-batch-size", type=int, default=30)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--embed-epochs", type=int, default=100)
    parser.add_argument("--dynamics-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=Path, help="Optional model checkpoint directory")
    parser.add_argument("--plot-dir", type=Path, help="Optional PNG output directory")
    return parser.parse_args()


def resolve(value, base):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def input_rows(args):
    required = ("dataset_name", "file_path", "modality", "cluster_key", "dimred_key",
                "count_layer", "time_key")
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
            "file_path": str(args.input), "modality": "lineage", "cluster_key": args.annotation_key,
            "dimred_key": args.coordinate_key, "pca_key": args.pca_key,
            "count_layer": args.count_layer, "time_key": args.time_key,
            "clone_key": args.clone_key, "clone_matrix_key": args.clone_matrix_key,
        }]
        base = Path.cwd()
    rows = []
    for source in sources:
        if None in source:
            raise ValueError("CSV row has more values than its header")
        row = {key: (value or "").strip() for key, value in source.items()}
        if any(not row.get(key) for key in required):
            raise ValueError("Dataset row has an empty required field")
        if row["modality"] != "lineage":
            raise ValueError("LineageVAE CSV modality must be lineage")
        name = row["dataset_name"]
        if name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError(f"Invalid dataset ID: {name!r}")
        clone_key = row.get("clone_key") or args.clone_key or ""
        clone_matrix_key = row.get("clone_matrix_key") or args.clone_matrix_key or ""
        if bool(clone_key) == bool(clone_matrix_key):
            raise ValueError(f"{name}: choose exactly one clone_key or clone_matrix_key")
        run_args = argparse.Namespace(**vars(args))
        run_args.dataset_id = name
        run_args.input = resolve(row["file_path"], base)
        run_args.output = args.output_dir / f"{name}.h5ad"
        run_args.checkpoint = args.checkpoint_dir / f"{name}.pt" if args.checkpoint_dir else None
        run_args.annotation_key = row["cluster_key"]
        run_args.coordinate_key = row["dimred_key"]
        run_args.pca_key = row.get("pca_key") or args.pca_key or ""
        run_args.count_layer = row["count_layer"]
        run_args.time_key = row["time_key"]
        run_args.clone_key = clone_key
        run_args.clone_matrix_key = clone_matrix_key
        for key in ("genes", "clone_batch_size", "embed_batch_size", "embed_epochs",
                    "dynamics_epochs", "seed", "threads", "min_distinct_days"):
            if row.get(key):
                setattr(run_args, key, int(row[key]))
        for key in ("learning_rate",):
            if row.get(key):
                setattr(run_args, key, float(row[key]))
        run_args.required_day = int(row["required_day"]) if row.get("required_day") else args.required_day
        if run_args.clone_batch_size < 2 or run_args.min_distinct_days < 1:
            raise ValueError(f"{name}: invalid clone training constraints")
        rows.append(run_args)
    if not rows or len({row.dataset_id for row in rows}) != len(rows):
        raise ValueError("Dataset list is empty or has duplicate IDs")
    return rows


def prepare_source(args):
    source = sc.read_h5ad(args.input)
    if args.annotation_key not in source.obs or source.obs[args.annotation_key].isna().any():
        raise ValueError(f"Missing or incomplete obs[{args.annotation_key!r}]")
    if args.count_layer == "X":
        counts = source.X.copy()
    elif args.count_layer in source.layers:
        counts = source.layers[args.count_layer].copy()
    else:
        raise KeyError(f"Missing layers[{args.count_layer!r}]")
    source.layers["raw_count"] = counts
    source, removed = drop_zero_expression_cells(source, "raw_count")
    if args.time_key not in source.obs:
        raise KeyError(f"Missing obs[{args.time_key!r}]")
    source.obs["Day"] = pd.to_numeric(source.obs[args.time_key], errors="raise")
    if args.clone_matrix_key:
        if args.clone_matrix_key not in source.obsm:
            raise KeyError(f"Missing obsm[{args.clone_matrix_key!r}]")
        clones = source.obsm[args.clone_matrix_key]
        clones = sparse.csr_matrix(clones, copy=True)
        clones.sum_duplicates()
        clones.eliminate_zeros()
        if not np.all(clones.data == 1) or not np.all(np.diff(clones.indptr) == 1):
            raise ValueError("Clone matrix needs exactly one one-hot membership per cell")
        source.obs["Lineage"] = [f"clone_{index}" for index in clones.indices]
        clone_source = f"obsm[{args.clone_matrix_key!r}]"
    else:
        if args.clone_key not in source.obs:
            raise KeyError(f"Missing obs[{args.clone_key!r}]")
        if source.obs[args.clone_key].isna().any():
            raise ValueError(f"obs[{args.clone_key!r}] contains missing clone IDs")
        source.obs["Lineage"] = source.obs[args.clone_key].astype(str)
        clone_source = f"obs[{args.clone_key!r}]"
    if args.coordinate_key not in source.obsm:
        raise KeyError(f"Missing obsm[{args.coordinate_key!r}]")
    coords = np.asarray(source.obsm[args.coordinate_key], dtype=float)
    if coords.shape != (source.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError("Coordinate key needs finite 2D cell coordinates")
    source.obsm["X_umap"] = coords.copy()
    if args.pca_key:
        if args.pca_key not in source.obsm:
            raise KeyError(f"Missing obsm[{args.pca_key!r}]")
        pca = np.asarray(source.obsm[args.pca_key], dtype=float)
        if pca.ndim != 2 or pca.shape[0] != source.n_obs or not np.isfinite(pca).all():
            raise ValueError("PCA key has invalid coordinates")
        source.obsm["X_pca"] = pca.copy()
    args.metadata = {
        "dataset_id": args.dataset_id, "annotation_key": args.annotation_key,
        "coordinate_key": args.coordinate_key, "count_source": args.count_layer,
        "time_source": args.time_key, "clone_source": clone_source,
        "zero_expression_cells_removed": removed,
        "required_day": "none" if args.required_day is None else args.required_day,
        "min_distinct_days": args.min_distinct_days,
    }
    return source


def validate_input(adata: ad.AnnData, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    for key in ("Day", "Lineage"):
        if key not in adata.obs:
            raise ValueError(f"Missing obs[{key!r}]")
    if "raw_count" not in adata.layers:
        raise ValueError("Missing layers['raw_count']")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Cell and gene names must be unique")
    raw = adata.layers["raw_count"]
    values = raw.data if sparse.issparse(raw) else np.asarray(raw).ravel()
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("raw_count is empty or contains non-finite values")
    if np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError("raw_count must contain nonnegative integer counts")
    days = pd.to_numeric(adata.obs["Day"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(days).all() or np.any(days < 0) or np.any(days != np.floor(days)):
        raise ValueError("Day must contain finite nonnegative integer time points")
    if adata.obs["Lineage"].isna().any():
        raise ValueError("Lineage contains missing clone IDs")
    lineages = adata.obs["Lineage"].astype(str).to_numpy()
    counts = pd.Series(lineages).value_counts()
    if (counts >= batch_size).sum() < 2:
        raise ValueError("Need at least two clones with enough cells for lineage batches")
    cell_totals = np.asarray(raw.sum(axis=1)).ravel()
    if np.any(cell_totals <= 0):
        raise ValueError("raw_count contains zero-library cells")
    return days.astype(np.int64), lineages


def select_genes(adata: ad.AnnData, n_genes: int) -> np.ndarray:
    if n_genes < 2 or n_genes > adata.n_vars:
        raise ValueError("--genes must be between 2 and the input gene count")
    temporary = ad.AnnData(X=adata.layers["raw_count"].copy(), var=adata.var.copy())
    sc.pp.filter_genes(temporary, min_counts=20)
    if temporary.n_vars < n_genes:
        raise ValueError(f"Only {temporary.n_vars} genes pass the 20-count filter")
    sc.pp.normalize_total(temporary, target_sum=1e4)
    sc.pp.log1p(temporary)
    sc.pp.highly_variable_genes(temporary, n_top_genes=n_genes, flavor="cell_ranger")
    chosen_names = temporary.var_names[temporary.var["highly_variable"].to_numpy(dtype=bool)]
    mask = adata.var_names.isin(chosen_names)
    if mask.sum() != n_genes:
        raise RuntimeError(f"HVG selection returned {mask.sum()} genes, expected {n_genes}")
    return mask


def normalization_matrix(counts: np.ndarray) -> np.ndarray:
    """The original LineageVAE Poisson exposure matrix, over selected genes."""
    row = counts.sum(axis=1, dtype=np.float64)
    column = counts.sum(axis=0, dtype=np.float64)
    if np.any(row <= 0) or np.any(column <= 0):
        raise ValueError("Selected expression contains a zero-library cell or zero-count gene")
    scale = counts.mean(dtype=np.float64) / (row.mean() * column.mean())
    return (scale * row[:, None] * column[None, :]).astype(np.float32)


def clone_indices(
    lineages: np.ndarray, batch_size: int, days: np.ndarray | None = None,
    required_day: int | None = None, min_distinct_days: int = 1,
):
    if (required_day is not None or min_distinct_days > 1) and days is None:
        raise ValueError("Clone time filters require day annotations")
    groups = {}
    for clone in pd.unique(lineages):
        indices = np.flatnonzero(lineages == clone)
        if len(indices) >= batch_size and (
            required_day is None or np.any(days[indices] == required_day)
        ) and (
            min_distinct_days == 1 or len(np.unique(days[indices])) >= min_distinct_days
        ):
            groups[str(clone)] = indices
    if not groups:
        raise ValueError("No eligible clones")
    return groups


def sample_clone_batch(indices: np.ndarray, days: np.ndarray, size: int, rng: np.random.Generator):
    """One clone per batch; include one cell from each observed time first."""
    chosen = [rng.choice(indices[days[indices] == day]) for day in np.unique(days[indices])]
    if len(chosen) > size:
        raise ValueError("A clone has more observed time points than the batch size")
    remainder = np.setdiff1d(indices, np.asarray(chosen), assume_unique=True)
    chosen.extend(rng.choice(remainder, size=size - len(chosen), replace=False).tolist())
    return np.asarray(chosen, dtype=np.int64)


def train_embedding(model, counts, exposure, days, args, device, rng):
    optimizer = torch.optim.Adam(
        list(model.enc_z.parameters()) + list(model.dec_z.parameters()), lr=args.learning_rate
    )
    history = []
    for epoch in range(args.embed_epochs):
        model.train()
        order = rng.permutation(len(days))
        total = 0.0
        seen = 0
        for start in range(0, len(order), args.embed_batch_size):
            indices = order[start : start + args.embed_batch_size]
            x = torch.from_numpy(counts[indices]).to(device)
            norm = torch.from_numpy(exposure[indices]).to(device)
            t = torch.from_numpy(days[indices].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            mu, logvar = model.enc_z(x, t)
            sigma = F.softplus(logvar) + 1e-5
            z = mu + sigma * torch.randn_like(mu)
            rate = model.dec_z(z, t) * norm
            reconstruction = -torch.distributions.Poisson(rate).log_prob(x).sum()
            kld = 0.5 * (mu.square() + sigma.square() - 1 - 2 * sigma.log()).sum()
            loss = (reconstruction + kld) / len(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite embedding loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            seen += len(indices)
        history.append(total / seen)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"embedding epoch {epoch + 1}/{args.embed_epochs}: {history[-1]:.4g}", flush=True)
    return history


def train_dynamics(model, counts, exposure, days, lineages, args, device, rng):
    groups = clone_indices(
        lineages, args.clone_batch_size, days, getattr(args, "required_day", None),
        getattr(args, "min_distinct_days", 1),
    )
    clones = np.asarray(list(groups))
    rng.shuffle(clones)
    n_val = max(1, int(round(len(clones) * 0.1)))
    train_clones, validation_clones = clones[n_val:], clones[:n_val]
    if len(train_clones) == 0:
        raise ValueError("No training clones remain after validation split")
    for parameter in model.enc_z.parameters():
        parameter.requires_grad_(False)
    for parameter in model.dec_z.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate
    )
    history = []
    validation_history = []
    fixed_validation = {
        clone: sample_clone_batch(groups[clone], days, args.clone_batch_size, rng)
        for clone in validation_clones
    }
    for epoch in range(args.dynamics_epochs):
        model.train()
        rng.shuffle(train_clones)
        total = 0.0
        for clone in train_clones:
            indices = sample_clone_batch(groups[clone], days, args.clone_batch_size, rng)
            x = torch.from_numpy(counts[indices]).to(device)
            norm = torch.from_numpy(exposure[indices]).to(device)
            t = torch.from_numpy(days[indices].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.elbo_loss(x, torch.zeros_like(x), t, norm, kinetics=False)
            loss = loss / len(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite dynamics loss at epoch {epoch + 1}, clone {clone}")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        history.append(total / len(train_clones))
        model.eval()
        with torch.no_grad():
            validation_total = 0.0
            for indices in fixed_validation.values():
                x = torch.from_numpy(counts[indices]).to(device)
                norm = torch.from_numpy(exposure[indices]).to(device)
                t = torch.from_numpy(days[indices].astype(np.float32)).to(device)
                validation_loss, _ = model.elbo_loss(
                    x, torch.zeros_like(x), t, norm, kinetics=False
                )
                if not torch.isfinite(validation_loss):
                    raise FloatingPointError(f"Non-finite validation loss at epoch {epoch + 1}")
                validation_total += float(validation_loss) / len(indices)
            validation_history.append(validation_total / len(fixed_validation))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"dynamics epoch {epoch + 1}/{args.dynamics_epochs}: "
                f"train={history[-1]:.4g}, val={validation_history[-1]:.4g}",
                flush=True,
            )
    return history, validation_history, train_clones.tolist(), validation_clones.tolist()


def log_normalize(values: torch.Tensor) -> torch.Tensor:
    return torch.log1p(values / values.sum(dim=1, keepdim=True) * 1e4)


def infer_velocity(model, counts, exposure, days, device, batch_size):
    n_cells, n_genes = counts.shape
    z_all = np.empty((n_cells, model.z_dim), dtype=np.float32)
    d_all = np.empty_like(z_all)
    velocity = np.zeros((n_cells, n_genes), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            stop = min(start + batch_size, n_cells)
            x = torch.from_numpy(counts[start:stop]).to(device)
            norm = torch.from_numpy(exposure[start:stop]).to(device)
            t = torch.from_numpy(days[start:stop].astype(np.float32)).to(device)
            z, _ = model.enc_z(x, t)
            delta, _ = model.enc_d(z)
            delta = model.d_coeff * delta
            prev = z - delta
            now_expr = log_normalize(model.dec_z(z, t) * norm)
            prev_expr = log_normalize(model.dec_z(prev, t - 1) * norm)
            v = now_expr - prev_expr
            v[t == 0] = 0
            z_all[start:stop] = z.cpu().numpy()
            d_all[start:stop] = delta.cpu().numpy()
            velocity[start:stop] = v.cpu().numpy()
    if not np.isfinite(velocity).all() or not np.isfinite(z_all).all() or not np.isfinite(d_all).all():
        raise FloatingPointError("Inference produced non-finite latent states or velocities")
    valid = days > 0
    if not np.any(np.linalg.norm(velocity[valid], axis=1) > 0):
        raise RuntimeError("All inferred velocities are zero")
    return z_all, d_all, velocity


def build_output(source, gene_mask, days, z, delta, velocity, metadata):
    result = source[:, gene_mask].copy()
    result.obs["Day"] = days
    result.obs["lineagevae_velocity_valid"] = days > 0
    result.var["LineageVAE_used"] = True
    result.X = result.layers["raw_count"].copy()
    sc.pp.normalize_total(result, target_sum=1e4)
    sc.pp.log1p(result)
    if "X_pca" not in result.obsm:
        sc.pp.pca(result, n_comps=min(30, result.n_vars - 1, result.n_obs - 1),
                  random_state=metadata["seed"])
    sc.pp.neighbors(
        result, n_neighbors=min(30, result.n_obs - 1),
        n_pcs=min(30, result.obsm["X_pca"].shape[1]),
        use_rep="X_pca", random_state=metadata["seed"],
    )
    connection = result.obsp["connectivities"].tocsr()
    smooth = connection @ result.X
    smooth = smooth / np.asarray(connection.sum(axis=1))
    result.layers["Ms"] = (
        smooth.toarray().astype(np.float32)
        if sparse.issparse(smooth)
        else np.asarray(smooth, dtype=np.float32)
    )
    result.layers["velocity"] = velocity
    result.obsm["X_lineagevae"] = z
    result.obsm["velocity_lineagevae"] = delta
    result.uns["lineagevae_run"] = metadata
    return result


def add_benchmark_velocity_outputs(result, n_jobs=4):
    """Save the gene-space graph, UMAP velocity, and plotting pseudotime."""
    import scvelo as scv

    scv.tl.velocity_graph(result, vkey="velocity", xkey="Ms", n_jobs=n_jobs)
    scv.tl.velocity_embedding(result, basis="umap", vkey="velocity")
    embedded = np.asarray(result.obsm["velocity_umap"], dtype=np.float64).copy()
    nonfinite = ~np.isfinite(embedded).all(axis=1)
    if nonfinite.any():
        # scVelo's projection takes the mean of an empty transition row. The
        # model deliberately assigns zero velocity at Day0, so those cells
        # have no outgoing graph edges and receive NaN rather than zero.
        zero_velocity = np.all(np.asarray(result.layers["velocity"]) == 0, axis=1)
        graph = result.uns["velocity_graph"].tocsr()
        graph_neg = result.uns["velocity_graph_neg"].tocsr()
        no_edges = (np.diff(graph.indptr) == 0) & (np.diff(graph_neg.indptr) == 0)
        unexpected = nonfinite & ~(zero_velocity & no_edges)
        if unexpected.any():
            raise ValueError(
                f"Non-finite UMAP velocity in {int(unexpected.sum())} cells with "
                "nonzero model velocity or outgoing graph edges"
            )
        embedded[nonfinite] = 0
        result.obsm["velocity_umap"] = embedded.astype(np.float32)
        result.uns["lineagevae_run"]["zero_velocity_embedding_rows"] = int(nonfinite.sum())
    scv.tl.velocity_pseudotime(result, vkey="velocity")
    pseudotime = result.obs["velocity_pseudotime"].to_numpy(dtype=float)
    span = np.ptp(pseudotime)
    if not np.isfinite(pseudotime).all() or span == 0:
        raise ValueError("LineageVAE velocity pseudotime is non-finite or constant")
    result.obs["lineagevae_pseudotime_normalized"] = (
        (pseudotime - pseudotime.min()) / span
    ).astype(np.float32)
    add_embedding_velocity_graph(
        result, vkey="velocity_umap", velocity_key="velocity_umap"
    )
    result.uns["lineagevae_run"]["benchmark_velocity_graph"] = (
        "uns['velocity_graph']: scVelo gene-space graph from layers['velocity'] and layers['Ms']"
    )


def run(args: argparse.Namespace, source: ad.AnnData):
    if min(args.embed_epochs, args.dynamics_epochs, args.embed_batch_size) < 1:
        raise ValueError("Training epochs and embedding batch size must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"Training {args.dataset_id} on {device}", flush=True)
    days, lineages = validate_input(source, args.clone_batch_size)
    mask = select_genes(source, args.genes)
    selected = source.layers["raw_count"][:, mask]
    counts = selected.toarray().astype(np.float32) if sparse.issparse(selected) else np.asarray(selected, dtype=np.float32)
    exposure = normalization_matrix(counts)
    eligible = clone_indices(
        lineages, args.clone_batch_size, days, getattr(args, "required_day", None),
        getattr(args, "min_distinct_days", 1),
    )
    if len(eligible) < 2:
        raise ValueError("Fewer than two eligible lineage groups for dynamics training")
    if any(len(np.unique(days[indices])) > args.clone_batch_size for indices in eligible.values()):
        raise ValueError("A lineage group has more observed days than the clone batch size")
    print(f"Cells={len(days)}, genes={counts.shape[1]}, eligible clones={len(eligible)}", flush=True)
    params = dict(
        x_dim=counts.shape[1], z_dim=10, enc_z_h_dim=50, dec_z_h_dim=50,
        enc_d_h_dim=50, num_enc_z_layers=2, num_dec_z_layers=2,
        num_enc_d_layers=2, t_num=int(days.max()), undifferentiated=None,
        kinetics=False,
    )
    model = LineageVAE(**params).to(device)
    model.device = device
    # The upstream model allocates d_coeff with torch.Tensor(1) but never initializes it.
    with torch.no_grad():
        model.d_coeff.fill_(0.05)
    embedding_loss = train_embedding(model, counts, exposure, days, args, device, rng)
    dynamics_loss, validation_loss, training_clones, validation_clones = train_dynamics(
        model, counts, exposure, days, lineages, args, device, rng
    )
    if not torch.isfinite(model.d_coeff).all():
        raise FloatingPointError("Non-finite trained d_coeff")
    z, delta, velocity = infer_velocity(model, counts, exposure, days, device, args.embed_batch_size)
    metadata = dict(
        model="LineageVAE original architecture, kinetics=False",
        velocity_definition="log1p normalized decoded expression at z(t) minus z(t-1)",
        time_unit="one numeric Day unit", genes=int(args.genes), seed=int(args.seed),
        clone_batch_size=int(args.clone_batch_size), embedding_epochs=int(args.embed_epochs),
        dynamics_epochs=int(args.dynamics_epochs), embedding_loss=np.asarray(embedding_loss),
        dynamics_loss=np.asarray(dynamics_loss), validation_loss=np.asarray(validation_loss),
        training_clones=training_clones,
        validation_clones=validation_clones, d_coeff=float(model.d_coeff.item()),
        source_gene_count=int(source.n_vars),
    )
    metadata.update(getattr(args, "metadata", {}))
    metadata["eligible_clone_count"] = int(len(eligible))
    metadata["eligible_clone_cell_count"] = int(sum(len(indices) for indices in eligible.values()))
    result = build_output(source, mask, days, z, delta, velocity, metadata)
    if args.checkpoint is not None:
        if args.checkpoint.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {args.checkpoint}")
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "model_params": params,
                    "gene_names": result.var_names.tolist()}, args.checkpoint)
    add_benchmark_velocity_outputs(result, n_jobs=args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite partial output: {temporary}")
    result.write_h5ad(temporary, compression="lzf")
    temporary.replace(args.output)
    print(f"Wrote {args.output}: {result.n_obs} cells x {result.n_vars} genes", flush=True)
    return result


def plot_result(result, annotation_key, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scvelo as scv

    if annotation_key not in result.obs:
        raise KeyError(f"Missing obs[{annotation_key!r}] for plotting")
    destination.mkdir(parents=True, exist_ok=True)
    plots = (
        ("embedding", lambda: scv.pl.scatter(result, basis="umap", color=annotation_key, show=False)),
        ("stream", lambda: scv.pl.velocity_embedding_stream(
            result, basis="umap", vkey="velocity", color=annotation_key, show=False)),
        ("grid", lambda: scv.pl.velocity_embedding_grid(
            result, basis="umap", vkey="velocity", color=annotation_key, show=False)),
        ("pseudotime", lambda: scv.pl.scatter(
            result, basis="umap", color="lineagevae_pseudotime_normalized", show=False)),
    )
    for name, draw in plots:
        draw()
        plt.savefig(destination / f"{name}.png", dpi=400, bbox_inches="tight")
        plt.close()


def main() -> None:
    args = arguments()
    for row in input_rows(args):
        print(f"Reading {row.dataset_id}", flush=True)
        result = run(row, prepare_source(row))
        if args.plot_dir:
            plot_result(result, row.annotation_key, args.plot_dir / row.dataset_id)


if __name__ == "__main__":
    main()
