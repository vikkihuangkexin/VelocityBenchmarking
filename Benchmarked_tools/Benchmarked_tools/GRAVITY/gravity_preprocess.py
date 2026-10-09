#!/usr/bin/env python3
"""
GRAVITY preprocessing (CPU only) for simulated data.

NOTE: Only used for simulated data inside the Docker image.

This script performs the CPU-heavy stages that precede deep-learning training:

    1) Read an h5ad, run filter_and_normalize / pca / neighbors / moments and
       export the long-format table ``<output-dir>/<ID>/cell_type_u_s.csv``.
    2) Aggregate the long table into the wide training table
       ``<output-dir>/<ID>/combine.csv``.

It deliberately does not import torch / pynvml / the GRAVITY training pipeline,
so it can run on CPU-only nodes. The produced ``cell_type_u_s.csv`` and
``combine.csv`` are the only inputs needed by the training stage; when both
already exist, the training script skips the corresponding step automatically.

Parallelism:
  - The upstream scanpy/scvelo preprocessing steps do not accept ``n_jobs`` and
    stay single-threaded.
  - GRAVITY's own ``adata_to_df_with_embed`` (per gene) and ``preprocess_counts``
    (per cell + iterrows) have no parallel option, so this script provides fork
    based multi-process versions controlled by ``--n-jobs`` (on Linux the AnnData
    object and long table are shared copy-on-write, so memory is not duplicated).

Usage:
    python gravity_preprocess.py --input samples.csv --output-dir ./out --simulate
    python gravity_preprocess.py --input samples.csv --output-dir ./out --simulate --n-jobs 8
"""

import argparse
import multiprocessing
import os
import traceback

# --- Compatibility patch: scVelo 0.2.5 relies on matplotlib.cbook.mplDeprecation,
#     which was removed in matplotlib >= 3.7. Add it before importing scVelo.
import matplotlib

matplotlib.use("Agg")
import matplotlib.cbook as _cbook

if not hasattr(_cbook, "mplDeprecation"):
    _cbook.mplDeprecation = DeprecationWarning

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv

from gravity.data.preprocessing import adata_to_df_with_embed, preprocess_counts

# ---- fork worker global contexts (shared copy-on-write, not pickled) ----
_EXPORT_CTX = {}   # holds adata / us_para / pos_map
_AGG_CTX = {}      # holds the dict grouped by cellIndex


def _csv_is_complete(csv_path):
    """Check that a long table CSV is complete (contains the cell metadata columns)."""
    required = {"gene_name", "unsplice", "splice", "cellID", "embedding1", "embedding2"}
    try:
        with open(csv_path) as handle:
            header = handle.readline().strip().split(",")
        return required.issubset(set(header))
    except Exception:
        return False


def pick_celltype_key(adata):
    """Cluster column name for simulated data (dyngen/scmultisim use ``milestone``)."""
    for key in ("milestone", "vis_annotation", "celltype", "cell_type"):
        if key in adata.obs:
            return key
    raise KeyError(f"No cluster column found in adata.obs; actual columns: {list(adata.obs.columns)}")


def preprocess_adata(adata, data_id, n_top_genes=2000, n_pcs=30, n_neighbors=30):
    """Preprocess simulated data (matches the training script, no parallel speedup).

    Copies ``X_dimred`` into ``X_umap`` when present, applies
    filter_and_normalize with ``min_shared_counts=None`` and a dynamic gene
    count, then pca / neighbors / moments.
    """
    adata.obs_names_make_unique()

    if "X_dimred" in adata.obsm:
        adata.obsm["X_umap"] = adata.obsm["X_dimred"].copy()
    elif "X_umap" not in adata.obsm:
        try:
            sc.tl.pca(adata)
            sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
            sc.tl.umap(adata)
        except Exception as exc:
            print(f"[WARN] {data_id} could not generate a 2D embedding: {exc}")

    if adata.n_vars < 10000:
        top_gene = (adata.n_vars // 500) * 500
        top_gene = min(top_gene, adata.n_vars, 500)
    else:
        top_gene = n_top_genes

    scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=top_gene)
    print(f"[INFO] {data_id} after filter_and_normalize: {adata.n_vars} genes", flush=True)

    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors, method="umap")
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    print(f"[INFO] {data_id} after moments: {adata.n_vars} genes", flush=True)
    return adata


# ===========================================================================
#  Parallel version 1: long table export (replaces adata_to_df_with_embed)
# ===========================================================================
def _export_gene_chunk(gene_chunk):
    """Worker: export a batch of genes as (gene_name, unsplice, splice) rows."""
    adata = _EXPORT_CTX["adata"]
    us = _EXPORT_CTX["us_para"]
    pos_map = _EXPORT_CTX["pos_map"]
    parts = []
    for gene in gene_chunk:
        pos = pos_map[gene]
        sub = adata[:, [pos]]
        u = np.asarray(sub.layers[us[0]][:, 0], dtype=np.float32)
        s = np.asarray(sub.layers[us[1]][:, 0], dtype=np.float32)
        parts.append(pd.DataFrame({"gene_name": gene, "unsplice": u, "splice": s}, copy=False))
    return pd.concat(parts, ignore_index=True) if parts else \
        pd.DataFrame(columns=["gene_name", "unsplice", "splice"])


def export_long_table_parallel(adata, save_path, celltype_key,
                               us_para=("Mu", "Ms"), embed_para="X_umap",
                               n_jobs=None):
    """Export the long table in parallel (column order matches adata_to_df_with_embed)."""
    gene_list = list(adata.var.index)
    n_genes = len(gene_list)
    n_cells = adata.n_obs

    n_workers = n_jobs or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, n_genes))

    if n_workers <= 1:
        adata_to_df_with_embed(
            adata, us_para=list(us_para), cell_type_para=celltype_key,
            embed_para=embed_para, save_path=save_path)
        return

    pos_map = {g: i for i, g in enumerate(adata.var.index)}

    _EXPORT_CTX["adata"] = adata
    _EXPORT_CTX["us_para"] = us_para
    _EXPORT_CTX["pos_map"] = pos_map

    chunks = np.array_split(np.arange(n_genes), n_workers)
    chunk_gene_lists = [[gene_list[i] for i in c] for c in chunks if len(c) > 0]

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        dfs = pool.map(_export_gene_chunk, chunk_gene_lists)

    # Merge the per-gene blocks in gene order (n_genes * n_cells rows, 3 columns).
    gene_df = pd.concat(dfs, ignore_index=True)

    # Cell metadata (repeated once per gene). Row order must align with gene_df:
    # the long table loops genes on the outside and cells on the inside, so the
    # cell sequence is tiled n_genes times.
    cell_id = np.asarray(adata.obs.index)
    clusters = np.asarray(adata.obs[celltype_key])
    emb = np.asarray(adata.obsm[embed_para])
    rep = np.tile(np.arange(n_cells), n_genes)
    meta = pd.DataFrame({
        "cellID": cell_id[rep],
        "clusters": clusters[rep],
        "embedding1": emb[rep, 0],
        "embedding2": emb[rep, 1],
    })

    full = pd.concat([gene_df.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    full.to_csv(save_path, index=False)


# ===========================================================================
#  Parallel version 2: long table -> wide training table (replaces preprocess_counts)
# ===========================================================================
def _agg_cell_chunk(cell_chunk):
    """Worker: aggregate a batch of cells into wide-table rows."""
    groups = _AGG_CTX["groups"]
    records = []
    for cid in cell_chunk:
        subset = groups[cid]
        first = subset.iloc[0]
        row = {
            "cellIndex": first["cellIndex"],
            "embedding1": first["embedding1"],
            "embedding2": first["embedding2"],
        }
        for _, entry in subset.iterrows():
            row[f"gene_u_s_{entry.gene_name}"] = str(
                (entry.gene_name, entry.unsplice, entry.splice))
        records.append(row)
    return pd.DataFrame(records)


def preprocess_counts_parallel(csv_path, output_csv, n_jobs=None):
    """Aggregate the long table into the wide training table (equivalent to preprocess_counts)."""
    if os.path.exists(output_csv):
        print(f"[INFO] {output_csv} already exists, skipping aggregation", flush=True)
        return

    data = pd.read_csv(csv_path)
    if "cellIndex" not in data.columns:
        data.insert(0, "cellIndex", pd.factorize(data["cellID"])[0])
    cols = ["cellIndex", "gene_name", "unsplice", "splice", "embedding1", "embedding2"]
    data = data[cols]

    # Group once by cellIndex (O(n_rows)); each cell then takes its own subset.
    groups = {cid: grp for cid, grp in data.groupby("cellIndex", sort=False)}
    cell_ids = list(groups.keys())

    n_workers = n_jobs or os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(cell_ids)))

    if n_workers <= 1:
        preprocess_counts(csv_path, output_csv, gene_order=None)
        return

    _AGG_CTX["groups"] = groups
    chunks = np.array_split(np.arange(len(cell_ids)), n_workers)
    chunk_cell_lists = [[cell_ids[i] for i in c] for c in chunks if len(c) > 0]

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        dfs = pool.map(_agg_cell_chunk, chunk_cell_lists)

    combined = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    combined.to_csv(output_csv, index=False)


def process_one(data_id, data_path, save_dir, n_jobs=None, force=False):
    """CPU preprocessing: h5ad -> cell_type_u_s.csv -> combine.csv. Returns 'skip'/'ok'/'error'."""
    csv_target = os.path.join(save_dir, "cell_type_u_s.csv")
    combine_target = os.path.join(save_dir, "combine.csv")

    # ---- Stage 1: h5ad -> long table ----
    if os.path.exists(csv_target) and _csv_is_complete(csv_target) and not force:
        print(f"[skip] {data_id} long table already exists, skipping", flush=True)
    else:
        os.makedirs(save_dir, exist_ok=True)
        adata = sc.read(data_path, cache=True)
        if adata.n_vars == 0 or adata.n_obs == 0:
            print(f"{data_id} shape error!", flush=True)
            return "error"
        adata = preprocess_adata(adata, data_id)
        celltype_key = pick_celltype_key(adata)
        export_long_table_parallel(
            adata, csv_target, celltype_key,
            us_para=("Mu", "Ms"), embed_para="X_umap", n_jobs=n_jobs)
        print(f"[OK] {data_id} long table exported: {csv_target}", flush=True)

    # ---- Stage 2: long table -> wide training table combine.csv ----
    preprocess_counts_parallel(csv_target, combine_target, n_jobs=n_jobs)
    print(f"[OK] {data_id} training table ready: {combine_target}", flush=True)
    return "ok"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GRAVITY simulated-data CPU preprocessing (multi-core)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Sample list CSV with columns 'ID' and 'path'")
    parser.add_argument("--output-dir", required=True, help="Result save root directory")
    parser.add_argument("--start", type=int, default=None, help="Start row index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End row index (exclusive)")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Number of parallel workers (0 = all CPU cores, 1 = serial, >1 = explicit count)",
    )
    parser.add_argument("--force", action="store_true", help="Ignore existing outputs and re-export")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Confirm simulated-data processing (required; this script is simulated-data only)",
    )
    return parser


def main(args=None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if not args.simulate:
        parser.error("gravity_preprocess.py only supports simulated data; pass --simulate")

    np.random.seed(args.seed)

    n_jobs = args.n_jobs if args.n_jobs > 0 else None
    print(f"Sample list: {args.input}", flush=True)
    print(f"Save root: {args.output_dir}", flush=True)
    print(f"Parallel workers: {n_jobs or os.cpu_count()}", flush=True)

    datalist = pd.read_csv(args.input)
    ok = skip = err = 0
    for i, row in datalist.iterrows():
        if args.start is not None and i < args.start:
            continue
        if args.end is not None and i >= args.end:
            break
        data_id = str(row["ID"])
        data_path = str(row["path"])
        save_dir = os.path.join(args.output_dir, data_id)
        print(f"\n===== [{i}] preprocessing: {data_id} =====", flush=True)
        try:
            status = process_one(data_id, data_path, save_dir, n_jobs=n_jobs, force=args.force)
            if status == "skip":
                skip += 1
            elif status == "ok":
                ok += 1
            else:
                err += 1
        except Exception as exc:
            err += 1
            print(f"[FAIL] {data_id}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Preprocessing done: ok {ok} | skipped {skip} | failed {err} | total {len(datalist)}")


if __name__ == "__main__":
    main()
