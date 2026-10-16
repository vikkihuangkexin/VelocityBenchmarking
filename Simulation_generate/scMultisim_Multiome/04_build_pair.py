#!/usr/bin/env python3
"""Build paired RNA/ATAC h5ad files and all reference figures."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from common import (
    build_differentiation_ground_truth,
    dataframe_to_uns,
    read_dense_f32,
    read_mtx,
    read_names,
    read_table_gz,
    write_json,
)


def log_normalize(counts: sparse.csr_matrix, target_sum: float = 1e4) -> sparse.csr_matrix:
    output = counts.astype(np.float32).tocsr(copy=True)
    totals = np.asarray(output.sum(axis=1)).ravel()
    scales = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    output = sparse.diags(scales.astype(np.float32)) @ output
    output.data = np.log1p(output.data)
    return output.tocsr()


def make_knn_graph(pcs: np.ndarray, n_neighbors: int = 15):
    n_neighbors = min(n_neighbors, len(pcs) - 1)
    model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(pcs)
    distances, indices = model.kneighbors(pcs)
    distances, indices = distances[:, 1:], indices[:, 1:]
    rows = np.repeat(np.arange(len(pcs)), n_neighbors)
    cols = indices.ravel()
    vals_d = distances.ravel().astype(np.float32)
    scale = np.median(distances, axis=1)
    scale[scale <= 0] = 1.0
    vals_c = np.exp(-(distances**2) / (2 * np.maximum(scale[:, None] ** 2, 1e-12))).ravel()
    dist = sparse.csr_matrix((vals_d, (rows, cols)), shape=(len(pcs), len(pcs))).maximum(
        sparse.csr_matrix((vals_d, (rows, cols)), shape=(len(pcs), len(pcs))).T
    )
    conn_directed = sparse.csr_matrix(
        (vals_c.astype(np.float32), (rows, cols)), shape=(len(pcs), len(pcs))
    )
    conn = conn_directed.maximum(conn_directed.T)
    return indices, conn.tocsr(), dist.tocsr()


def multivelo_tfidf_smooth(
    chromatin: sparse.csr_matrix, connectivities: sparse.csr_matrix
) -> sparse.csr_matrix:
    matrix = chromatin.astype(np.float64).tocsr()
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    tf = sparse.diags(np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)) @ matrix
    feature_totals = np.asarray(matrix.sum(axis=0)).ravel()
    idf = np.zeros_like(feature_totals)
    nonzero = feature_totals > 0
    idf[nonzero] = np.log1p(matrix.shape[0] / feature_totals[nonzero])
    tfidf = (tf @ sparse.diags(idf) * 1e4).tocsr()
    conn = connectivities.astype(np.float64).tocsr(copy=True)
    conn.setdiag(1.0)
    row_sums = np.asarray(conn.sum(axis=1)).ravel()
    normalized = sparse.diags(1.0 / np.maximum(row_sums, 1e-12)) @ conn
    return (normalized @ tfidf).astype(np.float32).tocsr()


def project_velocity(
    spliced: sparse.csr_matrix,
    velocity: np.ndarray,
    components: np.ndarray,
    pcs: np.ndarray,
    embedding: np.ndarray,
    neighbors: np.ndarray,
    true_time: np.ndarray,
) -> np.ndarray:
    expression = spliced.toarray().astype(np.float32)
    velocity_log = velocity / np.maximum(1.0 + expression, 1.0)
    velocity_pcs = velocity_log @ components.T.astype(np.float32)
    output = np.zeros((len(pcs), 2), dtype=np.float32)
    for i, js in enumerate(neighbors):
        delta = pcs[js] - pcs[i]
        v = velocity_pcs[i]
        denom = np.linalg.norm(delta, axis=1) * max(np.linalg.norm(v), 1e-12)
        cosine = np.divide(delta @ v, denom, out=np.zeros_like(denom), where=denom > 0)
        forward = true_time[js] >= true_time[i] - 1e-12
        weights = np.exp(5 * np.clip(cosine, -1, 1)) * forward
        if not np.any(weights > 0):
            weights = np.exp(5 * np.clip(cosine, -1, 1))
        weights /= max(weights.sum(), 1e-12)
        output[i] = np.sum((embedding[js] - embedding[i]) * weights[:, None], axis=0)
    scale = np.quantile(np.linalg.norm(output, axis=1), 0.95)
    if np.isfinite(scale) and scale > 0:
        output /= scale
    return output


def velocity_grid(embedding: np.ndarray, velocity: np.ndarray, n_grid: int = 36):
    gx = np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), n_grid)
    gy = np.linspace(embedding[:, 1].min(), embedding[:, 1].max(), n_grid)
    xx, yy = np.meshgrid(gx, gy)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    sigma = 0.06 * max(np.ptp(embedding[:, 0]), np.ptp(embedding[:, 1]), 1.0)
    dist2 = ((points[:, None] - embedding[None, :]) ** 2).sum(axis=2)
    weights = np.exp(-dist2 / (2 * sigma**2))
    density = weights.sum(axis=1)
    grid = weights @ velocity / np.maximum(density[:, None], 1e-12)
    grid[density < np.quantile(density, 0.50)] = 0
    return xx, yy, grid[:, 0].reshape(xx.shape), grid[:, 1].reshape(xx.shape), density.reshape(xx.shape)


def colors(values: pd.Series):
    categories = sorted(values.astype(str).unique())
    cmap = plt.get_cmap("tab10" if len(categories) <= 10 else "tab20")
    return categories, {x: cmap(i % cmap.N) for i, x in enumerate(categories)}


def plot_cells(ax, embedding, values, categories, palette, size=10, alpha=0.8):
    array = values.astype(str).to_numpy()
    for category in categories:
        mask = array == category
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1], s=size, alpha=alpha,
            color=palette[category], linewidths=0, label=category, zorder=1,
        )


def save(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def finite_matrix(value) -> bool:
    return np.isfinite(value.data if sparse.issparse(value) else np.asarray(value)).all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir).resolve()
    intermediate = outdir / "intermediate"
    figures = outdir / "figures"
    figures.mkdir(exist_ok=True)

    genes = read_names(intermediate / "gene_names.tsv")
    cells = read_names(intermediate / "cell_names.tsv")
    peaks = read_names(intermediate / "region_names.tsv")
    tfs = read_names(intermediate / "tf_names.tsv")
    spliced = read_mtx(intermediate / "spliced_genes_cells.mtx.gz").T.tocsr()
    unspliced = read_mtx(intermediate / "unspliced_genes_cells.mtx.gz").T.tocsr()
    peak_counts = read_mtx(intermediate / "atac_counts_regions_cells.mtx.gz").T.tocsr()
    peak_true = read_mtx(intermediate / "atac_true_regions_cells.mtx.gz").T.tocsr()
    region_to_gene = read_mtx(intermediate / "region_to_gene_regions_genes.mtx.gz", np.int8)
    region_to_tf = read_mtx(intermediate / "region_to_tf_regions_tfs.mtx.gz", np.int8)
    velocity = read_dense_f32(
        intermediate / "true_velocity_genes_cells.f32", (len(genes), len(cells))
    ).T.copy()
    if spliced.shape != (len(cells), len(genes)) or peak_counts.shape != (len(cells), len(peaks)):
        raise RuntimeError("Core matrix shapes do not match names")

    obs = pd.read_csv(intermediate / "cell_metadata.tsv.gz", sep="\t", index_col=0).loc[cells]
    gene_meta = pd.read_csv(intermediate / "gene_metadata.tsv.gz", sep="\t").set_index("gene").loc[genes]
    peak_meta = pd.read_csv(intermediate / "peak_metadata.tsv.gz", sep="\t").set_index("region").loc[peaks]
    obs["true_cell_type"] = obs["true_cell_type"].astype(str)
    obs["true_lineage"] = obs["true_lineage"].astype(str)
    true_time = pd.to_numeric(obs["true_time"], errors="raise").to_numpy()
    order, branch_pt, differentiation = build_differentiation_ground_truth(
        obs["true_cell_type"], true_time
    )
    obs["true_branch"] = obs["true_cell_type"].to_numpy()
    obs["true_cell_type_order"] = order
    obs["true_branch_pseudotime"] = branch_pt
    obs["velocity_ground_truth"] = obs["true_lineage"]
    obs["velocity_lineage"] = obs["true_branch"]
    obs["velocity_stage"] = obs["true_cell_type"]
    obs["velocity_stage_order"] = obs["true_cell_type_order"]

    # Verify that every peak is on the chromosome of its linked RNA gene.
    linked_chrom = gene_meta.loc[peak_meta["linked_gene"], "chrom"].to_numpy()
    if not np.array_equal(linked_chrom.astype(str), peak_meta["chrom"].astype(str).to_numpy()):
        raise RuntimeError("RNA gene and linked ATAC peak chromosomes do not match")

    log_expression = log_normalize(spliced)
    n_components = min(30, len(cells) - 1, len(genes) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=args.seed)
    pcs = svd.fit_transform(log_expression).astype(np.float32)
    neighbors, connectivities, distances = make_knn_graph(pcs)
    perplexity = min(30.0, max(5.0, (len(cells) - 1) / 10.0))
    embedding = TSNE(
        n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(pcs).astype(np.float32)
    velocity_2d = project_velocity(
        spliced, velocity, svd.components_, pcs, embedding, neighbors, true_time
    )
    xx, yy, grid_u, grid_v, density = velocity_grid(embedding, velocity_2d)

    categories, palette = colors(obs["true_cell_type"])
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    plot_cells(ax, embedding, obs["true_cell_type"], categories, palette, size=11)
    ax.set(title=f"{args.dataset_name}: reference tSNE", xlabel="tSNE 1", ylabel="tSNE 2")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, figures / "01_reference_tsne_celltype")

    # Cell-colored points with short, thick, black arrows placed directly on cells.
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    plot_cells(ax, embedding, obs["true_cell_type"], categories, palette, size=14, alpha=0.78)
    rng = np.random.default_rng(args.seed)
    arrow_count = min(300, len(cells))
    arrow_idx = np.sort(rng.choice(len(cells), size=arrow_count, replace=False))
    ax.quiver(
        embedding[arrow_idx, 0], embedding[arrow_idx, 1],
        velocity_2d[arrow_idx, 0], velocity_2d[arrow_idx, 1],
        color="black", angles="xy", scale_units="xy", scale=0.35,
        width=0.0042, headwidth=4.5, headlength=5.0, headaxislength=4.5,
        alpha=0.90, zorder=3,
    )
    ax.set(title="Ground-truth RNA velocity stream", xlabel="tSNE 1", ylabel="tSNE 2")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, figures / "02_true_velocity_stream_tsne")

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    plot_cells(ax, embedding, obs["true_cell_type"], categories, palette, size=9, alpha=0.62)
    valid = density > np.quantile(density, 0.58)
    ax.quiver(
        xx[valid], yy[valid], grid_u[valid], grid_v[valid], color="black",
        angles="xy", scale_units="xy", scale=0.30, width=0.0038,
        headwidth=4.2, headlength=4.8, alpha=0.88, zorder=3,
    )
    ax.set(title="Ground-truth RNA velocity grid", xlabel="tSNE 1", ylabel="tSNE 2")
    ax.legend(frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, figures / "03_true_velocity_grid_tsne")

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=obs["true_pseudotime"].to_numpy(float),
        cmap="plasma", s=12, alpha=0.85, linewidths=0,
    )
    fig.colorbar(scatter, ax=ax, label="True pseudotime")
    ax.set(title=f"{args.dataset_name}: ground-truth pseudotime", xlabel="tSNE 1", ylabel="tSNE 2")
    save(fig, figures / "04_true_pseudotime_tsne")

    reconstructed_chromatin = (peak_counts @ region_to_gene).astype(np.float32).tocsr()
    stored_chromatin = read_mtx(intermediate / "chromatin_genes_cells.mtx.gz").T.tocsr()
    difference = reconstructed_chromatin - stored_chromatin
    difference.eliminate_zeros()
    if difference.nnz:
        raise RuntimeError(f"chromatin aggregation mismatch: difference_nnz={difference.nnz}")
    mc = multivelo_tfidf_smooth(reconstructed_chromatin, connectivities)

    factors = np.load(intermediate / "cistopic_factors.npz")
    theta = factors["theta_cells_topics"].astype(np.float32)
    psi = factors["psi_topics_peaks"].astype(np.float32)
    cistopic_reconstructed = (theta @ psi).astype(np.float32)
    if cistopic_reconstructed.shape != peak_counts.shape:
        raise RuntimeError(
            f"obsm['cisTopic'] must be cells x peaks; got {cistopic_reconstructed.shape}, expected {peak_counts.shape}"
        )

    observed_diff = read_table_gz(intermediate / "sckinetics_diff_peaks.tsv.gz")
    true_diff = read_table_gz(intermediate / "true_differential_peaks.tsv.gz")
    observed_diff["selected"] = observed_diff["selected"].astype(str).str.lower().eq("true")
    true_diff["selected"] = true_diff["selected"].astype(str).str.lower().eq("true")
    observed_union = set(observed_diff.loc[observed_diff.selected, "region"].astype(str))
    true_union = set(true_diff.loc[true_diff.selected, "region"].astype(str))
    peak_meta["sckinetics_diff_peak"] = peak_meta.index.to_series().isin(observed_union).to_numpy()
    peak_meta["true_diff_peak"] = peak_meta.index.to_series().isin(true_union).to_numpy()
    peak_meta["n_counts"] = np.asarray(peak_counts.sum(axis=0)).ravel().astype(np.float32)
    peak_meta["n_cells"] = np.asarray((peak_counts > 0).sum(axis=0)).ravel().astype(np.int32)

    chromvar_status = pd.read_csv(intermediate / "chromvar_status.tsv.gz", sep="\t").iloc[0]
    chromvar = None
    chromvar_names = np.asarray([], dtype=str)
    if str(chromvar_status["available"]).lower() == "true":
        chromvar_names = read_names(intermediate / "chromvar_motif_names.tsv")
        chromvar = read_dense_f32(
            intermediate / "chromvar_cells_motifs.f32", (len(cells), len(chromvar_names))
        )

    gene_var = gene_meta.copy()
    gene_var["protein_coding"] = True
    gene_var["highly_variable"] = True
    gene_var["multivelo_gene"] = True
    gene_var["coordinate_system"] = "synthetic_human_like_v1"

    simulation = {
        "simulator": "scMultiSim",
        "dataset_name": args.dataset_name,
        "topology": args.topology,
        "seed": args.seed,
        "coordinate_system": "synthetic_human_like_v1",
        "coordinates_are_real_genome": False,
        "chromatin_definition": "ATAC peak counts @ region_to_gene",
        "Mc_definition": "MultiVelo TF-IDF followed by RNA-KNN smoothing",
        "chromVAR_mode": str(chromvar_status["mode"]),
        "chromVAR_available": chromvar is not None,
        "chromVAR_reason": str(chromvar_status["reason"]),
        "cisTopic_obsm_semantics": "cells_x_peaks reconstructed as theta @ psi",
        "cisTopic_topics_obsm_semantics": "cells_x_topics theta",
    }
    simulation_manifest = read_table_gz(intermediate / "simulation_manifest.tsv.gz").fillna("")
    simulation["run_metadata"] = simulation_manifest.iloc[0].to_dict()
    simulation["python_packages"] = {
        name: version(name) for name in
        ("numpy", "pandas", "scipy", "anndata", "matplotlib", "scikit-learn", "h5py")
    }

    rna = ad.AnnData(X=log_expression, obs=obs.copy(), var=gene_var)
    rna.layers["spliced"] = spliced.astype(np.float32)
    rna.layers["unspliced"] = unspliced.astype(np.float32)
    rna.layers["true_velocity"] = velocity.astype(np.float32)
    rna.layers["chromatin"] = reconstructed_chromatin.copy()
    rna.layers["atac_counts"] = reconstructed_chromatin.copy()
    rna.layers["Mc"] = mc
    rna.obsm["X_pca"] = pcs
    rna.obsm["X_tsne"] = embedding
    rna.obsm["true_velocity_tsne"] = velocity_2d
    rna.obsp["connectivities"] = connectivities
    rna.obsp["distances"] = distances
    rna.uns["differentiation_ground_truth"] = differentiation
    rna.uns["simulation"] = simulation

    atac = ad.AnnData(X=peak_counts.copy(), obs=obs.copy(), var=peak_meta.copy())
    atac.layers["atac_counts"] = peak_counts.copy()
    atac.layers["true_accessibility"] = peak_true.copy()
    atac.obsm["cisTopic"] = cistopic_reconstructed
    atac.obsm["cisTopic_topics"] = theta
    if chromvar is not None:
        atac.obsm["chromVAR"] = chromvar.astype(np.float32)
    atac.obsm["X_tsne"] = embedding
    atac.obsm["true_velocity_tsne"] = velocity_2d
    atac.varm["region_to_gene"] = region_to_gene
    atac.varm["region_to_tf"] = region_to_tf
    atac.uns["region_to_gene_genes"] = genes
    atac.uns["region_to_tf_names"] = tfs
    atac.uns["cisTopic_topics"] = np.asarray(
        [f"Topic_{i + 1:02d}" for i in range(theta.shape[1])], dtype=str
    )
    atac.uns["cisTopic_topic_peak"] = psi
    atac.uns["chromVAR_features"] = chromvar_names
    atac.uns["chromVAR_status"] = {
        "available": chromvar is not None,
        "mode": str(chromvar_status["mode"]),
        "reason": str(chromvar_status["reason"]),
    }
    atac.uns["sckinetics_diff_peaks"] = dataframe_to_uns(observed_diff)
    atac.uns["true_differential_peaks"] = dataframe_to_uns(true_diff)
    atac.uns["differentiation_ground_truth"] = differentiation
    atac.uns["simulation"] = simulation

    rna_path = outdir / f"{args.dataset_name}_rna.h5ad"
    atac_path = outdir / f"{args.dataset_name}_atac.h5ad"
    rna.uns["modality"] = "RNA"
    rna.uns["paired_atac_h5ad"] = atac_path.name
    atac.uns["modality"] = "ATAC"
    atac.uns["paired_rna_h5ad"] = rna_path.name
    rna.write_h5ad(rna_path, compression="gzip")
    atac.write_h5ad(atac_path, compression="gzip")

    # Reopen and enforce the contract that downstream benchmark code relies on.
    rna_check = ad.read_h5ad(rna_path)
    atac_check = ad.read_h5ad(atac_path)
    if not np.array_equal(rna_check.obs_names, atac_check.obs_names):
        raise AssertionError("RNA/ATAC cells or ordering differ")
    if (rna_check.layers["chromatin"] != rna_check.layers["atac_counts"]).nnz:
        raise AssertionError("RNA chromatin != atac_counts")
    if atac_check.obsm["cisTopic"].shape != atac_check.shape:
        raise AssertionError("ATAC obsm['cisTopic'] is not cells x peaks")
    if atac_check.obsm["cisTopic_topics"].shape != (atac_check.n_obs, theta.shape[1]):
        raise AssertionError("cisTopic_topics does not have cells x topics shape")
    reopened_reconstruction = (
        atac_check.layers["atac_counts"] @ atac_check.varm["region_to_gene"]
    ).tocsr()
    reopened_difference = reopened_reconstruction - rna_check.layers["chromatin"]
    reopened_difference.eliminate_zeros()
    if reopened_difference.nnz:
        raise AssertionError(
            "Reopened ATAC peak counts @ region_to_gene != RNA chromatin: "
            f"difference_nnz={reopened_difference.nnz}"
        )
    if {"chromvar", "cistopic"}.intersection(atac_check.obsm.keys()):
        raise AssertionError("Forbidden lowercase aliases found")
    if not finite_matrix(atac_check.obsm["cisTopic"]) or not finite_matrix(rna_check.layers["Mc"]):
        raise AssertionError("Non-finite cisTopic or Mc")
    if chromvar is not None and not finite_matrix(atac_check.obsm["chromVAR"]):
        raise AssertionError("Non-finite chromVAR")
    for key in ("sckinetics_diff_peak", "true_diff_peak"):
        if key not in atac_check.var:
            raise AssertionError(f"Missing atac.var[{key!r}]")
    for key in ("sckinetics_diff_peaks", "true_differential_peaks"):
        if key not in atac_check.uns:
            raise AssertionError(f"Missing atac.uns[{key!r}]")
    if "lineage_paths" not in atac_check.uns["differentiation_ground_truth"]:
        raise AssertionError("Missing differentiation lineage_paths")

    audit = {
        "dataset": args.dataset_name,
        "rna_h5ad": rna_path.name,
        "atac_h5ad": atac_path.name,
        "rna_shape_cells_genes": list(map(int, rna_check.shape)),
        "atac_shape_cells_peaks": list(map(int, atac_check.shape)),
        "paired_cells_exact_order": True,
        "rna_atac_linked_chromosomes_match": True,
        "chromatin_equals_atac_counts": True,
        "peak_to_gene_aggregation_difference_nnz": 0,
        "cisTopic_shape": list(map(int, atac_check.obsm["cisTopic"].shape)),
        "cisTopic_semantics": "cells_x_peaks",
        "cisTopic_topics_shape": list(map(int, atac_check.obsm["cisTopic_topics"].shape)),
        "chromVAR_available": "chromVAR" in atac_check.obsm,
        "chromVAR_shape": list(map(int, atac_check.obsm["chromVAR"].shape)) if "chromVAR" in atac_check.obsm else None,
        "n_sckinetics_diff_peaks": int(atac_check.var["sckinetics_diff_peak"].sum()),
        "n_true_diff_peaks": int(atac_check.var["true_diff_peak"].sum()),
        "lineage_paths": list(map(str, atac_check.uns["differentiation_ground_truth"]["lineage_paths"])),
        "figures_png_pdf": sorted(x.name for x in figures.glob("*.*")),
    }
    write_json(outdir / f"{args.dataset_name}_validation_audit.json", audit)
    # run.py publishes _SUCCESS after optional postprocessing and final validation.
    print(json.dumps(audit, indent=2))
    print("PAIR BUILD AND VALIDATION: PASSED")


if __name__ == "__main__":
    main()
