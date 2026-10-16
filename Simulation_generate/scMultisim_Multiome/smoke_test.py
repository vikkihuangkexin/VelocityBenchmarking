#!/usr/bin/env python3
"""Exercise processing with temporary artificial fixtures, not scientific results."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite
import gzip

from run import load_datasets, select_ids
from validate_pair import assert_equal_matrix, validate_pair

HERE = Path(__file__).resolve().parent


def fixture(outdir: Path, name: str) -> None:
    rng = np.random.default_rng(52)
    n_cells, n_genes, n_peaks, n_tfs, n_topics = 84, 100, 120, 8, 5
    intermediate = outdir / "intermediate"
    intermediate.mkdir(parents=True)
    cells = np.array([f"cell_{i:06d}" for i in range(n_cells)])
    genes = np.array([f"gene_{i:05d}" for i in range(n_genes)])
    primary = np.arange(n_peaks) % n_genes
    gene_start = 100000 + np.arange(n_genes) * 100000
    peak_start = gene_start[primary] - 5000 + (np.arange(n_peaks) // n_genes) * 500
    peaks = np.array([f"chr1:{start}-{start + 399}" for start in peak_start])
    mapping = sparse.csr_matrix((np.ones(n_peaks), (np.arange(n_peaks), primary)),
                                shape=(n_peaks, n_genes))
    tf_map = sparse.csr_matrix((np.ones(n_peaks), (np.arange(n_peaks), np.arange(n_peaks) % n_tfs)),
                               shape=(n_peaks, n_tfs))
    spliced = sparse.csr_matrix(rng.poisson(3, size=(n_genes, n_cells)), dtype=np.float32)
    unspliced = sparse.csr_matrix(rng.poisson(1, size=(n_genes, n_cells)), dtype=np.float32)
    counts = rng.poisson(1, size=(n_peaks, n_cells)).astype(np.float32)
    counts[:15, :28] += 3
    latent = counts + rng.random(counts.shape).astype(np.float32)
    matrices = {
        "spliced_genes_cells": spliced, "unspliced_genes_cells": unspliced,
        "atac_counts_regions_cells": sparse.csr_matrix(counts),
        "atac_true_regions_cells": sparse.csr_matrix(latent),
        "region_to_gene_regions_genes": mapping, "region_to_tf_regions_tfs": tf_map,
        "chromatin_genes_cells": mapping.T @ sparse.csr_matrix(counts),
    }
    for stem, matrix in matrices.items():
        with gzip.open(intermediate / f"{stem}.mtx.gz", "wb") as handle:
            mmwrite(handle, matrix)
    velocity = rng.normal(size=(n_genes, n_cells)).astype("<f4")
    velocity.ravel(order="F").tofile(intermediate / "true_velocity_genes_cells.f32")
    for stem, names in [("cell_names", cells), ("gene_names", genes), ("region_names", peaks),
                        ("tf_names", [f"TF_{i}" for i in range(n_tfs)])]:
        (intermediate / f"{stem}.tsv").write_text("\n".join(names) + "\n")
    labels = np.repeat(["0_1", "1_2", "1_3"], 28)
    times = np.concatenate([np.linspace(0, 0.4, 28), np.linspace(0.5, 1, 28), np.linspace(0.5, 1, 28)])
    pd.DataFrame({"cell": cells, "true_cell_type": labels, "true_lineage": labels,
                  "true_time": times, "true_pseudotime": times}).to_csv(
        intermediate / "cell_metadata.tsv.gz", sep="\t", index=False)
    pd.DataFrame({"gene": genes, "chrom": "chr1", "chromStart": gene_start,
                  "chromEnd": gene_start + 19999, "gene_type": "protein_coding"}).to_csv(
        intermediate / "gene_metadata.tsv.gz", sep="\t", index=False)
    pd.DataFrame({"region": peaks, "chrom": "chr1", "chromStart": peak_start,
                  "chromEnd": peak_start + 399, "linked_gene": genes[primary],
                  "linked_gene_index": primary + 1, "coordinate_system": "synthetic_human_like_v1"}).to_csv(
        intermediate / "peak_metadata.tsv.gz", sep="\t", index=False)
    pd.DataFrame([{"available": False, "mode": "synthetic_region_to_tf_with_pseudo_gc_background_no_reference_genome_scan",
                   "reason": "artificial_test_fixture", "n_motifs": 0}]).to_csv(
        intermediate / "chromvar_status.tsv.gz", sep="\t", index=False)
    pd.DataFrame([{"dataset_name": name, "topology": "bifurcation3", "seed": 52,
                   "requested_cells": n_cells, "requested_genes": n_genes,
                   "simulation_attempt_label": "artificial_test_fixture"}]).to_csv(
        intermediate / "simulation_manifest.tsv.gz", sep="\t", index=False)
    theta = rng.dirichlet(np.ones(n_topics), size=n_cells).astype(np.float32)
    psi = rng.dirichlet(np.ones(n_peaks), size=n_topics).astype(np.float32)
    np.savez_compressed(intermediate / "cistopic_factors.npz", theta_cells_topics=theta, psi_topics_peaks=psi)


def command(*args: str) -> None:
    result = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def expect_failure(callback, label: str) -> None:
    try:
        callback()
    except (ValueError, AssertionError):
        return
    raise AssertionError(f"Negative control was not rejected: {label}")


def main() -> None:
    datasets = load_datasets()
    assert len(datasets) == 25
    assert max(int(row["n_cells"]) for row in datasets.values()) <= 10000
    sparse_rows = [row for row in datasets.values() if "atac_p_zero" in row]
    assert len(sparse_rows) == 4 and {row["topology"] for row in sparse_rows} == {"linear", "bifurcation3"}
    assert select_ids("1-3,25-28", set(datasets)) == [1, 2, 3, 25, 26, 27, 28]
    expect_failure(lambda: select_ids("999", set(datasets)), "unknown dataset")
    with tempfile.TemporaryDirectory(prefix="multiome_processing_test_") as temporary:
        root = Path(temporary)
        name = datasets[20]["dataset_name"]
        source = root / name
        fixture(source, name)
        command(HERE / "02_compute_diff_peaks.py", "--outdir", source)
        command(HERE / "04_build_pair.py", "--outdir", source, "--dataset-name", name,
                "--topology", "bifurcation3", "--seed", "52")
        rna_path, atac_path = [source / f"{name}_{modality}.h5ad" for modality in ("rna", "atac")]
        rna, atac = ad.read_h5ad(rna_path), ad.read_h5ad(atac_path)
        assert validate_pair(rna, atac)["status"] == "PASS"
        expect_failure(lambda: validate_pair(rna, atac, True), "missing deviations")
        expect_failure(lambda: validate_pair(rna, atac[::-1].copy()), "shuffled paired cells")
        broken = rna.copy()
        broken.layers["chromatin"] = broken.layers["chromatin"] * 2
        expect_failure(lambda: validate_pair(broken, atac), "incorrect peak-to-gene aggregation")
        command(HERE / "05_add_tf_activity.py", "--atac", atac_path)
        atac = ad.read_h5ad(atac_path)
        assert validate_pair(rna, atac)["tf_activity_is_proxy"]
        expect_failure(lambda: validate_pair(rna, atac, True), "proxy passed as actual deviations")
        original_activity = atac.obsm["chromVAR"].copy()
        command(HERE / "05_add_tf_activity.py", "--atac", atac_path)
        assert np.array_equal(original_activity, ad.read_h5ad(atac_path).obsm["chromVAR"])
        (source / "_SUCCESS").write_text("artificial fixture validated\n")
        command(HERE / "downsample.py", "--root", root)
        command(HERE / "downsample.py", "--root", root)
        summary = pd.read_csv(root / "downsampling_31_40_summary.tsv", sep="\t")
        assert len(summary) == 10 and summary.groupby("fraction").size().eq(2).all()
        for row in summary.itertuples():
            sub_rna, sub_atac = ad.read_h5ad(row.rna_h5ad), ad.read_h5ad(row.atac_h5ad)
            assert sub_rna.n_obs == round(rna.n_obs * row.fraction)
            assert sub_rna.var_names.equals(rna.var_names) and sub_atac.var_names.equals(atac.var_names)
            positions = rna.obs_names.get_indexer(sub_rna.obs_names)
            assert np.array_equal(sub_rna.layers["true_velocity"], rna.layers["true_velocity"][positions])
            assert np.array_equal(sub_atac.obsm["chromVAR"], atac.obsm["chromVAR"][positions])
            assert_equal_matrix(sub_rna.X, rna.X[positions], "Downsampled RNA X")
            assert_equal_matrix(sub_atac.X, atac.X[positions], "Downsampled ATAC X")
            manifest = json.loads((root / row.dataset / "downsampling_manifest.json").read_text())
            assert manifest["selected_cell_names"] == sub_rna.obs_names.tolist()
            assert (root / row.dataset / "sckinetics_differential_peaks.bed").read_bytes() == (source / "sckinetics_differential_peaks.bed").read_bytes()
            validate_pair(sub_rna, sub_atac)
        figures = list((source / "figures").iterdir())
        assert len(figures) == 8 and all(path.stat().st_size > 100 for path in figures)
    print("PASS: retained manifests; differential peaks; paired H5AD export; four reference plots;")
    print("      labeled TF activity; negative controls; ten downsampling tasks and resume checks.")
    print("Not exercised: R simulation, actual chromVAR deviations, and pycisTopic fitting.")


if __name__ == "__main__":
    main()
