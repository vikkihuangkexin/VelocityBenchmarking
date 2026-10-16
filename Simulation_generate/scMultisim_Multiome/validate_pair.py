#!/usr/bin/env python3
"""Validate paired H5AD fields, alignment, and peak-to-gene reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from common import write_json


def finite(value) -> bool:
    return bool(np.isfinite(value.data if sparse.issparse(value) else np.asarray(value)).all())


def assert_equal_matrix(left, right, label: str) -> None:
    left, right = sparse.csr_matrix(left), sparse.csr_matrix(right)
    if left.shape != right.shape:
        raise ValueError(f"{label}: shapes differ")
    difference = left - right
    difference.eliminate_zeros()
    if difference.nnz:
        raise ValueError(f"{label}: {difference.nnz} entries differ")


def validate_pair(rna, atac, require_chromvar: bool = False) -> dict:
    if rna.n_obs < 2 or atac.n_vars < 1 or rna.n_vars < 1:
        raise ValueError("Empty or undersized paired data")
    if not rna.obs_names.equals(atac.obs_names):
        raise ValueError("RNA/ATAC cells or their order differ")
    if not rna.obs_names.is_unique or not rna.var_names.is_unique or not atac.var_names.is_unique:
        raise ValueError("Duplicated cell or feature names")
    for obj, keys in [(rna, ("spliced", "unspliced", "true_velocity", "chromatin", "atac_counts", "Mc")),
                      (atac, ("atac_counts", "true_accessibility"))]:
        for key in keys:
            if key not in obj.layers or obj.layers[key].shape != obj.shape or not finite(obj.layers[key]):
                raise ValueError(f"Missing, misaligned, or non-finite layer: {key}")
    for obj in (rna, atac):
        if obj.X is None or not finite(obj.X):
            raise ValueError("Missing or non-finite X matrix")
        for key in ("true_cell_type", "true_time", "true_pseudotime", "true_branch", "true_cell_type_order"):
            if key not in obj.obs or obj.obs[key].isna().any():
                raise ValueError(f"Missing ground-truth metadata: {key}")
            if not rna.obs[key].astype(str).equals(atac.obs[key].astype(str)):
                raise ValueError(f"Paired metadata differs: {key}")
        for key in ("X_tsne", "true_velocity_tsne"):
            if key not in obj.obsm or obj.obsm[key].shape != (obj.n_obs, 2) or not finite(obj.obsm[key]):
                raise ValueError(f"Missing or invalid embedding: {key}")
        if "lineage_paths" not in obj.uns.get("differentiation_ground_truth", {}):
            raise ValueError("Missing lineage_paths")
    counts = sparse.csr_matrix(atac.layers["atac_counts"])
    if (counts.data < 0).any() or not np.equal(counts.data, np.rint(counts.data)).all():
        raise ValueError("ATAC counts must be nonnegative integers")
    assert_equal_matrix(atac.X, counts, "ATAC X vs count layer")
    mapping = atac.varm.get("region_to_gene")
    if mapping is None or mapping.shape != (atac.n_vars, rna.n_vars):
        raise ValueError("Invalid region_to_gene shape")
    if not np.array_equal(np.asarray(atac.uns.get("region_to_gene_genes", []), dtype=str), rna.var_names):
        raise ValueError("Peak-to-gene columns do not match RNA gene order")
    mapping = sparse.csr_matrix(mapping)
    if not np.isin(mapping.data, [0, 1]).all() or (np.asarray(mapping.sum(axis=1)).ravel() < 1).any():
        raise ValueError("Peak-to-gene graph is not binary or has unlinked peaks")
    assert_equal_matrix(counts @ mapping, rna.layers["chromatin"], "Peak-to-gene aggregation")
    assert_equal_matrix(rna.layers["chromatin"], rna.layers["atac_counts"], "RNA chromatin aliases")
    if not np.array_equal(rna.var.loc[atac.var["linked_gene"], "chrom"].astype(str).to_numpy(),
                          atac.var["chrom"].astype(str).to_numpy()):
        raise ValueError("Linked peak/gene chromosomes differ")
    for key in ("sckinetics_diff_peak", "true_diff_peak"):
        if key not in atac.var:
            raise ValueError(f"Missing ATAC peak selection: {key}")
    for key in ("sckinetics_diff_peaks", "true_differential_peaks"):
        if key not in atac.uns:
            raise ValueError(f"Missing ATAC differential table: {key}")
    theta = np.asarray(atac.obsm.get("cisTopic_topics"))
    psi = np.asarray(atac.uns.get("cisTopic_topic_peak"))
    reconstruction = atac.obsm.get("cisTopic")
    if (theta.ndim != 2 or psi.ndim != 2 or theta.shape[0] != atac.n_obs or
        psi.shape != (theta.shape[1], atac.n_vars) or reconstruction is None or
        reconstruction.shape != atac.shape):
        raise ValueError("cisTopic factors/reconstruction have incompatible shapes")
    if not finite(theta) or not finite(psi) or not finite(reconstruction):
        raise ValueError("Non-finite cisTopic factors")
    if (theta < 0).any() or (psi < 0).any():
        raise ValueError("Negative topic factors")
    for start in range(0, atac.n_obs, 256):
        if not np.allclose(reconstruction[start:start + 256], theta[start:start + 256] @ psi,
                           atol=1e-6, rtol=1e-5):
            raise ValueError("cisTopic is inconsistent with theta @ psi")
    status = dict(atac.uns.get("chromVAR_status", {}))
    mode = str(status.get("mode", "unknown"))
    has_activity = "chromVAR" in atac.obsm
    is_proxy = "depth_normalized_zscore" in mode
    if bool(status.get("available", False)) != has_activity:
        raise ValueError("TF-activity availability metadata is inconsistent")
    if has_activity:
        activity = atac.obsm["chromVAR"]
        if activity.shape != (atac.n_obs, len(atac.uns.get("chromVAR_features", []))) or not finite(activity):
            raise ValueError("TF activity/names are invalid")
    if require_chromvar and (not has_activity or is_proxy):
        raise ValueError("Actual chromVAR deviations are required; missing or proxy-only")
    if {"chromvar", "cistopic"}.intersection(atac.obsm):
        raise ValueError("Unexpected lowercase aliases")
    return {
        "status": "PASS", "rna_shape": list(rna.shape), "atac_shape": list(atac.shape),
        "paired_cell_order": True, "peak_to_gene_reconstruction": True,
        "cisTopic_reconstruction": True, "tf_activity_available": has_activity,
        "tf_activity_mode": mode, "tf_activity_is_proxy": is_proxy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna", type=Path, required=True)
    parser.add_argument("--atac", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-chromvar", action="store_true")
    args = parser.parse_args()
    result = validate_pair(ad.read_h5ad(args.rna), ad.read_h5ad(args.atac), args.require_chromvar)
    if args.output:
        write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
