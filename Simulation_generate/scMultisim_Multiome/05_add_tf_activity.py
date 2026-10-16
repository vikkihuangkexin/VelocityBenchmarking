#!/usr/bin/env python3
"""Fill missing synthetic TF activity with the supplied depth-normalized proxy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

MODE = "synthetic_region_to_tf_depth_normalized_zscore"


def add_tf_activity(atac) -> bool:
    if "chromVAR" in atac.obsm:
        return False
    if "region_to_tf" not in atac.varm or "region_to_tf_names" not in atac.uns:
        raise ValueError("Missing stored region-to-TF graph or TF names")
    counts = sparse.csr_matrix(atac.layers["atac_counts"], dtype=np.float32)
    mapping = sparse.csr_matrix(atac.varm["region_to_tf"], dtype=np.float32)
    names = np.asarray(atac.uns["region_to_tf_names"], dtype=str)
    if mapping.shape != (atac.n_vars, len(names)):
        raise ValueError("Region-to-TF graph/names are not aligned")
    if atac.n_obs < 2:
        raise ValueError("At least two cells are required for sample standardization")
    raw = (counts @ mapping).toarray().astype(np.float32)
    depth = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    activity = np.log1p(raw / np.maximum(depth[:, None], 1.0) * 1e4)
    sd = activity.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    activity = (activity - activity.mean(axis=0)) / sd
    if not np.isfinite(activity).all():
        raise ValueError("Non-finite TF activity")
    atac.obsm["chromVAR"] = activity.astype(np.float32)
    atac.uns["chromVAR_features"] = names
    atac.uns["chromVAR_mode"] = MODE
    atac.uns["chromVAR_status"] = {
        "available": True, "mode": MODE, "is_chromvar_deviation": False,
        "reason": "Explicit synthetic TF-activity proxy; no background correction",
    }
    atac.uns["simulation"]["chromVAR_mode"] = MODE
    atac.uns["simulation"]["chromVAR_available"] = True
    atac.uns["simulation"]["chromVAR_reason"] = "synthetic_proxy"
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atac", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Default: atomically update --atac")
    args = parser.parse_args()
    atac = ad.read_h5ad(args.atac)
    if not add_tf_activity(atac):
        print("TF activity already exists; existing values retained")
        if args.output and args.output.resolve() != args.atac.resolve():
            if args.output.exists():
                raise FileExistsError(args.output)
            atac.write_h5ad(args.output, compression="gzip")
        return
    destination = args.output or args.atac
    if args.output and args.output.resolve() != args.atac.resolve() and args.output.exists():
        raise FileExistsError(args.output)
    temporary = destination.with_name(destination.stem + ".tf_activity_tmp.h5ad")
    atac.write_h5ad(temporary, compression="gzip")
    os.replace(temporary, destination)
    print(f"Stored labeled TF-activity proxy: {destination}; shape={atac.obsm['chromVAR'].shape}")


if __name__ == "__main__":
    main()
