#!/usr/bin/env python3
"""
RegVelo preprocessing step 1: convert an h5ad into the loom consumed by pySCENIC.

The real-data RegVelo pipeline obtains its prior GRN with pySCENIC, which reads
the expression matrix in loom format. This helper writes ``adata.loom`` (with the
``Gene`` and ``CellID`` columns pySCENIC expects) and records the detected species
in ``species.txt`` so that the pipeline can pick the matching TF list, ranking
database and motif annotation.

The species is detected from the file name: a case-insensitive ``Mm`` token means
mouse and an ``Hs`` (or ``peripheral``) token means human.

Installation:
    pip install scanpy loompy

Usage:
    python h5ad_to_loom.py --input data.h5ad --output-dir ./output --dataset-name 42_Mm_embryos.h5ad

The output directory receives:

    adata.loom    expression matrix for pySCENIC
    species.txt   detected species token (``mm`` or ``hs``)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def detect_species(filename: str) -> str:
    """Detect the species token (``mm`` / ``hs``) from a file name."""
    tokens = filename.replace(".", "_").split("_")
    for token in tokens:
        lowered = token.lower()
        if lowered == "mm":
            return "mm"
        if lowered == "hs":
            return "hs"
        if lowered == "peripheral":
            return "hs"
    raise ValueError(f"Could not detect species from '{filename}' (expected 'Mm' or 'Hs')")


def h5ad_to_loom(input_path, output_dir, dataset_name: Optional[str] = None) -> Path:
    """Write ``<output-dir>/adata.loom`` and ``<output-dir>/species.txt``."""
    import scanpy as sc

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    dataset_name = dataset_name or input_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read(input_path, cache=False)
    species = detect_species(dataset_name)
    print(f"[INFO] detected species: {species}")

    adata_loom = sc.AnnData(adata.X, obs=adata.obs.copy(), var=adata.var.copy())
    adata_loom.var["Gene"] = adata_loom.var_names
    adata_loom.obs["CellID"] = adata_loom.obs_names

    loom_path = output_dir / "adata.loom"
    adata_loom.write_loom(str(loom_path))
    (output_dir / "species.txt").write_text(species, encoding="utf-8")
    print(f"[INFO] loom written: {loom_path}")
    return loom_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RegVelo preprocessing: h5ad -> loom",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input H5AD file")
    parser.add_argument("--output-dir", required=True, help="Directory for adata.loom and species.txt")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="File name used for species detection; defaults to the input file name",
    )
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()
    return h5ad_to_loom(args.input, args.output_dir, args.dataset_name)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
