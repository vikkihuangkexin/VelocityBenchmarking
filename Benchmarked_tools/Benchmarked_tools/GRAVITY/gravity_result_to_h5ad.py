#!/usr/bin/env python3
"""
GRAVITY result CSV to H5AD converter.

GRAVITY's ``stage2.csv`` result has the same layout as a cellDancer result
table (``splice``/``unsplice``/``alpha``/``beta``/``gamma``/``splice_predict``/
``unsplice_predict`` columns per gene and cell), so ``celldancer.utilities.to_dynamo``
is used to build the AnnData object. This converter is shared by the real-data
path of ``GRAVITY.py`` and by the standalone Docker workflow.

Installation:
    pip install celldancer

Usage:
    python gravity_result_to_h5ad.py --input gravity_result.csv --output-dir ./output
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import celldancer.utilities as cdutil
import pandas as pd


def convert(result_csv, save_dir=None, data_file=None):
    """Read a GRAVITY result CSV and build an AnnData object.

    Parameters
    ----------
    result_csv
        Path to the GRAVITY result CSV.
    save_dir, data_file
        Optional. When both are given, the AnnData object is also written to
        ``<save_dir>/<data_file stem>_velo.h5ad``.

    Returns
    -------
    anndata.AnnData
        The converted dataset. Its native velocity keys (``velocity_S`` /
        ``velocity_U`` / ``obsm['velocity_cdr']``) are preserved and
        ``layers['velocity']`` is added as an alias of the gene-level spliced
        velocity so the benchmark-standard layer is always present.
    """
    result_csv = Path(result_csv)
    cell_dancer_df = pd.read_csv(result_csv)

    # to_dynamo expects a loss column; GRAVITY's stage2.csv has none, so add a placeholder.
    if "loss" not in cell_dancer_df.columns:
        cell_dancer_df["loss"] = 0.0

    adata = cdutil.to_dynamo(cell_dancer_df)

    # Benchmark-standard velocity layer (native GRAVITY/cellDancer keys are kept).
    if "velocity" not in adata.layers:
        for source_key in ("velocity_S", "velocity_s"):
            if source_key in adata.layers:
                adata.layers["velocity"] = adata.layers[source_key].copy()
                break

    if "velocity" not in adata.layers:
        raise RuntimeError(
            "The GRAVITY result table produced neither layers['velocity'] nor "
            "layers['velocity_S'], so layers['velocity'] cannot be exported."
        )

    if save_dir is not None and data_file is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{Path(data_file).stem}_velo.h5ad"
        adata.write_h5ad(out_path)
        print(f"[DONE] {out_path}")

    return adata


def run_convert(input_path, output_dir, dataset_name: Optional[str] = None, overwrite: bool = False) -> Path:
    """Convert a result CSV and write ``<output-dir>/<stem>_velo.h5ad``."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    stem = dataset_name or input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}_velo.h5ad"
    if out_path.exists() and not overwrite:
        print(f"Skipping existing output: {out_path}")
        return out_path

    adata = convert(input_path)
    adata.write_h5ad(out_path)
    print(f"[DONE] {out_path}")
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a GRAVITY result CSV to H5AD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="GRAVITY result CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset-name", default=None, help="Output file stem")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()
    return run_convert(
        input_path=args.input,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
