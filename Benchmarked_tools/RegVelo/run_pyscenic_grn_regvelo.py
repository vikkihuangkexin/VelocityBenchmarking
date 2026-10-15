#!/usr/bin/env python3
"""
RegVelo preprocessing step 2: build the pySCENIC GRN parquet and run RegVelo.

The real-data pipeline runs pySCENIC (grn -> ctx -> aucell) to obtain the prior
GRN. This helper turns the pySCENIC AUCell output (``pyscenic_output.loom``) into
the square 0/1 ``regulon_mat_processed_all_regulons.parquet`` and then invokes
``RegVelo.run_regvelo_analysis`` so the velocity is estimated in one step.

It is normally called by ``pipeline.sh`` after the pySCENIC aucell step, but it
can also be used stand-alone when a pySCENIC AUCell loom already exists.

Installation:
    pip install regvelo scanpy scvelo pandas numpy scikit-learn loompy pyarrow

Usage:
    python run_pyscenic_grn_regvelo.py \
        --input data.h5ad \
        --output-dir ./output \
        --dataset-name id_x \
        --cluster-key celltype \
        --pyscenic-loom ./output/id_x/pyscenic_output.loom
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from RegVelo import (  # noqa: E402 - imported after the sys.path setup above
    generate_grn_parquet,
    run_regvelo_analysis,
)


def run_pyscenic_grn(
    input_path,
    output_dir,
    cluster_key: str,
    dataset_name: Optional[str] = None,
    pyscenic_loom=None,
    grn_parquet=None,
    n_top_genes: int = 2000,
    max_epochs: int = 150,
    batch_size: Optional[int] = None,
    cuda: Optional[int] = None,
    seed: int = 2024,
    overwrite: bool = False,
):
    """Generate the GRN parquet (if needed) and run RegVelo on the dataset.

    ``output_dir`` is the root directory; the GRN parquet, the preprocessed
    ``pp.h5ad``, the saved model and the final ``<stem>.h5ad`` are all written to
    ``<output-dir>/<dataset-name>/``.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    dataset_name = dataset_name or input_path.stem
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    parquet = grn_parquet
    if parquet is None:
        if pyscenic_loom is None:
            default_loom = dataset_dir / "pyscenic_output.loom"
            if default_loom.exists():
                pyscenic_loom = default_loom
            else:
                raise ValueError("Provide --pyscenic-loom or --grn-parquet")
        parquet = generate_grn_parquet(
            pyscenic_loom, dataset_dir / "regulon_mat_processed_all_regulons.parquet"
        )
    else:
        print(f"[INFO] using existing GRN parquet: {parquet}")

    return run_regvelo_analysis(
        input_path=input_path,
        output_dir=output_dir,
        cluster_key=cluster_key,
        dataset_name=dataset_name,
        grn_parquet=str(parquet),
        n_top_genes=n_top_genes,
        max_epochs=max_epochs,
        batch_size=batch_size,
        cuda=cuda,
        seed=seed,
        overwrite=overwrite,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RegVelo preprocessing: pySCENIC GRN parquet + RegVelo velocity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input H5AD file")
    parser.add_argument("--output-dir", required=True, help="Output directory for the dataset")
    parser.add_argument("--dataset-name", default=None, help="Dataset folder name")
    parser.add_argument("--cluster-key", required=True, help="Column in adata.obs used for cell-type labels")
    parser.add_argument("--pyscenic-loom", dest="pyscenic_loom", default=None,
                        help="pySCENIC AUCell loom; used to generate the GRN parquet")
    parser.add_argument("--grn-parquet", dest="grn_parquet", default=None,
                        help="Pre-computed GRN parquet; skips parquet generation")
    parser.add_argument("--n-top-genes", dest="n_top_genes", type=int, default=2000,
                        help="Number of highly variable genes")
    parser.add_argument("--max-epochs", dest="max_epochs", type=int, default=150,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                        help="Mini-batch size (None = all cells)")
    parser.add_argument("--cuda", type=int, default=None,
                        help="CUDA device id; when omitted the freest GPU is auto-selected")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser


def main(args: Optional[argparse.Namespace] = None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()
    return run_pyscenic_grn(
        input_path=args.input,
        output_dir=args.output_dir,
        cluster_key=args.cluster_key,
        dataset_name=args.dataset_name,
        pyscenic_loom=args.pyscenic_loom,
        grn_parquet=args.grn_parquet,
        n_top_genes=args.n_top_genes,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        cuda=args.cuda,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
