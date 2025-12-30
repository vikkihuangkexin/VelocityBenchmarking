#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Precompute Ground Truth Velocity Embeddings for Simulation Data

This script processes simulated single-cell trajectory datasets and precomputes
ground truth velocity embeddings in low-dimensional space (X_dimred), saving
results as NPZ files for efficient downstream analysis.

Input: H5AD files containing:
  - obs: 'cell_id', 'milestone'
  - var: 'gene_id'
  - obsm: 'X_dimred' (2D embedding coordinates)
  - layers: 'ground_truth_velocity', 'spliced', 'unspliced'

Output: NPZ files containing:
  - gt_dimred: Ground truth velocity in 2D space (n_cells, 2)
  - X_dimred: 2D embedding coordinates (n_cells, 2)
  - cell_names: Original cell names
  - cell_names_unique: Unique cell names (if duplicates exist)

Usage:
    python get_gt_concurrent.py \\
        --input-dir /path/to/simdata_local \\
        --output-dir /path/to/output \\
        --topologies bifurcating cycle-simple trifurcating
"""

import os
import sys
import warnings
import argparse
import logging
import traceback
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import scanpy as sc
import scvelo as scv

# Suppress warnings and verbosity
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
scv.settings.verbosity = 0
scv.settings.presenter_view = False
scv.settings.plot_prefix = ""
scv.settings.show_progress_bar = False


def compute_gt_embedding(adata) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ground truth velocity embedding in dimred space.

    Args:
        adata: AnnData object with ground_truth_velocity in layers

    Returns:
        Tuple of (gt_dimred, X_dimred) arrays
    """
    # Compute neighbor graph
    sc.pp.neighbors(adata)

    # Compute velocity graph and embedding in dimred space
    scv.tl.velocity_graph(adata, vkey="ground_truth_velocity", n_jobs=10)
    scv.tl.velocity_embedding(adata, basis="dimred", vkey="ground_truth_velocity")

    gt_dimred = adata.obsm["ground_truth_velocity_dimred"]
    X_dimred = adata.obsm["X_dimred"]

    return gt_dimred, X_dimred


def save_gt_npz(
    output_dir: Path,
    topology: str,
    dataset_name: str,
    gt_dimred: np.ndarray,
    X_dimred: np.ndarray,
    cell_names: np.ndarray,
    cell_names_unique: Optional[np.ndarray] = None,
) -> Path:
    """
    Save ground truth data as NPZ file.

    Args:
        output_dir: Base output directory
        topology: Topology type (e.g., 'bifurcating')
        dataset_name: Dataset identifier
        gt_dimred: Ground truth velocity in 2D
        X_dimred: 2D embedding coordinates
        cell_names: Original cell names
        cell_names_unique: Unique cell names (optional)

    Returns:
        Path to saved NPZ file
    """
    topology_dir = output_dir / topology
    topology_dir.mkdir(parents=True, exist_ok=True)

    # Remove '_dataset.h5ad' suffix
    base_name = dataset_name.replace("_dataset.h5ad", "")
    output_path = topology_dir / f"{base_name}_gt_data.npz"

    # Prepare data for saving
    save_kwargs = {
        "gt_dimred": gt_dimred,
        "X_dimred": X_dimred,
        "cell_names": cell_names,
    }

    if cell_names_unique is not None:
        save_kwargs["cell_names_unique"] = cell_names_unique

    np.savez_compressed(output_path, **save_kwargs)
    return output_path


def process_single_dataset(
    topology: str,
    filename: str,
    filepath: Path
) -> Tuple[bool, str, bool, str, Optional[str]]:
    """
    Process a single dataset file.

    Args:
        topology: Topology type
        filename: Dataset filename
        filepath: Full path to H5AD file

    Returns:
        Tuple of (success, message, has_unique_names, base_name, traceback)
    """
    base_name = filename.replace("_dataset.h5ad", "")

    try:
        # Load data
        adata = sc.read_h5ad(filepath)

        # Save original cell names
        raw_cell_names = np.array(adata.obs_names.values, copy=True)

        # Make cell names unique for scvelo computation
        adata.obs_names_make_unique()

        # Compute ground truth embedding
        gt_dimred, X_dimred = compute_gt_embedding(adata)

        # Restore original cell names
        adata.obs_names = raw_cell_names

        # Check if unique names differ from original
        adata.obs_names_make_unique()
        new_cell_names = adata.obs_names.values

        cell_names_unique = None
        has_unique = False
        if not np.array_equal(new_cell_names, raw_cell_names):
            cell_names_unique = new_cell_names
            has_unique = True

        return True, f"{topology}/{filename}", has_unique, base_name, None

    except Exception as e:
        tb_str = traceback.format_exc()
        msg = f"{topology}/{filename}: {str(e)}"
        return False, msg, False, base_name, tb_str


def find_dataset_files(
    input_dir: Path,
    topologies: List[str]
) -> List[Tuple[str, str, Path]]:
    """
    Find all dataset files in specified topology directories.

    Args:
        input_dir: Base input directory
        topologies: List of topology types to process

    Returns:
        List of (topology, filename, filepath) tuples
    """
    dataset_files = []

    for topology in topologies:
        topology_path = input_dir / topology
        if not topology_path.exists():
            print(f"Warning: Topology directory not found: {topology_path}")
            continue

        for file in topology_path.iterdir():
            if file.name.endswith("_dataset.h5ad"):
                dataset_files.append((topology, file.name, file))

    return dataset_files


def main():
    parser = argparse.ArgumentParser(
        description="Precompute ground truth velocity embeddings for simulation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python get_gt_concurrent.py \\
      --input-dir /path/to/simdata_local \\
      --output-dir /path/to/simdata_gtkey \\
      --topologies bifurcating cycle-simple trifurcating

Expected input directory structure:
  input_dir/
    ├── bifurcating/
    │   ├── bifurcating_cell1000_gene1000_dataset.h5ad
    │   └── ...
    ├── cycle-simple/
    │   └── ...
    └── trifurcating/
        └── ...

Output directory structure:
  output_dir/
    ├── bifurcating/
    │   ├── bifurcating_cell1000_gene1000_gt_data.npz
    │   └── ...
    └── ...
        """
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Base directory containing simulation data (organized by topology)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for NPZ files",
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=[
            "bifurcating",
            "cycle-simple",
            "trifurcating",
            "linear-simple",
            "bifurcating-loop",
            "consecutive-bifurcating",
            "disconnected",
            "simulation-add",
            "cellsub-bifurcating",
            "genesub-bifurcating",
        ],
        help="List of topology types to process (default: all common topologies)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: {output_dir}/precompute_gt.log)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_file) if args.log_file else (output_dir / "precompute_gt.log")

    # Setup logging
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.ERROR,
        format="%(levelname)s - %(message)s",
    )

    # Find dataset files
    dataset_files = find_dataset_files(input_dir, args.topologies)
    if not dataset_files:
        print("No dataset files found")
        return

    print(f"Found {len(dataset_files)} dataset files")
    if args.verbose:
        print("Processing datasets:")
        for topology, filename, _ in dataset_files:
            print(f"  {topology}/{filename}")

    # Process datasets
    results = []
    for i, (topology, filename, filepath) in enumerate(dataset_files, 1):
        if args.verbose:
            print(f"[{i}/{len(dataset_files)}] Processing {topology}/{filename}...", end=" ")

        success, msg, has_unique, base_name, tb_str = process_single_dataset(
            topology, filename, filepath
        )

        # Save NPZ if successful
        if success:
            # Re-load to get computed data
            adata = sc.read_h5ad(filepath)
            raw_cell_names = np.array(adata.obs_names.values, copy=True)
            adata.obs_names_make_unique()
            gt_dimred, X_dimred = compute_gt_embedding(adata)
            adata.obs_names = raw_cell_names
            adata.obs_names_make_unique()

            cell_names_unique = None
            if has_unique:
                cell_names_unique = adata.obs_names.values

            save_gt_npz(
                output_dir,
                topology,
                filename,
                gt_dimred,
                X_dimred,
                raw_cell_names,
                cell_names_unique,
            )

            if args.verbose:
                print("✓")

        else:
            if args.verbose:
                print(f"✗ {msg}")

        results.append((success, msg, has_unique, base_name, tb_str))

    # Summary
    success_count = sum(1 for success, _, _, _, _ in results if success)
    failed_count = len(results) - success_count

    print(f"\n{'='*60}")
    print(f"Processing complete")
    print(f"  Total: {len(results)} datasets")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output: {output_dir}")
    print(f"  Log: {log_file}")
    print(f"{'='*60}")

    # Report failures
    failed_infos = [
        (msg, tb_str) for success, msg, _, _, tb_str in results if not success
    ]
    if failed_infos:
        print("\nFailed files:")
        for msg, _ in failed_infos:
            print(f"  - {msg}")

    # Report datasets with unique cell names
    unique_datasets = [
        base_name for success, _, has_unique, base_name, _ in results
        if success and has_unique
    ]
    if unique_datasets:
        print("\nDatasets with cell_names_unique:")
        for base_name in unique_datasets:
            print(f"  - {base_name}")

    # Write log
    with open(log_file, "w", encoding="utf-8") as f:
        for msg, tb_str in failed_infos:
            f.write(f"[ERROR] {msg}\n")
            if tb_str:
                f.write(tb_str + "\n")

        if unique_datasets:
            f.write("\nDatasets with cell_names_unique:\n")
            for base_name in unique_datasets:
                f.write(f"  - {base_name}\n")


if __name__ == "__main__":
    main()
