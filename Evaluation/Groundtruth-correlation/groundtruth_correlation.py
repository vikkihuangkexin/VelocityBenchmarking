#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ground Truth Correlation Metric

Calculates cosine similarity between predicted velocity and ground truth velocity
for RNA velocity prediction methods on simulated single-cell trajectory data.

This metric is specifically designed for simulated data where ground truth velocity
vectors are available. It computes the mean cosine similarity between predicted and
ground truth velocity vectors, either in high-dimensional gene space or low-dimensional
embedding space.

Features:
- Single-file and batch processing modes
- Tool-specific preprocessing for different RNA velocity methods
- Ground truth recovery from precomputed NPZ files
- Wide-format CSV output with automatic ranking
- Auto-detection and warnings for common data issues
- Structured error logging for easy troubleshooting

Supported Tools (with automatic preprocessing):
- PhyloVelo: Handles low-dimensional-only results
- cellDancer: Handles renamed cluster annotations
- TFvelo: Computes velocity from velo_hat and fit_scaling_y
- SDEvelo: Computes velocity graph and embedding
- TopicVelo: Handles alternative velocity key names

Usage:
    # Programmatic API
    from groundtruth_correlation import calculate_groundtruth_correlation
    result = calculate_groundtruth_correlation(
        adata_or_path='result.h5ad',
        method='VeloVAE',
        dataset_id='bifurcating_cell1000_gene1000',
        output_csv='results.csv'
    )

    # Command-line interface (single file)
    python groundtruth_correlation.py --input result.h5ad --method VeloVAE \\
        --dataset-id bifurcating_cell1000_gene1000 --output-csv results.csv

    # Command-line interface (batch)
    python groundtruth_correlation.py --metadata-csv datasets.csv \\
        --output-csv results.csv --output-dir results/

Author: RNA Velocity Benchmarking Team
Date: 2025-01-30
"""

import os
import sys
import warnings
import logging
import signal
import atexit
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import pandas as pd

try:
    import scanpy as sc
    import scvelo as scv
    from anndata import AnnData
except ImportError as e:
    print(f"ERROR: Required package not found: {e}")
    print("Please install: pip install scanpy scvelo anndata")
    sys.exit(1)

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# CONSTANTS
# =============================================================================

# Special dataset configurations
SIMULATION_ADD_DATASETS = {
    '1bifurcating-add_cell1000_gene1000',
    '2bifurcating-add_cell1000_gene1000',
    '3bifurcating-add_cell1000_gene1000',
    '4bifurcating-add_cell1000_gene1000',
}

# Dataset topologies (underscore → hyphen normalization)
TOPOLOGY_VARIANTS = {
    'bifurcating_loop': 'bifurcating-loop',
    'cycle_simple': 'cycle-simple',
    'linear_simple': 'linear-simple',
    'linear_bifurcating': 'linear-bifurcating',
    'consecutive_bifurcating': 'consecutive-bifurcating',
    'linear_linear': 'linear-linear',
    'genesub_bifurcating': 'genesub-bifurcating',
    'cellsub_bifurcating': 'cellsub-bifurcating',
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_dataset_name(dataset_name: str) -> str:
    """
    Normalize dataset name by replacing underscores with hyphens in topology names.

    Example:
        'bifurcating_loop_cell1000_gene1000' → 'bifurcating-loop_cell1000_gene1000'
        'cycle_simple_cell5000_gene500' → 'cycle-simple_cell5000_gene500'

    Args:
        dataset_name: Original dataset identifier

    Returns:
        Normalized dataset identifier with hyphens in topology part
    """
    if not dataset_name:
        return dataset_name

    # Replace known topology variants
    for old_variant, new_variant in TOPOLOGY_VARIANTS.items():
        if dataset_name.startswith(old_variant):
            return dataset_name.replace(old_variant, new_variant, 1)

    return dataset_name


def normalize_cell_name(cell_name: str, adjust_numeric_index: bool = False) -> str:
    """
    Standardize cell names to ensure consistent matching.

    Handles different cell naming conventions:
    - Ensures 'cell' prefix exists
    - Handles numeric-only indices (e.g., STT tool uses '0', '1', '2'...)

    Args:
        cell_name: Original cell identifier
        adjust_numeric_index: If True, add 1 to numeric indices (STT uses 0-based indexing)

    Returns:
        Normalized cell name (e.g., '0' → 'cell1', 'cell123' → 'cell123')
    """
    cell_name_str = str(cell_name).strip()

    # Check if already has 'cell' prefix
    if cell_name_str.lower().startswith('cell'):
        return cell_name_str

    # Handle numeric-only indices
    if cell_name_str.isdigit():
        numeric_idx = int(cell_name_str)
        if adjust_numeric_index:
            # STT uses 0-based indexing, adjust to 1-based
            numeric_idx += 1
        return f"cell{numeric_idx}"

    # Add 'cell' prefix to non-standard names
    return f"cell{cell_name_str}"


def infer_topology_from_dataset_id(dataset_id: str) -> Optional[str]:
    """
    Infer topology type from dataset identifier.

    Dataset IDs follow pattern: {topology}_{cellcount}_gene{genecount}
    Example: 'bifurcating-loop_cell1000_gene1000' → 'bifurcating-loop'

    Args:
        dataset_id: Dataset identifier

    Returns:
        Topology type string or None if cannot be inferred
    """
    if not dataset_id:
        return None

    # Normalize first
    normalized_id = normalize_dataset_name(dataset_id)

    # Handle simulation-add datasets (numeric prefix)
    if normalized_id in SIMULATION_ADD_DATASETS:
        # Extract topology from '1bifurcating-add' → 'bifurcating'
        return 'bifurcating'

    # Standard format: topology_cellXXX_geneXXX
    # Extract everything before first '_cell'
    if '_cell' in normalized_id:
        topology = normalized_id.split('_cell')[0]
        return topology

    # Fallback: extract before first '_'
    parts = normalized_id.split('_')
    if len(parts) >= 2:
        return parts[0]

    return None


def locate_npz_file(
    dataset_id: str,
    base_dir: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Locate precomputed ground truth NPZ file for a dataset.

    NPZ files contain:
    - gt_dimred: Ground truth velocity in 2D (n_cells, 2)
    - X_dimred: Original 2D embedding coordinates (n_cells, 2)
    - cell_names: Cell identifiers
    - cell_names_unique: Unique cell names (optional, for duplicate handling)

    Search strategy:
    1. Try original dataset_id
    2. Try with underscore→hyphen normalization
    3. Try with hyphen→underscore normalization

    Args:
        dataset_id: Dataset identifier
        base_dir: Base directory containing topology subdirectories
                  If None, returns None (GT recovery skipped)

    Returns:
        Tuple of (npz_file_path, status_message)
        - npz_file_path: Full path to NPZ file, or None if not found
        - status_message: Description of search result
    """
    if base_dir is None:
        return None, "GT NPZ base directory not provided (skipping GT recovery)"

    base_path = Path(base_dir)
    if not base_path.exists():
        return None, f"GT NPZ base directory does not exist: {base_dir}"

    # Infer topology
    topology = infer_topology_from_dataset_id(dataset_id)
    if topology is None:
        return None, f"Cannot infer topology from dataset_id: {dataset_id}"

    topology_dir = base_path / topology
    if not topology_dir.exists():
        return None, f"Topology directory not found: {topology_dir}"

    # Try multiple naming variations
    npz_filename_variants = [
        f"{dataset_id}_gt_data.npz",
        f"{normalize_dataset_name(dataset_id)}_gt_data.npz",
        f"{dataset_id.replace('-', '_')}_gt_data.npz",
    ]

    for npz_filename in npz_filename_variants:
        npz_path = topology_dir / npz_filename
        if npz_path.exists():
            return str(npz_path), f"Found GT NPZ file: {npz_filename}"

    return None, f"GT NPZ file not found in {topology_dir} (tried {len(npz_filename_variants)} variants)"


# =============================================================================
# CELL MATCHING FUNCTIONS
# =============================================================================

def match_cell_indices(
    adata_cell_names: np.ndarray,
    npz_cell_names: np.ndarray,
    npz_cell_names_unique: Optional[np.ndarray] = None,
    allow_partial_match: bool = True,
    min_match_ratio: float = 0.95,
    method_name: str = "Unknown"
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Match adata cells to NPZ cells using multiple strategies.

    Matching strategies (in priority order):
    1. Exact name matching (after normalization)
    2. Fallback to npz_cell_names_unique if provided
    3. Position-based matching (if cell counts match and X_dimred consistent)

    Args:
        adata_cell_names: Cell names from adata.obs_names (n_adata,)
        npz_cell_names: Cell names from NPZ file (n_npz,)
        npz_cell_names_unique: Unique cell names from NPZ (optional)
        allow_partial_match: Allow processing when match ratio < 1.0
        min_match_ratio: Minimum ratio of cells that must match
        method_name: Method name for error logging

    Returns:
        Tuple of (adata_indices, npz_indices, status_message)
        - adata_indices: Indices of matched cells in adata (n_matched,)
        - npz_indices: Corresponding indices in NPZ arrays (n_matched,)
        - status_message: Description of matching result
    """
    n_adata = len(adata_cell_names)
    n_npz = len(npz_cell_names)

    # Normalize cell names
    adata_normalized = np.array([
        normalize_cell_name(name, adjust_numeric_index=(method_name == 'STT'))
        for name in adata_cell_names
    ])

    # Strategy 1: Exact name matching
    npz_cell_dict = {name: idx for idx, name in enumerate(npz_cell_names)}

    adata_indices = []
    npz_indices = []

    for adata_idx, cell_name in enumerate(adata_normalized):
        if cell_name in npz_cell_dict:
            adata_indices.append(adata_idx)
            npz_indices.append(npz_cell_dict[cell_name])

    n_matched = len(adata_indices)
    match_ratio = n_matched / n_adata if n_adata > 0 else 0

    # Check match quality
    if match_ratio >= 1.0:
        return (
            np.array(adata_indices, dtype=int),
            np.array(npz_indices, dtype=int),
            f"Perfect match: {n_matched}/{n_adata} cells"
        )

    # Strategy 2: Try npz_cell_names_unique
    if npz_cell_names_unique is not None and match_ratio < min_match_ratio:
        npz_unique_dict = {name: idx for idx, name in enumerate(npz_cell_names_unique)}

        adata_indices_unique = []
        npz_indices_unique = []

        for adata_idx, cell_name in enumerate(adata_normalized):
            if cell_name in npz_unique_dict:
                adata_indices_unique.append(adata_idx)
                npz_indices_unique.append(npz_unique_dict[cell_name])

        n_matched_unique = len(adata_indices_unique)
        match_ratio_unique = n_matched_unique / n_adata if n_adata > 0 else 0

        if match_ratio_unique > match_ratio:
            return (
                np.array(adata_indices_unique, dtype=int),
                np.array(npz_indices_unique, dtype=int),
                f"Matched using unique names: {n_matched_unique}/{n_adata} cells"
            )

    # Strategy 3: Position-based matching (same cell count)
    if n_adata == n_npz and match_ratio < min_match_ratio:
        # Will be validated by check_dimred_consistency
        return (
            np.arange(n_adata, dtype=int),
            np.arange(n_npz, dtype=int),
            f"Position-based match: {n_adata} cells (requires X_dimred validation)"
        )

    # Check if match ratio is acceptable
    if match_ratio >= min_match_ratio and allow_partial_match:
        return (
            np.array(adata_indices, dtype=int),
            np.array(npz_indices, dtype=int),
            f"Partial match: {n_matched}/{n_adata} cells ({match_ratio:.1%})"
        )

    # Matching failed
    return (
        None,
        None,
        f"Match ratio too low: {n_matched}/{n_adata} cells ({match_ratio:.1%}) < {min_match_ratio:.1%}"
    )


def check_dimred_consistency(
    adata_dimred: np.ndarray,
    npz_dimred: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> bool:
    """
    Check if two X_dimred arrays are consistent (identical within tolerance).

    Used to validate position-based cell matching when cell names don't match
    but cell counts are equal.

    Args:
        adata_dimred: X_dimred from adata (n_cells, n_dims)
        npz_dimred: X_dimred from NPZ (n_cells, n_dims)
        rtol: Relative tolerance for np.allclose
        atol: Absolute tolerance for np.allclose

    Returns:
        True if arrays are consistent, False otherwise
    """
    if adata_dimred.shape != npz_dimred.shape:
        return False

    return np.allclose(adata_dimred, npz_dimred, rtol=rtol, atol=atol)


# =============================================================================
# GROUND TRUTH RECOVERY
# =============================================================================

def recover_ground_truth_velocity(
    adata: AnnData,
    dataset_id: str,
    method: str,
    velocity_key: str,
    gt_velocity_key: str = "ground_truth_velocity",
    gt_npz_base_dir: Optional[str] = None,
    raise_on_failure: bool = True,
    allow_partial_match: bool = True,
    min_match_ratio: float = 0.95
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Recover ground truth velocity from precomputed NPZ file and align with adata.

    This function:
    1. Locates the NPZ file for the dataset
    2. Loads ground truth velocity and X_dimred
    3. Matches cells between adata and NPZ
    4. Adds ground truth data to adata.obsm
    5. Replaces X_dimred with unified coordinates (if inconsistent)

    Note: PhyloVelo keeps original X_dimred since it only outputs low-dim results.

    Args:
        adata: AnnData object to enrich with ground truth
        dataset_id: Dataset identifier for locating NPZ file
        method: Method name (affects cell name normalization)
        velocity_key: Key for predicted velocity (used to detect PhyloVelo)
        gt_velocity_key: Key to store ground truth velocity
        gt_npz_base_dir: Base directory for GT NPZ files (optional)
        raise_on_failure: If True, raise exception on failure; if False, return error status
        allow_partial_match: Allow partial cell matching
        min_match_ratio: Minimum cell match ratio

    Returns:
        Tuple of (success, message, metadata_dict)
        - success: Boolean indicating if GT recovery succeeded
        - message: Status message describing result
        - metadata: Dict with 'cell_match_ratio', 'n_matched', 'n_total'
    """
    metadata = {'cell_match_ratio': 0.0, 'n_matched': 0, 'n_total': len(adata)}

    # Locate NPZ file
    npz_path, locate_msg = locate_npz_file(dataset_id, gt_npz_base_dir)

    if npz_path is None:
        error_msg = f"GT NPZ file not found for {dataset_id}: {locate_msg}"
        if raise_on_failure:
            raise FileNotFoundError(error_msg)
        return False, error_msg, metadata

    # Load NPZ data
    try:
        npz_data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        error_msg = f"Failed to load NPZ file {npz_path}: {str(e)}"
        if raise_on_failure:
            raise IOError(error_msg)
        return False, error_msg, metadata

    # Extract data from NPZ
    gt_dimred = npz_data.get('gt_dimred')
    X_dimred_npz = npz_data.get('X_dimred')
    cell_names_npz = npz_data.get('cell_names')
    cell_names_unique = npz_data.get('cell_names_unique', None)

    if gt_dimred is None or X_dimred_npz is None or cell_names_npz is None:
        error_msg = f"NPZ file missing required fields: {npz_path}"
        if raise_on_failure:
            raise ValueError(error_msg)
        return False, error_msg, metadata

    # Match cells
    adata_indices, npz_indices, match_msg = match_cell_indices(
        adata_cell_names=adata.obs_names.values,
        npz_cell_names=cell_names_npz,
        npz_cell_names_unique=cell_names_unique,
        allow_partial_match=allow_partial_match,
        min_match_ratio=min_match_ratio,
        method_name=method
    )

    if adata_indices is None or npz_indices is None:
        error_msg = f"Cell matching failed for {dataset_id}: {match_msg}"
        if raise_on_failure:
            raise ValueError(error_msg)
        return False, error_msg, metadata

    # Update metadata
    n_matched = len(adata_indices)
    n_total = len(adata)
    match_ratio = n_matched / n_total if n_total > 0 else 0
    metadata = {
        'cell_match_ratio': match_ratio,
        'n_matched': n_matched,
        'n_total': n_total
    }

    # Check if we need to subset adata
    if n_matched < n_total:
        # Subset adata to matched cells only
        adata._inplace_subset_obs(adata_indices)
        subset_msg = f" (subsetted to {n_matched} matched cells)"
    else:
        subset_msg = ""

    # Add ground truth velocity (low-dim)
    gt_dimred_matched = gt_dimred[npz_indices]
    adata.obsm[f"{gt_velocity_key}_dimred"] = gt_dimred_matched

    # Handle X_dimred replacement
    X_dimred_npz_matched = X_dimred_npz[npz_indices]

    # Check if adata already has X_dimred
    if 'X_dimred' in adata.obsm:
        X_dimred_adata = adata.obsm['X_dimred']

        # Check consistency
        is_consistent = check_dimred_consistency(X_dimred_adata, X_dimred_npz_matched)

        if not is_consistent:
            # PhyloVelo special handling: Keep original X_dimred
            if method == 'PhyloVelo' or 'phylovelo' in velocity_key.lower():
                success_msg = f"GT recovered for {dataset_id}{subset_msg}. " + \
                             f"X_dimred differs but kept original (PhyloVelo low-dim only). {match_msg}"
                return True, success_msg, metadata

            # For other methods: Replace X_dimred and delete old velocity embeddings
            # Delete old velocity embeddings (will be recomputed with unified X_dimred)
            keys_to_delete = []
            for obsm_key in list(adata.obsm.keys()):
                if obsm_key.endswith('_dimred') or obsm_key.endswith('_umap'):
                    if obsm_key not in [f"{gt_velocity_key}_dimred", 'X_dimred', 'X_umap']:
                        keys_to_delete.append(obsm_key)

            for key in keys_to_delete:
                del adata.obsm[key]

            # Replace X_dimred
            adata.obsm['X_dimred'] = X_dimred_npz_matched

            success_msg = f"GT recovered for {dataset_id}{subset_msg}. " + \
                         f"X_dimred replaced with unified coordinates (inconsistency detected). {match_msg}"
            return True, success_msg, metadata
    else:
        # Add X_dimred if missing
        adata.obsm['X_dimred'] = X_dimred_npz_matched

    success_msg = f"GT recovered for {dataset_id}{subset_msg}. {match_msg}"
    return True, success_msg, metadata


# =============================================================================
# TOOL-SPECIFIC PREPROCESSING
# =============================================================================

def _preprocess_phylovelo(adata: AnnData, velocity_key: str) -> List[str]:
    """
    PhyloVelo preprocessing: Handle low-dimensional-only results.

    PhyloVelo outputs velocity directly in low-dimensional space (obsm)
    rather than high-dimensional gene space (layers). We copy the obsm
    velocity to have a _dimred suffix for consistency with other methods.

    Args:
        adata: AnnData object
        velocity_key: Velocity key (typically 'phylovelo_velocity')

    Returns:
        List of applied preprocessing step names
    """
    steps = []

    # PhyloVelo stores velocity in obsm without _dimred suffix
    if 'phylovelo_velocity' in adata.obsm:
        adata.obsm['phylovelo_velocity_dimred'] = adata.obsm['phylovelo_velocity']
        steps.append('copy_phylovelo_velocity_to_dimred')

    return steps


def _preprocess_celldancer(adata: AnnData, velocity_key: str) -> List[str]:
    """
    cellDancer preprocessing: Handle renamed cluster annotations.

    cellDancer uses 'clusters' column instead of standard 'milestone' column.
    We rename it for consistency with other methods.

    Args:
        adata: AnnData object
        velocity_key: Velocity key (typically 'velocity')

    Returns:
        List of applied preprocessing step names
    """
    steps = []

    if 'clusters' in adata.obs.columns and 'milestone' not in adata.obs.columns:
        adata.obs['milestone'] = adata.obs['clusters']
        steps.append('rename_clusters_to_milestone')

    return steps


def _preprocess_topicvelo(adata: AnnData, velocity_key: str) -> List[str]:
    """
    TopicVelo preprocessing: Handle alternative velocity key names.

    TopicVelo may output 'variance_velocity' instead of 'velocity'.
    We check for the requested velocity_key and rename if needed.

    Args:
        adata: AnnData object
        velocity_key: Requested velocity key (typically 'velocity')

    Returns:
        List of applied preprocessing step names
    """
    steps = []

    # Check if velocity_key exists
    vkey_exists = (
        velocity_key in adata.layers or
        f"{velocity_key}_dimred" in adata.obsm or
        f"{velocity_key}_umap" in adata.obsm
    )

    # If not found, try renaming variance_velocity
    if not vkey_exists and 'variance_velocity' in adata.layers:
        adata.layers[velocity_key] = adata.layers['variance_velocity']
        steps.append(f'rename_variance_velocity_to_{velocity_key}')

    return steps


def _preprocess_tfvelo(adata: AnnData, velocity_key: str) -> List[str]:
    """
    TFvelo preprocessing: Compute velocity from velo_hat and fit_scaling_y.

    TFvelo outputs velo_hat (velocity estimate) and fit_scaling_y (scaling factors).
    The actual velocity is computed as: velocity = velo_hat / fit_scaling_y
    (element-wise division per gene).

    Args:
        adata: AnnData object
        velocity_key: Velocity key (should be 'velocity')

    Returns:
        List of applied preprocessing step names
    """
    steps = []

    if 'velo_hat' in adata.layers and 'fit_scaling_y' in adata.var:
        # Compute velocity: element-wise division
        # velo_hat: (n_cells, n_genes)
        # fit_scaling_y: (n_genes,)
        adata.layers['velocity'] = (
            adata.layers['velo_hat'] /
            np.expand_dims(adata.var['fit_scaling_y'].values, axis=0)
        )
        steps.append('compute_velocity_from_velo_hat')

    return steps


def _preprocess_sdevelo(adata: AnnData, velocity_key: str) -> List[str]:
    """
    SDEvelo preprocessing: No pre-GT operations needed.

    SDEvelo outputs velocity directly but needs velocity_graph and
    velocity_embedding computation after GT recovery (see post-GT preprocessing).

    Args:
        adata: AnnData object
        velocity_key: Velocity key (typically 'sde_velocity')

    Returns:
        Empty list (no pre-GT preprocessing)
    """
    return []


# Tool preprocessing registry
TOOL_PREPROCESSING_REGISTRY = {
    'PhyloVelo': _preprocess_phylovelo,
    'cellDancer': _preprocess_celldancer,
    'TFvelo': _preprocess_tfvelo,
    'SDEvelo': _preprocess_sdevelo,
    'TopicVelo': _preprocess_topicvelo,
}


def _apply_tool_specific_preprocessing(
    adata: AnnData,
    method: str,
    velocity_key: str
) -> List[str]:
    """
    Apply tool-specific preprocessing (Phase 1: Pre-GT recovery).

    This handles method-specific data format variations before ground truth
    recovery. Each tool may have unique requirements:
    - PhyloVelo: Low-dimensional results in obsm
    - cellDancer: Renamed cluster annotations
    - TopicVelo: Alternative velocity key names
    - TFvelo: Compute velocity from velo_hat
    - SDEvelo: No pre-processing (only post-GT)

    Args:
        adata: AnnData object to preprocess
        method: Method name (case-insensitive matching)
        velocity_key: Velocity key in adata

    Returns:
        List of applied preprocessing step names (for logging)
    """
    applied_steps = []

    # Case-insensitive method lookup
    method_lower = method.lower()
    for registry_method, handler in TOOL_PREPROCESSING_REGISTRY.items():
        if registry_method.lower() == method_lower:
            steps = handler(adata, velocity_key)
            applied_steps.extend(steps)
            break

    return applied_steps


def _fix_corrupted_neighbors(adata: AnnData):
    """
    Fix corrupted neighbor graph by recomputing from scratch.

    Common causes of corruption:
    - Categorical dtype issues in obs columns or obs_names
    - Duplicate cell names
    - Duplicate cell observations (same expression profile)

    This function:
    1. Converts categorical columns to strings
    2. Makes obs_names unique
    3. Removes duplicate cell observations
    4. Recomputes neighbor graph

    Args:
        adata: AnnData object with corrupted neighbor graph
    """
    # Convert categorical columns to string
    for col in adata.obs.columns:
        if hasattr(adata.obs[col], 'dtype') and isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].astype(str)

    # Convert obs_names if categorical
    if hasattr(adata.obs.index, 'dtype') and isinstance(adata.obs.index.dtype, pd.CategoricalDtype):
        adata.obs.index = adata.obs.index.astype(str)

    # Make obs_names unique
    adata.obs_names_make_unique()

    # Remove duplicate cells (same expression profile)
    duplicated_mask = adata.to_df().duplicated()
    if duplicated_mask.any():
        adata._inplace_subset_obs(~duplicated_mask)

    # Recompute neighbors
    sc.pp.neighbors(adata)


def _apply_post_gt_preprocessing(
    adata: AnnData,
    method: str,
    velocity_key: str
) -> List[str]:
    """
    Apply post-GT preprocessing (Phase 2: After unified X_dimred).

    This computes velocity embeddings using the standardized X_dimred basis
    from GT recovery. Some methods require this step:
    - TFvelo: velocity_graph + velocity_embedding with xkey='M_total'
    - SDEvelo: velocity_graph + velocity_embedding with xkey='Ms'

    Automatically handles corrupted neighbor graphs by recomputing.

    Args:
        adata: AnnData object with unified X_dimred
        method: Method name
        velocity_key: Velocity key

    Returns:
        List of applied preprocessing step names (for logging)
    """
    applied_steps = []
    method_lower = method.lower()

    # TFvelo: Compute velocity embedding
    if method_lower == 'tfvelo' and 'velocity' in adata.layers:
        try:
            scv.tl.velocity_graph(adata, vkey='velocity', xkey='M_total', n_jobs=10)
            scv.tl.velocity_embedding(adata, basis='dimred', vkey='velocity')
            applied_steps.append('compute_tfvelo_velocity_embedding')
        except Exception as e:
            if "neighbor graph" in str(e).lower():
                _fix_corrupted_neighbors(adata)
                scv.tl.velocity_graph(adata, vkey='velocity', xkey='M_total', n_jobs=10)
                scv.tl.velocity_embedding(adata, basis='dimred', vkey='velocity')
                applied_steps.extend(['fix_corrupted_neighbor_graph', 'compute_tfvelo_velocity_embedding'])
            else:
                raise

    # SDEvelo: Compute velocity embedding
    elif method_lower == 'sdevelo' and 'sde_velocity' in adata.layers:
        try:
            scv.tl.velocity_graph(adata, vkey='sde_velocity', xkey='Ms', n_jobs=10)
            scv.tl.velocity_embedding(adata, basis='dimred', vkey='sde_velocity')
            applied_steps.append('compute_sdevelo_velocity_embedding')
        except Exception as e:
            if "neighbor graph" in str(e).lower():
                _fix_corrupted_neighbors(adata)
                scv.tl.velocity_graph(adata, vkey='sde_velocity', xkey='Ms', n_jobs=10)
                scv.tl.velocity_embedding(adata, basis='dimred', vkey='sde_velocity')
                applied_steps.extend(['fix_corrupted_neighbor_graph', 'compute_sdevelo_velocity_embedding'])
            else:
                raise

    return applied_steps


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def determine_embedding_basis(adata: AnnData, velocity_key: str) -> str:
    """
    Determine which embedding basis to use ('dimred' or 'umap').

    Priority:
    1. Check if velocity_key_{basis} exists in obsm
    2. Prefer 'dimred' (simulation data default)
    3. Fall back to 'umap' if dimred not available

    Args:
        adata: AnnData object
        velocity_key: Velocity key name

    Returns:
        Basis string: 'dimred' or 'umap'
    """
    # Check if velocity embeddings exist
    has_vkey_dimred = f"{velocity_key}_dimred" in adata.obsm
    has_vkey_umap = f"{velocity_key}_umap" in adata.obsm

    if has_vkey_dimred:
        return 'dimred'
    elif has_vkey_umap:
        return 'umap'

    # Check X embeddings
    has_X_dimred = 'X_dimred' in adata.obsm
    has_X_umap = 'X_umap' in adata.obsm

    if has_X_dimred:
        return 'dimred'
    elif has_X_umap:
        return 'umap'

    # Default to dimred (simulation data convention)
    return 'dimred'


def ensure_velocity_embedding(
    adata: AnnData,
    velocity_key: str,
    basis: str,
    is_ground_truth: bool = False
):
    """
    Ensure velocity embedding exists, compute if missing.

    For predicted velocity: Computes using scvelo
    For ground truth velocity: Creates embedding from high-dim GT velocity

    Args:
        adata: AnnData object
        velocity_key: Velocity key in layers/obsm
        basis: Embedding basis ('dimred' or 'umap')
        is_ground_truth: If True, handle ground truth velocity specially
    """
    embedding_key = f"{velocity_key}_{basis}"

    # Check if embedding already exists
    if embedding_key in adata.obsm:
        return

    # For ground truth: embedding should be precomputed in NPZ
    if is_ground_truth:
        if f"{velocity_key}_dimred" in adata.obsm:
            return  # Already have GT embedding
        else:
            raise KeyError(f"Ground truth velocity embedding not found: {embedding_key}")

    # For predicted velocity: compute embedding
    if velocity_key not in adata.layers:
        raise KeyError(f"Velocity not found in layers: {velocity_key}")

    # Check if we have necessary data for velocity_embedding
    if f"X_{basis}" not in adata.obsm:
        raise KeyError(f"Embedding not found: X_{basis}")

    # Try to compute velocity_embedding
    try:
        scv.tl.velocity_graph(adata, vkey=velocity_key)
        scv.tl.velocity_embedding(adata, basis=basis, vkey=velocity_key)
    except Exception as e:
        # May fail if neighbor graph missing, try to fix
        if "neighbor" in str(e).lower():
            if f"X_{basis}" in adata.obsm:
                sc.pp.neighbors(adata)
                scv.tl.velocity_graph(adata, vkey=velocity_key)
                scv.tl.velocity_embedding(adata, basis=basis, vkey=velocity_key)
            else:
                raise
        else:
            raise


# =============================================================================
# VELOCITY LOADING & COMPUTATION
# =============================================================================

def _load_velocity_pair(
    adata: AnnData,
    vkey: str,
    gt_key: str,
    use_low_dim: bool,
    basis: str = 'dimred'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load predicted and ground truth velocity arrays.

    Args:
        adata: AnnData object
        vkey: Predicted velocity key
        gt_key: Ground truth velocity key
        use_low_dim: If True, use low-dim embeddings; if False, use high-dim layers
        basis: Embedding basis ('dimred' or 'umap')

    Returns:
        Tuple of (v_pred, v_gt), both as numpy arrays (n_cells, n_dims)

    Raises:
        KeyError: If required velocity data not found
        ValueError: If shapes don't match
    """
    if use_low_dim:
        # Low-dimensional velocity embeddings
        v_pred_key = f"{vkey}_{basis}"
        v_gt_key = f"{gt_key}_{basis}"

        if v_pred_key not in adata.obsm:
            raise KeyError(f"Predicted velocity embedding not found: {v_pred_key}")
        if v_gt_key not in adata.obsm:
            raise KeyError(f"Ground truth velocity embedding not found: {v_gt_key}")

        v_pred = adata.obsm[v_pred_key]
        v_gt = adata.obsm[v_gt_key]
    else:
        # High-dimensional velocity in layers
        if vkey not in adata.layers:
            raise KeyError(f"Predicted velocity not found in layers: {vkey}")
        if gt_key not in adata.layers:
            raise KeyError(f"Ground truth velocity not found in layers: {gt_key}")

        v_pred = adata.layers[vkey]
        v_gt = adata.layers[gt_key]

        # Convert sparse to dense if needed
        if hasattr(v_pred, 'toarray'):
            v_pred = v_pred.toarray()
        if hasattr(v_gt, 'toarray'):
            v_gt = v_gt.toarray()

    # Validate shapes
    if v_pred.shape != v_gt.shape:
        raise ValueError(
            f"Shape mismatch: predicted {v_pred.shape} vs ground truth {v_gt.shape}"
        )

    return v_pred, v_gt


def _cosine_similarity_per_cell(
    v_pred: np.ndarray,
    v_gt: np.ndarray
) -> Tuple[np.ndarray, int, int]:
    """
    Compute per-cell cosine similarity between predicted and ground truth velocity.

    Cosine similarity is mapped from [-1, 1] to [0, 1]:
    - -1 (opposite direction) → 0
    - 0 (orthogonal) → 0.5
    - +1 (same direction) → 1

    Cells with zero velocity (in either predicted or GT) are excluded.

    Args:
        v_pred: Predicted velocity (n_cells, n_dims)
        v_gt: Ground truth velocity (n_cells, n_dims)

    Returns:
        Tuple of (cos_similarities, n_total, n_valid)
        - cos_similarities: Array of cosine similarities [0, 1] for valid cells
        - n_total: Total number of cells
        - n_valid: Number of cells with non-zero velocity

    Raises:
        ValueError: If all cells have zero velocity
    """
    n_cells = v_pred.shape[0]

    # Compute L2 norms (per cell)
    norm_pred = np.linalg.norm(v_pred, axis=1)  # (n_cells,)
    norm_gt = np.linalg.norm(v_gt, axis=1)      # (n_cells,)

    # Find valid cells (non-zero velocity in both)
    valid_mask = (norm_pred > 0) & (norm_gt > 0)
    n_valid = valid_mask.sum()

    if n_valid == 0:
        raise ValueError("All cells have zero velocity (cannot compute cosine similarity)")

    # Compute dot products (per cell)
    dot_products = np.sum(v_pred[valid_mask] * v_gt[valid_mask], axis=1)  # (n_valid,)

    # Compute cosine similarity
    cos_sim = dot_products / (norm_pred[valid_mask] * norm_gt[valid_mask])

    # Map from [-1, 1] to [0, 1]
    cos_sim_01 = (cos_sim + 1.0) / 2.0

    return cos_sim_01, n_cells, n_valid


# =============================================================================
# CSV OUTPUT MANAGEMENT
# =============================================================================

def _update_wide_format_csv(
    output_path: Path,
    method: str,
    dataset_id: str,
    mean_cosine: float
):
    """
    Update wide-format CSV with incremental results.

    CSV Structure:
    - Rows: Methods
    - Columns: Method | dataset1 | dataset2 | ... | AVG | Reversed_rank
    - **Reversed_rank**: 1 = worst (lowest AVG), N = best (highest AVG)
      (FIXED from original implementation which had rank 1 = best)

    The CSV is sorted by AVG descending (best methods first) for readability,
    but the Reversed_rank values reflect that lower AVG = worse performance.

    Args:
        output_path: Path to output CSV file
        method: Method name
        dataset_id: Dataset identifier
        mean_cosine: Mean cosine similarity [0, 1]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing CSV or create new
    if output_path.exists():
        df = pd.read_csv(output_path)

        # Remove computed columns before updating
        if 'AVG' in df.columns:
            df = df.drop(columns=['AVG'])
        if 'Reversed_rank' in df.columns:
            df = df.drop(columns=['Reversed_rank'])
    else:
        df = pd.DataFrame(columns=['Method'])

    # Add dataset column if new
    if dataset_id not in df.columns:
        df[dataset_id] = np.nan

    # Update or insert row
    if method in df['Method'].values:
        row_idx = df[df['Method'] == method].index[0]
        df.at[row_idx, dataset_id] = mean_cosine
    else:
        new_row = {'Method': method, dataset_id: mean_cosine}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Compute AVG across all dataset columns
    data_columns = [col for col in df.columns if col != 'Method']
    df['AVG'] = df[data_columns].mean(axis=1, skipna=True)

    # **FIXED**: Reversed_rank with ascending=True
    # Rank 1 = lowest AVG (worst), Rank N = highest AVG (best)
    df['Reversed_rank'] = df['AVG'].rank(method='min', ascending=True).astype(int)

    # Sort by AVG descending for readability (best methods first)
    df = df.sort_values('AVG', ascending=False).reset_index(drop=True)

    # Round numeric columns
    for col in data_columns:
        if col in df.columns:
            df[col] = df[col].round(3)
    df['AVG'] = df['AVG'].round(3)

    # Save CSV
    df.to_csv(output_path, index=False)


# =============================================================================
# AUTO-DETECTION & WARNINGS
# =============================================================================

def _detect_common_issues(
    adata: AnnData,
    method: str,
    velocity_key: str
) -> List[str]:
    """
    Auto-detect common issues and return warning messages.

    Checks for:
    - Missing velocity key in layers/obsm
    - Corrupted neighbor graph
    - All-zero velocity values
    - Duplicate cell names
    - Missing X_dimred/X_umap
    - Missing milestone/cluster annotations
    - Tool-specific configuration mismatches

    Args:
        adata: AnnData object
        method: Method name
        velocity_key: Velocity key

    Returns:
        List of warning strings (empty if no issues detected)
    """
    warnings_list = []

    # 1. Check velocity key existence
    vkey_in_layers = velocity_key in adata.layers
    vkey_dimred_exists = f"{velocity_key}_dimred" in adata.obsm
    vkey_umap_exists = f"{velocity_key}_umap" in adata.obsm

    if not (vkey_in_layers or vkey_dimred_exists or vkey_umap_exists):
        available_layers = list(adata.layers.keys())[:5]
        warnings_list.append(
            f"WARNING: velocity_key '{velocity_key}' not found in layers or obsm. "
            f"Available layers: {available_layers}"
        )

    # 2. Check for all-zero velocity (in layers only)
    if vkey_in_layers:
        velocity_array = adata.layers[velocity_key]
        if hasattr(velocity_array, 'toarray'):
            velocity_array = velocity_array.toarray()

        max_abs = np.abs(velocity_array).max()
        if max_abs == 0:
            warnings_list.append(
                f"WARNING: All velocities are zero in layers['{velocity_key}']. "
                f"Check if method produced valid output."
            )
        elif max_abs < 1e-10:
            warnings_list.append(
                f"WARNING: Velocities are extremely small (max={max_abs:.2e}). "
                f"May indicate numerical issues."
            )

    # 3. Check for duplicate cell names
    if adata.obs_names.duplicated().any():
        n_dups = adata.obs_names.duplicated().sum()
        warnings_list.append(
            f"WARNING: {n_dups} duplicate cell names detected. "
            f"Will be made unique automatically."
        )

    # 4. Check for X_dimred/X_umap
    has_dimred = 'X_dimred' in adata.obsm
    has_umap = 'X_umap' in adata.obsm

    if not has_dimred and not has_umap:
        warnings_list.append(
            "WARNING: Neither X_dimred nor X_umap found in obsm. "
            "Low-dimensional analysis will fail."
        )
    elif has_umap and not has_dimred:
        warnings_list.append(
            "INFO: Using X_umap instead of X_dimred (simulation data typically uses X_dimred)"
        )

    # 5. Check for milestone/cluster annotation
    if 'milestone' not in adata.obs.columns and 'cluster' not in adata.obs.columns:
        available_cols = list(adata.obs.columns)[:5]
        warnings_list.append(
            "WARNING: No 'milestone' or 'cluster' column in obs. "
            f"Available columns: {available_cols}"
        )

    # 6. Tool-specific checks
    method_lower = method.lower()

    if method_lower == 'phylovelo':
        if 'phylovelo_velocity' not in adata.obsm:
            warnings_list.append(
                "WARNING: PhyloVelo expected to have 'phylovelo_velocity' in obsm. "
                "May fail if velocity_key not set correctly."
            )

    elif method_lower == 'tfvelo':
        if 'velo_hat' not in adata.layers or 'fit_scaling_y' not in adata.var:
            warnings_list.append(
                "WARNING: TFvelo preprocessing expects 'velo_hat' in layers "
                "and 'fit_scaling_y' in var. Check if preprocessing already done."
            )

    elif method_lower == 'sdevelo':
        if 'sde_velocity' not in adata.layers and velocity_key != 'sde_velocity':
            warnings_list.append(
                "WARNING: SDEvelo typically uses velocity_key='sde_velocity'. "
                f"Current velocity_key='{velocity_key}' may not match."
            )

    return warnings_list


# =============================================================================
# LOGGING SETUP
# =============================================================================

def _setup_logging(
    log_file: Optional[Path] = None,
    verbose: bool = True
) -> logging.Logger:
    """
    Setup unified error-only logging.

    - Console: INFO level (if verbose=True), suppressed otherwise
    - File: ERROR level only (structured messages)

    Args:
        log_file: Path to error log file (optional)
        verbose: If True, print INFO messages to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('groundtruth_correlation')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler (INFO or suppressed)
    if verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console)

    # File handler (ERROR only)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# MAIN API FUNCTION
# =============================================================================

def calculate_groundtruth_correlation(
    adata_or_path: Union[str, Path, AnnData],
    method: str,
    dataset_id: str,
    output_csv: str,
    velocity_key: str = "velocity",
    gt_velocity_key: str = "ground_truth_velocity",
    use_low_dim: bool = True,
    gt_npz_base_dir: Optional[str] = None,
    apply_tool_preprocessing: bool = True,
    raise_on_gt_failure: bool = True,
    allow_partial_cell_match: bool = True,
    min_cell_match_ratio: float = 0.95,
) -> Dict[str, Any]:
    """
    Calculate cosine similarity between predicted velocity and ground truth velocity.

    This is the main API function for single-dataset correlation analysis.

    Workflow:
    1. Load adata from file or use provided object
    2. Auto-detect common data issues
    3. Apply Phase 1 preprocessing (tool-specific)
    4. Recover ground truth velocity from NPZ file
    5. Apply Phase 2 preprocessing (velocity embedding computation)
    6. Compute cosine similarity
    7. Update wide-format CSV output

    Args:
        adata_or_path: AnnData object or path to H5AD file
        method: Method name (e.g., 'VeloVAE', 'scKINETICS', 'PhyloVelo')
        dataset_id: Dataset identifier (e.g., 'bifurcating_cell1000_gene1000')
        output_csv: Path to output CSV file (wide format: Method x datasets)
        velocity_key: Key for predicted velocity in adata.layers or adata.obsm
        gt_velocity_key: Key for ground truth velocity
        use_low_dim: Use low-dimensional (2D) velocity embeddings for comparison
                     If False, use high-dimensional velocity from layers
        gt_npz_base_dir: Base directory for precomputed GT NPZ files (optional)
        apply_tool_preprocessing: Apply tool-specific preprocessing
        raise_on_gt_failure: Raise exception if GT recovery fails (vs. silent failure)
        allow_partial_cell_match: Allow processing when not all cells match GT data
        min_cell_match_ratio: Minimum ratio of cells that must match

    Returns:
        Dictionary with keys:
            - 'method': Method name
            - 'dataset_id': Dataset identifier
            - 'mean_cosine': Mean cosine similarity [0, 1]
            - 'n_cells_total': Total number of cells
            - 'n_cells_valid': Number of cells with non-zero velocity
            - 'cell_match_ratio': Ratio of cells matched to GT
            - 'gt_recovery_status': Status message from GT recovery
            - 'preprocessing_applied': List of preprocessing step names
            - 'warnings': List of warning messages from auto-detection

    Raises:
        FileNotFoundError: If input file doesn't exist or GT NPZ not found
        ValueError: If GT recovery fails, velocity shapes don't match, or all velocities zero
        KeyError: If required velocity keys are missing
    """
    # Load adata
    if isinstance(adata_or_path, (str, Path)):
        adata_path = Path(adata_or_path)
        if not adata_path.exists():
            raise FileNotFoundError(f"Input file not found: {adata_path}")
        adata = sc.read_h5ad(adata_path)
    elif isinstance(adata_or_path, AnnData):
        adata = adata_or_path.copy()  # Work on copy to avoid modifying original
    else:
        raise TypeError(f"adata_or_path must be str, Path, or AnnData, got {type(adata_or_path)}")

    # Auto-detect common issues
    warnings_list = _detect_common_issues(adata, method, velocity_key)

    # Phase 1: Tool-specific preprocessing (pre-GT recovery)
    preprocessing_steps = []
    if apply_tool_preprocessing:
        steps = _apply_tool_specific_preprocessing(adata, method, velocity_key)
        preprocessing_steps.extend(steps)

    # Recover ground truth velocity
    gt_success, gt_message, gt_metadata = recover_ground_truth_velocity(
        adata=adata,
        dataset_id=dataset_id,
        method=method,
        velocity_key=velocity_key,
        gt_velocity_key=gt_velocity_key,
        gt_npz_base_dir=gt_npz_base_dir,
        raise_on_failure=raise_on_gt_failure,
        allow_partial_match=allow_partial_cell_match,
        min_match_ratio=min_cell_match_ratio
    )

    if not gt_success:
        # GT recovery failed
        result = {
            'method': method,
            'dataset_id': dataset_id,
            'mean_cosine': np.nan,
            'n_cells_total': len(adata),
            'n_cells_valid': 0,
            'cell_match_ratio': 0.0,
            'gt_recovery_status': gt_message,
            'preprocessing_applied': preprocessing_steps,
            'warnings': warnings_list,
            'success': False
        }
        return result

    # Phase 2: Post-GT preprocessing (velocity embedding computation)
    if apply_tool_preprocessing:
        steps = _apply_post_gt_preprocessing(adata, method, velocity_key)
        preprocessing_steps.extend(steps)

    # Determine embedding basis
    basis = determine_embedding_basis(adata, velocity_key)

    # Ensure velocity embeddings exist
    try:
        ensure_velocity_embedding(adata, velocity_key, basis, is_ground_truth=False)
        ensure_velocity_embedding(adata, gt_velocity_key, basis, is_ground_truth=True)
    except Exception as e:
        raise RuntimeError(f"Failed to ensure velocity embeddings: {str(e)}")

    # Load velocity pair
    v_pred, v_gt = _load_velocity_pair(
        adata=adata,
        vkey=velocity_key,
        gt_key=gt_velocity_key,
        use_low_dim=use_low_dim,
        basis=basis
    )

    # Compute cosine similarity
    cos_similarities, n_total, n_valid = _cosine_similarity_per_cell(v_pred, v_gt)
    mean_cosine = cos_similarities.mean()

    # Update CSV
    _update_wide_format_csv(
        output_path=Path(output_csv),
        method=method,
        dataset_id=dataset_id,
        mean_cosine=mean_cosine
    )

    # Return result
    result = {
        'method': method,
        'dataset_id': dataset_id,
        'mean_cosine': float(mean_cosine),
        'n_cells_total': n_total,
        'n_cells_valid': n_valid,
        'cell_match_ratio': gt_metadata['cell_match_ratio'],
        'gt_recovery_status': gt_message,
        'preprocessing_applied': preprocessing_steps,
        'warnings': warnings_list,
        'success': True
    }

    return result


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def calculate_groundtruth_correlation_batch(
    metadata_csv: str,
    output_csv: str,
    output_dir: Optional[str] = None,
    velocity_key_column: str = 'vkey',
    method_column: str = 'method',
    dataset_id_column: str = 'id',
    h5ad_path_column: str = 'path',
    gt_npz_base_dir: Optional[str] = None,
    apply_tool_preprocessing: bool = True,
    error_log_file: Optional[str] = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Batch process multiple method-dataset pairs from CSV metadata.

    CSV format (required columns):
    - method_column (default 'method'): Method name
    - dataset_id_column (default 'id'): Dataset identifier
    - h5ad_path_column (default 'path'): Full path to H5AD file
    - velocity_key_column (default 'vkey'): Velocity key (optional, defaults to 'velocity')

    Args:
        metadata_csv: CSV file with method-dataset metadata
        output_csv: Path to output CSV (wide format, updated incrementally)
        output_dir: Output directory (inferred from output_csv if not provided)
        velocity_key_column: Column name for velocity key
        method_column: Column name for method
        dataset_id_column: Column name for dataset ID
        h5ad_path_column: Column name for H5AD file path
        gt_npz_base_dir: Base directory for GT NPZ files
        apply_tool_preprocessing: Apply tool-specific preprocessing
        error_log_file: Path to error log file (default: {output_dir}/errors.log)
        n_jobs: Number of parallel jobs (currently only n_jobs=1 supported)
        verbose: Print progress messages

    Returns:
        DataFrame with processing summary (status, errors)
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(output_csv).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup error log
    if error_log_file is None:
        error_log_file = output_dir / 'errors.log'
    else:
        error_log_file = Path(error_log_file)

    logger = _setup_logging(error_log_file, verbose=verbose)

    # Load metadata CSV
    try:
        metadata_df = pd.read_csv(metadata_csv)
    except Exception as e:
        logger.error(f"Failed to load metadata CSV {metadata_csv}: {str(e)}")
        raise

    # Validate required columns
    required_columns = [method_column, dataset_id_column, h5ad_path_column]
    missing_columns = [col for col in required_columns if col not in metadata_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")

    # Add velocity_key column if missing (default to 'velocity')
    if velocity_key_column not in metadata_df.columns:
        metadata_df[velocity_key_column] = 'velocity'

    # Process each row
    n_total = len(metadata_df)
    results = []

    if verbose:
        print(f"\n{'='*80}")
        print(f"Processing {n_total} datasets from {metadata_csv}")
        print(f"Output CSV: {output_csv}")
        print(f"Error log: {error_log_file}")
        print(f"{'='*80}\n")

    for idx, row in metadata_df.iterrows():
        method = row[method_column]
        dataset_id = row[dataset_id_column]
        h5ad_path = row[h5ad_path_column]
        velocity_key = row[velocity_key_column]

        if verbose:
            print(f"[{idx+1}/{n_total}] Processing {method} | {dataset_id}...", end=' ')

        try:
            result = calculate_groundtruth_correlation(
                adata_or_path=h5ad_path,
                method=method,
                dataset_id=dataset_id,
                output_csv=output_csv,
                velocity_key=velocity_key,
                gt_npz_base_dir=gt_npz_base_dir,
                apply_tool_preprocessing=apply_tool_preprocessing,
                raise_on_gt_failure=False,  # Don't stop batch on individual failures
            )

            if result['success']:
                if verbose:
                    print(f"✓ Mean cosine: {result['mean_cosine']:.3f}")
                results.append({
                    'method': method,
                    'dataset_id': dataset_id,
                    'status': 'success',
                    'mean_cosine': result['mean_cosine'],
                    'error': None
                })
            else:
                if verbose:
                    print(f"✗ Failed: {result['gt_recovery_status']}")
                logger.error(f"GT_RECOVERY | {method} | {dataset_id} | {result['gt_recovery_status']}")
                results.append({
                    'method': method,
                    'dataset_id': dataset_id,
                    'status': 'failed',
                    'mean_cosine': np.nan,
                    'error': result['gt_recovery_status']
                })

        except Exception as e:
            if verbose:
                print(f"✗ Error: {str(e)[:50]}...")
            logger.error(f"PROCESSING_ERROR | {method} | {dataset_id} | {str(e)}")
            results.append({
                'method': method,
                'dataset_id': dataset_id,
                'status': 'error',
                'mean_cosine': np.nan,
                'error': str(e)
            })

    # Create summary DataFrame
    summary_df = pd.DataFrame(results)

    # Print summary statistics
    if verbose:
        n_success = (summary_df['status'] == 'success').sum()
        n_failed = (summary_df['status'] == 'failed').sum()
        n_error = (summary_df['status'] == 'error').sum()
        success_rate = (n_success / n_total * 100) if n_total > 0 else 0

        print(f"\n{'='*80}")
        print(f"Batch processing complete")
        print(f"  Total: {n_total} datasets")
        print(f"  Success: {n_success} ({success_rate:.1f}%)")
        print(f"  GT recovery failed: {n_failed}")
        print(f"  Processing errors: {n_error}")
        print(f"  Output: {output_csv}")
        print(f"  Error log: {error_log_file}")
        print(f"{'='*80}\n")

    return summary_df


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for Ground Truth Correlation"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate ground truth correlation for RNA velocity predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file mode
  python groundtruth_correlation.py \\
      --input result.h5ad \\
      --method VeloVAE \\
      --dataset-id bifurcating_cell1000_gene1000 \\
      --output-csv results/correlation.csv

  # Batch mode
  python groundtruth_correlation.py \\
      --metadata-csv SIMDATA_ALL.csv \\
      --output-csv results/correlation.csv \\
      --output-dir results/ \\
      --verbose

  # Custom GT directory
  python groundtruth_correlation.py \\
      --input result.h5ad \\
      --method scKINETICS \\
      --dataset-id cycle-simple_cell5000_gene500 \\
      --output-csv results/correlation.csv \\
      --gt-npz-dir /custom/path/to/npz \\
      --velocity-key sck_velocity
        """
    )

    # Mode selection (mutually exclusive)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--input', type=str,
                      help='Input H5AD file (single-file mode)')
    mode.add_argument('--metadata-csv', type=str,
                      help='CSV metadata file (batch mode)')

    # Single-file mode parameters
    single_group = parser.add_argument_group('single-file mode parameters')
    single_group.add_argument('--method', type=str,
                              help='Method name (required for single-file mode)')
    single_group.add_argument('--dataset-id', type=str,
                              help='Dataset ID (required for single-file mode)')

    # Batch mode parameters
    batch_group = parser.add_argument_group('batch mode parameters')
    batch_group.add_argument('--method-column', type=str, default='method',
                             help='Column name for method (default: method)')
    batch_group.add_argument('--dataset-id-column', type=str, default='id',
                             help='Column name for dataset ID (default: id)')
    batch_group.add_argument('--h5ad-path-column', type=str, default='path',
                             help='Column name for H5AD path (default: path)')
    batch_group.add_argument('--velocity-key-column', type=str, default='vkey',
                             help='Column name for velocity key (default: vkey)')

    # Common parameters
    parser.add_argument('--output-csv', type=str, required=True,
                        help='Output CSV file (wide format)')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory (default: inferred from output-csv)')
    parser.add_argument('--velocity-key', type=str, default='velocity',
                        help='Velocity key in adata.layers/obsm (default: velocity)')
    parser.add_argument('--gt-velocity-key', type=str, default='ground_truth_velocity',
                        help='GT velocity key (default: ground_truth_velocity)')
    parser.add_argument('--gt-npz-dir', type=str,
                        help='GT NPZ base directory (optional)')
    parser.add_argument('--use-high-dim', action='store_true',
                        help='Use high-dimensional velocity (default: low-dim)')
    parser.add_argument('--no-tool-preprocessing', action='store_true',
                        help='Disable tool-specific preprocessing')
    parser.add_argument('--no-raise-on-failure', action='store_true',
                        help='Continue on GT recovery failure (default: raise)')
    parser.add_argument('--min-cell-match-ratio', type=float, default=0.95,
                        help='Minimum cell match ratio (default: 0.95)')
    parser.add_argument('--error-log', type=str,
                        help='Error log file (default: {output-dir}/errors.log)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')

    args = parser.parse_args()

    # Validate mode-specific arguments
    if args.input and not (args.method and args.dataset_id):
        parser.error('--method and --dataset-id are required for single-file mode')

    # Infer output_dir
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    error_log = Path(args.error_log) if args.error_log else (output_dir / 'errors.log')
    logger = _setup_logging(error_log, verbose=args.verbose)

    # Single-file mode
    if args.input:
        try:
            result = calculate_groundtruth_correlation(
                adata_or_path=args.input,
                method=args.method,
                dataset_id=args.dataset_id,
                output_csv=args.output_csv,
                velocity_key=args.velocity_key,
                gt_velocity_key=args.gt_velocity_key,
                use_low_dim=not args.use_high_dim,
                gt_npz_base_dir=args.gt_npz_dir,
                apply_tool_preprocessing=not args.no_tool_preprocessing,
                raise_on_gt_failure=not args.no_raise_on_failure,
                min_cell_match_ratio=args.min_cell_match_ratio,
            )

            print(f"\n✓ Success: {args.method} | {args.dataset_id}")
            print(f"  Mean cosine: {result['mean_cosine']:.3f}")
            print(f"  Valid cells: {result['n_cells_valid']}/{result['n_cells_total']}")
            print(f"  Output: {args.output_csv}")

        except Exception as e:
            logger.error(f"{args.method}_{args.dataset_id}: {str(e)}")
            print(f"\n✗ Failed: {args.method} | {args.dataset_id}")
            print(f"  Error: {str(e)}")
            sys.exit(1)

    # Batch mode
    else:
        summary = calculate_groundtruth_correlation_batch(
            metadata_csv=args.metadata_csv,
            output_csv=args.output_csv,
            output_dir=output_dir,
            velocity_key_column=args.velocity_key_column,
            method_column=args.method_column,
            dataset_id_column=args.dataset_id_column,
            h5ad_path_column=args.h5ad_path_column,
            gt_npz_base_dir=args.gt_npz_dir,
            apply_tool_preprocessing=not args.no_tool_preprocessing,
            error_log_file=error_log,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
