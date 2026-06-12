#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ground Truth Correlation

Compare predicted velocity and ground truth velocity by cosine similarity on a
shared low-dimensional basis.

Design principles:
- Only low-dimensional comparison is supported; there is no high-dimensional cosine path.
- By default, the GT embedding is computed on the fly from adata.layers[gt_velocity_key].
- Reference files are optional overrides used to restore basis coordinates or provide
  precomputed low-dimensional GT vectors.
- Generic cleanup, neighbor repair, and embedding recomputation fallbacks are retained.
- No special handling based on specific tool names is kept.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import scanpy as sc
    import scvelo as scv
    from anndata import AnnData
except ImportError as exc:
    print(f"ERROR: Required package not found: {exc}")
    print("Please install: pip install scanpy scvelo anndata")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


SIMULATION_ADD_DATASETS = {
    "1_linear-simple_cell1000_gene10000",
    "2_linear-simple_cell10000_gene1000",
    "3_cycle-simple_cell10000_gene1000",
    "4_cycle-simple_cell1000_gene10000",
    "5_bifurcating_cell1000_gene10000",
    "6_bifurcating_cell500_gene10000",
    "7_bifurcating-loop_cell1000_gene10000",
    "8_bifurcating-loop_cell1000_gene10000",
    "9_consecutive-bifurcating_cell500_gene10000",
    "10_consecutive-bifurcating_cell1000_gene10000",
    "11_trifurcating_cell1000_gene10000",
    "12_trifurcating_cell1000_gene10000",
    "13_trifurcating_cell1000_gene10000",
    "14_linear-bifurcating_cell10000_gene1000",
    "15_linear-linear_cell10000_gene500",
    "16_bifurcating_cell500_gene500",
}

TOPOLOGY_VARIANTS = {
    "bifurcating_loop": "bifurcating-loop",
    "cycle_simple": "cycle-simple",
    "linear_simple_subset": "linear-simple",
    "linear_simple": "linear-simple",
    "linear_bifurcating": "linear-bifurcating",
    "consecutive_bifurcating": "consecutive-bifurcating",
    "linear_linear": "linear-linear",
    "genesub_bifurcating": "genesub-bifurcating",
    "cellsub_bifurcating": "cellsub-bifurcating",
    "bursting_tree": "Bursting-tree",
}


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalize dataset naming variants for reference file lookup."""
    if not dataset_name:
        return dataset_name

    lowered = dataset_name.lower()
    replacements = sorted(TOPOLOGY_VARIANTS.items(), key=lambda item: len(item[0]), reverse=True)
    for old_name, new_name in replacements:
        if old_name in lowered:
            start = lowered.find(old_name)
            if start != -1:
                return dataset_name[:start] + new_name + dataset_name[start + len(old_name):]
    return dataset_name


def normalize_cell_name(cell_name: str, adjust_numeric_index: bool = False) -> str:
    """Normalize cell names, adding a `cell` prefix when needed."""
    cell_name_str = str(cell_name).strip()

    if cell_name_str.lower().startswith("cell"):
        return cell_name_str

    if cell_name_str.isdigit():
        numeric_idx = int(cell_name_str)
        if adjust_numeric_index:
            numeric_idx += 1
        return f"cell{numeric_idx}"

    return f"cell{cell_name_str}"


def infer_topology_from_dataset_id(dataset_id: str) -> Optional[str]:
    """Infer the reference subdirectory from a dataset identifier."""
    if not dataset_id:
        return None

    normalized_id = normalize_dataset_name(dataset_id)
    if normalized_id in SIMULATION_ADD_DATASETS:
        return "simulation-add"

    topology_patterns = [
        ("consecutive-bifurcating", "consecutive-bifurcating"),
        ("genesub-bifurcating", "genesub-bifurcating"),
        ("cellsub-bifurcating", "cellsub-bifurcating"),
        ("Bursting-tree", "Bursting-tree"),
        ("bursting-tree", "Bursting-tree"),
        ("bifurcating-loop", "bifurcating-loop"),
        ("linear-simple", "linear-simple"),
        ("cycle-simple", "cycle-simple"),
        ("linear-bifurcating", "linear-bifurcating"),
        ("linear-linear", "linear-linear"),
        ("bifurcating", "bifurcating"),
        ("trifurcating", "trifurcating"),
        ("disconnected", "disconnected"),
        ("lineage-tracing", "lineage-tracing"),
    ]

    lowered = normalized_id.lower()
    for pattern, topology in topology_patterns:
        if pattern.lower() in lowered:
            return topology
    return None


def locate_npz_file(dataset_id: str, base_dir: Optional[str]) -> Tuple[Optional[Path], str]:
    """Locate a `*_reference_data.npz` file, keeping hyphen/underscore normalization."""
    if base_dir is None:
        return None, "reference directory not provided"

    base_path = Path(base_dir)
    if not base_path.exists():
        return None, f"reference directory does not exist: {base_dir}"

    topology = infer_topology_from_dataset_id(dataset_id)
    if topology is None:
        return None, f"cannot infer topology from dataset_id: {dataset_id}"

    topology_dir = base_path / topology
    if not topology_dir.exists():
        return None, f"topology directory not found: {topology_dir}"

    candidate_ids: List[str] = []
    for candidate in [
        dataset_id,
        normalize_dataset_name(dataset_id),
        dataset_id.replace("_", "-"),
        dataset_id.replace("-", "_"),
    ]:
        if candidate not in candidate_ids:
            candidate_ids.append(candidate)

    for candidate_id in candidate_ids:
        npz_path = topology_dir / f"{candidate_id}_reference_data.npz"
        if npz_path.exists():
            return npz_path, f"found reference file: {npz_path.name}"

    return None, f"reference file not found in {topology_dir}"


def check_dimred_consistency(
    adata_dimred: np.ndarray,
    npz_dimred: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Check whether two low-dimensional coordinate arrays are effectively identical."""
    if adata_dimred.shape != npz_dimred.shape:
        return False
    return np.allclose(adata_dimred, npz_dimred, rtol=rtol, atol=atol)


def get_basis_array(adata: AnnData, basis_name: str) -> Optional[np.ndarray]:
    """Return the coordinate array for a given basis."""
    return adata.obsm.get(f"X_{basis_name}")


def resolve_basis_name(
    adata: AnnData,
    basis_name: str = "dimred",
    *,
    allow_default_fallback: bool = True,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Resolve the effective basis name.

    Rules:
    - If the user explicitly requests a basis and it is missing, fail immediately.
    - Only when the default `dimred` is missing do we fall back to X_umap -> X_tsne.
    """
    requested_key = f"X_{basis_name}"
    if requested_key in adata.obsm:
        return basis_name

    is_default_dimred = basis_name == "dimred" and allow_default_fallback
    if not is_default_dimred:
        raise KeyError(f"Requested basis not found in adata.obsm: {requested_key}")

    warning_msg = "WARNING: X_dimred not found, trying fallback bases in order: X_umap -> X_tsne"
    if logger:
        logger.warning(warning_msg)
    else:
        print(warning_msg)

    for fallback_basis in ["umap", "tsne"]:
        if f"X_{fallback_basis}" in adata.obsm:
            return fallback_basis

    raise KeyError("No usable non-PCA basis found. Checked X_dimred, X_umap, X_tsne.")


def _ensure_adata(adata_or_path: Union[str, Path, AnnData]) -> AnnData:
    """Coerce the input into an AnnData copy."""
    if isinstance(adata_or_path, AnnData):
        return adata_or_path.copy()

    adata_path = Path(adata_or_path)
    if not adata_path.exists():
        raise FileNotFoundError(f"Input file not found: {adata_path}")
    return sc.read_h5ad(adata_path)


def _coerce_obs_metadata(adata: AnnData) -> None:
    """Normalize obs column types and obs_names."""
    for col in adata.obs.columns:
        if hasattr(adata.obs[col], "dtype") and isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].astype(str)

    if hasattr(adata.obs.index, "dtype") and isinstance(adata.obs.index.dtype, pd.CategoricalDtype):
        adata.obs.index = adata.obs.index.astype(str)

    adata.obs_names_make_unique()


def _remove_zero_expression_cells(adata: AnnData, xkey_to_use: Optional[str]) -> None:
    """Remove cells with all-zero expression in the selected expression layer."""
    layer_for_zero_filter = None
    if "spliced" in adata.layers:
        layer_for_zero_filter = "spliced"
    elif xkey_to_use in adata.layers:
        layer_for_zero_filter = xkey_to_use

    if layer_for_zero_filter is None:
        return

    nonzero_mask = np.asarray(adata.layers[layer_for_zero_filter].sum(axis=1)).ravel() != 0
    if not np.all(nonzero_mask):
        adata._inplace_subset_obs(nonzero_mask)


def _remove_duplicate_expression_cells(adata: AnnData) -> None:
    """Remove cells with duplicated expression profiles."""
    duplicated_mask = adata.to_df().duplicated()
    if duplicated_mask.any():
        adata._inplace_subset_obs(~duplicated_mask)


def _cleanup_for_neighbors(adata: AnnData, xkey_to_use: Optional[str]) -> None:
    """Perform generic cleanup before rebuilding neighbors."""
    _coerce_obs_metadata(adata)
    _remove_zero_expression_cells(adata, xkey_to_use)
    _remove_duplicate_expression_cells(adata)


def _recompute_neighbors_after_cleanup(
    adata: AnnData,
    xkey_to_use: Optional[str],
    max_rounds: int = 3,
) -> None:
    """Clean the data and iteratively rebuild neighbors."""
    _cleanup_for_neighbors(adata, xkey_to_use)

    for _ in range(max_rounds):
        adata.uns.pop("neighbors", None)
        for key in ["distances", "connectivities"]:
            if key in adata.obsp:
                del adata.obsp[key]

        sc.pp.neighbors(adata)

        neighbor_counts = np.asarray((adata.obsp["distances"] > 0).sum(1)).ravel()
        zero_neighbor_mask = neighbor_counts == 0
        if not zero_neighbor_mask.any():
            return

        adata._inplace_subset_obs(~zero_neighbor_mask)
        _cleanup_for_neighbors(adata, xkey_to_use)

    raise ValueError("Neighbor graph rebuild failed: zero-neighbor cells remain after cleanup")


def _determine_velocity_xkey(adata: AnnData) -> Optional[str]:
    """Pick the most suitable expression layer for velocity_graph."""
    for key in ["Ms", "spliced", "M_total"]:
        if key in adata.layers:
            return key
    return None


def _ensure_velocity_embedding(
    adata: AnnData,
    velocity_key: str,
    basis_name: str,
) -> None:
    """Ensure that a high-dimensional velocity layer has a low-dimensional embedding."""
    embedding_key = f"{velocity_key}_{basis_name}"
    if embedding_key in adata.obsm:
        return

    if velocity_key not in adata.layers:
        raise KeyError(f"Velocity layer not found: adata.layers['{velocity_key}']")

    basis_key = f"X_{basis_name}"
    if basis_key not in adata.obsm:
        raise KeyError(f"Basis coordinates not found: adata.obsm['{basis_key}']")

    xkey_to_use = _determine_velocity_xkey(adata)

    try:
        if xkey_to_use:
            scv.tl.velocity_graph(adata, vkey=velocity_key, xkey=xkey_to_use, n_jobs=16, approx=True)
        else:
            scv.tl.velocity_graph(adata, vkey=velocity_key, n_jobs=16, approx=True)
        scv.tl.velocity_embedding(adata, basis=basis_name, vkey=velocity_key)
    except Exception as exc:
        error_msg = str(exc).lower()
        if "neighbor" not in error_msg:
            raise

        _recompute_neighbors_after_cleanup(adata, xkey_to_use)
        if xkey_to_use:
            scv.tl.velocity_graph(adata, vkey=velocity_key, xkey=xkey_to_use, n_jobs=16, approx=True)
        else:
            scv.tl.velocity_graph(adata, vkey=velocity_key, n_jobs=16, approx=True)
        scv.tl.velocity_embedding(adata, basis=basis_name, vkey=velocity_key)

    if embedding_key not in adata.obsm:
        raise ValueError(f"velocity embedding was not created: {embedding_key}")


def match_cell_indices(
    adata_cell_names: np.ndarray,
    npz_cell_names: np.ndarray,
    npz_cell_names_unique: Optional[np.ndarray] = None,
    allow_partial_match: bool = True,
    min_match_ratio: float = 0.95,
    adjust_numeric_index: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    First match by cell names, then fall back to `cell_names_unique`.

    This function only returns name-based matching results. Position-based fallback is
    applied by the caller only if coordinate consistency can be verified.
    """
    normalized_adata_names = np.array(
        [normalize_cell_name(name, adjust_numeric_index=adjust_numeric_index) for name in adata_cell_names]
    )

    npz_name_to_idx = {str(name): idx for idx, name in enumerate(npz_cell_names)}
    adata_indices: List[int] = []
    npz_indices: List[int] = []

    for adata_idx, cell_name in enumerate(normalized_adata_names):
        if cell_name in npz_name_to_idx:
            adata_indices.append(adata_idx)
            npz_indices.append(npz_name_to_idx[cell_name])

    matched_count = len(adata_indices)
    total_count = len(normalized_adata_names)

    if matched_count == total_count:
        return np.arange(total_count), np.array(npz_indices), "cell_names exact match"

    if npz_cell_names_unique is not None:
        npz_unique_to_idx = {str(name): idx for idx, name in enumerate(npz_cell_names_unique)}
        adata_indices_unique: List[int] = []
        npz_indices_unique: List[int] = []

        for adata_idx, cell_name in enumerate(normalized_adata_names):
            if cell_name in npz_unique_to_idx:
                adata_indices_unique.append(adata_idx)
                npz_indices_unique.append(npz_unique_to_idx[cell_name])

        matched_count_unique = len(adata_indices_unique)
        if matched_count_unique == total_count:
            return np.arange(total_count), np.array(npz_indices_unique), "cell_names_unique exact match"

        if matched_count_unique > matched_count:
            adata_indices = adata_indices_unique
            npz_indices = npz_indices_unique
            matched_count = matched_count_unique

    if matched_count > 0:
        match_ratio = matched_count / total_count
        if allow_partial_match and match_ratio >= min_match_ratio:
            return (
                np.array(adata_indices),
                np.array(npz_indices),
                f"partial name match ({matched_count}/{total_count}={match_ratio:.1%})",
            )
        return None, None, f"match ratio too low ({matched_count}/{total_count}={match_ratio:.1%})"

    return None, None, "cell names do not match"


def _load_reference_arrays(
    npz_data: Any,
    reference_gt_key: str,
    reference_basis_key: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load the minimal set of arrays needed from a reference npz."""
    gt_dimred = npz_data.get(reference_gt_key)
    basis_coords = npz_data.get(reference_basis_key)
    cell_names = npz_data.get("cell_names")
    cell_names_unique = npz_data.get("cell_names_unique", None)
    return gt_dimred, basis_coords, cell_names, cell_names_unique


def _align_with_reference(
    adata: AnnData,
    dataset_id: str,
    basis_name: str,
    *,
    velocity_key: str,
    gt_npz_base_dir: Optional[str],
    reference_gt_key: str,
    reference_basis_key: str,
    gt_velocity_key: str,
    allow_partial_match: bool,
    min_cell_match_ratio: float,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Align cells and basis coordinates using an optional reference and inject low-dimensional GT when available.
    """
    metadata = {"cell_match_ratio": 1.0, "n_matched": len(adata), "n_total": len(adata), "used_reference": False}
    npz_path, locate_msg = locate_npz_file(dataset_id, gt_npz_base_dir)
    if npz_path is None:
        return False, locate_msg, metadata

    try:
        npz_data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        raise IOError(f"Failed to load reference npz {npz_path}: {exc}") from exc

    gt_dimred, basis_coords, cell_names, cell_names_unique = _load_reference_arrays(
        npz_data=npz_data,
        reference_gt_key=reference_gt_key,
        reference_basis_key=reference_basis_key,
    )
    if cell_names is None:
        raise ValueError(f"Reference file missing required field: cell_names ({npz_path})")
    if gt_dimred is None:
        warning_msg = (
            f"WARNING: reference file {npz_path.name} does not contain '{reference_gt_key}'. "
            "The script will fall back to projecting adata.layers ground truth if available."
        )
        if logger:
            logger.warning(warning_msg)
        else:
            print(warning_msg)

    _coerce_obs_metadata(adata)
    adata_cell_names = np.array(adata.obs_names)
    adata_indices, npz_indices, match_msg = match_cell_indices(
        adata_cell_names=adata_cell_names,
        npz_cell_names=cell_names,
        npz_cell_names_unique=cell_names_unique,
        allow_partial_match=allow_partial_match,
        min_match_ratio=min_cell_match_ratio,
    )

    if adata_indices is None:
        if basis_coords is not None and len(adata_cell_names) == len(cell_names):
            existing_basis = get_basis_array(adata, basis_name)
            if existing_basis is not None and check_dimred_consistency(existing_basis, basis_coords):
                adata_indices = np.arange(len(adata_cell_names))
                npz_indices = np.arange(len(cell_names))
                match_msg = "position fallback after name match failure"
            else:
                raise ValueError(f"Cell matching failed for {dataset_id}: {match_msg}")
        else:
            raise ValueError(f"Cell matching failed for {dataset_id}: {match_msg}")

    n_matched = len(adata_indices)
    n_total = len(adata_cell_names)
    metadata = {
        "cell_match_ratio": (n_matched / n_total) if n_total > 0 else 0.0,
        "n_matched": n_matched,
        "n_total": n_total,
        "used_reference": True,
    }

    if n_matched < n_total:
        adata._inplace_subset_obs(adata_indices)

    if gt_dimred is not None:
        adata.obsm[f"{gt_velocity_key}_{basis_name}"] = gt_dimred[npz_indices]

    if basis_coords is not None:
        new_basis_coords = basis_coords[npz_indices]
        existing_basis_coords = get_basis_array(adata, basis_name)
        basis_changed = existing_basis_coords is None or not check_dimred_consistency(
            existing_basis_coords,
            new_basis_coords,
        )

        if basis_changed:
            for stale_key in [
                f"{velocity_key}_{basis_name}",
                f"{velocity_key}_umap",
                f"{velocity_key}_tsne",
            ]:
                if stale_key in adata.obsm:
                    del adata.obsm[stale_key]

        adata.obsm[f"X_{basis_name}"] = new_basis_coords

    return True, f"{locate_msg}; {match_msg}", metadata


def _load_lowdim_velocity_pair(
    adata: AnnData,
    vkey: str,
    gt_key: str,
    basis_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load low-dimensional predicted and ground truth velocity arrays."""
    pred_key = f"{vkey}_{basis_name}"
    gt_emb_key = f"{gt_key}_{basis_name}"

    if pred_key not in adata.obsm:
        raise KeyError(f"Predicted low-dimensional velocity not found: adata.obsm['{pred_key}']")
    if gt_emb_key not in adata.obsm:
        raise KeyError(f"Ground truth low-dimensional velocity not found: adata.obsm['{gt_emb_key}']")

    v_pred = adata.obsm[pred_key]
    v_gt = adata.obsm[gt_emb_key]

    if hasattr(v_pred, "toarray"):
        v_pred = v_pred.toarray()
    if hasattr(v_gt, "toarray"):
        v_gt = v_gt.toarray()

    if v_pred.shape != v_gt.shape:
        raise ValueError(f"Shape mismatch: predicted {v_pred.shape} vs ground truth {v_gt.shape}")

    return v_pred, v_gt


def _cosine_similarity_per_cell(v_pred: np.ndarray, v_gt: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Compute per-cell cosine similarity and map values from [-1, 1] to [0, 1]."""
    pred_norms = np.linalg.norm(v_pred, axis=1)
    gt_norms = np.linalg.norm(v_gt, axis=1)
    valid_mask = (pred_norms > 0) & (gt_norms > 0)

    if not valid_mask.any():
        raise ValueError("All cells have zero vectors in predicted or ground truth velocity")

    dot_products = np.sum(v_pred[valid_mask] * v_gt[valid_mask], axis=1)
    cosine_raw = dot_products / (pred_norms[valid_mask] * gt_norms[valid_mask])
    cosine_raw = np.clip(cosine_raw, -1.0, 1.0)
    cosine_01 = (cosine_raw + 1.0) / 2.0
    return cosine_01, v_pred.shape[0], int(valid_mask.sum())


def _update_wide_format_csv(
    output_path: Path,
    method: str,
    dataset_id: str,
    mean_cosine: float,
) -> None:
    """Incrementally update the wide-format CSV and sort rows by Method."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        df = pd.read_csv(output_path)
        for stale_col in ["AVG", "Average", "Reversed_rank"]:
            if stale_col in df.columns:
                df = df.drop(columns=[stale_col])
    else:
        df = pd.DataFrame(columns=["Method"])

    if dataset_id not in df.columns:
        df[dataset_id] = np.nan

    if method in df["Method"].values:
        row_idx = df[df["Method"] == method].index[0]
        df.at[row_idx, dataset_id] = mean_cosine
    else:
        df = pd.concat([df, pd.DataFrame([{"Method": method, dataset_id: mean_cosine}])], ignore_index=True)

    data_columns = [col for col in df.columns if col != "Method"]
    for col in data_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(3)

    df = df.sort_values(by="Method", ascending=True, na_position="last").reset_index(drop=True)
    df.to_csv(output_path, index=False)


def _detect_common_issues(
    adata: AnnData,
    velocity_key: str,
    gt_velocity_key: str,
    requested_basis_name: str,
) -> List[str]:
    """Return lightweight warnings without any tool-specific assumptions."""
    warnings_list: List[str] = []

    if velocity_key not in adata.layers and not any(
        f"{velocity_key}_{basis}" in adata.obsm for basis in ["dimred", "umap", "tsne"]
    ):
        available_layers = list(adata.layers.keys())[:8]
        warnings_list.append(
            f"WARNING: velocity_key '{velocity_key}' not found in layers or low-dimensional obsm. "
            f"Available layers: {available_layers}"
        )

    if gt_velocity_key not in adata.layers:
        warnings_list.append(
            f"WARNING: ground truth layer '{gt_velocity_key}' not found in adata.layers. "
            "A reference directory is required if no low-dimensional GT is already available."
        )

    if f"X_{requested_basis_name}" not in adata.obsm and requested_basis_name != "dimred":
        warnings_list.append(f"WARNING: requested basis 'X_{requested_basis_name}' not found in adata.obsm.")

    if adata.obs_names.duplicated().any():
        warnings_list.append("WARNING: duplicate obs_names detected; they will be made unique automatically.")

    if "X_dimred" not in adata.obsm and "X_umap" not in adata.obsm and "X_tsne" not in adata.obsm:
        warnings_list.append("WARNING: no usable non-PCA basis found in adata.obsm.")

    return warnings_list


def _setup_logging(log_file: Optional[Path] = None, verbose: bool = True) -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("groundtruth_correlation")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    if verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    return logger


def calculate_groundtruth_correlation(
    adata_or_path: Union[str, Path, AnnData],
    method: str,
    dataset_id: str,
    output_csv: str,
    velocity_key: str = "velocity",
    gt_velocity_key: str = "ground_truth_velocity",
    basis_name: str = "dimred",
    gt_npz_base_dir: Optional[str] = None,
    reference_gt_key: str = "gt_dimred",
    reference_basis_key: str = "X_basis",
    raise_on_gt_failure: bool = True,
    allow_partial_cell_match: bool = True,
    min_cell_match_ratio: float = 0.95,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Compute ground truth correlation for one method on one dataset.

    This function only performs low-dimensional comparison:
    - Predicted velocity is projected from adata.layers[velocity_key] onto the target basis.
    - Ground truth velocity is projected from adata.layers[gt_velocity_key] onto the same basis by default.
    - If a reference is provided explicitly, its low-dimensional GT or basis coordinates can override the low-dimensional preparation step.
    """
    adata = _ensure_adata(adata_or_path)
    warnings_list = _detect_common_issues(adata, velocity_key, gt_velocity_key, basis_name)

    try:
        resolved_basis_name = resolve_basis_name(
            adata,
            basis_name=basis_name,
            allow_default_fallback=True,
            logger=logger,
        )
    except Exception:
        if raise_on_gt_failure:
            raise
        return {
            "method": method,
            "dataset_id": dataset_id,
            "mean_cosine": np.nan,
            "n_cells_total": len(adata),
            "n_cells_valid": 0,
            "cell_match_ratio": 0.0,
            "gt_recovery_status": "basis resolution failed",
            "warnings": warnings_list,
            "success": False,
        }

    gt_metadata = {"cell_match_ratio": 1.0, "n_matched": len(adata), "n_total": len(adata), "used_reference": False}
    gt_recovery_status = "using adata.layers GT"

    if gt_npz_base_dir:
        try:
            used_reference, reference_msg, gt_metadata = _align_with_reference(
                adata=adata,
                dataset_id=dataset_id,
                basis_name=resolved_basis_name,
                velocity_key=velocity_key,
                gt_npz_base_dir=gt_npz_base_dir,
                reference_gt_key=reference_gt_key,
                reference_basis_key=reference_basis_key,
                gt_velocity_key=gt_velocity_key,
                allow_partial_match=allow_partial_cell_match,
                min_cell_match_ratio=min_cell_match_ratio,
                logger=logger,
            )
            if used_reference:
                gt_recovery_status = reference_msg
        except Exception as exc:
            if raise_on_gt_failure:
                raise
            return {
                "method": method,
                "dataset_id": dataset_id,
                "mean_cosine": np.nan,
                "n_cells_total": len(adata),
                "n_cells_valid": 0,
                "cell_match_ratio": 0.0,
                "gt_recovery_status": str(exc),
                "warnings": warnings_list,
                "success": False,
            }

    if f"X_{resolved_basis_name}" not in adata.obsm:
        raise KeyError(f"Missing basis coordinates after preparation: adata.obsm['X_{resolved_basis_name}']")

    _ensure_velocity_embedding(adata, velocity_key, resolved_basis_name)

    gt_embedding_key = f"{gt_velocity_key}_{resolved_basis_name}"
    if gt_embedding_key not in adata.obsm:
        if gt_velocity_key not in adata.layers:
            error_msg = (
                f"Ground truth unavailable: neither low-dimensional '{gt_embedding_key}' nor "
                f"high-dimensional adata.layers['{gt_velocity_key}'] exists."
            )
            if raise_on_gt_failure:
                raise KeyError(error_msg)
            return {
                "method": method,
                "dataset_id": dataset_id,
                "mean_cosine": np.nan,
                "n_cells_total": len(adata),
                "n_cells_valid": 0,
                "cell_match_ratio": gt_metadata["cell_match_ratio"],
                "gt_recovery_status": error_msg,
                "warnings": warnings_list,
                "success": False,
            }
        _ensure_velocity_embedding(adata, gt_velocity_key, resolved_basis_name)

    v_pred, v_gt = _load_lowdim_velocity_pair(
        adata=adata,
        vkey=velocity_key,
        gt_key=gt_velocity_key,
        basis_name=resolved_basis_name,
    )

    cos_similarities, n_total, n_valid = _cosine_similarity_per_cell(v_pred, v_gt)
    mean_cosine = float(cos_similarities.mean())

    _update_wide_format_csv(
        output_path=Path(output_csv),
        method=method,
        dataset_id=dataset_id,
        mean_cosine=mean_cosine,
    )

    return {
        "method": method,
        "dataset_id": dataset_id,
        "mean_cosine": mean_cosine,
        "n_cells_total": n_total,
        "n_cells_valid": n_valid,
        "cell_match_ratio": gt_metadata["cell_match_ratio"],
        "gt_recovery_status": gt_recovery_status,
        "basis_name": resolved_basis_name,
        "warnings": warnings_list,
        "success": True,
    }


def calculate_groundtruth_correlation_batch(
    metadata_csv: str,
    output_csv: str,
    output_dir: Optional[str] = None,
    velocity_key_column: str = "vkey",
    method_column: str = "method",
    dataset_id_column: str = "id",
    h5ad_path_column: str = "path",
    gt_velocity_key: str = "ground_truth_velocity",
    basis_name: str = "dimred",
    gt_npz_base_dir: Optional[str] = None,
    reference_gt_key: str = "gt_dimred",
    reference_basis_key: str = "X_basis",
    error_log_file: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute ground truth correlation in batch mode from CSV metadata."""
    output_dir_path = Path(output_csv).parent if output_dir is None else Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    log_path = output_dir_path / "errors.log" if error_log_file is None else Path(error_log_file)
    logger = _setup_logging(log_path, verbose=verbose)

    metadata_df = pd.read_csv(metadata_csv)
    required_columns = [method_column, dataset_id_column, h5ad_path_column]
    missing_columns = [col for col in required_columns if col not in metadata_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")

    if velocity_key_column not in metadata_df.columns:
        metadata_df[velocity_key_column] = "velocity"

    results: List[Dict[str, Any]] = []
    n_total = len(metadata_df)

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Processing {n_total} datasets from {metadata_csv}")
        print(f"Output CSV: {output_csv}")
        print(f"Error log: {log_path}")
        print(f"{'=' * 80}\n")

    for idx, row in metadata_df.iterrows():
        method = row[method_column]
        dataset_id = row[dataset_id_column]
        h5ad_path = row[h5ad_path_column]
        velocity_key = row[velocity_key_column] if pd.notna(row[velocity_key_column]) else "velocity"

        if verbose:
            print(f"[{idx + 1}/{n_total}] Processing {method} | {dataset_id}...", end=" ")

        try:
            result = calculate_groundtruth_correlation(
                adata_or_path=h5ad_path,
                method=method,
                dataset_id=dataset_id,
                output_csv=output_csv,
                velocity_key=velocity_key,
                gt_velocity_key=gt_velocity_key,
                basis_name=basis_name,
                gt_npz_base_dir=gt_npz_base_dir,
                reference_gt_key=reference_gt_key,
                reference_basis_key=reference_basis_key,
                raise_on_gt_failure=False,
                logger=logger,
            )

            if result["success"]:
                if verbose:
                    print(f"✓ Mean cosine: {result['mean_cosine']:.3f}")
                results.append(
                    {
                        "method": method,
                        "dataset_id": dataset_id,
                        "status": "success",
                        "mean_cosine": result["mean_cosine"],
                        "error": None,
                    }
                )
            else:
                if verbose:
                    print(f"✗ Failed: {result['gt_recovery_status']}")
                logger.error(f"GROUNDTRUTH_FAILED | {method} | {dataset_id} | {result['gt_recovery_status']}")
                results.append(
                    {
                        "method": method,
                        "dataset_id": dataset_id,
                        "status": "failed",
                        "mean_cosine": np.nan,
                        "error": result["gt_recovery_status"],
                    }
                )
        except Exception as exc:
            if verbose:
                print(f"✗ Error: {str(exc)[:80]}")
            logger.error(f"PROCESSING_ERROR | {method} | {dataset_id} | {str(exc)}")
            results.append(
                {
                    "method": method,
                    "dataset_id": dataset_id,
                    "status": "error",
                    "mean_cosine": np.nan,
                    "error": str(exc),
                }
            )

    summary_df = pd.DataFrame(results)

    if verbose:
        n_success = int((summary_df["status"] == "success").sum()) if not summary_df.empty else 0
        n_failed = int((summary_df["status"] == "failed").sum()) if not summary_df.empty else 0
        n_error = int((summary_df["status"] == "error").sum()) if not summary_df.empty else 0
        success_rate = (n_success / n_total * 100) if n_total > 0 else 0

        print(f"\n{'=' * 80}")
        print("Batch processing complete")
        print(f"  Total: {n_total} datasets")
        print(f"  Success: {n_success} ({success_rate:.1f}%)")
        print(f"  Failed: {n_failed}")
        print(f"  Errors: {n_error}")
        print(f"  Output: {output_csv}")
        print(f"  Error log: {log_path}")
        print(f"{'=' * 80}\n")

    return summary_df


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate ground truth correlation for RNA velocity predictions on a low-dimensional basis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-file mode with default GT from adata.layers['ground_truth_velocity']
  python groundtruth_correlation.py \
      --input result.h5ad \
      --method VeloVAE \
      --dataset-id bifurcating_cell1000_gene1000 \
      --output-csv results/correlation.csv \
      --velocity-key velocity

  # Batch mode
  python groundtruth_correlation.py \
      --metadata-csv datasets.csv \
      --output-csv results/correlation.csv \
      --basis-name dimred

  # Use reference directory to override low-dimensional GT / basis coordinates
  python groundtruth_correlation.py \
      --input result.h5ad \
      --method MyMethod \
      --dataset-id bifurcating_cell1000_gene1000 \
      --output-csv results/correlation.csv \
      --gt-npz-dir simdata_reference
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=str, help="Input H5AD file (single-file mode)")
    mode.add_argument("--metadata-csv", type=str, help="CSV metadata file (batch mode)")

    single_group = parser.add_argument_group("single-file mode parameters")
    single_group.add_argument("--method", type=str, help="Method name (required for single-file mode)")
    single_group.add_argument("--dataset-id", type=str, help="Dataset ID (required for single-file mode)")

    batch_group = parser.add_argument_group("batch mode parameters")
    batch_group.add_argument("--method-column", type=str, default="method", help="Column name for method")
    batch_group.add_argument("--dataset-id-column", type=str, default="id", help="Column name for dataset ID")
    batch_group.add_argument("--h5ad-path-column", type=str, default="path", help="Column name for H5AD path")
    batch_group.add_argument("--velocity-key-column", type=str, default="vkey", help="Column name for velocity key")

    parser.add_argument("--output-csv", type=str, required=True, help="Output CSV file (wide format)")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: inferred from output-csv)")
    parser.add_argument("--velocity-key", type=str, default="velocity", help="Predicted velocity layer key")
    parser.add_argument(
        "--gt-velocity-key",
        type=str,
        default="ground_truth_velocity",
        help="Ground truth velocity layer key",
    )
    parser.add_argument(
        "--basis-name",
        type=str,
        default="dimred",
        help="Low-dimensional basis name (default: dimred)",
    )
    parser.add_argument(
        "--gt-npz-dir",
        type=str,
        help="Optional reference directory containing *_reference_data.npz files",
    )
    parser.add_argument(
        "--reference-gt-key",
        type=str,
        default="gt_dimred",
        help="Low-dimensional GT key inside reference npz (default: gt_dimred)",
    )
    parser.add_argument(
        "--reference-basis-key",
        type=str,
        default="X_basis",
        help="Basis coordinate key inside reference npz (default: X_basis)",
    )
    parser.add_argument(
        "--min-cell-match-ratio",
        type=float,
        default=0.95,
        help="Minimum acceptable cell name match ratio (default: 0.95)",
    )
    parser.add_argument("--error-log", type=str, help="Error log file")
    parser.add_argument("--verbose", action="store_true", help="Print progress to console")

    args = parser.parse_args()

    if args.input and not (args.method and args.dataset_id):
        parser.error("--method and --dataset-id are required for single-file mode")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    error_log = Path(args.error_log) if args.error_log else (output_dir / "errors.log")
    logger = _setup_logging(error_log, verbose=args.verbose)

    if args.input:
        try:
            result = calculate_groundtruth_correlation(
                adata_or_path=args.input,
                method=args.method,
                dataset_id=args.dataset_id,
                output_csv=args.output_csv,
                velocity_key=args.velocity_key,
                gt_velocity_key=args.gt_velocity_key,
                basis_name=args.basis_name,
                gt_npz_base_dir=args.gt_npz_dir,
                reference_gt_key=args.reference_gt_key,
                reference_basis_key=args.reference_basis_key,
                raise_on_gt_failure=True,
                min_cell_match_ratio=args.min_cell_match_ratio,
                logger=logger,
            )

            print(f"\n✓ Success: {args.method} | {args.dataset_id}")
            print(f"  Mean cosine: {result['mean_cosine']:.3f}")
            print(f"  Basis: {result['basis_name']}")
            print(f"  Valid cells: {result['n_cells_valid']}/{result['n_cells_total']}")
            print(f"  Output: {args.output_csv}")
        except Exception as exc:
            logger.error(f"{args.method}_{args.dataset_id}: {str(exc)}")
            print(f"\n✗ Failed: {args.method} | {args.dataset_id}")
            print(f"  Error: {str(exc)}")
            sys.exit(1)
    else:
        calculate_groundtruth_correlation_batch(
            metadata_csv=args.metadata_csv,
            output_csv=args.output_csv,
            output_dir=str(output_dir),
            velocity_key_column=args.velocity_key_column,
            method_column=args.method_column,
            dataset_id_column=args.dataset_id_column,
            h5ad_path_column=args.h5ad_path_column,
            gt_velocity_key=args.gt_velocity_key,
            basis_name=args.basis_name,
            gt_npz_base_dir=args.gt_npz_dir,
            reference_gt_key=args.reference_gt_key,
            reference_basis_key=args.reference_basis_key,
            error_log_file=str(error_log),
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
