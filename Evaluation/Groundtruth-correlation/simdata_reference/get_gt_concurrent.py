#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export simulation reference NPZ files.

RNA uses the original topology-specific coordinates, labels, and velocity
projection. Lineage references contain coordinates and labels without true
velocity. ATAC copies X_tsne and true_velocity_tsne directly; dataset IDs
containing 'linear' use pseudo_celltype, while other ATAC IDs use celltype.
All NPZ files use X_basis, cell_names, celltype, and, when available, gt_dimred.
"""

from __future__ import annotations

import argparse
import logging
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv

ALL_TOPOLOGIES = [
    "bifurcating",
    "consecutive-bifurcating",
    "disconnected",
    "trifurcating",
    "linear-simple",
    "cycle-simple",
    "bifurcating-loop",
    "simulation-add",
    "Bursting-tree",
    "cellsub-bifurcating",
    "genesub-bifurcating",
    "lineage-tracing",
]

SKIP_IN_ALL_MODE = {
    "Bursting-tree",
    "lineage-tracing",
}


@dataclass(frozen=True)
class DatasetSpec:
    topology: str
    dataset_name: str
    filepath: str
    output_path: str


@dataclass(frozen=True)
class ExportConfig:
    input_dir: str
    output_dir: str
    fallback_input_dir: Optional[str] = None
    bursting_meta_dir: Optional[str] = None
    bursting_annotated_dir: Optional[str] = None
    modality: str = "rna"
    n_jobs: int = 10
    skip_name_fragments: tuple[str, ...] = ()


def setup_logging(config: ExportConfig) -> str:
    os.makedirs(config.output_dir, exist_ok=True)
    log_file = os.path.join(config.output_dir, "export_reference_sidecar.log")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.ERROR,
        format="%(levelname)s - %(message)s",
    )
    return log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Input topology root; use --topologies . for a flat directory")
    parser.add_argument("--output-dir", required=True, help="Directory for reference NPZ files and the export log")
    parser.add_argument("--fallback-input-dir", help="Optional second root, searched after --input-dir")
    parser.add_argument("--modality", choices=("rna", "atac"), default="rna")
    parser.add_argument("--bursting-meta-dir", help="Root containing <dataset_id>/cell_metadata.csv")
    parser.add_argument("--bursting-annotated-dir", help="Directory containing annotated <dataset_id>.h5ad files")
    parser.add_argument("--n-jobs", type=int, default=10, help="Workers for RNA true-velocity graph computation")
    parser.add_argument("--skip-name-fragment", action="append", default=[], help="Skip dataset IDs containing this string; repeat to exclude several patterns")
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["all"],
        help="Topology directory names, '.' for flat input, or 'all' for the standard RNA topologies",
    )
    parser.add_argument(
        "--exclude-topologies",
        nargs="*",
        default=[],
        help="Topology directory names to exclude",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing reference NPZ files",
    )
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    for name in ("input_dir", "fallback_input_dir", "bursting_meta_dir", "bursting_annotated_dir"):
        path = getattr(args, name)
        if path is not None and not Path(path).is_dir():
            parser.error(f"--{name.replace('_', '-')} is not a directory: {path}")
    return args


def get_base_name(filename: str) -> str:
    if filename.endswith("_dataset.h5ad"):
        return filename[:-len("_dataset.h5ad")]
    if filename.endswith(".h5ad"):
        return filename[:-len(".h5ad")]
    return filename


def normalize_topologies(topologies: Iterable[str], excludes: Iterable[str]) -> list[str]:
    normalized = set()
    requested_all = False
    for topology in topologies:
        if topology not in {"all", "."} and (
            not topology or topology == ".." or Path(topology).name != topology
            or "/" in topology or "\\" in topology
        ):
            raise ValueError(f"Topology must be one directory name: {topology!r}")
        if topology == "all":
            requested_all = True
            normalized.update(ALL_TOPOLOGIES)
        else:
            normalized.add(topology)

    if requested_all:
        normalized.difference_update(SKIP_IN_ALL_MODE)

    for topology in excludes:
        normalized.discard(topology)

    ordered = [topology for topology in ALL_TOPOLOGIES if topology in normalized]
    extras = sorted(normalized - set(ordered))
    return ordered + extras


def get_topology_source_dirs(topology: str, config: ExportConfig) -> list[str]:
    candidates: list[str] = []

    if topology == "lineage-tracing" and config.modality == "rna":
        candidates.append(os.path.join(config.input_dir, "lineage_tracing"))
    else:
        for dirname in (topology, topology.replace("-", "_")):
            candidates.append(os.path.join(config.input_dir, dirname))

    if config.fallback_input_dir is not None:
        candidates.append(os.path.join(config.fallback_input_dir, topology))

    ordered: list[str] = []
    seen = set()
    for path in candidates:
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def should_skip_dataset(dataset_name: str, config: ExportConfig) -> bool:
    return any(fragment in dataset_name for fragment in config.skip_name_fragments)


def resolve_dataset_specs(topology: str, overwrite: bool, config: ExportConfig) -> list[DatasetSpec]:
    output_dir = os.path.join(config.output_dir, topology)
    os.makedirs(output_dir, exist_ok=True)
    specs: list[DatasetSpec] = []

    specs_by_name: dict[str, DatasetSpec] = {}
    for source_dir in get_topology_source_dirs(topology, config):
        if not os.path.isdir(source_dir):
            continue

        if topology == "lineage-tracing" and config.modality == "rna":
            for rep_name in sorted(os.listdir(source_dir)):
                rep_dir = os.path.join(source_dir, rep_name)
                if not os.path.isdir(rep_dir):
                    continue
                filepath = os.path.join(rep_dir, "adata.h5ad")
                if not os.path.exists(filepath):
                    continue
                dataset_name = f"lineage-tracing_{rep_name}"
                if should_skip_dataset(dataset_name, config) or dataset_name in specs_by_name:
                    continue
                output_path = os.path.join(output_dir, f"{dataset_name}_reference_data.npz")
                if not overwrite and os.path.exists(output_path):
                    continue
                specs_by_name[dataset_name] = DatasetSpec(topology, dataset_name, filepath, output_path)
            continue

        for filename in sorted(os.listdir(source_dir)):
            if not filename.endswith(".h5ad"):
                continue
            filepath = os.path.join(source_dir, filename)
            dataset_name = get_base_name(filename)
            if should_skip_dataset(dataset_name, config) or dataset_name in specs_by_name:
                continue
            output_path = os.path.join(output_dir, f"{dataset_name}_reference_data.npz")
            if not overwrite and os.path.exists(output_path):
                continue
            specs_by_name[dataset_name] = DatasetSpec(topology, dataset_name, filepath, output_path)

    specs.extend(specs_by_name[name] for name in sorted(specs_by_name))
    return specs


def matrix_2d(matrix, n_cells: int, name: str, require_finite: bool = True) -> np.ndarray:
    """Copy two reference dimensions without changing values or row order."""
    values = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != n_cells or values.shape[1] < 2:
        raise ValueError(f"{name} must have n_cells rows and at least two columns")
    values = np.asarray(values[:, :2]).copy()
    if require_finite and not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite coordinates or velocities")
    return values


def get_display_basis(adata, topology: str, modality: str = "rna") -> np.ndarray:
    if modality == "atac":
        if "X_tsne" not in adata.obsm:
            raise ValueError("ATAC input requires obsm['X_tsne']")
        return matrix_2d(adata.obsm["X_tsne"], adata.n_obs, "X_tsne")
    if topology == "lineage-tracing":
        if "X_umap" not in adata.obsm:
            raise ValueError("Missing obsm['X_umap']")
        return np.asarray(adata.obsm["X_umap"])[:, :2].copy()

    if topology == "Bursting-tree":
        if "X_tsne" in adata.obsm:
            return np.asarray(adata.obsm["X_tsne"])[:, :2].copy()
        if "X_dimred" in adata.obsm:
            return np.asarray(adata.obsm["X_dimred"])[:, :2].copy()
        raise ValueError("Missing X_tsne/X_dimred coordinates")

    if "X_dimred" in adata.obsm:
        return np.asarray(adata.obsm["X_dimred"])[:, :2].copy()
    if "X_umap" in adata.obsm:
        return np.asarray(adata.obsm["X_umap"])[:, :2].copy()
    raise ValueError("Missing X_dimred/X_umap coordinates")


def get_celltype(adata, topology: str, dataset_name: str = "", modality: str = "rna") -> np.ndarray:
    if modality == "atac":
        key = "pseudo_celltype" if "linear" in dataset_name.lower() else "celltype"
        if key not in adata.obs.columns or adata.obs[key].isna().any():
            raise ValueError(f"{dataset_name}: ATAC input requires complete obs[{key!r}]")
        return adata.obs[key].astype(str).to_numpy()
    if topology == "lineage-tracing":
        if "synthetic_celllabel" not in adata.obs.columns:
            raise ValueError("Missing obs['synthetic_celllabel']")
        return adata.obs["synthetic_celllabel"].astype(str).to_numpy()

    for key in ("milestone", "synthetic_celllabel", "pop", "edge_id"):
        if key in adata.obs.columns:
            return adata.obs[key].astype(str).to_numpy()
    raise ValueError("Missing a label column for celltype")


def load_bursting_meta(dataset_name: str, config: ExportConfig) -> pd.DataFrame:
    if config.bursting_meta_dir is None:
        raise ValueError("Bursting-tree export requires --bursting-meta-dir")
    meta_path = os.path.join(config.bursting_meta_dir, dataset_name, "cell_metadata.csv")
    if not os.path.exists(meta_path):
        raise ValueError(f"{dataset_name}: Missing Bursting metadata file {meta_path}")

    meta_df = pd.read_csv(meta_path)
    required_columns = ["cell_id", "edge_id", "cell_time_ref", "milestone_id", "lineage_id"]
    missing = [col for col in required_columns if col not in meta_df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: Bursting metadata is missing columns {missing}")

    if "pop_raw" not in meta_df.columns:
        if "pop" in meta_df.columns:
            meta_df["pop_raw"] = meta_df["pop"]
        else:
            meta_df["pop_raw"] = meta_df["edge_id"]
    if "milestone" not in meta_df.columns:
        meta_df["milestone"] = meta_df["milestone_id"]

    for col in ["cell_id", "edge_id", "milestone_id", "lineage_id", "pop_raw", "milestone"]:
        meta_df[col] = meta_df[col].astype(str)
    meta_df["cell_time_ref"] = meta_df["cell_time_ref"].astype(float)
    return meta_df


def load_bursting_annotated_payload(dataset_name: str, config: ExportConfig) -> dict[str, np.ndarray]:
    if config.bursting_annotated_dir is None:
        raise ValueError("Bursting-tree export requires --bursting-annotated-dir")
    annotated_path = os.path.join(config.bursting_annotated_dir, f"{dataset_name}.h5ad")
    if not os.path.exists(annotated_path):
        raise ValueError(f"{dataset_name}: Missing Bursting annotated H5AD {annotated_path}")

    annotated = sc.read_h5ad(annotated_path)
    if "cell_id" in annotated.obs.columns:
        cell_ids = annotated.obs["cell_id"].astype(str).to_numpy()
    else:
        cell_ids = annotated.obs_names.astype(str).to_numpy()

    if "X_dimred" in annotated.obsm:
        x_basis = np.asarray(annotated.obsm["X_dimred"])[:, :2].copy()
    elif "X_tsne" in annotated.obsm:
        x_basis = np.asarray(annotated.obsm["X_tsne"])[:, :2].copy()
    else:
        raise ValueError(f"{dataset_name}: Annotated H5AD is missing X_dimred/X_tsne")

    if "ground_truth_velocity_dimred" in annotated.obsm:
        gt_dimred = np.asarray(annotated.obsm["ground_truth_velocity_dimred"])[:, :2].copy()
    elif "ground_truth_velocity_tsne" in annotated.obsm:
        gt_dimred = np.asarray(annotated.obsm["ground_truth_velocity_tsne"])[:, :2].copy()
    else:
        raise ValueError(f"{dataset_name}: Annotated H5AD is missing ground_truth_velocity_dimred/tsne")

    return {
        "cell_id": cell_ids,
        "X_basis": x_basis,
        "gt_dimred": gt_dimred,
    }


def compute_gt_embedding(adata, n_jobs: int = 10) -> np.ndarray:
    if "X_dimred" not in adata.obsm and "X_tsne" in adata.obsm:
        adata.obsm["X_dimred"] = np.asarray(adata.obsm["X_tsne"]).copy()

    sc.pp.neighbors(adata)
    scv.tl.velocity_graph(adata, vkey="ground_truth_velocity", n_jobs=n_jobs)
    scv.tl.velocity_embedding(adata, basis="dimred", vkey="ground_truth_velocity")
    return np.asarray(adata.obsm["ground_truth_velocity_dimred"])[:, :2].copy()


def save_npz(
    output_path: str,
    x_basis: np.ndarray,
    cell_names: np.ndarray,
    celltype: np.ndarray,
    gt_dimred: Optional[np.ndarray],
    cell_names_unique: Optional[np.ndarray],
    extra_fields: Optional[dict[str, np.ndarray]] = None,
) -> None:
    n_cells = len(cell_names)
    x_basis = matrix_2d(x_basis, n_cells, "X_basis")
    if len(celltype) != n_cells:
        raise ValueError("celltype must have one label per reference cell")
    if gt_dimred is not None:
        gt_dimred = matrix_2d(gt_dimred, n_cells, "gt_dimred", require_finite=False)
    if extra_fields and any(len(values) != n_cells for values in extra_fields.values()):
        raise ValueError("Extra reference fields must have one value per cell")
    save_kwargs = {
        "X_basis": x_basis,
        "cell_names": cell_names,
        "celltype": celltype,
    }
    if gt_dimred is not None:
        save_kwargs["gt_dimred"] = gt_dimred
    if cell_names_unique is not None:
        save_kwargs["cell_names_unique"] = cell_names_unique
    if extra_fields:
        save_kwargs.update(extra_fields)
    np.savez_compressed(output_path, **save_kwargs)


def process_single_dataset(spec: DatasetSpec, config: ExportConfig) -> tuple[bool, str, bool, str, Optional[str]]:
    try:
        adata = sc.read_h5ad(spec.filepath)
        raw_cell_names = np.array(adata.obs_names.values, copy=True)
        x_basis = get_display_basis(adata, spec.topology, config.modality)
        celltype = get_celltype(adata, spec.topology, spec.dataset_name, config.modality)

        gt_dimred = None
        extra_fields: dict[str, np.ndarray] = {}
        if config.modality == "atac":
            if "true_velocity_tsne" not in adata.obsm:
                raise ValueError("ATAC input requires obsm['true_velocity_tsne']")
            # The ATAC truth is already expressed on the supplied tSNE basis.
            gt_dimred = matrix_2d(
                adata.obsm["true_velocity_tsne"], adata.n_obs,
                "true_velocity_tsne", require_finite=False,
            )
        elif spec.topology != "lineage-tracing":
            adata.obs_names_make_unique()
            gt_dimred = compute_gt_embedding(adata, n_jobs=config.n_jobs)
            adata.obs_names = raw_cell_names

        if config.modality == "rna" and spec.topology == "Bursting-tree":
            meta_df = load_bursting_meta(spec.dataset_name, config)
            annotated_payload = load_bursting_annotated_payload(spec.dataset_name, config)
            ref_index = {cell_id: idx for idx, cell_id in enumerate(annotated_payload["cell_id"])}
            missing_ids = [cell_id for cell_id in meta_df["cell_id"].tolist() if cell_id not in ref_index]
            if missing_ids:
                raise ValueError(f"{spec.dataset_name}: Annotated payload is missing {len(missing_ids)} cell IDs")
            ref_indices = np.array([ref_index[cell_id] for cell_id in meta_df["cell_id"].tolist()], dtype=int)

            x_basis = annotated_payload["X_basis"][ref_indices]
            gt_dimred = annotated_payload["gt_dimred"][ref_indices]
            celltype = meta_df["pop_raw"].astype(str).to_numpy()
            extra_fields = {
                "cell_id": meta_df["cell_id"].astype(str).to_numpy(),
                "edge_id": meta_df["edge_id"].astype(str).to_numpy(),
                "cell_time_ref": meta_df["cell_time_ref"].to_numpy(dtype=float),
                "milestone_id": meta_df["milestone_id"].astype(str).to_numpy(),
                "lineage_id": meta_df["lineage_id"].astype(str).to_numpy(),
                "pop_raw": meta_df["pop_raw"].astype(str).to_numpy(),
                "milestone": meta_df["milestone"].astype(str).to_numpy(),
            }

        adata.obs_names_make_unique()
        new_cell_names = np.asarray(adata.obs_names.values)
        cell_names_unique = None
        has_unique = False
        if not np.array_equal(new_cell_names, raw_cell_names):
            cell_names_unique = new_cell_names
            has_unique = True

        save_npz(
            output_path=spec.output_path,
            x_basis=x_basis,
            cell_names=raw_cell_names,
            celltype=celltype,
            gt_dimred=gt_dimred,
            cell_names_unique=cell_names_unique,
            extra_fields=extra_fields,
        )
        return True, f"{spec.topology}/{spec.dataset_name}", has_unique, spec.dataset_name, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{spec.topology}/{spec.dataset_name}: {exc}", False, spec.dataset_name, traceback.format_exc()


def main() -> None:
    args = parse_args()
    config = ExportConfig(
        input_dir=args.input_dir, output_dir=args.output_dir,
        fallback_input_dir=args.fallback_input_dir,
        bursting_meta_dir=args.bursting_meta_dir,
        bursting_annotated_dir=args.bursting_annotated_dir,
        modality=args.modality, n_jobs=args.n_jobs,
        skip_name_fragments=tuple(args.skip_name_fragment),
    )
    log_file = setup_logging(config)

    selected_topologies = normalize_topologies(args.topologies, args.exclude_topologies)
    if "all" in args.topologies:
        skipped = [topo for topo in ALL_TOPOLOGIES if topo in SKIP_IN_ALL_MODE]
        if skipped:
            print(
                "The all preset skips: " + ", ".join(skipped),
                flush=True,
            )
            print(
                "To export these topologies, specify --topologies Bursting-tree lineage-tracing",
                flush=True,
            )

    dataset_specs: list[DatasetSpec] = []
    for topology in selected_topologies:
        specs = resolve_dataset_specs(topology, overwrite=args.overwrite, config=config)
        dataset_specs.extend(specs)
        print(
            f"Topology {topology}: {len(specs)} datasets pending",
            flush=True,
        )

    if not dataset_specs:
        print("No datasets are pending; check input directories, topology selection, and existing outputs")
        return

    results = []
    total = len(dataset_specs)
    for index, spec in enumerate(dataset_specs, start=1):
        print(
            f"[{index}/{total}] Processing {spec.topology}/{spec.dataset_name}",
            flush=True,
        )
        result = process_single_dataset(spec, config)
        results.append(result)
        success, msg, _, _, _ = result
        status = "Completed" if success else "Failed"
        print(f"[{index}/{total}] {status}: {msg}", flush=True)

    success_count = sum(1 for success, _, _, _, _ in results if success)
    failed_infos = [(msg, tb) for success, msg, _, _, tb in results if not success]
    cell_names_unique_list = [
        base_name
        for success, _, has_unique, base_name, _ in results
        if success and has_unique
    ]

    print(f"Total: {success_count} succeeded, {len(failed_infos)} failed")
    if failed_infos:
        print("\nFailed files:")
        for msg, _ in failed_infos:
            print(f"  - {msg}")

    if cell_names_unique_list:
        print("\nDatasets with cell_names_unique:")
        for base_name in cell_names_unique_list:
            print(f"  - {base_name}")

    with open(log_file, "w", encoding="utf-8") as handle:
        for msg, tb_str in failed_infos:
            handle.write(f"[ERROR] {msg}\n")
            if tb_str is not None:
                handle.write(tb_str + "\n")
        handle.write("\nDatasets with cell_names_unique:\n")
        for base_name in cell_names_unique_list:
            handle.write(f"  - {base_name}\n")

    if failed_infos:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
