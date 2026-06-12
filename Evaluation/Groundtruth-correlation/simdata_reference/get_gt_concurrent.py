#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出统一格式的模拟数据 sidecar npz。

输出字段：
- X_basis: 原始展示坐标
  - 普通模拟数据来自 ``obsm['X_dimred']``
  - Bursting-tree 来自 ``obsm['X_tsne']``（若缺失则退到 ``X_dimred``）
  - lineage-tracing 来自 ``obsm['X_umap']``
- cell_names: 原始 obs_names
- cell_names_unique: 如唯一化后名称发生变化则额外保存
- celltype: 统一细胞类型标签
  - 普通模拟数据优先取 ``obs['milestone']``
  - lineage-tracing 取 ``obs['synthetic_celllabel']``
- gt_dimred: 仅有 ground-truth velocity 的数据保存
- Bursting-tree 额外字段：
  - cell_id / edge_id / cell_time_ref / milestone_id / lineage_id / pop_raw / milestone
"""

from __future__ import annotations

import argparse
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import warnings


warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
scv.settings.verbosity = 0
scv.settings.presenter_view = False
scv.settings.plot_prefix = ""
scv.settings.show_progress_bar = False

SIMDATA_LOCAL_DIR = "/data_d/Velocity/data/simdata_local"
SIMULATION2_DIR = "/data_d/Velocity/data/simulation/simulation2"
OUTPUT_DIR = "/data_d/Velocity/LXiao/simdata_reference"
LOG_FILE = os.path.join(OUTPUT_DIR, "export_reference_sidecar.log")
BURSTING_META_BASE_DIR = "/data_d/Velocity/LXiao/sim_roseplot/bursting_meta"
BURSTING_ANNOTATED_DIR = "/data_d/Velocity/LXiao/sim_roseplot/.codex-local/archive/2026-05-25_sim_roseplot_cleanup/bursting_ground_truth_tsne/annotated_h5ad"

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

SKIP_DATASET_NAME_FRAGMENTS = (
    "cell1000_gene50000",
    "cell1000_gene100000",
)


@dataclass(frozen=True)
class DatasetSpec:
    topology: str
    dataset_name: str
    filepath: str
    output_path: str


def setup_logging() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        filemode="w",
        level=logging.ERROR,
        format="%(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出统一 sidecar npz")
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["all"],
        help="要处理的拓扑类型；默认 all",
    )
    parser.add_argument(
        "--exclude-topologies",
        nargs="*",
        default=[],
        help="要排除的拓扑类型",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 npz",
    )
    return parser.parse_args()


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


def get_topology_source_dir(topology: str) -> str:
    if topology == "lineage-tracing":
        return os.path.join(SIMULATION2_DIR, "lineage_tracing")
    return os.path.join(SIMDATA_LOCAL_DIR, topology)


def get_topology_source_dirs(topology: str) -> list[str]:
    candidates: list[str] = []

    if topology == "lineage-tracing":
        candidates.append(os.path.join(SIMULATION2_DIR, "lineage_tracing"))
    else:
        for dirname in (topology, topology.replace("-", "_")):
            candidates.append(os.path.join(SIMULATION2_DIR, dirname))

    candidates.append(os.path.join(SIMDATA_LOCAL_DIR, topology))

    ordered: list[str] = []
    seen = set()
    for path in candidates:
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def should_skip_dataset(dataset_name: str) -> bool:
    return any(fragment in dataset_name for fragment in SKIP_DATASET_NAME_FRAGMENTS)


def resolve_dataset_specs(topology: str, overwrite: bool) -> list[DatasetSpec]:
    output_dir = os.path.join(OUTPUT_DIR, topology)
    os.makedirs(output_dir, exist_ok=True)
    specs: list[DatasetSpec] = []

    specs_by_name: dict[str, DatasetSpec] = {}
    for source_dir in get_topology_source_dirs(topology):
        if not os.path.isdir(source_dir):
            continue

        if topology == "lineage-tracing":
            for rep_name in sorted(os.listdir(source_dir)):
                rep_dir = os.path.join(source_dir, rep_name)
                if not os.path.isdir(rep_dir):
                    continue
                filepath = os.path.join(rep_dir, "adata.h5ad")
                if not os.path.exists(filepath):
                    continue
                dataset_name = f"lineage-tracing_{rep_name}"
                if should_skip_dataset(dataset_name) or dataset_name in specs_by_name:
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
            if should_skip_dataset(dataset_name) or dataset_name in specs_by_name:
                continue
            output_path = os.path.join(output_dir, f"{dataset_name}_reference_data.npz")
            if not overwrite and os.path.exists(output_path):
                continue
            specs_by_name[dataset_name] = DatasetSpec(topology, dataset_name, filepath, output_path)

    specs.extend(specs_by_name[name] for name in sorted(specs_by_name))
    return specs


def get_display_basis(adata, topology: str) -> np.ndarray:
    if topology == "lineage-tracing":
        if "X_umap" not in adata.obsm:
            raise ValueError("缺少 X_umap")
        return np.asarray(adata.obsm["X_umap"])[:, :2].copy()

    if topology == "Bursting-tree":
        if "X_tsne" in adata.obsm:
            return np.asarray(adata.obsm["X_tsne"])[:, :2].copy()
        if "X_dimred" in adata.obsm:
            return np.asarray(adata.obsm["X_dimred"])[:, :2].copy()
        raise ValueError("缺少 X_tsne/X_dimred")

    if "X_dimred" in adata.obsm:
        return np.asarray(adata.obsm["X_dimred"])[:, :2].copy()
    if "X_umap" in adata.obsm:
        return np.asarray(adata.obsm["X_umap"])[:, :2].copy()
    raise ValueError("缺少 X_dimred/X_umap")


def get_celltype(adata, topology: str) -> np.ndarray:
    if topology == "lineage-tracing":
        if "synthetic_celllabel" not in adata.obs.columns:
            raise ValueError("缺少 synthetic_celllabel")
        return adata.obs["synthetic_celllabel"].astype(str).to_numpy()

    for key in ("milestone", "synthetic_celllabel", "pop", "edge_id"):
        if key in adata.obs.columns:
            return adata.obs[key].astype(str).to_numpy()
    raise ValueError("缺少可写入 celltype 的标签列")


def load_bursting_meta(dataset_name: str) -> pd.DataFrame:
    meta_path = os.path.join(BURSTING_META_BASE_DIR, dataset_name, "cell_metadata.csv")
    if not os.path.exists(meta_path):
        raise ValueError(f"{dataset_name}: 缺少 Bursting metadata 文件 {meta_path}")

    meta_df = pd.read_csv(meta_path)
    required_columns = ["cell_id", "edge_id", "cell_time_ref", "milestone_id", "lineage_id"]
    missing = [col for col in required_columns if col not in meta_df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: Bursting metadata 缺少列 {missing}")

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


def load_bursting_annotated_payload(dataset_name: str) -> dict[str, np.ndarray]:
    annotated_path = os.path.join(BURSTING_ANNOTATED_DIR, f"{dataset_name}.h5ad")
    if not os.path.exists(annotated_path):
        raise ValueError(f"{dataset_name}: 缺少 Bursting annotated h5ad {annotated_path}")

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
        raise ValueError(f"{dataset_name}: annotated h5ad 缺少 X_dimred/X_tsne")

    if "ground_truth_velocity_dimred" in annotated.obsm:
        gt_dimred = np.asarray(annotated.obsm["ground_truth_velocity_dimred"])[:, :2].copy()
    elif "ground_truth_velocity_tsne" in annotated.obsm:
        gt_dimred = np.asarray(annotated.obsm["ground_truth_velocity_tsne"])[:, :2].copy()
    else:
        raise ValueError(f"{dataset_name}: annotated h5ad 缺少 ground_truth_velocity_dimred/tsne")

    return {
        "cell_id": cell_ids,
        "X_basis": x_basis,
        "gt_dimred": gt_dimred,
    }


def compute_gt_embedding(adata) -> np.ndarray:
    if "X_dimred" not in adata.obsm and "X_tsne" in adata.obsm:
        adata.obsm["X_dimred"] = np.asarray(adata.obsm["X_tsne"]).copy()

    sc.pp.neighbors(adata)
    scv.tl.velocity_graph(adata, vkey="ground_truth_velocity", n_jobs=10)
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


def process_single_dataset(spec: DatasetSpec) -> tuple[bool, str, bool, str, Optional[str]]:
    try:
        adata = sc.read_h5ad(spec.filepath)
        raw_cell_names = np.array(adata.obs_names.values, copy=True)
        x_basis = get_display_basis(adata, spec.topology)
        celltype = get_celltype(adata, spec.topology)

        gt_dimred = None
        extra_fields: dict[str, np.ndarray] = {}
        if spec.topology != "lineage-tracing":
            adata.obs_names_make_unique()
            gt_dimred = compute_gt_embedding(adata)
            adata.obs_names = raw_cell_names

        if spec.topology == "Bursting-tree":
            meta_df = load_bursting_meta(spec.dataset_name)
            annotated_payload = load_bursting_annotated_payload(spec.dataset_name)
            ref_index = {cell_id: idx for idx, cell_id in enumerate(annotated_payload["cell_id"])}
            missing_ids = [cell_id for cell_id in meta_df["cell_id"].tolist() if cell_id not in ref_index]
            if missing_ids:
                raise ValueError(f"{spec.dataset_name}: annotated payload 缺少 {len(missing_ids)} 个 cell_id")
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
    setup_logging()

    selected_topologies = normalize_topologies(args.topologies, args.exclude_topologies)
    if "all" in args.topologies:
        skipped = [topo for topo in ALL_TOPOLOGIES if topo in SKIP_IN_ALL_MODE]
        if skipped:
            print(
                "all 模式默认跳过拓扑: " + ", ".join(skipped),
                flush=True,
            )
            print(
                "如需重导这些拓扑，请显式写入 --topologies Bursting-tree lineage-tracing",
                flush=True,
            )

    dataset_specs: list[DatasetSpec] = []
    for topology in selected_topologies:
        specs = resolve_dataset_specs(topology, overwrite=args.overwrite)
        dataset_specs.extend(specs)
        print(
            f"拓扑 {topology}: 待处理 {len(specs)} 个数据集",
            flush=True,
        )

    if not dataset_specs:
        print("未找到任何待处理数据集")
        return

    results = []
    total = len(dataset_specs)
    for index, spec in enumerate(dataset_specs, start=1):
        print(
            f"[{index}/{total}] 开始处理 {spec.topology}/{spec.dataset_name}",
            flush=True,
        )
        result = process_single_dataset(spec)
        results.append(result)
        success, msg, _, _, _ = result
        status = "完成" if success else "失败"
        print(f"[{index}/{total}] {status}: {msg}", flush=True)

    success_count = sum(1 for success, _, _, _, _ in results if success)
    failed_infos = [(msg, tb) for success, msg, _, _, tb in results if not success]
    cell_names_unique_list = [
        base_name
        for success, _, has_unique, base_name, _ in results
        if success and has_unique
    ]

    print(f"总计: 成功 {success_count} 个, 失败 {len(failed_infos)} 个")
    if failed_infos:
        print("\n失败的文件:")
        for msg, _ in failed_infos:
            print(f"  - {msg}")

    if cell_names_unique_list:
        print("\n以下数据存在 cell_names_unique：")
        for base_name in cell_names_unique_list:
            print(f"  - {base_name}")

    with open(LOG_FILE, "w", encoding="utf-8") as handle:
        for msg, tb_str in failed_infos:
            handle.write(f"[ERROR] {msg}\n")
            if tb_str is not None:
                handle.write(tb_str + "\n")
        handle.write("\n以下数据存在 cell_names_unique：\n")
        for base_name in cell_names_unique_list:
            handle.write(f"  - {base_name}\n")


if __name__ == "__main__":
    main()
