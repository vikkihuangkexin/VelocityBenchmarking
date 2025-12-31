#!/usr/bin/env python3
"""
Angle Consistency (Rose Plot) metric for RNA velocity.

This module implements a reproducible metric that compares predicted RNA velocity
directions against reference differentiation directions in a low-dimensional embedding.
It supports both:
- Real datasets: reference directions are derived from differentiation paths (cell types,
  time, or phase) fitted on UMAP coordinates (external CSVs or computed from the input).
- Simulated datasets: reference directions are derived from milestone-based trajectory
  templates (topology-aware).

Entry points:
- Python API: run_unified_angle_consistency(), run_batch_from_csv()
- CLI: python angle_consistency.py --input ...  OR  python angle_consistency.py --csv-file ...

Reproducibility defaults:
- Parallelism defaults to n_jobs=1 (user-overridable for scvelo.velocity_graph).
- External real-UMAP CSV directory is optional; when absent, UMAP will be computed.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import logging
import multiprocessing
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from anndata import AnnData
from matplotlib.patches import Wedge
from scipy.interpolate import splprep, splev
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans


DEFAULT_REAL_UMAP_DIRNAME = "real-umap"

ANGLE_COLUMNS = [
  "0-30", "30-60", "60-80", "80-90", "60-90",
  "90-100", "100-120", "90-120", "120-150", "150-180",
  "valid_ratio", "cv_coefficient",
]


def _setup_error_logging(error_log_path: Optional[str]) -> None:
  """
  Configure process-wide ERROR-only logging.

  - Root logger: ERROR-only, optional file handler.
  - Simulation pipeline: also sets the shared error-log file if provided.
  """
  root_logger = logging.getLogger()
  root_logger.setLevel(logging.ERROR)

  # Avoid unexpected stderr output via logging.lastResort when no handlers exist.
  if not any(isinstance(h, logging.NullHandler) for h in root_logger.handlers):
    root_logger.addHandler(logging.NullHandler())

  if error_log_path:
    log_path = Path(error_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in root_logger.handlers[:]:
      if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == str(log_path):
        break
    else:
      file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
      file_handler.setLevel(logging.ERROR)
      file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
      root_logger.addHandler(file_handler)

    set_unified_log_file(str(log_path))


def _configure_runtime_defaults() -> None:
  """
  Apply runtime defaults that should be consistent across CLI and API usage.

  Notes:
  - Keep this idempotent (safe to call multiple times).
  - Avoid doing this at import-time to reduce side effects when users import the module.
  """
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  warnings.filterwarnings("ignore", category=FutureWarning)
  warnings.filterwarnings("ignore", category=UserWarning)

  # Configure matplotlib for PDF compatibility (Illustrator-friendly text)
  mpl.rcParams["pdf.fonttype"] = 42
  mpl.rcParams["ps.fonttype"] = 42

  # Keep scanpy/scvelo quiet by default; users can raise verbosity externally if needed.
  sc.settings.verbosity = 0
  scv.settings.verbosity = 0


def resolve_real_umap_dir(umap_dir: Optional[str]) -> Optional[str]:
  """
  Resolve the external real-UMAP CSV directory.

  Priority:
  1) Explicit --umap-dir / umap_dir argument
  2) Env vars: REAL_ROSEPLOT_UMAP_DIR or ANGLE_CONSISTENCY_REAL_UMAP_DIR
  3) A folder named DEFAULT_REAL_UMAP_DIRNAME in:
     - current working directory
     - next to this script

  Returns None if no directory is found. In that case, the real-data pipeline will
  fall back to computing UMAP from the input h5ad (if possible).
  """
  if umap_dir:
    p = Path(str(umap_dir))
    if p.is_dir():
      return str(p)
    logging.error(
      "Provided umap_dir does not exist or is not a directory: %s (falling back to auto-detection / compute UMAP)",
      umap_dir,
    )
    return None

  env_dir = os.environ.get("REAL_ROSEPLOT_UMAP_DIR") or os.environ.get("ANGLE_CONSISTENCY_REAL_UMAP_DIR")
  if env_dir:
    return env_dir

  candidates = [
    os.path.join(os.getcwd(), DEFAULT_REAL_UMAP_DIRNAME),
    os.path.join(os.path.dirname(__file__), DEFAULT_REAL_UMAP_DIRNAME),
  ]
  for candidate in candidates:
    if os.path.isdir(candidate):
      return candidate

  return None


def _normalize_data_type(value: str) -> str:
  if value is None:
    raise ValueError("data_type is required (expected: 'real' or 'sim').")
  v = str(value).strip().lower()
  if v in {"real", "r"}:
    return "real"
  if v in {"sim", "simulation", "s"}:
    return "sim"
  raise ValueError(f"Unknown data_type '{value}'. Expected 'real' or 'sim'.")


def _read_json_maybe_path(value: Optional[str]) -> Optional[Union[List[str], List[List[str]]]]:
  if not value:
    return None
  maybe_path = Path(value)
  raw = value
  if maybe_path.exists() and maybe_path.is_file():
    raw = maybe_path.read_text(encoding="utf-8")
  try:
    parsed = json.loads(raw)
  except Exception as e:
    raise ValueError(
      "Failed to parse differentiation paths. Provide a JSON string or a JSON file path."
    ) from e

  if isinstance(parsed, list) and (len(parsed) == 0 or isinstance(parsed[0], str)):
    return [str(x) for x in parsed]
  if isinstance(parsed, list) and isinstance(parsed[0], list):
    return [[str(x) for x in path] for path in parsed]

  raise ValueError("Invalid differentiation paths JSON. Expected a list[str] or list[list[str]].")


def _coerce_float(value: Any) -> float:
  if value is None:
    return float("nan")
  try:
    if isinstance(value, str) and value.strip() == "":
      return float("nan")
    return float(value)
  except Exception:
    return float("nan")


def _build_benchmark_long_rows(
  data_type: str,
  tool: str,
  dataset: str,
  group_type: str,
  group_to_stats: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
  rows: List[Dict[str, Any]] = []
  for group_name, stats in group_to_stats.items():
    row: Dict[str, Any] = {
      "tool": tool,
      "dataset": dataset,
      "data_type": data_type,
      "group_type": group_type,
      "group": group_name,
    }
    for col in ANGLE_COLUMNS:
      row[col] = _coerce_float(stats.get(col))
    rows.append(row)
  return pd.DataFrame(rows)


def write_benchmark_long(
  benchmark_csv_path: str,
  new_rows: pd.DataFrame,
  key_cols: Optional[List[str]] = None,
) -> None:
  """Upsert long-format benchmark rows into benchmark.csv."""
  if key_cols is None:
    key_cols = ["tool", "dataset", "data_type", "group_type", "group"]

  output_path = Path(benchmark_csv_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  if output_path.exists():
    existing = pd.read_csv(output_path)
    for col in key_cols:
      if col not in existing.columns:
        existing[col] = ""
    existing_key = existing[key_cols].astype(str)
    new_key = new_rows[key_cols].astype(str)
    merged_key = existing_key.apply(lambda r: "\x1f".join(r.values), axis=1)
    new_key_ser = new_key.apply(lambda r: "\x1f".join(r.values), axis=1)
    existing = existing.loc[~merged_key.isin(set(new_key_ser))].copy()
    combined = pd.concat([existing, new_rows], ignore_index=True)
  else:
    combined = new_rows

  combined.to_csv(output_path, index=False)


def update_benchmark_total_wide(
  benchmark_total_csv_path: str,
  tool: str,
  dataset: str,
  value_0_60: float,
  rank_1_is_worst: bool = True,
) -> None:
  """
  Upsert a single (tool, dataset) value into benchmark_total.csv (wide format),
  then recompute Mean and Rank.

  Rank behavior (default):
  - Higher Mean is better, but Rank=1 is the worst (smallest Mean).
    This is intentionally reversed compared to common "Rank=1 is best" conventions.
  """
  out_path = Path(benchmark_total_csv_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  if out_path.exists():
    df = pd.read_csv(out_path)
  else:
    df = pd.DataFrame(columns=["Tool"])

  for col in ["Mean", "Rank"]:
    if col in df.columns:
      df = df.drop(columns=[col])

  if "Tool" not in df.columns:
    df.insert(0, "Tool", "")

  if dataset not in df.columns:
    df[dataset] = np.nan

  if tool in df["Tool"].astype(str).values:
    row_idx = df.index[df["Tool"].astype(str) == str(tool)][0]
    df.at[row_idx, dataset] = value_0_60
  else:
    new_row: Dict[str, Any] = {"Tool": tool, dataset: value_0_60}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

  dataset_cols = [c for c in df.columns if c not in {"Tool", "Mean", "Rank"}]
  for c in dataset_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

  df["Mean"] = df[dataset_cols].mean(axis=1, skipna=True)

  if rank_1_is_worst:
    df["Rank"] = df["Mean"].rank(method="min", ascending=True)
    df = df.sort_values("Mean", ascending=False)
  else:
    df["Rank"] = df["Mean"].rank(method="min", ascending=False)
    df = df.sort_values("Mean", ascending=False)

  df["Rank"] = df["Rank"].where(~df["Mean"].isna(), np.nan)
  df["Rank"] = df["Rank"].astype("Int64")

  for c in dataset_cols + ["Mean"]:
    df[c] = df[c].round(2)

  df.to_csv(out_path, index=False)


def run_unified_angle_consistency(
  *,
  data_type: str,
  input_h5ad: Union[str, Path],
  output_dir: Union[str, Path],
  tool: str,
  dataset: str,
  velocity_key: str = "velocity",
  n_jobs: int = 1,
  benchmark_csv_path: Optional[str] = None,
  benchmark_total_csv_path: Optional[str] = None,
  error_log_path: Optional[str] = None,
  # Real-only
  cluster_key: Optional[str] = None,
  differentiation_paths: Optional[Union[List[str], List[List[str]]]] = None,
  umap_dir: Optional[str] = None,
  # Sim-only
  milestone_key: str = "milestone",
  topology_type: Optional[str] = None,
  npz_base_dir: Optional[str] = None,
  plot_reference_curves: bool = False,
  reference_curves_dir: Optional[str] = None,
  gt_velocity_enhancement: Optional[bool] = None,
) -> Dict[str, Any]:
  """
  Unified entrypoint.

  Returns:
    dict with keys: status, data_type, tool, dataset, pdf_path/png_path, total_ratio_0_60
  """
  output_dir_path = Path(output_dir)
  resolved_error_log_path = error_log_path or str(output_dir_path / "errors.log")
  _setup_error_logging(resolved_error_log_path)
  _configure_runtime_defaults()

  data_type_norm = _normalize_data_type(data_type)
  output_dir = str(output_dir_path)
  tool = str(tool)
  dataset = str(dataset)

  try:
    n_jobs_int = int(n_jobs)
  except Exception:
    return {"status": "error", "message": f"Invalid n_jobs '{n_jobs}' (expected a positive integer)."}
  if n_jobs_int < 1:
    return {"status": "error", "message": f"Invalid n_jobs '{n_jobs}' (expected >= 1)."}

  input_path = Path(input_h5ad)
  if not input_path.exists():
    return {"status": "error", "message": f"Input file not found: {input_path}"}

  try:
    adata = sc.read_h5ad(str(input_path))
  except Exception as e:
    return {"status": "error", "message": f"Failed to read h5ad: {e}"}

  if data_type_norm == "real":
    resolved_umap_dir = resolve_real_umap_dir(umap_dir)

    result = analyze_single_dataset(
      adata=adata,
      output_dir=output_dir,
      tool=tool,
      dataset=dataset,
      velocity_key=velocity_key,
      n_jobs=n_jobs_int,
      cluster_key=cluster_key,
      differentiation_paths=differentiation_paths,
      umap_dir=resolved_umap_dir,
      h5ad_path=str(input_path),
      benchmark_csv=None,
      benchmark_total_csv=None,
    )

    if result.get("status") != "success":
      return {"status": result.get("status", "error"), "message": result.get("message", "Unknown error")}

    angle_stats = result.get("angle_distribution") or {}
    total_ratio = _coerce_float(result.get("total_ratio_0_60"))

    if benchmark_csv_path:
      df_long = _build_benchmark_long_rows(
        data_type="real",
        tool=tool,
        dataset=dataset,
        group_type="celltype",
        group_to_stats=angle_stats,
      )
      write_benchmark_long(benchmark_csv_path, df_long)

    if benchmark_total_csv_path:
      update_benchmark_total_wide(benchmark_total_csv_path, tool, dataset, total_ratio, rank_1_is_worst=True)

    return {
      "status": "success",
      "data_type": "real",
      "tool": tool,
      "dataset": dataset,
      "pdf_path": result.get("pdf_path"),
      "png_path": result.get("png_path"),
      "total_ratio_0_60": total_ratio,
    }

  # sim
  resolved_npz_base_dir = resolve_npz_base_dir(npz_base_dir)
  if topology_type:
    try:
      topology_type = normalize_topology_type(topology_type)
    except Exception as e:
      hint = (
        " Valid topology examples: bifurcating, trifurcating, cycle-simple, linear-simple, disconnected. "
        "If this is a real dataset ID, check data_type='real'."
      )
      return {"status": "error", "message": f"Invalid topology_type '{topology_type}': {e}.{hint}"}

  if not preprocess_adata_for_method(adata, tool, velocity_key, dataset):
    return {
      "status": "error",
      "message": f"Simulation preprocessing failed: {tool} | {dataset}. See error log for details.",
    }

  gt_restore_ok = False
  try:
    gt_restore_ok, _ = add_ground_truth_velocity_dimred(
      adata=adata,
      dataset_id=dataset,
      method_name=tool,
      velocity_key=velocity_key,
      raise_on_failure=False,
      npz_base_dir=resolved_npz_base_dir,
    )
  except Exception:
    # add_ground_truth_velocity_dimred(raise_on_failure=False) should not raise, but keep it robust.
    gt_restore_ok = False

  if not compute_velocity_embeddings_for_method(adata, tool, velocity_key, dataset, n_jobs=n_jobs_int):
    return {
      "status": "error",
      "message": f"Simulation velocity embedding computation failed: {tool} | {dataset}. See error log for details.",
    }

  # Default behavior (semantic match to sim_roseplot.py CLI):
  # enable GT-based enhancements iff GT restore succeeds.
  if gt_velocity_enhancement is None:
    use_gt_enhance = bool(gt_restore_ok)
  elif bool(gt_velocity_enhancement):
    use_gt_enhance = bool(gt_restore_ok)
  else:
    use_gt_enhance = False
  try:
    sim_result = analyze_simulation_velocity_consistency(
      adata=adata,
      base_dir=output_dir,
      tool=tool,
      dataset=dataset,
      topology_type=topology_type,
      velocity_key=velocity_key,
      milestone_key=milestone_key,
      n_jobs=n_jobs_int,
      benchmark_csv_path=None,
      npz_base_dir=resolved_npz_base_dir,
      plot_reference_curves=plot_reference_curves,
      reference_curves_dir=reference_curves_dir,
      use_gt_velocity_enhancement=use_gt_enhance,
    )
  except Exception as e:
    hint = ""
    if str(dataset).strip() and str(dataset).strip()[0].isdigit():
      hint = (
        " Hint: dataset starts with digits; if this is a real dataset ID, check data_type='real' "
        "or confirm the simulated dataset name/topology."
      )
    return {"status": "error", "message": f"Simulation analysis failed: {e}.{hint}"}

  if sim_result.get("status") != "success":
    return {"status": sim_result.get("status", "error"), "message": sim_result.get("message", "Unknown error")}

  angle_stats_sim = sim_result.get("angle_distribution_stats") or {}
  total_ratio_sim = _coerce_float(sim_result.get("total_ratio_0_60"))

  if benchmark_csv_path:
    df_long = _build_benchmark_long_rows(
      data_type="sim",
      tool=tool,
      dataset=str(sim_result.get("dataset_name", dataset)),
      group_type="milestone",
      group_to_stats=angle_stats_sim,
    )
    write_benchmark_long(benchmark_csv_path, df_long)

  if benchmark_total_csv_path:
    update_benchmark_total_wide(
      benchmark_total_csv_path,
      tool,
      str(sim_result.get("dataset_name", dataset)),
      total_ratio_sim,
      rank_1_is_worst=True,
    )

  return {
    "status": "success",
    "data_type": "sim",
    "tool": tool,
    "dataset": str(sim_result.get("dataset_name", dataset)),
    "pdf_path": sim_result.get("rose_pdf_path"),
    "png_path": sim_result.get("rose_png_path"),
    "total_ratio_0_60": total_ratio_sim,
    "topology_type": sim_result.get("topology_type"),
    "embedding_basis": sim_result.get("embedding_basis"),
  }


def _pick_first_present(row: pd.Series, candidates: List[str]) -> Optional[str]:
  for c in candidates:
    if c in row.index:
      v = row.get(c)
      if v is None:
        continue
      s = str(v).strip()
      if s != "" and s.lower() != "nan":
        return s
  return None


def run_batch_from_csv(
  *,
  csv_path: Union[str, Path],
  output_dir: Union[str, Path],
  benchmark_csv_path: str,
  benchmark_total_csv_path: str,
  error_log_path: Optional[str] = None,
  default_data_type: Optional[str] = None,
  default_velocity_key: str = "velocity",
  default_milestone_key: str = "milestone",
  default_n_jobs: int = 1,
  default_umap_dir: Optional[str] = None,
  default_npz_base_dir: Optional[str] = None,
  plot_reference_curves: bool = False,
  default_gt_velocity_enhancement: Optional[bool] = None,
) -> Dict[str, Any]:
  """
  Batch runner from a unified CSV schema.

  Required columns (canonical):
  - data_type: real|sim (optional if default_data_type is provided)
  - tool
  - dataset
  - path

  Backward-compatible aliases accepted:
  - tool: method
  - dataset: dataset_id, id
  - path: h5ad_path
  - velocity_key: vkey, velocity_key
  - milestone_key: milestone_key
  - topology_type: topology_type
  - cluster_key: cluster_key
  - umap_dir: umap_dir
  - npz_base_dir: npz_base_dir, gtkey_dir
  """
  output_dir_path = Path(output_dir)
  resolved_error_log_path = error_log_path or str(output_dir_path / "errors.log")
  _setup_error_logging(resolved_error_log_path)
  _configure_runtime_defaults()

  csv_path = Path(csv_path)
  if not csv_path.exists():
    return {"status": "error", "message": f"CSV file not found: {csv_path}"}

  try:
    df = pd.read_csv(csv_path, dtype=str)
  except Exception as e:
    return {"status": "error", "message": f"Failed to read CSV: {e}"}

  ok: List[Dict[str, Any]] = []
  bad: List[Dict[str, Any]] = []

  def _parse_optional_bool(raw: Any) -> Optional[bool]:
    if raw is None:
      return None
    s = str(raw).strip().lower()
    if s in {"", "nan", "none", "null", "auto"}:
      return None
    if s in {"1", "true", "t", "yes", "y"}:
      return True
    if s in {"0", "false", "f", "no", "n"}:
      return False
    raise ValueError(f"Invalid boolean value: '{raw}' (expected true/false/auto).")

  def _parse_optional_int(raw: Any, *, default: int) -> int:
    if raw is None:
      return int(default)
    s = str(raw).strip().lower()
    if s in {"", "nan", "none", "null"}:
      return int(default)
    try:
      # Accept both "4" and "4.0" in CSV cells.
      v = int(float(s))
    except Exception as e:
      raise ValueError(f"Invalid integer value: '{raw}'") from e
    if v < 1:
      raise ValueError(f"Invalid n_jobs value: '{raw}' (expected >= 1).")
    return v

  def _read_json_paths_from_csv_cell(raw: Optional[str], csv_parent: Path) -> Optional[Union[List[str], List[List[str]]]]:
    if not raw:
      return None
    maybe_path = Path(str(raw))
    if not maybe_path.is_absolute():
      candidate = (csv_parent / maybe_path).resolve()
      if candidate.exists() and candidate.is_file():
        return _read_json_maybe_path(str(candidate))
    return _read_json_maybe_path(str(raw))

  def _resolve_existing_path(raw: str, *, csv_parent: Path, expect_dir: Optional[bool] = None) -> str:
    """
    Resolve relative paths in CSV against the CSV file directory when possible.

    - If raw is absolute, return it.
    - If raw is relative and (csv_parent / raw) exists, return the resolved path.
    - Otherwise, return raw unchanged (so callers can still interpret it elsewhere).
    """
    if not raw:
      return raw
    p = Path(str(raw))
    if p.is_absolute():
      return str(p)

    candidate = (csv_parent / p).resolve()
    if not candidate.exists():
      return str(raw)
    if expect_dir is True and not candidate.is_dir():
      return str(raw)
    if expect_dir is False and not candidate.is_file():
      return str(raw)
    return str(candidate)

  for idx, row in df.iterrows():
    tool = _pick_first_present(row, ["tool", "method"])
    dataset = _pick_first_present(row, ["dataset", "dataset_id", "id", "dataset_name"])
    path = _pick_first_present(row, ["path", "h5ad_path", "input"])
    data_type = _pick_first_present(row, ["data_type", "datatype", "type"]) or default_data_type
    velocity_key = _pick_first_present(row, ["velocity_key", "vkey"]) or default_velocity_key

    raw_n_jobs = _pick_first_present(row, ["n_jobs", "njobs", "num_workers"])
    try:
      n_jobs = _parse_optional_int(raw_n_jobs, default=default_n_jobs)
    except Exception as e:
      bad.append(
        {
          "row": int(idx),
          "tool": tool or "",
          "dataset": dataset or "",
          "error": f"Invalid n_jobs '{raw_n_jobs}': {e}",
        }
      )
      continue

    milestone_key = _pick_first_present(row, ["milestone_key"]) or default_milestone_key
    topology_type = _pick_first_present(row, ["topology_type", "topology"])
    cluster_key = _pick_first_present(row, ["cluster_key", "cluster"])
    differentiation_paths_raw = _pick_first_present(row, ["differentiation_paths", "differentiation_path", "paths"])

    csv_parent = csv_path.parent
    if path:
      path = _resolve_existing_path(path, csv_parent=csv_parent, expect_dir=False)

    umap_dir = _pick_first_present(row, ["umap_dir", "real_umap_dir"]) or default_umap_dir
    if umap_dir:
      umap_dir = _resolve_existing_path(umap_dir, csv_parent=csv_parent, expect_dir=True)

    npz_base_dir = _pick_first_present(row, ["npz_base_dir", "gtkey_dir"]) or default_npz_base_dir
    if npz_base_dir:
      npz_base_dir = _resolve_existing_path(npz_base_dir, csv_parent=csv_parent, expect_dir=True)

    if not tool or not dataset or not path:
      bad.append({"row": int(idx), "error": "Missing required columns: tool/method, dataset/dataset_id/id, path/h5ad_path"})
      continue
    if not data_type:
      bad.append({"row": int(idx), "tool": tool, "dataset": dataset, "error": "Missing data_type (real|sim)."})
      continue

    plot_ref_raw = _pick_first_present(row, ["plot_reference_curves", "plot_ref_curves"])
    reference_curves_dir = _pick_first_present(row, ["reference_curves_dir", "ref_curves_dir"])
    gt_enhance_raw = _pick_first_present(row, ["gt_velocity_enhancement", "use_gt_velocity_enhancement"])

    try:
      differentiation_paths = _read_json_paths_from_csv_cell(differentiation_paths_raw, csv_path.parent)
      plot_ref_opt = _parse_optional_bool(plot_ref_raw)
      plot_ref = bool(plot_reference_curves) if plot_ref_opt is None else bool(plot_ref_opt)
      gt_enhance = default_gt_velocity_enhancement if gt_enhance_raw is None else _parse_optional_bool(gt_enhance_raw)
    except Exception as e:
      bad.append({"row": int(idx), "tool": tool, "dataset": dataset, "error": str(e)})
      continue

    try:
      res = run_unified_angle_consistency(
        data_type=data_type,
        input_h5ad=path,
        output_dir=output_dir,
        tool=tool,
        dataset=dataset,
        velocity_key=velocity_key,
        n_jobs=n_jobs,
        benchmark_csv_path=benchmark_csv_path,
        benchmark_total_csv_path=benchmark_total_csv_path,
        error_log_path=resolved_error_log_path,
        cluster_key=cluster_key,
        differentiation_paths=differentiation_paths,
        umap_dir=umap_dir,
        milestone_key=milestone_key,
        topology_type=topology_type,
        npz_base_dir=npz_base_dir,
        plot_reference_curves=plot_ref,
        reference_curves_dir=reference_curves_dir,
        gt_velocity_enhancement=gt_enhance,
      )
      if res.get("status") == "success":
        ok.append({"tool": tool, "dataset": res.get("dataset", dataset), "data_type": res.get("data_type", data_type)})
      else:
        bad.append({"row": int(idx), "tool": tool, "dataset": dataset, "error": res.get("message", "Unknown error")})
    except Exception as e:
      bad.append({"row": int(idx), "tool": tool, "dataset": dataset, "error": str(e)})

  return {
    "status": "completed",
    "total": int(len(df)),
    "successful": int(len(ok)),
    "failed": int(len(bad)),
    "successful_details": ok,
    "failed_details": bad,
    "benchmark_csv": str(benchmark_csv_path),
    "benchmark_total_csv": str(benchmark_total_csv_path),
  }


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Unified RNA velocity angle-consistency metric (real + simulation)."
  )

  input_group = parser.add_mutually_exclusive_group(required=True)
  input_group.add_argument("--input", help="Input .h5ad path (single dataset).")
  input_group.add_argument("--csv-file", help="Input CSV path (batch mode).")

  parser.add_argument("--data-type", help="Dataset type: real|sim (required for single mode; optional for CSV if CSV has a data_type column).")
  parser.add_argument("--tool", help="Tool/method name (required for single mode).")
  parser.add_argument("--dataset", help="Dataset name/id (required for single mode).")
  parser.add_argument("--output-dir", required=True, help="Output directory root for plots.")

  parser.add_argument("--velocity-key", default="velocity", help="Velocity key (default: velocity).")
  parser.add_argument(
    "--n-jobs",
    type=int,
    default=1,
    help="Parallel jobs for scvelo.velocity_graph (default: 1 for reproducibility).",
  )

  parser.add_argument("--benchmark-csv", help="Output benchmark.csv path (default: <output-dir>/benchmark.csv).")
  parser.add_argument("--benchmark-total-csv", help="Output benchmark_total.csv path (default: <output-dir>/benchmark_total.csv).")
  parser.add_argument("--error-log", help="Error-only log file path (optional).")

  # Real-only options
  parser.add_argument("--cluster-key", help="Real data: cluster/celltype column name in adata.obs (optional).")
  parser.add_argument(
    "--umap-dir",
    help=(
      "Real data: external UMAP CSV directory (optional). "
      "Default search: ./real-umap or next to this script; can also be set via env ANGLE_CONSISTENCY_REAL_UMAP_DIR."
    ),
  )
  parser.add_argument("--differentiation-paths", help="Real data: JSON string or JSON file path for differentiation paths (optional).")

  # Sim-only options
  parser.add_argument("--milestone-key", default="milestone", help="Simulation: milestone column in adata.obs (default: milestone).")
  parser.add_argument("--topology-type", help="Simulation: topology type (optional; inferred if omitted).")
  parser.add_argument("--npz-base-dir", help="Simulation: Simdata-GTkey root directory (optional; auto-resolved if omitted).")
  parser.add_argument("--plot-reference-curves", action="store_true", help="Simulation: save reference-curve QA plot.")
  parser.add_argument("--reference-curves-dir", help="Simulation: output directory for reference-curve QA plot (optional).")

  gt_group = parser.add_mutually_exclusive_group()
  gt_group.add_argument(
    "--use-gt-velocity-enhancement",
    dest="gt_velocity_enhancement",
    action="store_true",
    default=None,
    help="Simulation: force-enable optional GT-velocity-based guide-point tweaks (requires GT restore).",
  )
  gt_group.add_argument(
    "--no-gt-velocity-enhancement",
    dest="gt_velocity_enhancement",
    action="store_false",
    default=None,
    help="Simulation: disable GT-velocity-based guide-point tweaks (default: auto when GT is available).",
  )

  args = parser.parse_args()

  output_dir = Path(args.output_dir)
  benchmark_csv = args.benchmark_csv or str(output_dir / "benchmark.csv")
  benchmark_total = args.benchmark_total_csv or str(output_dir / "benchmark_total.csv")

  if args.csv_file:
    res = run_batch_from_csv(
      csv_path=args.csv_file,
      output_dir=output_dir,
      benchmark_csv_path=benchmark_csv,
      benchmark_total_csv_path=benchmark_total,
      error_log_path=args.error_log,
      default_data_type=args.data_type,
      default_velocity_key=args.velocity_key,
      default_milestone_key=args.milestone_key,
      default_n_jobs=args.n_jobs,
      default_umap_dir=args.umap_dir,
      default_npz_base_dir=args.npz_base_dir,
      plot_reference_curves=bool(args.plot_reference_curves),
      default_gt_velocity_enhancement=args.gt_velocity_enhancement,
    )

    if res.get("status") != "completed":
      print(res.get("message", "Error"))
      return 1

    print(f"Completed: {res['successful']}/{res['total']} succeeded")
    if res["failed"] > 0:
      print(f"Failed: {res['failed']}")
      for item in res.get("failed_details", [])[:20]:
        tool = item.get("tool", "")
        dataset = item.get("dataset", "")
        err = item.get("error", "Unknown error")
        print(f"- {tool} | {dataset} | {err}")
      if res["failed"] > 20:
        print("... (more failures omitted)")

    print(f"benchmark.csv: {benchmark_csv}")
    print(f"benchmark_total.csv: {benchmark_total}")
    return 0 if res["failed"] == 0 else 2

  # single mode
  if not args.input:
    print("Error: --input is required for single-dataset mode.")
    return 1

  if not args.tool or not args.dataset or not args.data_type:
    print("Error: --data-type, --tool, and --dataset are required for single-dataset mode.")
    return 1

  differentiation_paths = _read_json_maybe_path(args.differentiation_paths)

  res = run_unified_angle_consistency(
    data_type=args.data_type,
    input_h5ad=args.input,
    output_dir=output_dir,
    tool=args.tool,
    dataset=args.dataset,
    velocity_key=args.velocity_key,
    n_jobs=args.n_jobs,
    benchmark_csv_path=benchmark_csv,
    benchmark_total_csv_path=benchmark_total,
    error_log_path=args.error_log,
    cluster_key=args.cluster_key,
    differentiation_paths=differentiation_paths,
    umap_dir=args.umap_dir,
    milestone_key=args.milestone_key,
    topology_type=args.topology_type,
    npz_base_dir=args.npz_base_dir,
    plot_reference_curves=bool(args.plot_reference_curves),
    reference_curves_dir=args.reference_curves_dir,
    gt_velocity_enhancement=args.gt_velocity_enhancement,
  )

  if res.get("status") != "success":
    print(res.get("message", "Error"))
    return 1

  print(f"Success: {res['tool']} | {res['dataset']} | 0-60: {res['total_ratio_0_60']:.2f}%")
  if res.get("pdf_path"):
    print(f"PDF: {res['pdf_path']}")
  if res.get("png_path"):
    print(f"PNG: {res['png_path']}")
  print(f"benchmark.csv: {benchmark_csv}")
  print(f"benchmark_total.csv: {benchmark_total}")
  return 0




# === BEGIN INLINED REAL_ROSEPLOT (from real_roseplot.py) ===
"""
Real Data Velocity Angle Consistency Analysis (Rose Plot)

This module analyzes the directional consistency between predicted RNA velocity
and reference directions derived from differentiation trajectories in real
single-cell datasets. Results are visualized as rose plots showing angle
distributions per cell type.

Author: RNA Velocity Benchmark Team
"""

# Runtime defaults are configured via _configure_runtime_defaults() from the
# unified entrypoints (CLI + API).


# =============================================================================
# Default Configuration
# =============================================================================

# Default cluster key mapping for known datasets
# Users can override by passing cluster_key parameter
DEFAULT_CLUSTER_KEYS: Dict[str, str] = {
    # ClusterName
    "1": "ClusterName",
    # cell_type
    "2": "cell_type", "3": "cell_type", "4": "cell_type", "6": "cell_type",
    "8": "cell_type", "9": "cell_type", "10": "cell_type", "14": "cell_type",
    "18": "cell_type", "19": "cell_type", "20": "cell_type", "21": "cell_type",
    "22": "cell_type", "23": "cell_type", "25": "cell_type", "27": "cell_type",
    "28": "cell_type", "33": "cell_type", "34": "cell_type", "35": "cell_type",
    "37": "cell_type", "38": "cell_type", "39": "cell_type", "42": "cell_type",
    "44": "cell_type", "46": "cell_type", "49": "cell_type", "50": "cell_type",
    "51": "cell_type", "52": "cell_type", "53": "cell_type", "54": "cell_type",
    "55": "cell_type", "56": "cell_type", "57": "cell_type", "59": "cell_type",
    "58": "Class",
    # Celltype / celltype / type2
    "5": "Celltype", "11": "celltype", "13": "type2",
    "15": "Annotation", "17": "celltype", "24": "celltype", "26": "celltype",
    "43": "celltype", "47": "celltype", "48": "original_clusters",
    # clusters
    "7": "clusters", "40": "clusters", "41": "clusters",
    # phase
    "29": "phase", "30": "phase", "31": "phase", "32": "phase",
    # Special types
    "12": "cell_cycle_phase", "45": "lineage_cat",
    "16": "time", "36": "time",
}

# Default differentiation paths for known datasets
# Format: dataset_id -> list of cell type sequences (single or multiple paths)
DEFAULT_DIFFERENTIATION_PATHS: Dict[str, Any] = {
    # Single-path datasets
    "1": ["RadialGlia", "nIPC", "Nbl1", "ImmGranule1", "Granule"],
    "2": ["sympathoblasts", "Schwann cell precursors", "chromaffin"],
    "5": ["TA.Early", "Enterocyte.Progenitor",
          "Enterocyte.Immature.Proximal", "Enterocyte.Mature.Proximal"],
    "6": ["Radial glia", "Neuroblast", "Immature neuron", "Neuron"],
    "7": ["Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine"],
    "8": ["DIplotene/Secondary spermatocytes", "Early Round spermatids",
          "Mid Round spermatids", "Late Round spermatids"],
    "10": ["NF Oligo", "OPC", "Oligo"],
    "11": ["Blood progenitors 1", "Blood progenitors 2",
           "Erythroid1", "Erythroid2", "Erythroid3"],
    "12": ["M", "M-G1", "G1-S", "S", "G2-M"],
    "13": ["MEMP", "Early Erythroid", "Mid  Erythroid", "Late Erythroid"],
    "14": ["Stem cells", "TA cells", "Enterocytes"],
    "15": ["Neuroblast", "AC/HC"],
    "17": ["COPs", "NFOLs", "MFOLs"],
    "24": ["RG, Astro, OPC", "IPC", "V-SVZ", "Upper Layer", "Deeper Layer"],
    "25": ["embryonic day 3", "embryonic day 4", "embryonic day 5",
           "embryonic day 6", "embryonic day 7"],
    "26": ["TAC-1", "TAC-2", "IRS"],
    "28": ["HSC", "MPP", "LMPP", "MEP", "Erythrocyte"],
    "34": ["Early mesenchyme", "Limb mesenchyme",
           "Chondrocyte progenitors", "Chondrocytes and osteoblasts"],
    "35": ["Epididymal cells  ", "Embryonic stem cells  ",
           "Hematopoietic stem cells  "],
    "37": ["Neural progenitors", "Sensory progenitors"],
    "40": ["Undifferentiated", "DC-like-monocyte"],
    "42": ["Acinar", "Ductal"],
    "43": ["Blood progenitors 1", "Blood progenitors 2",
           "Erythroid1", "Erythroid2", "Erythroid3"],
    "45": ["5", "6", "7", "8", "9", "10", "11", "12"],
    "46": ["Pre-EMT", "Early EMT 1", "Mesenchymal-1",
           "Mesenchymal-2", "AT1-like"],
    "47": ["CD34+", "CD19+ B"],
    "48": ["HSCs #1.", "HSCs #2", "LMPPs #1", "LMPPs #2",
           "Myel. prog. #1", "Myel. prog. #2", "Myel. prog. #3"],
    "49": ["Quiescent-like", "ILC2-like", "ILC3-like"],
    "50": ["HSC", "MEP-like", "Ery"],
    "51": ["spermatocytes", "round spermatids", "elongating spermatids"],
    "52": ["Immature myocardial cells", "Cardiomyocytes-1"],
    "53": ["Ectoderm", "Neural tube", "Radial glia", "Neuroblast", "Neuron"],
    "57": ["Immature myocardial cells", "Cardiomyocytes-1"],
    "58": ["Gastrulation", "Ectoderm", "Neural tube", "Neuroblast", "Neuron"],

    # Multi-path datasets (list of lists)
    # Dataset 27: supports both "nlPC/ExN" and "nIPC/ExN" variants
    "27": [["ExM", "RG/Astro", "nlPC/ExN"], ["ExDp", "SP", "mGPC/OPC"]],
    "33": [["Neural stem cells", "Gliogenic progenitors"],
           ["Differentiating GABA interneurons", "GABA interneurons"]],
    "38": [["Neuronal progenitor cells", "Neuronal cells"],
           ["Neural progenitor cells", "Interneuron cells"]],
    "39": [["Granulocyte macrophage progenitor", "CD14+ monocyte", "CD16+ monocyte"],
           ["Hematopoietic stem cells", "Erythroid progenitor"]],
    "41": [["OPC", "OL"], ["Radial Glia-like", "Astrocytes"],
           ["Neuroblast", "Granule immature"]],
    "44": [["Angioblasts", "Primitive blood"],
           ["Visceral endoderm", "Gut endoderm"]],
    "54": [["Lateral plate mesoderm", "Splanchnic mesoderm", "Cardiomyocytes"],
           ["NMP", "Mixed mesenchymal mesoderm", "Haematoendothelial progenitors",
            "Blood progenitors", "Erythroid"]],
    "55": [["Lateral plate mesoderm", "Splanchnic mesoderm", "Cardiomyocytes"],
           ["NMP", "Mixed mesenchymal mesoderm", "Haematoendothelial progenitors",
            "Blood progenitors", "Erythroid"]],
    "56": [["Lateral plate mesoderm", "Splanchnic mesoderm", "Cardiomyocytes"],
           ["NMP", "Mixed mesenchymal mesoderm", "Haematoendothelial progenitors",
            "Blood progenitors", "Erythroid"]],
    "59": [["Lateral plate mesoderm", "Splanchnic mesoderm", "Cardiomyocytes"],
           ["NMP", "Mixed mesenchymal mesoderm", "Haematoendothelial progenitors",
            "Blood progenitors", "Erythroid"]],
}

# Cell type name variants that should be treated as equivalent
# Format: canonical_name -> [alternative_names]
CELLTYPE_VARIANTS: Dict[str, List[str]] = {
    "nlPC/ExN": ["nIPC/ExN"],
    "nIPC/ExN": ["nlPC/ExN"],
}

# Special label formatting for long cell type names
LABEL_FORMATTING: Dict[str, Dict[str, str]] = {
    "5": {
        "Enterocyte.Progenitor": "Enterocyte\nProgenitor",
        "Enterocyte.Immature.Proximal": "Enterocyte\nImmature.Proximal",
        "Enterocyte.Mature.Proximal": "Enterocyte\nMature.Proximal",
    },
    "8": {"DIplotene/Secondary spermatocytes": "DIplotene/Secondary\nspermatocytes"},
    "33": {"Differentiating GABA interneurons": "Differentiating\nGABA interneurons"},
    "34": {
        "Chondrocyte progenitors": "Chondrocyte\nprogenitors",
        "Chondrocytes and osteoblasts": "Chondrocytes & osteoblasts",
    },
    "39": {"Granulocyte macrophage progenitor": "Granulocyte\nmacrophage progenitor"},
    "54": {
        "Mixed mesenchymal mesoderm": "Mixed mesenchymal\nmesoderm",
        "Haematoendothelial progenitors": "Haematoendothelial\nprogenitors",
    },
    "55": {
        "Mixed mesenchymal mesoderm": "Mixed mesenchymal\nmesoderm",
        "Haematoendothelial progenitors": "Haematoendothelial\nprogenitors",
    },
    "56": {
        "Mixed mesenchymal mesoderm": "Mixed mesenchymal\nmesoderm",
        "Haematoendothelial progenitors": "Haematoendothelial\nprogenitors",
    },
    "59": {
        "Mixed mesenchymal mesoderm": "Mixed mesenchymal\nmesoderm",
        "Haematoendothelial progenitors": "Haematoendothelial\nprogenitors",
    },
}

# Color palette for rose plots
PALETTE_34 = [
    "#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4", "#66c2a5",
    "#3288bd", "#abdda4", "#e6f598", "#fee08b", "#f46d43", "#e7298a",
    "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c",
    "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a", "#ffff99", "#b15928",
    "#8dd3c7", "#bc80bd", "#ccebc5", "#ffed6f", "#999999", "#8B0000",
    "#006400", "#FF69B4", "#00CED1", "#FFD700",
]

# Rose plot angle bin configuration
ROSE_PLOT_N_BINS = 18
ROSE_PLOT_MAX_ANGLE = 180


# =============================================================================
# Utility Functions
# =============================================================================

def get_optimal_n_jobs() -> int:
    """
    Deprecated helper retained for backward compatibility.

    This project now defaults to n_jobs=1 for reproducibility; prefer passing n_jobs
    explicitly via the unified API/CLI instead of using CPU-based heuristics.
    """
    try:
        cpu_count = multiprocessing.cpu_count()
        return max(min(cpu_count // 2, 32), 1)
    except Exception:
        return 1


def extract_numeric_prefix(dataset: str) -> str:
    """Extract numeric prefix from dataset ID (e.g., '28_test' -> '28')."""
    match = re.match(r"^(\d+)", str(dataset))
    return match.group(1) if match else dataset


def parse_time_value(time_str: str) -> Tuple[float, str]:
    """Parse time value, extracting numeric part and unit."""
    if time_str.lower() == "dmso" or not any(c.isdigit() for c in time_str):
        return float("inf"), "other"

    num_str = ""
    for char in time_str:
        if char.isdigit() or char == ".":
            num_str += char
        else:
            break
    unit = time_str[len(num_str):]

    try:
        return float(num_str) if num_str else float("inf"), unit
    except ValueError:
        return float("inf"), "other"


def sort_time_clusters(clusters: List[str]) -> List[str]:
    """Sort clusters by time value."""
    filtered = [c for c in clusters if str(c).lower() != "dmso"]
    parsed = [(c, *parse_time_value(str(c))) for c in filtered]
    sorted_vals = sorted(parsed, key=lambda x: (x[1], x[2]))
    return [item[0] for item in sorted_vals]


def format_cell_type_label(cell_type: str, dataset: str) -> str:
    """Format cell type label with line breaks for long names."""
    num_prefix = extract_numeric_prefix(dataset)
    if num_prefix in LABEL_FORMATTING:
        return LABEL_FORMATTING[num_prefix].get(cell_type, cell_type)
    return cell_type


def resolve_celltype_variant(
    cell_type: str,
    available_types: List[str]
) -> Optional[str]:
    """
    Resolve cell type name to available variant.

    Handles cases like 'nlPC/ExN' vs 'nIPC/ExN' where different datasets
    may use different naming conventions.
    """
    if cell_type in available_types:
        return cell_type

    # Check variants
    if cell_type in CELLTYPE_VARIANTS:
        for variant in CELLTYPE_VARIANTS[cell_type]:
            if variant in available_types:
                return variant

    return None


def resolve_differentiation_paths(
    paths: Union[List[str], List[List[str]]],
    available_types: List[str]
) -> Union[List[str], List[List[str]]]:
    """
    Resolve differentiation paths by matching cell type variants.

    Automatically handles naming variants (e.g., nlPC/ExN vs nIPC/ExN).
    """
    def resolve_single_path(path: List[str]) -> List[str]:
        resolved = []
        for ct in path:
            resolved_ct = resolve_celltype_variant(ct, available_types)
            if resolved_ct:
                resolved.append(resolved_ct)
            else:
                resolved.append(ct)  # Keep original if no match found
        return resolved

    if not paths:
        return paths

    # Check if multi-path (list of lists)
    if isinstance(paths[0], list):
        return [resolve_single_path(p) for p in paths]
    else:
        return resolve_single_path(paths)


# =============================================================================
# Data Preprocessing Functions
# =============================================================================

def fix_sparse_matrix_format(adata: sc.AnnData) -> sc.AnnData:
    """Convert dense matrices in obsp to sparse CSR format."""
    for key in ["connectivities", "distances"]:
        if key in adata.obsp and isinstance(adata.obsp[key], np.ndarray):
            adata.obsp[key] = csr_matrix(adata.obsp[key])
    return adata


def fix_data_types(adata: sc.AnnData) -> Optional[sc.AnnData]:
    """
    Fix data type issues in AnnData object.

    - Normalizes integer count matrices
    - Ensures var_names and obs_names are strings
    """
    try:
        # Handle integer count matrices
        if adata.X.dtype.kind in ["i", "u"]:
            logging.info(f"Normalizing integer matrix (dtype: {adata.X.dtype})")
            scv.pp.filter_and_normalize(adata)
        elif adata.X.dtype.kind != "f":
            logging.error(f"Invalid X matrix dtype: {adata.X.dtype}")
            return None

        # Ensure string indices
        adata.var_names = adata.var_names.astype(str)
        adata.obs_names = adata.obs_names.astype(str)

        return adata
    except Exception as e:
        logging.error(f"Data type fix failed: {e}")
        return None


def robust_neighbors_computation(adata: sc.AnnData) -> Optional[sc.AnnData]:
    """Compute neighbors with error handling."""
    try:
        if adata.n_vars == 0:
            logging.error("No genes in data")
            return None

        fixed_adata = fix_data_types(adata)
        if fixed_adata is None:
            return None

        scv.pp.neighbors(fixed_adata)
        return fix_sparse_matrix_format(fixed_adata)
    except Exception as e:
        logging.error(f"Neighbors computation failed: {e}")
        return None


def repair_neighbor_graph(
    adata: sc.AnnData,
    vkey: str,
    n_jobs: int
) -> Tuple[bool, sc.AnnData]:
    """Repair corrupted neighbor graph by removing isolated/duplicate cells."""
    try:
        # Remove isolated cells (zero neighbors)
        if "connectivities" in adata.obsp:
            neighbor_counts = np.array(
                adata.obsp["connectivities"].sum(axis=1)
            ).flatten()
            isolated_mask = neighbor_counts == 0
            n_isolated = isolated_mask.sum()

            if n_isolated > 0:
                logging.warning(f"Removing {n_isolated} isolated cells")
                adata = adata[~isolated_mask].copy()

        # Remove duplicate cells
        scv.pp.remove_duplicate_cells(adata)

        # Recompute neighbors and velocity graph
        scv.pp.neighbors(adata)
        scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=n_jobs)

        return True, adata
    except Exception as e:
        logging.error(f"Neighbor graph repair failed: {e}")
        return False, adata


def robust_velocity_graph_computation(
    adata: sc.AnnData,
    vkey: str = "velocity",
    n_jobs: int = 1,
) -> Tuple[bool, sc.AnnData]:
    """
    Compute velocity graph with fallback strategies.

    Handles:
    - WRITEBACKIFCOPY errors (fallback to single-core)
    - Corrupted neighbor graphs (repair and retry)
    - show_progress_bar compatibility issues
    """
    def _try_compute(n_jobs: int) -> Tuple[bool, Optional[Exception]]:
        try:
            scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=n_jobs,
                                  show_progress_bar=False)
            return True, None
        except TypeError as e:
            if "show_progress_bar" in str(e):
                scv.tl.velocity_graph(adata, vkey=vkey, n_jobs=n_jobs)
                return True, None
            return False, e
        except Exception as e:
            return False, e

    try:
        requested_jobs = int(n_jobs)
    except Exception:
        requested_jobs = 1
    requested_jobs = max(requested_jobs, 1)
    adata = fix_sparse_matrix_format(adata)

    try:
        # Try requested parallelism first (default: 1 for reproducibility)
        success, error = _try_compute(requested_jobs)
        if success:
            return True, adata

        if isinstance(error, ValueError):
            error_str = str(error).lower()

            # WRITEBACKIFCOPY error: retry with single core
            if "writebackifcopy" in error_str and "read-only" in error_str:
                logging.warning("WRITEBACKIFCOPY error, retrying single-core")
                success, _ = _try_compute(1)
                return success, adata

            # Corrupted neighbor graph: repair and retry
            if "neighbor graph" in error_str and "corrupted" in error_str:
                logging.warning("Corrupted neighbor graph, attempting repair")
                repaired, repaired_adata = repair_neighbor_graph(adata, vkey, requested_jobs)
                if repaired:
                    return True, repaired_adata
                if requested_jobs != 1:
                    return repair_neighbor_graph(repaired_adata, vkey, 1)
                return False, repaired_adata

        # Generic fallback: if requested_jobs != 1, retry single-core once.
        if requested_jobs != 1:
            success, _ = _try_compute(1)
            return success, adata

        logging.error(f"Velocity graph computation failed: {error}")
        return False, adata

    except Exception as e:
        logging.error(f"Velocity graph computation error: {e}")
        return False, adata


# =============================================================================
# UMAP Handling Functions
# =============================================================================

def load_external_umap(
    dataset: str,
    h5ad_path: str,
    umap_dir: str
) -> Optional[pd.DataFrame]:
    """
    Load external UMAP coordinates from CSV file.

    Expected CSV format: index column with cell names, UMAP1, UMAP2 columns.
    """
    if umap_dir is None or not os.path.isdir(umap_dir):
        return None

    num_prefix = extract_numeric_prefix(dataset)

    # Special handling for dataset 10
    if num_prefix == "10":
        filename = ("new_10_mouse_oldbrain_with_celltype_addUmap.csv"
                   if "new" in h5ad_path.lower()
                   else "10_mouse_oldbrain_GSE129788_with_celltype_addUmap.csv")
        file_path = Path(umap_dir) / filename
        if file_path.exists():
            return pd.read_csv(file_path, index_col=0)

    # Special handling for dataset 36 (try two files)
    if num_prefix == "36":
        for filename in [
            "36_new_GPT_mouse_spermary_cells_with_celltype_addUmap.csv",
            "36_mouse_spermary_cells_with_celltype_addUmap.csv"
        ]:
            file_path = Path(umap_dir) / filename
            if file_path.exists():
                return pd.read_csv(file_path, index_col=0)
        return None

    # Generic search for matching UMAP file
    for filename in os.listdir(umap_dir):
        if filename.endswith("_addUmap.csv") and str(dataset) in filename:
            parts = filename.replace("_addUmap.csv", "").split("_")
            if str(dataset) in parts:
                return pd.read_csv(Path(umap_dir) / filename, index_col=0)

    return None


def match_umap_coordinates(
    adata: sc.AnnData,
    umap_df: pd.DataFrame,
    min_match_rate: float = 0.99
) -> Optional[sc.AnnData]:
    """
    Match and assign UMAP coordinates from external DataFrame.

    Returns filtered AnnData with matched cells if match rate >= min_match_rate.
    """
    # Remove duplicate indices
    if umap_df.index.duplicated().any():
        umap_df = umap_df[~umap_df.index.duplicated(keep="first")]

    # Calculate match rate
    common_cells = set(umap_df.index) & set(adata.obs_names)
    match_rate = len(common_cells) / len(adata.obs_names)

    if match_rate < min_match_rate or not common_cells:
        logging.error(
            "External UMAP match failed "
            f"(match_rate={match_rate:.4f}, min_match_rate={min_match_rate}, "
            f"n_common={len(common_cells)}, n_adata={len(adata.obs_names)}, n_umap={len(umap_df.index)})"
        )
        return None

    # Filter to common cells
    common_list = list(common_cells)
    adata_filtered = adata[common_list].copy()

    # Align UMAP coordinates
    umap_aligned = umap_df.reindex(adata_filtered.obs_names).dropna()
    if len(umap_aligned) != adata_filtered.n_obs:
        adata_filtered = adata_filtered[umap_aligned.index].copy()

    # Assign coordinates
    coords = umap_aligned[["UMAP1", "UMAP2"]].values
    if coords.shape == (adata_filtered.n_obs, 2):
        adata_filtered.obsm["X_umap"] = coords
        return adata_filtered

    logging.error(
        "External UMAP assignment failed "
        f"(coords_shape={coords.shape}, expected=({adata_filtered.n_obs}, 2))"
    )
    return None


def ensure_umap(
    adata: sc.AnnData,
    dataset: str,
    tool: str,
    h5ad_path: str,
    umap_dir: Optional[str] = None
) -> Optional[sc.AnnData]:
    """
    Ensure AnnData has UMAP coordinates.

    Priority:
    1. Existing X_umap
    2. original_umap in obsm
    3. External UMAP file (if umap_dir provided)
    4. Compute new UMAP
    """
    # Already has UMAP
    if "X_umap" in adata.obsm:
        return adata

    # Use original_umap if available
    if "original_umap" in adata.obsm:
        orig = adata.obsm["original_umap"]
        if orig.shape[0] == adata.n_obs and orig.shape[1] >= 2:
            adata.obsm["X_umap"] = orig[:, :2]
            return adata

    # Try external UMAP file
    if umap_dir:
        umap_df = load_external_umap(dataset, h5ad_path, umap_dir)
        if umap_df is not None:
            result = match_umap_coordinates(adata, umap_df)
            if result is not None:
                return result

    # Compute UMAP
    try:
        if "neighbors" not in adata.uns:
            if adata.n_vars == 0:
                logging.error(f"No genes for UMAP computation: {tool}_{dataset}")
                return None
            neighbors_adata = robust_neighbors_computation(adata)
            if neighbors_adata is None:
                return None
            adata = neighbors_adata

        sc.tl.umap(adata)
        return adata
    except Exception as e:
        logging.error(f"UMAP computation failed: {tool}_{dataset}: {e}")
        return None


# =============================================================================
# Velocity Preprocessing Functions
# =============================================================================

def preprocess_tfvelo(adata: sc.AnnData) -> Optional[sc.AnnData]:
    """
    Preprocess TFvelo data: compute velocity from velo_hat and fit_scaling_y.

    velocity = velo_hat / fit_scaling_y
    """
    if "velocity" in adata.layers:
        return adata

    if "velo_hat" not in adata.layers or "fit_scaling_y" not in adata.var.columns:
        logging.error("TFvelo requires 'velo_hat' layer and 'fit_scaling_y' in var")
        return None

    try:
        n_cells = adata.shape[0]
        scaling_y = np.array(adata.var["fit_scaling_y"])
        expanded_scaling = np.expand_dims(scaling_y, 0).repeat(n_cells, axis=0)
        adata.layers["velocity"] = adata.layers["velo_hat"] / expanded_scaling
        return adata
    except Exception as e:
        logging.error(f"TFvelo preprocessing failed: {e}")
        return None


def preprocess_phylovelo(
    adata: sc.AnnData,
    vkey: str
) -> sc.AnnData:
    """
    Preprocess PhyloVelo data: rename velocity embedding key.

    PhyloVelo stores low-dimensional velocity directly in obsm.
    """
    velocity_dimred_key = f"{vkey}_umap"
    if vkey in adata.obsm and velocity_dimred_key not in adata.obsm:
        adata.obsm[velocity_dimred_key] = adata.obsm[vkey]
    return adata


def ensure_velocity_embedding(
    adata: sc.AnnData,
    dataset: str,
    tool: str,
    vkey: str = "velocity",
    n_jobs: int = 1,
) -> Tuple[bool, sc.AnnData]:
    """
    Ensure velocity embedding exists in UMAP space.

    Computes velocity_graph and velocity_embedding if needed.
    """
    velocity_dimred_key = f"{vkey}_umap"

    if velocity_dimred_key in adata.obsm:
        return True, adata

    # Handle layer name variants (M_u -> Mu, M_s -> Ms)
    if "M_u" in adata.layers and "Mu" not in adata.layers:
        adata.layers["Mu"] = adata.layers["M_u"].copy()
    if "M_s" in adata.layers and "Ms" not in adata.layers:
        adata.layers["Ms"] = adata.layers["M_s"].copy()

    # Ensure moments (Ms, Mu layers)
    if "Ms" not in adata.layers or "Mu" not in adata.layers:
        fixed_adata = robust_neighbors_computation(adata)
        if fixed_adata is None:
            logging.error(f"Neighbors failed: {tool}_{dataset}")
            return False, adata

        try:
            scv.pp.moments(fixed_adata)
            adata = fixed_adata
        except Exception as e:
            logging.error(f"Moments computation failed: {tool}_{dataset}: {e}")
            return False, adata
    else:
        # Ensure neighbor graph exists
        if "neighbors" not in adata.uns or "connectivities" not in adata.obsp:
            fixed_adata = robust_neighbors_computation(adata)
            if fixed_adata is None:
                return False, adata
            adata = fixed_adata
        else:
            adata = fix_sparse_matrix_format(adata)

    # Compute velocity graph
    success, adata = robust_velocity_graph_computation(adata, vkey, n_jobs=n_jobs)
    if not success:
        logging.error(f"Velocity graph failed: {tool}_{dataset}")
        return False, adata

    # Compute velocity embedding
    try:
        scv.tl.velocity_embedding(adata, basis="umap", vkey=vkey)
        return True, adata
    except Exception as e:
        logging.error(f"Velocity embedding failed: {tool}_{dataset}: {e}")
        return False, adata


# =============================================================================
# Cluster Key Detection
# =============================================================================

def detect_cluster_key(adata: sc.AnnData) -> Optional[str]:
    """
    Auto-detect cluster key column in adata.obs.

    Priority: time > phase > cell_type > cluster_name > cluster > leiden
    """
    patterns = [
        (r"^time$", re.IGNORECASE),
        (r"^phase$", re.IGNORECASE),
        (r"[Cc]ell[._]?[Tt]ype[s]?", 0),
        (r"[Cc]luster[s]?[._]?[Nn]ame[s]?", re.IGNORECASE),
        (r"^cluster[s]?$", re.IGNORECASE),
        (r"^leiden$", re.IGNORECASE),
    ]

    for pattern, flags in patterns:
        for col in adata.obs.columns:
            if flags == re.IGNORECASE:
                if re.fullmatch(pattern, col, flags):
                    return col
            else:
                if re.search(pattern, col):
                    return col

    return None


def get_cluster_key(
    adata: sc.AnnData,
    dataset: str,
    user_cluster_key: Optional[str] = None
) -> Optional[str]:
    """
    Get cluster key, with user override support.

    Priority: user_cluster_key > default for dataset > auto-detect
    """
    # User override
    if user_cluster_key and user_cluster_key in adata.obs.columns:
        return user_cluster_key

    # Default for dataset
    num_prefix = extract_numeric_prefix(dataset)
    default_key = DEFAULT_CLUSTER_KEYS.get(num_prefix) or DEFAULT_CLUSTER_KEYS.get(dataset)
    if default_key and default_key in adata.obs.columns:
        return default_key

    # Auto-detect
    return detect_cluster_key(adata)


def process_time_column(
    adata: sc.AnnData,
    cluster_key: str,
    dataset: str
) -> None:
    """
    Process time-based cluster column: add units and sort.

    Handles datasets 18 and 37 which may have numeric-only time values.
    """
    if cluster_key not in adata.obs:
        return

    num_prefix = extract_numeric_prefix(dataset)
    unique_vals = adata.obs[cluster_key].astype(str).unique().tolist()

    # Add 'h' unit for datasets 18/37 if all numeric
    if num_prefix in ["18", "37"] and cluster_key.lower() == "time":
        if all(str(x).isdigit() for x in unique_vals if str(x) != "nan"):
            adata.obs[cluster_key] = (
                adata.obs[cluster_key].astype(str)
                .apply(lambda x: f"{x}h" if x.isdigit() else x)
            )
            unique_vals = adata.obs[cluster_key].astype(str).unique().tolist()

    # Sort by time value
    if all(str(x).endswith("h") and str(x)[:-1].isdigit() for x in unique_vals):
        categories = sorted(unique_vals, key=lambda x: int(str(x)[:-1]))
        adata.obs[cluster_key] = pd.Categorical(
            adata.obs[cluster_key].astype(str),
            categories=categories,
            ordered=True
        )
    elif all(str(x).endswith("min") and str(x)[:-3].isdigit() for x in unique_vals):
        categories = sorted(unique_vals, key=lambda x: int(str(x)[:-3]))
        adata.obs[cluster_key] = pd.Categorical(
            adata.obs[cluster_key].astype(str),
            categories=categories,
            ordered=True
        )


# =============================================================================
# Path Validation Functions
# =============================================================================

def validate_differentiation_paths(
    paths: Union[List[str], List[List[str]]],
    available_clusters: List[str],
    dataset: str,
    cluster_key: str
) -> List[List[str]]:
    """
    Validate differentiation paths against available clusters.

    For time/phase data: filter to available clusters without raising errors.
    For other data: raise error if any expected cell type is missing.

    Returns list of validated paths (always list of lists).
    """
    is_time_or_phase = cluster_key.lower() in ["time", "phase"] or \
                       set(["G1", "S", "G2M"]).issubset(set(available_clusters))

    # Normalize to list of lists
    if not paths:
        return []
    if not isinstance(paths[0], list):
        paths = [paths]

    # Resolve cell type variants
    paths = resolve_differentiation_paths(paths, available_clusters)

    valid_paths = []
    for idx, path in enumerate(paths):
        available_in_path = [c for c in path if c in available_clusters]
        missing = [c for c in path if c not in available_clusters]

        if is_time_or_phase:
            # For time/phase: just use what's available
            if available_in_path:
                valid_paths.append(available_in_path)
        else:
            # For cell type data: require all specified types
            if missing:
                raise ValueError(
                    f"Dataset '{dataset}' path {idx+1} missing cell types: {missing}"
                )
            if len(available_in_path) < 2:
                raise ValueError(
                    f"Dataset '{dataset}' path {idx+1} has only "
                    f"{len(available_in_path)} cell types (need >=2)"
                )
            valid_paths.append(available_in_path)

    return valid_paths


# =============================================================================
# Reference Direction Calculation
# =============================================================================

def calculate_reference_directions(
    adata: sc.AnnData,
    cluster_key: str,
    differentiation_paths: Optional[List[List[str]]] = None
) -> np.ndarray:
    """
    Calculate reference directions from spline-fitted trajectories.

    For each cell, finds the closest point on the trajectory spline
    and returns the tangent direction at that point.
    """
    X_umap = adata.obsm["X_umap"]
    reference_dirs = np.zeros((adata.n_obs, 2))

    unique_clusters = adata.obs[cluster_key].unique().astype(str).tolist()

    # Detect data type
    is_phase = cluster_key.lower() == "phase" or \
               set(["G1", "S", "G2M"]).issubset(set(unique_clusters))
    is_time = cluster_key.lower() == "time" or any(
        ("min" in c.lower() or "h" in c.lower() or "hour" in c.lower() or
         "sec" in c.lower() or "day" in c.lower()) and any(ch.isdigit() for ch in c)
        for c in unique_clusters
    )

    # Determine paths to use
    if differentiation_paths and isinstance(differentiation_paths[0], list):
        paths = differentiation_paths
    elif differentiation_paths:
        paths = [differentiation_paths]
    elif is_phase:
        phase_order = ["G1", "S", "G2M"]
        paths = [[c for c in phase_order if c in unique_clusters]]
    elif is_time:
        paths = [sort_time_clusters(unique_clusters)]
    else:
        paths = [unique_clusters]

    # Filter to valid clusters
    paths = [[c for c in path if c in unique_clusters] for path in paths]
    paths = [p for p in paths if len(p) >= 2]

    if not paths:
        return reference_dirs

    # Process each path
    for path in paths:
        # Compute cluster centers
        centers = []
        for cluster in path:
            mask = adata.obs[cluster_key].astype(str) == cluster
            if mask.sum() > 0:
                center = X_umap[mask].mean(axis=0)
                centers.append((cluster, center))

        if len(centers) < 2:
            continue

        # Sort by path order
        order_dict = {c: i for i, c in enumerate(path)}
        centers.sort(key=lambda x: order_dict.get(str(x[0]), 999))

        center_coords = np.array([c[1] for c in centers])
        center_clusters = [c[0] for c in centers]

        # Handle cyclic data (phase)
        is_cycle = is_phase and all(
            str(c) in ["G1", "S", "G2M"] for c in center_clusters
        )
        if is_cycle and len(center_coords) == 3:
            center_coords = np.vstack([center_coords, center_coords[0]])

        # Fit spline
        k = min(3, len(center_coords) - 1)
        try:
            tck, _ = splprep(
                [center_coords[:, 0], center_coords[:, 1]],
                s=0, k=k, per=is_cycle
            )
        except Exception:
            continue

        # Calculate reference direction for each cell in this path
        path_mask = adata.obs[cluster_key].astype(str).isin(path)

        for i in np.where(path_mask)[0]:
            point = X_umap[i]

            # Find closest point on spline
            min_dist = float("inf")
            min_u = 0
            for u_val in np.linspace(0, 1, 200):
                curve_point = np.array(splev(u_val, tck))
                dist = np.linalg.norm(point - curve_point)
                if dist < min_dist:
                    min_dist = dist
                    min_u = u_val

            # Get tangent direction
            deriv = np.array(splev(min_u, tck, der=1))
            norm = np.linalg.norm(deriv)
            if norm > 0:
                reference_dirs[i] = deriv / norm

    return reference_dirs


# =============================================================================
# Angle Statistics Functions
# =============================================================================

def calculate_angle_statistics(
    angles: np.ndarray,
    adata: sc.AnnData,
    cluster_key: str,
    valid_clusters: List[str],
    valid_mask: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Calculate angle distribution statistics per cell type.

    Returns dict mapping cell_type -> {bin_label: percentage, ...}
    """
    bins = [(0, 30), (30, 60), (60, 80), (80, 90), (60, 90),
            (90, 100), (100, 120), (90, 120), (120, 150), (150, 180)]
    labels = ["0-30", "30-60", "60-80", "80-90", "60-90",
              "90-100", "100-120", "90-120", "120-150", "150-180"]

    stats = {}
    for cell_type in valid_clusters:
        cell_mask = (adata.obs[cluster_key].astype(str) == cell_type) & valid_mask

        if cell_mask.sum() == 0:
            stats[cell_type] = {lab: np.nan for lab in labels}
            stats[cell_type]["valid_ratio"] = np.nan
            continue

        cell_angles = angles[cell_mask]
        total_cells = len(cell_angles)
        valid_angles = cell_angles[~np.isnan(cell_angles)]

        if total_cells == 0:
            stats[cell_type] = {lab: np.nan for lab in labels}
            stats[cell_type]["valid_ratio"] = np.nan
            continue

        # Compute bin percentages
        bin_counts = {}
        for i, (min_a, max_a) in enumerate(bins):
            count = np.sum((valid_angles >= min_a) & (valid_angles <= max_a))
            bin_counts[labels[i]] = count / total_cells * 100

        bin_counts["valid_ratio"] = len(valid_angles) / total_cells * 100
        stats[cell_type] = bin_counts

    return stats


def calculate_total_angle_ratio(
    angles: np.ndarray,
    valid_mask: np.ndarray
) -> float:
    """Calculate percentage of cells with angles in 0-60 degree range."""
    valid_angles = angles[valid_mask]
    valid_angles = valid_angles[~np.isnan(valid_angles)]

    if len(valid_angles) == 0:
        return np.nan

    count_0_60 = np.sum((valid_angles >= 0) & (valid_angles <= 60))
    return (count_0_60 / len(valid_angles)) * 100


# =============================================================================
# Rose Plot Visualization
# =============================================================================

def plot_rose_diagram(
    adata: sc.AnnData,
    reference_directions: np.ndarray,
    predicted_velocities: np.ndarray,
    cluster_key: str,
    pdf_path: str,
    png_path: str,
    tool: str,
    differentiation_paths: List[List[str]],
    dataset: str
) -> None:
    """
    Generate rose diagram showing angle distribution per cell type.

    Creates a multi-panel figure with one rose plot per cell type,
    organized by differentiation paths.
    """
    n_paths = len(differentiation_paths)
    max_path_len = max(len(p) for p in differentiation_paths)

    fig_width = min(max_path_len * 2.5, 20)
    fig_height = max(n_paths * 2.5, 4)

    fig, axes = plt.subplots(n_paths, max_path_len, figsize=(fig_width, fig_height))

    # Handle axis array shape
    if n_paths == 1 and max_path_len == 1:
        axes = np.array([[axes]])
    elif n_paths == 1:
        axes = axes.reshape(1, -1)
    elif max_path_len == 1:
        axes = axes.reshape(-1, 1)

    n_bins = ROSE_PLOT_N_BINS
    bin_width = ROSE_PLOT_MAX_ANGLE / n_bins
    bins = np.linspace(0, ROSE_PLOT_MAX_ANGLE, n_bins + 1)
    bin_centers = bins[:-1] + bin_width / 2

    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=ROSE_PLOT_MAX_ANGLE)

    color_idx = 0
    for path_idx, path in enumerate(differentiation_paths):
        for cell_idx, cell_type in enumerate(path):
            ax = axes[path_idx, cell_idx]
            ax.set_aspect("equal")

            # Get cells of this type
            cell_mask = adata.obs[cluster_key].astype(str) == cell_type

            # Calculate angles
            angles = []
            for j in np.where(cell_mask)[0]:
                ref = reference_directions[j]
                pred = predicted_velocities[j]

                if np.all(ref == 0) or np.all(pred == 0):
                    continue

                dot = np.dot(ref, pred)
                norm_prod = np.linalg.norm(ref) * np.linalg.norm(pred)
                cos_angle = np.clip(dot / (norm_prod + 1e-10), -1, 1)
                angles.append(np.degrees(np.arccos(cos_angle)))

            # Compute histogram
            hist, _ = np.histogram(angles, bins=bins)
            max_count = max(hist) if hist.max() > 0 else 1

            cell_color = PALETTE_34[color_idx % len(PALETTE_34)]
            color_idx += 1

            # Draw semicircle outline
            radius = 1.0
            theta = np.linspace(0, np.pi, 100)
            ax.plot(radius * np.cos(theta), radius * np.sin(theta),
                   color=cell_color, linewidth=0.8)
            ax.plot([-radius, radius], [0, 0], color=cell_color, linewidth=0.8)

            # Draw bin dividers
            for angle in bins[1:-1]:
                rad = np.radians(angle)
                ax.plot([0, radius * np.cos(rad)], [0, radius * np.sin(rad)],
                       color="gray", linestyle="--", linewidth=0.15, alpha=0.3)

            # Draw wedges
            for j, count in enumerate(hist):
                if count == 0:
                    continue

                height = count / max_count * radius
                wedge = Wedge(
                    (0, 0), height, bins[j], bins[j+1],
                    facecolor=cmap(norm(bin_centers[j])),
                    edgecolor=cell_color, alpha=0.7, linewidth=0.3
                )
                ax.add_artist(wedge)

            ax.set_xlim(-radius * 1.2, radius * 1.1)
            ax.set_ylim(-0.1, radius * 1.1)
            ax.axis("off")

            # Cell type label
            label = format_cell_type_label(cell_type, dataset)
            ax.text(0, radius * 1.25, label, ha="center", va="bottom",
                   fontsize=11, fontweight="bold", color=cell_color)

        # Hide unused axes
        for empty_idx in range(len(path), max_path_len):
            axes[path_idx, empty_idx].axis("off")

    # Title
    title = f"{tool} Velocity-Reference Direction Alignment"
    if n_paths > 1:
        title += f" ({n_paths} Independent Trajectories)"
    fig.suptitle(title, x=0.5, y=0.97, fontsize=14, fontweight="bold")

    # Colorbar
    cax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdYlGn),
        cax=cax, orientation="horizontal"
    )
    cb.set_ticks([0, 90, 180])
    cb.set_ticklabels(["180°", "90°", "0°"])

    plt.subplots_adjust(bottom=0.12, left=0.05, right=0.95, top=0.9,
                       wspace=0.05, hspace=0.3)

    # Save
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(pdf_path, format="pdf", dpi=400, bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=400, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_single_dataset(
    adata: sc.AnnData,
    output_dir: str,
    tool: str,
    dataset: str,
    velocity_key: str = "velocity",
    n_jobs: int = 1,
    cluster_key: Optional[str] = None,
    differentiation_paths: Optional[Union[List[str], List[List[str]]]] = None,
    umap_dir: Optional[str] = None,
    h5ad_path: Optional[str] = None,
    benchmark_csv: Optional[str] = None,
    benchmark_total_csv: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze velocity consistency for a single dataset.

    Parameters
    ----------
    adata : AnnData
        Single-cell data with velocity layer
    output_dir : str
        Base output directory
    tool : str
        Name of velocity method (e.g., 'scVelo', 'VeloVAE')
    dataset : str
        Dataset identifier (e.g., '28', '22_CTCL')
    velocity_key : str
        Key for velocity layer (default: 'velocity')
    cluster_key : str, optional
        Column name for cell type annotation. Auto-detected if None
    differentiation_paths : list, optional
        Custom differentiation paths. Uses built-in defaults if None
    umap_dir : str, optional
        Directory containing external UMAP CSV files
    h5ad_path : str, optional
        Path to h5ad file (used for UMAP file matching)
    benchmark_csv : str, optional
        Path for detailed benchmark CSV output
    benchmark_total_csv : str, optional
        Path for summary benchmark CSV output

    Returns
    -------
    dict
        Analysis results including status, paths, and statistics
    """
    num_prefix = extract_numeric_prefix(dataset)

    # Create output directories
    method_dir = Path(output_dir) / tool
    pdf_dir = method_dir / "pdf"
    png_dir = method_dir / "png"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Preprocess based on method
        if tool == "TFvelo":
            adata = preprocess_tfvelo(adata)
            if adata is None:
                return {"status": "error", "message": "TFvelo preprocessing failed"}

        if tool == "PhyloVelo":
            adata = preprocess_phylovelo(adata, velocity_key)

        # Ensure UMAP exists
        adata = ensure_umap(adata, dataset, tool,
                           h5ad_path or "", umap_dir)
        if adata is None:
            return {"status": "error", "message": "UMAP computation failed"}

        # Remove duplicates
        adata.obs_names_make_unique()
        if hasattr(adata, 'to_df'):
            try:
                adata = adata[~adata.to_df().duplicated(), :]
            except Exception:
                pass

        # Ensure velocity embedding
        velocity_dimred_key = f"{velocity_key}_umap"
        if velocity_dimred_key not in adata.obsm:
            success, adata = ensure_velocity_embedding(
                adata, dataset, tool, velocity_key, n_jobs=n_jobs
            )
            if not success:
                return {"status": "error",
                       "message": "Velocity embedding computation failed"}

        # Get cluster key
        resolved_cluster_key = get_cluster_key(adata, dataset, cluster_key)
        if resolved_cluster_key is None:
            return {"status": "error", "message": "No valid cluster key found"}

        # Process time column if needed
        process_time_column(adata, resolved_cluster_key, dataset)

        # Get available clusters
        available_clusters = adata.obs[resolved_cluster_key].unique().astype(str).tolist()

        # Get differentiation paths
        if differentiation_paths is None:
            paths = DEFAULT_DIFFERENTIATION_PATHS.get(num_prefix) or \
                   DEFAULT_DIFFERENTIATION_PATHS.get(dataset)
        else:
            paths = differentiation_paths

        # For time/phase data without predefined paths
        is_time_or_phase = resolved_cluster_key.lower() in ["time", "phase"] or \
                          set(["G1", "S", "G2M"]).issubset(set(available_clusters))

        if paths is None:
            if is_time_or_phase:
                if resolved_cluster_key.lower() == "phase" or \
                   set(["G1", "S", "G2M"]).issubset(set(available_clusters)):
                    paths = [c for c in ["G1", "S", "G2M"] if c in available_clusters]
                else:
                    paths = sort_time_clusters(available_clusters)
            else:
                return {"status": "skipped",
                       "message": f"No differentiation paths for dataset {dataset}"}

        # Validate paths
        try:
            validated_paths = validate_differentiation_paths(
                paths, available_clusters, dataset, resolved_cluster_key
            )
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        if not validated_paths:
            return {"status": "skipped",
                   "message": "No valid cells in differentiation paths"}

        # Get all valid clusters
        valid_clusters = [c for path in validated_paths for c in path]
        valid_mask = adata.obs[resolved_cluster_key].astype(str).isin(valid_clusters)

        # Calculate reference directions
        reference_dirs = calculate_reference_directions(
            adata, resolved_cluster_key, validated_paths
        )

        # Get predicted velocities
        predicted_vels = adata.obsm[velocity_dimred_key]

        # Calculate angles
        angles = np.full(adata.n_obs, np.nan)
        cosine_sim = np.full(adata.n_obs, np.nan)

        for i in np.where(valid_mask)[0]:
            ref = reference_dirs[i]
            pred = predicted_vels[i]

            if np.all(ref == 0) or np.all(pred == 0):
                continue

            norm_prod = np.linalg.norm(ref) * np.linalg.norm(pred)
            if norm_prod < 1e-6:
                continue

            cos_val = np.clip(np.dot(ref, pred) / norm_prod, -1, 1)
            cosine_sim[i] = cos_val
            angles[i] = np.degrees(np.arccos(cos_val))

        # Calculate statistics
        angle_stats = calculate_angle_statistics(
            angles, adata, resolved_cluster_key, valid_clusters, valid_mask
        )
        total_ratio = calculate_total_angle_ratio(angles, valid_mask)

        # Write unified benchmark CSVs (optional).
        # Note: angle bins are not mutually exclusive (e.g. 60-90 overlaps with 60-80/80-90).
        if benchmark_csv:
            df_long = _build_benchmark_long_rows(
                data_type="real",
                tool=tool,
                dataset=dataset,
                group_type="celltype",
                group_to_stats=angle_stats,
            )
            write_benchmark_long(benchmark_csv, df_long)
        if benchmark_total_csv:
            update_benchmark_total_wide(
                benchmark_total_csv,
                tool,
                dataset,
                float(total_ratio) if total_ratio == total_ratio else np.nan,
                rank_1_is_worst=True,
            )

        # Generate rose plot
        pdf_path = str(pdf_dir / f"{tool}_{dataset}_anglerose.pdf")
        png_path = str(png_dir / f"{tool}_{dataset}_anglerose.png")

        plot_rose_diagram(
            adata, reference_dirs, predicted_vels,
            resolved_cluster_key, pdf_path, png_path,
            tool, validated_paths, dataset
        )

        # Compute per-cluster statistics
        cluster_consistency = {}
        cluster_mean_angle = {}
        for cluster in valid_clusters:
            mask = (adata.obs[resolved_cluster_key].astype(str) == cluster) & valid_mask
            if mask.sum() > 0:
                cluster_consistency[cluster] = float(np.nanmean(cosine_sim[mask]))
                cluster_mean_angle[cluster] = float(np.nanmean(angles[mask]))

        return {
            "status": "success",
            "pdf_path": pdf_path,
            "png_path": png_path,
            "cluster_key": resolved_cluster_key,
            "n_paths": len(validated_paths),
            "cluster_consistency": cluster_consistency,
            "cluster_mean_angle": cluster_mean_angle,
            "total_ratio_0_60": total_ratio,
            "angle_distribution": angle_stats,
        }

    except Exception as e:
        logging.error(f"Analysis failed for {tool}_{dataset}: {e}")
        return {"status": "error", "message": str(e)}


# === BEGIN INLINED SIM_ROSEPLOT (from sim_roseplot.py) ===
# pylint: disable=C0301,C0302,C0303
"""
Angle consistency between predicted RNA velocity vectors and a reference differentiation direction
for simulated datasets.

Method overview:
1) Define differentiation paths using milestones (cell states) and topology type.
2) Fit reference trajectory curve(s) in a low-dimensional embedding (X_dimred / X_umap)
   using spline interpolation.
3) For each cell, compute the tangent direction at its closest point on the curve
   as the reference direction.
4) Compare predicted velocity (in the same embedding) against the reference direction
   using cosine similarity / angles and visualize via rose plots.

Supported topology types:
- Standard: bifurcating, linear-simple, trifurcating, cycle-simple, disconnected, ...
- Subsampled: genesub-bifurcating, cellsub-bifurcating (share the same milestone configuration as bifurcating)
- Special shapes: linear-bifurcating, linear-linear

Important special handling (metric-specific; keep for reproducibility):
- linear-simple-subset is treated as linear-simple for plotting and output naming.
"""

BasisType = Literal["dimred", "umap"]


AngleDistribution = TypedDict(
    "AngleDistribution",
    {
        "0-30": float,
        "30-60": float,
        "60-80": float,
        "80-90": float,
        "60-90": float,
        "90-100": float,
        "100-120": float,
        "90-120": float,
        "120-150": float,
        "150-180": float,
        "valid_ratio": float,
        "cv_coefficient": float,
    },
    total=False,
)
# NOTE: The angle-bin keys are coupled with calculate_angle_distribution_stats_simulation
# and the CSV writing schema. If you change bins, update this TypedDict and ANGLE_COLUMNS
# together to keep the schema consistent.


class AnalysisResult(TypedDict, total=False):
    """Return schema for the main analysis entrypoint."""
    status: str
    dataset_name: str
    normalized_filename: str
    topology_type: str
    embedding_basis: BasisType
    rose_pdf_path: str
    rose_png_path: str
    milestone_consistency: Dict[str, float]
    milestone_mean_angle: Dict[str, float]
    n_valid_milestones: int
    valid_milestones: List[str]
    angle_distribution_stats: Dict[str, AngleDistribution]
    reference_curve_png_path: str


def normalize_dataset_name(dataset_name: str) -> str:
    """
    Normalize dataset name by converting topology keywords from underscore form to hyphen form.

    Example:
        bifurcating_loop_cell1000_gene2000 -> bifurcating-loop_cell1000_gene2000
        14_consecutive_bifurcating_cell5000_gene1000 -> 14_consecutive-bifurcating_cell5000_gene1000
    """
    topology_replacements = {
        'bifurcating_loop': 'bifurcating-loop',
        'consecutive_bifurcating': 'consecutive-bifurcating',
        'cycle_simple': 'cycle-simple',
        'linear_simple': 'linear-simple',
        'linear_simple_subset': 'linear-simple',  # normalize subset to linear-simple for naming
        'genesub_bifurcating': 'genesub-bifurcating',
        'cellsub_bifurcating': 'cellsub-bifurcating',
        'linear_bifurcating': 'linear-bifurcating',
        'linear_linear': 'linear-linear'
    }

    normalized_name = dataset_name
    lower_name = dataset_name.lower()
    sorted_replacements = sorted(topology_replacements.items(), key=lambda x: len(x[0]), reverse=True)

    for old_format, new_format in sorted_replacements:
        if old_format in lower_name:
            start_idx = lower_name.find(old_format)
            if start_idx != -1:
                normalized_name = (
                    dataset_name[:start_idx] +
                    new_format +
                    dataset_name[start_idx + len(old_format):]
                )
                break

    return normalized_name


def normalize_filename(tool: str, dataset: str) -> str:
    """Normalize output filename using the normalized dataset name."""
    normalized_dataset = normalize_dataset_name(dataset)
    return f"{tool}_{normalized_dataset}"


def normalize_topology_type(topology_type: str) -> str:
    """Normalize topology type name (underscore -> hyphen)."""
    standard_topologies = {
        "bifurcating": "bifurcating",
        "bifurcating_loop": "bifurcating-loop",
        "bifurcating-loop": "bifurcating-loop",
        "consecutive_bifurcating": "consecutive-bifurcating",
        "consecutive-bifurcating": "consecutive-bifurcating",
        "cycle_simple": "cycle-simple",
        "cycle-simple": "cycle-simple",
        "disconnected": "disconnected",
        "linear_simple": "linear-simple",
        "linear-simple": "linear-simple",
        "linear_simple_subset": "linear-simple-subset",
        "linear-simple-subset": "linear-simple-subset",
        "trifurcating": "trifurcating",
        # Additional / special topology aliases
        "genesub_bifurcating": "genesub-bifurcating",
        "genesub-bifurcating": "genesub-bifurcating",
        "cellsub_bifurcating": "cellsub-bifurcating", 
        "cellsub-bifurcating": "cellsub-bifurcating",
        "linear_bifurcating": "linear-bifurcating",
        "linear-bifurcating": "linear-bifurcating",
        "linear_linear": "linear-linear",
        "linear-linear": "linear-linear"
    }
    
    normalized = standard_topologies.get(topology_type.lower())
    if normalized is None:
        raise ValueError(f"Unknown topology type: {topology_type}")
    
    return normalized


def determine_embedding_basis(adata: AnnData, velocity_key: str) -> BasisType:
    """
    Decide which embedding basis to use.

    Priority:
    1) Use an existing velocity embedding (e.g. {vkey}_dimred / {vkey}_umap) if available.
    2) Otherwise fall back to X_dimred, then X_umap.
    """
    if f"{velocity_key}_dimred" in adata.obsm and 'X_dimred' in adata.obsm:
        return 'dimred'

    if f"{velocity_key}_umap" in adata.obsm and 'X_umap' in adata.obsm:
        return 'umap'

    if 'X_dimred' in adata.obsm:
        return 'dimred'
    if 'X_umap' in adata.obsm:
        return 'umap'
    raise ValueError("Neither 'X_umap' nor 'X_dimred' is available in adata.obsm.")


def ensure_velocity_embedding_sim(adata: AnnData, velocity_key: str, basis: BasisType, n_jobs: int = 1) -> None:
    """Ensure {velocity_key}_{basis} exists in adata.obsm; compute it if missing."""
    velocity_embedding_key = f"{velocity_key}_{basis}"

    if velocity_embedding_key not in adata.obsm:
        xkey_to_use = None
        if 'Ms' in adata.layers:
            xkey_to_use = 'Ms'
        elif 'spliced' in adata.layers:
            xkey_to_use = 'spliced'
        elif 'M_total' in adata.layers:
            xkey_to_use = 'M_total'

        try:
            if xkey_to_use is not None:
                scv.tl.velocity_graph(adata, vkey=velocity_key, xkey=xkey_to_use, n_jobs=int(n_jobs))
            else:
                scv.tl.velocity_graph(adata, vkey=velocity_key, n_jobs=int(n_jobs))
        except ValueError as e:
            error_msg = str(e)
            if "neighbor graph seems to be corrupted" in error_msg or "Consider recomputing via pp.neighbors" in error_msg:
                adata.obs_names_make_unique()
                duplicated_mask = adata.to_df().duplicated()
                if duplicated_mask.any():
                    adata._inplace_subset_obs(~duplicated_mask)

                sc.pp.neighbors(adata)

                if xkey_to_use is not None:
                    scv.tl.velocity_graph(adata, vkey=velocity_key, xkey=xkey_to_use, n_jobs=int(n_jobs))
                else:
                    scv.tl.velocity_graph(adata, vkey=velocity_key, n_jobs=int(n_jobs))
            else:
                raise ValueError(f"Failed to compute velocity graph: {str(e)}") from e

        scv.tl.velocity_embedding(adata, basis=basis, vkey=velocity_key)


def calculate_angle_distribution_stats_simulation(
    angles: np.ndarray,
    adata: AnnData,
    milestone_key: str,
    valid_milestones: List[str],
    valid_cells_mask: Union[np.ndarray, pd.Series]
) -> Dict[str, Dict[str, float]]:
    """Compute per-milestone angle distribution statistics."""
    if isinstance(valid_cells_mask, pd.Series):
        valid_cells_mask = valid_cells_mask.values

    angle_stats: Dict[str, Dict[str, float]] = {}

    angle_bins = [
        (0, 30), (30, 60), (60, 80), (80, 90), (60, 90),
        (90, 100), (100, 120), (90, 120),
        (120, 150), (150, 180)
    ]
    bin_labels = [
        "0-30", "30-60", "60-80", "80-90", "60-90",
        "90-100", "100-120", "90-120",
        "120-150", "150-180"
    ]
    
    for milestone in valid_milestones:
        milestone_mask = (adata.obs[milestone_key].astype(str) == milestone) & valid_cells_mask

        if np.sum(milestone_mask) == 0:
            stats_dict = {label: np.nan for label in bin_labels}
            stats_dict["valid_ratio"] = np.nan
            angle_stats[milestone] = stats_dict
            continue

        milestone_angles = angles[milestone_mask]
        total_cells_original = len(milestone_angles)
        milestone_angles_valid = milestone_angles[~np.isnan(milestone_angles)]
        valid_cells_count = len(milestone_angles_valid)

        if total_cells_original == 0:
            stats_dict = {label: np.nan for label in bin_labels}
            stats_dict["valid_ratio"] = np.nan
            angle_stats[milestone] = stats_dict
            continue

        if valid_cells_count == 0:
            stats_dict = {label: 0.0 for label in bin_labels}
            stats_dict["valid_ratio"] = 0.0
            angle_stats[milestone] = stats_dict
            continue

        bin_counts = {}
        for i, (min_angle, max_angle) in enumerate(angle_bins):
            count = np.sum((milestone_angles_valid >= min_angle) & (milestone_angles_valid <= max_angle))
            bin_counts[bin_labels[i]] = count / total_cells_original * 100

        valid_ratio = valid_cells_count / total_cells_original * 100
        bin_counts["valid_ratio"] = valid_ratio

        if valid_cells_count > 1:
            mean_angle = float(np.mean(milestone_angles_valid))
            std_angle = float(np.std(milestone_angles_valid, ddof=1))
            if mean_angle != 0:
                cv_coefficient = (std_angle / mean_angle) * 100
            else:
                cv_coefficient = np.nan
        else:
            cv_coefficient = np.nan

        bin_counts["cv_coefficient"] = cv_coefficient

        angle_stats[milestone] = bin_counts

    return angle_stats


def calculate_total_0_60_ratio(
    angles: np.ndarray,
    valid_cells_mask: Union[np.ndarray, pd.Series]
) -> float:
    """Compute the percentage of cells whose angle falls within [0, 60] degrees."""
    if isinstance(valid_cells_mask, pd.Series):
        valid_cells_mask = valid_cells_mask.values

    valid_angles = angles[valid_cells_mask]
    valid_angles_no_nan = valid_angles[~np.isnan(valid_angles)]

    if len(valid_angles_no_nan) == 0:
        return np.nan

    count_0_60 = np.sum((valid_angles_no_nan >= 0) & (valid_angles_no_nan <= 60))
    total_cells = len(valid_angles_no_nan)
    ratio_0_60 = (count_0_60 / total_cells) * 100

    return ratio_0_60


def get_display_topology_type(topology_type: str) -> str:
    """Topology name used for display/output naming (linear-simple-subset -> linear-simple)."""
    normalized = normalize_topology_type(topology_type)

    if normalized == 'linear-simple-subset':
        return 'linear-simple'

    return normalized


def get_expected_milestones_for_topology(topology_type: str, dataset_name: str = "") -> List[str]:
    """
    Return the expected milestone list for a given topology.

    Note: consecutive-bifurcating has multiple variants and is disambiguated by dataset_name patterns.
    """
    normalized_topology = normalize_topology_type(topology_type)

    if normalized_topology == "consecutive-bifurcating":
        normalized_name = normalize_dataset_name(dataset_name)

        variant2_patterns = [
            "consecutive-bifurcating_cell500_gene500",
            "consecutive-bifurcating_cell50000_gene1000",
            "consecutive-bifurcating_cell100000_gene1000",
            "10_consecutive-bifurcating_cell1000_gene10000"
        ]
        variant3_patterns = [
            "consecutive-bifurcating_cell1000_gene500",
            "consecutive-bifurcating_cell1000_gene50000"
        ]

        if any(pattern in normalized_name for pattern in variant2_patterns + variant3_patterns):
            return ["sA", "sB", "sC", "sCmid", "sD", "sDmid", "sE", "sEndE", "sF", "sEndF", "sG", "sEndG"]

        return ["sA", "sB", "sC", "sCmid", "sD", "sEndD", "sE", "sEmid", "sF", "sEndF", "sG", "sEndG"]

    milestone_sets = {
        "bifurcating": ["sA", "sB", "sBmid", "sC", "sEndC", "sD", "sEndD"],
        "genesub-bifurcating": ["sA", "sB", "sBmid", "sC", "sEndC", "sD", "sEndD"],
        "cellsub-bifurcating": ["sA", "sB", "sBmid", "sC", "sEndC", "sD", "sEndD"],
        "bifurcating-loop": ["sA", "sB", "sC", "sD"],
        "cycle-simple": ["s1", "s2", "s3"],
        "disconnected": ["left sA", "left sB", "left sC", "left sEndC", "right s1", "right s2", "right s3"],
        "linear-simple": ["s1", "s2"],
        "linear-simple-subset": ["s1", "s2"],
        "trifurcating": ["sA", "sB", "sC", "sCmid", "sD", "sEndD", "sE", "sEndE", "sF", "sEndF"],
        "linear-bifurcating": ["left sA", "left sB", "left sC", "left sEndC", "right sA", "right sB", "right sBmid", "right sC", "right sEndC", "right sD", "right sEndD"],
        "linear-linear": ["left sA", "left sB", "left sC", "left sEndC", "right sA", "right sB", "right sC", "right sEndC"]
    }

    if normalized_topology not in milestone_sets:
        raise ValueError(f"Unknown topology type: {normalized_topology}")

    return milestone_sets[normalized_topology]


def infer_topology_from_dataset_name(
    dataset_name: str,
    adata: AnnData,
    milestone_key: str
) -> str:
    """
    Infer topology type from dataset_name and (optionally) milestone patterns.

    Expected naming patterns:
    - Standard: {topology}_cellN_geneM
    - Added datasets: <number>_{topology}_cellN_geneM
    - Subsampled: cellsub_* / genesub_* (treated as bifurcating)
    """
    normalized_name = dataset_name.lower()

    if normalized_name.startswith('cellsub') or normalized_name.startswith('genesub'):
        return 'bifurcating'

    topology_keywords = [
        'consecutive-bifurcating', 'consecutive_bifurcating',
        'bifurcating-loop', 'bifurcating_loop',
        'linear-simple-subset', 'linear_simple_subset',
        'linear-bifurcating', 'linear_bifurcating',
        'cycle-simple', 'cycle_simple',
        'linear-simple', 'linear_simple',
        'linear-linear', 'linear_linear',
        'bifurcating', 'trifurcating', 'disconnected',
    ]

    topology_mapping = {
        'consecutive-bifurcating': 'consecutive-bifurcating',
        'consecutive_bifurcating': 'consecutive-bifurcating',
        'bifurcating-loop': 'bifurcating-loop',
        'bifurcating_loop': 'bifurcating-loop',
        'linear-simple-subset': 'linear-simple',
        'linear_simple_subset': 'linear-simple',
        'linear-bifurcating': 'linear-bifurcating',
        'linear_bifurcating': 'linear-bifurcating',
        'cycle-simple': 'cycle-simple',
        'cycle_simple': 'cycle-simple',
        'linear-simple': 'linear-simple',
        'linear_simple': 'linear-simple',
        'linear-linear': 'linear-linear',
        'linear_linear': 'linear-linear',
        'bifurcating': 'bifurcating',
        'trifurcating': 'trifurcating',
        'disconnected': 'disconnected',
    }

    for keyword in topology_keywords:
        if keyword in normalized_name:
            inferred_topology = topology_mapping[keyword]
            return inferred_topology

    if milestone_key in adata.obs.columns:
        available_milestones = set(adata.obs[milestone_key].astype(str).unique())
        has_left = any('left' in m.lower() for m in available_milestones)
        has_right = any('right' in m.lower() for m in available_milestones)

        if has_left and has_right:
            has_bifurcation = any('bmid' in m.lower() or 'cmid' in m.lower() for m in available_milestones)
            has_cycle = 's1' in available_milestones or 's2' in available_milestones or 's3' in available_milestones

            if has_bifurcation and has_cycle:
                return 'disconnected'
            if has_bifurcation:
                return 'linear-bifurcating'
            return 'linear-linear'

        if available_milestones == {'s1', 's2', 's3'}:
            return 'cycle-simple'

        if available_milestones == {'s1', 's2'}:
            return 'linear-simple'

        bifurcating_milestones = {'sA', 'sB', 'sBmid', 'sC', 'sEndC', 'sD', 'sEndD'}
        if available_milestones == bifurcating_milestones:
            return 'bifurcating'

    raise ValueError(
        f"Failed to infer topology type from dataset_name '{dataset_name}'.\n"
        f"Please specify --topology-type explicitly, or ensure dataset_name contains a topology keyword.\n"
        f"Supported: bifurcating, linear-simple, cycle-simple, trifurcating, "
        f"consecutive-bifurcating, bifurcating-loop, disconnected, linear-bifurcating, linear-linear.\n"
        f"Subsampled datasets: names starting with cellsub_* or genesub_* (treated as bifurcating)."
    )


def detect_disconnected_variant(dataset_name: str) -> str:
    """
    Detect disconnected variant by dataset_name.

    - standard: left is a curved polyline; right is a closed loop (most datasets)
    - chaotic: right cluster is irregular; fit an ellipse via boundary points (a few datasets)
    """
    chaotic_datasets = [
        'disconnected_cell10000_gene10000',
        'disconnected_cell50000_gene1000',
        'disconnected_cell100000_gene1000'
    ]

    normalized_name = dataset_name.lower().replace('-', '_')

    for pattern in chaotic_datasets:
        if pattern in normalized_name:
            return 'chaotic'

    return 'standard'


def get_trajectory_segments_for_topology(
    topology_type: str,
    adata: AnnData,
    milestone_key: str,
    dataset_name: str
) -> List[Tuple[List[str], bool]]:
    """
    Return a list of non-duplicated trajectory segments for the given topology.

    Each segment is drawn once to form the correct branching/tree structure and
    avoid re-drawing shared trunk segments.

    Returns:
        List[(milestone_sequence, is_closed_loop)]
    """
    normalized_topology = normalize_topology_type(topology_type)

    if normalized_topology in ("linear-simple", "linear-simple-subset"):
        return [(["s1", "s2"], False)]

    if normalized_topology == "cycle-simple":
        return [(["s1", "s2", "s3"], True)]

    if normalized_topology in ("bifurcating", "genesub-bifurcating", "cellsub-bifurcating"):
        return [
            (["sA", "sB", "sBmid"], False),
            (["sBmid", "sC", "sEndC"], False),
            (["sBmid", "sD", "sEndD"], False)
        ]

    if normalized_topology == "bifurcating-loop":
        # T-shape + D-like arc: trunk (sA→sB), short branch (sB→sD),
        # long arc branch (sB→sC, ~210 degrees).
        return [
            (["sA", "sB"], False),
            (["sB", "sD"], False),
            (["sB", "sC"], False)
        ]

    if normalized_topology == "consecutive-bifurcating":
        normalized_name = normalize_dataset_name(dataset_name)

        variant2_patterns = [
            "consecutive-bifurcating_cell500_gene500",
            "consecutive-bifurcating_cell50000_gene1000",
            "consecutive-bifurcating_cell100000_gene1000",
            "10_consecutive-bifurcating_cell1000_gene10000"
        ]
        if any(pattern in normalized_name for pattern in variant2_patterns):
            return [
                (["sA", "sB", "sC", "sCmid"], False),
                (["sCmid", "sE", "sEndE"], False),
                (["sCmid", "sD", "sDmid"], False),
                (["sDmid", "sF", "sEndF"], False),
                (["sDmid", "sG", "sEndG"], False)
            ]

        variant3_patterns = [
            "consecutive-bifurcating_cell1000_gene500",
            "consecutive-bifurcating_cell1000_gene50000"
        ]
        if any(pattern in normalized_name for pattern in variant3_patterns):
            return [
                (["sA", "sB", "sC", "sCmid"], False),
                (["sCmid", "sG", "sEndG"], False),
                (["sCmid", "sD", "sDmid"], False),
                (["sDmid", "sF", "sEndF"], False),
                (["sDmid", "sE", "sEndE"], False)
            ]

        available_milestones = set(adata.obs[milestone_key].unique())

        if "sEndD" in available_milestones and "sEmid" in available_milestones:
            return [
                (["sA", "sB", "sC", "sCmid"], False),
                (["sCmid", "sD", "sEndD"], False),
                (["sCmid", "sE", "sEmid"], False),
                (["sEmid", "sF", "sEndF"], False),
                (["sEmid", "sG", "sEndG"], False)
            ]

        if "sDmid" in available_milestones and "sEndE" in available_milestones:
            raise ValueError(
                "Detected consecutive-bifurcating variant 2/3 by milestone patterns, but cannot\n"
                "disambiguate by dataset_name. Variant 2 and 3 share the same milestone set and\n"
                "must be matched by dataset_name patterns.\n"
                f"dataset_name: {dataset_name}"
            )

        raise ValueError(
            "Unrecognized consecutive-bifurcating variant.\n"
            f"dataset_name: {dataset_name}\n"
            f"available milestones: {sorted(available_milestones)}"
        )

    if normalized_topology == "trifurcating":
        return [
            (["sA", "sB", "sC", "sCmid"], False),
            (["sCmid", "sD", "sEndD"], False),
            (["sCmid", "sE", "sEndE"], False),
            (["sCmid", "sF", "sEndF"], False)
        ]

    if normalized_topology == "disconnected":
        variant = detect_disconnected_variant(dataset_name)

        if variant == 'chaotic':
            return [
                (["left sA", "left sB", "left sC", "left sEndC"], False),
                (["right s1", "right s2", "right s3"], True)
            ]
        else:
            return [
                (["left sA", "left sB", "left sC", "left sEndC"], False),
                (["right s1", "right s2", "right s3"], True)
            ]

    if normalized_topology == "linear-bifurcating":
        return [
            (["left sA", "left sB", "left sC", "left sEndC"], False),
            (["right sA", "right sB", "right sBmid"], False),
            (["right sBmid", "right sC", "right sEndC"], False),
            (["right sBmid", "right sD", "right sEndD"], False)
        ]

    if normalized_topology == "linear-linear":
        return [
            (["left sA", "left sB", "left sC", "left sEndC"], False),
            (["right sA", "right sB", "right sC", "right sEndC"], False)
        ]

    raise ValueError(f"Unknown topology type: {normalized_topology}")


def normalize_milestone_names(milestone_name: str) -> str:
    """Normalize milestone names by replacing '_' / '-' with spaces (e.g. 'left_sA' -> 'left sA')."""
    normalized = milestone_name.replace('_', ' ').replace('-', ' ')
    normalized = ' '.join(normalized.split())
    return normalized


def standardize_milestone_column(adata: AnnData, milestone_key: str) -> None:
    """Standardize the milestone column by normalizing all milestone names."""
    current_milestones = adata.obs[milestone_key].astype(str)
    normalized_milestones = current_milestones.apply(normalize_milestone_names)
    adata.obs[milestone_key] = normalized_milestones


def validate_milestones_for_simulation(
    expected_milestones: List[str],
    available_milestones: np.ndarray,
    dataset_name: str
) -> List[str]:
    """Validate milestones for simulated datasets; required milestones must be present."""
    available_milestones_set = set(available_milestones.astype(str))
    expected_milestones_set = set(expected_milestones)

    missing_milestones = expected_milestones_set - available_milestones_set

    if missing_milestones:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required milestones: {sorted(missing_milestones)}"
        )

    extra_milestones = available_milestones_set - expected_milestones_set
    if extra_milestones:
        pass

    return expected_milestones


def ensure_milestone_order(adata: AnnData, milestone_key: str, expected_milestones: List[str]) -> None:
    """Ensure milestone column is categorical and ordered by expected_milestones."""
    current_milestones = adata.obs[milestone_key].astype(str)

    available_milestones = set(current_milestones.unique())
    ordered_categories = [ms for ms in expected_milestones if ms in available_milestones]

    extra_milestones = sorted(available_milestones - set(expected_milestones))
    if extra_milestones:
        ordered_categories.extend(extra_milestones)

    adata.obs[milestone_key] = pd.Categorical(
        current_milestones,
        categories=ordered_categories,
        ordered=True
    )


def get_spline_degree_for_segment(
    topology_type: str,
    segment_idx: int,
    num_milestones: int,
    is_cycle: bool = False
) -> int:
    """
    Choose spline degree (k) based on topology/segment type.

    Heuristic:
    - Closed loops: k=3 for smoothness
    - Curved branches/arcs: k=3 to capture curvature
    - Lines/branches: k=2 to allow mild curvature while remaining stable
    """
    normalized_topology = normalize_topology_type(topology_type)

    if is_cycle:
        return 3

    if normalized_topology in ('bifurcating', 'genesub-bifurcating', 'cellsub-bifurcating'):
        return 2

    if normalized_topology == 'trifurcating':
        return 2

    if normalized_topology == 'consecutive-bifurcating':
        return 2

    if normalized_topology == 'linear-bifurcating':
        return 2

    if normalized_topology == 'linear-simple':
        return 3

    if normalized_topology == 'cycle-simple':
        return 3

    if normalized_topology == 'bifurcating-loop':
        if segment_idx == 2:
            return 3
        return 2

    if normalized_topology == 'disconnected':
        return 2

    if normalized_topology == 'linear-linear':
        return 2

    return 2


def filter_outliers_by_distance(
    embedding: np.ndarray,
    mask: Union[np.ndarray, pd.Series],
    mad_threshold: float = 3.0,
    min_cells: int = 10,
    milestone_name: Optional[str] = None
) -> Union[np.ndarray, pd.Series]:
    """
    Filter outlier cells by distance to the cluster center using MAD.

    Keep cells within: median_distance + mad_threshold * MAD.
    """
    cell_coords = embedding[mask]

    if len(cell_coords) <= min_cells:
        return mask

    adjusted_threshold = mad_threshold
    if milestone_name is not None and 'right' in milestone_name.lower():
        adjusted_threshold = 2.0

    center = np.median(cell_coords, axis=0)
    distances = np.sqrt(np.sum((cell_coords - center)**2, axis=1))

    median_distance = np.median(distances)
    mad = np.median(np.abs(distances - median_distance))

    if mad < 1e-10:
        return mask

    threshold = median_distance + adjusted_threshold * mad

    inlier_indices = np.where(distances <= threshold)[0]

    if len(inlier_indices) < min_cells:
        inlier_indices = np.argsort(distances)[:min_cells]

    new_mask = mask.copy()
    original_indices = np.where(mask)[0]
    outlier_indices = np.setdiff1d(np.arange(len(cell_coords)), inlier_indices)
    new_mask[original_indices[outlier_indices]] = False

    return new_mask


def calculate_extended_linear_segment_points(
    embedding: np.ndarray,
    adata: AnnData,
    milestone_key: str,
    segment_milestones: List[str],
    start_milestone: str
) -> np.ndarray:
    """
    Compute extended guide points for an elongated linear segment.

    This is used for long, bar-shaped milestones (e.g. 'left sA' in disconnected):
    - pick a start point from cells farthest from the next milestone
    - add intermediate guide points by distance quantiles so the fitted curve
      covers the full extent of the cluster
    """
    milestones = adata.obs[milestone_key].astype(str)

    milestone_cells = {}
    milestone_centers = {}
    for ms in segment_milestones:
        mask = milestones == ms
        if np.sum(mask) > 0:
            filtered_mask = filter_outliers_by_distance(embedding, mask, milestone_name=ms)
            cells = embedding[filtered_mask]
            milestone_cells[ms] = cells
            milestone_centers[ms] = np.median(cells, axis=0)

    if start_milestone not in milestone_cells or len(milestone_cells) < 2:
        return np.array([milestone_centers[ms] for ms in segment_milestones if ms in milestone_centers])

    start_idx = segment_milestones.index(start_milestone)
    if start_idx < len(segment_milestones) - 1:
        next_milestone = segment_milestones[start_idx + 1]
    else:
        next_milestone = segment_milestones[start_idx - 1] if start_idx > 0 else None

    if next_milestone is None or next_milestone not in milestone_centers:
        return np.array([milestone_centers[ms] for ms in segment_milestones if ms in milestone_centers])

    start_cells = milestone_cells[start_milestone]
    next_center = milestone_centers[next_milestone]

    distances_to_next = np.linalg.norm(start_cells - next_center, axis=1)

    farthest_threshold = np.percentile(distances_to_next, 90)
    farthest_cells = start_cells[distances_to_next >= farthest_threshold]
    start_point = np.median(farthest_cells, axis=0) if len(farthest_cells) > 0 else start_cells[np.argmax(distances_to_next)]

    closest_threshold = np.percentile(distances_to_next, 20)
    closest_cells = start_cells[distances_to_next <= closest_threshold]
    connection_point = np.median(closest_cells, axis=0) if len(closest_cells) > 0 else start_cells[np.argmin(distances_to_next)]

    enhanced_points = [start_point]

    distances_from_start = np.linalg.norm(start_cells - start_point, axis=1)
    sorted_indices = np.argsort(distances_from_start)
    sorted_cells = start_cells[sorted_indices]

    n_segments = 4
    n_cells = len(sorted_cells)
    for i in range(1, n_segments):
        seg_start = int(i * n_cells / n_segments)
        seg_end = int((i + 1) * n_cells / n_segments)
        if seg_end > seg_start:
            segment_cells = sorted_cells[seg_start:seg_end]
            guide_point = np.median(segment_cells, axis=0)
            enhanced_points.append(guide_point)

    enhanced_points.append(connection_point)

    for ms in segment_milestones[start_idx + 1:]:
        if ms in milestone_centers:
            enhanced_points.append(milestone_centers[ms])

    return np.array(enhanced_points)


def calculate_bifurcating_loop_guide_points(
    embedding: np.ndarray,
    adata: AnnData,
    milestone_key: str,
    centers: List[Tuple[str, np.ndarray]],
    segment_milestones: List[str]
) -> np.ndarray:
    """
    Compute guide points for the bifurcating-loop topology.

    Guide points are inferred from the empirical cell distribution (no fixed spatial orientation
    assumptions). The layout can appear as multiple L-like configurations.

    Core strategy:
    1) Estimate the fork point at the A/B interface:
       midpoint between (A cells closest to B center) and (B cells closest to those A cells).
    2) "Closer-to-target" selection:
       for B→D use B cells closer to D; for B→C use B cells closer to C.
    3) For the B→C loop-like arc:
       start from the fork, move along the B arc to reach the C tip, then return toward the fork
       following the C cell distribution.
    4) Sort by distance proportion rather than by angle to avoid dependence on absolute orientation.
    """
    center_dict = {name: coord for name, coord in centers}
    milestones = adata.obs[milestone_key].astype(str)

    sA_mask = milestones == "sA"
    sB_mask = milestones == "sB"
    sC_mask = milestones == "sC"
    sD_mask = milestones == "sD"

    sA_cells = embedding[sA_mask]
    sB_cells = embedding[sB_mask]
    sC_cells = embedding[sC_mask]
    sD_cells = embedding[sD_mask]

    sA_center = np.median(sA_cells, axis=0) if len(sA_cells) > 0 else center_dict.get("sA")
    sB_center = np.median(sB_cells, axis=0) if len(sB_cells) > 0 else center_dict.get("sB")
    sC_center = np.median(sC_cells, axis=0) if len(sC_cells) > 0 else center_dict.get("sC")
    sD_center = np.median(sD_cells, axis=0) if len(sD_cells) > 0 else center_dict.get("sD")

    def find_fork_point():
        """Estimate the interface point between sA and sB (fork point)."""
        if len(sA_cells) == 0 or len(sB_cells) == 0:
            return sB_center if sB_center is not None else np.array([0, 0])

        dist_sA_to_sB = np.linalg.norm(sA_cells - sB_center, axis=1)
        sA_closest_idx = np.argsort(dist_sA_to_sB)[:max(1, int(len(sA_cells) * 0.2))]
        sA_closest = sA_cells[sA_closest_idx]
        sA_bridge = np.median(sA_closest, axis=0)

        dist_sB_to_sA_bridge = np.linalg.norm(sB_cells - sA_bridge, axis=1)
        sB_closest_idx = np.argsort(dist_sB_to_sA_bridge)[:max(1, int(len(sB_cells) * 0.15))]
        sB_closest = sB_cells[sB_closest_idx]
        sB_bridge = np.median(sB_closest, axis=0)

        fork_point = (sA_bridge + sB_bridge) / 2

        return fork_point

    fork_point = find_fork_point()

    if segment_milestones == ["sA", "sB"]:
        # Heuristic: once a proper distal start point is found, use linear interpolation.
        # Rationale: sA→sB should be a straight trunk, so we interpolate directly to the fork.
        if len(sA_cells) < 3:
            if sA_center is None:
                return np.array([c[1] for c in centers])
            enhanced_points = [sA_center]
            for j in range(1, 3):
                interp = sA_center + j * (fork_point - sA_center) / 3
                enhanced_points.append(interp)
            enhanced_points.append(fork_point)
            return np.array(enhanced_points)

        # Distances from each sA cell to the fork and to the sA centroid.
        dist_to_fork = np.linalg.norm(sA_cells - fork_point, axis=1)
        sA_centroid = np.mean(sA_cells, axis=0)
        dist_to_centroid = np.linalg.norm(sA_cells - sA_centroid, axis=1)

        # Pick a distal start point:
        # among the farthest 20% from the fork, take boundary-like cells far from the centroid.
        far_threshold = np.percentile(dist_to_fork, 80)
        far_mask = dist_to_fork >= far_threshold
        far_cells = sA_cells[far_mask]

        if len(far_cells) > 3:
            far_dist_to_centroid = dist_to_centroid[far_mask]
            far_boundary_mask = far_dist_to_centroid >= np.percentile(far_dist_to_centroid, 50)
            sA_start = np.median(far_cells[far_boundary_mask], axis=0)
        else:
            sA_start = np.median(far_cells, axis=0)

        # Linear interpolation to enforce a straight segment.
        n_points = 4
        enhanced_points = [sA_start]
        for j in range(1, n_points):
            interp = sA_start + j * (fork_point - sA_start) / n_points
            enhanced_points.append(interp)
        enhanced_points.append(fork_point)

        return np.array(enhanced_points)

    elif segment_milestones == ["sB", "sD"]:
        if sD_center is None:
            return np.array([c[1] for c in centers])

        if len(sB_cells) == 0:
            enhanced_points = [fork_point]
            interp = fork_point + (sD_center - fork_point) / 2
            enhanced_points.append(interp)
            enhanced_points.append(sD_center)
            return np.array(enhanced_points)

        dist_to_sD = np.linalg.norm(sB_cells - sD_center, axis=1)
        dist_to_sC = np.linalg.norm(sB_cells - sC_center, axis=1) if sC_center is not None else np.full(len(sB_cells), np.inf)
        closer_to_sD = dist_to_sD < dist_to_sC
        sB_to_sD = sB_cells[closer_to_sD]

        if len(sB_to_sD) < 5:
            enhanced_points = [fork_point]
            interp = fork_point + (sD_center - fork_point) / 2
            enhanced_points.append(interp)
            enhanced_points.append(sD_center)
            return np.array(enhanced_points)

        dist_from_fork = np.linalg.norm(sB_to_sD - fork_point, axis=1)
        dist_to_sD_from_sB = np.linalg.norm(sB_to_sD - sD_center, axis=1)
        total_dist = dist_from_fork + dist_to_sD_from_sB + 1e-6
        progress = dist_from_fork / total_dist

        n_segments = 3
        enhanced_points = [fork_point]
        for i in range(n_segments):
            seg_mask = (progress >= i / n_segments) & (progress < (i + 1) / n_segments)
            if np.sum(seg_mask) > 0:
                seg_center = np.median(sB_to_sD[seg_mask], axis=0)
                enhanced_points.append(seg_center)
        enhanced_points.append(sD_center)

        return np.array(enhanced_points)

    elif segment_milestones == ["sB", "sC"]:
        if sC_center is None:
            return np.array([c[1] for c in centers])

        if len(sB_cells) == 0 or len(sC_cells) == 0:
            enhanced_points = [fork_point]
            for j in range(1, 6):
                interp = fork_point + j * (sC_center - fork_point) / 6
                enhanced_points.append(interp)
            enhanced_points.append(sC_center)
            return np.array(enhanced_points)

        dist_sC_from_fork = np.linalg.norm(sC_cells - fork_point, axis=1)
        sC_tip_idx = np.argsort(dist_sC_from_fork)[-max(1, int(len(sC_cells) * 0.1)):]
        sC_tip = np.median(sC_cells[sC_tip_idx], axis=0)

        dist_to_sC = np.linalg.norm(sB_cells - sC_center, axis=1)
        dist_to_sD = np.linalg.norm(sB_cells - sD_center, axis=1) if sD_center is not None else np.full(len(sB_cells), np.inf)
        closer_to_sC = dist_to_sC < dist_to_sD
        sB_to_sC = sB_cells[closer_to_sC]

        if len(sB_to_sC) < 5:
            enhanced_points = [fork_point]
            for j in range(1, 6):
                interp = fork_point + j * (sC_center - fork_point) / 6
                enhanced_points.append(interp)
            enhanced_points.append(sC_center)
            return np.array(enhanced_points)

        dist_from_fork_sB = np.linalg.norm(sB_to_sC - fork_point, axis=1)
        dist_to_tip_from_sB = np.linalg.norm(sB_to_sC - sC_tip, axis=1)
        total_dist_sB = dist_from_fork_sB + dist_to_tip_from_sB + 1e-6
        progress_sB = dist_from_fork_sB / total_dist_sB

        enhanced_points = [fork_point]

        for i in range(4):
            seg_mask = (progress_sB >= i / 4) & (progress_sB < (i + 1) / 4)
            if np.sum(seg_mask) > 0:
                seg_center = np.median(sB_to_sC[seg_mask], axis=0)
                enhanced_points.append(seg_center)

        enhanced_points.append(sC_tip)

        sC_non_tip_mask = dist_sC_from_fork < np.percentile(dist_sC_from_fork, 90)
        sC_non_tip = sC_cells[sC_non_tip_mask]

        if len(sC_non_tip) >= 3:
            dist_non_tip = np.linalg.norm(sC_non_tip - fork_point, axis=1)
            sorted_idx = np.argsort(dist_non_tip)[::-1]
            sorted_sC = sC_non_tip[sorted_idx]

            n_seg = 2
            for i in range(n_seg):
                start_idx = int(i * len(sorted_sC) / n_seg)
                end_idx = int((i + 1) * len(sorted_sC) / n_seg)
                if end_idx > start_idx:
                    seg_center = np.median(sorted_sC[start_idx:end_idx], axis=0)
                    enhanced_points.append(seg_center)

        final_point = fork_point + 0.03 * (sA_center - fork_point)
        enhanced_points.append(final_point)

        return np.array(enhanced_points)

    else:
        return np.array([c[1] for c in centers])


def calculate_cycle_simple_guide_points(
    embedding: np.ndarray,
    adata: AnnData,
    milestone_key: str,
    centers: List[Tuple[str, np.ndarray]]
) -> np.ndarray:
    """
    Compute guide points for cycle-simple (circular closed loop).

    Strategy:
    - estimate a global center
    - for each milestone, pick an outer boundary representative (far from the center)
    - sort milestone representatives by polar angle to ensure a smooth circular connection
    """
    milestone_names = ['s1', 's2', 's3']
    all_milestone_cells = []
    milestone_cells_dict = {}

    for ms_name in milestone_names:
        mask = adata.obs[milestone_key].astype(str) == ms_name
        if np.sum(mask) > 0:
            cells = embedding[mask]
            all_milestone_cells.append(cells)
            milestone_cells_dict[ms_name] = cells

    if len(all_milestone_cells) < 3:
        return np.array([c[1] for c in centers])

    all_cells = np.vstack(all_milestone_cells)
    global_center = np.median(all_cells, axis=0)

    circle_points = []

    for ms_name in milestone_names:
        if ms_name not in milestone_cells_dict:
            continue

        cells = milestone_cells_dict[ms_name]

        distances = np.linalg.norm(cells - global_center, axis=1)

        threshold = np.percentile(distances, 70)
        outer_mask = distances >= threshold
        outer_cells = cells[outer_mask]

        if len(outer_cells) > 0:
            circle_point = np.median(outer_cells, axis=0)
        else:
            circle_point = np.median(cells, axis=0)

        circle_points.append((ms_name, circle_point))

    points_with_angles = []
    for ms_name, point in circle_points:
        vector = point - global_center
        angle = np.arctan2(vector[1], vector[0])
        points_with_angles.append((ms_name, point, angle))

    points_with_angles.sort(key=lambda x: x[2])

    enhanced_points = []
    n_points = len(points_with_angles)

    for i in range(n_points):
        current_name, current_point, current_angle = points_with_angles[i]
        next_idx = (i + 1) % n_points
        next_name, next_point, next_angle = points_with_angles[next_idx]

        enhanced_points.append(current_point)

        if current_name in milestone_cells_dict and next_name in milestone_cells_dict:
            cells1 = milestone_cells_dict[current_name]
            cells2 = milestone_cells_dict[next_name]

            combined_cells = np.vstack([cells1, cells2])

            distances_to_center = np.linalg.norm(combined_cells - global_center, axis=1)
            outer_threshold = np.percentile(distances_to_center, 60)
            outer_cells = combined_cells[distances_to_center >= outer_threshold]

            if len(outer_cells) > 10:
                vectors = outer_cells - global_center
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])

                angle_diff = next_angle - current_angle
                if angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                    angles[angles < 0] += 2 * np.pi
                    current_angle_adjusted = current_angle if current_angle >= 0 else current_angle + 2 * np.pi
                    next_angle_adjusted = next_angle + 2 * np.pi
                elif angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                    angles[angles > 0] -= 2 * np.pi
                    current_angle_adjusted = current_angle
                    next_angle_adjusted = next_angle - 2 * np.pi
                else:
                    current_angle_adjusted = current_angle
                    next_angle_adjusted = next_angle

                if current_angle_adjusted < next_angle_adjusted:
                    between_mask = (angles >= current_angle_adjusted) & (angles <= next_angle_adjusted)
                else:
                    between_mask = (angles >= next_angle_adjusted) & (angles <= current_angle_adjusted)

                between_cells = outer_cells[between_mask]

                if len(between_cells) > 5:
                    between_angles = angles[between_mask]
                    sorted_indices = np.argsort(between_angles)
                    sorted_cells = between_cells[sorted_indices]

                    n_cells = len(sorted_cells)
                    for j in range(1, 5):
                        start_idx = int((j - 1) * n_cells / 4)
                        end_idx = int(j * n_cells / 4)
                        if end_idx > start_idx:
                            segment_cells = sorted_cells[start_idx:end_idx]
                            guide_point = np.median(segment_cells, axis=0)
                            enhanced_points.append(guide_point)
                    continue

        radius = np.linalg.norm(current_point - global_center)

        angle_diff = next_angle - current_angle
        if angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        elif angle_diff > np.pi:
            angle_diff -= 2 * np.pi

        for j in range(1, 5):
            interp_angle = current_angle + j * angle_diff / 5
            guide_point = global_center + radius * np.array([np.cos(interp_angle), np.sin(interp_angle)])
            enhanced_points.append(guide_point)

    enhanced_points_array = np.array(enhanced_points)
    angles_all = np.arctan2(
        enhanced_points_array[:, 1] - global_center[1],
        enhanced_points_array[:, 0] - global_center[0]
    )
    angles_all = np.unwrap(angles_all)
    order = np.argsort(angles_all)
    enhanced_points_array = enhanced_points_array[order]
    angles_all = angles_all[order]

    radii_all = np.linalg.norm(enhanced_points_array - global_center, axis=1)

    densified = []
    m = len(enhanced_points_array)
    for i in range(m):
        densified.append(enhanced_points_array[i])
        j = (i + 1) % m

        angle_i = angles_all[i]
        angle_j = angles_all[j]
        if angle_j <= angle_i:
            angle_j += 2 * np.pi

        gap = angle_j - angle_i
        steps = max(1, int(np.ceil(gap / (np.pi / 12))))
        radius_interp = (radii_all[i] + radii_all[j]) / 2

        for k in range(1, steps):
            a = angle_i + gap * k / steps
            densified.append(global_center + radius_interp * np.array([np.cos(a), np.sin(a)]))

    return np.array(densified)


def calculate_disconnected_ellipse_guide_points(
    embedding: np.ndarray,
    adata: AnnData,
    milestone_key: str
) -> np.ndarray:
    """
    Compute guide points for the disconnected (standard) right-side closed ellipse.

    Strategy:
    - traverse along the right s1 boundary from far-from-s3 to near-s3
    - add an anchor point at the right s3 center
    - traverse along the right s2 boundary from near-s3 to far-from-s3
    - close the loop along the bottom arc back to the far end of right s1
    """
    milestone_names = ['right s1', 'right s2', 'right s3']
    milestones = adata.obs[milestone_key].astype(str)

    all_cells = []
    milestone_cells_dict = {}

    for ms_name in milestone_names:
        mask = milestones == ms_name
        if np.sum(mask) > 0:
            filtered_mask = filter_outliers_by_distance(embedding, mask, milestone_name=ms_name)
            cells = embedding[filtered_mask]
            all_cells.append(cells)
            milestone_cells_dict[ms_name] = cells

    if len(all_cells) < 3:
        return np.array([np.median(cells, axis=0) for cells in all_cells])

    s1_cells = milestone_cells_dict.get('right s1', np.array([]))
    s2_cells = milestone_cells_dict.get('right s2', np.array([]))
    s3_cells = milestone_cells_dict.get('right s3', np.array([]))

    if len(s1_cells) == 0 or len(s2_cells) == 0 or len(s3_cells) == 0:
        return np.array([np.median(cells, axis=0) for cells in all_cells if len(cells) > 0])

    s3_center = np.median(s3_cells, axis=0)
    all_cells_concat = np.vstack(all_cells)
    global_center = np.median(all_cells_concat, axis=0)

    def get_edge_points_along_milestone(cells, start_point, end_point, n_points=6):
        """Sample representative points along a milestone from start_point to end_point."""
        if len(cells) < 5:
            return [np.median(cells, axis=0)]

        direction = end_point - start_point
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            return [np.median(cells, axis=0)]
        direction = direction / direction_norm

        projections = np.dot(cells - start_point, direction)
        sorted_indices = np.argsort(projections)
        sorted_cells = cells[sorted_indices]

        edge_points = []
        n_cells = len(sorted_cells)

        for i in range(n_points):
            seg_start = int(i * n_cells / n_points)
            seg_end = int((i + 1) * n_cells / n_points)
            if seg_end > seg_start:
                segment_cells = sorted_cells[seg_start:seg_end]
                edge_point = np.median(segment_cells, axis=0)
                edge_points.append(edge_point)

        return edge_points

    s1_dist_to_s3 = np.linalg.norm(s1_cells - s3_center, axis=1)
    s1_far_from_s3 = s1_cells[np.argmax(s1_dist_to_s3)]
    s1_near_to_s3 = s1_cells[np.argmin(s1_dist_to_s3)]

    s2_dist_to_s3 = np.linalg.norm(s2_cells - s3_center, axis=1)
    s2_far_from_s3 = s2_cells[np.argmax(s2_dist_to_s3)]
    s2_near_to_s3 = s2_cells[np.argmin(s2_dist_to_s3)]

    enhanced_points = []

    s1_edge_points = get_edge_points_along_milestone(s1_cells, s1_far_from_s3, s1_near_to_s3, n_points=5)
    enhanced_points.extend(s1_edge_points)

    enhanced_points.append(s3_center)

    s2_edge_points = get_edge_points_along_milestone(s2_cells, s2_near_to_s3, s2_far_from_s3, n_points=5)
    enhanced_points.extend(s2_edge_points)

    s3_direction = s3_center - global_center
    s3_direction_norm = np.linalg.norm(s3_direction)
    if s3_direction_norm > 1e-6:
        s3_direction = s3_direction / s3_direction_norm
    else:
        s3_direction = np.array([0, 1])

    s1_s2_combined = np.vstack([s1_cells, s2_cells])

    projections_to_s3 = np.dot(s1_s2_combined - global_center, s3_direction)

    far_from_s3_threshold = np.percentile(projections_to_s3, 30)
    far_from_s3_cells = s1_s2_combined[projections_to_s3 <= far_from_s3_threshold]

    if len(far_from_s3_cells) >= 5:
        closure_direction = s1_far_from_s3 - s2_far_from_s3
        closure_direction_norm = np.linalg.norm(closure_direction)
        if closure_direction_norm > 1e-6:
            closure_direction = closure_direction / closure_direction_norm

        closure_projections = np.dot(far_from_s3_cells - s2_far_from_s3, closure_direction)
        sorted_indices = np.argsort(closure_projections)
        sorted_closure_cells = far_from_s3_cells[sorted_indices]

        n_closure_points = 4
        for i in range(n_closure_points):
            seg_start = int(i * len(sorted_closure_cells) / n_closure_points)
            seg_end = int((i + 1) * len(sorted_closure_cells) / n_closure_points)
            if seg_end > seg_start:
                segment_cells = sorted_closure_cells[seg_start:seg_end]
                closure_point = np.median(segment_cells, axis=0)
                enhanced_points.append(closure_point)
    else:
        mid_point = (s2_far_from_s3 + s1_far_from_s3) / 2
        offset = -s3_direction * np.linalg.norm(s1_far_from_s3 - s2_far_from_s3) * 0.3
        arc_mid = mid_point + offset

        for i in range(1, 4):
            t = i / 4
            p0 = s2_far_from_s3
            p1 = arc_mid
            p2 = s1_far_from_s3
            interp_point = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
            enhanced_points.append(interp_point)

    return np.array(enhanced_points)


def calculate_chaotic_ellipse_via_boundary(
    embedding: np.ndarray,
    adata: AnnData,
    milestone_key: str,
    dataset_name: str
) -> np.ndarray:
    """
    For the chaotic disconnected variant, fit a closed ellipse-like loop on the right cluster
    without relying on milestone ordering.

    Strategy:
    - collect all right-side cells
    - extract boundary points using a radial sampling heuristic
    - handle tail-like outliers (for certain large-cell datasets) separately
    """
    milestone_names = ['right s1', 'right s2', 'right s3']
    milestones = adata.obs[milestone_key].astype(str)

    all_right_cells = []
    s3_cells = None

    for ms_name in milestone_names:
        mask = milestones == ms_name
        if np.sum(mask) > 0:
            cells = embedding[mask]
            all_right_cells.append(cells)
            if ms_name == 'right s3':
                s3_cells = cells

    if len(all_right_cells) == 0:
        return np.array([])

    all_right_concat = np.vstack(all_right_cells)

    has_tail = 'cell50000' in dataset_name or 'cell100000' in dataset_name

    if has_tail and s3_cells is not None:
        return calculate_ellipse_with_tail(all_right_concat, s3_cells)
    else:
        return calculate_ellipse_boundary(all_right_concat)


def calculate_ellipse_boundary(cells: np.ndarray, n_boundary_points: int = 36) -> np.ndarray:
    """
    Compute boundary points of a 2D point cloud to form an ellipse-like closed loop.

    Strategy: cast rays from the center in multiple directions and pick farthest points.
    """
    if len(cells) < 10:
        return cells

    center = np.median(cells, axis=0)

    angles = np.linspace(0, 2*np.pi, n_boundary_points, endpoint=False)
    boundary_points = []

    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])

        relative_positions = cells - center
        projections = np.dot(relative_positions, direction)

        threshold = np.percentile(projections, 95)
        far_cells = cells[projections >= threshold]

        if len(far_cells) > 0:
            boundary_point = np.median(far_cells, axis=0)
            boundary_points.append(boundary_point)
        else:
            boundary_point = cells[np.argmax(projections)]
            boundary_points.append(boundary_point)

    return np.array(boundary_points)


def calculate_ellipse_with_tail(
    all_cells: np.ndarray,
    s3_cells: np.ndarray
) -> np.ndarray:
    """
    Handle tail-like shapes (e.g. an extended 'right s3' protrusion in large-cell datasets).

    Strategy:
    - estimate tail direction using s3 vs overall center
    - keep additional boundary points in the tail direction while fitting the main boundary elsewhere
    """
    s3_center = np.median(s3_cells, axis=0)
    overall_center = np.median(all_cells, axis=0)

    tail_direction = s3_center - overall_center
    tail_direction_norm = np.linalg.norm(tail_direction)
    if tail_direction_norm < 1e-6:
        return calculate_ellipse_boundary(all_cells)
    tail_direction = tail_direction / tail_direction_norm

    n_boundary_points = 30
    angles = np.linspace(0, 2*np.pi, n_boundary_points, endpoint=False)
    boundary_points = []

    tail_angle = np.arctan2(tail_direction[1], tail_direction[0])

    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])

        relative_positions = all_cells - overall_center
        projections = np.dot(relative_positions, direction)

        angle_diff = abs(angle - tail_angle)
        if angle_diff > np.pi:
            angle_diff = 2*np.pi - angle_diff

        if angle_diff < np.pi/6:
            threshold = np.percentile(projections, 80)
        else:
            threshold = np.percentile(projections, 95)

        far_cells = all_cells[projections >= threshold]

        if len(far_cells) > 0:
            boundary_point = np.median(far_cells, axis=0)
            boundary_points.append(boundary_point)
        else:
            boundary_point = all_cells[np.argmax(projections)]
            boundary_points.append(boundary_point)

    boundary_points = np.array(boundary_points)
    s3_projections = np.dot(s3_cells - overall_center, tail_direction)
    main_body_threshold = np.percentile(s3_projections, 50)
    tail_cells = s3_cells[s3_projections >= main_body_threshold]

    if len(tail_cells) > 5:
        tail_cells_sorted_indices = np.argsort(np.dot(tail_cells - overall_center, tail_direction))
        tail_cells_sorted = tail_cells[tail_cells_sorted_indices]

        n_tail_points = 4
        tail_guide_points = []
        for i in range(n_tail_points):
            seg_start = int(i * len(tail_cells_sorted) / n_tail_points)
            seg_end = int((i + 1) * len(tail_cells_sorted) / n_tail_points)
            if seg_end > seg_start:
                segment_cells = tail_cells_sorted[seg_start:seg_end]
                guide_point = np.median(segment_cells, axis=0)
                tail_guide_points.append(guide_point)

        result_points = list(boundary_points)

        tail_angle_range = np.pi / 4
        indices_to_remove = []
        for i, bp in enumerate(boundary_points):
            bp_angle = np.arctan2(bp[1] - overall_center[1], bp[0] - overall_center[0])
            angle_diff = abs(bp_angle - tail_angle)
            if angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff
            if angle_diff < tail_angle_range:
                indices_to_remove.append(i)

        result_points = [bp for i, bp in enumerate(boundary_points) if i not in indices_to_remove]

        insert_idx = min(indices_to_remove) if indices_to_remove else 0
        for i, tp in enumerate(tail_guide_points):
            result_points.insert(insert_idx + i, tp)

        return np.array(result_points)

    return boundary_points


def calculate_linear_simple_guide_points(
    embedding: np.ndarray,
    adata: AnnData,
    milestone_key: str,
    centers: List[Tuple[str, np.ndarray]]
) -> np.ndarray:
    """
    Compute guide points for linear-simple (often U-shaped).

    Problem:
    - s1 can span most of the U-shape (start → bottom), while s2 occupies only a short tail.
      Using only the s1 center may lose the true start-end geometry.

    Strategy:
    - detect a long-spanning milestone (typically s1)
    - find two endpoints (distal start and the end closer to s2)
    - insert intermediate guide points along the cell distribution to recover the U-shape
    """
    if len(centers) != 2:
        return np.array([c[1] for c in centers])

    ms1_name, ms1_center = centers[0]
    ms2_name, ms2_center = centers[1]

    mask1 = adata.obs[milestone_key].astype(str) == ms1_name
    mask2 = adata.obs[milestone_key].astype(str) == ms2_name

    if np.sum(mask1) == 0 or np.sum(mask2) == 0:
        return np.array([c[1] for c in centers])

    cells1 = embedding[mask1]
    cells2 = embedding[mask2]

    if len(cells1) > 1:
        distances1 = pdist(cells1)
        if distances1.size > 0:
            max_dist1 = np.max(distances1)
        else:
            max_dist1 = 0
    else:
        max_dist1 = 0

    center_distance = np.linalg.norm(ms2_center - ms1_center)

    if max_dist1 > center_distance * 0.8 and len(cells1) > 10:
        distances_to_s2 = np.linalg.norm(cells1 - ms2_center, axis=1)

        farthest_idx = np.argmax(distances_to_s2)
        start_point = cells1[farthest_idx]

        closest_threshold = np.percentile(distances_to_s2, 30)
        closest_mask = distances_to_s2 <= closest_threshold
        closest_cells = cells1[closest_mask]

        if len(closest_cells) > 0:
            connection_point = np.median(closest_cells, axis=0)
        else:
            nearest_idx = np.argmin(distances_to_s2)
            connection_point = cells1[nearest_idx]

        enhanced_points = []
        enhanced_points.append(start_point)

        distances_to_start = np.linalg.norm(cells1 - start_point, axis=1)
        distances_to_connection = np.linalg.norm(cells1 - connection_point, axis=1)

        direct_distance = np.linalg.norm(connection_point - start_point)

        path_mask = (distances_to_start + distances_to_connection) <= direct_distance * 1.5
        path_cells = cells1[path_mask]

        if len(path_cells) > 10:
            path_distances = np.linalg.norm(path_cells - start_point, axis=1)
            sorted_indices = np.argsort(path_distances)
            sorted_path_cells = path_cells[sorted_indices]

            n_segments = 4
            n_cells = len(sorted_path_cells)
            for i in range(1, n_segments):
                start_idx = int(i * n_cells / n_segments)
                end_idx = int((i + 1) * n_cells / n_segments)
                if end_idx > start_idx:
                    segment_cells = sorted_path_cells[start_idx:end_idx]
                    guide_point = np.median(segment_cells, axis=0)
                    enhanced_points.append(guide_point)
        else:
            for i in range(1, 5):
                t = i / 5
                guide_point = start_point + t * (connection_point - start_point)
                enhanced_points.append(guide_point)

        enhanced_points.append(connection_point)

        combined_cells = np.vstack([closest_cells, cells2])

        distances_to_conn = np.linalg.norm(combined_cells - connection_point, axis=1)
        distances_to_s2_center = np.linalg.norm(combined_cells - ms2_center, axis=1)

        transition_distance = np.linalg.norm(ms2_center - connection_point)
        transition_mask = (distances_to_conn + distances_to_s2_center) <= transition_distance * 1.5
        transition_cells = combined_cells[transition_mask]

        if len(transition_cells) > 5:
            transition_distances = np.linalg.norm(transition_cells - connection_point, axis=1)
            sorted_indices = np.argsort(transition_distances)
            sorted_transition_cells = transition_cells[sorted_indices]

            n_trans = len(sorted_transition_cells)
            for i in range(1, 4):
                start_idx = int(i * n_trans / 4)
                end_idx = int((i + 1) * n_trans / 4)
                if end_idx > start_idx:
                    segment_cells = sorted_transition_cells[start_idx:end_idx]
                    guide_point = np.median(segment_cells, axis=0)
                    enhanced_points.append(guide_point)
        else:
            for i in range(1, 4):
                t = i / 4
                guide_point = connection_point + t * (ms2_center - connection_point)
                enhanced_points.append(guide_point)

        enhanced_points.append(ms2_center)

        return np.array(enhanced_points)

    else:
        return None


def calculate_arc_guide_points(
    embedding: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    center1: np.ndarray,
    center2: np.ndarray,
    n_guide_points: int = 3
) -> np.ndarray:
    """
    Compute arc-like guide points between two milestones using the empirical cell distribution.

    Strategy: split the parameterized path into n segments; for each segment, pick the median of
    cells near the corresponding linear interpolation point as the guide point.
    """
    cells1 = embedding[mask1]
    cells2 = embedding[mask2]
    all_cells = np.vstack([cells1, cells2])

    if len(all_cells) == 0:
        return np.array([
            center1 + (i + 1) / (n_guide_points + 1) * (center2 - center1)
            for i in range(n_guide_points)
        ])

    direction = center2 - center1
    distance = np.linalg.norm(direction)

    if distance < 1e-6:
        # Centers overlap: return repeated center points.
        return np.tile(center1, (n_guide_points, 1))

    guide_points = []

    for i in range(n_guide_points):
        t = (i + 1) / (n_guide_points + 1)
        linear_point = center1 + t * direction

        distances = np.linalg.norm(all_cells - linear_point, axis=1)
        reasonable_distance = distance * 0.3
        nearby_mask = distances <= reasonable_distance

        if np.sum(nearby_mask) > 0:
            nearby_cells = all_cells[nearby_mask]
            guide_point = np.median(nearby_cells, axis=0)
        else:
            guide_point = linear_point

        guide_points.append(guide_point)

    return np.array(guide_points)


def identify_finger_branches_by_x_bins(
    cells: np.ndarray,
    fork_point: np.ndarray,
    n_bins: int = 5,
    min_cells: int = 100
) -> List[np.ndarray]:
    """
    Identify finger-like branches by binning along the x-axis.

    For downward finger-like branches, separation is primarily along x.
    """
    if len(cells) < min_cells:
        return [cells]

    x_min, x_max = cells[:, 0].min(), cells[:, 0].max()
    x_range = x_max - x_min

    if x_range < 0.05:
        return [cells]

    bin_edges = np.linspace(x_min, x_max, n_bins + 1)

    branches = []
    for i in range(n_bins):
        mask = (cells[:, 0] >= bin_edges[i]) & (cells[:, 0] < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (cells[:, 0] >= bin_edges[i]) & (cells[:, 0] <= bin_edges[i + 1])

        bin_cells = cells[mask]
        if len(bin_cells) >= min_cells:
            branches.append(bin_cells)

    return branches if branches else [cells]


def get_finger_branch_endpoint(
    branch_cells: np.ndarray,
    fork_point: np.ndarray,
    direction: str = 'down'
) -> np.ndarray:
    """Get a finger-branch endpoint (down: smallest y; right: largest x)."""
    if direction == 'down':
        y_coords = branch_cells[:, 1]
        threshold = np.percentile(y_coords, 10)
        endpoint_cells = branch_cells[y_coords <= threshold]
    elif direction == 'right':
        x_coords = branch_cells[:, 0]
        threshold = np.percentile(x_coords, 90)
        endpoint_cells = branch_cells[x_coords >= threshold]
    else:
        distances = np.linalg.norm(branch_cells - fork_point, axis=1)
        threshold = np.percentile(distances, 90)
        endpoint_cells = branch_cells[distances >= threshold]

    if len(endpoint_cells) == 0:
        endpoint_cells = branch_cells

    return np.median(endpoint_cells, axis=0)


def calculate_consecutive_bifurcating_complex_segments(
    embedding: np.ndarray,
    adata: AnnData,
    milestone_key: str,
    segment_milestones: List[str],
    dataset_name: str,
    center_dict: Dict[str, np.ndarray]
) -> Optional[List[np.ndarray]]:
    """
    Build multiple straight segments for complex branches in large consecutive-bifurcating datasets.

    Supported datasets (identified by dataset_name patterns):
    - cell100000_gene1000: multiple finger-like branches on the right side
    - cell50000_gene1000: 2 branches for sE-sEndE, 4 branches for sF-sEndF, 4 branches for sG-sEndG
    """
    normalized_name = normalize_dataset_name(dataset_name)

    is_cell100000 = 'consecutive-bifurcating_cell100000_gene1000' in normalized_name
    is_cell50000 = 'consecutive-bifurcating_cell50000_gene1000' in normalized_name

    if not is_cell100000 and not is_cell50000:
        return None

    milestones = adata.obs[milestone_key].astype(str)

    # ========== cell50000 special handling ==========
    if is_cell50000:
        # Collect cells needed for the cell50000 case
        sE_cells = embedding[milestones == 'sE']
        sEndE_cells = embedding[milestones == 'sEndE']
        sF_cells = embedding[milestones == 'sF']
        sEndF_cells = embedding[milestones == 'sEndF']
        sG_cells = embedding[milestones == 'sG']
        sEndG_cells = embedding[milestones == 'sEndG']
        sCmid_cells = embedding[milestones == 'sCmid']
        sDmid_cells = embedding[milestones == 'sDmid']

        sDmid_center = center_dict.get('sDmid')
        sCmid_center = center_dict.get('sCmid')

        # If a milestone center is missing from center_dict, compute it from data.
        if sCmid_center is None:
            sCmid_mask = (milestones == 'sCmid').values
            if np.sum(sCmid_mask) > 0:
                sCmid_filtered_mask = filter_outliers_by_distance(embedding, sCmid_mask, milestone_name='sCmid')
                sCmid_center = np.median(embedding[sCmid_filtered_mask], axis=0)

        if sDmid_center is None:
            sDmid_mask = (milestones == 'sDmid').values
            if np.sum(sDmid_mask) > 0:
                sDmid_filtered_mask = filter_outliers_by_distance(embedding, sDmid_mask, milestone_name='sDmid')
                sDmid_center = np.median(embedding[sDmid_filtered_mask], axis=0)

        if sDmid_center is None:
            return None

        segments = []

        # --- sE-sEndE: 2 branches ---
        if segment_milestones == ['sCmid', 'sE', 'sEndE']:
            if len(sE_cells) == 0 or len(sEndE_cells) == 0 or sCmid_center is None:
                return None

            sE_all = np.vstack([sE_cells, sEndE_cells])
            sE_median_y = np.median(sE_all[:, 1])
            sE_branch1 = sE_all[sE_all[:, 1] < sE_median_y]  # lower branch
            sE_branch2 = sE_all[sE_all[:, 1] >= sE_median_y]  # upper branch

            # Upper branch start: sCmid center
            sE_start2 = sCmid_center.copy()

            # Lower branch start: slightly left-below sCmid
            sCmid_left = sCmid_cells[sCmid_cells[:, 0] < sCmid_center[0]]
            sE_start1_base = np.median(sCmid_left, axis=0) if len(sCmid_left) > 0 else sCmid_center
            sE_start1 = np.array([sE_start1_base[0] - 0.04, sE_start1_base[1] - 0.10])

            # Endpoints
            sE_end1 = np.median(sE_branch1[sE_branch1[:, 0] <= np.percentile(sE_branch1[:, 0], 15)], axis=0)
            sE_end2 = np.median(sE_branch2[sE_branch2[:, 0] <= np.percentile(sE_branch2[:, 0], 15)], axis=0)

            # Add line segments: upper branch, connector, lower branch
            segments.append(np.array([sE_start2, sE_end2]))  # upper branch
            segments.append(np.array([sE_start2, sE_start1]))  # connector to lower branch start
            segments.append(np.array([sE_start1, sE_end1]))  # lower branch

            return segments

        # --- sF-sEndF: 4 branches ---
        if segment_milestones == ['sDmid', 'sF', 'sEndF']:
            if len(sF_cells) == 0 or len(sEndF_cells) == 0:
                return None

            # Use KMeans on sEndF cells to find 4 endpoints.
            kmeans_F = KMeans(n_clusters=4, random_state=42, n_init=10)
            labels_F = kmeans_F.fit_predict(sEndF_cells)

            endpoints_F = []
            for i in range(4):
                endpoint_cells = sEndF_cells[labels_F == i]
                if len(endpoint_cells) > 0:
                    endpoint = np.median(endpoint_cells, axis=0)
                    endpoints_F.append(endpoint)
            endpoints_F.sort(key=lambda e: e[0])  # sort by x (left -> right)

            # Define start points for each branch.
            # Branch 1 (leftmost): lower-left of sF, then shift further left.
            sF_leftmost = sF_cells[sF_cells[:, 0] <= np.percentile(sF_cells[:, 0], 25)]
            sF_start1 = np.median(sF_leftmost[sF_leftmost[:, 1] <= np.percentile(sF_leftmost[:, 1], 30)], axis=0)
            sF_start1 = np.array([sF_start1[0] - 0.04, sF_start1[1]])

            # Branch 2: slightly right to branch 1.
            sF_left_mid = sF_cells[(sF_cells[:, 0] > np.percentile(sF_cells[:, 0], 20)) &
                                  (sF_cells[:, 0] <= np.percentile(sF_cells[:, 0], 40))]
            if len(sF_left_mid) > 0:
                sF_start2 = np.median(sF_left_mid[sF_left_mid[:, 1] <= np.percentile(sF_left_mid[:, 1], 25)], axis=0)
            else:
                sF_start2 = np.array([sF_start1[0] + 0.05, sF_start1[1]])

            # Branch 3: sDmid center.
            sF_start3 = sDmid_center.copy()

            # Branch 4 (rightmost): far-right of sDmid.
            sDmid_far_right = sDmid_cells[sDmid_cells[:, 0] >= np.percentile(sDmid_cells[:, 0], 85)]
            if len(sDmid_far_right) > 0:
                sF_start4 = np.median(sDmid_far_right, axis=0)
            else:
                sF_start4 = np.array([sDmid_center[0] + 0.08, sDmid_center[1]])

            sF_starts = [sF_start1, sF_start2, sF_start3, sF_start4]

            # Add line segments.
            for i, endpoint in enumerate(endpoints_F):
                if i < len(sF_starts):
                    segments.append(np.array([sF_starts[i], endpoint]))

            return segments if segments else None

        # --- sG-sEndG: 4 branches ---
        if segment_milestones == ['sDmid', 'sG', 'sEndG']:
            if len(sG_cells) == 0 or len(sEndG_cells) == 0:
                return None

            # Use KMeans on sEndG cells to find 4 endpoints.
            kmeans_G = KMeans(n_clusters=4, random_state=42, n_init=10)
            labels_G = kmeans_G.fit_predict(sEndG_cells)

            endpoints_G = []
            for i in range(4):
                endpoint_cells = sEndG_cells[labels_G == i]
                if len(endpoint_cells) > 0:
                    endpoint = np.median(endpoint_cells, axis=0)
                    endpoints_G.append(endpoint)
            endpoints_G.sort(key=lambda e: -e[1])  # sort by y (top -> bottom)

            # For each endpoint, pick an sG start point around the endpoint's y coordinate.
            sG_starts = []
            for idx, endpoint in enumerate(endpoints_G):
                y_range = 0.025
                sG_in_y = sG_cells[(sG_cells[:, 1] >= endpoint[1] - y_range) &
                                  (sG_cells[:, 1] <= endpoint[1] + y_range)]
                if len(sG_in_y) < 20:
                    sG_in_y = sG_cells[(sG_cells[:, 1] >= endpoint[1] - 0.04) &
                                      (sG_cells[:, 1] <= endpoint[1] + 0.04)]
                if len(sG_in_y) > 0:
                    start_point = np.median(sG_in_y[sG_in_y[:, 0] <= np.percentile(sG_in_y[:, 0], 20)], axis=0)
                else:
                    start_point = np.array([sDmid_center[0] + 0.05, endpoint[1]])
                sG_starts.append(start_point)

            # For the lowest branch (#4), reuse the start of branch #3.
            if len(sG_starts) >= 4:
                sG_starts[3] = sG_starts[2].copy()

            # Add segments: first connectors from sDmid center to starts, then branch segments.
            for i, (start, endpoint) in enumerate(zip(sG_starts, endpoints_G)):
                # Connector from sDmid center to branch start.
                segments.append(np.array([sDmid_center, start]))
                # Branch segment.
                segments.append(np.array([start, endpoint]))

            return segments if segments else None

        return None

    # ========== cell100000 special handling (original logic) ==========
    if is_cell100000:
        if segment_milestones not in [['sDmid', 'sF', 'sEndF'], ['sDmid', 'sG', 'sEndG']]:
            return None

        # Collect cells and centers needed for the cell100000 case
        sF_cells = embedding[milestones == 'sF']
        sEndF_cells = embedding[milestones == 'sEndF']
        sG_cells = embedding[milestones == 'sG']
        sEndG_cells = embedding[milestones == 'sEndG']

        sDmid_center = center_dict.get('sDmid')
        sD_center = center_dict.get('sD')

        # If sDmid center is missing from center_dict, compute it from data.
        if sDmid_center is None:
            sDmid_mask = (milestones == 'sDmid').values
            if np.sum(sDmid_mask) > 0:
                sDmid_filtered_mask = filter_outliers_by_distance(embedding, sDmid_mask, milestone_name='sDmid')
                sDmid_center = np.median(embedding[sDmid_filtered_mask], axis=0)

        # If sD center is missing from center_dict, compute it from data (used for downward sG branches).
        if sD_center is None:
            sD_mask = (milestones == 'sD').values
            if np.sum(sD_mask) > 0:
                sD_filtered_mask = filter_outliers_by_distance(embedding, sD_mask, milestone_name='sD')
                sD_center = np.median(embedding[sD_filtered_mask], axis=0)

        if sDmid_center is None:
            return None

        sF_top = sF_cells[sF_cells[:, 1] >= np.percentile(sF_cells[:, 1], 90)] if len(sF_cells) > 0 else np.array([])
        sG_left = sG_cells[sG_cells[:, 0] <= np.percentile(sG_cells[:, 0], 10)] if len(sG_cells) > 0 else np.array([])

        if len(sF_top) > 0 and len(sG_left) > 0:
            fork2_point = np.median(np.vstack([sF_top, sG_left]), axis=0)
        elif len(sF_top) > 0:
            fork2_point = np.median(sF_top, axis=0)
        elif len(sG_left) > 0:
            fork2_point = np.median(sG_left, axis=0)
        else:
            fork2_point = sDmid_center

        segments = []

        if segment_milestones == ['sDmid', 'sF', 'sEndF']:
            sF_all = np.vstack([sF_cells, sEndF_cells]) if len(sF_cells) > 0 and len(sEndF_cells) > 0 else (
                sF_cells if len(sF_cells) > 0 else sEndF_cells
            )

            if len(sF_all) == 0:
                return None

            sF_below = sF_all[sF_all[:, 1] < sDmid_center[1]]

            if len(sF_below) < 50:
                sF_below = sF_all

            sF_fingers = identify_finger_branches_by_x_bins(sF_below, sDmid_center, n_bins=6, min_cells=200)

            for finger_cells in sF_fingers:
                endpoint = get_finger_branch_endpoint(finger_cells, sDmid_center, direction='down')
                segments.append(np.array([sDmid_center, endpoint]))

            segments.append(np.array([sDmid_center, fork2_point]))

            return segments

        elif segment_milestones == ['sDmid', 'sG', 'sEndG']:
            sG_all = np.vstack([sG_cells, sEndG_cells]) if len(sG_cells) > 0 and len(sEndG_cells) > 0 else (
                sG_cells if len(sG_cells) > 0 else sEndG_cells
            )

            if len(sG_all) == 0:
                return None

            sG_right = sG_all[sG_all[:, 0] > fork2_point[0]]

            if len(sG_right) > 50:
                endpoint_sG_right = get_finger_branch_endpoint(sG_right, fork2_point, direction='right')
                segments.append(np.array([fork2_point, endpoint_sG_right]))

            if sD_center is not None:
                sG_below = sG_all[sG_all[:, 1] < sD_center[1]]

                if len(sG_below) > 100:
                    sG_down_fingers = identify_finger_branches_by_x_bins(sG_below, sD_center, n_bins=4, min_cells=100)

                    for finger_cells in sG_down_fingers:
                        endpoint = get_finger_branch_endpoint(finger_cells, sD_center, direction='down')
                        segments.append(np.array([sD_center, endpoint]))

            return segments if segments else None

    return None


def calculate_smoothing_parameter(
    curve_points: np.ndarray,
    topology_type: str,
    is_cycle: bool,
    k: int
) -> float:
    """
    Compute spline smoothing parameter based on topology and curve characteristics.

    Heuristic:
    - Closed loops: larger smoothing to preserve natural curvature
    - Arc-like (linear-simple): larger smoothing to preserve the arc
    - Mostly linear: smaller smoothing to stay closer to milestone centers
    - bifurcating-loop: larger smoothing due to pronounced curvature/arc segment
    """
    if len(curve_points) <= 2:
        return 0

    normalized_topology = normalize_topology_type(topology_type)

    distances = np.sum((curve_points[1:] - curve_points[:-1])**2, axis=1)
    avg_distance = np.mean(distances)
    n_points = len(curve_points)

    base_smoothing = n_points * avg_distance

    if is_cycle or normalized_topology == 'cycle-simple':
        smoothing_factor = 0.07 if k == 3 else 0.05
    elif normalized_topology == 'bifurcating-loop':
        smoothing_factor = 0.15 if k == 3 else 0.05
    elif normalized_topology == 'linear-simple':
        smoothing_factor = 0.12
    elif normalized_topology in ('bifurcating', 'genesub-bifurcating', 'cellsub-bifurcating',
                                  'trifurcating', 'consecutive-bifurcating'):
        smoothing_factor = 0.08
    else:
        smoothing_factor = 0.05

    return base_smoothing * smoothing_factor


def calculate_reference_directions_from_trajectory(
    adata: AnnData,
    milestone_key: str,
    topology_type: str,
    dataset_name: str,
    basis: BasisType = 'dimred',
    use_gt_velocity_enhancement: bool = False,
    ground_truth_velocity_key: str = "ground_truth_velocity"
) -> np.ndarray:
    """
    Fit trajectory curve(s) from milestone centers and compute per-cell reference directions.

    Modes:
    - use_gt_velocity_enhancement=False (default):
      spline interpolation based on milestone centers only (same as the real-data pipeline).
    - use_gt_velocity_enhancement=True:
      insert intermediate guide points using ground-truth velocity direction (optional for simulations).
    """
    embedding_key = f"X_{basis}"
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm")

    x_embedding = adata.obsm[embedding_key]
    reference_directions = np.zeros((adata.n_obs, 2))

    gt_velocity_embedding = None
    if use_gt_velocity_enhancement:
        gt_velocity_key = f"{ground_truth_velocity_key}_{basis}"
        if gt_velocity_key not in adata.obsm:
            use_gt_velocity_enhancement = False
        else:
            gt_velocity_embedding = adata.obsm[gt_velocity_key]

    standardize_milestone_column(adata, milestone_key)

    segments = get_trajectory_segments_for_topology(
        topology_type, adata, milestone_key, dataset_name
    )

    calculated_cells = set()

    for segment_idx, (segment_milestones, is_cycle) in enumerate(segments):
        available_milestones = set(adata.obs[milestone_key].astype(str).unique())
        valid_segment = [m for m in segment_milestones if m in available_milestones]

        if len(valid_segment) < 2:
            continue

        centers = []
        for milestone in valid_segment:
            mask = adata.obs[milestone_key].astype(str) == milestone
            if np.sum(mask) > 0:
                filtered_mask = filter_outliers_by_distance(x_embedding, mask, milestone_name=milestone)
                center = np.median(x_embedding[filtered_mask], axis=0)
                centers.append((milestone, center))

        def get_milestone_order(center_tuple, segment_list):
            milestone_name = str(center_tuple[0])
            return segment_list.index(milestone_name) if milestone_name in segment_list else 999

        centers.sort(key=partial(get_milestone_order, segment_list=valid_segment))

        center_coords = np.array([c[1] for c in centers])

        if len(center_coords) < 2:
            continue

        normalized_topology = normalize_topology_type(topology_type)

        needs_high_density = False
        needs_cell_guided = False

        center_dict = {name: coord for name, coord in centers}

        complex_segments = calculate_consecutive_bifurcating_complex_segments(
            x_embedding, adata, milestone_key, valid_segment, dataset_name, center_dict
        )
        if complex_segments is not None:
            segment_cells_mask = adata.obs[milestone_key].astype(str).isin(valid_segment)

            for cell_idx in np.where(segment_cells_mask)[0]:
                if cell_idx in calculated_cells:
                    continue

                point = x_embedding[cell_idx]

                min_dist = float("inf")
                best_tangent = None

                for seg_points in complex_segments:
                    start_point = seg_points[0]
                    end_point = seg_points[1]

                    line_vec = end_point - start_point
                    line_len = np.linalg.norm(line_vec)
                    if line_len < 1e-6:
                        continue

                    line_unit = line_vec / line_len

                    point_vec = point - start_point

                    proj_len = np.dot(point_vec, line_unit)

                    proj_len = max(0, min(line_len, proj_len))

                    closest_point = start_point + proj_len * line_unit

                    dist = np.linalg.norm(point - closest_point)

                    if dist < min_dist:
                        min_dist = dist
                        best_tangent = line_unit

                if best_tangent is not None:
                    reference_directions[cell_idx] = best_tangent
                    calculated_cells.add(cell_idx)

            continue

        if normalized_topology == 'bifurcating-loop':
            curve_points = calculate_bifurcating_loop_guide_points(
                x_embedding, adata, milestone_key, centers, valid_segment
            )
        elif normalized_topology == 'cycle-simple' and is_cycle:
            curve_points = calculate_cycle_simple_guide_points(
                x_embedding, adata, milestone_key, centers
            )
        elif normalized_topology == 'linear-simple':
            linear_curve_points = calculate_linear_simple_guide_points(
                x_embedding, adata, milestone_key, centers
            )
            if linear_curve_points is not None:
                curve_points = linear_curve_points
            else:
                curve_points = None
        elif normalized_topology == 'disconnected':
            if 'left sA' in valid_segment:
                curve_points = calculate_extended_linear_segment_points(
                    x_embedding, adata, milestone_key, valid_segment, 'left sA'
                )
            elif is_cycle and 'right s1' in valid_segment:
                variant = detect_disconnected_variant(dataset_name)
                if variant == 'chaotic':
                    curve_points = calculate_chaotic_ellipse_via_boundary(
                        x_embedding, adata, milestone_key, dataset_name
                    )
                else:
                    curve_points = calculate_disconnected_ellipse_guide_points(
                        x_embedding, adata, milestone_key
                    )
            else:
                curve_points = None
        elif normalized_topology == 'linear-bifurcating' and 'left sA' in valid_segment:
            curve_points = calculate_extended_linear_segment_points(
                x_embedding, adata, milestone_key, valid_segment, 'left sA'
            )
        elif normalized_topology == 'linear-linear' and 'right sA' in valid_segment:
            curve_points = calculate_extended_linear_segment_points(
                x_embedding, adata, milestone_key, valid_segment, 'right sA'
            )
        else:
            curve_points = None

        if curve_points is None:
            num_milestones = len(center_coords)
            needs_high_density = (
                is_cycle or
                normalized_topology == 'linear-simple' or
                num_milestones == 2
            )

            needs_cell_guided = (
                is_cycle or
                normalized_topology == 'linear-simple'
            )

            enhanced_points = []
            for i in range(len(center_coords)):
                enhanced_points.append(center_coords[i])

                if i < len(center_coords) - 1:
                    current_milestone = centers[i][0]
                    next_milestone = centers[i+1][0]

                    if needs_cell_guided:
                        mask1 = adata.obs[milestone_key].astype(str) == current_milestone
                        mask2 = adata.obs[milestone_key].astype(str) == next_milestone

                        n_guides = 4 if needs_high_density else 2
                        guide_points = calculate_arc_guide_points(
                            x_embedding, mask1, mask2,
                            center_coords[i], center_coords[i+1],
                            n_guide_points=n_guides
                        )
                        enhanced_points.extend(guide_points)

                    elif needs_high_density:
                        for j in range(1, 5):
                            interp_point = center_coords[i] + j * (center_coords[i+1] - center_coords[i]) / 5
                            enhanced_points.append(interp_point)
                    else:
                        point1 = center_coords[i] + (center_coords[i+1] - center_coords[i]) / 3
                        point2 = center_coords[i] + 2 * (center_coords[i+1] - center_coords[i]) / 3
                        enhanced_points.append(point1)
                        enhanced_points.append(point2)

            curve_points = np.array(enhanced_points)

        if is_cycle and len(curve_points) >= 3:
            first_milestone = centers[0][0]
            last_milestone = centers[-1][0]

            if needs_cell_guided:
                mask1 = adata.obs[milestone_key].astype(str) == last_milestone
                mask2 = adata.obs[milestone_key].astype(str) == first_milestone

                n_guides = 4 if needs_high_density else 2
                guide_points = calculate_arc_guide_points(
                    x_embedding, mask1, mask2,
                    center_coords[-1], center_coords[0],
                    n_guide_points=n_guides
                )
                for guide_point in guide_points:
                    curve_points = np.vstack([curve_points, guide_point])

            elif needs_high_density:
                last_center = center_coords[-1]
                first_center = center_coords[0]
                for j in range(1, 5):
                    interp_point = last_center + j * (first_center - last_center) / 5
                    curve_points = np.vstack([curve_points, interp_point])

            curve_points = np.vstack([curve_points, curve_points[0]])

        if use_gt_velocity_enhancement and gt_velocity_embedding is not None:
            center_milestones = [c[0] for c in centers]
            enhanced_curve_points = []

            for i, point in enumerate(curve_points):
                enhanced_curve_points.append(point)

                if i < len(center_milestones):
                    milestone = center_milestones[min(i, len(center_milestones)-1)]
                    mask = adata.obs[milestone_key].astype(str) == milestone

                    if mask.sum() > 0:
                        avg_gt_velocity = np.mean(gt_velocity_embedding[mask], axis=0)
                        velocity_norm = np.linalg.norm(avg_gt_velocity)

                        if velocity_norm > 1e-6 and i < len(curve_points) - 1:
                            direction = avg_gt_velocity / velocity_norm
                            adjustment = 0.1 * velocity_norm * direction
                            adjusted_point = point + adjustment
                            if np.linalg.norm(adjusted_point - point) < np.linalg.norm(curve_points[i+1] - point) * 0.3:
                                enhanced_curve_points[-1] = adjusted_point

            curve_points = np.array(enhanced_curve_points)


        num_original_milestones = len(center_coords)
        k_ideal = get_spline_degree_for_segment(
            topology_type, segment_idx, num_original_milestones, is_cycle
        )
        k = min(k_ideal, len(curve_points) - 1)

        try:
            s = calculate_smoothing_parameter(curve_points, topology_type, is_cycle, k)

            # pylint: disable=unbalanced-tuple-unpacking
            tck, _ = splprep(
                [curve_points[:, 0], curve_points[:, 1]],
                s=s,
                k=k,
                per=is_cycle
            )
        except Exception:
            continue

        segment_cells_mask = adata.obs[milestone_key].astype(str).isin(valid_segment)

        for i in np.where(segment_cells_mask)[0]:
            if i in calculated_cells:
                continue

            point = x_embedding[i]

            min_dist = float("inf")
            min_u = 0

            for u_val in np.linspace(0, 1, 200):
                curve_point = np.array(splev(u_val, tck))
                dist = np.linalg.norm(point - curve_point)

                if dist < min_dist:
                    min_dist = dist
                    min_u = u_val

            deriv = np.array(splev(min_u, tck, der=1))
            norm = np.linalg.norm(deriv)
            if norm > 0:
                reference_directions[i] = deriv / norm
                calculated_cells.add(i)

    return reference_directions


def plot_angle_rose_diagrams_simulation(
    adata: AnnData,
    angles: np.ndarray,
    milestone_key: str,
    valid_milestones: List[str],
    pdf_path: str,
    png_path: str,
    tool: str
) -> None:
    """Plot per-milestone angle rose diagrams for simulated datasets."""

    n_milestones = len(valid_milestones)

    if n_milestones <= 4:
        cols = n_milestones
        rows = 1
    elif n_milestones <= 8:
        cols = 4
        rows = 2
    else:
        cols = 5
        rows = (n_milestones + cols - 1) // cols

    fig_width = min(cols * 2.5, 20)
    fig_height = max(rows * 2.5, 4)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    n_bins = ROSE_PLOT_N_BINS
    bin_width = ROSE_PLOT_MAX_ANGLE / n_bins
    bins = np.linspace(0, ROSE_PLOT_MAX_ANGLE, n_bins + 1)
    bin_centers = bins[:-1] + bin_width / 2

    cmap = plt.get_cmap('RdYlGn_r')
    norm = mcolors.Normalize(vmin=0, vmax=ROSE_PLOT_MAX_ANGLE)

    for idx, milestone in enumerate(valid_milestones):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        ax.set_aspect("equal")

        milestone_indices = adata.obs[milestone_key].astype(str) == milestone
        milestone_angles = angles[milestone_indices]
        milestone_angles = milestone_angles[~np.isnan(milestone_angles)]

        if len(milestone_angles) == 0:
            ax.text(0, 0, "No Data", ha="center", va="center", fontsize=12)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.1, 1.2)
            ax.axis("off")
            continue

        hist, _ = np.histogram(milestone_angles, bins=bins)
        if hist.size > 0:
            max_val = np.max(hist)
            max_count = max_val if max_val > 0 else 1
        else:
            max_count = 1

        milestone_color = PALETTE_34[idx % len(PALETTE_34)]

        radius = 1.0
        theta = np.linspace(0, np.pi, 100)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        ax.plot(x, y, color=milestone_color, linestyle="-", linewidth=0.8)
        ax.plot([-radius, radius], [0, 0], color=milestone_color, linestyle="-", linewidth=0.8)

        for angle_val in bins[1:-1]:
            plotting_angle_rad = np.radians(angle_val)
            ax.plot(
                [0, radius * np.cos(plotting_angle_rad)],
                [0, radius * np.sin(plotting_angle_rad)],
                color="gray", linestyle="--", linewidth=0.15, alpha=0.3
            )

        for j, count_val in enumerate(hist):
            if count_val == 0:
                continue

            data_angle_start = float(bins[j])
            data_angle_end = float(bins[j + 1])
            height = count_val / max_count * radius

            edge_width = 0.3 if n_bins > 12 else 0.5

            wedge = Wedge(
                (0, 0), height, data_angle_start, data_angle_end,
                facecolor=cmap(norm(bin_centers[j])),
                edgecolor=milestone_color,
                alpha=0.7, linewidth=edge_width
            )
            ax.add_artist(wedge)
        
        ax.set_xlim(-radius * 1.2, radius * 1.1)
        ax.set_ylim(-0.1, radius * 1.1)
        ax.axis("off")
        
        # Panel title
        ax.text(0, radius * 1.25, milestone, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=milestone_color)
    
    # Hide unused axes
    for idx in range(n_milestones, rows * cols):
        row = idx // cols
        col = idx % cols
        # Defensive: ensure the object has axis()
        ax_to_hide = axes[row, col]
        if hasattr(ax_to_hide, 'axis'):
            ax_to_hide.axis("off")
    
    # Suptitle
    title = f"{tool} Velocity-Reference Direction Alignment"
    fig.suptitle(title, x=0.5, y=0.97, fontsize=14, fontweight="bold")
    
    # Colorbar
    reversed_cmap = plt.get_cmap('RdYlGn')
    # Explicit cax to avoid backend differences
    cax = fig.add_axes((0.15, 0.05, 0.7, 0.02))
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=reversed_cmap),
        cax=cax, orientation="horizontal"
    )
    cb.set_ticks([0, 90, 180])
    cb.set_ticklabels(["180°", "90°", "0°"])
    
    plt.subplots_adjust(
        bottom=0.12, left=0.05, right=0.95, top=0.9,
        wspace=0.05, hspace=0.3
    )
    
    plt.savefig(pdf_path, format="pdf", dpi=400, bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=400, bbox_inches="tight")
    plt.close()


def plot_reference_curves_visualization(
    adata: AnnData,
    milestone_key: str,
    topology_type: str,
    dataset_name: str,
    basis: BasisType,
    png_path: str,
    tool: str,
    use_gt_velocity_enhancement: bool = False
) -> None:
    """
    Visualize fitted reference trajectory curves (debug/QA plot).

    Args:
        adata: AnnData object.
        milestone_key: Column in adata.obs for milestones.
        topology_type: Topology name.
        dataset_name: Dataset identifier/name.
        basis: Embedding basis ('dimred' or 'umap').
        png_path: Output PNG path.
        tool: Tool/method name used in titles.
        use_gt_velocity_enhancement: Whether to apply optional GT-velocity-based guide-point tweaks.
    """
    # Embedding
    embedding_key = f"X_{basis}"
    x_embedding = adata.obsm[embedding_key]

    # Normalize milestone names (must be done before get_trajectory_segments_for_topology
    # because detect_disconnected_variant depends on standardized milestone labels).
    standardize_milestone_column(adata, milestone_key)

    # Optional: load GT velocity embedding for guide-point enhancement
    gt_velocity_embedding = None
    if use_gt_velocity_enhancement:
        gt_velocity_key = f"ground_truth_velocity_{basis}"
        if gt_velocity_key in adata.obsm:
            gt_velocity_embedding = adata.obsm[gt_velocity_key]
        else:
            use_gt_velocity_enhancement = False

    # Trajectory segments (use segments, not full paths, to avoid re-drawing shared trunks)
    segments = get_trajectory_segments_for_topology(
        topology_type, adata, milestone_key, dataset_name
    )

    # Figure
    _, ax = plt.subplots(figsize=(10, 10))

    # Colors per milestone
    milestones = adata.obs[milestone_key].astype(str).unique()
    milestone_colors = {ms: PALETTE_34[i % len(PALETTE_34)] for i, ms in enumerate(milestones)}

    # Scatter plot (colored by milestone)
    for milestone in milestones:
        mask = adata.obs[milestone_key].astype(str) == milestone
        points = x_embedding[mask]
        ax.scatter(
            points[:, 0], points[:, 1],
            c=milestone_colors[milestone],
            label=milestone,
            s=10,
            alpha=0.5,
            edgecolors='none'
        )

    # Track milestones whose centers are already plotted to avoid duplicates at branch points
    plotted_milestones = set()

    # Plot curve(s) per segment
    for segment_idx, (segment_milestones, is_cycle) in enumerate(segments):
        # Keep only milestones that exist in this dataset
        available_milestones = set(adata.obs[milestone_key].astype(str).unique())
        valid_segment = [m for m in segment_milestones if m in available_milestones]

        if len(valid_segment) < 2:
            continue

        # Compute per-milestone centers (after outlier filtering)
        centers = []
        for milestone in valid_segment:
            mask = adata.obs[milestone_key].astype(str) == milestone
            if np.sum(mask) > 0:
                # Outlier filtering with milestone-specific threshold heuristics
                filtered_mask = filter_outliers_by_distance(x_embedding, mask, milestone_name=milestone)
                # Median center from filtered cells
                center = np.median(x_embedding[filtered_mask], axis=0)
                centers.append((milestone, center))

        # Sort by the milestone order within the segment
        def get_milestone_order(center_tuple, segment_list):
            milestone_name = str(center_tuple[0])
            return segment_list.index(milestone_name) if milestone_name in segment_list else 999

        centers.sort(key=partial(get_milestone_order, segment_list=valid_segment))
        center_coords = np.array([c[1] for c in centers])

        if len(center_coords) < 2:
            continue

        # Plot milestone centers (only once per milestone across segments)
        centers_to_plot = []
        for milestone, center in centers:
            if milestone not in plotted_milestones:
                centers_to_plot.append(center)
                plotted_milestones.add(milestone)

        if centers_to_plot:
            centers_to_plot_array = np.array(centers_to_plot)
            ax.scatter(
                centers_to_plot_array[:, 0], centers_to_plot_array[:, 1],
                c='black',
                s=150,
                marker='*',
                edgecolors='white',
                linewidths=1.5,
                zorder=10
            )

        # ===== Control points (match calculate_reference_directions_from_trajectory) =====
        # Heuristic: adjust control-point density / guidance based on topology and curve characteristics.
        normalized_topology = normalize_topology_type(topology_type)

        # Control flags
        needs_high_density = False
        needs_cell_guided = False

        # Build center_dict for special-case helper functions
        center_dict = {name: coord for name, coord in centers}

        # Special case 0: complex branches in consecutive-bifurcating_cell100000_gene1000
        # This dataset has multiple finger-like branches on the right side; draw multiple straight segments.
        complex_segments = calculate_consecutive_bifurcating_complex_segments(
            x_embedding, adata, milestone_key, valid_segment, dataset_name, center_dict
        )
        if complex_segments is not None:
            # Draw multiple straight segments
            for seg_points in complex_segments:
                ax.plot(
                    seg_points[:, 0], seg_points[:, 1],
                    color='black',
                    linestyle='-',
                    linewidth=2.5,
                    zorder=5,
                    alpha=0.8
                )
                # Mark endpoints
                ax.scatter(
                    seg_points[1, 0], seg_points[1, 1],
                    c='red',
                    s=50,
                    marker='o',
                    zorder=15,
                    alpha=0.8
                )
            # Done for this segment; skip spline fitting below.
            continue

        # Special case 1: bifurcating-loop (T-shape + D-like arc)
        if normalized_topology == 'bifurcating-loop':
            # Use the dedicated guide-point builder for the current segment.
            curve_points = calculate_bifurcating_loop_guide_points(
                x_embedding, adata, milestone_key, centers, valid_segment
            )
        # Special case 2: cycle-simple (circular closed loop)
        elif normalized_topology == 'cycle-simple' and is_cycle:
            # Use boundary points on the circle instead of passing through the center.
            curve_points = calculate_cycle_simple_guide_points(
                x_embedding, adata, milestone_key, centers
            )
        # Special case 3: linear-simple (often U-shaped): when s1 spans widely, use endpoints not the center.
        elif normalized_topology == 'linear-simple':
            # Try the dedicated U-shape handler.
            linear_curve_points = calculate_linear_simple_guide_points(
                x_embedding, adata, milestone_key, centers
            )
            if linear_curve_points is not None:
                # Wide-span case handled.
                curve_points = linear_curve_points
            else:
                # Not wide enough; fall back to the generic logic below.
                curve_points = None  # fall back to the generic logic below
        # Special case 4: disconnected segments (left linear vs right ellipse loop)
        elif normalized_topology == 'disconnected':
            # Decide whether this segment is the left linear part or the right ellipse loop.
            if 'left sA' in valid_segment:
                # Left linear segment: extend the start of 'left sA'.
                curve_points = calculate_extended_linear_segment_points(
                    x_embedding, adata, milestone_key, valid_segment, 'left sA'
                )
            elif is_cycle and 'right s1' in valid_segment:
                # Right ellipse loop segment.
                variant = detect_disconnected_variant(dataset_name)
                if variant == 'chaotic':
                    # Chaotic: fit by boundary.
                    curve_points = calculate_chaotic_ellipse_via_boundary(
                        x_embedding, adata, milestone_key, dataset_name
                    )
                else:
                    # Standard: use milestone-based distribution.
                    curve_points = calculate_disconnected_ellipse_guide_points(
                        x_embedding, adata, milestone_key
                    )
            else:
                curve_points = None  # use generic logic
        # Special case 5: linear-bifurcating left linear segment (extend 'left sA')
        elif normalized_topology == 'linear-bifurcating' and 'left sA' in valid_segment:
            curve_points = calculate_extended_linear_segment_points(
                x_embedding, adata, milestone_key, valid_segment, 'left sA'
            )
        # Special case 6: linear-linear right linear segment (extend 'right sA')
        elif normalized_topology == 'linear-linear' and 'right sA' in valid_segment:
            curve_points = calculate_extended_linear_segment_points(
                x_embedding, adata, milestone_key, valid_segment, 'right sA'
            )
        else:
            curve_points = None  # use generic logic

        # If no special-case handler produced curve_points, use the generic logic below.
        if curve_points is None:
            # Decide whether high-density interpolation is needed.
            num_milestones = len(center_coords)
            needs_high_density = (
                is_cycle or  # closed loops need more control points
                normalized_topology == 'linear-simple' or  # arc-like shapes benefit from more points
                num_milestones == 2  # two-milestone segments need more intermediate points
            )

            # Decide whether to use cell-guided arc points based on the empirical distribution.
            # For circular/arc-like trajectories, this helps avoid unrealistic shortcuts.
            needs_cell_guided = (
                is_cycle or  # closed loops (cycle-simple, disconnected-right)
                normalized_topology == 'linear-simple'  # U-shaped arc
            )

            enhanced_points = []
            for i in range(len(center_coords)):
                enhanced_points.append(center_coords[i])

                if i < len(center_coords) - 1:
                    # Current and next milestone
                    current_milestone = centers[i][0]
                    next_milestone = centers[i+1][0]

                    if needs_cell_guided:
                        # Cell-guided arc points
                        mask1 = adata.obs[milestone_key].astype(str) == current_milestone
                        mask2 = adata.obs[milestone_key].astype(str) == next_milestone

                        # Compute guide points along the empirical path between two milestones.
                        n_guides = 4 if needs_high_density else 2
                        guide_points = calculate_arc_guide_points(
                            x_embedding, mask1, mask2,
                            center_coords[i], center_coords[i+1],
                            n_guide_points=n_guides
                        )
                        enhanced_points.extend(guide_points)

                    elif needs_high_density:
                        # High density: insert 4 intermediate points (5 segments).
                        for j in range(1, 5):
                            interp_point = center_coords[i] + j * (center_coords[i+1] - center_coords[i]) / 5
                            enhanced_points.append(interp_point)
                    else:
                        # Standard density: insert 2 intermediate points (3 segments).
                        point1 = center_coords[i] + (center_coords[i+1] - center_coords[i]) / 3
                        point2 = center_coords[i] + 2 * (center_coords[i+1] - center_coords[i]) / 3
                        enhanced_points.append(point1)
                        enhanced_points.append(point2)

            curve_points = np.array(enhanced_points)

        # Closed-loop handling: append the first point to the end and add end-to-start intermediates.
        if is_cycle and len(curve_points) >= 3:
            # First and last milestone
            first_milestone = centers[0][0]
            last_milestone = centers[-1][0]

            if needs_cell_guided:
                # Connect the ends using cell-guided arc points.
                mask1 = adata.obs[milestone_key].astype(str) == last_milestone
                mask2 = adata.obs[milestone_key].astype(str) == first_milestone

                n_guides = 4 if needs_high_density else 2
                guide_points = calculate_arc_guide_points(
                    x_embedding, mask1, mask2,
                    center_coords[-1], center_coords[0],
                    n_guide_points=n_guides
                )
                for guide_point in guide_points:
                    curve_points = np.vstack([curve_points, guide_point])

            elif needs_high_density:
                # Insert 4 interpolation points between the last and first milestone centers.
                last_center = center_coords[-1]
                first_center = center_coords[0]
                for j in range(1, 5):
                    interp_point = last_center + j * (first_center - last_center) / 5
                    curve_points = np.vstack([curve_points, interp_point])

            curve_points = np.vstack([curve_points, curve_points[0]])

        # ===== Optional: GT-velocity enhancement =====
        # If enabled, lightly adjust control points along the mean GT velocity direction.
        if use_gt_velocity_enhancement and gt_velocity_embedding is not None:
            # Small adjustment near each milestone center.
            center_milestones = [c[0] for c in centers]
            enhanced_curve_points = []

            for i, point in enumerate(curve_points):
                enhanced_curve_points.append(point)

                # Identify the associated milestone (by index along centers).
                if i < len(center_milestones):
                    milestone = center_milestones[min(i, len(center_milestones)-1)]
                    mask = adata.obs[milestone_key].astype(str) == milestone

                    if mask.sum() > 0:
                        # Mean ground-truth velocity for this milestone.
                        avg_gt_velocity = np.mean(gt_velocity_embedding[mask], axis=0)
                        velocity_norm = np.linalg.norm(avg_gt_velocity)

                        # Apply a small adjustment if velocity is informative.
                        if velocity_norm > 1e-6 and i < len(curve_points) - 1:
                            direction = avg_gt_velocity / velocity_norm
                            # Small adjustment (10%).
                            adjustment = 0.1 * velocity_norm * direction
                            adjusted_point = point + adjustment
                            # Constrain the adjustment within a reasonable range.
                            if np.linalg.norm(adjusted_point - point) < np.linalg.norm(curve_points[i+1] - point) * 0.3:
                                enhanced_curve_points[-1] = adjusted_point

            curve_points = np.array(enhanced_curve_points)

        # Fit spline curve (match calculate_reference_directions_from_trajectory).
        num_original_milestones = len(center_coords)
        k_ideal = get_spline_degree_for_segment(
            topology_type, segment_idx, num_original_milestones, is_cycle
        )
        # Ensure k <= (#control_points - 1).
        k = min(k_ideal, len(curve_points) - 1)

        try:
            # Dynamic smoothing parameter
            s = calculate_smoothing_parameter(curve_points, topology_type, is_cycle, k)

            # pylint: disable=unbalanced-tuple-unpacking
            tck, _ = splprep(
                [curve_points[:, 0], curve_points[:, 1]],
                s=s,
                k=k,
                per=is_cycle
            )

            # Generate dense points on the fitted curve
            u_fine = np.linspace(0, 1, 500)
            curve_smooth = np.array(splev(u_fine, tck)).T

            # Plot curve (solid line)
            ax.plot(
                curve_smooth[:, 0], curve_smooth[:, 1],
                color='black',
                linestyle='-',  # solid line
                linewidth=2.5,
                label=f'Segment {segment_idx+1}',
                zorder=5,
                alpha=0.8
            )

        except Exception:
            # Fallback: connect guide points directly if spline fitting fails.
            if is_cycle:
                curve_points_plot = np.vstack([curve_points, curve_points[0]])
            else:
                curve_points_plot = curve_points

            ax.plot(
                curve_points_plot[:, 0], curve_points_plot[:, 1],
                color='black',
                linestyle='-',
                linewidth=2.5,
                label=f'Segment {segment_idx+1}',
                zorder=5,
                alpha=0.8
            )

    # Title
    display_topology = get_display_topology_type(topology_type)
    title = f"{tool} - {dataset_name}\nReference Trajectory Curves ({display_topology})"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Clean axes
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend
    handles, labels = ax.get_legend_handles_labels()

    # Place legend outside on the right
    ax.legend(
        handles, labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        framealpha=0.9,
        ncol=1
    )

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()


def analyze_simulation_velocity_consistency(
    adata: AnnData,
    base_dir: str,
    tool: str,
    dataset: str,
    topology_type: Optional[str] = None,
    velocity_key: str = "velocity",
    milestone_key: str = "milestone",
    n_jobs: int = 1,
    ground_truth_velocity_key: str = "ground_truth_velocity",
    benchmark_csv_path: Optional[str] = None,
    use_gt_velocity_enhancement: bool = False,
    plot_reference_curves: bool = False,
    reference_curves_dir: Optional[str] = None,
    npz_base_dir: Optional[str] = None
) -> AnalysisResult:
    """
    Analyze velocity direction consistency for a simulated dataset.

    Pipeline:
    1) Define differentiation paths using milestones and topology type.
    2) Fit reference trajectory curve(s) in a low-dimensional embedding.
    3) For each cell, compute the tangent direction at its closest point on the curve.
    4) Compare predicted velocity direction against the reference direction via cosine/angle.

    Args:
        adata: Input AnnData.
        base_dir: Output root directory.
        tool: Tool/method name used in output paths.
        dataset: Dataset identifier/name (normalized internally).
        topology_type: Topology type. If None, inferred from dataset / milestones.
        velocity_key: Predicted velocity key in layers/obsm.
        milestone_key: Milestone column name in adata.obs.
        n_jobs: Parallel jobs for scvelo.velocity_graph when low-dim velocities need to be computed.
        ground_truth_velocity_key: Ground-truth velocity key (optional enhancement).
        benchmark_csv_path: If provided, write unified benchmark.csv (long format).
        use_gt_velocity_enhancement: Whether to apply optional GT-velocity-based guide-point tweaks.
        plot_reference_curves: Whether to save a reference-curve QA plot.
        reference_curves_dir: Output directory for reference-curve plot.
        npz_base_dir: GT npz base directory (passed through; used by GT restoration helpers).
    """

    dataset_name = normalize_dataset_name(dataset)

    method_dir = os.path.join(base_dir, tool)
    pdf_dir = os.path.join(method_dir, "pdf")
    png_dir = os.path.join(method_dir, "png")

    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    try:
        if milestone_key not in adata.obs.columns:
            raise KeyError(f"Milestone key '{milestone_key}' not found in adata.obs")

        has_high_dim_velocity = velocity_key in adata.layers
        has_low_dim_velocity = f"{velocity_key}_dimred" in adata.obsm or f"{velocity_key}_umap" in adata.obsm

        if not has_high_dim_velocity and not has_low_dim_velocity:
            raise ValueError(
                f"Predicted velocity '{velocity_key}' is missing in both adata.layers and adata.obsm "
                f"({velocity_key}_dimred / {velocity_key}_umap)."
            )

        if topology_type is None:
            topology_type = infer_topology_from_dataset_name(
                dataset_name=dataset_name,
                adata=adata,
                milestone_key=milestone_key
            )

        if 'X_dimred' not in adata.obsm and 'X_umap' not in adata.obsm:
            raise ValueError("Neither X_dimred nor X_umap is available in adata.obsm")

        basis = determine_embedding_basis(adata, velocity_key)

        if has_high_dim_velocity:
            ensure_velocity_embedding_sim(adata, velocity_key, basis, n_jobs=n_jobs)
        else:
            velocity_embedding_key = f"{velocity_key}_{basis}"
            if velocity_embedding_key not in adata.obsm:
                raise ValueError(
                    f"Low-dimensional velocity result expected but '{velocity_embedding_key}' is missing in adata.obsm"
                )

        if use_gt_velocity_enhancement:
            gt_velocity_dimred_key = f"{ground_truth_velocity_key}_dimred"
            gt_velocity_basis_key = f"{ground_truth_velocity_key}_{basis}"

            if gt_velocity_dimred_key in adata.obsm:
                if basis != 'dimred' and gt_velocity_basis_key not in adata.obsm:
                    adata.obsm[gt_velocity_basis_key] = adata.obsm[gt_velocity_dimred_key]
            elif ground_truth_velocity_key in adata.layers:
                ensure_velocity_embedding_sim(adata, ground_truth_velocity_key, basis, n_jobs=n_jobs)
            else:
                # No GT velocity available (neither low-dim nor high-dim); disable enhancement.
                use_gt_velocity_enhancement = False

        # Normalize milestone names
        standardize_milestone_column(adata, milestone_key)

        # Expected milestone list for this topology
        expected_milestones = get_expected_milestones_for_topology(topology_type, dataset_name)

        # Available milestones in data
        available_milestones = adata.obs[milestone_key].unique()

        # Validate required milestones
        validated_milestones = validate_milestones_for_simulation(
            expected_milestones, available_milestones, dataset_name
        )

        # Ensure milestone order (categorical)
        ensure_milestone_order(adata, milestone_key, validated_milestones)

        # Re-order for stable coloring/plotting
        ordered_valid_milestones_for_plot = [ms for ms in expected_milestones if ms in validated_milestones]

        # ===== Core: reference directions from fitted trajectory curve(s) =====
        reference_directions = calculate_reference_directions_from_trajectory(
            adata=adata,
            milestone_key=milestone_key,
            topology_type=topology_type,
            dataset_name=dataset_name,
            basis=basis,
            use_gt_velocity_enhancement=use_gt_velocity_enhancement,
            ground_truth_velocity_key=ground_truth_velocity_key
        )

        # Predicted velocity in the chosen embedding
        predicted_velocity_embedding_key = f"{velocity_key}_{basis}"

        if predicted_velocity_embedding_key not in adata.obsm:
            raise ValueError(f"Predicted velocity embedding '{predicted_velocity_embedding_key}' not found in adata.obsm")

        predicted_velocities_embedding = adata.obsm[predicted_velocity_embedding_key]

        # Compute angles (in embedding space) against the reference tangent directions.
        valid_cells_mask_series = adata.obs[milestone_key].astype(str).isin(ordered_valid_milestones_for_plot)
        valid_cells_mask = valid_cells_mask_series.values
        angles = np.full(adata.n_obs, np.nan)
        cosine_similarities = np.full(adata.n_obs, np.nan)

        valid_indices = np.where(valid_cells_mask)[0]
        for i in valid_indices:
            pred_vel = predicted_velocities_embedding[i]
            ref_dir = reference_directions[i]

            # Skip zero vectors
            if np.all(pred_vel == 0) or np.all(ref_dir == 0):
                continue

            # Norms
            pred_norm = np.linalg.norm(pred_vel)
            ref_norm = np.linalg.norm(ref_dir)

            if pred_norm < 1e-6 or ref_norm < 1e-6:
                continue

            # Cosine similarity and angle
            cos_angle = np.clip(np.dot(pred_vel, ref_dir) / (pred_norm * ref_norm), -1, 1)
            cosine_similarities[i] = cos_angle
            angles[i] = np.degrees(np.arccos(cos_angle))
        
        # Angle distribution stats (always computed for reproducibility; writing is optional)
        angle_stats = calculate_angle_distribution_stats_simulation(
            angles, adata, milestone_key, ordered_valid_milestones_for_plot, valid_cells_mask
        )

        # Overall 0-60 degree percentage (always computed; writing is optional)
        ratio_0_60 = calculate_total_0_60_ratio(angles, valid_cells_mask)

        # Optional: write unified benchmark.csv (long format) for simulations.
        # The unified pipeline writes benchmark_total.csv separately.
        if benchmark_csv_path:
            df_long = _build_benchmark_long_rows(
                data_type="sim",
                tool=tool,
                dataset=dataset_name,
                group_type="milestone",
                group_to_stats=angle_stats,
            )
            write_benchmark_long(benchmark_csv_path, df_long)

        # Output paths
        normalized_filename = normalize_filename(tool, dataset_name)

        # Use *_angle_consistency suffix for outputs
        pdf_path = os.path.join(pdf_dir, f"{normalized_filename}_angle_consistency.pdf")
        png_path = os.path.join(png_dir, f"{normalized_filename}_angle_consistency.png")
        
        # Rose plot
        plot_angle_rose_diagrams_simulation(
            adata, angles, milestone_key, ordered_valid_milestones_for_plot,
            pdf_path, png_path, tool
        )

        # Optional: reference-curve QA plot
        reference_curve_png_path = None
        if plot_reference_curves:
            # Output directory for QA plot
            if reference_curves_dir is None:
                ref_curves_dir = os.path.join(base_dir, tool, "reference")
            else:
                ref_curves_dir = reference_curves_dir

            os.makedirs(ref_curves_dir, exist_ok=True)

            reference_curve_png_path = os.path.join(
                ref_curves_dir,
                f"{normalized_filename}_reference_curves.png"
            )

            plot_reference_curves_visualization(
                adata=adata,
                milestone_key=milestone_key,
                topology_type=topology_type,
                dataset_name=dataset_name,
                basis=basis,
                png_path=reference_curve_png_path,
                method_name=tool,
                use_gt_velocity_enhancement=use_gt_velocity_enhancement
            )

        # Per-milestone summary stats
        milestone_consistency: Dict[str, float] = {}
        milestone_mean_angle: Dict[str, float] = {}
        
        for milestone in ordered_valid_milestones_for_plot:
            mask = (adata.obs[milestone_key].astype(str) == milestone) & valid_cells_mask
            if np.sum(mask) > 0:
                milestone_consistency[milestone] = float(np.nanmean(cosine_similarities[mask]))
                milestone_mean_angle[milestone] = float(np.nanmean(angles[mask]))
        
        result: Dict[str, Any] = {
            "status": "success",
            "dataset_name": dataset_name,
            "normalized_filename": normalized_filename,
            "topology_type": get_display_topology_type(topology_type),
            "embedding_basis": basis,
            "rose_pdf_path": pdf_path,
            "rose_png_path": png_path,
            "milestone_consistency": milestone_consistency,
            "milestone_mean_angle": milestone_mean_angle,
            "n_valid_milestones": len(ordered_valid_milestones_for_plot),
            "valid_milestones": ordered_valid_milestones_for_plot,
            "total_ratio_0_60": float(ratio_0_60) if ratio_0_60 == ratio_0_60 else np.nan,
            "angle_distribution_stats": angle_stats,
        }

        if reference_curve_png_path is not None:
            result["reference_curve_png_path"] = reference_curve_png_path

        return result

    except Exception as e:
        error_msg = f"Error while analyzing dataset '{dataset_name}': {str(e)}"
        raise ValueError(error_msg) from e

# ============================================================================
# Ground Truth Velocity Helper Functions
# ============================================================================

# Ground-truth npz base directory resolution.
#
# Recommended (most explicit) ways to configure:
# - CLI flag: --npz-base-dir /path/to/Simdata-GTkey
# - Env var: SIM_ROSEPLOT_NPZ_BASE_DIR=/path/to/Simdata-GTkey
#
# Convenience default for GitHub reproducibility:
# If neither is provided, we will look for a folder named "Simdata-GTkey"
# in the current working directory and next to this script file.
DEFAULT_GTKEY_DIRNAME = "Simdata-GTkey"


def resolve_npz_base_dir(npz_base_dir: Optional[str]) -> Optional[str]:
    """Resolve the root directory that contains <topology>/*_gt_data.npz files."""
    if npz_base_dir:
        return npz_base_dir

    env_dir = os.environ.get("SIM_ROSEPLOT_NPZ_BASE_DIR") or os.environ.get("SIM_ROSEPLOT_GTKEY_DIR")
    if env_dir:
        return env_dir

    candidates = [
        os.path.join(os.getcwd(), DEFAULT_GTKEY_DIRNAME),
        os.path.join(os.path.dirname(__file__), DEFAULT_GTKEY_DIRNAME),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    return None

_unified_logger = None


def get_unified_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Return the shared error logger (file-backed)."""
    global _unified_logger

    if _unified_logger is None:
        _unified_logger = logging.getLogger('unified_errors')
        _unified_logger.setLevel(logging.ERROR)
        _unified_logger.propagate = False

        if not _unified_logger.handlers:
            if log_file is None:
                log_file = 'unified_errors.log'

            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            handler.setLevel(logging.ERROR)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            _unified_logger.addHandler(handler)

    return _unified_logger


def set_unified_log_file(log_file: str) -> None:
    """Set the output file used by the shared error logger."""
    global _unified_logger

    if _unified_logger is not None:
        for handler in _unified_logger.handlers[:]:
            _unified_logger.removeHandler(handler)
            handler.close()
        _unified_logger = None

    get_unified_logger(log_file)


def infer_topology_from_dataset(dataset: str) -> Optional[str]:
    """Infer topology name from dataset for locating the GT npz subdirectory."""
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

    if dataset in SIMULATION_ADD_DATASETS:
        return 'simulation-add'

    normalized_id = dataset.lower()

    topology_patterns = [
        ('consecutive-bifurcating', 'consecutive-bifurcating'),
        ('consecutive_bifurcating', 'consecutive-bifurcating'),
        ('genesub-bifurcating', 'genesub-bifurcating'),
        ('genesub_bifurcating', 'genesub-bifurcating'),
        ('cellsub-bifurcating', 'cellsub-bifurcating'),
        ('cellsub_bifurcating', 'cellsub-bifurcating'),
        ('bifurcating-loop', 'bifurcating-loop'),
        ('bifurcating_loop', 'bifurcating-loop'),
        ('linear-simple', 'linear-simple'),
        ('linear_simple', 'linear-simple'),
        ('cycle-simple', 'cycle-simple'),
        ('cycle_simple', 'cycle-simple'),
        ('linear-bifurcating', 'linear-bifurcating'),
        ('linear_bifurcating', 'linear-bifurcating'),
        ('linear-linear', 'linear-linear'),
        ('linear_linear', 'linear-linear'),
        ('bifurcating', 'bifurcating'),
        ('trifurcating', 'trifurcating'),
        ('disconnected', 'disconnected'),
    ]

    for pattern, topology in topology_patterns:
        if pattern in normalized_id:
            return topology

    return None


def locate_npz_file(dataset: str, npz_base_dir: str) -> Optional[str]:
    """Locate the GT npz file path for a given dataset."""
    topology = infer_topology_from_dataset(dataset)
    if topology is None:
        return None

    # Try the original dataset
    npz_path = os.path.join(npz_base_dir, topology, f"{dataset}_gt_data.npz")
    if os.path.exists(npz_path):
        return npz_path

    # Try underscore -> hyphen
    alt_id = dataset.replace('_', '-')
    alt_path = os.path.join(npz_base_dir, topology, f"{alt_id}_gt_data.npz")
    if os.path.exists(alt_path):
        return alt_path

    # Try hyphen -> underscore
    alt_id2 = dataset.replace('-', '_')
    alt_path2 = os.path.join(npz_base_dir, topology, f"{alt_id2}_gt_data.npz")
    if os.path.exists(alt_path2):
        return alt_path2

    return None


def normalize_cell_name(cell_name: str, adjust_numeric_index: bool = False) -> str:
    """Normalize cell name to ensure it starts with 'cell'."""
    if not cell_name.lower().startswith('cell'):
        if adjust_numeric_index and cell_name.isdigit():
            return f"cell{int(cell_name) + 1}"
        else:
            return f"cell{cell_name}"
    return cell_name


def match_cell_indices(
    adata_cell_names: np.ndarray,
    npz_cell_names: np.ndarray,
    npz_cell_names_unique: Optional[np.ndarray] = None,
    allow_partial: bool = True,
    min_match_ratio: float = 0.95,
    method_name: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Match adata cell names to npz cell indices."""
    adjust_numeric_index = method_name == 'STT'

    normalized_adata_names = np.array([
        normalize_cell_name(str(n), adjust_numeric_index=adjust_numeric_index)
        for n in adata_cell_names
    ])

    npz_name_to_idx = {str(name): idx for idx, name in enumerate(npz_cell_names)}

    adata_indices = []
    npz_indices = []

    for adata_idx, name in enumerate(normalized_adata_names):
        if name in npz_name_to_idx:
            adata_indices.append(adata_idx)
            npz_indices.append(npz_name_to_idx[name])

    matched_count = len(adata_indices)
    total_count = len(normalized_adata_names)

    if matched_count == total_count:
        return np.arange(total_count), np.array(npz_indices), "Exact match via cell_names"

    # Try cell_names_unique
    if npz_cell_names_unique is not None:
        npz_unique_to_idx = {str(name): idx for idx, name in enumerate(npz_cell_names_unique)}

        adata_indices_unique = []
        npz_indices_unique = []

        for adata_idx, name in enumerate(normalized_adata_names):
            if name in npz_unique_to_idx:
                adata_indices_unique.append(adata_idx)
                npz_indices_unique.append(npz_unique_to_idx[name])

        matched_count_unique = len(adata_indices_unique)

        if matched_count_unique == total_count:
            return np.arange(total_count), np.array(npz_indices_unique), "Exact match via cell_names_unique"

        if matched_count_unique > matched_count:
            adata_indices = adata_indices_unique
            npz_indices = npz_indices_unique
            matched_count = matched_count_unique

    if matched_count > 0:
        match_ratio = matched_count / total_count

        if match_ratio >= min_match_ratio and allow_partial:
            return (
                np.array(adata_indices),
                np.array(npz_indices),
                f"Partial match ({matched_count}/{total_count}={match_ratio:.1%}); subsetting to matched cells"
            )
        else:
            return (
                None,
                None,
                f"Match ratio too low ({matched_count}/{total_count}={match_ratio:.1%}; requires >= {min_match_ratio:.0%})"
            )

    return None, None, "No matching cell names"


def check_dimred_consistency(
    adata_dimred: np.ndarray,
    npz_dimred_subset: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> bool:
    """Check whether two embeddings are numerically consistent."""
    if adata_dimred.shape != npz_dimred_subset.shape:
        return False
    return np.allclose(adata_dimred, npz_dimred_subset, rtol=rtol, atol=atol)


def add_ground_truth_velocity_dimred(
    adata,
    dataset_id: str,
    method_name: Optional[str] = None,
    velocity_key: str = "velocity",
    raise_on_failure: bool = True,
    allow_partial_match: bool = True,
    min_match_ratio: float = 0.95,
    npz_base_dir: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Restore unified ground-truth (GT) velocity embedding from a *_gt_data.npz file.

    Steps:
    1) Locate and load the npz file.
    2) Match cell indices between AnnData and npz.
    3) Add gt_dimred into adata.obsm['ground_truth_velocity_dimred'].
    4) Replace X_dimred with the unified coordinate system (except for PhyloVelo).
    5) If X_dimred mismatches and a high-dimensional velocity exists, drop old velocity embeddings
       so they can be recomputed later under the unified embedding.
    """
    method_prefix = f"{method_name}_" if method_name else ""

    npz_base_dir = resolve_npz_base_dir(npz_base_dir)
    if not npz_base_dir:
        msg = (
            f"{method_prefix}{dataset_id}: GT npz base directory is not provided.\n"
            "Please pass --npz-base-dir, set SIM_ROSEPLOT_NPZ_BASE_DIR, or place a 'Simdata-GTkey' folder\n"
            "in the current working directory or next to this script."
        )
        if raise_on_failure:
            raise ValueError(msg)
        return False, msg

    # Locate npz file
    npz_path = locate_npz_file(dataset_id, npz_base_dir)
    if npz_path is None:
        msg = f"{method_prefix}{dataset_id}: GT npz file not found"
        if raise_on_failure:
            raise ValueError(msg)
        return False, msg

    # Read npz file
    try:
        npz_data = np.load(npz_path, allow_pickle=True)
        gt_dimred = npz_data['gt_dimred']
        X_dimred_npz = npz_data['X_dimred']
        cell_names = npz_data['cell_names']
        cell_names_unique = npz_data.get('cell_names_unique', None)
    except Exception as e:
        msg = f"{method_prefix}{dataset_id}: Failed to read GT npz file - {str(e)}"
        if raise_on_failure:
            raise ValueError(msg) from e
        return False, msg

    # Convert categorical columns to strings to avoid obs_names_make_unique issues
    for col in adata.obs.columns:
        if isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
            adata.obs[col] = adata.obs[col].astype(str)

    # Convert obs index if categorical
    if isinstance(adata.obs.index.dtype, pd.CategoricalDtype):
        adata.obs.index = adata.obs.index.astype(str)

    adata.obs_names_make_unique()

    # Match cell indices
    adata_cell_names = np.array(adata.obs_names)
    adata_indices, npz_indices, match_msg = match_cell_indices(
        adata_cell_names, cell_names, cell_names_unique,
        allow_partial=allow_partial_match,
        min_match_ratio=min_match_ratio,
        method_name=method_name
    )

    # If name matching fails, try position-based matching (only when sizes are identical)
    if adata_indices is None:
        if len(adata_cell_names) == len(cell_names):
            existing_dimred = adata.obsm.get('X_dimred') or adata.obsm.get('X_umap')

            if existing_dimred is not None and check_dimred_consistency(existing_dimred, X_dimred_npz):
                adata_indices = np.arange(len(adata_cell_names))
                npz_indices = np.arange(len(cell_names))
                match_msg = "Position-based match (different cell names but identical X_dimred)"
            else:
                error_logger = get_unified_logger()
                error_logger.error(
                    f"Cell match failed | {method_prefix}{dataset_id} | {match_msg} | "
                    f"n_adata={len(adata_cell_names)}, n_npz={len(cell_names)}"
                )
                msg = f"{method_prefix}{dataset_id}: Cell matching failed - {match_msg}"
                if raise_on_failure:
                    raise ValueError(msg)
                return False, msg
        else:
            error_logger = get_unified_logger()
            error_logger.error(
                f"Cell match failed | {method_prefix}{dataset_id} | {match_msg} | "
                f"n_adata={len(adata_cell_names)}, n_npz={len(cell_names)}"
            )
            msg = f"{method_prefix}{dataset_id}: Cell matching failed - {match_msg}"
            if raise_on_failure:
                raise ValueError(msg)
            return False, msg

    # For partial matches, subset AnnData
    if len(adata_indices) < len(adata_cell_names):
        error_logger = get_unified_logger()
        error_logger.error(
            f"Partial cell match | {method_prefix}{dataset_id} | {match_msg} | "
            f"subsetting AnnData to matched cells (n_matched={len(adata_indices)}, n_total={len(adata_cell_names)})"
        )
        adata._inplace_subset_obs(adata_indices)

    # Subset npz arrays
    gt_dimred_subset = gt_dimred[npz_indices]
    X_dimred_subset = X_dimred_npz[npz_indices]

    # Add gt_dimred
    adata.obsm['ground_truth_velocity_dimred'] = gt_dimred_subset

    # PhyloVelo special case: keep existing obsm (only low-dim results available)
    if method_name == 'PhyloVelo':
        if 'X_dimred' not in adata.obsm and 'X_umap' in adata.obsm:
            adata.obsm['X_dimred'] = adata.obsm['X_umap'].copy()
        return True, f"{method_prefix}{dataset_id}: Added gt_dimred; kept original obsm ({match_msg})"

    # Other methods: replace X_dimred with the unified coordinate system
    adata.obsm['X_dimred_ori'] = X_dimred_subset

    existing_dimred_key = 'X_dimred' if 'X_dimred' in adata.obsm else ('X_umap' if 'X_umap' in adata.obsm else None)
    existing_dimred = adata.obsm.get(existing_dimred_key) if existing_dimred_key else None

    dimred_mismatch = False
    vkey_dimred = f"{velocity_key}_dimred"
    vkey_umap = f"{velocity_key}_umap"

    if existing_dimred is not None and not check_dimred_consistency(existing_dimred, X_dimred_subset):
        dimred_mismatch = True

        has_vkey_dimred = vkey_dimred in adata.obsm
        has_vkey_umap = vkey_umap in adata.obsm

        if has_vkey_dimred or has_vkey_umap:
            error_logger = get_unified_logger()
            error_logger.error(
                f"X_dimred mismatch | {method_prefix}{dataset_id} | "
                f"existing velocity embedding: {vkey_dimred}={has_vkey_dimred}, {vkey_umap}={has_vkey_umap}"
            )

        # If high-dimensional velocity exists, drop old embeddings so they can be recomputed
        if velocity_key in adata.layers:
            if has_vkey_dimred:
                del adata.obsm[vkey_dimred]
            if has_vkey_umap:
                del adata.obsm[vkey_umap]

        if 'X_dimred' in adata.obsm:
            del adata.obsm['X_dimred']
        if 'X_umap' in adata.obsm:
            del adata.obsm['X_umap']

    # Rename X_dimred_ori -> X_dimred
    X_dimred_final = adata.obsm['X_dimred_ori'].copy()
    del adata.obsm['X_dimred_ori']
    adata.obsm['X_dimred'] = X_dimred_final

    if dimred_mismatch:
        return True, f"{method_prefix}{dataset_id}: Added gt_dimred; replaced X_dimred"
    else:
        return True, f"{method_prefix}{dataset_id}: Added gt_dimred ({match_msg})"


# ============================================================================
# Tool-Specific Preprocessing Functions
# ============================================================================

def preprocess_adata_for_method(
    adata: AnnData,
    tool: str,
    velocity_key: str,
    dataset: str
) -> bool:
    """
    Method-specific preprocessing (phase 1: before unifying X_dimred).

    Returns:
        bool: Whether preprocessing succeeded.
    """
    try:
        if tool == 'PhyloVelo':
            if 'phylovelo_velocity' in adata.obsm:
                adata.obsm['phylovelo_velocity_dimred'] = adata.obsm['phylovelo_velocity']
            else:
                error_logger = get_unified_logger()
                error_logger.error(
                    f"Preprocess failed | {tool}_{dataset} | missing obsm['phylovelo_velocity']"
                )
                return False

        elif tool == 'cellDancer':
            if 'clusters' in adata.obs:
                adata.obs['milestone'] = adata.obs['clusters']

        elif tool == 'TopicVelo':
            vkey_exists = (
                velocity_key in adata.layers or
                f"{velocity_key}_dimred" in adata.obsm or
                f"{velocity_key}_umap" in adata.obsm
            )
            if not vkey_exists and 'variance_velocity' in adata.layers:
                adata.layers[velocity_key] = adata.layers['variance_velocity']

        elif tool == 'TFvelo':
            if 'velo_hat' in adata.layers and 'fit_scaling_y' in adata.var:
                adata.layers['velocity'] = adata.layers['velo_hat'] / np.expand_dims(
                    adata.var['fit_scaling_y'], 0
                )

        return True

    except Exception as e:
        error_logger = get_unified_logger()
        error_logger.error(f"Preprocess error | {tool}_{dataset} | {str(e)}")
        return False


def compute_velocity_embeddings_for_method(
    adata: AnnData,
    tool: str,
    velocity_key: str,
    dataset: str,
    n_jobs: int = 1,
) -> bool:
    """
    Method-specific velocity embedding computation (phase 3: after unifying X_dimred).

    Returns:
        bool: Whether computation succeeded.
    """
    try:
        if tool == 'TFvelo' and 'velocity' in adata.layers:
            try:
                scv.tl.velocity_graph(adata, vkey='velocity', xkey='M_total', n_jobs=int(n_jobs))
            except Exception as e:
                if "neighbor graph" in str(e):
                    # Fix corrupted/missing neighbor graph
                    for col in adata.obs.columns:
                        if isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
                            adata.obs[col] = adata.obs[col].astype(str)
                    if isinstance(adata.obs.index.dtype, pd.CategoricalDtype):
                        adata.obs.index = adata.obs.index.astype(str)
                    adata.obs_names_make_unique()
                    duplicated_mask = adata.to_df().duplicated()
                    if duplicated_mask.any():
                        adata._inplace_subset_obs(~duplicated_mask)
                    sc.pp.neighbors(adata)
                    scv.tl.velocity_graph(adata, vkey='velocity', xkey='M_total', n_jobs=int(n_jobs))
                else:
                    raise
            scv.tl.velocity_embedding(adata, basis='dimred', vkey='velocity')

        elif tool == 'SDEvelo' and 'sde_velocity' in adata.layers:
            try:
                scv.tl.velocity_graph(adata, vkey='sde_velocity', xkey='Ms', n_jobs=int(n_jobs))
                scv.tl.velocity_embedding(adata, basis='dimred', vkey='sde_velocity')
            except Exception as e:
                if "neighbor graph" in str(e):
                    # Fix corrupted/missing neighbor graph
                    for col in adata.obs.columns:
                        if isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
                            adata.obs[col] = adata.obs[col].astype(str)
                    if isinstance(adata.obs.index.dtype, pd.CategoricalDtype):
                        adata.obs.index = adata.obs.index.astype(str)
                    adata.obs_names_make_unique()
                    duplicated_mask = adata.to_df().duplicated()
                    if duplicated_mask.any():
                        adata._inplace_subset_obs(~duplicated_mask)
                    sc.pp.neighbors(adata)
                    scv.tl.velocity_graph(adata, vkey='sde_velocity', xkey='Ms', n_jobs=int(n_jobs))
                    scv.tl.velocity_embedding(adata, basis='dimred', vkey='sde_velocity')
                else:
                    raise

        return True

    except Exception as e:
        error_logger = get_unified_logger()
        error_logger.error(f"Embedding computation error | {tool}_{dataset} | {str(e)}")
        return False


if __name__ == "__main__":
  raise SystemExit(main())
