#!/usr/bin/env python3
"""
GRAVITY simulated-data training driver (consumes ``combine.csv``).

NOTE: Only used for simulated data inside the Docker image.

This script runs the GRAVITY deep-learning stages for a list of simulated
samples. It mirrors the logging/timing conventions of the other simulated-data
drivers in this repository:

1. Samples are read from a CSV and processed in a loop; samples whose result is
   already present (``<output-dir>/<ID>/gravity_result.csv``) are skipped.
2. Per-sample outputs live in ``<output-dir>/<ID>/``:
       cell_type_u_s.csv     long table (intermediate)
       gravity_result.csv    final velocity result (skip marker)
       cpu_mem_log.txt
       killed/               "running" marker
       combine.csv / stage1.csv / stage2.csv / *.ckpt
3. Logs live in ``<output-dir>/log_file/``:
       log_time.txt  log_resources.txt  log_error.txt
       log_total_resources.txt  log_run_stdout.txt
4. Caches / GPU memory are cleared before each sample.
5. Per-stage timings are written to ``log_time.txt``:
       Time elapsed              training start -> velocity obtained (excludes IO)
       Preprocessing time        read + preprocess + export long table
       Parameter generation time GRAVITY two-stage training
       Velocity computation time cell velocity computation

Simulated-data notes:
  - Input h5ad files (from dyngen / scmultisim) contain ``obsm['X_dimred']``,
    ``obs['milestone']`` and ``layers['spliced'/'unspliced']``.
  - Simulated data has no species information, so the prior network is always
    ``None`` (no NicheNet prior).
  - No CSV -> h5ad conversion is done here (that belongs to the real-data path).

Usage:
    python gravity_sim_from_combine.py --input samples.csv --output-dir ./out --simulate
    python gravity_sim_from_combine.py --input samples.csv --output-dir ./out --simulate --from-combine
"""

import argparse
import datetime
import gc
import multiprocessing
import os
import sys
import time
import traceback

# Must be set before importing torch / gravity.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# --- Compatibility patch: scVelo 0.2.5 relies on matplotlib.cbook.mplDeprecation,
#     which was removed in matplotlib >= 3.7. Add it before importing gravity.
import matplotlib

matplotlib.use("Agg")
import matplotlib.cbook as _cbook

if not hasattr(_cbook, "mplDeprecation"):
    _cbook.mplDeprecation = DeprecationWarning
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import psutil
import pynvml
import scanpy as sc
import scvelo as scv
import torch

from gravity import PipelineConfig, run_pipeline
from gravity.data.preprocessing import adata_to_df_with_embed
from gravity.velocity import compute_cell_velocity_

_LOG = {
    "dir": None,
    "time": None,
    "performance": None,
    "error": None,
    "total": None,
    "stdout": None,
}


def setup_logging(log_dir):
    """Initialize the log directory and redirect stdout/stderr to log_run_stdout.txt."""
    os.makedirs(log_dir, exist_ok=True)
    _LOG.update({
        "dir": log_dir,
        "time": os.path.join(log_dir, "log_time.txt"),
        "performance": os.path.join(log_dir, "log_resources.txt"),
        "error": os.path.join(log_dir, "log_error.txt"),
        "total": os.path.join(log_dir, "log_total_resources.txt"),
        "stdout": os.path.join(log_dir, "log_run_stdout.txt"),
    })
    handle = open(_LOG["stdout"], "a", buffering=1, encoding="utf-8")
    sys.stdout = handle
    sys.stderr = handle
    return log_dir


def write_logs(log_data, log_type):
    timestamp = log_data.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if log_type == "time":
        with open(_LOG["time"], "a") as handle:
            handle.write(f'[{timestamp}] {log_data["message"]}\n')
    elif log_type == "performance":
        with open(_LOG["performance"], "a") as handle:
            handle.write(f'[{timestamp}] {log_data["message"]}\n')
    elif log_type == "error":
        with open(_LOG["error"], "a") as handle:
            handle.write(f'[{timestamp}] Error: {log_data["message"]}\n')
    elif log_type == "total":
        pd.DataFrame([log_data]).to_csv(
            _LOG["total"], mode="a", header=not os.path.exists(_LOG["total"]), index=False, sep="\t")


def monitor_resources(pid, interval, shared_list):
    """Monitor CPU/memory usage of a process and append samples to a shared list."""
    process = psutil.Process(pid)
    num_cpus = psutil.cpu_count()
    process.cpu_percent(interval=None)
    while True:
        try:
            raw_cpu = process.cpu_percent(interval=interval)
            norm_cpu = raw_cpu / num_cpus
            mem = process.memory_percent()
            shared_list.append((time.time(), raw_cpu, norm_cpu, mem))
            time.sleep(interval)
        except psutil.NoSuchProcess:
            break


def monitor_gpu_memory(device_ids, interval, log_list):
    """Monitor GPU memory usage (records the busiest visible GPU)."""
    try:
        pynvml.nvmlInit()
        if device_ids is None:
            device_ids = list(range(pynvml.nvmlDeviceGetCount()))
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in device_ids]
        while True:
            used = 0
            for handle in handles:
                used = max(used, pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            log_list.append(used)
            time.sleep(interval)
    except Exception as exc:
        print(f"GPU monitoring error: {exc}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def clear_memory_intensive():
    """Aggressively free caches / GPU memory (called before each sample)."""
    collected = gc.collect()
    print(f"Garbage collector cleared {collected} objects")

    try:
        np._globals._clear_cache()
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        plt.close("all")
    except Exception:
        pass


def _csv_is_complete(csv_path):
    """Check that the intermediate long table CSV is complete."""
    required = {"gene_name", "unsplice", "splice", "cellID", "embedding1", "embedding2"}
    try:
        with open(csv_path) as handle:
            header = handle.readline().strip().split(",")
        cols = set(header)
        if not required.issubset(cols):
            print(f"[WARN] incomplete intermediate CSV (missing: {required - cols}), re-exporting.")
            return False
        return True
    except Exception as exc:
        print(f"[WARN] could not check CSV ({exc}), re-exporting.")
        return False


def long_table_from_combine(combine_csv, out_csv):
    """Rebuild the long table (``cell_type_u_s.csv``) from the wide ``combine.csv``.

    ``combine.csv`` columns: cellIndex, embedding1, embedding2, gene_u_s_<gene>...
    Each ``gene_u_s_<gene>`` cell is a ``'(gene_name, unsplice, splice)'`` tuple
    string. The long table columns must match ``adata_to_df_with_embed`` output
    order because ``PreprocessDataset`` depends on that order.
    """
    import ast

    df = pd.read_csv(combine_csv)
    gene_cols = [c for c in df.columns if c.startswith("gene_u_s_")]
    if not gene_cols:
        raise ValueError(f"combine.csv has no gene_u_s_ columns: {combine_csv}")

    long_df = df.melt(
        id_vars=["cellIndex", "embedding1", "embedding2"],
        value_vars=gene_cols,
        var_name="_gene_col",
        value_name="_triplet",
    )
    parsed = long_df["_triplet"].astype(str).map(ast.literal_eval)
    long_df["gene_name"] = parsed.map(lambda t: t[0])
    long_df["unsplice"] = parsed.map(lambda t: t[1])
    long_df["splice"] = parsed.map(lambda t: t[2])
    long_df["cellID"] = "cell_" + long_df["cellIndex"].astype(str)
    long_df["clusters"] = "unknown"   # non-NaN placeholder so pandas does not parse it as NaN

    long_df = long_df[
        ["gene_name", "unsplice", "splice", "cellID", "clusters", "embedding1", "embedding2"]
    ]
    long_df.to_csv(out_csv, index=False)
    return long_df


def pick_celltype_key(adata):
    """Cluster column name for simulated data (dyngen/scmultisim use ``milestone``)."""
    for key in ("milestone", "vis_annotation", "celltype", "cell_type"):
        if key in adata.obs:
            return key
    raise KeyError(f"No cluster column found in adata.obs; actual columns: {list(adata.obs.columns)}")


def preprocess_adata(adata, data_id, n_top_genes=2000, n_pcs=30, n_neighbors=30):
    """Preprocess simulated data: X_dimred -> X_umap, then filter/normalize + moments."""
    adata.obs_names_make_unique()

    if "X_dimred" in adata.obsm:
        adata.obsm["X_umap"] = adata.obsm["X_dimred"].copy()
    elif "X_umap" not in adata.obsm:
        try:
            sc.tl.pca(adata)
            sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
            sc.tl.umap(adata)
        except Exception as exc:
            print(f"[WARN] {data_id} could not generate a 2D embedding: {exc}")

    if adata.n_vars < 10000:
        top_gene = (adata.n_vars // 500) * 500
        top_gene = min(top_gene, adata.n_vars, 500)
    else:
        top_gene = n_top_genes

    scv.pp.filter_and_normalize(adata, min_shared_counts=None, n_top_genes=top_gene)
    print(f"[INFO] {data_id} after filter_and_normalize: {adata.n_vars} genes", flush=True)

    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors, method="umap")
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    print(f"[INFO] {data_id} after moments: {adata.n_vars} genes", flush=True)
    return adata


def process_one(data_id, data_path, save_dir,
                stage1_epochs=6, stage2_epochs=6, stage1_lr=1e-6, stage2_lr=1e-4,
                batch_size=16, num_workers=8, force_rerun=False, from_combine=False):
    """Process one simulated sample. Returns (status, message) with status in skip/ok/error.

    With ``from_combine=True`` training starts directly from ``<save_dir>/combine.csv``,
    skipping h5ad reading / preprocessing / long-table export; the long table is
    rebuilt if it is missing.
    """
    os.makedirs(save_dir, exist_ok=True)
    result_csv = os.path.join(save_dir, "gravity_result.csv")

    if os.path.exists(result_csv) and not force_rerun:
        print(f"[skip] {data_id} already has {result_csv}, skipping", flush=True)
        return "skip", None

    combine_csv = os.path.join(save_dir, "combine.csv")
    csv_target = os.path.join(save_dir, "cell_type_u_s.csv")

    use_combine = from_combine and os.path.exists(combine_csv)
    if from_combine and not os.path.exists(combine_csv):
        print(f"[WARN] {data_id}: from_combine requested but {combine_csv} is missing, falling back to h5ad", flush=True)

    if use_combine:
        preprocess_start = time.time()
        if os.path.exists(csv_target) and _csv_is_complete(csv_target):
            print(f"[INFO] long table already exists, reusing: {csv_target}", flush=True)
        else:
            print(f"[INFO] rebuilding long table from combine.csv: {csv_target}", flush=True)
            long_table_from_combine(combine_csv, csv_target)
        csv_path = csv_target
        preprocess_time = time.time() - preprocess_start
        print(f"[{data_id}] from_combine preparation finished in {preprocess_time:.2f} s", flush=True)
    else:
        try:
            adata = sc.read(data_path, cache=True)
        except Exception as exc:
            return "error", f"{data_id} read failed: {exc}"
        if adata.n_vars == 0 or adata.n_obs == 0:
            print(f"{data_id} shape error!", flush=True)
            return "error", f"{data_id} shape error"

        preprocess_start = time.time()
        adata = preprocess_adata(adata, data_id)
        celltype_key = pick_celltype_key(adata)

        if os.path.exists(csv_target) and _csv_is_complete(csv_target):
            print(f"[INFO] intermediate CSV already exists, skipping export: {csv_target}", flush=True)
        else:
            adata_to_df_with_embed(
                adata,
                us_para=["Mu", "Ms"],
                cell_type_para=celltype_key,
                embed_para="X_umap",
                save_path=csv_target,
            )
        csv_path = csv_target
        preprocess_time = time.time() - preprocess_start
        print(f"[{data_id}] preprocessing finished in {preprocess_time:.2f} s", flush=True)

    os.makedirs(os.path.join(save_dir, "killed"), exist_ok=True)   # running marker

    process = psutil.Process(os.getpid())
    start_time = time.time()
    process.cpu_percent(interval=None)
    manager = multiprocessing.Manager()
    cpu_mem_log = manager.list()
    gpu_mem_log = manager.list()
    monitor_proc = multiprocessing.Process(target=monitor_resources,
                                           args=(process.pid, 0.05, cpu_mem_log))
    monitor_proc.start()
    gpu_monitor_proc = multiprocessing.Process(target=monitor_gpu_memory,
                                               args=(None, 0.1, gpu_mem_log))
    gpu_monitor_proc.start()

    print(f"GRAVITY calculation for {data_id}", flush=True)
    print(f"Start time and resource recording for {data_id}", flush=True)

    cfg = PipelineConfig(
        raw_counts=str(csv_path),
        workdir=save_dir,
        prior_network=None,          # simulated data has no species prior
        accelerator="gpu",
        devices=1,
        batch_size=batch_size,
        num_workers=num_workers,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        stage1_lr=stage1_lr,
        stage2_lr=stage2_lr,
    )
    params_start = time.time()
    outputs = run_pipeline(cfg)
    params_time = time.time() - params_start
    print(f"[{data_id}] parameter generation finished in {params_time:.2f} s", flush=True)
    print(f"[INFO] stage2.csv: {outputs['stage2_csv']}", flush=True)

    velocity_start = time.time()
    stage2_df = pd.read_csv(outputs["stage2_csv"])
    try:
        result_df, _ = compute_cell_velocity_(stage2_df)
    except Exception as exc:
        print(f"[WARN] cell velocity computation failed, using raw stage2 result: {exc}", flush=True)
        result_df = stage2_df
    if "loss" not in result_df.columns:
        result_df["loss"] = 0.0

    end_time = time.time()
    elapsed_time = end_time - start_time
    velocity_time = end_time - velocity_start

    result_df.to_csv(result_csv, index=False)
    print(f"[{data_id}] velocity computation finished in {velocity_time:.2f} s", flush=True)
    print(f"[INFO] result CSV: {result_csv}", flush=True)

    monitor_proc.terminate()
    monitor_proc.join()
    gpu_monitor_proc.terminate()
    gpu_monitor_proc.join()

    cpu_norm_list = [norm_cpu for _, _, norm_cpu, _ in cpu_mem_log]
    max_cpu_peak = max(cpu_norm_list) if cpu_norm_list else 0
    max_gpu_mem_bytes = max(gpu_mem_log) if gpu_mem_log else 0
    max_gpu_mem_gb = max_gpu_mem_bytes / (1024 ** 3)

    write_logs({"message": f"File {data_id}: Time elapsed: {elapsed_time:.2f} seconds"}, "time")
    write_logs({"message": f"File {data_id}: Preprocessing time: {preprocess_time:.2f} seconds"}, "time")
    write_logs({"message": f"File {data_id}: Parameter generation time: {params_time:.2f} seconds"}, "time")
    write_logs({"message": f"File {data_id}: Velocity computation time: {velocity_time:.2f} seconds"}, "time")

    process_info = process.memory_info()
    cpu_percent = process.cpu_percent(interval=0.1)
    normalized_cpu_percent = cpu_percent / psutil.cpu_count()
    write_logs(
        {"message": f"PID: {process.pid}, Memory usage: {process_info.rss / (1024 ** 2):.2f} MB, "
                    f"CPU: {normalized_cpu_percent:.2f} %"},
        "performance",
    )
    write_logs({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": data_id,
        "elapsed_time": elapsed_time,
        "PID": process.pid,
        "VIRT_MB": process_info.vms / (1024 ** 2),
        "RES_MB": process_info.rss / (1024 ** 2),
        "CPU_peak": max_cpu_peak,
        "MEM_percent": process.memory_percent(),
        "GPU_peak_GB": max_gpu_mem_gb,
    }, "total")

    with open(os.path.join(save_dir, "cpu_mem_log.txt"), "w") as handle:
        handle.write("Timestamp\tRaw_CPU_percent\tNormalized_CPU_percent\tMemory_percent\n")
        for ts, raw_cpu, norm_cpu, mem_val in cpu_mem_log:
            stamp = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            handle.write(f"{stamp}\t{raw_cpu:.2f}\t{norm_cpu:.2f}\t{mem_val:.2f}\n")

    try:
        os.rmdir(os.path.join(save_dir, "killed"))
    except OSError:
        pass
    return "ok", None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GRAVITY simulated-data training driver (consumes combine.csv)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Sample list CSV with columns 'ID' and 'path'")
    parser.add_argument("--output-dir", required=True, help="Result and log root directory")
    parser.add_argument("--start", type=int, default=None, help="Start row index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End row index (exclusive)")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Confirm simulated-data processing (required; this script is simulated-data only)",
    )
    parser.add_argument("--from-combine", action="store_true",
                        help="Start training directly from <output-dir>/<ID>/combine.csv")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Ignore the 'already finished' check and rerun every sample")
    parser.add_argument("--stage1-epochs", type=int, default=6)
    parser.add_argument("--stage2-epochs", type=int, default=6)
    parser.add_argument("--stage1-lr", type=float, default=1e-6)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2024)
    return parser


def main(args=None):
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()

    if not args.simulate:
        parser.error("gravity_sim_from_combine.py only supports simulated data; pass --simulate")

    np.random.seed(args.seed)

    save_root = args.output_dir
    log_dir = setup_logging(os.path.join(save_root, "log_file"))

    print(f"Sample list: {args.input}", flush=True)
    print(f"Result and log root: {save_root}", flush=True)
    print(f"Log directory: {log_dir}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}", flush=True)

    datalist = pd.read_csv(args.input)

    success_ids = []
    failed_ids = []
    skipped_ids = []

    for i, row in datalist.iterrows():
        if args.start is not None and i < args.start:
            continue
        if args.end is not None and i >= args.end:
            break

        data_id = str(row["ID"])
        data_path = str(row["path"])
        data_file = os.path.basename(data_path)
        save_dir = os.path.join(save_root, data_id)

        if os.path.exists(os.path.join(save_dir, "killed")):
            print(f"[skip] {i}__{data_id} has a running marker {save_dir}/killed, skipping", flush=True)
            skipped_ids.append(data_id)
            continue

        if os.path.exists(os.path.join(save_dir, "gravity_result.csv")) and not args.force_rerun:
            print(f"[skip] {i}__{data_id} already finished, skipping", flush=True)
            skipped_ids.append(data_id)
            clear_memory_intensive()
            continue

        print(f"\n===== [{i}] processing: {data_id} | {data_file} =====", flush=True)
        clear_memory_intensive()
        try:
            status, message = process_one(
                data_id=data_id,
                data_path=data_path,
                save_dir=save_dir,
                stage1_epochs=args.stage1_epochs,
                stage2_epochs=args.stage2_epochs,
                stage1_lr=args.stage1_lr,
                stage2_lr=args.stage2_lr,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                force_rerun=args.force_rerun,
                from_combine=args.from_combine,
            )
            if status == "skip":
                skipped_ids.append(data_id)
            elif status == "error":
                print(f"[error] {data_id}: {message}", flush=True)
                write_logs({"message": f"File {data_id}: {message}"}, "error")
                os.makedirs(os.path.join(save_root, f"error_{data_id}"), exist_ok=True)
                failed_ids.append(data_id)
            else:
                success_ids.append(data_id)
        except Exception as exc:
            failed_ids.append(data_id)
            print("\n" + "!" * 80)
            print(f"ERROR processing data_id: {data_id}")
            print(f"Error type: {type(exc).__name__}")
            print(f"Error message: {exc}")
            print("Traceback:")
            traceback.print_exc()
            print("!" * 80, flush=True)
            write_logs({"message": f"File {data_id}: {type(exc).__name__}: {exc} | "
                                   f"{traceback.format_exc().replace(chr(10), ' ')}"}, "error")
        finally:
            clear_memory_intensive()

    print("\n\n" + "=" * 80)
    print("ALL DATASETS FINISHED")
    print("=" * 80)
    print(f"\nTotal datasets: {len(datalist)}")
    print(f"Successful: {len(success_ids)}")
    print(f"Skipped: {len(skipped_ids)}")
    print(f"Failed: {len(failed_ids)}")


if __name__ == "__main__":
    main()
