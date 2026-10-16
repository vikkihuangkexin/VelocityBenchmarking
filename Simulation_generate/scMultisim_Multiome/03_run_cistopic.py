#!/usr/bin/env python3
"""Run pycisTopic and retain both low-rank factors with unambiguous semantics."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from common import read_mtx, read_names, write_json


def as_dataframe(value, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if sparse.issparse(value):
        value = value.toarray()
    array = np.asarray(value)
    if array.ndim != 2:
        raise RuntimeError(f"{name} is not 2D: {array.shape}")
    return pd.DataFrame(array)


def model_matrix(model, names: tuple[str, ...], label: str) -> pd.DataFrame:
    for name in names:
        if hasattr(model, name) and getattr(model, name) is not None:
            return as_dataframe(getattr(model, name), f"{label}:{name}")
    available = sorted(k for k in vars(model) if not k.startswith("_"))
    raise RuntimeError(f"Missing {label}; checked {names}; model has {available}")


def orient_cell_topic(frame: pd.DataFrame, cells: np.ndarray, n_topics: int) -> np.ndarray:
    cell_set = set(cells)
    if set(map(str, frame.index)) == cell_set:
        frame.index = frame.index.astype(str)
        array = frame.loc[cells].to_numpy()
    elif set(map(str, frame.columns)) == cell_set:
        transposed = frame.T
        transposed.index = transposed.index.astype(str)
        array = transposed.loc[cells].to_numpy()
    elif frame.shape == (len(cells), n_topics):
        array = frame.to_numpy()
    elif frame.shape == (n_topics, len(cells)):
        array = frame.to_numpy().T
    else:
        raise RuntimeError(f"Cannot orient cell-topic matrix {frame.shape}")
    return np.asarray(array, dtype=np.float32)


def orient_topic_peak(frame: pd.DataFrame, peaks: np.ndarray, n_topics: int) -> np.ndarray:
    peak_set = set(peaks)
    if set(map(str, frame.columns)) == peak_set:
        frame.columns = frame.columns.astype(str)
        array = frame.loc[:, peaks].to_numpy()
    elif set(map(str, frame.index)) == peak_set:
        transposed = frame.T
        transposed.columns = transposed.columns.astype(str)
        array = transposed.loc[:, peaks].to_numpy()
    elif frame.shape == (n_topics, len(peaks)):
        array = frame.to_numpy()
    elif frame.shape == (len(peaks), n_topics):
        array = frame.to_numpy().T
    else:
        raise RuntimeError(f"Cannot orient topic-peak matrix {frame.shape}")
    return np.asarray(array, dtype=np.float32)


def normalize_rows(array: np.ndarray) -> np.ndarray:
    totals = array.sum(axis=1, keepdims=True)
    return np.divide(array, totals, out=np.zeros_like(array), where=totals > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--topics", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if min(args.topics, args.iterations, args.cpus) < 1:
        parser.error("topics, iterations, and cpus must be positive")
    outdir = Path(args.outdir).resolve()
    intermediate = outdir / "intermediate"

    import pycisTopic
    from pycisTopic.cistopic_class import create_cistopic_object
    from pycisTopic.lda_models import run_cgs_models

    peaks = read_names(intermediate / "region_names.tsv")
    cells = read_names(intermediate / "cell_names.tsv")
    counts = read_mtx(
        intermediate / "atac_counts_regions_cells.mtx.gz", dtype=np.float64
    )
    if counts.shape != (len(peaks), len(cells)):
        raise RuntimeError("ATAC matrix is not peaks x cells")
    if counts.data.size and (counts.data < 0).any():
        raise RuntimeError("ATAC contains negative values")
    counts.data = np.rint(counts.data).astype(np.int32)
    counts = counts.astype(np.int32)
    if (np.asarray(counts.sum(axis=0)).ravel() == 0).any():
        raise RuntimeError("Zero-depth ATAC cells would be dropped by cisTopic; paired cells must be retained")

    obj = create_cistopic_object(
        fragment_matrix=counts,
        cell_names=cells.tolist(),
        region_names=peaks.tolist(),
        min_frag=1,
        min_cell=1,
        is_acc=1,
        project=outdir.name,
        tag_cells=False,
    )
    if list(map(str, obj.cell_names)) != cells.tolist() or list(map(str, obj.region_names)) != peaks.tolist():
        raise RuntimeError("cisTopic changed the paired cell or peak order")
    models = run_cgs_models(
        obj,
        n_topics=[args.topics],
        n_cpu=args.cpus,
        n_iter=args.iterations,
        random_state=args.seed,
        alpha=50,
        alpha_by_topic=True,
        eta=0.1,
        eta_by_topic=False,
        save_path=None,
    )
    if len(models) != 1:
        raise RuntimeError(f"Expected one model, obtained {len(models)}")
    model = models[0]
    obj.add_LDA_model(model)

    theta_raw = model_matrix(
        model,
        ("cell_topic", "cell_topic_distribution", "document_topic", "doc_topic"),
        "cell-topic",
    )
    psi_raw = model_matrix(
        model,
        ("topic_region", "topic_region_distribution", "topic_word", "topic_feature"),
        "topic-peak",
    )
    theta = normalize_rows(orient_cell_topic(theta_raw, cells, args.topics))
    psi = normalize_rows(orient_topic_peak(psi_raw, peaks, args.topics))
    if theta.shape != (len(cells), args.topics):
        raise RuntimeError(f"theta shape is {theta.shape}")
    if psi.shape != (args.topics, len(peaks)):
        raise RuntimeError(f"psi shape is {psi.shape}")
    if not np.isfinite(theta).all() or not np.isfinite(psi).all():
        raise RuntimeError("cisTopic factors contain non-finite values")

    np.savez_compressed(
        intermediate / "cistopic_factors.npz",
        theta_cells_topics=theta,
        psi_topics_peaks=psi,
    )
    if str(__import__("os").environ.get("SAVE_CISTOPIC_OBJECT", "0")) == "1":
        with open(intermediate / "cistopic_object.pkl", "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Skipping cistopic_object.pkl because SAVE_CISTOPIC_OBJECT=0")
    audit = {
        "backend": "pycisTopic.run_cgs_models",
        "pycisTopic_version": getattr(pycisTopic, "__version__", "unknown"),
        "theta_semantics": "cells_x_topics",
        "psi_semantics": "topics_x_peaks",
        "required_obsm_cisTopic_semantics": "theta_at_psi = cells_x_peaks",
        "n_cells": len(cells),
        "n_peaks": len(peaks),
        "n_topics": args.topics,
        "n_iterations": args.iterations,
        "seed": args.seed,
    }
    write_json(intermediate / "cistopic_audit.json", audit)
    print("theta cells x topics:", theta.shape)
    print("psi topics x peaks:", psi.shape)
    print("final obsm['cisTopic'] will be cells x peaks:", (len(cells), len(peaks)))
    print("CISTOPIC STAGE: PASSED")


if __name__ == "__main__":
    main()
