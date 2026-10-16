#!/usr/bin/env python3
"""Aggregate scored, finite cell-level rescaled cosines into a method/task matrix."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def prepare(raw_path, source_root):
    frame = pd.read_csv(raw_path, float_precision='round_trip')
    required = {'method', 'id', 'cell_id', 'cosine', 'rescaled_cosine', 'scored'}
    if not required.issubset(frame.columns) or frame.duplicated(['method', 'id', 'cell_id']).any():
        raise ValueError('Invalid or duplicate cell-level downsampling observations')
    scored = frame.scored.astype(str).str.lower().map({'true': True, 'false': False})
    if scored.isna().any():
        raise ValueError('Unknown scored flag')
    finite = np.isfinite(frame.rescaled_cosine)
    if not frame.loc[finite, 'rescaled_cosine'].between(0, 1).all():
        raise ValueError('Rescaled cosine outside [0,1]')
    both = finite & np.isfinite(frame.cosine)
    np.testing.assert_allclose(frame.loc[both, 'rescaled_cosine'],
        (frame.loc[both, 'cosine'] + 1) / 2, rtol=1e-12, atol=1e-12)
    selected = frame[scored & finite]
    grouped = selected.groupby(['method', 'id']).rescaled_cosine.agg(['mean', 'count'])
    methods = pd.read_csv(source_root / 'measurements_raw/ATAC/methods.csv').method.tolist()
    if not set(frame.method).issubset(methods):
        raise ValueError('Raw downsampling contains an unknown method')
    tasks = sorted(frame.id.unique(), key=lambda s: int(s.split('_', 1)[0]))
    matrix = grouped['mean'].unstack('id').reindex(index=methods, columns=tasks)
    matrix.index.name = 'method'
    folder = source_root / 'measurements_raw/ATAC/measurement/downsampling'
    folder.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(folder / 'GroundTruthCorrelation_ATAC_downsampling_02_summary.csv', float_format='%.17g')
    counts = grouped.reset_index().rename(columns={'id': 'dataset_id', 'mean': 'value', 'count': 'n_scored_cells'})
    counts.to_csv(folder / 'aggregation_counts.csv', index=False, float_format='%.17g')
    receipt = {'source_name': raw_path.name, 'sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
               'source_rows': len(frame), 'selected_rows': len(selected), 'methods': len(methods),
               'tasks': len(tasks), 'aggregation': 'mean of finite rescaled_cosine where scored=True',
               'cell_keys_unique': True, 'r_labels': 'retained verbatim; no biological semantics inferred'}
    (folder / 'aggregation_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2] / 'Source_data')
    args = parser.parse_args()
    result = prepare(args.raw.resolve(), args.source_root.resolve())
    print(f'Prepared {result.shape[0]} methods x {result.shape[1]} tasks')
