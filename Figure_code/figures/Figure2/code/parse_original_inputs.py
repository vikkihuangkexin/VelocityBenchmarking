#!/usr/bin/env python3
"""Read the updated matrices and the original downsampling CSV from scratch.

TransitionScore is a filename alias for the baseline Transition V1 metric.
The locked real-data ID map is an English-only projection of Table S3v2.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import re

import pandas as pd
from io_atomic import save_csv

P = Path(__file__).resolve().parents[1]
methods = pd.read_csv(P / 'data/methods.csv').sort_values('method_order').method.tolist()
excluded = {'Region Velocity', 'PhyloVelo'}
assert len(methods) == len(set(methods)) == 28


def method_key(value):
    value = str(value).strip().replace('\u03ba', 'k').replace('\u2013', '-').replace('\u2014', '-')
    return re.sub(r'\s+', ' ', value).casefold()


aliases = {method_key(m): m for m in methods + sorted(excluded)}
aliases.update({method_key(k): v for k, v in {
    'Region-Velocity': 'Region Velocity', 'Tsvelo': 'TSvelo',
    'scVelo stochastic': 'scVelo-stochastic', 'scVelo dynamical': 'scVelo-dynamical',
    'Pyro Velocity': 'Pyro-Velocity', 'kappa-velo': 'k-velo',
}.items()})
missing = {'', 'na', 'nan', 'n/a', 'none', 'null', 'error'}


def parse(token):
    text = str(token).strip()
    if text.casefold() in missing:
        return math.nan, 'missing_marker', 'empty cell' if not text else 'missing marker'
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f'Unknown nonnumeric token: {token!r}') from exc
    if not math.isfinite(value):
        raise ValueError(f'Unexpected nonfinite value: {token!r}')
    status = 'valid_zero' if value == 0 else 'valid_negative' if value < 0 else 'valid_positive'
    return value, status, ''


dictionary = pd.read_csv(P / 'data/real_dataset_ids.csv', dtype=str)
assert not dictionary.dataset_number.duplicated().any()
assert not dictionary.dataset_id.duplicated().any()
mapping = dict(zip(dictionary.dataset_number, dictionary.dataset_id))
save_csv(dictionary, P / 'tables/real_dataset_dictionary.csv')

metric_names = {
    'AngleConsistency': 'Angle consistency', 'CBDir': 'CBDir',
    'ConsistencyScore': 'Consistency score', 'ICCoh': 'ICCoh',
    'GroundTruthCorrelation': 'Ground-truth correlation',
    'TransitionScoreV1': 'Transition V1', 'TransitionScore': 'Transition V1',
}
expected = {(m, t) for m in ['Angle consistency', 'CBDir', 'Consistency score', 'ICCoh', 'Transition V1']
            for t in ['real', 'simulation']} | {('Ground-truth correlation', 'simulation')}
assessed = {'DeepKINET', 'DeepVelo', 'InterVelo', 'LatentVelo', 'SvelvetVAE', 'VeloAE', 'VeloVAE', 'veloVI'}
rows = {'primary': [], 'std': []}
exclusion, manifest, catalog = [], [], []
task_headers = {}

for folder in ['summary', 'std', 'downsampling']:
    files = sorted((P / 'data/raw' / folder).glob('*.csv'))
    assert len(files) == (1 if folder == 'downsampling' else 11), (folder, len(files))
    observed = set()
    for path in files:
        data = path.read_bytes()
        short = path.relative_to(P / 'data/raw').as_posix()
        match = re.fullmatch(r'(\w+)_scRNA_(real|simulation)_(02_summary|03_batchrun_std)\.csv', path.name)
        if not match:
            raise ValueError(f'Unrecognized input file: {short}')
        prefix, typ, version = match.groups()
        assert (version == '03_batchrun_std') == (folder == 'std'), short
        metric = metric_names[prefix]
        if folder == 'downsampling':
            assert metric == 'Ground-truth correlation' and typ == 'simulation', short
            metric = 'Performance under downsampling'
        key = metric, typ
        assert key not in observed, (folder, key)
        observed.add(key)
        group = 'std' if folder == 'std' else 'primary'
        matrix = list(csv.reader(io.StringIO(data.decode('utf-8-sig'))))
        assert matrix and matrix[0][0].strip().casefold() in {'', 'method'}, short
        headers = matrix[0][1:]
        assert headers and len(headers) == len(set(headers)), short
        datasets = [mapping[h.strip()] if typ == 'real' else h.strip() for h in headers]
        assert all(datasets) and len(datasets) == len(set(datasets)), short
        task_headers[folder, metric, typ] = set(datasets)
        seen = {}
        for ri, row in enumerate(matrix[1:], 2):
            if not row or not any(str(x).strip() for x in row):
                continue
            assert len(row) == len(matrix[0]), (short, ri, 'unequal row length')
            raw_method = row[0].strip()
            method = aliases.get(method_key(raw_method))
            if method is None:
                raise ValueError(f'Unknown method: {raw_method} in {short}')
            if method in excluded:
                exclusion.append({'method_raw': raw_method, 'method': method, 'source_path': short,
                                  'CSV_row': ri, 'reason': 'excluded before task standardisation'})
                continue
            if method in seen:
                raise ValueError(f'Duplicate canonical method: {method} in {short}')
            seen[method] = ri, row, raw_method
        if group == 'std':
            assert set(seen) == assessed, (short, set(seen))
        selected = methods if group == 'primary' else [m for m in methods if m in seen]
        for h, dataset in zip(headers, datasets):
            catalog.append({'metric': metric, 'data_type': typ, 'dataset_id': dataset,
                            'dataset_raw': h, 'source_path': short, 'source_kind': group})
        for method in selected:
            present = method in seen
            ri, row, raw_method = seen[method] if present else (None, None, method)
            for ci, (h, dataset) in enumerate(zip(headers, datasets), 2):
                token = row[ci - 1] if present else ''
                value, status, reason = parse(token) if present else (math.nan, 'method_not_provided', 'method row absent from source')
                if group == 'std' and math.isfinite(value) and value < 0:
                    raise ValueError(f'Negative SD: {short}:{ri}:{ci}')
                rows[group].append({
                    'method': method, 'method_raw': raw_method, 'metric': metric, 'data_type': typ,
                    'dataset_id': dataset, 'dataset_raw': h, 'raw_token': token, 'clean_value': value,
                    'value_status': status, 'missing_reason': reason,
                    'is_zero': math.isfinite(value) and value == 0,
                    'is_negative': math.isfinite(value) and value < 0,
                    'source_family': 'summary_csv',
                    'source_role': 'batch_run_std_supplement' if group == 'std' else 'downsampling' if folder == 'downsampling' else 'new_summary_primary',
                    'source_path': short,
                    'source_locator': f'row={ri};column={ci};header={h}' if present else '',
                    'method_present': present, 'dataset_present': True,
                })
        manifest.append({'path': path.relative_to(P).as_posix(), 'source_role': folder,
                         'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    assert observed == ({('Performance under downsampling', 'simulation')} if folder == 'downsampling' else expected)

for metric, typ in expected:
    assert task_headers['summary', metric, typ] == task_headers['std', metric, typ]
for group, filename in [('primary', 'primary_clean_long.csv'), ('std', 'batch_sd_clean_long.csv')]:
    frame = pd.DataFrame(rows[group])
    assert not frame.duplicated(['method', 'metric', 'data_type', 'dataset_id']).any()
    save_csv(frame, P / 'data' / filename)
save_csv(pd.DataFrame(exclusion), P / 'tables/excluded_method_audit.csv')
save_csv(pd.DataFrame(catalog), P / 'tables/source_task_catalog.csv')
save_csv(pd.DataFrame(manifest), P / 'tables/original_input_manifest.csv')
result = {'status': 'PASS', 'primary_rows': len(rows['primary']), 'batch_sd_rows': len(rows['std']),
          'input_files': len(manifest), 'excluded_method_rows': len(exclusion),
          'raw_inputs_reparsed': True, 'missing_SD_not_filled': True}
(P / 'validation').mkdir(exist_ok=True)
(P / 'validation/original_parser_checks.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
