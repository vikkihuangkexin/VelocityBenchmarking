#!/usr/bin/env python3
"""Independent numerical verification from Source_data, without importing scoring code.

Uses column-wise NumPy calculations and explicit method loops. Validates input
hashes, all five dimension scores/ranks, Overall and source-preserving zero/NA rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def verify(results, source):
    cfg = json.loads((results / 'audit/effective_config.json').read_text(encoding='utf-8'))
    manifest = pd.read_csv(results / 'audit/input_manifest.csv')
    for row in manifest.itertuples():
        data = (source / row.path).read_bytes()
        if len(data) != row.bytes or hashlib.sha256(data).hexdigest() != row.sha256:
            raise AssertionError(f'Source data changed since calculation: {row.path}')
    methods = pd.read_csv(source / cfg['methods_csv']).sort_values('method_order').method.tolist()
    def norm(s):
        return re.sub('[^a-z0-9]', '', str(s).replace('κ', 'k').replace('–', '-').replace('—', '-').lower())
    aliases = {norm(m): m for m in methods}
    aliases.update({norm(k): v for k, v in cfg.get('method_aliases', {}).items()})
    real_map = {}
    if cfg.get('real_id_map_csv'):
        mp = pd.read_csv(source / cfg['real_id_map_csv'], dtype=str)
        real_map = dict(zip(mp.dataset_number, mp.dataset_id))

    def standardize(v):
        a = np.array(v, dtype=float)
        out = np.full(a.shape, np.nan)
        ok = np.isfinite(a)
        if ok.sum() >= 2:
            sd = np.std(a[ok], ddof=1)
            if sd > 0:
                out[ok] = (a[ok] - np.mean(a[ok])) / sd
        return out

    def mean(v):
        a = np.asarray(v, dtype=float)
        return float(np.mean(a[np.isfinite(a)])) if np.isfinite(a).any() else np.nan

    def load_matrix(entry, sd=False):
        if entry.get('optional') and not (source / entry['path']).is_file():
            return pd.DataFrame(columns=['method', 'metric', 'data_type', 'dataset_id', 'value', 'z'])
        d = pd.read_csv(source / entry['path'], dtype=str, keep_default_na=False, encoding='utf-8-sig')
        selected = {}
        for row in d.itertuples(index=False, name=None):
            name = aliases.get(norm(row[0]))
            if name:
                selected[name] = row[1:]
        roster = [m for m in methods if m in selected] if sd else methods
        columns = list(d.columns[1:])
        values = pd.DataFrame.from_dict(selected, orient='index', columns=columns).reindex(roster)
        values = values.replace(r'^\s*(?:NA|na|NaN|nan|n/a|N/A|none|None|null|error)?\s*$', np.nan, regex=True)
        values = values.apply(pd.to_numeric, errors='raise')
        records = []
        for col in columns:
            vals = values[col].to_numpy(dtype=float)
            scores = standardize(vals)
            dataset = real_map.get(col, col) if entry['data_type'] == 'real' else col
            for i, method in enumerate(roster):
                records.append({'method': method, 'metric': entry['metric'], 'data_type': entry['data_type'],
                                'dataset_id': dataset, 'value': vals[i], 'z': scores[i]})
        return pd.DataFrame(records)

    def matrices(entries):
        frames = [load_matrix(e) for e in entries]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=['method', 'metric', 'data_type', 'dataset_id', 'value', 'z'])

    measurement = matrices(cfg['measurement_inputs'])
    expected = {}
    for dimension in ['accuracy', 'consistency']:
        scores = []
        for method in methods:
            metric_scores = []
            for metric in cfg[dimension + '_metrics']:
                all_rows = measurement[measurement.metric.eq(metric)]
                f = all_rows[all_rows.method.eq(method)]
                if cfg['data_type_weighting'] == 'pooled':
                    metric_scores.append(mean(f.z))
                else:
                    by_type = [mean(f[f.data_type.eq(t)].z) for t in all_rows.data_type.unique()]
                    metric_scores.append(float(np.mean(by_type)) if by_type else np.nan)
            scores.append(mean(metric_scores))
        expected[dimension] = pd.Series(scores, index=methods)
    observed_tasks = pd.read_csv(results / 'audit/measurement_task_z.csv')
    keys = ['method', 'metric', 'data_type', 'dataset_id']
    reference_tasks = measurement.set_index(keys).sort_index()
    actual_tasks = observed_tasks.set_index(keys).sort_index()
    if not reference_tasks.index.equals(actual_tasks.index):
        raise AssertionError('Measurement task universe changed')
    np.testing.assert_allclose(reference_tasks.z, actual_tasks.z, rtol=1e-11, atol=1e-11, equal_nan=True)

    resource = pd.concat([pd.read_csv(source / p, float_precision='round_trip') for p in cfg['resource_csvs']])
    resource = resource[resource.protocol.isin(cfg['resource_protocols']) & resource.measure.isin(cfg['resource_components'])].copy()
    resource['method'] = resource.method.map(lambda m: aliases.get(norm(m)))
    resource = resource[resource.method.notna()]
    resource_z = {}
    for (protocol, measure, dataset), f in resource.groupby(['protocol', 'measure', 'dataset_id']):
        v = f.set_index('method').clean_value.reindex(methods).to_numpy(dtype=float)
        resource_z[(protocol, measure, dataset)] = standardize(-v)
    expected['scalability'] = pd.Series([
        mean([mean([v[i] for (p, measure, d), v in resource_z.items() if measure == m])
              for m in cfg['resource_components']]) for i, method in enumerate(methods)], index=methods)

    down = matrices(cfg['downsampling_inputs'])
    down_by_dataset = down.groupby(['method', 'data_type', 'dataset_id']).z.mean()
    down_scores = []
    for method in methods:
        if down.empty:
            down_scores.append(np.nan)
            continue
        vals = down_by_dataset.xs(method, level='method')
        group_scores = []
        for pattern in cfg['downsampling_groups'].values():
            group_scores.append(mean([v for (t, d), v in vals.items() if re.search(pattern, d)]))
        down_scores.append(mean(group_scores))
    down_standardized = standardize(down_scores)
    sd_standardized = np.full(len(methods), np.nan)
    true_sd_zeros = 0
    if cfg['stability_mode'] == 'combined':
        sd = pd.concat([load_matrix(e, sd=True) for e in cfg['batch_sd_inputs']], ignore_index=True)
        true_sd_zeros = int(sd.value.eq(0).sum())
        dataset_sd = sd.groupby(['method', 'data_type', 'dataset_id']).z.mean()
        batch_scores = []
        types = sd.data_type.unique()
        for method in methods:
            scores = [mean([v for (m, t, d), v in dataset_sd.items() if m == method and t == typ]) for typ in types]
            batch_scores.append(float(np.mean(scores)))
        sd_standardized = standardize(-np.asarray(batch_scores))
        observed_sd = pd.read_csv(results / 'stability/batch_sd_raw.csv')
        if true_sd_zeros != int(observed_sd.value.eq(0).sum()):
            raise AssertionError('True SD zeros changed')
    st = []
    for i, method in enumerate(methods):
        values = [down_standardized[i]]
        weights = [cfg['stability_weights']['downsampling']]
        if cfg['stability_mode'] == 'combined':
            values.append(sd_standardized[i])
            weights.append(cfg['stability_weights']['batchrun'])
        mask = np.isfinite(values)
        st.append(np.average(np.asarray(values)[mask], weights=np.asarray(weights)[mask]) if mask.any() else np.nan)
    expected['stability'] = pd.Series(st, index=methods)

    rules = pd.read_csv(source / cfg['usability_rules_csv'])
    items = pd.read_csv(source / cfg['usability_items_csv'])
    items['method'] = items.method.map(lambda m: aliases.get(norm(m)))
    u = []
    for method in methods:
        category_values = []
        for cat, weight in cfg['usability_category_weights'].items():
            r = rules[rules.category.eq(cat)]
            f = items[items.method.eq(method) & items.category.eq(cat)].set_index('subcategory').reindex(r.subcategory)
            category_values.append(float(np.sum(f.subscore.to_numpy() * r.subcategory_weight.to_numpy())) * weight)
        u.append(float(np.sum(category_values)))
    expected['usability'] = pd.Series(u, index=methods)

    expected_ranks = {}
    dimensions = ['accuracy', 'consistency', 'scalability', 'stability', 'usability']
    counts, max_errors = {}, {}
    provisional = []
    for dimension in dimensions:
        actual = pd.read_csv(results / dimension / 'scores_ranks.csv', float_precision='round_trip').set_index('method').reindex(methods)
        wanted = expected[dimension]
        np.testing.assert_allclose(wanted, actual.score, rtol=1e-11, atol=1e-11, equal_nan=True, err_msg=dimension)
        rank = wanted.rank(ascending=False, method='average', na_option='keep')
        np.testing.assert_allclose(rank, actual['rank'], equal_nan=True, err_msg=dimension + ' rank')
        np.testing.assert_allclose(len(methods) + 1 - rank, actual.reversed_rank, equal_nan=True)
        expected_ranks[dimension] = len(methods) + 1 - rank
        if dimension == 'scalability' and cfg.get('scalability_all_failed_policy') == 'bottom_tie_effective_zero':
            raw_count = resource.groupby('method').clean_value.count().reindex(methods).fillna(0)
            expected_ranks[dimension] = expected_ranks[dimension].where(raw_count.ne(0), 0.0)
            np.testing.assert_allclose(expected_ranks[dimension], actual.effective_reversed_rank, equal_nan=True)
        if dimension in ['accuracy', 'consistency']:
            declared = [e['path'] for e in cfg['measurement_inputs'] if e['metric'] in cfg[dimension + '_metrics']]
        elif dimension == 'stability':
            declared = [e['path'] for e in cfg['downsampling_inputs'] +
                        (cfg['batch_sd_inputs'] if cfg['stability_mode'] == 'combined' else [])]
        elif dimension == 'scalability':
            declared = cfg['resource_csvs']
        else:
            declared = [cfg['usability_items_csv'], cfg['usability_rules_csv']]
        available = sum((source / p).is_file() for p in declared)
        status = 'complete' if available == len(declared) else 'partial' if available else 'unavailable'
        if not actual.input_status.eq(status).all():
            raise AssertionError(f'Input status mismatch: {dimension}')
        if status != 'complete':
            provisional.append(dimension)
            expected_ranks[dimension] = pd.Series(np.nan, index=methods)
        counts[dimension] = int(wanted.notna().sum())
        difference = (wanted - actual.score).dropna().abs()
        max_errors[dimension] = float(difference.max()) if len(difference) else None
    total = pd.Series(0.0, index=methods)
    for d in dimensions:
        total = total + cfg['overall_weights'][d] * expected_ranks[d]
    actual = pd.read_csv(results / 'overall/scores_ranks.csv', float_precision='round_trip').set_index('method').reindex(methods)
    np.testing.assert_allclose(total, actual.score, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(total.rank(ascending=False, method='average'), actual['rank'], equal_nan=True)
    counts['overall'] = int(total.notna().sum())
    report = {'status': 'PASS', 'verification': 'independent recomputation from Source_data; no scoring imports',
              'n_methods': len(methods), 'files_hash_verified': len(manifest), 'scores_available': counts,
              'max_absolute_score_errors': max_errors, 'missing_overall_methods': total[total.isna()].index.tolist(),
              'numeric_zeros_preserved': int(measurement.value.eq(0).sum()),
              'negative_measurements_preserved': int(measurement.value.lt(0).sum()),
              'true_SD_zeros_preserved': true_sd_zeros}
    report['provisional_or_unavailable_dimensions'] = provisional
    (results / 'audit/independent_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1] / 'Source_data')
    args = parser.parse_args()
    verify(args.results.resolve(), args.source_root.resolve())
