"""Numerical rules shared by all five dimensions. No historical results as inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd


DIMENSIONS = ['accuracy', 'consistency', 'scalability', 'stability', 'usability']


def sample_z(values: pd.Series) -> pd.Series:
    """Sample z; an unidentifiable task remains NA, including true equal zeros."""
    values = values.astype(float)
    n, sd = values.count(), values.std(ddof=1)
    if n < 2 or not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (values - values.mean()) / sd


def task_z(frame: pd.DataFrame, keys: list[str], *, lower_is_better=False) -> pd.DataFrame:
    """Standardize across selected methods within each metric/dataset task."""
    out = frame.copy()
    if out.duplicated(['method'] + keys).any():
        raise ValueError('Duplicate method/task before standardization')
    out['directed_value'] = -out.value if lower_is_better else out.value
    stat = out.groupby(keys, dropna=False).directed_value.agg(['count', 'mean', 'std'])
    stat = stat.rename(columns={'count': 'task_n', 'mean': 'task_mean', 'std': 'task_sd'})
    out = out.join(stat, on=keys)
    ok = out.task_n.ge(2) & out.task_sd.gt(0)
    out['z'] = ((out.directed_value - out.task_mean) / out.task_sd).where(ok)
    out['z_status'] = np.select(
        [out.value.isna(), out.task_n.lt(2), out.task_sd.eq(0)],
        ['raw_missing', 'fewer_than_2_methods', 'all_values_equal'], default='valid')
    return out


def rank_scores(scores: pd.Series, methods: list[str]) -> pd.DataFrame:
    """Rank 1 is best. NA has no rank; display order never becomes a numeric rank."""
    out = scores.reindex(methods).rename('score').rename_axis('method').reset_index()
    out['rank'] = out.score.rank(ascending=False, method='average', na_option='keep')
    out['reversed_rank'] = len(methods) + 1 - out['rank']
    out['normalized_rank'] = out.reversed_rank / len(methods)
    out['fixed_method_order'] = np.arange(1, len(methods) + 1)
    out = out.sort_values(['rank', 'fixed_method_order'], na_position='last', kind='stable')
    out['display_order'] = np.arange(1, len(methods) + 1)
    return out.reset_index(drop=True)


def metric_summary(z: pd.DataFrame, weighting='balanced') -> pd.DataFrame:
    """Within-type task means, then strict 50:50 real/simulation or pooled sensitivity.

    A metric present only for simulations (e.g. GT) uses simulation only.
    If a shared metric has no usable result for a method in one type, its balanced
    combined score is NA. An absent metric may be omitted at dimension averaging,
    with effective weights exported explicitly.
    """
    rows = []
    methods = list(z.method.drop_duplicates())
    for metric, metric_frame in z.groupby('metric', sort=False):
        types = list(metric_frame.data_type.drop_duplicates())
        for method in methods:
            f = metric_frame[metric_frame.method.eq(method)]
            row = {'method': method, 'metric': metric}
            type_scores = []
            for typ in types:
                t = f[f.data_type.eq(typ)]
                v = t.value.dropna()
                s = t.z.mean()
                type_scores.append(s)
                row.update({f'{typ}_score': s, f'{typ}_n_valid': int(t.z.notna().sum()),
                            f'{typ}_n_expected': len(t), f'{typ}_raw_mean': v.mean(),
                            f'{typ}_raw_se': v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan})
            if weighting == 'balanced':
                row['score'] = float(np.mean(type_scores)) if np.isfinite(type_scores).all() else np.nan
            elif weighting == 'pooled':
                row['score'] = f.z.mean()
            else:
                raise ValueError(f'Unknown data-type weighting: {weighting}')
            row['type_weighting'] = weighting
            row['expected_types'] = ';'.join(types)
            row['n_z_valid'] = int(f.z.notna().sum())
            row['n_expected'] = len(f)
            row['coverage'] = row['n_z_valid'] / len(f) if len(f) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def combine_components(components: pd.DataFrame, methods: list[str], weights=None,
                       strict=False) -> pd.DataFrame:
    """Components indexed by method, columns named by their actual definitions."""
    components = components.reindex(methods)
    weights = pd.Series(weights if weights is not None else {c: 1.0 for c in components})
    if set(weights.index) != set(components.columns) or not (weights > 0).all():
        raise ValueError('Component weights must be positive and match components')
    present = components.notna()
    denom = present.mul(weights, axis=1).sum(axis=1)
    numerator = components.mul(weights, axis=1).sum(axis=1, min_count=1)
    score = numerator / denom.replace(0, np.nan)
    if strict:
        score = score.where(present.all(axis=1))
    out = rank_scores(score, methods).set_index('method')
    out['n_components'] = present.sum(axis=1)
    out['expected_components'] = len(components.columns)
    out['missing_components'] = present.apply(lambda r: ';'.join(r.index[~r]), axis=1)
    for col in components:
        out[f'component_{col}'] = components[col]
        eff = (weights[col] / denom).where(present[col])
        out[f'effective_weight_{col}'] = eff.where(score.notna()) if strict else eff
    return out.reset_index()


def dimension_from_metrics(z, metrics, methods, weighting):
    summary = metric_summary(z[z.metric.isin(metrics)], weighting)
    wide = summary.pivot(index='method', columns='metric', values='score').reindex(columns=metrics)
    return summary, combine_components(wide, methods)


def operational_scores(z, metrics, methods, weighting):
    """Supplement only: fixed task universe, q=(N+1-r)/N; failed task q=0."""
    x = z[z.metric.isin(metrics)].copy()
    keys = ['metric', 'data_type', 'dataset_id']
    x['task_rank'] = x.groupby(keys).directed_value.rank(ascending=False, method='average')
    x['q'] = ((len(methods) + 1 - x.task_rank) / len(methods)).fillna(0.0)
    # q is task performance with a coverage penalty; do not restandardize it.
    q = x.copy()
    q['z'] = q.q
    summary = metric_summary(q, weighting)
    wide = summary.pivot(index='method', columns='metric', values='score').reindex(columns=metrics)
    return x, combine_components(wide, methods)
