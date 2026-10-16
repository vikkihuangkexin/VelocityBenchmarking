"""Runtime and CPU RAM: reverse sample-z per workload, equal component means."""
from pathlib import Path
import sys
import pandas as pd


def apply_all_failed_policy(result, raw_counts, methods, policy):
    """Spatial approved exception: retain missing scientific score; expose display/effective ranks."""
    result = result.copy()
    result['n_raw_valid'] = result.method.map(raw_counts).fillna(0).astype(int)
    result['entire_dimension_failure'] = result.n_raw_valid.eq(0)
    result['display_rank'] = result['rank']
    result['effective_reversed_rank'] = result.reversed_rank
    if policy == 'bottom_tie_effective_zero':
        mask = result.entire_dimension_failure
        n_failed = int(mask.sum())
        if n_failed:
            result.loc[mask, 'display_rank'] = len(methods) - (n_failed - 1) / 2
            result.loc[mask, 'effective_reversed_rank'] = 0.0
    elif policy != 'missing':
        raise ValueError('Unknown entire Scalability failure policy')
    return result


def calculate(inputs, config):
    from common.core import task_z, combine_components
    parts = [inputs.long_resource(p) for p in config['resource_csvs']]
    source = pd.concat(parts, ignore_index=True)
    spec = config['resource_components']
    if source.duplicated(['method', 'protocol', 'measure', 'dataset_id']).any():
        raise ValueError('Duplicate resource task across source files')
    for measure, unit in config.get('resource_units', {}).items():
        if not source.loc[source.measure.eq(measure), 'unit'].eq(unit).all():
            raise ValueError(f'Unexpected resource unit for {measure}; expected {unit}')
    x = source[source.protocol.isin(config['resource_protocols']) & source.measure.isin(spec)].copy()
    if set(x.measure) != set(spec):
        raise ValueError('Resource component missing from inputs')
    catalog = x[['protocol', 'measure', 'dataset_id']].drop_duplicates()
    roster = pd.DataFrame({'method': inputs.methods})
    grid = roster.merge(catalog, how='cross')
    x = grid.merge(x, on=['method', 'protocol', 'measure', 'dataset_id'], how='left', validate='one_to_one')
    z = task_z(x, ['protocol', 'measure', 'dataset_id'], lower_is_better=True)
    parts = z.groupby(['method', 'measure']).z.mean().unstack('measure').reindex(columns=spec)
    result = combine_components(parts, inputs.methods)
    counts = z.groupby('method').z.agg(['count', 'size'])
    result['n_z_valid'] = result.method.map(counts['count'])
    result['n_expected'] = result.method.map(counts['size'])
    result['coverage'] = result.n_z_valid / result.n_expected
    policy = config.get('scalability_all_failed_policy', 'missing')
    if policy == 'bottom_tie_effective_zero' and config['profile'] != 'Spatial':
        raise ValueError('The entire-failure display/effective-rank exception is Spatial only')
    result = apply_all_failed_policy(result, z.groupby('method').value.count(), inputs.methods, policy)
    # Other protocols and VRAM are exported for inspection but cannot enter score.
    display = source[~source.protocol.isin(config['resource_protocols']) | ~source.measure.isin(spec)]
    return z, result, display


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run import main
    main(forced_stage='scalability')
