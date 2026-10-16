"""Declared sensitivity scenarios; never optimize weights to a target method order."""
import itertools
import copy
import pandas as pd


def calculate(tables, z, config, methods, out):
    from common.core import dimension_from_metrics
    from overall.compute_overall_rank import calculate as overall
    baseline = tables['overall'].set_index('method')['rank']
    scenarios, weights = [], []
    # A small, explicit local grid around the configured five-dimension weights.
    for a, c, s in itertools.product([.625, .650, .675], [.075, .100, .125], [.075, .100, .125]):
        remainder = 1 - a - c - s
        w = {'accuracy': a, 'consistency': c, 'scalability': s,
             'stability': remainder / 2, 'usability': remainder / 2}
        cfg = copy.deepcopy(config)
        cfg['overall_weights'] = w
        sid = f'A{a:.3f}_C{c:.3f}_Sc{s:.3f}'
        f = overall(tables, cfg, methods)
        f['scenario'] = sid
        scenarios.append(f[['scenario', 'method', 'score', 'rank']])
        weights.append({'scenario': sid, **w})
    long = pd.concat(scenarios, ignore_index=True)
    ranks = long.groupby('method')['rank'].agg(['min', 'max', 'median', 'count']).reset_index()
    ranks['baseline_rank'] = ranks.method.map(baseline)
    alternate = []
    for label, weighting in [('pooled_real_simulation', 'pooled')]:
        alt = dict(tables)
        for d in ['accuracy', 'consistency']:
            _, alt[d] = dimension_from_metrics(z, config[d + '_metrics'], methods, weighting)
            if 'overall_eligible' in tables[d]:
                alt[d] = alt[d].merge(tables[d][['method', 'overall_eligible']], on='method', validate='one_to_one')
        f = overall(alt, config, methods)
        f['scenario'] = label
        alternate.append(f[['scenario', 'method', 'score', 'rank']])
    alt = dict(tables)
    alt['stability'] = pd.read_csv(out / 'stability/downsampling_only_sensitivity.csv')
    if 'overall_eligible' in tables['stability']:
        alt['stability'] = alt['stability'].drop(columns=['overall_eligible'], errors='ignore').merge(
            tables['stability'][['method', 'overall_eligible']], on='method', validate='one_to_one')
    f = overall(alt, config, methods)
    f['scenario'] = 'downsampling_only_stability'
    alternate.append(f[['scenario', 'method', 'score', 'rank']])
    return {'weight_grid': pd.DataFrame(weights), 'weight_sensitivity_long': long,
            'weight_sensitivity_summary': ranks, 'alternative_definitions': pd.concat(alternate, ignore_index=True)}
