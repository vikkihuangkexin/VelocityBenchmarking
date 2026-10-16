"""RNA: downsampling plus measured batch-run SD; multimodal: downsampling only."""
from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd


def calculate(inputs, config):
    from common.core import task_z, sample_z, combine_components
    raw = inputs.matrices(config['downsampling_inputs'])
    if raw.empty:
        if not config.get('allow_missing_sources'):
            raise ValueError('No downsampling inputs configured')
        components = pd.DataFrame({'downsampling': np.nan}, index=inputs.methods)
        result = combine_components(components, inputs.methods)
        result['downsampling_task_score'] = np.nan
        result['batchrun_normalized_variability'] = np.nan
        result['downsampling_n_valid'] = 0
        result['downsampling_n_expected'] = np.nan
        result['batchrun_measured'] = False
        result['stability_mode'] = config['stability_mode']
        return {'downsampling_only_sensitivity': result.copy()}, result
    keys = ['metric', 'data_type', 'dataset_id']
    down_z = task_z(raw, keys)
    # Equal metrics within each perturbation; equal perturbations within each
    # configured family (cell/gene), then equal available perturbation families.
    down_dataset = down_z.groupby(['method', 'data_type', 'dataset_id'], as_index=False).agg(
        downsampling_z=('z', 'mean'), n_valid_metrics=('z', 'count'), n_expected_metrics=('z', 'size'))
    down_dataset['n_supplied_metrics'] = down_dataset.n_expected_metrics
    down_dataset['n_expected_metrics'] = len({e['metric'] for e in config['downsampling_inputs'] if isinstance(e, dict)}) or down_dataset.n_expected_metrics
    groups = config['downsampling_groups']
    def group_of(dataset):
        hits = [name for name, pattern in groups.items() if re.search(pattern, dataset)]
        if len(hits) != 1:
            raise ValueError(f'Downsampling task must match exactly one group: {dataset}, {hits}')
        return hits[0]
    down_dataset['perturbation_group'] = down_dataset.dataset_id.map(group_of)
    group_means = down_dataset.groupby(['method', 'perturbation_group']).downsampling_z.mean().unstack()
    down_score = group_means.mean(axis=1).reindex(inputs.methods)
    down_standardized = sample_z(down_score)
    components = pd.DataFrame({'downsampling': down_standardized}, index=inputs.methods)
    details = {'downsampling_raw': raw, 'downsampling_task_z': down_z,
               'downsampling_one_value_per_dataset': down_dataset,
               'downsampling_group_scores': group_means.reset_index()}
    batch_method = pd.Series(np.nan, index=inputs.methods, dtype=float)
    if config['stability_mode'] == 'combined':
        std = inputs.matrices(config['batch_sd_inputs'])
        if std.empty:
            raise ValueError('Combined RNA Stability requires measured batch-run SD inputs')
        # Positive z here: smaller = less variability. Keep exact SD zeros.
        sd_z = task_z(std, keys)
        batch_dataset = sd_z.groupby(['method', 'data_type', 'dataset_id'], as_index=False).agg(
            normalized_variability=('z', 'mean'), n_valid_metrics=('z', 'count'),
            n_expected_metrics=('z', 'size'), n_true_sd_zeros=('value', lambda x: int(x.eq(0).sum())))
        bm = batch_dataset.groupby(['method', 'data_type']).normalized_variability.mean().unstack()
        # Strict 50:50 when the SD dataset includes both real and simulation.
        if {'real', 'simulation'}.issubset(bm.columns):
            batch_method = (.5 * bm.real + .5 * bm.simulation).reindex(inputs.methods)
        else:
            batch_method = bm.mean(axis=1).reindex(inputs.methods)
        # Restore common component scale before merging; high stability = low SD.
        components['batchrun'] = sample_z(-batch_method)
        details.update({'batch_sd_raw': std, 'batch_sd_task_z': sd_z,
                        'batch_sd_one_value_per_dataset': batch_dataset})
    elif config['stability_mode'] != 'downsampling_only':
        raise ValueError('stability_mode must be combined or downsampling_only')
    result = combine_components(components, inputs.methods,
                                {k: config['stability_weights'][k] for k in components})
    result['downsampling_task_score'] = result.method.map(down_score)
    result['batchrun_normalized_variability'] = result.method.map(batch_method)
    count = down_z.groupby('method').z.count()
    result['downsampling_n_valid'] = result.method.map(count).fillna(0).astype(int)
    result['downsampling_n_expected'] = down_z.groupby('method').size().reindex(result.method).to_numpy()
    result['batchrun_measured'] = result.method.isin(batch_method.dropna().index)
    result['stability_mode'] = config['stability_mode']
    # This explicitly labeled sensitivity never replaces the configured main score.
    details['downsampling_only_sensitivity'] = combine_components(
        components[['downsampling']], inputs.methods)
    return details, result


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run import main
    main(forced_stage='stability')
