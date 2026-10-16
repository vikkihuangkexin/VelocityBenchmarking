#!/usr/bin/env python3
"""One entry point for current calculations; defaults to the repository's Source_data."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from common.core import DIMENSIONS, task_z, operational_scores
from common.inputs import Inputs


ROOT = Path(__file__).resolve().parent
MODULES = {
    'accuracy': 'accuracy.compute_accuracy_metrics',
    'consistency': 'consistency.compute_consistency_metrics',
    'scalability': 'scalability.compute_scalability_rank',
    'stability': 'stability.compute_stability_rank',
    'usability': 'usability.compute_usability_rank',
    'overall': 'overall.compute_overall_rank',
}


def save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g')


def validate_config(cfg):
    if cfg.get('profile') not in {'scRNA', 'ATAC', 'Multiome', 'Spatial'}:
        raise ValueError('profile must be scRNA, ATAC or Spatial')
    if cfg['profile'] in {'ATAC', 'Multiome', 'Spatial'} and cfg['stability_mode'] != 'downsampling_only':
        raise ValueError('Multiome/Spatial Stability must use downsampling only')
    all_metrics = {e['metric'] for e in cfg['measurement_inputs']}
    if set(cfg['accuracy_metrics']) & set(cfg['consistency_metrics']):
        raise ValueError('Accuracy/Consistency metric lists must be disjoint')
    for component in ['accuracy_metrics', 'consistency_metrics']:
        if not cfg[component] or not set(cfg[component]).issubset(all_metrics):
            raise ValueError(f'Unknown or empty metric list: {component}')
    if cfg['stability_mode'] == 'combined' and not cfg['batch_sd_inputs']:
        raise ValueError('Combined RNA Stability needs batch-run SD inputs')
    allowed_types = {'real', 'simulation'}
    for section in ['measurement_inputs', 'downsampling_inputs', 'batch_sd_inputs']:
        if any(e['data_type'] not in allowed_types for e in cfg[section]):
            raise ValueError(f'Invalid data_type in {section}')


def main(forced_stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--profile', choices=['scRNA', 'ATAC', 'Spatial', 'all'], default='scRNA')
    parser.add_argument('--source-root', type=Path, default=ROOT.parent / 'Source_data')
    parser.add_argument('--output', type=Path, help='New/empty output directory; default Results/PROFILE')
    parser.add_argument('--stage', choices=['all', 'check', *DIMENSIONS, 'overall', 'sensitivity'], default='all')
    parser.add_argument('--skip-validation', action='store_true', help='Skip independent output verification')
    args = parser.parse_args()
    if args.profile == 'all':
        if args.config:
            raise ValueError('--config cannot be combined with --profile all')
        for profile in ['scRNA', 'ATAC', 'Spatial']:
            command = [sys.executable, str(ROOT / 'run.py'), '--profile', profile,
                       '--source-root', str(args.source_root), '--stage', forced_stage or args.stage]
            if args.output:
                command += ['--output', str(args.output / profile)]
            if args.skip_validation:
                command.append('--skip-validation')
            subprocess.run(command, check=True)
        return
    stage = forced_stage or args.stage
    args.config = args.config or ROOT / 'config' / (args.profile + '.json')
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(cfg)
    source = args.source_root.resolve()
    out = (args.output or ROOT.parent / 'Results' / cfg['profile']).resolve()
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Output must be separate from Source_data')
    if out == ROOT or ROOT.is_relative_to(out) or out.is_relative_to(ROOT):
        raise ValueError('Place calculation outputs outside Figure_code')
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Use a new/empty output directory: {out}')
    inputs = Inputs(source, cfg)
    if stage == 'check':
        inputs.matrices(cfg['measurement_inputs'])
        inputs.matrices(cfg['downsampling_inputs'])
        if cfg['stability_mode'] == 'combined':
            inputs.matrices(cfg['batch_sd_inputs'])
        for p in cfg['resource_csvs']:
            inputs.long_resource(p)
        inputs.read(cfg['usability_items_csv'])
        inputs.read(cfg['usability_rules_csv'])
        inputs.save_audit(out / 'audit')
        print(f'Input schema check passed: {len(inputs.methods)} methods, {len(inputs.manifest)} supplied files, '
              f'{len(inputs.missing_inputs)} explicitly unavailable source files')
        return
    selected = DIMENSIONS if stage in {'all', 'overall', 'sensitivity'} else [stage]
    tables = {}
    z = None
    if set(selected) & {'accuracy', 'consistency'}:
        raw = inputs.matrices(cfg['measurement_inputs'])
        z = task_z(raw, ['metric', 'data_type', 'dataset_id'])
        save(z, out / 'audit/measurement_task_z.csv')
        cov = z.groupby(['method', 'metric', 'data_type'], as_index=False).agg(
            n_raw_valid=('value', 'count'), n_z_valid=('z', 'count'), n_expected=('z', 'size'),
            true_zeros=('value', lambda x: int(x.eq(0).sum())),
            negatives=('value', lambda x: int(x.lt(0).sum())))
        cov['coverage'] = cov.n_raw_valid / cov.n_expected
        save(cov, out / 'audit/coverage.csv')
    for dimension in selected:
        module = importlib.import_module(MODULES[dimension])
        folder = out / dimension
        if dimension in {'accuracy', 'consistency'}:
            summary, table = module.calculate(z, cfg, inputs.methods)
            save(summary, folder / 'metric_scores.csv')
            save(z[z.metric.isin(cfg[dimension + '_metrics'])], folder / 'task_z.csv')
            op_detail, op = operational_scores(z, cfg[dimension + '_metrics'], inputs.methods,
                                                cfg['data_type_weighting'])
            save(op_detail, folder / 'operational_task_q_supplement.csv')
            save(op, folder / 'operational_ranks_supplement.csv')
        elif dimension == 'scalability':
            detail, table, display = module.calculate(inputs, cfg)
            save(detail, folder / 'task_z.csv')
            save(display, folder / 'resources_display_only.csv')
        elif dimension == 'stability':
            details, table = module.calculate(inputs, cfg)
            for name, frame in details.items():
                save(frame, folder / (name + '.csv'))
        elif dimension == 'usability':
            items, cats, table = module.calculate(inputs, cfg)
            save(items, folder / 'item_contributions.csv')
            save(cats, folder / 'category_scores.csv')
        if dimension in {'accuracy', 'consistency'}:
            declared = [e for e in cfg['measurement_inputs'] if e['metric'] in cfg[dimension + '_metrics']]
        elif dimension == 'stability':
            declared = cfg['downsampling_inputs'] + (cfg['batch_sd_inputs'] if cfg['stability_mode'] == 'combined' else [])
        elif dimension == 'scalability':
            declared = [{'path': p} for p in cfg['resource_csvs']]
        else:
            declared = [{'path': cfg['usability_items_csv']}, {'path': cfg['usability_rules_csv']}]
        available = sum((source / e['path']).is_file() for e in declared)
        status = 'complete' if available == len(declared) else 'partial' if available else 'unavailable'
        table['n_source_inputs_expected'] = len(declared)
        table['n_source_inputs_available'] = available
        table['input_status'] = status
        table['overall_eligible'] = status == 'complete'
        tables[dimension] = table
        save(table, folder / 'scores_ranks.csv')
        save(table[['method', 'score', 'rank', 'reversed_rank', 'normalized_rank', 'input_status', 'overall_eligible']],
             out / 'ranks' / (dimension + '_rank.csv'))
        print(f'{dimension}: {int(table.score.notna().sum())}/{len(inputs.methods)} available scores; source set {status}', flush=True)
    if stage in {'all', 'overall', 'sensitivity'}:
        overall = importlib.import_module(MODULES['overall']).calculate(tables, cfg, inputs.methods)
        tables['overall'] = overall
        save(overall, out / 'overall/scores_ranks.csv')
        save(overall[['method', 'score', 'rank', 'reversed_rank', 'normalized_overall_score']],
             out / 'ranks/overall_rank.csv')
        wide = pd.DataFrame(index=pd.Index(inputs.methods, name='method'))
        for dimension, table in tables.items():
            t = table.set_index('method')
            for col in ['score', 'rank', 'reversed_rank']:
                wide[dimension + '_' + col] = t[col]
            if 'input_status' in t:
                wide[dimension + '_input_status'] = t.input_status
        save(wide.reset_index(), out / 'ranks/all_dimensions.csv')
        from sensitivity.compute_weight_sensitivity import calculate
        for name, frame in calculate(tables, z, cfg, inputs.methods, out).items():
            save(frame, out / 'sensitivity' / (name + '.csv'))
        print(f'overall: {int(overall.score.notna().sum())}/{len(inputs.methods)} complete five-dimension scores')
    inputs.save_audit(out / 'audit')
    metadata = {'pipeline_version': '4.0.0', 'profile': cfg['profile'], 'stage': stage,
                'source_root': str(source), 'output_root': str(out),
                'config_path': str(args.config.resolve()), 'n_methods': len(inputs.methods),
                'transition_version': cfg.get('transition_version'),
                'batchrun_n_runs': 'unknown: supplied SD summaries do not encode run count',
                'notes': ['No missing metric/SD/dimension is replaced by zero in the main ranking.',
                          'Operational q failure=0 exists only in the explicitly named supplement.',
                          'Only configured resource protocols enter Scalability; VRAM is display-only.']}
    (out / 'audit/run_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    if stage in {'all', 'overall', 'sensitivity'} and not args.skip_validation:
        subprocess.run([sys.executable, str(ROOT / 'verify.py'), '--results', str(out),
                        '--source-root', str(source)], check=True)
    print(f'Completed: {out}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
