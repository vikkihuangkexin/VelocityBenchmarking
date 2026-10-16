"""Focused scientific-rule regression tests; fixtures are synthetic unit-test data."""
import json
from pathlib import Path
import sys
import unittest
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.core import sample_z, task_z, rank_scores, combine_components, metric_summary
from overall.compute_overall_rank import calculate as overall
from stability.compute_stability_rank import calculate as stability
from scalability.compute_scalability_rank import apply_all_failed_policy
from common.inputs import Inputs
from stability.prepare_atac_downsampling import prepare as prepare_atac


class CalculationRules(unittest.TestCase):
    def test_sample_sd_and_true_zero(self):
        actual = sample_z(pd.Series([0.0, 1.0, 2.0, np.nan]))
        np.testing.assert_allclose(actual, [-1, 0, 1, np.nan], equal_nan=True)

    def test_equal_values_and_single_method_stay_missing(self):
        self.assertTrue(sample_z(pd.Series([0.0, 0.0])).isna().all())
        self.assertTrue(sample_z(pd.Series([1.0, np.nan])).isna().all())

    def test_negative_metrics_are_not_missing(self):
        f = pd.DataFrame({'method': ['a', 'b'], 'task': ['x', 'x'], 'value': [-.5, .5]})
        z = task_z(f, ['task'])
        self.assertTrue(z.z.notna().all())
        self.assertLess(z.z.iloc[0], 0)

    def test_reverse_cost_and_missing_rank(self):
        f = pd.DataFrame({'method': ['a', 'b', 'c'], 'task': ['x'] * 3, 'value': [1, 2, np.nan]})
        z = task_z(f, ['task'], lower_is_better=True)
        self.assertGreater(z.z.iloc[0], z.z.iloc[1])
        r = rank_scores(pd.Series({'a': 1.0, 'b': 1.0, 'c': np.nan}), ['a', 'b', 'c']).set_index('method')
        self.assertEqual(r.loc['a', 'rank'], 1.5)
        self.assertTrue(pd.isna(r.loc['c', 'rank']))

    def test_no_batchrun_is_not_zero(self):
        components = pd.DataFrame({'downsampling': [1., 1.], 'batchrun': [-1., np.nan]}, index=['a', 'b'])
        r = combine_components(components, ['a', 'b']).set_index('method')
        self.assertEqual(r.loc['a', 'score'], 0)
        self.assertEqual(r.loc['b', 'score'], 1)
        self.assertTrue(pd.isna(r.loc['b', 'component_batchrun']))

    def test_balanced_type_requires_both_and_preserves_zero(self):
        f = pd.DataFrame({'method': ['a', 'a'], 'metric': ['CBDir'] * 2,
                          'data_type': ['real', 'simulation'], 'value': [0., np.nan], 'z': [1., np.nan]})
        self.assertTrue(pd.isna(metric_summary(f, 'balanced').score.iloc[0]))
        self.assertEqual(metric_summary(f, 'pooled').score.iloc[0], 1)

    def test_strict_overall_missing_dimension(self):
        dimensions = ['accuracy', 'consistency', 'scalability', 'stability', 'usability']
        tables = {d: pd.DataFrame({'method': ['a', 'b'], 'reversed_rank': [2., 1.]}) for d in dimensions}
        tables['stability'].loc[1, 'reversed_rank'] = np.nan
        cfg = {'overall_weights': dict(zip(dimensions, [.65, .1, .1, .075, .075])), 'overall_missing_policy': 'strict'}
        r = overall(tables, cfg, ['a', 'b']).set_index('method')
        self.assertAlmostEqual(r.loc['a', 'score'], 2)
        self.assertTrue(pd.isna(r.loc['b', 'score']))
        self.assertTrue(pd.isna(r.loc['b', 'rank']))

    def test_multimodal_stability_never_reads_batchrun(self):
        class UnitInputs:
            methods = ['a', 'b', 'c']
            def matrices(self, entries):
                if entries != ['down']:
                    raise AssertionError('Attempted to read batchrun for multimodal Stability')
                return pd.DataFrame([{'method': m, 'metric': metric, 'data_type': 'simulation',
                                      'dataset_id': dataset, 'value': value}
                    for metric in ['CBDir', 'ICCoh'] for dataset in ['cellsub-x', 'genesub-x']
                    for m, value in zip(self.methods, [1., 2., 3.])])
        cfg = {'downsampling_inputs': ['down'], 'downsampling_groups': {'cell': '^cellsub-', 'gene': '^genesub-'},
               'stability_mode': 'downsampling_only', 'stability_weights': {'downsampling': 1.0}}
        detail, r = stability(UnitInputs(), cfg)
        self.assertNotIn('batch_sd_raw', detail)
        self.assertEqual(r.loc[r['rank'].eq(1), 'method'].iloc[0], 'c')
        self.assertTrue(r.n_components.eq(1).all())

    def test_spatial_entire_failure_has_separate_display_and_effective_rank(self):
        methods = list('abcdefg')
        f = rank_scores(pd.Series([5., 4., 3., 2., 1., np.nan, np.nan], index=methods), methods)
        f = apply_all_failed_policy(f, pd.Series([11, 11, 11, 11, 11, 0, 0], index=methods), methods,
                                    'bottom_tie_effective_zero').set_index('method')
        self.assertEqual(f.loc['f', 'display_rank'], 6.5)
        self.assertEqual(f.loc['g', 'effective_reversed_rank'], 0)
        self.assertTrue(pd.isna(f.loc['f', 'score']))

    def test_derived_real_downsampling_does_not_use_main_real_id_map(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'methods.csv').write_text('method_order,method\n1,a\n2,b\n')
            (p / 'ids.csv').write_text('dataset_number,dataset_id\n55,55_Gg_Heart\n')
            (p / 'down.csv').write_text('method,cellsub-chickenheart_60pct\na,1\nb,2\n')
            reader = Inputs(p, {'methods_csv': 'methods.csv', 'real_id_map_csv': 'ids.csv'})
            f = reader.matrix({'path': 'down.csv', 'metric': 'CBDir', 'data_type': 'real', 'role': 'downsampling'})
            self.assertEqual(f.dataset_id.unique().tolist(), ['cellsub-chickenheart_60pct'])

    def test_provisional_dimension_does_not_enter_overall(self):
        dimensions = ['accuracy', 'consistency', 'scalability', 'stability', 'usability']
        tables = {d: pd.DataFrame({'method': ['a', 'b'], 'reversed_rank': [2., 1.],
                                  'overall_eligible': [True, True]}) for d in dimensions}
        tables['consistency']['overall_eligible'] = False
        cfg = {'overall_weights': dict(zip(dimensions, [.65, .1, .1, .075, .075])), 'overall_missing_policy': 'strict'}
        self.assertTrue(overall(tables, cfg, ['a', 'b']).score.isna().all())

    def test_spatial_metric_contract(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'config/Spatial.json').read_text())
        self.assertEqual(cfg['consistency_metrics'], ['STC', 'SVC'])
        self.assertEqual({e['metric'] for e in cfg['downsampling_inputs']},
                         {'Angle consistency', 'CBDir', 'Transition V1', 'STC', 'SVC'})
        self.assertFalse(any(e.get('optional') for e in cfg['measurement_inputs'] + cfg['downsampling_inputs']))

    def test_cell_level_gt_aggregation_honors_scored_and_true_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            folder = p / 'measurements_raw/ATAC'
            folder.mkdir(parents=True)
            (folder / 'methods.csv').write_text('method_order,method\n1,a\n2,b\n')
            raw = p / 'raw.csv'
            raw.write_text('method,id,cell_id,cosine,rescaled_cosine,scored\n'
                           'a,26_downsample60_r1,c1,-1,0,True\n'
                           'a,26_downsample60_r1,c2,1,1,False\n'
                           'b,26_downsample60_r1,c1,0,0.5,True\n')
            actual = prepare_atac(raw, p)
            self.assertEqual(actual.loc['a'].iloc[0], 0)
            self.assertEqual(actual.loc['b'].iloc[0], .5)


if __name__ == '__main__':
    unittest.main()
