"""Five-dimension Overall: weighted reversed ranks; missing dimensions stay NA."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def calculate(tables, config, methods):
    from common.core import DIMENSIONS, combine_components
    weights = config['overall_weights']
    if set(weights) != set(DIMENSIONS) or not np.isclose(sum(weights.values()), 1):
        raise ValueError('Overall weights must cover the five dimensions and sum to one')
    if config.get('overall_missing_policy') != 'strict':
        raise ValueError('Main Overall requires strict missing policy; no zero-fill or renormalization')
    # Use comparable ranks, rather than adding U in 0..10 units to scientific z.
    components = pd.DataFrame({d: tables[d].set_index('method').reversed_rank for d in DIMENSIONS})
    if config.get('scalability_all_failed_policy') == 'bottom_tie_effective_zero':
        components['scalability'] = tables['scalability'].set_index('method').effective_reversed_rank
    for d in DIMENSIONS:
        if 'overall_eligible' in tables[d]:
            eligibility = tables[d].set_index('method').overall_eligible.reindex(components.index)
            components.loc[~eligibility.fillna(False).astype(bool), d] = np.nan
    result = combine_components(components, methods, weights, strict=True)
    result['normalized_overall_score'] = result.score / len(methods)
    result['score_definition'] = 'weighted_reversed_ranks'
    # A strict Overall row with missing dimensions has no actual contributions.
    for d in DIMENSIONS:
        result[f'weighted_{d}_contribution'] = weights[d] * result[f'component_{d}']
        result[f'configured_weight_{d}'] = weights[d]
        result[f'effective_weight_{d}'] = np.where(result.score.notna(), weights[d], np.nan)
    return result


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run import main
    main(forced_stage='overall')
