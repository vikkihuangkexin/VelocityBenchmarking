"""Recompute usability from the 24 item scores and declared category rules."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def calculate(inputs, config):
    from common.core import combine_components
    rules = inputs.read(config['usability_rules_csv'])
    items = inputs.read(config['usability_items_csv'])
    items['method'] = [inputs.canonical(m, config['usability_items_csv'], i + 2)
                       for i, m in enumerate(items.method)]
    items = items[items.method.notna()].copy()
    cats = config['usability_category_weights']
    if len(rules) != 24 or rules.duplicated(['category', 'subcategory']).any():
        raise ValueError('Usability rules require 24 unique subitems')
    if items.duplicated(['method', 'category', 'subcategory']).any():
        raise ValueError('Duplicate usability item')
    if set(rules.category) != set(cats) or not np.isclose(sum(cats.values()), 1):
        raise ValueError('Invalid usability category weights')
    for category, weight in cats.items():
        r = rules[rules.category.eq(category)]
        if not np.allclose(r.category_weight, weight) or not np.isclose(r.subcategory_weight.sum(), 1):
            raise ValueError(f'Invalid usability subweights: {category}')
    expected_counts = config.get('usability_expected_item_counts', {})
    for category, count in expected_counts.items():
        if int(rules.category.eq(category).sum()) != count:
            raise ValueError(f'Unexpected usability item count for {category}')
    if not items.category.isin(cats).all():
        raise ValueError('Unknown usability category')
    checked = items.merge(rules[['category', 'subcategory', 'category_weight', 'subcategory_weight']],
                          on=['category', 'subcategory'], how='left', suffixes=('', '_rule'), validate='many_to_one')
    if checked.category_weight_rule.isna().any():
        raise ValueError('Unknown usability subcategory')
    for col in ['category_weight', 'subcategory_weight']:
        if not np.allclose(checked[col], checked[col + '_rule']):
            raise ValueError(f'Usability item/rule mismatch: {col}')
    checked['subscore'] = pd.to_numeric(checked.subscore, errors='raise')
    valid = checked.subscore.dropna()
    if not valid.between(0, 10).all():
        raise ValueError('Usability subscore outside 0..10')
    checked['recomputed_contribution'] = checked.subscore * checked.subcategory_weight_rule
    rows = []
    for method in inputs.methods:
        for category, weight in cats.items():
            r = rules[rules.category.eq(category)]
            f = checked[checked.method.eq(method) & checked.category.eq(category)]
            complete = len(f) == len(r) and f.subscore.notna().all()
            old = f.reported_category_score.dropna().unique() if 'reported_category_score' in f else []
            if len(old) > 1:
                raise ValueError(f'Conflicting reported category scores: {method}/{category}')
            score = f.recomputed_contribution.sum() if complete else np.nan
            rows.append({'method': method, 'category': category, 'category_weight': weight,
                         'score': score, 'n_items': len(f), 'expected_items': len(r),
                         'n_valid_items': int(f.subscore.notna().sum()), 'complete': complete,
                         'reported_category_score': old[0] if len(old) else np.nan,
                         'reported_minus_recomputed': old[0] - score if len(old) else np.nan})
    categories = pd.DataFrame(rows)
    wide = categories.pivot(index='method', columns='category', values='score')
    result = combine_components(wide, inputs.methods, cats, strict=True)
    return checked, categories, result


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run import main
    main(forced_stage='usability')
