"""Consistency score and ICCoh are independent of directional Accuracy."""
from pathlib import Path
import sys


def calculate(z, config, methods):
    from common.core import dimension_from_metrics
    return dimension_from_metrics(z, config['consistency_metrics'], methods, config['data_type_weighting'])


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run import main
    main(forced_stage='consistency')
