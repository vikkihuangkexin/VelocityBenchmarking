# ATAC Overall ranking

Companion table: [overall_rank.csv](overall_rank.csv). One row per method; 9 methods and 9 available category scores.

These definitions describe the existing table. Scores, ranks, component values, and missing values are unchanged.

## Calculation and interpretation

All five components are reversed ranks based on the full roster N=9. They are not raw z scores, usability scores, normalized scores, or final Overall ranks. Exact Overall score ties retain average ranks.

All five required dimension components must be present. If a dimension is missing, the Overall score and effective weights remain missing. Configured weights and available configured weighted terms may remain populated for inspection; they do not define a partial Overall score.

## Columns

Column names below follow the CSV header order exactly.

| Column | Meaning |
|---|---|
| `method` | Canonical method name used in the source matrices and fixed modality roster. |
| `score` | Weighted sum of five category reversed ranks: 0.65 Accuracy + 0.10 Consistency + 0.10 Scalability + 0.075 Stability + 0.075 Usability. Higher is better. Missing required components yield a missing Overall score. |
| `rank` | Descending rank of score within this modality and category. Rank 1 is best. Exact score ties receive the average occupied rank. Missing scores have missing ranks. |
| `reversed_rank` | 9 + 1 - rank, where 9 is the full modality roster size, including methods with missing scores. Higher is better. Missing scientific ranks remain missing. |
| `normalized_rank` | reversed_rank / 9. A rank-derived value, not a normalized category score. The best rank maps to 1; missing ranks remain missing. |
| `fixed_method_order` | Original 1-based position in the fixed methods.csv roster. Used only to keep deterministic ordering of ties and missing rows. It is not a performance rank. |
| `display_order` | 1-based row position after sorting scientific rank and then fixed_method_order, with missing ranks last. This plotting order is unique even for ties and must not replace rank. |
| `n_components` | Number of nonmissing component values supplied to this score calculation. A count of metric or category components, not datasets or successful runs. |
| `expected_components` | Number of configured components for this category. Compare with n_components to identify incomplete component coverage. |
| `missing_components` | Semicolon-separated names of components without usable values. An empty field means none are missing; it is not a zero score. |
| `component_accuracy` | Accuracy reversed rank from its own category table, using the full modality roster. This is not its raw category score. |
| `effective_weight_accuracy` | Weight of the Accuracy reversed-rank component when the full Overall score is available: 0.65. Blank when Overall is unavailable. Weights are not redistributed across missing dimensions. |
| `component_consistency` | Consistency reversed rank from its own category table, using the full modality roster. This is not its raw category score. |
| `effective_weight_consistency` | Weight of the Consistency reversed-rank component when the full Overall score is available: 0.1. Blank when Overall is unavailable. Weights are not redistributed across missing dimensions. |
| `component_scalability` | Scalability reversed rank from its own category table, using the full modality roster. This is not its raw category score. |
| `effective_weight_scalability` | Weight of the Scalability reversed-rank component when the full Overall score is available: 0.1. Blank when Overall is unavailable. Weights are not redistributed across missing dimensions. |
| `component_stability` | Stability reversed rank from its own category table, using the full modality roster. This is not its raw category score. |
| `effective_weight_stability` | Weight of the Stability reversed-rank component when the full Overall score is available: 0.075. Blank when Overall is unavailable. Weights are not redistributed across missing dimensions. |
| `component_usability` | Usability reversed rank from its own category table, using the full modality roster. This is not its raw category score. |
| `effective_weight_usability` | Weight of the Usability reversed-rank component when the full Overall score is available: 0.075. Blank when Overall is unavailable. Weights are not redistributed across missing dimensions. |
| `normalized_overall_score` | Overall score / 9. Normalizes the weighted reversed-rank score by the full modality roster size. This differs from normalized_rank, which is derived from the final Overall rank. |
| `score_definition` | Literal weighted_reversed_ranks: Overall combines category reversed ranks, rather than raw dimension scores or category rank numbers. |
| `weighted_accuracy_contribution` | configured_weight_accuracy times component_accuracy. This configured weighted term can remain populated even if Overall is missing because another dimension is missing; it must not be summed into a substitute partial Overall score. |
| `configured_weight_accuracy` | Fixed configuration weight for Accuracy: 0.65. Retained even when the final Overall score is missing. This is not estimated from data. |
| `weighted_consistency_contribution` | configured_weight_consistency times component_consistency. This configured weighted term can remain populated even if Overall is missing because another dimension is missing; it must not be summed into a substitute partial Overall score. |
| `configured_weight_consistency` | Fixed configuration weight for Consistency: 0.1. Retained even when the final Overall score is missing. This is not estimated from data. |
| `weighted_scalability_contribution` | configured_weight_scalability times component_scalability. This configured weighted term can remain populated even if Overall is missing because another dimension is missing; it must not be summed into a substitute partial Overall score. |
| `configured_weight_scalability` | Fixed configuration weight for Scalability: 0.1. Retained even when the final Overall score is missing. This is not estimated from data. |
| `weighted_stability_contribution` | configured_weight_stability times component_stability. This configured weighted term can remain populated even if Overall is missing because another dimension is missing; it must not be summed into a substitute partial Overall score. |
| `configured_weight_stability` | Fixed configuration weight for Stability: 0.075. Retained even when the final Overall score is missing. This is not estimated from data. |
| `weighted_usability_contribution` | configured_weight_usability times component_usability. This configured weighted term can remain populated even if Overall is missing because another dimension is missing; it must not be summed into a substitute partial Overall score. |
| `configured_weight_usability` | Fixed configuration weight for Usability: 0.075. Retained even when the final Overall score is missing. This is not estimated from data. |

## Missing values and source status

An empty numeric field means unavailable, not zero. An empty missing_components field means no component is missing. True zeros and negative standardized scores remain valid when present. Source-file completeness, numeric task coverage, and availability of a method score are separate concepts.

Calculation implementation: `Figure_code/overall/compute_overall_rank.py` and `Figure_code/common/core.py`. Configuration: `Figure_code/config/ATAC.json`.
