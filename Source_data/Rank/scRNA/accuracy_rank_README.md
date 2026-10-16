# scRNA Accuracy ranking

Companion table: [accuracy_rank.csv](accuracy_rank.csv). One row per method; 28 methods and 28 available category scores.

These definitions describe the existing table. Scores, ranks, component values, and missing values are unchanged.

## Calculation and interpretation

Per-task z uses the sample SD (ddof=1) across the selected method roster. Fewer than two usable methods or zero SD yields a missing z. Valid raw zeros and negative values are retained. Negative component scores indicate below-task-average relative performance, not invalid measurements.

For metrics present in both data types, real and simulation dataset means receive equal 50:50 weights, independent of their dataset counts. Ground-truth correlation, where present, uses simulations only. Available metric components receive equal weights with their effective weights reported explicitly.

## Columns

Column names below follow the CSV header order exactly.

| Column | Meaning |
|---|---|
| `method` | Canonical method name used in the source matrices and fixed modality roster. |
| `score` | Equal-weight mean of the available standardized directional metric components listed below. Unitless; higher is better. A fully missing component set yields a missing score. |
| `rank` | Descending rank of score within this modality and category. Rank 1 is best. Exact score ties receive the average occupied rank. Missing scores have missing ranks. |
| `reversed_rank` | 28 + 1 - rank, where 28 is the full modality roster size, including methods with missing scores. Higher is better. Missing scientific ranks remain missing. |
| `normalized_rank` | reversed_rank / 28. A rank-derived value, not a normalized category score. The best rank maps to 1; missing ranks remain missing. |
| `fixed_method_order` | Original 1-based position in the fixed methods.csv roster. Used only to keep deterministic ordering of ties and missing rows. It is not a performance rank. |
| `display_order` | 1-based row position after sorting scientific rank and then fixed_method_order, with missing ranks last. This plotting order is unique even for ties and must not replace rank. |
| `n_components` | Number of nonmissing component values supplied to this score calculation. A count of metric or category components, not datasets or successful runs. |
| `expected_components` | Number of configured components for this category. Compare with n_components to identify incomplete component coverage. |
| `missing_components` | Semicolon-separated names of components without usable values. An empty field means none are missing; it is not a zero score. |
| `component_Angle consistency` | Unitless standardized score for the Angle consistency directional metric. Average valid dataset z values within real and simulation separately, then combine those two means 50:50. If a required type has no usable values, the component is missing. Higher is better; this is not a raw metric value or a rank. |
| `effective_weight_Angle consistency` | Actual weight of this component in the category score. Configured components are equally weighted; available components are renormalized to sum to 1. The weight is blank when this component is missing. |
| `component_CBDir` | Unitless standardized score for the CBDir directional metric. Average valid dataset z values within real and simulation separately, then combine those two means 50:50. If a required type has no usable values, the component is missing. Higher is better; this is not a raw metric value or a rank. |
| `effective_weight_CBDir` | Actual weight of this component in the category score. Configured components are equally weighted; available components are renormalized to sum to 1. The weight is blank when this component is missing. |
| `component_Transition V1` | Unitless standardized score for the Transition score version 1 directional metric. Average valid dataset z values within real and simulation separately, then combine those two means 50:50. If a required type has no usable values, the component is missing. Higher is better; this is not a raw metric value or a rank. |
| `effective_weight_Transition V1` | Actual weight of this component in the category score. Configured components are equally weighted; available components are renormalized to sum to 1. The weight is blank when this component is missing. |
| `component_Ground-truth correlation` | Unitless standardized score for the simulation ground-truth correlation metric. Mean of valid per-dataset sample-z values in simulations only. Higher is better; this is not a raw metric value or a rank. |
| `effective_weight_Ground-truth correlation` | Actual weight of this component in the category score. Configured components are equally weighted; available components are renormalized to sum to 1. The weight is blank when this component is missing. |
| `n_source_inputs_expected` | Number of declared source files for this category. This counts files, not methods, datasets, subitems, or valid numeric measurements. |
| `n_source_inputs_available` | Number of those declared source files that exist. A present file may still contain missing values for a particular method. |
| `input_status` | Source-file availability: complete when all declared files exist, partial when some exist, unavailable when none exist. This does not measure numeric task coverage. |
| `overall_eligible` | Source-file eligibility flag: True when this category has its complete declared file set. It is not a guarantee that this method has a category score or a complete Overall score. |

## Missing values and source status

An empty numeric field means unavailable, not zero. An empty missing_components field means no component is missing. True zeros and negative standardized scores remain valid when present. Source-file completeness, numeric task coverage, and availability of a method score are separate concepts.

Calculation implementation: `Figure_code/accuracy/compute_accuracy_metrics.py` and `Figure_code/common/core.py`. Configuration: `Figure_code/config/scRNA.json`.
