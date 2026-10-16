# scRNA Scalability ranking

Companion table: [scalability_rank.csv](scalability_rank.csv). One row per method; 28 methods and 28 available category scores.

These definitions describe the existing table. Scores, ranks, component values, and missing values are unchanged.

## Calculation and interpretation

The expected grid has 136 entries per method: 68 selected workloads times two measures (runtime and CPU RAM). The three source files include GPU VRAM for display, but GPU measurements do not contribute to score, n_z_valid, n_expected, coverage, or n_raw_valid.

Runtime and CPU RAM are reversed before sample-z standardization within each workload. Averages of those unitless values are used as components; column suffixes identify the raw input units, not the units of the component scores.

## Columns

Column names below follow the CSV header order exactly.

| Column | Meaning |
|---|---|
| `method` | Canonical method name used in the source matrices and fixed modality roster. |
| `score` | Equal-weight mean of the available runtime and CPU RAM reverse-z components. Unitless; higher means better resource efficiency. GPU VRAM does not enter this score. |
| `rank` | Descending rank of score within this modality and category. Rank 1 is best. Exact score ties receive the average occupied rank. Missing scores have missing ranks. |
| `reversed_rank` | 28 + 1 - rank, where 28 is the full modality roster size, including methods with missing scores. Higher is better. Missing scientific ranks remain missing. |
| `normalized_rank` | reversed_rank / 28. A rank-derived value, not a normalized category score. The best rank maps to 1; missing ranks remain missing. |
| `fixed_method_order` | Original 1-based position in the fixed methods.csv roster. Used only to keep deterministic ordering of ties and missing rows. It is not a performance rank. |
| `display_order` | 1-based row position after sorting scientific rank and then fixed_method_order, with missing ranks last. This plotting order is unique even for ties and must not replace rank. |
| `n_components` | Number of nonmissing component values supplied to this score calculation. A count of metric or category components, not datasets or successful runs. |
| `expected_components` | Number of configured components for this category. Compare with n_components to identify incomplete component coverage. |
| `missing_components` | Semicolon-separated names of components without usable values. An empty field means none are missing; it is not a zero score. |
| `component_runtime_seconds` | Mean reverse sample-z runtime score over the selected workloads. Computed as (workload mean runtime - method runtime) / workload sample SD. Unitless; higher means faster relative performance, not seconds. |
| `effective_weight_runtime_seconds` | Actual weight of this component in the category score. Configured components are equally weighted; available components are renormalized to sum to 1. The weight is blank when this component is missing. |
| `component_ram_gb` | Mean reverse sample-z CPU RAM score over the selected workloads. Computed as (workload mean RAM - method RAM) / workload sample SD. Unitless; higher means lower relative RAM use, not GB. |
| `effective_weight_ram_gb` | Actual weight of this component in the category score. Configured components are equally weighted; available components are renormalized to sum to 1. The weight is blank when this component is missing. |
| `n_z_valid` | Number of resource workload-by-measure entries with a valid standardized z value. Counts runtime and CPU RAM separately; does not include display-only GPU VRAM. |
| `n_expected` | Number of entries in the selected resource workload-by-measure grid per method. Runtime and CPU RAM each contribute one entry per workload. |
| `coverage` | n_z_valid / n_expected. Fraction of resource entries usable for standardization. A raw valid entry can lack z if fewer than two methods are valid or the workload SD is zero. |
| `n_raw_valid` | Number of nonmissing raw runtime/CPU RAM measurements in the selected resource grid, before standardization. A true numeric zero is counted as a measurement, not treated as missing. |
| `entire_dimension_failure` | True exactly when n_raw_valid is zero: no usable raw runtime or CPU RAM measurements for this method. It does not refer to missing GPU data. |
| `display_rank` | Plotting rank, identical to scientific rank in this modality. A method with no resource score keeps a missing display rank; no spatial failure exception is applied. |
| `effective_reversed_rank` | Reversed rank available for Overall, identical to reversed_rank in this modality. Missing scientific ranks remain missing. |
| `n_source_inputs_expected` | Number of declared source files for this category. This counts files, not methods, datasets, subitems, or valid numeric measurements. |
| `n_source_inputs_available` | Number of those declared source files that exist. A present file may still contain missing values for a particular method. |
| `input_status` | Source-file availability: complete when all declared files exist, partial when some exist, unavailable when none exist. This does not measure numeric task coverage. |
| `overall_eligible` | Source-file eligibility flag: True when this category has its complete declared file set. It is not a guarantee that this method has a category score or a complete Overall score. |

## Missing values and source status

An empty numeric field means unavailable, not zero. An empty missing_components field means no component is missing. True zeros and negative standardized scores remain valid when present. Source-file completeness, numeric task coverage, and availability of a method score are separate concepts.

Calculation implementation: `Figure_code/scalability/compute_scalability_rank.py` and `Figure_code/common/core.py`. Configuration: `Figure_code/config/scRNA.json`.
