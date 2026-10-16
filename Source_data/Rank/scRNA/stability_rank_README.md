# scRNA Stability ranking

Companion table: [stability_rank.csv](stability_rank.csv). One row per method; 28 methods and 26 available category scores.

These definitions describe the existing table. Scores, ranks, component values, and missing values are unchanged.

## Calculation and interpretation

There are 10 expected metric-by-dataset downsampling entries per method in this release. There is one selected GT metric on ten perturbation datasets, giving ten entries.

Cell and gene perturbation families are averaged separately and then equally combined. Eight methods have usable measured batch-run variability. Both components are standardized again across methods before combining them with configured weights 0.5/0.5. If only downsampling is available its effective weight is 1; an unmeasured batch-run SD is not set to zero. k-velo and GRAVITY lack both components and keep missing Stability scores and ranks.

downsampling_task_score and component_downsampling differ by the final across-method sample-z transformation. The published Stability score is higher-is-better; only batchrun_normalized_variability has the lower-is-better interpretation.

## Columns

Column names below follow the CSV header order exactly.

| Column | Meaning |
|---|---|
| `method` | Canonical method name used in the source matrices and fixed modality roster. |
| `score` | Weighted mean of the available standardized Stability components using their effective weights. Unitless; higher is better. Missing SD is never substituted with zero. |
| `rank` | Descending rank of score within this modality and category. Rank 1 is best. Exact score ties receive the average occupied rank. Missing scores have missing ranks. |
| `reversed_rank` | 28 + 1 - rank, where 28 is the full modality roster size, including methods with missing scores. Higher is better. Missing scientific ranks remain missing. |
| `normalized_rank` | reversed_rank / 28. A rank-derived value, not a normalized category score. The best rank maps to 1; missing ranks remain missing. |
| `fixed_method_order` | Original 1-based position in the fixed methods.csv roster. Used only to keep deterministic ordering of ties and missing rows. It is not a performance rank. |
| `display_order` | 1-based row position after sorting scientific rank and then fixed_method_order, with missing ranks last. This plotting order is unique even for ties and must not replace rank. |
| `n_components` | Number of nonmissing component values supplied to this score calculation. A count of metric or category components, not datasets or successful runs. |
| `expected_components` | Number of configured components for this category. Compare with n_components to identify incomplete component coverage. |
| `missing_components` | Semicolon-separated names of components without usable values. An empty field means none are missing; it is not a zero score. |
| `component_downsampling` | Sample-z standardization across methods of downsampling_task_score. Higher is better. This adds a method-level standardization after the per-task aggregation. |
| `effective_weight_downsampling` | Actual weight of this component among available Stability components. scRNA configures 0.5 downsampling and 0.5 batch-run stability; when only one is available its effective weight is 1. An absent component has a blank weight, not an assumed zero measurement. |
| `component_batchrun` | Sample-z standardization across methods of minus batchrun_normalized_variability. Higher is better because lower measured variability is more stable. Missing measured variability remains missing. |
| `effective_weight_batchrun` | Actual weight of this component among available Stability components. scRNA configures 0.5 downsampling and 0.5 batch-run stability; when only one is available its effective weight is 1. An absent component has a blank weight, not an assumed zero measurement. |
| `downsampling_task_score` | Mean of within-task metric z scores: equal available metrics within a perturbation dataset, then equal datasets within each configured perturbation family, then equal available families. This precedes the final across-method standardization in component_downsampling. |
| `batchrun_normalized_variability` | Method-level variability from measured batch-run SDs: sample-z SD within each metric/dataset across measured methods, average metrics per dataset, average datasets within real/simulation, then combine those types 50:50 when both are present. Lower is more stable; a negative value means below-average relative variability, not negative SD. Do not compare absolute values. Zero is a standardized reference, not proof of zero raw SD. |
| `downsampling_n_valid` | Number of downsampling metric-by-dataset entries with usable z values. This counts metric entries, not necessarily distinct datasets. |
| `downsampling_n_expected` | Number of downsampling metric-by-dataset entries in the declared task grid for this method. This is not the number of successful runs. |
| `batchrun_measured` | True when a usable method-level batch-run variability score is available after the configured aggregation. False does not mean SD=0 or deterministic behavior. |
| `stability_mode` | Configured Stability mode: combined for scRNA, downsampling_only for ATAC and spatial. combined allows the recorded available-component weighting; it does not prove both components were measured for every method. |
| `n_source_inputs_expected` | Number of declared source files for this category. This counts files, not methods, datasets, subitems, or valid numeric measurements. |
| `n_source_inputs_available` | Number of those declared source files that exist. A present file may still contain missing values for a particular method. |
| `input_status` | Source-file availability: complete when all declared files exist, partial when some exist, unavailable when none exist. This does not measure numeric task coverage. |
| `overall_eligible` | Source-file eligibility flag: True when this category has its complete declared file set. It is not a guarantee that this method has a category score or a complete Overall score. |

## Missing values and source status

An empty numeric field means unavailable, not zero. An empty missing_components field means no component is missing. True zeros and negative standardized scores remain valid when present. Source-file completeness, numeric task coverage, and availability of a method score are separate concepts.

Calculation implementation: `Figure_code/stability/compute_stability_rank.py` and `Figure_code/common/core.py`. Configuration: `Figure_code/config/scRNA.json`.
