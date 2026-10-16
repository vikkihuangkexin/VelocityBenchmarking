# scRNA Usability ranking

Companion table: [usability_rank.csv](usability_rank.csv). One row per method; 28 methods and 28 available category scores.

These definitions describe the existing table. Scores, ranks, component values, and missing values are unchanged.

## Calculation and interpretation

The 24 subitems comprise six Installability, five Code quality, five Usage guidance, five Maintenance, and three Publication quality items. Each category is recalculated from subitem scores times declared within-category subweights. Source-file counts do not count these 24 subitems.

Final weights are Installability 0.25, Code quality 0.25, Usage guidance 0.20, Maintenance 0.15, and Publication quality 0.15. A missing category makes the final score unavailable; the remaining category weights are not redistributed.

## Columns

Column names below follow the CSV header order exactly.

| Column | Meaning |
|---|---|
| `method` | Canonical method name used in the source matrices and fixed modality roster. |
| `score` | Weighted sum of five recomputed usability category scores, on the 0-10 scale. All five are required. Higher is better; this score is not z-standardized. |
| `rank` | Descending rank of score within this modality and category. Rank 1 is best. Exact score ties receive the average occupied rank. Missing scores have missing ranks. |
| `reversed_rank` | 28 + 1 - rank, where 28 is the full modality roster size, including methods with missing scores. Higher is better. Missing scientific ranks remain missing. |
| `normalized_rank` | reversed_rank / 28. A rank-derived value, not a normalized category score. The best rank maps to 1; missing ranks remain missing. |
| `fixed_method_order` | Original 1-based position in the fixed methods.csv roster. Used only to keep deterministic ordering of ties and missing rows. It is not a performance rank. |
| `display_order` | 1-based row position after sorting scientific rank and then fixed_method_order, with missing ranks last. This plotting order is unique even for ties and must not replace rank. |
| `n_components` | Number of nonmissing component values supplied to this score calculation. A count of metric or category components, not datasets or successful runs. |
| `expected_components` | Number of configured components for this category. Compare with n_components to identify incomplete component coverage. |
| `missing_components` | Semicolon-separated names of components without usable values. An empty field means none are missing; it is not a zero score. |
| `component_Code quality` | Recomputed Code quality category score on the 0-10 item scale: sum of subitem score times its within-category subweight. This is not z-standardized. |
| `effective_weight_Code quality` | Usability weight for Code quality: 0.25. All five category scores are required; effective weights are blank if the final Usability score is unavailable. |
| `component_Installability` | Recomputed Installability category score on the 0-10 item scale: sum of subitem score times its within-category subweight. This is not z-standardized. |
| `effective_weight_Installability` | Usability weight for Installability: 0.25. All five category scores are required; effective weights are blank if the final Usability score is unavailable. |
| `component_Maintenance` | Recomputed Maintenance category score on the 0-10 item scale: sum of subitem score times its within-category subweight. This is not z-standardized. |
| `effective_weight_Maintenance` | Usability weight for Maintenance: 0.15. All five category scores are required; effective weights are blank if the final Usability score is unavailable. |
| `component_Publication quality` | Recomputed Publication quality category score on the 0-10 item scale: sum of subitem score times its within-category subweight. This is not z-standardized. |
| `effective_weight_Publication quality` | Usability weight for Publication quality: 0.15. All five category scores are required; effective weights are blank if the final Usability score is unavailable. |
| `component_Usage guidance` | Recomputed Usage guidance category score on the 0-10 item scale: sum of subitem score times its within-category subweight. This is not z-standardized. |
| `effective_weight_Usage guidance` | Usability weight for Usage guidance: 0.2. All five category scores are required; effective weights are blank if the final Usability score is unavailable. |
| `n_source_inputs_expected` | Number of declared source files for this category. This counts files, not methods, datasets, subitems, or valid numeric measurements. |
| `n_source_inputs_available` | Number of those declared source files that exist. A present file may still contain missing values for a particular method. |
| `input_status` | Source-file availability: complete when all declared files exist, partial when some exist, unavailable when none exist. This does not measure numeric task coverage. |
| `overall_eligible` | Source-file eligibility flag: True when this category has its complete declared file set. It is not a guarantee that this method has a category score or a complete Overall score. |

## Missing values and source status

An empty numeric field means unavailable, not zero. An empty missing_components field means no component is missing. True zeros and negative standardized scores remain valid when present. Source-file completeness, numeric task coverage, and availability of a method score are separate concepts.

Calculation implementation: `Figure_code/usability/compute_usability_rank.py` and `Figure_code/common/core.py`. Configuration: `Figure_code/config/scRNA.json`.
