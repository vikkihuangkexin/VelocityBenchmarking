# Spatial rank summary

Companion table: [rank_summary.csv](rank_summary.csv). One row per method; 7 methods.

All columns retain the category-specific scientific ranks. Lower ranks are better.

## Columns

| Column | Meaning |
|---|---|
| `method` | Canonical method name. Rows follow the display order in accuracy_rank.csv. |
| `accuracy_rank` | The scientific rank column copied from accuracy_rank.csv for this method. Rank 1 is best; ties retain average ranks and missing ranks stay NA. This is not a new rank calculated from other columns. |
| `consistency_rank` | The scientific rank column copied from consistency_rank.csv for this method. Rank 1 is best; ties retain average ranks and missing ranks stay NA. This is not a new rank calculated from other columns. |
| `scalability_rank` | The scientific rank column copied from scalability_rank.csv for this method. Rank 1 is best; ties retain average ranks and missing ranks stay NA. This is not a new rank calculated from other columns. |
| `stability_rank` | The scientific rank column copied from stability_rank.csv for this method. Rank 1 is best; ties retain average ranks and missing ranks stay NA. This is not a new rank calculated from other columns. |
| `usability_rank` | The scientific rank column copied from usability_rank.csv for this method. Rank 1 is best; ties retain average ranks and missing ranks stay NA. This is not a new rank calculated from other columns. |
| `scalability_display_rank` | Spatial Scalability display_rank copied from scalability_rank.csv. STT and spaVelo have 6.5 under the bottom-tie display policy, while scalability_rank remains NA for them. |

## Missing values and ordering

NA denotes an unavailable scientific rank, not zero or a last-place value. Row position is a display order and does not assign a rank to an unranked method.

scalability_rank is the scientific rank and remains NA for STT and spaVelo. scalability_display_rank reports the separate 6.5 display tie. Neither column changes the original scientific scores.

See the matching category README for the score definition, weights, source coverage, and interpretation of other fields.
