# scRNA rank summary

Companion table: [rank_summary.csv](rank_summary.csv). One row per method; 28 methods.

All columns retain the category-specific scientific ranks. Lower ranks are better. Do not average these rank columns to reproduce Overall; Overall uses the configured weighted reversed ranks.

## Columns

| Column | Meaning |
|---|---|
| `method` | Canonical method name. The row order follows Overall display_order from overall_rank.csv, including any unranked methods at the end. |
| `accuracy_rank` | The scientific rank column copied from accuracy_rank.csv for this method. Rank 1 is best; ties retain average ranks and missing ranks stay NA. This is not a new rank calculated from other columns. |
| `consistency_rank` | The scientific rank column copied from consistency_rank.csv for this method. Rank 1 is best; ties retain average ranks and missing ranks stay NA. This is not a new rank calculated from other columns. |
| `scalability_rank` | The scientific rank column copied from scalability_rank.csv for this method. Rank 1 is best; ties retain average ranks and missing ranks stay NA. This is not a new rank calculated from other columns. |
| `stability_rank` | The scientific rank column copied from stability_rank.csv for this method. Rank 1 is best; ties retain average ranks and missing ranks stay NA. This is not a new rank calculated from other columns. |
| `usability_rank` | The scientific rank column copied from usability_rank.csv for this method. Rank 1 is best; ties retain average ranks and missing ranks stay NA. This is not a new rank calculated from other columns. |
| `overall_rank` | The scientific rank column copied from overall_rank.csv for this method. Rank 1 is best; ties retain average ranks and missing ranks stay NA. This is not a new rank calculated from other columns. |

## Missing values and ordering

NA denotes an unavailable scientific rank, not zero or a last-place value. Row position is a display order and does not assign a rank to an unranked method.

k-velo and GRAVITY retain NA Stability and Overall ranks. The other category ranks remain as calculated.

See the matching category README for the score definition, weights, source coverage, and interpretation of other fields.
