# Benchmark rankings

These ranking tables are stored in `Source_data/Rank`.
The three modality folders are `scRNA`, `ATAC`, and `spatial`.

Each folder contains five category files:

- `accuracy_rank.csv`
- `consistency_rank.csv`
- `scalability_rank.csv`
- `stability_rank.csv`
- `usability_rank.csv`

`rank_summary.csv` combines the five rank columns into one method-by-category
table. Rows follow the Accuracy display order. These summaries retain
the category ranks; they do not calculate ranks across columns or modalities.

## Per-table column definitions

Every CSV has an adjacent `*_README.md` explaining every physical column in header order. Both category tables and the combined rank summaries are documented.

| Table | scRNA | ATAC | spatial |
|---|---|---|---|
| `accuracy_rank.csv` | [README](scRNA/accuracy_rank_README.md) | [README](ATAC/accuracy_rank_README.md) | [README](spatial/accuracy_rank_README.md) |
| `consistency_rank.csv` | [README](scRNA/consistency_rank_README.md) | [README](ATAC/consistency_rank_README.md) | [README](spatial/consistency_rank_README.md) |
| `scalability_rank.csv` | [README](scRNA/scalability_rank_README.md) | [README](ATAC/scalability_rank_README.md) | [README](spatial/scalability_rank_README.md) |
| `stability_rank.csv` | [README](scRNA/stability_rank_README.md) | [README](ATAC/stability_rank_README.md) | [README](spatial/stability_rank_README.md) |
| `usability_rank.csv` | [README](scRNA/usability_rank_README.md) | [README](ATAC/usability_rank_README.md) | [README](spatial/usability_rank_README.md) |
| `rank_summary.csv` | [README](scRNA/rank_summary_README.md) | [README](ATAC/rank_summary_README.md) | [README](spatial/rank_summary_README.md) |

Category scores and ranks are unchanged. Read `rank` as the scientific rank; the separate spatial resource display-rank field retains its specified policy.

## Agreement with the previous analysis

The 15 category tables retain their existing scores, ranks, ties, coverage,
component values, and missing values. The summary tables retain the five
category ranks for each method and follow the Accuracy display order.

## Reading the files

- `score`: the original category score. Higher is better. Scores from different
  categories are not directly comparable.
- `rank`: the category's scientific rank, with 1 best. Exact ties keep average
  ranks. Unavailable scientific scores retain missing ranks.
- `reversed_rank`: roster size plus 1 minus the scientific rank.
- `display_order`: plotting row order, not a replacement rank.
- Component, effective-weight, coverage, and source-status columns are retained
  exactly as calculated.

Blank fields in the original category tables and `NA` in the summary tables
both indicate missing values. Neither is a zero score or a last-place rank.
Do not convert `display_order` into a scientific rank or renumber tied ranks.

## Confirmed scope and missing-data rules

Accuracy includes directional metrics only. Spatial main Consistency uses STC
and SVC; its downsampling uses Angle consistency, CBDir, Transition V1, STC,
and SVC. The updated ATAC downsampling matrix is aggregated from the supplied
cell-level values. RNA Stability combines downsampling with the measured
batch-run SD inputs. Missing batch-run SD is never assumed to be zero.

## Reproduce the calculations

From the repository root, using the previously supplied `Figure_code` and
`Source_data` folders:

```bash
python Figure_code/run.py --profile all --output Results_rank_check
```

The calculation engine calls independent verification. For each modality and
category, compare `Results_rank_check/PROFILE/CATEGORY/scores_ranks.csv` with
the shared columns in the corresponding
`Source_data/Rank/MODALITY/CATEGORY_rank.csv`. The calculation profile
name is `Spatial`; its release folder here is lowercase `spatial`.
