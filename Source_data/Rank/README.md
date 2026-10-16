# Benchmark rankings

Place this `rank` folder at the VelocityBenchmarking repository root.
The three modality folders are `scRNA`, `ATAC`, and `spatial`.

Each folder contains six category files:

- `accuracy_rank.csv`
- `consistency_rank.csv`
- `scalability_rank.csv`
- `stability_rank.csv`
- `usability_rank.csv`
- `overall_rank.csv`

`rank_summary.csv` combines the six rank columns into one method-by-category
table. Rows follow the original Overall display order. These summaries retain
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
| `overall_rank.csv` | [README](scRNA/overall_rank_README.md) | [README](ATAC/overall_rank_README.md) | [README](spatial/overall_rank_README.md) |
| `rank_summary.csv` | [README](scRNA/rank_summary_README.md) | [README](ATAC/rank_summary_README.md) | [README](spatial/rank_summary_README.md) |

The CSV files are unchanged from the previous release. Read `rank` as the scientific rank; the separate spatial resource display/effective-rank fields retain their specified policy.

## Agreement with the previous analysis

All three modalities were recalculated from the source CSV matrices using the
latest supplied calculation code and independently verified against those
matrices. The 18 category files were then compared with the latest previously
verified analysis in this conversation. Every complete category table is
byte-identical, including scores, ranks, ties, coverage, component values,
and missing values. ATAC and spatial ranks also exactly match the latest
previously generated figure-data tables. Scores in those figure-data exports
differ only by CSV serialization rounding below 1e-12.

`verification.json` records these comparisons, file hashes, and the current
source-file hashes. Earlier preliminary analyses using different inputs or
metric selections are not the reference for this release.

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

scRNA contains 28 methods. The Stability and Overall scores for k-velo and
GRAVITY remain unavailable, so those two ranks remain missing. ATAC has nine
methods with all six categories available.

Spatial has seven methods. STT and spaVelo have no successful Scalability
measurements: their scientific `score`, `rank`, and `reversed_rank` remain
missing. The Scalability file preserves `display_rank=6.5` and
`effective_reversed_rank=0` under the previously confirmed resource-failure
policy. Overall uses that effective value. The spatial summary adds a separate
`scalability_display_rank` column; it does not replace the missing scientific
rank. All seven spatial Overall scores remain available under this policy.

Overall uses the original weighted reversed ranks:

```text
0.65 * Accuracy + 0.10 * Consistency + 0.10 * Scalability
+ 0.075 * Stability + 0.075 * Usability
```

The terms above are reversed ranks, not raw category scores. Missing dimensions
do not trigger weight redistribution. The explicit spatial resource-failure
policy is retained as specified above.

## Reproduce the calculations

From the repository root, using the previously supplied `Figure_code` and
`Source_data` folders:

```bash
python Figure_code/run.py --profile all --output Results_rank_check
```

The calculation engine calls independent verification. For each modality and
category, compare `Results_rank_check/PROFILE/CATEGORY/scores_ranks.csv` with
the corresponding `rank/MODALITY/CATEGORY_rank.csv`. The calculation profile
name is `Spatial`; its release folder here is lowercase `spatial`.
