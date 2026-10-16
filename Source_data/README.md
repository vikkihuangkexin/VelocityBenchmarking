# Source data

Input data for the code in **`Figure_code/`**. Every CSV is a measurement matrix
exported from the study's supplementary tables; a few files are provenance
manifests that record where each matrix came from.

## Layout

```
multimodal_sources.csv              provenance manifest (upload -> repo file, sha256)
measurements_raw/
  scRNA/  ATAC/  Spatial/
    methods.csv                     method order used as the row order
    real_dataset_ids.csv            real dataset number -> dataset id
    ATAC/dataset_pairs.csv          scRNA <-> ATAC paired datasets
    ATAC/sparse_tasks.csv           four sparse-stress tasks
    measurement/
      summary/                      accuracy / consistency metric matrices
      downsampling/                 downsampling-perturbation stability
      std/                          multi-run batch-run SD (scRNA only)
      measurements_corr/            paired metric vs ground-truth values (scRNA only)
      auxiliary/                    per-cell eSVC values (Spatial only)
scalability/                        runtime / CPU RAM / GPU VRAM
usability/                          scoring rules and per-item scores
```

## `measurements_raw/`

Shared index files:

| File | Contents |
|---|---|
| `methods.csv` | `method_order`, `method` — row order of every matrix in that modality |
| `real_dataset_ids.csv` | `dataset_number`, `dataset_id` |
| `ATAC/dataset_pairs.csv` | `pair_id`, RNA/ATAC dataset names and ids, `pair_label` |
| `ATAC/sparse_tasks.csv` | `task_order`, `dataset_id`, `topology`, `sparsity_label`, `shape`, `nominal_cells`, `nominal_genes` |

`measurement/summary/` — one wide matrix per metric, `method` x `dataset_id`:

| Metric | scRNA | ATAC | Spatial |
|---|---|---|---|
| `AngleConsistency` | real + simulation | real + simulation | real |
| `CBDir` | real + simulation | real + simulation | real |
| `ConsistencyScore` | real + simulation | real + simulation | — |
| `ICCoh` | real + simulation | real + simulation | — |
| `TransitionScore` | real + simulation | real + simulation | real |
| `GroundTruthCorrelation` | simulation | simulation | — |
| `STC`, `SVC` | — | — | real |

File name pattern: `<Metric>_<Modality>_<real\|simulation>_02_summary.csv`, where
`02` is an export version tag.

`measurement/downsampling/` — robustness to downsampling, supplied for all three
modalities:

| File | Modality |
|---|---|
| `GroundTruthCorrelation_<Modality>_downsampling_02_summary.csv` | scRNA, ATAC |
| `AngleConsistency/CBDir/TransitionScore/STC/SVC_Spatial_downsampling_02_summary.csv` | Spatial |
| `aggregation_counts.csv` (per-run value + `n_scored_cells`) | ATAC |
| `aggregation_receipt.json`, `source_comparison.json` | ATAC (export provenance) |

`measurement/std/` — scRNA only. `*_03_batchrun_std.csv` hold the per-metric
standard deviation across the multi-run batch runs (deep-learning tools).
ATAC and Spatial use `stability_mode = downsampling_only`.

`measurements_corr/` — scRNA only. `gtc_paired_values.csv` has
`metric`, `method`, `dataset_id`, `simulation_family`, `metric_value`,
`groundtruth_correlation`; it feeds the Figure 2 correlation panels.

`auxiliary/` — Spatial only. Per-cell eSVC values:
`esvc_{all_cell,annotation_matched}_Spatial_{real,downsampling}.csv`.

## `scalability/`

`{scRNA,ATAC,Spatial}_{runtime,cpu_ram,gpu_vram}.csv`. Long format with
`protocol`, `measure`, `dataset_id`, `unit`, `cells`, `genes`, `method`,
`clean_value`, `value_status` and full source locators (`source_path`,
`source_locator`, `source_workbook`). Missing entries stay in the file with a
`value_status` such as `method_not_provided` instead of being dropped. Only the
`top_scRNA` protocol enters the Scalability score; VRAM does not.

## `usability/`

| File | Contents |
|---|---|
| `usability_rules.csv` | 24 subcategories with category/subcategory weights (Table S7) |
| `{scRNA,ATAC,Spatial}_usability_items.csv` | per-method item scores, `evidence_note`, `source_url` (Table S8) |

