# Angle Consistency (Rose Plot)

Single-file RNA velocity angle-consistency metric (rose plot) for real and simulated datasets.

Copy to run elsewhere:
- `angle_consistency.py`
- Optional resource folders: `real-umap/` (real external UMAP) and `Simdata-GTkey/` (simulation GT npz)

## Evaluation

    Angle Consistency (Rose Plot)
    result = run_unified_angle_consistency(
        data_type="real",  # or "sim"
        input_h5ad="/path/to/result.h5ad",
        output_dir="/path/to/out",
        tool="scVelo",
        dataset="27",  # real dataset id or sim dataset name
        benchmark_csv_path="/path/to/out/benchmark.csv",
        benchmark_total_csv_path="/path/to/out/benchmark_total.csv",
    )

Input: A `.h5ad` file path. The file must contain predicted velocity (`velocity_key`, default: `velocity`) in `adata.layers` or as a precomputed low-dimensional embedding in `adata.obsm` (e.g. `velocity_umap` / `velocity_dimred`). Real datasets also need a cluster/celltype column in `adata.obs` (auto-detected or set via `cluster_key`). Simulation datasets also need milestone labels in `adata.obs[milestone_key]` (default: `milestone`).

Output: Returns a dictionary with `status`, plot paths, and `total_ratio_0_60` (% of valid cells with angles in [0, 60]). If CSV paths are provided, it writes:
- `benchmark.csv` (long format: tool/dataset/data_type/group_type/group + angle bins + `valid_ratio` + `cv_coefficient`)
- `benchmark_total.csv` (wide format: Tool + per-dataset 0–60% + Mean + Rank)
  - Rank rule: higher Mean is better, but Rank=1 is the worst (smallest Mean)
Errors are logged to `<output-dir>/errors.log` by default (override via `--error-log`).

Optional parameters supported by `run_unified_angle_consistency(...)`:
- Common: `velocity_key`, `n_jobs`, `benchmark_csv_path`, `benchmark_total_csv_path`, `error_log_path`
- Real only: `cluster_key`, `differentiation_paths`, `umap_dir`
- Sim only: `milestone_key`, `topology_type`, `npz_base_dir`, `plot_reference_curves`, `reference_curves_dir`, `gt_velocity_enhancement`

Follow-up: Run `run_batch_from_csv(...)` or `python angle_consistency.py --csv-file ...` to generate comparable `benchmark_total.csv` across multiple methods/datasets.

## CLI

Single dataset:
```bash
python angle_consistency.py \
  --input /path/to/result.h5ad \
  --data-type real \
  --tool scVelo \
  --dataset 27 \
  --output-dir /path/to/out \
  --n-jobs 1
```

Batch CSV:
```bash
python angle_consistency.py \
  --csv-file /path/to/jobs.csv \
  --output-dir /path/to/out
```

## Batch CSV schema

Required columns:
- `data_type` (`real` or `sim`), `tool`, `dataset`, `path`

Optional columns (only used when relevant):
- Common: `velocity_key`, `n_jobs`
- Real: `cluster_key`, `umap_dir`, `differentiation_paths` (JSON string or JSON file path; relative to the CSV file)
- Sim: `milestone_key`, `topology_type`, `npz_base_dir` (or `gtkey_dir`), `plot_reference_curves`, `reference_curves_dir`, `gt_velocity_enhancement` (`true/false/auto`)

Aliases accepted (legacy CSVs): `method`→`tool`, `dataset_id|id|dataset_name`→`dataset`, `h5ad_path|input`→`path`, `vkey`→`velocity_key`.

## Notes on external real UMAP (real-umap/)

Real datasets can optionally use an external UMAP CSV folder (default search: `./real-umap` or next to `angle_consistency.py`).
If you rename the folder after downloading from GitHub, pass `--umap-dir /path/to/your-folder` (or set env `ANGLE_CONSISTENCY_REAL_UMAP_DIR`).
If no external folder is found/provided, the script will attempt to compute UMAP from the input `.h5ad`.
