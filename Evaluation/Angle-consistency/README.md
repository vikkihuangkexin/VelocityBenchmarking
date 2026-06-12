# Angle Consistency

This metric compares predicted RNA velocity directions against reference differentiation directions on a shared low-dimensional basis. It is implemented as a single public script, `angle_consistency.py`, which supports both real and simulated datasets.

The score reported in `benchmark_total.csv` is the percentage of cells whose angle between predicted velocity and reference direction falls in the `0-60` degree range.

## Core Behavior

**Real datasets.** The script reads low-dimensional coordinates from the input h5ad, builds reference differentiation paths from `cluster_key` and `differentiation_paths`, fits trajectory curves, and compares predicted velocity directions against the tangents. It does not recompute UMAP/tSNE/PCA coordinates. The default basis is `umap`; if the requested basis is missing the script stops with an error. External coordinate CSV injection is not supported.

**Simulated datasets.** The script compares predicted velocity directions against reference trajectory directions. Coordinates may come from the input h5ad or from a public reference directory (`simdata_reference`) containing `*_reference_data.npz` files, organized into topology subdirectories and using a unified coordinate key `X_basis`.

For the 11 built-in public topology families, the original fine-tuned reference construction is preserved (topology-specific guide points, spline degree, and smoothing). For user-defined or unknown topologies, the user must supply `differentiation_paths`; the script falls back to a basic spline workflow over milestone centers, which is more general but not guaranteed to match the built-in precision.

### Reference requirements by topology

| Topology        | Label key             | Reference                      |
|-----------------|-----------------------|--------------------------------|
| ordinary sim    | `milestone`           | optional (if h5ad has basis, labels, velocity) |
| `lineage-tracing` | `synthetic_celllabel` | required — supplies unified `X_basis` and `celltype` |
| `Bursting-tree` | `pop`                 | required — supplies `edge_id`, `cell_time_ref`, `lineage_id`, `milestone_id` |

Datasets without ground-truth velocity (e.g. `lineage-tracing`) are not hard-coded to skip; they follow the same validation path and fail with a clear error if required inputs are missing.

## Inputs

**Predicted velocity** is expected as either high-dimensional velocity in `adata.layers[velocity_key]`, or a precomputed low-dimensional embedding in `adata.obsm[f"{velocity_key}_{basis_name}"]`. For simulated datasets, `velocity_key` also names the high-dimensional layer used when embeddings must be recomputed.

**Labels.** Real datasets need a cluster/cell-type column in `adata.obs`, set via `cluster_key` (auto-detected if omitted). Simulated default label keys: `milestone` (ordinary), `pop` (`Bursting-tree`), `synthetic_celllabel` (`lineage-tracing`); override with `milestone_key`.

## Outputs

`benchmark_total.csv` is the main benchmark output: first column `Method`, one column per dataset, rows sorted alphabetically by method (no `Mean`/`Rank` columns).

`benchmark.csv` (per-group angle-bin statistics) is disabled by default; enable with `--save-detailed-csv`.

Each run also writes one PDF and one PNG rose plot, plus optional QA plots for simulated reference curves.

## Command Line Usage

Single real dataset:

```bash
python angle_consistency.py \
  --input /path/to/result.h5ad \
  --data-type real \
  --method scVelo \
  --dataset 28 \
  --output-dir /path/to/output \
  --velocity-key velocity
```

Single simulated dataset (add `--basis-name`, `--milestone-key`, and `--differentiation-paths "Root|Intermediate|BranchA;Root|Intermediate|BranchB"` for unknown topologies):

```bash
python angle_consistency.py \
  --input /path/to/result.h5ad \
  --data-type sim \
  --method scVelo \
  --dataset bifurcating_cell1000_gene10000 \
  --output-dir /path/to/output \
  --velocity-key velocity \
  --reference-dir /path/to/simdata_reference
```

Add `--save-detailed-csv` to any run to write the detailed long-format CSV.

## Batch CSV

Required schema: `method,id,path,vkey` (display label, dataset id, input h5ad path, predicted velocity key).

Optional columns: `data_type`, `basis_name`, `cluster_key`, `differentiation_paths`, `milestone_key`, `topology_type`, `reference_dir`. No legacy aliases are supported.

```csv
method,id,path,vkey,data_type,basis_name,reference_dir
scVelo,28,/path/to/real_28.h5ad,velocity,real,umap,
scVelo,bifurcating_cell1000_gene10000,/path/to/sim_bif.h5ad,velocity,sim,dimred,/path/to/simdata_reference
MyMethod,custom_topology_case1,/path/to/custom_sim.h5ad,velocity,sim,dimred,
```

`differentiation_paths` uses `;` between paths and `|` within a path (commas conflict with the CSV separator, hyphens may appear in cell type names):

```text
Root|Intermediate|BranchA;Root|Intermediate|BranchB
```

## Python API

Main entry points: `run_unified_angle_consistency(...)` and `run_batch_from_csv(...)`.

Defaults: real datasets use `basis_name="umap"`, simulated use `basis_name="dimred"`, `lineage-tracing` is forced to `umap`, and `benchmark.csv` is written only when `save_detailed_csv=True`.

## Notes

- `method` is only a display label used in file names and result tables.
- The script keeps generic robustness logic (neighbor repair, duplicate-cell cleanup, low-dimensional velocity recomputation) but no tool-specific preprocessing branches.
- Reference lookup applies dataset-name and underscore/hyphen topology normalization when locating `*_reference_data.npz`.
