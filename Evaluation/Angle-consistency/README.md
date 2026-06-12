# Evaluation

## Angle Consistency

This metric compares predicted RNA velocity directions against reference differentiation directions on a shared low-dimensional basis.

It is implemented as a single public script:

- `angle_consistency.py`

The script supports both:

- real datasets
- simulated datasets

The score reported in `benchmark_total.csv` is the percentage of cells whose angle between predicted velocity and reference direction falls in the `0-60` degree range.

## Core Behavior

### Real datasets

For real datasets, the script:

1. reads low-dimensional coordinates from the input h5ad
2. builds reference differentiation paths from `cluster_key` and `differentiation_paths`
3. fits low-dimensional trajectory curves
4. compares predicted low-dimensional velocity directions against the trajectory tangents

Important rules:

- The public script does not recompute UMAP, tSNE, or PCA coordinates.
- The default basis is `umap`.
- If the default `umap` basis is missing, the script prints a warning and stops.
- If the user explicitly requests another basis such as `tsne` or `pca` and it is missing, the script fails immediately.
- External coordinate CSV injection is not supported in the public version.

### Simulated datasets

For simulated datasets, the script compares predicted low-dimensional velocity directions against reference trajectory directions.

Simulated inputs may use:

- coordinates already present in the input h5ad
- or a public reference directory containing `*_reference_data.npz`

The public reference directory is:

- `simdata_reference`

Reference files use:

- topology subdirectories
- file naming pattern `*_reference_data.npz`
- a unified low-dimensional coordinate key `X_basis`

#### Scope of simulation generality

The public script supports two different levels of simulation handling.

For the 11 built-in public topology families provided by the team:

- the topology-specific reference-direction construction logic is kept
- topology-specific guide points, spline degree choices, and smoothing settings are preserved
- these built-in topologies use the original fine-tuned reference construction logic

For user-defined or otherwise unknown simulated topologies:

- the script does not apply built-in guide-point logic
- the user must provide `differentiation_paths`
- the script falls back to a basic spline workflow based on milestone centers
- this fallback is more general but is not guaranteed to match the precision of the built-in fine-tuned topology implementations

This boundary is intentional. The public script is reproducible for the built-in topologies and only moderately general for arbitrary user-defined simulated data.

#### Reference requirements by topology type

For ordinary simulated topologies:

- a reference is optional if the input h5ad already contains a usable basis, milestone labels, and predicted velocity information

For `lineage-tracing`:

- the input h5ad is expected to use `adata.obs['synthetic_celllabel']`
- the reference npz stores the same labels under `celltype`
- the metric uses a dedicated graph-based reference-direction algorithm
- `lineage-tracing` depends on the reference to provide unified `X_basis` and `celltype`

For `Bursting-tree`:

- the metric depends on special reference metadata such as `edge_id`, `cell_time_ref`, `lineage_id`, `milestone_id`, and related fields
- if these fields are not available, the script prints a warning and asks for a valid `reference_dir`
- if no valid reference file can be located, the script stops with an error

In other words:

- “reference is optional” applies only to ordinary simulated topologies
- `lineage-tracing` depends on the reference for unified basis and labels
- `Bursting-tree` effectively requires the reference

Note:

- datasets such as `lineage-tracing` that do not provide ground-truth velocity are not hard-coded to skip
- they follow the same general validation path as any other unsupported input and fail with a clear error if required inputs are missing

## Inputs

### Predicted velocity

The script expects predicted velocity in one of the following forms:

- high-dimensional velocity in `adata.layers[velocity_key]`
- or a precomputed low-dimensional embedding in `adata.obsm[f"{velocity_key}_{basis_name}"]`

For simulated datasets, `velocity_key` is also used as the high-dimensional velocity layer name when low-dimensional embeddings need to be recomputed.

### Real dataset labels

Real datasets require:

- a cluster or cell type column in `adata.obs`

You may provide it explicitly with `cluster_key`. If omitted, the script tries to detect a suitable column automatically.

### Simulated dataset labels

Default label keys are:

- ordinary simulated datasets: `milestone`
- `Bursting-tree`: `pop`
- `lineage-tracing`: `synthetic_celllabel`

You may override `milestone_key`, but the public datasets should work out of the box with these defaults.

## Outputs

The script writes rose plots and summary CSV files under the chosen output directory.

### Wide summary table

`benchmark_total.csv` is the main benchmark output.

Rules:

- the first column is `Method`
- one column is added per dataset
- `Mean` and `Rank` are not produced
- rows are sorted alphabetically by `Method`

### Detailed long table

`benchmark.csv` is optional and disabled by default.

Enable it with:

- `--save-detailed-csv`

This file stores per-group angle-bin statistics.

### Figures

Each run writes:

- one PDF rose plot
- one PNG rose plot

Optional QA plots for simulated reference curves can also be saved.

## Command Line Usage

### Single real dataset

```bash
python angle_consistency.py \
  --input /path/to/result.h5ad \
  --data-type real \
  --method scVelo \
  --dataset 28 \
  --output-dir /path/to/output \
  --velocity-key velocity
```

### Single simulated dataset

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

### Unknown simulated topology with user-defined paths

```bash
python angle_consistency.py \
  --input /path/to/custom_simulation.h5ad \
  --data-type sim \
  --method MyMethod \
  --dataset custom_topology_case1 \
  --output-dir /path/to/output \
  --velocity-key velocity \
  --basis-name dimred \
  --milestone-key milestone \
  --differentiation-paths "Root|Intermediate|BranchA;Root|Intermediate|BranchB"
```

### Save the detailed long-format CSV

```bash
python angle_consistency.py \
  --input /path/to/result.h5ad \
  --data-type real \
  --method scVelo \
  --dataset 28 \
  --output-dir /path/to/output \
  --save-detailed-csv
```

## Batch CSV

The minimum required schema is:

```csv
method,id,path,vkey
```

Meaning:

- `method`: display label in outputs
- `id`: dataset identifier
- `path`: input h5ad path
- `vkey`: predicted velocity key

Optional extension columns are:

- `data_type`
- `basis_name`
- `cluster_key`
- `differentiation_paths`
- `milestone_key`
- `topology_type`
- `reference_dir`

No legacy aliases are supported in the public version.

### Example batch CSV

```csv
method,id,path,vkey,data_type,basis_name,reference_dir
scVelo,28,/path/to/real_28.h5ad,velocity,real,umap,
scVelo,bifurcating_cell1000_gene10000,/path/to/sim_bif.h5ad,velocity,sim,dimred,/path/to/simdata_reference
MyMethod,custom_topology_case1,/path/to/custom_sim.h5ad,velocity,sim,dimred,
```

### `differentiation_paths` encoding

`differentiation_paths` uses:

- `;` between paths
- `|` within each path

Example:

```text
Root|Intermediate|BranchA;Root|Intermediate|BranchB
```

This format is used because:

- commas would conflict with CSV separators
- hyphens may already appear in real dataset cell type names

## Python API

Main entry points:

- `run_unified_angle_consistency(...)`
- `run_batch_from_csv(...)`

Important defaults:

- real datasets default to `basis_name="umap"`
- simulated datasets default to `basis_name="dimred"`
- `lineage-tracing` is forced to `umap`
- `benchmark.csv` is written only when `save_detailed_csv=True`

## Notes

- `method` is only a display label used in file names and result tables.
- The public script keeps generic robustness logic such as neighbor repair, duplicate-cell cleanup, and low-dimensional velocity recomputation when a valid high-dimensional velocity layer is available.
- The public script does not keep tool-specific preprocessing branches.
- Reference lookup keeps dataset-name normalization and underscore/hyphen topology normalization for locating `*_reference_data.npz`.
