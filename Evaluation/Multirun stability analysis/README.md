# Multirun stability analysis

Compare saved RNA velocity vectors across repeated runs of the same method on the same dataset. Each pair of runs is scored using the rescaled cosine similarities of the velocity vectors for matched cells.

## Inputs

Prepare a CSV with one row per result H5AD. All fields below are required.

| CSV field | Value |
| --- | --- |
| `method` | Method name |
| `dataset_id` | Dataset identifier |
| `run_id` | Run identifier, unique within a method and dataset |
| `path` | Path to the run's result H5AD |
| `vkey` | Velocity layer key in `adata.layers` |

```csv
method,dataset_id,run_id,path,vkey
MyMethod,example_real,r1,results/real_r1.h5ad,velocity
MyMethod,example_real,r3,results/real_r3.h5ad,velocity
MyMethod,example_real,r5,results/real_r5.h5ad,velocity
AnotherMethod,example_sim,r2,results/sim_r2.h5ad,spliced_velocity
```

Paths relative to the input CSV resolve from its directory. Run IDs may be nonconsecutive. Pairs follow the row order within each method and dataset. The example produces three pairs for `MyMethod`. The script reports and skips `AnotherMethod` because it has only one run. Five runs produce ten pairs.

Cells are matched by `obs_names`, and feature columns are aligned by `var_names`. Each file needs unique cell and feature names and finite velocity values. Runs in each group must have the same feature set, and each pair must share at least one cell. For learned latent velocities, supply vectors in a common coordinate system across runs.

## Run

Install AnnData (0.11 or later), h5py, NumPy and SciPy in the metric environment. Copy the example CSV to `inputs.csv` and enter your result paths and layer names. Run the following command from the repository root.

```bash
python "Evaluation/Multirun stability analysis/multirun_stability_analysis.py" \
  --manifest inputs.csv --output results/multirun_stability.csv
```

The script reads saved layers from the result H5AD files and converts sparse velocity layers to dense arrays.

## Outputs and calculation

The output CSV contains one row per run pair. The `tool_name` and `dataset_name` columns come from `method` and `dataset_id` in the input.

| Column | Contents |
| --- | --- |
| `group` | Run IDs joined by `_vs_` |
| `group_cosine` | Mean rescaled cosine across matched cells in this pair |
| `average_cosine` | Mean of the pair-level `group_cosine` values for this method and dataset |

For each matched cell, the cosine similarity `c` is mapped to `(c + 1) / 2`. If either velocity vector has zero norm, the cell contributes a score of zero. Higher scores indicate closer agreement across runs.

The same `average_cosine` is shown for every pair in a method and dataset. To report an equally weighted mean across datasets, take one `average_cosine` value per dataset and average those values.
