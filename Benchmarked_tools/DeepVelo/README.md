# DeepVelo

Run DeepVelo on real or simulated scRNA data and save gene-space velocity,
projected velocity, and scVelo pseudotime.

Use the team's [DeepVelo Python 3.11 fork](https://github.com/sd68515/DeepVelo_py311).
Its `environment-py311.yml` records the benchmark's package versions. The
fork documents an editable checkout installation and separate PyTorch
installation; follow its README for CPU or CUDA setup.

## Inputs and run

The H5AD file needs `spliced` and `unspliced` layers, an existing annotation
column in `adata.obs`, and existing finite 2D coordinates in `adata.obsm`.
At least 30 cells and 30 genes must remain after preprocessing.
Run the commands below from the repository root.

```bash
python Benchmarked_tools/DeepVelo/DeepVelo.py --input sample.h5ad --output-dir results \
  --kind real --coordinate-key X_umap --cluster-key cell_type
```

For simulated data, use `--kind sim` and supply its actual coordinate key.
The script preserves input coordinates; it does not compute or substitute
UMAP coordinates. `--dataset-id`, `--seed`, and `--overwrite` are optional.

For several datasets, supply all keys in a CSV:

```csv
dataset_name,file_path,modality,cluster_key,dimred_key
sample_a,data/sample_a.h5ad,real,cell_type,X_umap
sample_b,data/sample_b.h5ad,sim,milestone,X_dimred
```

Paths may be absolute or relative to the CSV. Run:

```bash
python Benchmarked_tools/DeepVelo/DeepVelo.py --metadata-file datasets.csv --output-dir results
```

## Outputs

The result is `results/<dataset_id>/<dataset_id>_deepvelo.h5ad`.

| H5AD location | Outputs |
| --- | --- |
| `layers` | `velocity`, `cell_specific_beta`, `cell_specific_gamma` |
| `obsm` | `velocity_<basis>` alongside the supplied coordinates |
| `obs` | `velocity_pseudotime` |
| `uns` | `deepvelo_run`, including preprocessing and non-finite cleanup counts |

Existing results are skipped unless `--overwrite` is set.

## Preprocessing and training

The single-run and repeated-training entries both use 100 epochs,
`shuffle=False`, `validation_split=0`, `topC=30`, `topG=20`, and DeepVelo's
data-derived spliced-loss coefficient. They remove jointly empty or duplicate
spliced/unspliced profiles before training. Real datasets use
`min_shared_counts=20`; simulations omit expression-threshold gene filters.

After training, non-finite values in at most 20% of **velocity matrix
elements** trigger removal of every affected cell and gene. A larger fraction
stops the run. The H5AD records the counts and resulting dimensions in
`adata.uns['deepvelo_run']`.

## Independent repeated training

Five repetitions are available from
[`Batch_run/DeepVelo_robust.py`](../../Batch_run/DeepVelo_robust.py):

```bash
python Batch_run/DeepVelo_robust.py --input-csv datasets.csv \
  --output-dir repeated/deepvelo --runs 5
```

The repeated-run entry requires all five shared CSV columns. It preprocesses
once using `--preprocess-seed`, then fits a fresh model per run. Run indices
start at 1 and use `--base-seed + run_index`. Results are
`<output-dir>/<modality>/<dataset_name>/DeepVelo_<dataset_name>_r<run>.h5ad`;
existing results are skipped. The repeated result metadata is
`uns['deepvelo_batchrun']`.
