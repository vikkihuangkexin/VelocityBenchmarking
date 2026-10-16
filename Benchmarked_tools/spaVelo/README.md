# spaVelo

This standalone entry reproduces the spatial denoising workflow used for the
benchmark. Use the [team spaVelo source fork](https://github.com/sd68515/spavelo_benchmark),
which contains the benchmark changes to `experiments/preprocess_real.py` and
`spaVelo.py`.

## Inputs

The H5AD input needs `spliced` and `unspliced` layers, an annotation column in
`obs`, and finite two-dimensional coordinates in `obsm`. Coordinates default
to `obsm['spatial']`; use `--coordinate-key` or CSV `dimred_key` for another
key. The annotation key is explicit; no key is inferred from dataset IDs.
The script also provides scVelo's `obsm['X_spatial']` alias by copying
`obsm['spatial']` when it is missing. Newly created aliases use the original
coordinates; the scaled model coordinates are kept separately in
`spatial_model` during fitting.

## Install and run

From the VelocityBenchmarking repository root, clone the fork next to this
checkout and make its root and `experiments/` importable through `PYTHONPATH`:

```bash
git clone https://github.com/sd68515/spavelo_benchmark.git ../spavelo_benchmark
```

```bash
PYTHONPATH="$PWD/../spavelo_benchmark:$PWD/../spavelo_benchmark/experiments${PYTHONPATH:+:$PYTHONPATH}" \
  python Benchmarked_tools/spaVelo/spaVelo.py \
  --input data/tissue.h5ad --dataset-id tissue \
  --cluster-key cell_type \
  --output-dir results/spaVelo
```

For several datasets, provide a CSV. Required columns are
`dataset_name,file_path,modality,cluster_key`; `modality` must be `spatial`.
`dimred_key` is optional and defaults to `spatial`:

```csv
dataset_name,file_path,modality,cluster_key,dimred_key
tissue,data/tissue.h5ad,spatial,cell_type,
```

```bash
PYTHONPATH="$PWD/../spavelo_benchmark:$PWD/../spavelo_benchmark/experiments${PYTHONPATH:+:$PYTHONPATH}" \
  python Benchmarked_tools/spaVelo/spaVelo.py \
  --metadata-file datasets.csv --output-dir results/spaVelo
```

CSV file paths may be absolute or relative to the CSV. The script scales the
declared coordinates to `[0, 20]` for spatial gene selection and model fitting,
then restores the original coordinates in the saved result. It uses spaVelo's
Moran-based `spatial` gene selection without a separate HVG pass, the local
Dynamo initialization, seed 144, and the benchmark's model settings. Defaults
are 5,000 training iterations, patience 100, batch size 256, and CPU; use
`--device cuda` or `--device cuda:<index>` when appropriate. `--threads` sets
PyTorch and Moran worker counts. `--seed`, `--maxiter`, `--patience`, and
`--batch-size` are explicit overrides.

## Outputs

Each dataset writes `results/spaVelo/<dataset_id>/model.pt` and
`adata_out.h5ad`. The latter contains `layers['pred_velocity']` and
`layers['Ms']`, the original spatial coordinates, and run settings in
`uns['spavelo_run']`. The input `X_umap`, if present, is preserved. Use
`--overwrite` before rerunning into a nonempty dataset output directory.

## Fork behavior

The required preprocessing changes use the declared spatial coordinates for
Squidpy neighbors, accept sparse inputs, allow UMAP calculation to be skipped,
set Moran worker count, and treat Dynamo's per-gene SVD failure as an invalid
gene. The training changes remove retained backward graphs and allow quiet
progress reporting and less frequent recorder writes. The spatial-neighbor
and invalid-gene changes can affect selected genes and results.
