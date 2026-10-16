# VeloVAE

Run VeloVAE on real or simulated scRNA data and save velocity, model time, and
plots. The separate repeated-training entry preprocesses once and fits
independent models. Install [VeloVAE](https://github.com/welch-lab/VeloVAE) and
its dependencies, including Scanpy, scVelo, PyTorch, NumPy, pandas, SciPy, and
Matplotlib:

```bash
git clone https://github.com/welch-lab/VeloVAE.git ../VeloVAE
python -m pip install -e ../VeloVAE
```

## Inputs

Each H5AD needs `layers['spliced']`, `layers['unspliced']`, an annotation
column in `obs`, and coordinates in `obsm`. Supply `--cluster-key` and the
actual `--dimred-key` (default `X_umap`). Existing finite 2D coordinates are
preserved. If that key is missing, this single-run entry computes UMAP and
stores it under the requested key; provide the original coordinates when
comparing methods in a common basis. The repeated-training entry requires
existing coordinates.

## Run

Commands below run from the repository root. For real data:

```bash
python Benchmarked_tools/VeloVAE/VeloVAE.py \
  --input data/sample.h5ad --output-dir results/velovae \
  --fig-dir figures/velovae --cluster-key cell_type --device auto
```

For simulated data, supply the simulation settings explicitly:

```bash
python Benchmarked_tools/VeloVAE/VeloVAE.py \
  --input data/simulation.h5ad --output-dir results/velovae \
  --fig-dir figures/velovae --cluster-key milestone --dimred-key X_dimred \
  --dim-z 4 --zero-threshold --hvg-scheme simulated --device auto
```

For several inputs, all five CSV columns below are required. Relative
`file_path` values resolve from the CSV directory. Dataset names must be
unique path components.

```csv
dataset_name,file_path,modality,cluster_key,dimred_key
sample,data/sample.h5ad,real,cell_type,X_umap
simulation,data/simulation.h5ad,sim,milestone,X_dimred
```

```bash
python Benchmarked_tools/VeloVAE/VeloVAE.py \
  --metadata-file datasets.csv --output-dir results/velovae \
  --fig-dir figures/velovae --device auto
```

Optional CSV fields `dim_z`, `zero_threshold`, and `hvg_scheme` override the
per-modality defaults. Empty fields use defaults: `5,false,real` for `real`
and `4,true,simulated` for `sim`. Boolean fields accept true/false, 1/0, or
yes/no. Missing files and failed datasets are reported and cause a nonzero
exit status after the remaining datasets have been attempted.

## Outputs

For one input, the result is
`<output-dir>/VeloVAE_<input_stem_without_dataset_suffix>_plot.h5ad`.
CSV mode adds a `<dataset_name>/` directory. Result filenames derive from the
input filename, not the CSV name. Figures are PNG/PDF scatter, stream, grid,
and model-time plots under `<fig-dir>/png/` and `pdf/`; CSV mode adds a
dataset-name directory. Model diagnostics are under the dataset output's
`figure/` directory. Existing filenames can be overwritten by the single-run
entry; choose a fresh output directory for a separate analysis.

| H5AD location | Main outputs |
| --- | --- |
| `layers` | `vae_velocity`, `vae_uhat`, `vae_shat` |
| `obsm` | `vae_z`, `vae_velocity_<basis>` |
| `obs` | `vae_time`, `vae_time_normalized` |
| `var` | `vae_alpha`, `vae_beta`, `vae_gamma` |

`vae_time_normalized` rescales model time to `[0,1]`; a constant time becomes
zero. It is distinct from scVelo velocity pseudotime.

## Independent repeated training

Use the same five required CSV fields with
[`Batch_run/velovae_robust.py`](../../Batch_run/velovae_robust.py):

```bash
python Batch_run/velovae_robust.py --input-csv datasets.csv \
  --output-dir repeated/velovae --runs 5 --device auto
```

Each result is
`<output-dir>/<modality>/<dataset_name>/VeloVAE_<dataset_name>_r<run>.h5ad`.
Existing results are skipped. Shared preprocessing uses `--preprocess-seed`;
run indices start at 1 and use `--base-seed + run_index`. The driver derives
settings from `modality` and does not read the single-run optional overrides.
It removes jointly empty/duplicate simulation profiles and degenerate moment
genes before ODE initialization. It saves scVelo velocity pseudotime as well
as model time.

## Preprocessing and advanced options

The single-run entry removes duplicate `X` rows via a dense DataFrame; this
can use substantial memory. `--zero-threshold` disables shared-count/cell
gene thresholds. HVG targets depend on the input gene count:

| Scheme | Requested HVGs |
| --- | --- |
| `real` | Below 1,500 genes: 500; otherwise 2,000 |
| `simulated` | Below 1,500: 500; below 10,000: 2,000; below 50,000: 4,000; otherwise 5,000 |

PCA uses 30 components below 15,000 cells, otherwise 50. Batch sizes are 64
below 1,500 cells, 128 below 15,000, and 256 otherwise. Small inputs still
need enough cells/genes for the upstream PCA and training routines.
`--device` accepts `auto` (default), `cpu`, or a PyTorch CUDA device;
`--n-jobs` controls single-run post-analysis parallelism (default 1).
