# spaVelo Script

spaVelo infers spatial RNA velocity by denoising spliced/unspliced counts with a
spatial Gaussian-process variational autoencoder. It is a spatial method: the model
needs the expression layers **and** the spatial coordinates of every cell.

## Installation

spaVelo is distributed as a script-only repository (no packaging), so the checkout
is imported through `--repo-dir` rather than installed:

```bash
git clone https://github.com/1062638515/spavelo.git /opt/spavelo
pip install 'numpy==1.26.4' 'pandas==2.2.1' 'scipy==1.12.0' 'anndata==0.10.6' \
    'scanpy==1.9.8' 'scvelo==0.3.2' 'scikit-learn==1.4.2' 'matplotlib==3.8.3' \
    'seaborn==0.13.2' 'h5py==3.10.0' 'numba==0.59.1' 'leidenalg==0.10.2' \
    'pyro-ppl==1.9.0' 'jax==0.4.20' 'jaxlib==0.4.20' 'ipykernel==6.28.0' tqdm 'loompy==3.0.7'
# PyTorch 2.2.1 with CUDA 12.1 is installed by the Dockerfile.
```

## Input Requirements

The input H5AD file must contain:

- `adata.layers["spliced"]` and `adata.layers["unspliced"]` (or the layers named by
  `--spliced-layer` / `--unspliced-layer`)
- `adata.obsm["spatial"]` — the 2D spatial coordinates (or the key named by
  `--spatial-key`)

`adata.obs[<cluster-key>]` is optional for spaVelo and only validated when passed.

The wrapper then runs the upstream preprocessing: `normalize_total` + `log1p`,
`highly_variable_genes` (2000 by default), a min-max scaled spatial grid in
`obsm["spatial"]` (the raw coordinates are kept in `obsm["spatial_raw"]`) and the
spatial GP inducing points. There is no simulated-data branch because both real and
simulated spatial datasets share the same input layout.

## Output

For an input `test.h5ad` with `dataset_name=id_test` the result is written to:

```text
output/
└── id_test/
    ├── test.h5ad    # result with layers['velocity']
    ├── model.pt     # saved spaVelo model weights
    └── args.csv     # the Config fields of the run, as written by the original driver
```

The naming is exactly the one of the original driver: its `run_all` loop computes
`name = path.stem`, `folder = root / name` and `out = folder / f"{name}.h5ad"`, and the
wrapper reproduces those statements, so the result file keeps the input stem verbatim
(no suffix rewriting) and `model.pt` / `args.csv` sit next to it. The native file name
is already the benchmark name, so no extra alias file is produced (the multi-omic
wrappers, which keep a method-specific native name, additionally write `rc.h5ad`).

The output H5AD contains:

- `.layers["velocity"]` — the inferred velocity (copied from the native `pred_velocity`)
- `.layers["pred_velocity"]` — the native spaVelo velocity key, preserved
- `.layers["prior_prob"]`, `.layers["Ms"]`, `.layers["Mu"]` — model inputs
- `.obsm["spatial_raw"]` / `.obsm["spatial"]` — original and scaled coordinates
- `.uns["spavelo_run"]` — run metadata (input path, parameters, native velocity key)

## Parameters

- `--input`: input H5AD file (single-file mode). Default: none
- `--metadata-file`: metadata CSV/TSV file for batch processing. Default: none
- `--output-dir`: root output directory. Required
- `--dataset-name`: dataset folder name in single-file mode. Default: input file stem
- `--spatial-key`: spatial coordinates key in `adata.obsm`. Default: `spatial`
- `--cluster-key`: column in `adata.obs` used for cell labels. Default: none
- `--spliced-layer`: spliced counts layer. Default: `spliced`
- `--unspliced-layer`: unspliced counts layer. Default: `unspliced`
- `--repo-dir`: spaVelo checkout containing `spaVeloDenoise.py` and `experiments/`. Default: `$SPAVELO_REPO_DIR` or `/opt/spavelo`
- `--device`: torch device, for example `cuda:0` or `cpu`. Default: `cuda:0`
- `--batch-size`: training batch size. Default: 256
- `--maxiter`: maximum training iterations. Default: 5000
- `--patience`: early-stopping patience. Default: 100
- `--overwrite`: overwrite existing outputs. Default: False
- `--seed`: random seed. Default: 144

The remaining model hyper-parameters are the `Config` fields of the original driver and
are not exposed on the command line (they keep their original values): `train_size`
0.95, `lr` 1e-3, `weight_decay` 1e-6, `encoder_layers` [256, 128], `decoder_layers`
[128, 256], `GP_dim` 2, `Normal_dim` 8, `dynamicVAE` True, `init_beta` / `min_beta` /
`max_beta` 4 / 1 / 25, `KL_loss` 0.025, `KL_weight` 15.0, `penalty_scale` 10.0,
`kernel_scale` 20.0, `loc_range` 20.0, `inducing_point_steps` 15
(`grid_inducing_points` True), `dirichlet_concentration` 0.25, `n_samples` 1 (train) /
25 (`get_velocity`), `gene_selection` `spatial`, `use_raw` False, `no_rho` False,
`dtype` `float64` and the 2000 highly variable genes of the original preprocessing.

## Usage

```bash
python spaVelo.py --input spatial.h5ad --output-dir ./output --spatial-key spatial
python spaVelo.py --input spatial.h5ad --output-dir ./output --maxiter 2000 --batch-size 128
python spaVelo.py --metadata-file datasets.csv --output-dir ./output
```

## Metadata File Format

Required columns:

- `dataset_name`
- `file_path`

Optional columns:

- `cluster_key` → defaults to empty (spaVelo does not require labels)
- `spatial_key` → defaults to `spatial`

```csv
dataset_name,file_path,cluster_key,spatial_key
cortex,/data/spatial/cortex.h5ad,,spatial
```

## Notes

- The whole pipeline is a statement-for-statement port of the original driver: the same
  `Config` class and field values, the same `spliced` / `unspliced` / `spatial` /
  `spatial_raw` / `prior_prob` / `pred_velocity` keys and the same local variable names.
  Only the argument parsing, the metadata batch loop and the `uns["spavelo_run"]` record
  are wrapper plumbing.
- The spatial GP is the defining feature: `obsm["spatial"]` is required and the
  coordinates are min-max scaled to `loc_range` before the inducing-point grid is
  built, matching the upstream driver.
- `--repo-dir` replaces the hard-coded repository path of the original script; the
  container default points at the cloned checkout.
- spaVelo does not plot, so this wrapper carries no plotting code or figure palette.
- The Dockerfile standardizes the conda environment name to `spavelo`.
- Verification scope: this wrapper was checked statically (AST parse, CLI construction,
  parameter-by-parameter comparison with `velocity_spatial_code/spavelo_run.py`) and the
  pipeline was executed once with stubbed scientific packages to confirm the output
  naming, the `args.csv` artefact and the `Config` values reaching the model call. No
  image was built and no dataset was run here, so the numerical behaviour of the
  upstream model on real spatial data was not re-confirmed.
