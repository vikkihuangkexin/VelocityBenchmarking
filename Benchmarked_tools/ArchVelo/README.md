# ArchVelo Script

ArchVelo infers RNA velocity from paired single-cell RNA-seq and ATAC-seq
profiles by extracting shared archetypal chromatin accessibility programs and
modelling their regulatory influence on transcription. This wrapper normalizes
the original multi-omic driver into the VelocityBenchmarking interface and
exports the velocity field in `layers['velocity']`.

## Installation

```bash
pip install git+https://github.com/pritykinlab/ArchVelo.git
```

ArchVelo's own dependency list pins the required MultiVelo fork
(`github.com/MariaAvdeeva/MultiVelo-for-ArchVelo`) and a CUDA-enabled
`torch==2.3.1`, so no extra installation step is required.

## Input Requirements

- An `.h5ad` file passed through `--input`.
- Required layers:
  - `layers['spliced']` and `layers['unspliced']` (RNA) — used for the
    multi-omic chromatin model.
  - a chromatin accessibility layer, by default `layers['Mc']`, selected with
    `--extra-layers`.
- `obs[cluster_key]` with cell/cluster labels (`--cluster-key`, default
  `celltype`). In `--simulate` mode the column is created if missing and every
  cell is labelled `milestone`.
- A 2D embedding in `obsm` (`--dimred-key`, default `X_umap`). If the requested
  key is missing, `obsm['X_tsne']` (or `obsm['X_dimred']`) is mirrored into
  `obsm['X_umap']`; if none is available a UMAP is computed on the fly.
- The RNA and chromatin matrices share the same `obs_names` and `var_names`
  (guaranteed here because both objects are derived from the same input file).
- `Ms`/`Mu` moment layers are reused if present; otherwise they are computed
  with `scvelo.pp.moments` after the neighbour graph is built.

A metadata CSV/TSV file can be supplied through `--metadata-file` for batch
runs. Required columns: `dataset_name`, `file_path`, `cluster_key`. Optional
columns: `dimred_key`, `extra_layers`, `simulate`, `num_comps`.

## Output

```text
<output-dir>/
├── log_file/
│   └── archvelo_run.log
└── <dataset-name>/
    ├── <stem>_ArchVelo.h5ad           # native name, layers['velocity']
    ├── rc.h5ad                        # alias of the file above
    ├── archetypes/
    │   ├── cell_on_peaks_<k>_comps.csv
    │   ├── peak_on_peaks_<k>_comps.csv
    │   └── C_train_on_peaks_<k>_comps.csv
    ├── modeling_results/
    │   ├── multivelo_result_denoised_chrom.h5ad
    │   ├── adata_atac_AA_denoised.h5ad
    │   └── archvelo_results_pars.p
```

The output H5AD contains `layers['velocity']` (copied from the native
`layers['velo_s']`) together with ArchVelo's native keys (`velo_s`, `velo_s_*`,
`s`, `u`, `c`, `fit_t`, per-archetype component layers, `obs`, `var`, `obsm`).

## Parameters

- `--input`: input H5AD file for single-file mode.
- `--metadata-file`: metadata CSV/TSV file for batch processing.
- `--output-dir`: root output directory. Required.
- `--dataset-name`: dataset folder name in single-file mode. Default: input file stem.
- `--cluster-key`: obs column used for labels and plots. Default: `celltype`.
- `--dimred-key`: embedding key in `adata.obsm`. Default: `X_umap`.
- `--extra-layers`: comma-separated `adata.layers` keys; the first entry is the chromatin accessibility layer. Default: `Mc`.
- `--num-comps`: number of archetypes extracted from the chromatin matrix. Default: 10.
- `--n-jobs`: number of parallel jobs for model fitting. Default: 10.
- `--n-neighbors`: number of neighbours for ArchVelo model fitting. Default: 50.
- `--n-pcs`: number of principal components for ArchVelo model fitting. Default: 30.
- `--simulate`: simulated-data branch: cluster label forced to `milestone`, `obsm['X_dimred']` mirrored to `obsm['X_umap']`. Default: `False`.
- `--overwrite`: overwrite existing outputs. Default: `False`.
- `--seed`: random seed. Default: 2024.

## Usage

```bash
python ArchVelo.py --input multiome.h5ad --output-dir results --cluster-key celltype

python ArchVelo.py --input sim.h5ad --output-dir results --cluster-key milestone \
    --dimred-key X_dimred --simulate

python ArchVelo.py --input multiome.h5ad --output-dir results --cluster-key celltype \
    --extra-layers Mc,Ma --num-comps 12

python ArchVelo.py --metadata-file datasets.csv --output-dir results
```
