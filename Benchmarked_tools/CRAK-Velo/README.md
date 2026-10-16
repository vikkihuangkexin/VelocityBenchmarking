# CRAK-Velo Script

CRAK-Velo is a semi-mechanistic model that incorporates chromatin accessibility
data into the estimation of RNA velocities. This wrapper normalizes the original
config-driven driver into the VelocityBenchmarking interface, runs the upstream
pipeline and writes the fitted result with the velocity in `layers['velocity']`.

## Installation

```bash
pip install scanpy==1.8.2 scvelo==0.2.5 anndata==0.8.0 umap-learn==0.5.3 \
    numba==0.55.2 llvmlite==0.38.1 scipy==1.7.3 unitvelo \
    matplotlib==3.5.3 pandas==1.5.3 numpy==1.22.4
conda install -c conda-forge -c bioconda pybedtools
git clone https://github.com/StatBiomed/CRAK-Velo.git /opt/CRAK-Velo
```

## Input Requirements

Single-file mode needs three inputs:

- `--input`: RNA `.h5ad` with `layers['spliced']` and `layers['unspliced']` and
  gene coordinates in `var` (`chrom`, `chromStart`, `chromEnd`).
- `--atac-input`: chromatin accessibility `.h5ad` with region coordinates in
  `var` (`chrom`, `chromStart`, `chromEnd`) and the smoothed accessibility
  embedding in `obsm['cisTopic']`.
- `--config`: a base JSON template, for example one of the files shipped in
  `crak-velo/config` (`config_main_HSPC.json`, `config_main_10X_mouse_brain.json`).
  The wrapper overrides `adata_path`, `adata_atac_path`, `save_dir`,
  `base_trainer.save_dir`, `cluster_name`, `name`, `logger_config_path`,
  `system.seed` and `preprocessing.basis`/`window`, and writes the runtime config
  next to the outputs.

Additional requirements:

- `obs[cluster_key]` with cluster/cell-type labels (`--cluster-key`, default
  `celltype`). In `--simulate` mode a prepared copy of the RNA file is written
  with every cell labelled `milestone` and `obsm['X_dimred']` mirrored into the
  configured basis when available.
- `pybedtools` (and the `bedtools` binary) must be importable, because the
  upstream code intersects ATAC regions with genes.
- The upstream `crak-velo/main.py` must be reachable. `--crak-main` or the
  `CRAK_VELO_MAIN` environment variable override the default search path
  (`/opt/CRAK-Velo/crak-velo/main.py`, then `./crak-velo/main.py`).

Batch mode uses `--metadata-file` with required columns `dataset_name`,
`file_path`, `cluster_key` and optional columns `atac_file_path`, `config`,
`basis`, `window`, `simulate`.

## Output

```text
<output-dir>/
├── log_file/
│   └── crakvelo_run.log
└── <dataset-name>/
    ├── <stem>.h5ad                    # native name, layers['velocity']
    ├── rc.h5ad                        # alias of the file above
    ├── config.json                    # runtime config used for the run
    ├── config_logger.json             # only written when the upstream logger config is missing
    ├── prepared/                      # only in --simulate mode
    │   └── <stem>_prepared.h5ad
    └── crak_output/                   # upstream working directory
        ├── checkpoints/<name>/<run_id>/adata_rna_fit.h5ad
        └── checkpoints/<name>/<run_id>/adata_atac_fit.h5ad
```

The exported H5AD keeps the upstream fit and adds `uns['crakvelo_run']` run
metadata. The velocity is available as `layers['velocity']`.

## Parameters

- `--input`: RNA H5AD file for single-file mode.
- `--metadata-file`: metadata CSV/TSV file for batch processing.
- `--atac-input`: ATAC H5AD file for single-file mode.
- `--config`: base CRAK-Velo JSON config template. Required in single-file mode.
- `--crak-main`: path to the upstream `crak-velo/main.py`. Default: `/opt/CRAK-Velo/crak-velo/main.py` (or `CRAK_VELO_MAIN`).
- `--output-dir`: root output directory. Required.
- `--dataset-name`: dataset folder name in single-file mode. Default: input file stem.
- `--cluster-key`: obs column holding cluster/cell-type labels. Default: `celltype`.
- `--basis`: embedding basis used by the upstream model, overriding `preprocessing.basis`. Default: value from the config template.
- `--window`: window length used to intersect ATAC regions with genes. Default: 10000.
- `--simulate`: simulated-data branch: label cells `milestone` and mirror `obsm['X_dimred']` into the configured basis. Default: `False`.
- `--overwrite`: overwrite existing outputs. Default: `False`.
- `--seed`: random seed. Default: 2024.

## Usage

```bash
python CRAK-Velo.py --input rna.h5ad --atac-input atac.h5ad \
    --config crak-velo/config/config_main_HSPC.json \
    --output-dir results --cluster-key celltype

python CRAK-Velo.py --input sim_rna.h5ad --atac-input sim_atac.h5ad \
    --config config/config_simdata_01.json \
    --output-dir results --cluster-key milestone --basis tsne --simulate

python CRAK-Velo.py --metadata-file datasets.csv --output-dir results \
    --crak-main /opt/CRAK-Velo/crak-velo/main.py
```
