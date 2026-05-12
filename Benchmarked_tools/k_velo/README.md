# k-velo Script

This script preprocesses data and computes velocities using the `k-velo` pipeline (`velocity` package).

## Installation

```bash
git clone https://github.com/ValerieMarot/velocity_package && \
cd velocity_package && \
pip install -e ."
```

## Usage

- AnnData `.h5ad` file passed via `--data_path`.
- Required internals:
  - `layers['spliced']` and `layers['unspliced']`.
  - Optional: `uns['top_genes']` (used in `--simulate` mode if present).
  - `obs` column for cell/cluster labels (default key: `cell_type`) or provide `--celltype_key`.

Outputs
- Saved AnnData at `<save_dir>/<input_filename>` containing computed velocity results and fitted parameters.

CLI options (high-level)
- `--data_path` (required): input `.h5ad` file
- `--save_dir` (required): directory to write output
- `--celltype_key`: obs key for cell/cluster labels (default: `cell_type`)
- `--hvgs_n`: number of HVGs to select (default: 1000)
- `--hvgs_theta`: theta parameter for HVG selection (default: 100)
- `--minlim`: minimum counts threshold for high unspliced gene selection (default: 3)
- `--impute_n_neighbours`: neighbors for imputation (default: 30)
- `--impute_n_pcs`: PCs for imputation (default: 15)
- `--simulate`: if set, skips HVG and high-unspliced filtering, sets all cells' cluster label to `'milestone'`, and if `obsm['X_dimred']` exists copies it to `obsm['X_umap']`.

Notes
- The script no longer contains hardcoded local paths.
- In `--simulate` mode the preprocessing is relaxed to avoid aggressive filtering of simulated genes.
