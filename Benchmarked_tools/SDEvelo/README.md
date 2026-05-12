# SDEvelo Script

## Installation

```bash
conda create -n velo --yes python=3.9.0
conda activate velo
pip install sdevelo
```

## Train SDEvelo model and save processed AnnData with velocity results.

## Inputs
- An AnnData `.h5ad` file provided to `--data_path`.
- Required internals (recommended):
  - `layers['spliced']` and `layers['unspliced']` for velocity computation
  - `obs` column for cell types (default key: `cell_type`) or provide `--celltype_key`
  - Optional: `obsm['X_dimred']` (will be copied to `obsm['X_umap']` in `--simulate` mode)

Outputs
- Saved AnnData at `<save_dir>/adata.h5ad` containing training/velocity results.

CLI
- `--data_path` (required): path to input `.h5ad` file
- `--save_dir` (required): directory to write outputs
- `--celltype_key`: obs column used for cell/cluster labels (default: `cell_type`)
- `--gpu`: GPU id to use; set `-1` for CPU
- `--n_epochs`: number of training epochs (default follows `sdevelo.Config`)
- `--simulate`: if set, relax preprocessing (sets `min_shared_counts=None`), sets all cells' cluster label to `'milestone'`, and if present copies `obsm['X_dimred']` to `obsm['X_umap']`.

Notes
- The script no longer uses hardcoded local paths; provide `--data_path` and `--save_dir`.
- If using simulated data, pass `--simulate` to avoid aggressive filtering and ensure embeddings/cluster labels are compatible with downstream evaluation.
