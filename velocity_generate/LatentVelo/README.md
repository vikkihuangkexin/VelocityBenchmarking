# LatentVelo Script
==========

Purpose
- Preprocess data and train LatentVelo model for velocity computation.

Inputs
- AnnData `.h5ad` file provided via `--data_path`.
- Required internals:
  - `layers['spliced']` and `layers['unspliced']`.
  - `obs` column for cell types (default key: `cell_type`) or provide `--celltype_key`.
  - Optional: `uns['top_genes']` (used in `--simulate` mode to set `n_top_genes`).

Outputs
- Saved AnnData at `<save_dir>/latent_adata.h5ad` containing latent space and velocity results.

CLI options
- `--data_path` (required): input `.h5ad` file
- `--save_dir` (required): directory to save outputs
- `--celltype_key`: obs key for cell/cluster labels (default: `cell_type`)
- `--simulate`: if set, relax preprocessing (set `min_shared_counts=None`, use `top_genes` if available), set all cells' cluster label to `'milestone'`, and copy `obsm['X_dimred']` to `obsm['X_umap']` if present
- `--min_shared_counts`: minimum shared counts for gene filtering (default: 5)
- `--n_top_genes`: number of top genes to select (default: 200)
- `--latent_dim`: latent dimension for the model (default: 40)
- `--encoder_hidden`: encoder hidden dimension (default: 45)
- `--zr_dim`: ZR dimension (default: 2)
- `--h_dim`: H dimension (default: 3)
- `--batch_size`: batch size for training (default: 1000)
- `--epochs`: number of training epochs (default: 50)
- `--grad_clip`: gradient clipping value (default: 100)
- `--random_seed`: random seed for reproducibility (default: 521)

Notes
- The script no longer uses hardcoded local paths; provide `--data_path` and `--save_dir`.
- In `--simulate` mode, preprocessing is relaxed to handle simulated data appropriately.