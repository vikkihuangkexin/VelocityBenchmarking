# Pyro-Velocity Script
=============

Purpose
- Train PyroVelocity model and compute velocities using probabilistic modeling.

Inputs
- AnnData `.h5ad` file provided via `--data_path`.
- Required internals:
  - `layers['spliced']` and `layers['unspliced']`.
  - `obsm['umap']` or specified `--umap_key` for embedding.
  - `obs` column for cell types (default key: `cell_type`) or provide `--celltype_key`.
  - Optional: `uns['top_genes']` (used in `--simulate` mode to set `n_top_genes`).

Outputs
- Processed data and model outputs saved in `<save_dir>` directory, including velocity results.

CLI options
- `--data_path` (required): input `.h5ad` file
- `--save_dir` (required): directory to save outputs
- `--celltype_key`: obs key for cell/cluster labels (default: `cell_type`)
- `--umap_key`: obsm key for UMAP coordinates (default: `umap`)
- `--simulate`: if set, relax preprocessing (set `min_shared_counts=None`), set all cells' cluster label to `'milestone'`, and copy `obsm['X_dimred']` to `obsm['X_umap']` if present
- `--n_top_genes`: number of top genes to select (default: 2000)
- `--min_shared_counts`: minimum shared counts for gene filtering (default: 5)
- `--max_epochs`: maximum training epochs (default: 300)
- `--batch_size`: batch size for training (default: -1, auto)

Notes
- The script generates a YAML config file and runs the PyroVelocity workflow.
- In `--simulate` mode, preprocessing is relaxed and cluster labels are adjusted for simulated data.