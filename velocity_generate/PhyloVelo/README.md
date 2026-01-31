# PhyloVelo Script
=========

Purpose
- Compute phylogenetic velocity and pseudotime using the phylovelo package.

Inputs
- AnnData `.h5ad` file provided via `--data_path`.
- Required internals:
  - `layers['spliced']` and `layers['unspliced']` (used for count data).
  - `obsm['umap']` or specified `--umap_key` for embedding.
  - `obs` column for cell types (default key: `cell_type`) or provide `--celltype_key`.
  - Optional: `uns['top_genes']` (used in `--simulate` mode to set `n_top_genes`).

Outputs
- Saved AnnData at `<save_dir>/adata.h5ad` containing velocity embeddings and pseudotime.

CLI options
- `--data_path` (required): input `.h5ad` file
- `--save_dir` (required): directory to save outputs
- `--celltype_key`: obs key for cell/cluster labels (default: `cell_type`)
- `--umap_key`: obsm key for UMAP coordinates (default: `umap`)
- `--simulate`: if set, set all cells' cluster label to `'milestone'` and copy `obsm['X_dimred']` to `obsm['X_umap']` if present
- `--n_top_genes`: number of top genes to select (default: 2000)
- `--min_count`: minimum count for normalization filter (default: 10)
- `--lineage_path`: lineage paths for phylogenetic analysis

Notes
- The script no longer uses hardcoded local paths.
- In `--simulate` mode, preprocessing is relaxed and cluster labels are set appropriately.
