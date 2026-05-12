# scRNAKinetics Script

## Installation

```bash
pip install scanpy==1.10.3 scvelo==0.3.3
pip install RNAkinetics
```

## Purpose
- Compute RNA kinetics and velocity using scRNAKinetics package with CytoTRACE pseudotime.

## Inputs
- AnnData `.h5ad` file provided via `--data_path`.
- Required internals:
  - `layers['spliced']` and `layers['unspliced']`.
  - Optional: `uns['top_genes']` (used in `--simulate` mode to set `n_top_genes`).

Outputs
- Saved AnnData at `<save_dir>/lores.h5ad` containing kinetics and velocity results.

CLI options
- `--data_path` (required): input `.h5ad` file
- `--save_dir` (required): directory to save outputs
- `--simulate`: if set, set all cells' cluster label to `'milestone'` and copy `obsm['X_dimred']` to `obsm['X_umap']` if present
- `--n_top_genes`: number of top genes to select (default: 200)
- `--num_iter`: number of iterations for kinetics inference (default: 10)
- `--n_jobs`: number of jobs for parallel processing (default: 64)

Notes
- The script no longer uses hardcoded local paths.
- In `--simulate` mode, preprocessing is relaxed and cluster labels are set appropriately.