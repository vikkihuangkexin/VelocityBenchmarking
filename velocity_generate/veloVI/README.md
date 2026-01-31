Usage: VELOVI runner

This script runs the VELOVI model on a single AnnData `.h5ad` file and writes velocity outputs and plots.

Basic usage:

python velovi.py --data_dir path/to/data.h5ad --data_file NAME --save_dir path/to/out

Options:
- `--data_dir` (required): path to input `.h5ad`
- `--data_file` (required): filename identifier used for output naming
- `--save_dir` (required): directory to save outputs
- `--gpu_numbers` (default: "0"): comma-separated GPU indices
- `--batch_size` (default: 1024): training batch size
- `--simulate` / `--no-simulate`: treat input as simulation (default: simulate)

Behavior:
- When `--simulate` is used (default), preprocessing uses simulation-friendly defaults: maps
  `obsm['X_dimred']` to `X_umap` if present, selects `top_gene` according to `n_vars`, and
  uses `scv.pp.filter_and_normalize(..., min_shared_counts=None, n_top_genes=top_gene)` followed
  by `sc.pp.pca`/`sc.pp.neighbors`/`scv.pp.moments`.
- When `--no-simulate`, preprocessing uses `find_cluster_column` and conservative defaults.
