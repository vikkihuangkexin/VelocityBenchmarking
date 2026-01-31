Usage: scVelo runner

This script runs stochastic + dynamical scVelo flows for a single AnnData `.h5ad` file.

Basic usage:

python scvelo.py --data_dir path/to/data.h5ad --data_file NAME --save_dir path/to/out

Options:
- `--data_dir` (required): path to input `.h5ad`
- `--data_file` (required): filename used for output naming
- `--save_dir` (required): directory to save outputs
- `--simulate` / `--no-simulate`: treat input as simulation (default: simulate)

Behavior:
- Default is `--simulate`: use simulation preprocessing (map `X_dimred` to `X_umap` if available,
  choose `top_gene` based on `n_vars`, call `scv.pp.filter_and_normalize(..., min_shared_counts=None, n_top_genes=top_gene)`,
  then `sc.pp.pca`/`sc.pp.neighbors`/`scv.pp.moments`).
- With `--no-simulate`, preprocessing uses `find_cluster_column` and conservative defaults.
