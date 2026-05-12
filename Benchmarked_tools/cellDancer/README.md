# cellDancer Script

This script converts an AnnData `.h5ad` to CellDancer inputs, computes gene-wise velocities, and aggregates to cell-level velocities. It writes a `*_velo.h5ad` file and a `cell_velo.csv`.

## Installation

```bash
pip install celldancer
pip install GPUtil
pip install scvelo
```

## Usage

```bash
python celldancer_S.py --data_dir path/to/data.h5ad --data_file NAME --save_dir path/to/out
```

## Options

- `--data_dir` (required): path to input `.h5ad`
- `--data_file` (required): filename used for outputs
- `--save_dir` (required): directory to save results
- `--n_jobs` (optional): number of parallel jobs (default auto)
- `--simulate` / `--no-simulate`: treat input as simulation (default: simulate)

## Behavior

- With `--simulate` (default), preprocessing uses simulation defaults (map `X_dimred` to `X_umap`, set `top_gene` according to `n_vars`, use `scv.pp.filter_and_normalize(..., min_shared_counts=None, n_top_genes=top_gene)`, then `sc.pp.pca`/`sc.pp.neighbors`/`scv.pp.moments`).
- With `--no-simulate`, preprocessing uses `find_cluster_column` and conservative defaults.
