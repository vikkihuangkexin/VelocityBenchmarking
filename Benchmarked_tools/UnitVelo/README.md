# UnitVelo Script

## Installation

```bash
conda create -n UnitVelo --yes python=3.9.0
conda activate UnitVelo
pip install unitvelo GPUtil pynvml
```

This script runs UnitVelo on a single AnnData `.h5ad` file and saves velocity outputs and plots.

## Usage

python unitvelo.py --data_dir path/to/data.h5ad --save_dir path/to/out

Options:
- `--data_dir` (required): input `.h5ad` file
- `--save_dir` (required): output directory
- `--gpu`: CUDA_VISIBLE_DEVICES (default: "1")
- `--normalize` (flag): pass `normalize=True` to `run_model`
- `--simulate` / `--no-simulate`: treat input as simulation (default: simulate)

Behavior differences:
- If `--simulate` is set (default), preprocessing follows the simulation rules: attempts
  to use `obsm['X_dimred']` as UMAP, sets `top_gene` based on `n_vars`, and uses
  `scv.pp.filter_and_normalize(..., min_shared_counts=None, n_top_genes=top_gene)` plus `sc.pp.pca`/`sc.pp.neighbors`/`scv.pp.moments`.
- If `--no-simulate`, preprocessing uses `find_cluster_column` and conservative defaults.
