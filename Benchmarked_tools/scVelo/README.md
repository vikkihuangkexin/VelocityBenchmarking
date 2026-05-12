# scVelo Script

## Installation

```bash
conda create -n scVelo -c conda-forge --yes python=3.8.0
conda activate scVelo
pip install numpy==1.21.1 scvelo==0.2.5 GPUtil
```

## Overview

This runner executes stochastic and dynamical velocity workflows in scVelo for a single AnnData `.h5ad` dataset.

## Usage

```bash
python scvelo.py --data_dir path/to/data.h5ad --data_file NAME --save_dir path/to/out
```

## Required arguments

- `--data_dir` : path to input `.h5ad` file
- `--data_file` : base name used for output files
- `--save_dir` : output directory for generated files

## Optional arguments

- `--simulate` / `--no-simulate` : treat input as simulation data (default: `--simulate`)

## Behavior

- Default mode (`--simulate`) uses simulation-friendly preprocessing:
  - maps `obsm['X_dimred']` to `obsm['X_umap']` when available
  - selects `top_gene` based on `n_vars`
  - runs `scv.pp.filter_and_normalize(..., min_shared_counts=None, n_top_genes=top_gene)`
  - runs `sc.pp.pca`, `sc.pp.neighbors`, and `scv.pp.moments`
- `--no-simulate` uses a more conservative preprocessing path and relies on cluster annotations found by `find_cluster_column()`.

## Notes

This README is intended to document the script interface and expected behavior for `Benchmarked_tools/scVelo/scvelo.py`.