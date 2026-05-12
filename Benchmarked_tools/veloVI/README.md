# VeloVI Script

## Installation

```bash
conda create -n veloVI --yes python=3.8.0
conda activate veloVI
conda install --yes pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install --yes -c conda-forge scvi-tools
pip install velovi GPUtil scvelo==0.2.5
```

## Overview

This runner executes the VELOVI model on a single AnnData `.h5ad` dataset and saves velocity outputs and plots.

## Usage

```bash
python velovi.py --data_dir path/to/data.h5ad --data_file NAME --save_dir path/to/out
```

## Required arguments

- `--data_dir` : path to input `.h5ad` file
- `--data_file` : base name used for outputs
- `--save_dir` : directory to save generated files

## Optional arguments

- `--gpu_numbers` : comma-separated GPU indices (default: `0`)
- `--batch_size` : training batch size (default: `1024`)
- `--simulate` / `--no-simulate` : treat input as simulation data (default: `--simulate`)

## Behavior

- Default mode (`--simulate`) uses simulation-friendly preprocessing:
  - maps `obsm['X_dimred']` to `obsm['X_umap']` if available
  - selects `top_gene` based on `n_vars`
  - runs `scv.pp.filter_and_normalize(..., min_shared_counts=None, n_top_genes=top_gene)`
  - runs `sc.pp.pca`, `sc.pp.neighbors`, and `scv.pp.moments`
- `--no-simulate` uses conservative defaults and cluster annotations from `find_cluster_column()`.

## Notes

This README documents the command interface for `Benchmarked_tools/veloVI/velovi.py`.