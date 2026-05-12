# DeepVelo Script

This script performs velocity analysis using DeepVelo on single-cell RNA sequencing data.

## Installation

```bash
pip install deepvelo
```

## Input Requirements

The input h5ad file must contain 'spliced' and 'unspliced' layers.

## Output

The velocity results are stored in the 'velocity' layer of the output h5ad file.

## Parameters

- `--data_dir`: Input h5ad data file path. Default: ./test.h5ad
- `--save_dir`: Result saving directory. Default: ./test
- `--simulate`: Whether the data is simulation data. If true, adjusts preprocessing parameters and sets X_umap from X_dimred if available.

## Usage

```bash
python DeepVelo.py --data_dir data.h5ad --save_dir results --simulate
```

For more details, see https://github.com/bowang-lab/DeepVelo