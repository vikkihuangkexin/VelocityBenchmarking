# Velocycle Script

This script performs cell cycle phase inference using Velocycle on single-cell RNA sequencing data.

## Installation

For installation details, see https://github.com/lamanno-epfl/velocycle

## Input Requirements

The input h5ad file must contain 'spliced' and 'unspliced' layers.

## Output

The cell cycle phase inference results are stored in the 'velocycle_phase' column of the output h5ad file.

## Parameters

- `--data_dir`: Input h5ad file (required)
- `--save_dir`: Output directory (required)
- `--num_steps`: Number of training steps. Default: 1000
- `--lr_start`: Starting learning rate. Default: 0.03
- `--lr_end`: Ending learning rate. Default: 0.005
- `--simulate`: Whether the data is simulation data. If true, sets X_umap from X_dimred if available.

## Usage

```bash
python velocycle.py --data_dir data.h5ad --save_dir results --simulate
```

For more details, see https://github.com/lamanno-epfl/velocycle