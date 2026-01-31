# Cell2fate Script

This script performs velocity analysis using Cell2fate on single-cell RNA sequencing data.

## Installation

```bash
pip install git+https://github.com/BayraktarLab/cell2fate
```

## Input Requirements

The input h5ad file must contain 'spliced' and 'unspliced' layers.

## Output

The velocity results are stored in the 'Velocity' layer of the output h5ad file.

## Parameters

- `--data_dir`: Input h5ad data file path (required)
- `--save_dir`: Result saving directory. Default: ./
- `--max_epochs`: Training epochs (1000–3000 recommended). Default: 1000
- `--batch_size`: Batch size (GPU: 512, CPU: 256). Default: 512
- `--cells_per_cluster`: Number of cells per cluster for training data preparation. Default: 100000
- `--simulate`: Whether the data is simulation data. If true, adjusts preprocessing parameters and sets cluster column to 'milestone', and sets X_umap from X_dimred if available.

## Usage

```bash
python cell2fate.py --data_dir data.h5ad --save_dir results --max_epochs 1500 --batch_size 256 --cells_per_cluster 50000 --simulate
```

For more details, see https://github.com/BayraktarLab/cell2fate and https://cell2fate.readthedocs.io/en/latest/