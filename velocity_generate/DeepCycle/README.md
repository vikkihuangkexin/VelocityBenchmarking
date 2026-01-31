# DeepCycle Script

This script performs cell cycle phase inference using DeepCycle on single-cell RNA sequencing data.

## Installation

For installation details, see https://github.com/andreariba/DeepCycle

## Input Requirements

The input h5ad file must contain 'spliced' and 'unspliced' layers, and must have been processed with scvelo.pp.moments.

## Output

The cell cycle phase inference results are stored in the 'cell_cycle_theta' column of the output AnnData file.

## Parameters

- `--input_adata`: Input AnnData file preprocessed with velocyto and scvelo (moments) (required)
- `--gene_list`: Subset of genes to run the inference on (required)
- `--base_gene`: Gene used to have an initial guess of the phase (required)
- `--expression_threshold`: Unspliced/spliced expression threshold (required)
- `--gpu`: Use GPUs (optional)
- `--hotelling`: Use Hotelling filter (optional)
- `--output_adata`: Output AnnData file (required)
- `--output_dir`: Output directory for results (default: .)
- `--simulate`: Whether the data is simulation data. If true, sets X_umap from X_dimred if available.

## Usage

```bash
python DeepCycle.py \
  --input_adata test.h5ad \
  --gene_list go_annotation/GO_cell_cycle_annotation_mouse.txt \
  --base_gene Nusap1 \
  --expression_threshold 0.5 \
  --gpu \
  --hotelling \
  --output_adata result_DeepCycle.h5ad \
  --simulate
```

For more details, see https://github.com/andreariba/DeepCycle