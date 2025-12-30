# Ground Truth NPZ Data Preparation

## Overview

This directory contains precomputed ground truth (GT) velocity embeddings for simulation data. The NPZ files store GT velocity in low-dimensional space for efficient comparison with predicted velocity results.

## Required Data Structure

Before using the Ground Truth Correlation metric, organize your simulation data as follows:

### Input Data Structure (simdata_local)

```
simdata_local/
├── bifurcating/
│   ├── bifurcating_cell1000_gene1000_dataset.h5ad
│   ├── bifurcating_cell2000_gene1000_dataset.h5ad
│   └── ...
├── cycle-simple/
│   ├── cycle-simple_cell1000_gene1000_dataset.h5ad
│   └── ...
├── trifurcating/
│   ├── trifurcating_cell1000_gene1000_dataset.h5ad
│   └── ...
├── linear-simple/
├── bifurcating-loop/
├── consecutive-bifurcating/
├── disconnected/
├── simulation-add/
├── cellsub-bifurcating/
└── genesub-bifurcating/
```

**Required fields in H5AD files:**
- `obs`: `cell_id`, `milestone`
- `var`: `gene_id`
- `obsm`: `X_dimred` (2D embedding coordinates)
- `layers`: `ground_truth_velocity`, `spliced`, `unspliced`

### Output NPZ Structure (simdata_gtkey)

After running the precomputation script, NPZ files will be organized as:

```
simdata_gtkey/
├── bifurcating/
│   ├── bifurcating_cell1000_gene1000_gt_data.npz
│   ├── bifurcating_cell2000_gene1000_gt_data.npz
│   └── ...
├── cycle-simple/
│   ├── cycle-simple_cell1000_gene1000_gt_data.npz
│   └── ...
└── ...
```

**NPZ file contents:**
- `gt_dimred`: Ground truth velocity in 2D space (n_cells, 2)
- `X_dimred`: 2D embedding coordinates (n_cells, 2)
- `cell_names`: Original cell names
- `cell_names_unique`: Unique cell names (if duplicates exist)

## Usage

### Precompute GT Embeddings

If you have the original simulation H5AD files, precompute GT embeddings:

```bash
python get_gt_concurrent.py \
    --input-dir /path/to/simdata_local \
    --output-dir /path/to/simdata_gtkey \
    --topologies bifurcating cycle-simple trifurcating \
    --verbose
```

**Parameters:**
- `--input-dir`: Base directory containing simulation H5AD files
- `--output-dir`: Output directory for NPZ files
- `--topologies`: List of topology types to process (optional, defaults to all)
- `--verbose`: Print detailed progress (optional)

### Use Precomputed NPZ Files

When running Ground Truth Correlation metric, specify the NPZ directory:

```python
from groundtruth_correlation import calculate_groundtruth_correlation

result = calculate_groundtruth_correlation(
    adata_or_path='result.h5ad',
    method='VeloVAE',
    dataset_id='bifurcating_cell1000_gene1000',
    output_csv='results.csv',
    gt_npz_base_dir='/path/to/simdata_gtkey'  # Point to NPZ directory
)
```

Or via command line:

```bash
python groundtruth_correlation.py \
    --input result.h5ad \
    --method VeloVAE \
    --dataset-id bifurcating_cell1000_gene1000 \
    --output-csv results.csv \
    --gt-npz-dir /path/to/simdata_gtkey
```

## Important Notes

### Dataset ID Format

Dataset IDs must use **hyphens** (not underscores) in topology names:

- ✅ Correct: `bifurcating-loop_cell1000_gene1000`
- ✅ Correct: `cycle-simple_cell5000_gene500`
- ❌ Wrong: `bifurcating_loop_cell1000_gene1000`
- ❌ Wrong: `cycle_simple_cell5000_gene500`

### Topology Types

Supported topology types:
- `bifurcating`
- `cycle-simple`
- `trifurcating`
- `linear-simple`
- `bifurcating-loop`
- `consecutive-bifurcating`
- `disconnected`
- `simulation-add`
- `cellsub-bifurcating` (gene-subsampled bifurcating)
- `genesub-bifurcating` (cell-subsampled bifurcating)

### File Naming Convention

Files must follow the pattern: `{topology}_{cellcount}_gene{genecount}_dataset.h5ad`

Examples:
- `bifurcating_cell1000_gene1000_dataset.h5ad`
- `cycle-simple_cell5000_gene500_dataset.h5ad`
- `trifurcating_cell2000_gene1000_dataset.h5ad`

Output NPZ files will use the same naming with `_gt_data.npz` suffix:
- `bifurcating_cell1000_gene1000_gt_data.npz`

## Downloading Data

**If you don't have the original simulation data:**

1. Download the precomputed NPZ files from [repository/release link]
2. Extract to a directory (e.g., `/path/to/simdata_gtkey`)
3. Ensure the directory structure matches the format above
4. Use the `--gt-npz-dir` parameter when running the metric

**If you have the original H5AD files:**

1. Organize them according to the input structure above
2. Run `get_gt_concurrent.py` to precompute NPZ files
3. Use the output directory with the metric

## Requirements

```bash
pip install scanpy scvelo numpy
```

## Troubleshooting

### "GT NPZ file not found"

**Cause:** Dataset ID doesn't match NPZ file naming

**Solution:** Check dataset ID format (hyphens vs underscores) and verify the file exists:
```bash
ls /path/to/simdata_gtkey/{topology}/{dataset_id}_gt_data.npz
```

### "Topology directory not found"

**Cause:** Topology directory missing in NPZ base directory

**Solution:** Ensure all required topology directories exist in the NPZ base directory, or run precomputation for missing topologies.
