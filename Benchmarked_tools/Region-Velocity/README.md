# Region-Velocity Script

This directory contains scripts to run Region-Velocity analysis on single-cell RNA sequencing data. Region-Velocity is an R package for estimating region-specific velocities.

## Overview

The pipeline consists of three steps:
1. **Setup environment and install R package** (`run_regionvelocity_0.bash`)
2. **Run Region-Velocity analysis in R** (`run_regionvelocity_1.r`)
3. **Import results into AnnData and visualize with scVelo** (`run_regionvelocity_2.py`)

## Dependencies

- Conda/Mamba for environment management
- R with RegionVelocity package
- Python with scVelo, scanpy, etc.

## Installation

1. Create the conda environment:
   ```bash
   bash run_regionvelocity_0.bash
   ```

   This will create a `regionvelocity` environment and install the RegionVelocity R package from GitHub.

2. Activate the environment:
   ```bash
   conda activate regionvelocity
   ```

## Usage

### Step 1: Run Region-Velocity Analysis

```bash
Rscript run_regionvelocity_1.r -o output_csv_dir -n 8
```

This runs the Region-Velocity analysis on built-in spermatogenesis data and exports results to CSV files.

### Step 2: Import and Visualize

```bash
python run_regionvelocity_2.py --input-h5ad input.h5ad --csv-dir output_csv_dir --output-h5ad output.h5ad --fig-dir figures --n-jobs 4
```

This imports the Region-Velocity results into an AnnData object, runs scVelo downstream analysis, and generates visualization plots.

## Parameters

### run_regionvelocity_1.r

- `-o, --outdir`: Output directory for CSV files (default: regionvelocity_csv)
- `-n, --n-cores`: Number of cores for computation (default: 8)

### run_regionvelocity_2.py

- `--input-h5ad`: Input AnnData file (required)
- `--csv-dir`: Directory with Region-Velocity CSV outputs (required)
- `--output-h5ad`: Output AnnData file with velocity layers (required)
- `--fig-dir`: Directory to save figures (required)
- `--n-jobs`: CPU cores for velocity graph computation (default: 1)

## Outputs

- **CSV files**: Region-Velocity results (velocities, projections, parameters, etc.)
- **AnnData (.h5ad)**: Processed data with velocity layers for scVelo
- **Figures**: UMAP scatter, velocity stream, grid, and pseudotime plots in PNG and PDF formats

## Notes

- The built-in data in `run_regionvelocity_1.r` is spermatogenesis data. Modify the script to use your own data.
- Ensure the input AnnData matches the data used in the R step.
- Visualization uses fixed color palette for three cell types.