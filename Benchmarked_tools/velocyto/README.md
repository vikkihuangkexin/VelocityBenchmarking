# Velocyto Runner

## Overview

This runner converts an AnnData `.h5ad` dataset into a loom file, runs Velocyto or compatible VelocytoLoom analysis, and saves velocity outputs and plots.

## Usage

```bash
python velocyto.py --data_dir path/to/data.h5ad --data_file NAME --save_dir path/to/out
```

## Required arguments

- `--data_dir` : path to input `.h5ad` file
- `--data_file` : base name used for output files
- `--save_dir` : directory to save generated outputs
- `--simulate` / `--no-simulate` : treat input as simulation data (default: `--simulate`)

## Behavior

- Default mode (`--simulate`) assumes the dataset is simulated and:
  - maps `obsm['X_dimred']` to `obsm['X_umap']` if needed
  - writes the loom file into the simulation loom directory
  - applies simulation preprocessing rules for gene selection, PCA, neighbors, and moments
- `--no-simulate` uses existing UMAP if available, can fall back to backup CSV data, deduplicates cells as needed, and follows more conservative scVelo preprocessing defaults.

## VelocytoLoom compatibility

- The script imports `VelocytoLoom` from `analysis_1`:
  `from analysis_1 import VelocytoLoom`
- Use `analysis_1.VelocytoLoom` when your loom file was not created by the standard velocyto R toolchain and may lack fields such as `ambiguous`.

## Outputs

- `{save_dir}/{NAME}.hdf5` : Velocyto HDF5 summary
- `{save_dir}/{NAME}_velo.h5ad` : AnnData with computed velocities
- `{save_dir}/grid_arrows_umap_legend.pdf` : example grid arrow figure

## Notes

This README documents the interface for `Benchmarked_tools/velocyto/velocyto.py`.