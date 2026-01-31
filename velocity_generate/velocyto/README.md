Usage: velocyto runner

This script prepares a loom file and runs Velocyto / VelocytoLoom analysis to compute
gene-wise and cell-wise velocities, then saves HDF5/H5AD outputs and plots.

Basic usage:

python velocyto.py --data_dir path/to/data.h5ad --data_file NAME --save_dir path/to/out

Options:
- `--data_dir` (required): path to input `.h5ad`
- `--data_file` (required): filename used for outputs
- `--save_dir` (required): directory to save outputs
- `--simulate` / `--no-simulate`: treat input as simulation (default: simulate)

Behavior and preprocessing:
- When `--simulate` is used (default), the script assumes the dataset is simulated and:
  - maps `obsm['X_dimred']` to `obsm['X_umap']` if needed,
  - writes the loom to the simulation loom directory,
  - follows simulation preprocessing choices (top genes selection, PCA, neighbors, moments).
- When `--no-simulate`, the script will attempt to use existing UMAP (or a backup CSV),
  de-duplicate cells if needed, and use conservative scVelo preprocessing defaults.

Velocyto vs custom VelocytoLoom usage:
- The code imports `VelocytoLoom` from `analysis_1` via:
  `from analysis_1 import VelocytoLoom`
- Use `VelocytoLoom` from `analysis_1` when your loom/layers were NOT produced by the
  standard velocyto R toolchain (i.e. were created externally or converted by custom code).
  Such looms often do NOT include an `ambiguous` column in the cell attributes and therefore
  cannot be handled by `velocyto.VelocytoLoom` directly. In that case, `analysis_1.VelocytoLoom`
  provides compatible loading and handling for those looms.

Outputs:
- `{save_dir}/{NAME}.hdf5` : Velocyto HDF5 summary
- `{save_dir}/{NAME}_velo.h5ad` : AnnData with velocity results
- `{save_dir}/grid_arrows_umap_legend.pdf` : example figure of grid arrows
