# STT Velocity Script

## Installation

STT is a project-specific package. Install required Python packages:

```bash
pip install scanpy scvelo numpy scipy
# also ensure the local `stt` package is importable (pip install -e . if available)
```

## Usage

Single-file mode (real data):

```bash
python STT.py --data_dir /path/to/data.h5ad --save_dir ./output/STT --celltype_key cell_type
```

Simulated data (precomputed low-dim coordinates in `X_dimred`):

```bash
python STT.py --data_dir simulated.h5ad --save_dir ./output/STT --simulate --celltype_key milestone
```

## Input requirements

- AnnData `.h5ad` file with an expression matrix in `adata.X`.
- `adata.obs` should contain a cell-type column (default `cell_type`) unless `--simulate` is used.
- For simulated data, if `adata.obsm['X_dimred']` exists it will be copied to `adata.obsm['X_umap']` for plotting.

## Output

- `{save_dir}/adata_aggr.h5ad`: aggregated AnnData returned by `st.tl.dynamical_iteration`.

## Parameters (high level)

- `--simulate`: Treat input as simulation; relax preprocessing (sets `min_shared_counts=None`, `n_top_genes` auto).
- `--moments_n_neighbors`: `n_neighbors` passed to `scv.pp.moments` (default: 50).
- `--dyn_n_states`: Number of states for `st.tl.dynamical_iteration` (auto if not set).
- `--dyn_n_iter`: Iterations for `st.tl.dynamical_iteration` (default: 15).
- `--dyn_weight_connectivities`: `weight_connectivities` (default: 0.5).
- `--dyn_n_components`: `n_components` (default: 21).
- `--dyn_n_neighbors`: `n_neighbors` for dynamical iteration (default: 20).
- `--dyn_thresh_ms_gene`: `thresh_ms_gene` (default: 0.2).
- `--dyn_use_spatial`: `use_spatial` flag (default: True).
- `--dyn_spa_weight`: spatial weight (default: 0.3).
- `--dyn_thresh_entropy`: entropy threshold (default: 0.1).

## Notes

- The script exposes `st.tl.dynamical_iteration` parameters via CLI so you can reproduce previous runs.
- For simulated data the script sets `adata.obs['attractor']='milestone'` and will copy `X_dimred` to `X_umap` when available.

## Example CSV (batch-driving – not implemented here)

You can call the script per dataset in a loop; output directories should be unique per dataset/run to avoid overwriting.

## Contact / References

See the `stt` package documentation for details on `dynamical_iteration`.

