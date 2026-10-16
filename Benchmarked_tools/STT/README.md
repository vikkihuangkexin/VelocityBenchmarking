# STT

This standalone entry script supports scRNA (`real` and `sim`) and spatial
(`spatial-real` and `spatial-sim`) workflows. The scRNA modes retain the
existing public preprocessing and model settings; the spatial modes use the
spatial benchmark workflow. Use the [team STT source fork](https://github.com/sd68515/STT_benchmark),
which adapts CellRank 1/2 imports and passes the explicit spliced layer and
dot-product rule to its transition kernel. These transition-input changes can
affect results in either modality. From the VelocityBenchmarking repository
root, clone the fork next to this checkout and add its package directory to
`PYTHONPATH`:

```bash
git clone https://github.com/sd68515/STT_benchmark.git ../STT_benchmark
export PYTHONPATH="$PWD/../STT_benchmark/example_notebooks${PYTHONPATH:+:$PYTHONPATH}"
```

Install compatible Scanpy, scVelo, NumPy, SciPy, pandas, and CellRank
dependencies. Spatial inputs
with more than 2,000 genes also need `scikit-misc` for Scanpy's
`seurat_v3` highly variable gene selection.

## Inputs and run

Each input is an H5AD file with `spliced` and `unspliced` layers, an annotation
column in `adata.obs`, and finite two-dimensional coordinates in `adata.obsm`.
Supply the existing annotation and coordinate keys explicitly. All five CSV
columns shown below are required. Run commands from the repository root:


```bash
python Benchmarked_tools/STT/STT.py --input sample.h5ad --output-dir results --dataset-id sample \
  --modality real --cluster-key cell_type --coordinate-key X_umap

python Benchmarked_tools/STT/STT.py --input simulated.h5ad --output-dir results --dataset-id simulation \
  --modality sim --cluster-key milestone --coordinate-key X_dimred

python Benchmarked_tools/STT/STT.py --input heart.h5ad --output-dir results --dataset-id heart \
  --modality spatial-real --cluster-key cell_type --coordinate-key spatial
```

For several files, use a CSV with these columns:

```csv
dataset_name,file_path,modality,cluster_key,dimred_key
rna_a,data/rna_a.h5ad,real,cell_type,X_umap
rna_sim,data/rna_sim.h5ad,sim,milestone,X_dimred
heart,data/heart.h5ad,spatial-real,cell_type,spatial
heart_sim,data/heart_sim.h5ad,spatial-sim,cell_type,spatial
```

Input paths may be absolute or relative to the CSV file.
Dataset IDs are used as supplied and must be unique. Run the CSV with:

```bash
python Benchmarked_tools/STT/STT.py --metadata-file datasets.csv --output-dir results
```

## Outputs

Each result is `results/<dataset_id>/adata_aggr.h5ad`. Existing results require
`--overwrite`. The aggregated object contains STT's `vj` velocity and `Ms`
layers, the supplied annotation and coordinates, and `uns['stt_run_config']`.
Spatial modes also provide `obsm['X_spatial']` for scVelo calls with
`basis='spatial'`. If that alias is missing, the script copies
`obsm['spatial']` to it.

## Preprocessing and mode settings

| Setting | `real` / `sim` | `spatial-real` / `spatial-sim` |
| --- | --- | --- |
| Gene preprocessing | Real: `min_shared_counts=20`, 2,000 top genes; simulation: no shared-count threshold, original gene-count rule | Up to 2,000 `seurat_v3` highly variable genes, then `min_shared_counts=1` |
| Expression neighbors | `scv.pp.moments(n_neighbors=50)` | PCA up to 30 components, expression neighbors up to 30, then scVelo moments |
| Spatial neighbors | None constructed by this entry | 20-neighbor graph from supplied coordinates, passed as `spa_conn_key='spatial'` |
| Dynamical iteration | 15 iterations, connectivity weight 0.5, 21 components, 20 neighbors, `thresh_ms_gene=0.2`, `use_spatial=True`, spatial weight 0.3, entropy threshold 0.1 | 15 iterations, connectivity weight 0.5, `max(20, number of states)` components, 20 neighbors, `thresh_ms_gene=0`, `use_spatial=True`, spatial weight 0.3, entropy threshold 0.1 |

Both workflows derive states from the supplied annotation by default. The
scRNA `sim` mode retains its original filtering rule and copies the
supplied coordinates to `X_umap` when that key is absent. The scRNA modes keep the
public entry's dynamical settings, including `use_spatial=True`; the entry
does not construct a spatial graph for this workflow. Its settings can be
changed with `--moments-n-neighbors` and the `--dyn-*` options.

Both spatial modes filter annotation classes with fewer than 10 cells before
fitting. Neither scRNA mode applies this category filter.
`--spatial-thresh-ms-gene` and `--spatial-n-jobs`
expose the spatial workflow's post-fit gene threshold and job count. The
method must return at least two states and a finite, nonzero `vj` layer.
