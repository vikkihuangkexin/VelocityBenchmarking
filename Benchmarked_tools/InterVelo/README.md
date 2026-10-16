# InterVelo

This entry point runs the benchmark's real RNA, simulated RNA, and ATAC
workflows with the public [InterVelo Python 3.11 fork](https://github.com/sd68515/InterVelo_py311).
Install the fork and the pinned NumPy/Numba versions:

```bash
pip install git+https://github.com/sd68515/InterVelo_py311.git
pip install "numpy==1.26.4" "numba==0.59.1"
```

Install a compatible PyTorch build for your CPU or CUDA environment before
running the script.

## Input

Each H5AD file needs `adata.layers['spliced']`,
`adata.layers['unspliced']`, an annotation column in `adata.obs`, and
coordinates in `adata.obsm`. ATAC also needs `adata.layers['Mc']`.
The coordinate key is supplied explicitly; the script preserves those
coordinates for velocity projection and plots. When this expression-only
method is compared on a spatial dataset, use the applicable `real` or `sim`
mode; spatial coordinates are not an extra model input.

| Modality | Filtering | Matrix used as `X` | Model input |
| --- | --- | --- | --- |
| `real` | `min_shared_counts=20`; remove zero-spliced cells | `spliced` | `[Ms, Mu]` |
| `sim` | scVelo defaults | Input `X` | `[Ms, Mu]` |
| `atac` | `min_shared_counts=20` | Input `X` | `[Ms, Mu, Mc]` |

The source contains an optional, commented-out line for removing cells with
identical `X` rows. It converts `X` to a dense DataFrame and can exhaust
memory on large datasets.

## Run one dataset

Run commands from the repository root.

```bash
python Benchmarked_tools/InterVelo/InterVelo.py --input sample.h5ad --output-dir results \
  --modality real --cluster-key cell_type --dimred-key X_umap
```

`--modality` accepts `real`, `sim`, or `atac`.
The input coordinate key may be `X_umap`, `X_dimred`, `X_spatial`, or another
existing `adata.obsm` key. scVelo requires an internal `X_<basis>` key for
projection, so the script copies the selected coordinates there when needed.
It does not compute replacement coordinates.

## Run a dataset CSV

```csv
dataset_name,file_path,modality,cluster_key,dimred_key
sample_a,data/sample_a.h5ad,real,cell_type,X_umap
sample_b,data/sample_b.h5ad,sim,milestone,X_dimred
sample_c,data/sample_c.h5ad,atac,cell_type,X_umap
```

`file_path` can be absolute or relative to the CSV. All five columns are
required. Dataset names identify output folders and need not follow the
paper's ID scheme.

```bash
python Benchmarked_tools/InterVelo/InterVelo.py --metadata-file datasets.csv --output-dir results
```

Use `--seed` to choose the training seed, `--overwrite` to replace an
existing H5AD result, and `--save-pdf` to save PDF plots alongside PNGs.

## Output

For dataset `sample_a`, the main output is
`results/sample_a/sample_a_plot.h5ad`, with figures under `plot/` and model
files under `saved/`. The H5AD contains the model's `velocity` layer and
`pseudotime`, the projected `velocity_<basis>` coordinates, and run settings
in `adata.uns['intervelo_run']`. InterVelo may also produce
`velocity_pseudotime`; `intervelo_pseudotime_normalized` rescales its model
`pseudotime` to the unit interval for plotting.

## Independent repeated training

Independent repeated training is provided by
[`Batch_run/intervelo_robust.py`](../../Batch_run/intervelo_robust.py):

```bash
python Batch_run/intervelo_robust.py --input-csv datasets.csv \
  --output-dir repeated/intervelo --runs 5
```

It requires all five shared CSV columns and preprocesses once using
`--preprocess-seed`. Run indices start at 1 and use `--base-seed + run_index`.
Results are `<output-dir>/<modality>/<dataset_name>/InterVelo_<dataset_name>_r<run>.h5ad`;
existing results are skipped. The repeated result settings are stored in
`uns['intervelo_batchrun']`.
