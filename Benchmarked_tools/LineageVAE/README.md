# LineageVAE

This standalone entry trains the original [LineageVAE](https://github.com/LzrRacer/LineageVAE)
architecture on lineage-labeled count data and exports benchmark velocities.
Clone the upstream repository and make its `src` directory importable; its
`modules.py` imports `funcs.py` from that same directory. Install its declared
dependencies in the same environment as Scanpy, scVelo, and PyTorch.

```bash
git clone https://github.com/LzrRacer/LineageVAE.git external/LineageVAE
export PYTHONPATH="$PWD/external/LineageVAE/src:$PYTHONPATH"
```

## Inputs

The H5AD needs integer counts in `X` or a declared layer, a numeric,
nonnegative integer time column in `obs`, clone IDs in `obs` or one-hot clone
membership in `obsm`, a cell annotation column, and finite two-dimensional
coordinates. The script removes zero-count cells, selects 1,000 highly
variable genes from the original counts, and trains the embedding and lineage
dynamics for 100 epochs each by default. It initializes the upstream model's
otherwise uninitialized `d_coeff` to 0.05 before training. Cell and gene
names must be unique. At least two clones must satisfy the training filters
and contain at least `clone_batch_size` cells each; a clone batch must fit at
least one cell from every observed time. The selected count matrix must have
positive cell and gene totals.

## Run

Run commands from the repository root.

```bash
python Benchmarked_tools/LineageVAE/lineagevae.py \
  --input data/lineage.h5ad --output-dir results/lineagevae \
  --annotation-key cell_type --coordinate-key X_umap \
  --count-layer raw_count --time-key Day --clone-key Lineage \
  --clone-batch-size 30
```

For several datasets, provide a CSV. Required columns are
`dataset_name,file_path,modality,cluster_key,dimred_key,count_layer,time_key`,
plus exactly one of `clone_key` or `clone_matrix_key`. All modality-specific fields are
explicit; the script makes no decision from a dataset ID. Paths in `file_path`
resolve relative to the CSV directory. A clone matrix needs exactly one
nonzero value, equal to 1, per cell. `required_day` and `min_distinct_days` filter eligible
training clones, and `clone_batch_size` controls how many cells from each
clone enter one dynamics batch.

```csv
dataset_name,file_path,modality,cluster_key,dimred_key,count_layer,time_key,clone_key,clone_matrix_key,clone_batch_size,required_day,min_distinct_days
clones,data/clones.h5ad,lineage,cell_type,X_umap,matrix,time_info,,X_clone,20,2,1
sampled,data/sampled.h5ad,lineage,cell_type,X_umap,raw_count,Day,Lineage,,30,,1
sparse_clones,data/sparse_clones.h5ad,lineage,cell_type,X_umap,total,day,Lineage,,2,,2
```

```bash
python Benchmarked_tools/LineageVAE/lineagevae.py \
  --metadata-file datasets.csv --output-dir results/lineagevae \
  --plot-dir results/lineagevae_plots
```

The CSV may also set `pca_key`, `genes`, `embed_batch_size`, `embed_epochs`,
`dynamics_epochs`, `learning_rate`, `seed`, and `threads` per dataset. The
default clone batch size is 30, learning rate is `1e-4`, and seed is 42.
`--device auto|cpu|cuda` selects hardware. Use `--checkpoint-dir` to save model
weights separately from the H5AD output.

## Outputs

The saved `<output-dir>/<dataset_id>.h5ad` contains `layers['velocity']`, a
cell-by-gene forward change in log-normalized decoded expression over one
numeric time unit. Cells at time zero have zero model velocity. It also
contains `obsm['velocity_umap']`, scVelo's gene-space velocity graph,
`obs['velocity_pseudotime']`, and a separate embedding-derived 2D cosine
graph under `uns['velocity_umap_graph']`. The 2D graph is a benchmark
comparison artifact, not the gene-space graph. Optional plots use the saved
embedding, projected velocity, and pseudotime. Input and resource paths are
not written to result metadata. Existing H5AD outputs and checkpoints are
refused; choose a fresh output directory for another run. The result retains
`layers['raw_count']`, `layers['Ms']`, modeling `obs['Day']` and
`obs['Lineage']`, and settings/coverage in `uns['lineagevae_run']`.
