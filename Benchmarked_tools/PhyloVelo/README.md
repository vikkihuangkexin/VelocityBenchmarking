# PhyloVelo

This standalone script runs PhyloVelo on scRNA and lineage datasets. Install
the [benchmark fork](https://github.com/sd68515/PhyloVelo_forward_benchmark),
which adds forward velocity projection and a two-dimensional comparison graph:

```bash
pip install git+https://github.com/sd68515/PhyloVelo_forward_benchmark.git
```

## Inputs and single run

Provide an H5AD with integer counts in `X` or a named layer, a cell annotation
in `obs`, and finite two-dimensional coordinates in `obsm`. Cell and gene names
must be unique. The script removes cells with zero total counts, selects 1,000
highly variable genes from the count matrix, fits gene slopes, computes
pseudotime from the backward projection, and saves a forward projection for
direction-based metrics. Both modes use the same count and inference pipeline.

For scRNA data, supply one or more ordered annotation paths. Each path label
is mapped to its zero-based position; a label shared between branches must
have the same position in every path. This repairs the old entry's use of
non-numeric annotation strings as the inference clock.

```bash
python Benchmarked_tools/PhyloVelo/phylovelo.py \
  --input data/rna.h5ad --output-dir results/phylovelo \
  --modality scrna --annotation-key cell_type --coordinate-key X_umap \
  --lineage-path progenitor,intermediate,terminal_a \
  --lineage-path progenitor,intermediate,terminal_b
```

If a numeric time column exists, use `--clock-key time` to use it instead of
the path-position clock. The inference clock is stored in
`obs['phylovelo_clock']`; a clock also used as an evaluation label cannot give
an independent time-direction evaluation.

For lineage data, specify the integer count layer and a numeric observation
clock explicitly. Sampling day is a temporal proxy rather than measured cell
generation; choose a clock appropriate to the supplied dataset.

```bash
python Benchmarked_tools/PhyloVelo/phylovelo.py \
  --input data/lineage.h5ad --output-dir results/phylovelo \
  --modality lineage --annotation-key cell_type --coordinate-key X_umap \
  --count-layer raw_count --clock-key cell_generation
```

The clone-size clock used for one benchmark dataset is an explicit alternative:

```bash
python Benchmarked_tools/PhyloVelo/phylovelo.py \
  --input data/clones.h5ad --output-dir results/phylovelo \
  --modality lineage --annotation-key cell_type --coordinate-key X_umap \
  --count-layer matrix --clock-mode clone_size --clone-key X_clone \
  --cutoff 0.97
```

`X_clone` must have exactly one nonzero value, equal to 1, per cell. The clock is
`round(log(clone size), 1)`; training uses clones with 11-99 cells and projects
all retained cells. This clock requires at least 100 training cells and three
distinct eligible clone sizes. Change these limits with `--clone-min` and `--clone-max`.
The historical 0.97 cutoff for this dataset is supplied explicitly; the
general default is 0.95.

## CSV input

The CSV requires `dataset_name,file_path,modality,cluster_key,dimred_key`;
`modality` is `scrna` or `lineage`. Additional clock fields are required as
described above. For multiple datasets, pass a CSV with explicit keys. Relative `file_path`
values resolve from the CSV directory. For a path clock, separate branches
with `|` and labels within a branch with commas. Quote this CSV field.

```csv
dataset_name,file_path,modality,cluster_key,dimred_key,count_layer,clock_mode,clock_key,clone_key,lineage_paths
rna,data/rna.h5ad,scrna,cell_type,X_umap,X,path,,,"progenitor,intermediate,terminal_a|progenitor,intermediate,terminal_b"
lineage,data/lineage.h5ad,lineage,cell_type,X_umap,raw_count,obs,cell_generation,,
clones,data/clones.h5ad,lineage,cell_type,X_umap,matrix,clone_size,,X_clone,
```

```bash
python Benchmarked_tools/PhyloVelo/phylovelo.py \
  --metadata-file datasets.csv --output-dir results/phylovelo \
  --plot-dir results/phylovelo_plots
```

Optional CSV columns override CLI defaults per row: `genes`, `cutoff`,
`embedding_neighbors`, `pseudotime_neighbors`, `pseudotime_sample`,
`graph_neighbors`, `clone_min`, `clone_max`, and `seed`. Defaults are 1,000
genes, cutoff 0.95, 15 embedding neighbors, 0.1 pseudotime sampling fraction,
30 graph neighbors, and seed 42. Pseudotime uses 10 neighbors for scRNA and
30 for lineage unless specified. For example, a lineage dataset historically
run with 10 pseudotime neighbors should set that CSV column to 10.

## Outputs

Each row writes `<output-dir>/<dataset_id>.h5ad`. The output contains
`var['phylovelo_gene_velocity']`, forward
`obsm['phylovelo_velocity_umap']`, backward
`obsm['phylovelo_velocity_backward_umap']`,
`obs['phylovelo_pseudotime']`, and the forward
`uns['phylovelo_velocity_graph']`. The graph uses cosine alignment of the
forward two-dimensional velocity with neighboring embedding displacements.
The `--plot-dir` option saves embedding, stream, grid, and pseudotime PNGs
using the backward projection. `--overwrite` permits replacing existing H5AD
outputs.

## Projection and clock interpretation

PhyloVelo infers a single slope per gene. The official projection points
backward toward ancestral cells. The fork's `direction='forward'` recomputes
neighbor weights and the projection from the fitted slopes; it does not merely
negate a backward vector. The script copies both projections after one slope
fit: forward for benchmark direction metrics and backward for basic figures.
The fork leaves the original pseudotime calculation unchanged. These 2D
outputs are benchmark adaptations, not cell-by-gene velocities.
