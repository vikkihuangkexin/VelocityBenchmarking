# TopoVelo

This standalone script provides two explicit TopoVelo benchmark workflows.
`scrna` retains the existing public post-analysis and plot workflow.
`spatial` uses the spatial benchmark training workflow and saves the model and
AnnData without that post-analysis or plot pass. Both modes use the supplied
two-dimensional coordinates to build TopoVelo's spatial graph; `scrna` is the
name of the existing public workflow, not a claim that TopoVelo ignores
coordinates.

Use the [team TopoVelo source fork](https://github.com/sd68515/TopoVelo_benchmark)
with compatible Scanpy, scVelo, and PyTorch dependencies:

```bash
git clone https://github.com/sd68515/TopoVelo_benchmark.git ../TopoVelo_benchmark
python -m pip install --no-deps -e ../TopoVelo_benchmark
```

## Input

Each H5AD input needs `spliced` and `unspliced` layers, an annotation column
in `adata.obs`, and finite two-dimensional coordinates in `adata.obsm`.
Provide the exact annotation and coordinate keys. Dataset IDs default to the
input stem; annotation/coordinate keys are never guessed from it. All five
CSV columns below are required. Run commands from the repository root:

```bash
python Benchmarked_tools/TopoVelo/topovelo.py --input sample.h5ad --output-dir results \
  --dataset-id sample --modality scrna \
  --cluster-key cell_type --coordinate-key X_umap

python Benchmarked_tools/TopoVelo/topovelo.py --input tissue.h5ad --output-dir results \
  --dataset-id tissue --modality spatial \
  --cluster-key cell_type --coordinate-key spatial
```

The input coordinates are copied to TopoVelo's internal `X_spatial` key.
Before calls using `basis='spatial'` or `embed='spatial'`, the script checks
that `X_spatial` exists and copies `obsm['spatial']` if the alias is missing.
For several datasets, use a CSV:

```csv
dataset_name,file_path,modality,cluster_key,dimred_key
sample,data/sample.h5ad,scrna,cell_type,X_umap
tissue,data/tissue.h5ad,spatial,cell_type,spatial
```

```bash
python Benchmarked_tools/TopoVelo/topovelo.py --metadata-file datasets.csv --output-dir results
```

CSV paths may be absolute or relative to the CSV. Dataset IDs must be unique.

## Outputs

The script creates
`results/<dataset_id>/adata_out.h5ad` and `results/<dataset_id>/model/`.
Use `--overwrite` if the dataset output directory already contains files.
`--device` accepts `auto` (default), `cpu`, `cuda`, or `cuda:<index>`.

## Preprocessing and mode settings

| Setting | `scrna` | `spatial` |
| --- | --- | --- |
| Preprocessing | Existing public `tpv.preprocess` defaults: 2,000 genes, minimum cell count 1, minimum expressed genes 1, UMAP off | Up to 2,000 genes, same count filters, UMAP off, annotation passed as `cluster_key`; cell loss stops the run |
| Graph | Public `KNN` default or `radius` with radius 30 | `KNN` with up to 16 neighbors |
| VAE | Time horizon 20, latent dimension 5, hidden sizes `(50,25,50)`, graph decoder and attention on | Time horizon 20, latent dimension 5, hidden sizes chosen from retained gene count, graph decoder and attention on |
| Training | Learning rates `2e-4`, `5e-3`, `2e-4`; other TopoVelo defaults | Same learning rates; batch size 128/256/512 by cell count, with validation interval `2 * batches_per_epoch`; trains with `cluster_key`, `embed='spatial'`, and `plot=False` |
| Post-analysis | TopoVelo `post_analysis`, scVelo velocity graph and pseudotime, plus stream, grid, and pseudotime plots in PNG/PDF | No TopoVelo post-analysis or plotting; scVelo graph and pseudotime are computed only when model time is constant |

The `scrna` options `--prep-*`, `--graph-*`, `--vae-*`, `--lr*`, and
`--post-*` retain the public entry's tunable settings. The spatial workflow
uses the spatial benchmark settings; `--spatial-batch-size` and
`--spatial-n-jobs` control its batch size and scVelo fallback jobs.

### Time fields and saved result

The result keeps TopoVelo's raw model time in `obs['gat_time']`. When it has a
discernible range, its min-max-scaled copy is stored separately in
`obs['gat_time_normalized']`. scVelo pseudotime, when computed, remains in
scVelo's default `obs['gat_velocity_pseudotime']` field. In `scrna` mode it is
computed after `post_analysis`; in `spatial` mode it is a fallback only when
model time is constant. These fields represent different calculations and
must not be interchanged.

The script writes `adata_out.h5ad` again after the selected postprocessing so
the stored file includes the final time and graph fields. It also records
the selected mode and settings in `uns['topovelo_run']`. The gene-space
velocity key is `layers['gat_velocity']`. A spatial run with usable model time
may have no saved velocity graph or projection; direction metrics can project
`gat_velocity` from its expression layer and neighbors when needed. `scrna` retains the
public post-analysis plot calls and four named plot types; `spatial` saves
the model and AnnData only.

## Fork behavior

The fork
restores compatibility with newer Matplotlib, limits parallel initialization
to eight threads by default (`TOPOVELO_N_JOBS` overrides this), and reduces
training memory. Its `float32` input conversion can change numerical results.
These source changes apply to both script modalities; the scRNA training and
post-analysis parameters remain those of the existing public entry.
