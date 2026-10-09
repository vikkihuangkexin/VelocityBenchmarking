# TIVelo Script

Wraps the TIVelo RNA velocity workflow for the VelocityBenchmarking pipeline. Real and
simulated datasets are handled by the same script; pass `--simulate` to use the
simulated-data branch. Datasets whose cluster transition graph is disconnected are
skipped gracefully instead of crashing.

## Installation

```bash
git clone https://github.com/cuhklinlab/TIVelo.git
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install scanpy python-igraph leidenalg scvelo==0.3.1 igraph louvain pybind11 optax==0.2.3
pip install TIVelo
```

## Input Requirements

- `adata.layers["spliced"]` and `adata.layers["unspliced"]`.
- A clustering column in `adata.obs` passed through `--cluster-key`.

Real data:

- `scv.pp.filter_and_normalize(min_shared_counts=20, n_top_genes=2000)`, `sc.pp.pca`,
  `sc.pp.neighbors(n_pcs=30, n_neighbors=30)` and `scv.pp.moments(n_pcs=None, n_neighbors=None)`.

Simulated data (`--simulate`):

- Uses the pre-computed `adata.obsm["X_dimred"]` embedding (copied into `obsm["X_umap"]`).
- Sets `min_shared_counts=None` and derives `n_top_genes` from the number of genes when the
  dataset has fewer than 2000 genes.
- Labels the cells with the `milestone` cluster when that column is absent.

## Output

For an input `test.h5ad` with `dataset_name=id_test` the result is written to:

```text
output/
└── id_test/
    ├── test.h5ad        # velocity result (layers["velocity"])
    └── native/          # untouched TIVelo outputs (h5ad, figures)
```

The output H5AD keeps TIVelo's native keys (`layers["velocity"]`, `uns["child_dict"]`,
`uns["path_dict"]`, `uns["velocity_graph"]`) and adds `uns["tivelo_run"]` run metadata.

Datasets with an unconnected cluster transition graph (`ValueError` containing
"unconnected sub-graphs", the simulated-only "max() arg is an empty sequence", or
`ArpackNoConvergence`) are reported and skipped; no output file is written for them.

## Parameters

- `--input`: input H5AD file (single-file mode).
- `--metadata-file`: metadata CSV/TSV file (batch mode).
- `--output-dir`: root output directory. Required.
- `--dataset-name`: dataset folder name in single-file mode. Default: input file stem.
- `--cluster-key`: column in `adata.obs` used for labels. Default: `milestone` with `--simulate`, `vis_annotation` otherwise.
- `--dimred-key`: embedding key in `adata.obsm`. Default: `X_umap`.
- `--simulate`: use the simulated-data branch. Default: `False`.
- `--n-top-genes`: number of highly variable genes. Default: `2000`.
- `--min-shared-counts`: minimum shared counts for the real-data gene filter. Default: `20`.
- `--n-pcs`: principal components for the neighbor graph. Default: `30`.
- `--n-neighbors`: neighbors for the neighbor graph. Default: `30`.
- `--n-epochs`: number of training epochs. Default: `100`.
- `--batch-size`: training batch size. Default: `1024`.
- `--loss-fun`: training loss function. Default: `mse`.
- `--no-filter-genes`: disable the internal gene filter. Default: filtering enabled.
- `--no-constrain`: disable the monotonicity constraint. Default: constraint enabled.
- `--velocity-key`: key used to store the velocity layer. Default: `velocity`.
- `--show-dti`: plot the directed transition graph. Default: `False`.
- `--adjust-dti`: adjust the directed transition graph. Default: `False`.
- `--measure-performance`: compute the internal TIVelo performance metrics. Default: `False`.
- `--n-jobs`: number of parallel jobs, `-1` uses all available cores. Default: `-1`.
- `--overwrite`: overwrite existing outputs. Default: `False`.
- `--seed`: random seed. Default: `2024`.

## Usage

```bash
python TIVelo.py --input data.h5ad --output-dir results --cluster-key celltype
python TIVelo.py --input sim.h5ad --output-dir results --cluster-key milestone --simulate
python TIVelo.py --metadata-file datasets.csv --output-dir results
```
