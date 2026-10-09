# Dynamo Script

Wraps the Dynamo RNA velocity workflow for the VelocityBenchmarking pipeline. Real and
simulated datasets are handled by the same script; pass `--simulate` to use the
simulated-data branch.

## Installation

```bash
pip install dynamo-release scanpy scvelo pyarrow simplejson GPUtil pynvml
```

## Input Requirements

- `adata.layers["spliced"]` and `adata.layers["unspliced"]` (Dynamo derives its moments from them).
- A clustering column in `adata.obs` passed through `--cluster-key`.

Real data:

- Uses `adata.obsm["X_umap"]` when a UMAP-like embedding already exists; otherwise it
  computes one with `sc.pp.neighbors(n_pcs=30, n_neighbors=30)` and `sc.tl.umap`.
- Gene filtering uses `dyn.pp.recipe_monocle(n_top_genes=2000, fg_kwargs={'shared_count': 20}, num_dim=30)`.

Simulated data (`--simulate`):

- Uses the pre-computed `adata.obsm["X_dimred"]` embedding (copied into `obsm["X_umap"]`).
- Relaxes the gene filter to `shared_count=1` and derives `n_top_genes` from the number of
  genes when the dataset has fewer than 2000 genes.
- Labels the cells with the `milestone` cluster when that column is absent.

## Output

For an input `test.h5ad` with `dataset_name=id_test` the result is written to:

```text
output/
└── id_test/
    ├── test.h5ad
    └── plot/
        ├── stream_arrow.pdf
        ├── grid_arrow.pdf
        └── full_arrow.pdf
```

The output H5AD keeps Dynamo's native keys and adds scVelo-compatible ones:

- `layers["velocity"]` — copy of `layers["velocity_S"]` for scVelo compatibility.
- `layers["velocity_S"]`, `obsm["velocity_umap"]` — Dynamo native velocity results.
- `uns["dynamo_run"]` — run metadata (input path, cluster key, model, output path).

The `plot/` PDFs are the `dyn.pl` streamline / grid / cell-wise vector plots that
`code/dynamo_1.py` writes; a plotting failure never discards a completed fit.

## Parameters

- `--input`: input H5AD file (single-file mode).
- `--metadata-file`: metadata CSV/TSV file (batch mode).
- `--output-dir`: root output directory. Required.
- `--dataset-name`: dataset folder name in single-file mode. Default: input file stem.
- `--cluster-key`: column in `adata.obs` used for labels. Default: `milestone` with `--simulate`, `vis_annotation` otherwise.
- `--dimred-key`: embedding key in `adata.obsm`. Default: `X_umap`.
- `--simulate`: use the simulated-data branch. Default: `False`.
- `--n-top-genes`: number of highly variable genes. Default: `2000`.
- `--n-pcs`: principal components for the neighbor graph. Default: `30`.
- `--n-neighbors`: neighbors for the neighbor graph. Default: `30`.
- `--num-dim`: dimensionality for `recipe_monocle`. Default: `30`.
- `--n-pca-components`: PCA components for `reduceDimension`. Default: `30`.
- `--shared-count`: minimum shared counts for the real-data gene filter. Default: `20`.
- `--model`: Dynamo dynamics model. Default: `stochastic`.
- `--fallback-model`: model used when the primary model raises `LinAlgError`. Default: `deterministic`.
- `--overwrite`: overwrite existing outputs. Default: `False`.
- `--seed`: random seed. Default: `2024`.

## Usage

```bash
python Dynamo.py --input data.h5ad --output-dir results --cluster-key celltype
python Dynamo.py --input sim.h5ad --output-dir results --cluster-key milestone --simulate
python Dynamo.py --metadata-file datasets.csv --output-dir results
```
