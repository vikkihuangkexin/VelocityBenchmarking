# MoFlow Script

MoFlow is a deep learning framework for multi-omic RNA velocity modeling that extends
the relay velocity model (cellDancer) by incorporating chromatin accessibility. It
couples the RNA modality with an auxiliary omic layer such as chromatin accessibility
gene activity.

## Installation

```bash
pip install moflow
```

For more information, see the [MoFlow repository](https://github.com/AriHong/MoFlow).

## Input Requirements

The input H5AD file must contain:

- `adata.layers["spliced"]`
- `adata.layers["unspliced"]`
- an auxiliary omic layer in `adata.layers`, by default `Mc` (chromatin accessibility gene activity), selected with `--extra-layers`
- `adata.obs[cluster_key]` (real data) or use `--simulate` and the `milestone` label is set automatically
- `adata.obsm["X_umap"]`, or a source embedding that is remapped to `X_umap` through `--dimargs`
- `adata.obsm["X_dimred"]` for simulated data, copied to `X_umap` when present

The RNA modality is passed to MoFlow as one AnnData object and the auxiliary omic matrix is passed as a second AnnData object whose expression matrix is set to the selected auxiliary layer.

## Output

For an input file `test.h5ad` with `--dataset-name test`, the output is written to:

```text
<output-dir>/
└── test/
    ├── test_moflow.h5ad    # native name, layers["velocity"]
    └── rc.h5ad             # alias of the file above
```

The output H5AD contains the exported velocity in `layers["velocity"]` together with the native MoFlow result keys. Run metadata is stored in `uns["moflow_run"]`.

## Parameters

- `--input`: Input H5AD file for single-file mode.
- `--metadata-file`: Metadata CSV/TSV file for batch processing.
- `--output-dir`: Root output directory (required).
- `--dataset-name`: Dataset folder name in single-file mode. Default: input file stem.
- `--cluster-key`: Column name in `adata.obs` used for labels in single-file mode.
- `--dimred-key`: Dimensionality reduction key in `adata.obsm`; use `X_umap` for real data and `X_dimred` for simulated data. Default: `X_umap`
- `--extra-layers`: Comma-separated `adata.layers` keys for the auxiliary omic matrix; MoFlow consumes a single matrix, so only the first entry is used. Default: `Mc`
- `--dimargs`: Comma-separated embedding remaps applied before training, each as `<source>` or `<source>:<target>`; the target defaults to `X_umap`. Default: `X_tsne:X_umap`
- `--simulate`: Simulated-data branch, sets the `milestone` label, copies `X_dimred` to `X_umap`, and relaxes the filtering. Default: `False`
- `--n-jobs`: Number of parallel jobs used by MoFlow. Default: `10`
- `--device`: Torch device passed to the MoFlow model, for example `cuda` or `cpu`; the model default is used when unset.
- `--preprocess-input`: Opt-in flag that runs `scv.pp.filter_and_normalize` before MoFlow. Disabled by default because `code/Moflow_sim.py` passes the input to the model without preprocessing. Default: `False`
- `--min-shared-counts`: Only used with `--preprocess-input`: minimum shared counts for `filter_and_normalize`. Default: `20`
- `--n-top-genes`: Only used with `--preprocess-input`: number of highly variable genes retained. Default: `2000`
- `--overwrite`: Overwrite existing outputs. Default: `False`
- `--seed`: Random seed. Default: `2024`

## Usage

```bash
python MoFlow.py --input data.h5ad --output-dir results --cluster-key celltype
python MoFlow.py --input sim.h5ad --output-dir results --cluster-key milestone --simulate
python MoFlow.py --input multiome.h5ad --output-dir results --cluster-key celltype --extra-layers Mc
python MoFlow.py --metadata-file datasets.csv --output-dir results
```
