# DeepKINET Script

DeepKINET is a deep generative model that estimates single-cell mRNA splicing and
degradation kinetics from raw spliced/unspliced counts. This wrapper runs the
DeepKINET kinetics workflow on one dataset or over a metadata table and writes a
`<stem>_velo.h5ad` result file.

## Installation

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/3254c/DeepKINET.git
cd DeepKINET && pip install -e .
pip install anndata==0.10.3 scanpy==1.9.6 scvelo==0.2.5
pip install pandas==1.5.3 numpy==1.23.5 einops==0.7.0
pip install leidenalg==0.10.1 umap-learn==0.5.5
pip install matplotlib==3.7.1 seaborn==0.12.2
```

The installed `deepkinet` package provides the `workflow` and `utils` entry
points that this script imports.

## Input Requirements

- `adata.layers["spliced"]` and `adata.layers["unspliced"]`.
- **Integer counts are required.** DeepKINET models the raw count-generating
  process and raises a `ValueError` for non-integer spliced/unspliced values.
  This wrapper re-reads the raw file and copies back only the filtered genes'
  raw layers, then either casts integer counts to `float32` or restores
  non-integer values to per-cell integer counts (each cell's smallest non-zero
  value is treated as one count).
- `adata.obs[cluster_key]` for the cell-type/cluster labels.
- Real data: `min_shared_counts=20`, `n_top_genes=2000`, `scv.pp.moments(n_pcs=30, n_neighbors=30, method='umap')`.
- Simulated data (`--simulate`): `adata.obsm["X_dimred"]` is copied to `X_umap`,
  `min_shared_counts=None`, a dynamic `n_top_genes` is used when the dataset has
  fewer than 2000 genes, and the cluster label is `milestone`.

## Output

```text
<output-dir>/
└── <dataset-name>/
    ├── <stem>_velo.h5ad        # DeepKINET result (native velocity keys preserved)
    ├── .deepkinet_opt.pt       # training checkpoint
    └── latent_velocity.png     # latent velocity embedding figure
```

## Parameters

- `--input`: Input H5AD file for single-file mode.
- `--metadata-file`: Metadata CSV/TSV file for batch processing.
- `--output-dir`: Root output directory. Required.
- `--dataset-name`: Dataset folder name in single-file mode. Default: input file stem.
- `--cluster-key`: Column name in `adata.obs` used for labels; defaults to `milestone` with `--simulate`. Required for real data.
- `--dimred-key`: Embedding key in `adata.obsm`. Default: `X_umap`.
- `--simulate`: Use simulated-data preprocessing. Default: `False`.
- `--n-top-genes`: Number of highly variable genes for real data. Default: `2000`.
- `--min-shared-counts`: Minimum shared counts for real-data gene filtering. Default: `20`.
- `--n-pcs`: Number of principal components for moments computation. Default: `30`.
- `--n-neighbors`: Number of neighbors for moments computation. Default: `30`.
- `--epochs`: Training epochs for each kinetics stage. Default: `2000`.
- `--overwrite`: Overwrite existing outputs. Default: `False`.
- `--seed`: Random seed. Default: `2024`.

## Usage

```bash
python DeepKINET.py --input data.h5ad --output-dir results --cluster-key celltype
python DeepKINET.py --input sim.h5ad --output-dir results --cluster-key milestone --simulate
python DeepKINET.py --metadata-file datasets.csv --output-dir results
```

## Metadata File Format

Required columns:

- `dataset_name`
- `file_path`

Optional columns:

- `cluster_key` -> defaults to empty (use `milestone` when `simulate` is true)
- `dimred_key` -> defaults to `X_umap`
- `simulate` -> defaults to `False`

### Example CSV

```csv
dataset_name,file_path,cluster_key,dimred_key,simulate
1,/data/real/pancreas.h5ad,celltype,X_umap,False
bifurcation_sim,/data/sim/bifurcation_dataset.h5ad,milestone,X_dimred,True
```
