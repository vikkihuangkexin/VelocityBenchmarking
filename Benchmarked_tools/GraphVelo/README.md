# GraphVelo Script

GraphVelo learns a gene-gene velocity graph on top of an scVelo dynamical velocity estimate,
using the MACk score to select the most informative genes and correcting the velocity field
through message passing over that graph.

## Installation

```bash
pip install graphvelo scvelo==0.2.5 GPUtil pynvml numpy==1.23.5 pygam
```

The conda environment used by the container is `graphvelo` (Python 3.8). GraphVelo is a
CPU method; no GPU build is required.

The extra `pygam` package is not declared by `graphvelo` but is imported by
`graphvelo/gam.py`, so `import graphvelo` fails without it.

## Input Requirements

The input H5AD file must contain:

- `adata.layers["spliced"]`
- `adata.layers["unspliced"]`
- `adata.obs[<cluster-key>]` — used for labels and plots

The wrapper runs the scVelo dynamical pipeline (`recover_dynamics`, `velocity(mode='dynamical')`,
`latent_time`, `velocity_graph`), builds `Ms`/`Mu` moments, converts the connectivities to kNN
indices with `adj_to_knn`, computes the MACk score and trains GraphVelo on the top MACk genes.

Real data: `min_shared_counts=20`, `n_top_genes=2000`, cluster key of your choice, embedding in
`adata.obsm["X_umap"]`.

Simulated data (`--simulate`): `min_shared_counts` stays at `20` (the value used by both the
real entry point and the simulated block of the original driver), `n_top_genes` is derived from
the gene count when fewer than 2000 genes are present, `adata.obsm["X_dimred"]` is copied to
`adata.obsm["X_umap"]`, and the
cluster label defaults to `milestone`.

### Multi-omics Input (`GraphVelo_multiome.py`)

`GraphVelo_multiome.py` additionally requires:

- `--rna-input` — RNA H5AD file containing `adata.layers["Mc"]` (gene activity)
- `--atac-input` — ATAC peak H5AD file

## Output

For an input file named `<stem>.h5ad` with `--dataset-name <dataset-name>`, the output
structure is:

```text
<output-dir>/
└── <dataset-name>/
    ├── <stem>.h5ad          # result with layers['velocity']
    ├── rc.h5ad              # GraphVelo_multiome.py only: alias of <stem>_graphvelo.h5ad
    └── plot/
        ├── GraphVelo_<basis>_stream.png
        ├── GraphVelo_<basis>_stream_gv.png
        └── GraphVelo_<basis>_grid_gv.png
```

The result H5AD contains:

- `adata.layers["velocity"]` — scVelo dynamical velocity
- `adata.layers["velocity_gvs"]`, `adata.layers["velocity_gvu"]` — GraphVelo velocity projected on `Ms` / `Mu`
- `adata.obsm["gv_pca"]`, `adata.obsm["gv_umap"]` — GraphVelo velocity projected on `X_pca` / `X_umap`
- `adata.uns["graphvelo_run"]` — run metadata (input path, parameters, seed)

`GraphVelo_multiome.py` writes `adata.layers["velocity_gv"]`, `adata.layers["velocity_c"]`,
`adata.obsm["gv_pca"]`, `adata.obsm["gv_tsne"]` and `adata.uns["graphvelo_multiome_run"]`.

## Parameters

- `--input`: input H5AD file (single-file mode, `GraphVelo.py`). Default: none
- `--rna-input`: input RNA H5AD file (single-file mode, `GraphVelo_multiome.py`). Default: none
- `--atac-input`: input ATAC peak H5AD file (`GraphVelo_multiome.py`). Default: none
- `--metadata-file`: metadata CSV/TSV file (batch mode). Default: none
- `--output-dir`: output directory. Required
- `--dataset-name`: dataset folder name in single-file mode. Default: input file stem
- `--cluster-key`: column in `adata.obs` used for labels; required for real data and defaults to `milestone` with `--simulate`. Default: none (RNA) / `celltype` (multi-omics)
- `--dimred-key`: embedding key in `adata.obsm`; use `X_dimred` for simulated data. Default: `X_umap`
- `--simulate`: use the simulated-data preprocessing branch. Default: False
- `--n-top-genes`: number of highly variable genes. Default: 2000
- `--min-shared-counts`: minimum shared counts for real-data filtering. Default: 20
- `--n-pcs`: principal components used for the neighbour graph. Default: 30
- `--n-neighbors`: neighbours used for the scanpy neighbour graph. Default: 30
- `--n-jobs`: parallel jobs for scVelo and the MACk score. Default: 8
- `--n-mack-genes`: number of top MACk genes used by GraphVelo. Default: 200
- `--n-lsi-components`: LSI components fitted on the ATAC peaks (multi-omics). Default: 50
- `--wnn-k`: neighbours for the weighted nearest-neighbour graph (multi-omics). Default: 50
- `--overwrite`: overwrite existing outputs. Default: False
- `--seed`: random seed. Default: 2024

## Usage

```bash
python GraphVelo.py --input data.h5ad --output-dir results --cluster-key celltype
python GraphVelo.py --input sim.h5ad --output-dir results --cluster-key milestone --simulate
python GraphVelo.py --metadata-file datasets.csv --output-dir results

python GraphVelo_multiome.py --rna-input rna.h5ad --atac-input atac.h5ad \
    --output-dir results --cluster-key celltype
python GraphVelo_multiome.py --metadata-file multiome.csv --output-dir results
```

### Metadata File Format

`GraphVelo.py` — required columns: `dataset_name`, `file_path`.
Optional columns: `cluster_key` (default `milestone`), `dimred_key` (default `X_umap`),
`simulate` (default `False`).

```csv
dataset_name,file_path,cluster_key,dimred_key,simulate
1,/data/real/pancreas.h5ad,celltype,X_umap,False
bifurcation_sim,/data/sim/bifurcation_dataset.h5ad,milestone,X_dimred,True
```

`GraphVelo_multiome.py` — required columns: `dataset_name`, `rna_path`, `atac_path`.
Optional columns: `cluster_key` (default `celltype`), `dimred_key` (default `X_umap`).

```csv
dataset_name,rna_path,atac_path,cluster_key
cortex,/data/multiome/cortex_rna.h5ad,/data/multiome/cortex_atac.h5ad,celltype
```
