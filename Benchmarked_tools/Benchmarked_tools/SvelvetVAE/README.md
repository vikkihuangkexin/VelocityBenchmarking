# SvelvetVAE Script

SvelvetVAE (velvetVAE) is a variational-autoencoder based RNA velocity method that learns a
latent-space neighbourhood structure together with a kinetic parameter (gamma) to recover
cell-specific velocities from spliced and unspliced counts.

## Installation

```bash
pip install "pip<24.1"
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install "jax[cuda11_local]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install chex==0.1.7 flax==0.6.11
pip install numpyro==0.12.1 optax==0.1.7 mudata==0.2.0
pip install scvi-tools==0.19.0 scanpy==1.9.8 scvelo==0.2.5 anndata==0.8.0
git clone https://github.com/rorymaizels/velvetVAE.git /opt/velvetVAE
pip install --no-deps -e /opt/velvetVAE
```

The conda environment used by the container is `svelvetvae` (Python 3.9). Install a
GPU-enabled PyTorch build that matches your own CUDA driver if you do not use the container.

## Input Requirements

The input H5AD file must contain:

- `adata.layers["spliced"]`
- `adata.layers["unspliced"]`
- `adata.obs[<cluster-key>]` — used for labels and plots

The `total` layer is built as `spliced + unspliced` when it is missing, and the three layers
are densified and cast to `float32` because velvetVAE builds float64 tensors otherwise.

Real data: `min_shared_counts=20`, `n_top_genes=2000`, cluster key of your choice, embedding in
`adata.obsm["X_umap"]`.

Simulated data (`--simulate`): the shared-count filter is relaxed
(`min_shared_counts=None`), `n_top_genes` is derived from the gene count when fewer than 2000
genes are present, `adata.obsm["X_dimred"]` is copied to `adata.obsm["X_umap"]`, and the
cluster label defaults to `milestone`.

## Output

For an input file named `<stem>.h5ad` with `--dataset-name <dataset-name>`, the output
structure is:

```text
<output-dir>/
└── <dataset-name>/
    ├── <stem>.h5ad          # result with layers['velocity']
    └── plot/
        └── SvelvetVAE_<basis>_stream.png
```

The result H5AD contains:

- `adata.layers["velocity"]` — predicted velocity
- `adata.uns["svelvetvae_run"]` — run metadata (input path, parameters, seed)

## Parameters

- `--input`: input H5AD file (single-file mode). Default: none
- `--metadata-file`: metadata CSV/TSV file (batch mode). Default: none
- `--output-dir`: output directory. Required
- `--dataset-name`: dataset folder name in single-file mode. Default: input file stem
- `--cluster-key`: column in `adata.obs` used for labels; required for real data and defaults to `milestone` with `--simulate`. Default: none
- `--dimred-key`: embedding key in `adata.obsm`; use `X_dimred` for simulated data. Default: `X_umap`
- `--simulate`: use the simulated-data preprocessing branch. Default: False
- `--n-latent`: latent space dimension. Default: 50
- `--max-epochs`: maximum number of training epochs. Default: 100
- `--freeze-vae-after-epochs`: epoch after which the VAE is frozen. Default: 20
- `--constrain-vf-after-epochs`: epoch after which the velocity field is constrained. Default: 20
- `--lr`: learning rate. Default: 0.01
- `--knn-neighbors`: neighbours for the velvetVAE neighbourhood graph. Default: 100
- `--n-top-genes`: number of highly variable genes. Default: 2000
- `--min-shared-counts`: minimum shared counts for real-data filtering. Default: 20
- `--n-pcs`: principal components used for the neighbour graph. Default: 30
- `--n-neighbors`: neighbours used for the scanpy neighbour graph. Default: 30
- `--no-linear-decoder`: disable the linear decoder. Default: False
- `--overwrite`: overwrite existing outputs. Default: False
- `--save-plots`: also write the velocity stream plot (computes `velocity_graph` as a side effect). Disabled by default because the original `code/svelvet.py` has this plotting block commented out. Default: False
- `--seed`: random seed. Default: 2024

## Usage

```bash
python SvelvetVAE.py --input data.h5ad --output-dir results --cluster-key celltype
python SvelvetVAE.py --input sim.h5ad --output-dir results --cluster-key milestone --simulate
python SvelvetVAE.py --metadata-file datasets.csv --output-dir results
```

### Metadata File Format

Required columns: `dataset_name`, `file_path`.

Optional columns: `cluster_key` (default `milestone`), `dimred_key` (default `X_umap`),
`simulate` (default `False`).

```csv
dataset_name,file_path,cluster_key,dimred_key,simulate
1,/data/real/pancreas.h5ad,celltype,X_umap,False
bifurcation_sim,/data/sim/bifurcation_dataset.h5ad,milestone,X_dimred,True
```
