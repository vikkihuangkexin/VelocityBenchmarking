# MultiVeloVAE Script

MultiVeloVAE is a probabilistic framework for RNA velocity inference from
multi-lineage, multi-omic and multi-sample single-cell data. It models gene
expression and chromatin accessibility on a shared time scale, supports multi-sample
inference from datasets with partially overlapping modalities, accounts for lineage
bifurcations, and enables statistical testing of velocity parameters across cell
types and time. Upstream repository:
[https://github.com/welch-lab/MultiVeloVAE](https://github.com/welch-lab/MultiVeloVAE).

`Benchmarked_tools/MultiVeloVAE/MultiVeloVAE.py` reimplements the host-side driver
(`MultiVeloVAE_simdata.py`) in the VelocityBenchmarking interface and follows the
upstream API documented below. Its batch entry point is
`Batch_run/multivelovae_robust.py`.

## Installation

```bash
pip install multivelovae
```

The benchmark Docker image installs the package together with the pinned scientific
stack used to generate the manuscript figures:

```bash
pip install multivelovae==0.1.0 torch==2.0.0 numpy==1.23.5 scipy==1.10.1 pandas==1.5.3 \
    scikit-learn==1.2.2 anndata==0.8.0 scanpy==1.9.3 scvelo==0.2.5 matplotlib==3.7.3 \
    seaborn==0.12.2 umap-learn==0.5.3 multivelo==0.1.2 hnswlib igraph leidenalg loess \
    tqdm ipywidgets psutil nvidia-ml-py GPUtil
```

The image environment is named `MultiVeloVAE`. Additional documentation and example
notebooks are available at
[https://multivelovae.readthedocs.io/en/latest](https://multivelovae.readthedocs.io/en/latest)
and in the upstream `paper-notebooks/` directory.

## Input Requirements

MultiVeloVAE works on a single AnnData object that carries the multi-omic layers on
a shared set of cells:

- `adata.layers["Ms"]`: spliced (RNA) expression.
- `adata.layers["Mu"]`: unspliced (RNA) expression.
- `adata.layers["Mc"]`: chromatin accessibility gene activity. This layer is read
  directly by the model and is **not** normalized internally, so it must already be
  scaled / smoothed.
- `adata.obs`: cell metadata used for the multi-sample / multi-lineage design.

Important preprocessing note (upstream limitation): `VAEChrom` reads the `Mc`, `Mu`
and `Ms` layers without normalizing them. If `Mu`/`Ms` are raw integer counts, some
genes can take a constant value across the selected cells, which makes the gene
covariance exactly zero and raises
`numpy.linalg.LinAlgError: 1-th leading minor of the array is not positive definite`
during model initialization. The upstream preprocessing pipeline avoids this by
running `sc.pp.log1p`, `sc.pp.scale` and, crucially,
`scv.pp.moments(adata, n_pcs=30, n_neighbors=30)` so that the moment layers become
kNN-smoothed continuous values. Any benchmark driver must apply this preprocessing
before constructing the model.

## Output

MultiVeloVAE stores its inference results inside the AnnData object:

- fitted velocities in `adata.layers["velocity"]` (plus the model's native
  per-gene kinetic parameters), which is the key consumed by the benchmark metrics.
- plotting and analysis helpers exposed through `multivelovae.plotting` /
  `multivelovae.plotting_chrom`.

The driver writes the native MultiVeloVAE file and the benchmark alias into
`<output-dir>/<dataset-name>/`:

- `<dataset-name>_MultiVeloVAE.h5ad` — the AnnData written by `save_anndata`, with the
  fitted velocity copied to `layers['velocity']` (the native `layers[f'{key}_velocity']`
  is kept as well).
- `rc.h5ad` — byte-identical copy of that file, for the `Batch_run` contract.
- `model/` — saved encoder/decoder weights; `figures/` — training curves.

## Parameters

MultiVeloVAE is driven upstream through the Python API (the model class is `VAEChrom`,
exposed as `multivelovae.VAEChrom`). The upstream training / inference parameters
(with their reference values) are:

- `seed`: random seed for reproducibility. Reference: `0`
- `device`: `cuda` when a GPU is available, otherwise `cpu`
- `n_epochs` / `max_epoch`: number of training epochs. Reference: method default used
  in the upstream `paper-notebooks/` examples
- `batch_size`: mini-batch size for model training
- `lr`: optimizer learning rate
- `t`: shared latent time scale used to couple expression and accessibility
- `Ms` / `Mu` / `Mc` layers: input layers listed in the Input Requirements section
- `n_samples` (evaluation): number of cells / samples used for multi-sample inference

### Benchmark driver (`MultiVeloVAE.py`)

The driver reimplements the host-side `MultiVeloVAE_simdata.py` driver and uses the
same defaults as that script:

- `--input`: input H5AD file. Required in single-file mode.
- `--metadata-file`: metadata CSV/TSV file for batch processing. Mutually exclusive with
  `--input`. Default: None
- `--output-dir`: root output directory. Required
- `--dataset-name`: dataset folder name in single-file mode. Default: input file stem
- `--cluster-key`: `obs` column used for labels and plots. Default: `leiden` (falls
  back to the first available label column when `leiden` is absent)
- `--embed`: embedding used for plots, stored as `X_<embed>`. Default: `tsne` (falls
  back to an available embedding when `X_tsne` is absent)
- `--dimred-key`: fallback embedding key in `adata.obsm`. Default: `X_umap`
- `--atac-layer`: `adata.layers` key holding the chromatin activity matrix. Default: `Mc`
- `--batch-size`: model mini-batch size. Default: `32`
- `--n-top-genes`: number of highly variable genes kept by `filter_and_normalize`.
  Default: `2000`
- `--device`: torch device. Default: `cuda:0`
- `--key`: model key used for the output layers, e.g. `vae_velocity`. Default: `vae`
- `--simulate`: simulated-data branch; uses `obsm['X_dimred']`, relaxes the filter and
  sets the cluster label to `milestone`. Default: `False`
- `--min-shared-counts`: `min_shared_counts` passed to `filter_and_normalize`.
  Default: unset (`None`, as in the original)
- `--overwrite`: overwrite existing outputs. Default: `False`
- `--seed`: random seed for numpy and torch. Default: `2022`

The driver runs `filter_and_normalize` + PCA + `sc.pp.neighbors(..., method="umap")`
+ `scv.pp.moments` on the input, mirroring the sibling simulation scripts.

## Usage

The upstream package exposes the model through a single import:

```python
import multivelovae as vv

# VAEChrom is the multi-omic / multi-sample velocity model.
model = vv.VAEChrom(...)
```

The complete end-to-end workflow (preprocessing, training and plotting) is provided
by the upstream notebooks in
[`paper-notebooks/`](https://github.com/welch-lab/MultiVeloVAE/tree/main/paper-notebooks),
with processed AnnData objects shared on
[figshare](https://figshare.com/articles/dataset/Post-processed_anndata_objects_for_MultiVeloVAE/30280333).

### Batch entry point

The single-file CLI is `python MultiVeloVAE.py --input ... --output-dir ... --cluster-key ...`,
and the corresponding batch driver is:

```bash
python Batch_run/multivelovae_robust.py
```

It runs each dataset five times with a distinct seed, writing `<output-dir>/<id>_r<run>/`
with `pp.h5ad` and `rc.h5ad`.
