# DynaVelo Script

DynaVelo is a latent neural ODE model that learns multiomic velocities of single
cells from paired RNA expression and TF motif accessibility, infers dynamic and
cell-state-specific gene regulatory networks, and performs in-silico perturbations
to predict velocity changes. Upstream repository:
[https://github.com/karbalayghareh/DynaVelo](https://github.com/karbalayghareh/DynaVelo).

`Benchmarked_tools/DynaVelo/DynaVelo.py` reimplements the host-side driver
(`DynaVelo_simdata_multigpu.py`) in the VelocityBenchmarking interface and follows the
upstream API documented below. Its batch entry point is `Batch_run/dynavelo_robust.py`.

## Installation

The upstream project installs from a conda environment file:

```bash
git clone https://github.com/karbalayghareh/DynaVelo.git
cd DynaVelo
conda env create -f environment.yml
conda activate dynavelo
```

The benchmark Docker image rebuilds the equivalent environment with pip (the
upstream `environment.yml` locks exact conda builds that are no longer available,
so the same version numbers are used instead):

```bash
pip install torch==1.12.1 numpy==1.22.4 scipy==1.10.1 pandas==1.5.3 scikit-learn==1.0.2 \
    matplotlib==3.7.1 h5py==3.7.0 llvmlite==0.36.0 numba==0.53.1 python-igraph==0.10.4 \
    leidenalg==0.9.1 umap-learn==0.5.4 anndata==0.10.2 scanpy==1.9.5 scvelo==0.2.4 \
    torchdiffeq==0.2.3 torchsde==0.2.5 torchmetrics==0.11.1 functorch==0.2.1 multivelo==0.1.2
```

The image environment is named `dynavelo` and exposes the package through
`PYTHONPATH=/opt/DynaVelo`.

## Input Requirements

DynaVelo is a multi-omic method and requires **two** AnnData objects:

- `adata_rna`: shape `[n_cells, n_genes]`. Contains preprocessed (log-normalized)
  RNA expression values and RNA velocity estimates computed with scVelo in
  `adata_rna.layers["velocity"]` / `adata_rna.layers["spliced"]` /
  `adata_rna.layers["unspliced"]`. The overall trajectory should be checked for
  biological plausibility before training, because scVelo velocities are sensitive
  to the selected gene set.
- `adata_atac`: shape `[n_cells, n_tfs]`. Contains TF motif accessibility z-scores
  from chromVAR. Cells must be aligned one-to-one with `adata_rna`.

Both objects must share the same cell ordering. Because two matrices are needed, a
benchmark driver for this method also needs a paired-input convention (for example
an RNA/ATAC key pair inside a single multiome H5AD) that is not defined by the
upstream code.

The RNA input must **already carry the `Ms`/`Mu` moment layers**: the original
`DynaVelo_simdata_multigpu.py` consumes them as-is and only calls
`scv.tl.recover_dynamics` / `scv.pp.neighbors` / `scv.tl.velocity` on top of them. The
wrapper does the same and recomputes the moments only when `--preprocess-input` is set.

## Output

The trained model writes predictions back into AnnData objects:

- `adata_rna_pred` and `adata_atac_pred`, produced by `model.evaluate(...)` with
  predicted latent times, RNA (and motif) velocities, plus their per-cell variance
  when sampling mode is used.
- Optional Jacobian matrices describing the gene regulatory network. Four tensors
  are produced by `model.calculate_jacobians(...)`:
  `J_vx_x [n_cells, n_genes, n_genes]`, `J_vy_x [n_cells, n_tfs, n_genes]`,
  `J_vx_y [n_cells, n_genes, n_tfs]`, and `J_vy_y [n_cells, n_tfs, n_tfs]`.
  Because these are dense 3D tensors, Jacobians are computed only for a subset of
  genes of interest (all TFs are always included in that subset).
- Optional in-silico perturbation results from
  `model.predict_perturbation(...)`, which reports the velocity changes caused by
  setting the perturbed genes to their observed minimum (TFs also have their motif
  accessibility set to the minimum).

The driver writes the native DynaVelo files and the benchmark alias into
`<output-dir>/<dataset-name>/`:

- `<dataset-name>_adata_rna_pred_DynaVelo.h5ad` — predicted RNA AnnData with
  `layers['velocity']` (copied from the native `vx_pred_mean`; the scVelo input
  velocity that DynaVelo consumed is kept as `layers['velocity_scvelo_input']`).
- `<dataset-name>_adata_atac_pred_DynaVelo.h5ad` — predicted TF motif AnnData.
- `rc.h5ad` — byte-identical copy of the RNA prediction, for the `Batch_run` contract.
- `checkpoints/` — model weights and per-epoch training logs.

## Parameters

DynaVelo is used upstream through a Python API rather than a command-line interface.
The upstream model / training parameters (with their upstream defaults) are:

- `seed`: random seed for numpy and torch. Default: `0`
- `gpu`: CUDA device index. Default: `0`
- `batch_size`: DataLoader batch size. Default: `128`
- `lr`: Adam learning rate. Default: `1e-3`
- `max_epoch`: number of training epochs passed to `model.fit(...)`. Default: `200`
- test fraction: share of cells held out for the test dataset. Default: `0.1`
- `mode`: model mode used before evaluation; `evaluation-sample` samples the latent
  posterior, `evaluation-fixed` uses the mean latent time and initial point.
  Default: `evaluation-fixed`
- `n_samples`: number of posterior samples used by `evaluation-sample`. Default: `20`
- `epsilon`: finite-difference step used by `model.calculate_jacobians(...)`.
  Default: `1e-4`
- `x_dim` / `y_dim`: input dimensions of the model, set to
  `adata_rna.shape[1]` / `adata_atac.shape[1]`

### Benchmark driver (`DynaVelo.py`)

The driver keeps the original per-dataset pipeline (scVelo prior, then DynaVelo
training/evaluation) and exposes the following command-line parameters:

- `--rna-input` (alias `--input`): RNA AnnData file. Required in single-file mode.
- `--atac-input`: ATAC AnnData file carrying `obsm['chromVAR']`. Optional; when omitted
  the matrix is read from `obsm['chromVAR']` of the RNA file. Default: None
- `--metadata-file`: metadata CSV/TSV file for batch processing. Mutually exclusive with
  `--rna-input`. Default: None
- `--output-dir`: root output directory. Required
- `--dataset-name`: dataset folder name in single-file mode. Default: input file stem
- `--cluster-key`: column in `adata.obs` used for the cell-type weights; use `milestone`
  for simulated data. Default: `celltype`
- `--batch-size`: mini-batch size. Default: 32
- `--max-epoch`: number of training epochs. Default: 200
- `--n-samples`: posterior samples drawn at evaluation. Default: 20
- `--learning-rate`: Adam learning rate. Default: `1e-3`
- `--test-fraction`: fraction of cells held out for testing. Default: 0.1
- `--preprocess-input`: opt-in flag that runs `filter_and_normalize` + PCA + `moments`
  before the scVelo prior. It is disabled by default because the original input already
  carries the `Ms`/`Mu` moment layers; enabling it recomputes them from raw counts and
  changes what the model sees. Default: `False`
- `--n-top-genes`: only used with `--preprocess-input` (number of highly variable genes).
  Default: `2000`
- `--min-shared-counts`: only used with `--preprocess-input` (minimum shared counts).
  Default: `20`
- `--device`: Torch device, for example `cuda:0` or `cpu`; auto-selected when unset.
  Default: None
- `--simulate`: simulated-data branch; labels the cells `milestone` (no `obsm` remap and
  no extra filtering, matching the original driver). Default: `False`
- `--overwrite`: overwrite existing outputs. Default: `False`
- `--seed`: random seed. Default: 2024

## Usage

The upstream training and prediction workflow is:

```python
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dynavelo.models import MultiomeDataset
from dynavelo.models import DynaVelo

# set seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# set gpu
gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

# datasets and dataloaders
dataset = MultiomeDataset(adata_rna, adata_atac)
N_test = int(0.1 * len(dataset))
idx_random = np.random.permutation(len(dataset))
idx_test = idx_random[:N_test]
idx_train = idx_random[N_test:]
dataset_train = MultiomeDataset(adata_rna[idx_train], adata_atac[idx_train])
dataset_test = MultiomeDataset(adata_rna[idx_test], adata_atac[idx_test])

dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=0)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

# model
model = DynaVelo(x_dim=adata_rna.shape[1], y_dim=adata_atac.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# train
model.fit(dataloader_train, dataloader_test, optimizer, max_epoch=200)

# predict velocities and latent times
model.mode = 'evaluation-sample'
adata_rna_pred, adata_atac_pred = model.evaluate(adata_rna, adata_atac, dataloader, n_samples=20)
```

### Batch entry point

The single-file CLI is `python DynaVelo.py --rna-input ... --output-dir ... --cluster-key ...`,
and the corresponding batch driver is:

```bash
python Batch_run/dynavelo_robust.py
```

It runs each dataset five times with a distinct seed, writing `<output-dir>/<id>_r<run>/
with `pp.h5ad` and `rc.h5ad`.
