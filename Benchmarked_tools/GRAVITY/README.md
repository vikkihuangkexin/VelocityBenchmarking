# GRAVITY Script

GRAVITY predicts RNA velocity and regulatory rewiring by dynamic
regulatory-mechanism-enhanced deep learning. It consumes a cellDancer-style
long-format count table and runs a two-stage optimization (cell-wise trajectory
recovery, then gene-wise kinetic refinement). This wrapper runs one dataset or a
metadata table and writes a `<stem>.h5ad` result file.

## Installation

```bash
pip install --index-url https://download.pytorch.org/whl/cu117 "torch==2.0.1+cu117"
git clone https://github.com/CSUBioGroup/GRAVITY.git
cd GRAVITY && pip install -e .
pip install "scvelo==0.2.5" GPUtil pynvml psutil
pip install celldancer
```

`celldancer` is required by `gravity_result_to_h5ad.py`, which converts the
result CSV into an AnnData object.

## Included scripts

| Script | Purpose |
|--------|---------|
| `GRAVITY.py` | Real + simulated main pipeline, switched by `--simulate`. |
| `gravity_preprocess.py` | CPU-only preprocessing that writes `combine.csv`. **Only used for simulated data inside the Docker image.** |
| `gravity_sim_from_combine.py` | Deep-learning step that consumes `combine.csv`. **Only used for simulated data inside the Docker image.** |
| `gravity_result_to_h5ad.py` | Result CSV to H5AD converter. |

### Helper-script parameters

`gravity_result_to_h5ad.py`:

- `--input`: GRAVITY result CSV. Required.
- `--output-dir`: output directory. Required.
- `--dataset-name`: output file stem. Default: input file stem.
- `--overwrite`: overwrite existing outputs. Default: `False`.

`gravity_preprocess.py`:

- `--input`: sample-list CSV with columns `ID` and `path`. Required.
- `--output-dir`: result save root directory. Required.
- `--start` / `--end`: row index range of the sample list (inclusive / exclusive). Default: None
- `--n-jobs`: number of parallel workers (`0` = all CPU cores, `1` = serial, `>1` = explicit count). Default: `0`
- `--force`: ignore existing outputs and re-export. Default: `False`
- `--simulate`: confirm simulated-data processing; required because this script is simulated-data only. Default: `False`
- `--seed`: random seed. Default: `2024`

`gravity_sim_from_combine.py`:

- `--input`: sample-list CSV with columns `ID` and `path`. Required.
- `--output-dir`: result and log root directory. Required.
- `--start` / `--end`: row index range of the sample list (inclusive / exclusive). Default: None
- `--simulate`: confirm simulated-data processing; required because this script is simulated-data only. Default: `False`
- `--from-combine`: start training directly from `<output-dir>/<ID>/combine.csv`. Default: `False`
- `--force-rerun`: ignore the "already finished" check and rerun every sample. Default: `False`
- `--stage1-epochs` / `--stage2-epochs`: cell-wise / gene-wise stage epochs. Default: `6` / `6`
- `--stage1-lr` / `--stage2-lr`: cell-wise / gene-wise stage learning rates. Default: `1e-6` / `1e-4`
- `--batch-size`: mini-batch size. Default: `16`
- `--num-workers`: data loader workers. Default: `8`
- `--seed`: random seed. Default: `2024`

## Input Requirements

- `adata.layers["spliced"]` and `adata.layers["unspliced"]`.
- `adata.obs[cluster_key]` for the cell-type labels.
- Real data: `scv.pp.filter_and_normalize(min_shared_counts=20, n_top_genes=2000)`
  and moments are performed by GRAVITY's `export_intermediate_from_h5ad`, using
  the embedding key given by `--dimred-key`. The species is detected from the
  file name (`Mm` -> mouse, `Hs` -> human) to select the matching
  `prior_data/nichenet_<species>.zip` network shipped with GRAVITY; override with
  `--prior-network`.
- Simulated data (`--simulate`): `adata.obsm["X_dimred"]` is used as the 2D
  embedding, `min_shared_counts=None`, and a dynamic gene count is used for
  smaller datasets. No prior network is used. Recommended cluster key `milestone`.

## Output

```text
<output-dir>/
└── <dataset-name>/
    ├── <stem>.h5ad            # main result, contains layers['velocity']
    ├── cell_type_u_s.csv      # long-format count table (intermediate)
    ├── combine.csv            # wide training table (intermediate)
    ├── stage1.csv / stage2.csv
    ├── stage1.ckpt / stage2.ckpt
    └── gravity_result.csv     # per-cell velocity result CSV
```

## Parameters

- `--input`: Input H5AD file for single-file mode.
- `--metadata-file`: Metadata CSV/TSV file for batch processing.
- `--output-dir`: Root output directory. Required.
- `--dataset-name`: Dataset folder name in single-file mode. Default: input file stem.
- `--cluster-key`: Column name in `adata.obs` used as cell-type labels; defaults to `milestone` with `--simulate`. Required for real data.
- `--dimred-key`: Embedding key in `adata.obsm`. Default: `X_umap`.
- `--simulate`: Use simulated-data preprocessing. Default: `False`.
- `--preprocessed`: Treat the input as already filtered/normalized with moment layers (for example a `pp.h5ad`), so it is exported without re-normalizing. Default: `False`.
- `--prior-network`: Path to the prior TF-target network archive; detected from the file name for real data. Default: `None`.
- `--n-top-genes`: Number of highly variable genes. Default: `2000`.
- `--min-shared-counts`: Minimum shared counts for real-data gene filtering. Default: `20`.
- `--n-pcs`: Number of principal components. Default: `30`.
- `--n-neighbors`: Number of neighbors. Default: `30`.
- `--stage1-epochs`: Cell-wise stage epochs. Default: `6`.
- `--stage2-epochs`: Gene-wise stage epochs. Default: `6`.
- `--stage1-lr`: Cell-wise stage learning rate. Default: `1e-6`.
- `--stage2-lr`: Gene-wise stage learning rate. Default: `1e-4`.
- `--batch-size`: Mini-batch size. Default: `128`.
- `--num-workers`: Data loader workers. Default: `8`.
- `--overwrite`: Overwrite existing outputs. Default: `False`.
- `--seed`: Random seed. Default: `2024`.

## Usage

```bash
python GRAVITY.py --input data.h5ad --output-dir results --cluster-key celltype
python GRAVITY.py --input sim.h5ad --output-dir results --cluster-key milestone --simulate
python GRAVITY.py --metadata-file datasets.csv --output-dir results
python gravity_result_to_h5ad.py --input gravity_result.csv --output-dir results
python gravity_preprocess.py --input samples.csv --output-dir results --simulate
python gravity_sim_from_combine.py --input samples.csv --output-dir results --simulate
```

## Metadata File Format

Required columns:

- `dataset_name`
- `file_path`

Optional columns:

- `cluster_key` -> defaults to empty (use `milestone` when `simulate` is true)
- `dimred_key` -> defaults to `X_umap`
- `simulate` -> defaults to `False`
- `prior_network` -> defaults to empty (auto-detected from the file name)

### Example CSV

```csv
dataset_name,file_path,cluster_key,dimred_key,simulate,prior_network
1,/data/real/pancreas_Mm.h5ad,celltype,X_umap,False,
bifurcation_sim,/data/sim/bifurcation.h5ad,milestone,X_dimred,True,
```
