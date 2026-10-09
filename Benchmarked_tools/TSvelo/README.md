# TSvelo Script

TSvelo infers RNA velocity by jointly modeling splicing, transcription and a
transcription-factor (TF) regulatory network with a neural ODE.

TSvelo is distributed as two scripts because the simulated-data branch is
genuinely different from the real-data branch:

- `TSvelo.py` — real data. The TF network is looked up from the ENCODE / ChEA
  TF databases.
- `TSvelo_sim.py` — simulated data. The a-priori regulatory network that
  generated the data is injected directly, so no TF-database lookup is used.

## Installation

Install the Python stack (versions mirror the project Docker image) and the
TSvelo package itself from the upstream repository:

```bash
pip install pandas==2.0.3 anndata==0.9.2 scanpy==1.9.8 numpy==1.24.4 scipy==1.10.1
pip install numba==0.58.1 matplotlib==3.7.5 scvelo==0.3.2 torch==2.4.1
pip install torchdiffeq==0.2.4 typing_extensions mygene==3.2.2 leidenalg==0.10.2 pygam==0.9.1
pip install git+https://github.com/lijc0804/TSvelo.git
```

The real-data script additionally needs the ENCODE / ChEA TF databases that ship
with the TSvelo repository (`ENCODE/processed/`, `ChEA/ChEA_2016.txt`). Point
`--tsvelo-db-dir` (or the `TSVELO_DB_DIR` environment variable) at the folder that
contains them.

## Input Requirements

Both scripts read an `.h5ad` file that contains:

- `adata.layers["spliced"]`
- `adata.layers["unspliced"]`
- a cell-label column in `adata.obs` (passed via `--cluster-key`)

Optional:

- `adata.obsm["X_umap"]` — used for plots. For simulated data, if it is missing
  but `adata.obsm["X_dimred"]` is present, the latter is copied to `X_umap`.

For `TSvelo_sim.py`, the simulated objects are expected to carry the GRN prior as
described below:

- scmultisim ("Bursting-tree") data names its regulatory genes with pure numbers
  (`"1"`, `"2"`, ...). The matching long-format GRN is supplied with
  `--grn-100-csv` / `--grn-1139-csv`.
- dyngen data names its TFs `*_TF1`. The true GRN edge list can be supplied with
  `--dyngen-feature-network-csv`; if it is absent, a GRN is rebuilt from the top
  Spearman-correlated TFs (`--dyngen-correlation-top-k`).

## Output

For an input `data.h5ad` and `--dataset-name id_x`, the output structure is:

```text
<output-dir>/
└── id_x/
    ├── data.h5ad          (final AnnData with layers['velocity'])
    ├── pp.h5ad            (preprocessed AnnData)
    ├── TSvelo.h5ad        (internal merged TSvelo result)
    ├── figures/           (velocity plots)
    ├── figures_l0/        (per-lineage training plots)
    ├── l0_pp.h5ad         (per-lineage preprocessed branches)
    └── l0_TSvelo.h5ad
```

The final `<stem>.h5ad` contains:

- `.layers["velocity"]` — the inferred velocity (NaN outside selected genes)
- `.layers["U"]`, `.layers["S"]`, `.layers["alpha"]`, `.layers["du_dt"]`, `.layers["ds_dt"]`
- `.obs["t_steps"]`, `.obs["t"]` — the inferred latent time
- `.var["beta"]`, `.var["gamma"]`, `.var["W_bias"]`, `.var["selected_genes"]`
- `.varm["W"]` — the (trained) regulatory weight matrix
- `.uns["U_t"]`, `.uns["S_t"]`, `.uns["loss_mask"]`, `.uns["tsvelo_run"]`

## Parameters

### `TSvelo.py` (real data)

- `--input`: input `.h5ad` file (single-file mode). Default: None
- `--metadata-file`: metadata CSV/TSV for batch mode. Default: None
- `--output-dir`: root output directory. Required
- `--dataset-name` / `--dataset_name`: dataset folder name. Default: input file stem
- `--cluster-key` / `--cluster_key`: column in `adata.obs` with cell labels. Default: `vis_annotation` (required on the CLI in single-file mode)
- `--save-name` / `--save_name`: suffix appended to the dataset folder name. Default: empty
- `--preprocess`: force re-preprocessing. Default: False
- `--n-jobs` / `--n_jobs`: number of parallel jobs. Default: -1
- `--n-neighbors` / `--n_neighbors`: number of neighbors for the KNN graph. Default: 30
- `--n-top-genes` / `--n_top_genes`: number of highly variable genes. Default: 2000
- `--n-selected-genes` / `--n_selected_genes`: number of selected velocity genes. Default: 100
- `--TF-databases` / `--TF_databases`: TF databases to use, e.g. `ENCODE ChEA`. Default: `ENCODE ChEA`
- `--N-steps` / `--N_steps`: number of time steps. Default: 500
- `--N-EPOCH` / `--N_EPOCH`: maximum number of EM epochs. Default: 10
- `--num-epochs` / `--num_epochs`: maximum number of neural-ODE epochs. Default: 100
- `--min-decrease` / `--min_decrease`: minimum decrease for early stopping. Default: 0
- `--n-genes2show` / `--n_genes2show`: number of genes to plot during training. Default: 0
- `--cuda`: CUDA device id, exposed through `CUDA_VISIBLE_DEVICES`. Default: None
- `--tsvelo-db-dir`: directory holding the `ENCODE/` and `ChEA/` TF databases. Default: `/opt/TSvelo`
- `--overwrite`: overwrite existing outputs. Default: False
- `--seed`: random seed. Default: 2024

### `TSvelo_sim.py` (simulated data)

- `--input`: input `.h5ad` file. Default: None
- `--metadata-file`: metadata CSV for batch mode. Default: None
- `--output-dir`: root output directory. Required
- `--dataset-name`: dataset folder name. Default: input file stem
- `--cluster-key`: column in `adata.obs` with cell labels. Default: `vis_annotation`
- `--sim-type`: `scmultisim`, `dyngen` or `auto`. Default: `auto`
- `--grn-100-csv`: scmultisim GRN csv for small networks (~100 regulators). Default: `GRN_params_100.csv`
- `--grn-1139-csv`: scmultisim GRN csv for large networks (~1100 regulators). Default: `GRN_params_1139.csv`
- `--dyngen-feature-network-csv`: dyngen model edge list (true GRN). Default: `dyngen_bifurcating_cell10000_gene500_feature_network.csv`
- `--dyngen-correlation-top-k`: top-k correlated TFs per gene when rebuilding a dyngen GRN. Default: 10
- `--n-jobs`, `--n-neighbors`, `--n-selected-genes`, `--N-steps`, `--N-EPOCH`, `--num-epochs`,
  `--min-decrease`, `--n-genes2show`, `--cuda`, `--save-pdf`, `--overwrite`, `--seed`: as above,
  with simulated-data defaults `--N-steps 500`, `--N-EPOCH 10`, `--num-epochs 100`

## Usage

### Real Data

```bash
python TSvelo.py \
    --input data.h5ad \
    --output-dir ./output \
    --cluster-key celltype
```

### Real Data, Named Dataset

```bash
python TSvelo.py \
    --input data.h5ad \
    --output-dir ./output \
    --dataset-name pancreas \
    --cluster-key celltype
```

### Simulated Data

```bash
python TSvelo_sim.py \
    --input simulated.h5ad \
    --output-dir ./output \
    --cluster-key vis_annotation
```

### Simulated Data with Explicit GRN Files

```bash
python TSvelo_sim.py \
    --input simulated.h5ad \
    --output-dir ./output \
    --grn-100-csv GRN_params_100.csv \
    --grn-1139-csv GRN_params_1139.csv \
    --dyngen-feature-network-csv dyngen_bifurcating_cell10000_gene500_feature_network.csv
```

### Batch Mode

```bash
python TSvelo.py --metadata-file datasets.csv --output-dir ./output
python TSvelo_sim.py --metadata-file sim_datasets.csv --output-dir ./output
```

The batch metadata file requires the columns `dataset_name`, `file_path` and
`cluster_key` (an optional `save_name` column is also accepted).

## Method Notes

- On real data TSvelo builds the TF network from ENCODE / ChEA and derives an
  initial regulatory matrix `W` from TF-target co-expression; `W`, the splicing
  rates `beta`/`gamma` and the latent time are then optimized jointly.
- PAGA lineages are detected automatically and one model is trained per lineage;
  the per-lineage results are merged when more than one lineage is found.
- On simulated data the GRN prior replaces the ENCODE / ChEA lookup and, for
  dyngen data without a stored model, the prior is rebuilt by correlation.
- `--cuda` only sets `CUDA_VISIBLE_DEVICES` when it is not already defined, so an
  externally pinned GPU is never overridden.
