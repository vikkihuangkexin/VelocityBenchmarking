# RegVelo Script

RegVelo is a regulatory-network-aware RNA velocity method: it extends veloVI with
a gene regulatory network (GRN) prior so that transcription rates are explained by
transcription-factor activity.

RegVelo needs a prior GRN, and that prior is obtained differently for each data
type, so the method ships two entry scripts together with the shared helpers,
data files and pipelines that feed them:

- `RegVelo.py` — real data. The prior GRN is computed with pySCENIC
  (grn -> ctx -> aucell) and consumed as a 0/1 regulator-target matrix.
- `RegVelo_sim.py` — simulated data. The scmultisim GRN is taken from the shipped
  csv priors and the dyngen GRN is rebuilt by Spearman correlation; the helpers
  live in `grn_utils.py`.
- `h5ad_to_loom.py` / `run_pyscenic_grn_regvelo.py` — the real-data
  preprocessing steps (loom conversion and GRN parquet generation).
- `pipeline.sh` / `pipeline_sim.sh` — end-to-end real / simulated pipelines.

> **pySCENIC is required only for real data.** The simulated-data workflow
> (`RegVelo_sim.py` / `pipeline_sim.sh`) never calls pySCENIC. The Docker image in
> this project is used to train the simulated-data workflow, so its pySCENIC
> installation is **not validated** and is left commented out in the Dockerfile
> (see `Dockerfile/RegVelo/Dockerfile`).

## Installation

Install the RegVelo environment (Python 3.10) and the auxiliary packages used to
build the GRN prior:

```bash
pip install regvelo scanpy scvelo GPUtil pynvml psutil
pip install pandas numpy scipy scikit-learn loompy pyarrow
```

A GPU-enabled PyTorch build is installed automatically as a dependency of
`regvelo`.

**Only the real-data workflow needs pySCENIC** (for the `grn` / `ctx` / `aucell`
steps):

```bash
pip install pyscenic
```

The simulated-data workflow (the one this project's Docker image is built and
validated for) does not require pySCENIC at all. In `Dockerfile/RegVelo/Dockerfile`
the pySCENIC installation is therefore written but **commented out** — uncomment
it if you need the real-data pipeline inside the container. pySCENIC may also live
in its own environment; `pipeline.sh` selects it through `${PYSCENIC_ENV}` (which
defaults to `${CONDA_ENV}`).

## Included Scripts

| Script | Purpose |
|--------|---------|
| `RegVelo.py` | Real-data entry point (pySCENIC GRN prior). |
| `RegVelo_sim.py` | Simulated-data entry point (a-priori GRN prior). |
| `grn_utils.py` | GRN helpers used by the simulated entry point. |
| `h5ad_to_loom.py` | Real-data preprocessing: convert an h5ad into the pySCENIC loom. |
| `run_pyscenic_grn_regvelo.py` | Real-data preprocessing: build the GRN parquet and run `RegVelo.py`. |
| `pipeline.sh` | Real-data pipeline: loom -> pySCENIC -> GRN parquet -> RegVelo. Requires pySCENIC. |
| `pipeline_sim.sh` | Simulated-data pipeline: prior GRN -> RegVelo. Does not use pySCENIC. |
| `GRN_params_100.csv` | scmultisim prior GRN for small networks (~100 regulators). |
| `GRN_params_1139.csv` | scmultisim prior GRN for large networks (~1100 regulators). |
| `dyngen_bifurcating_cell10000_gene500_feature_network.csv` | Reference dyngen GRN edge list (not read by default; see below). |

All scripts resolve the bundled GRN csv files relative to their own directory, so
the pipeline can be launched from any working directory.

## Input Requirements

Both scripts read an `.h5ad` file that contains:

- `adata.layers["spliced"]` and `adata.layers["unspliced"]` (turned into `Ms`/`Mu`)
- a cell-label column in `adata.obs` (passed via `--cluster-key`)

`RegVelo.py` additionally needs a GRN prior:

- `--grn-parquet`: a pre-computed pySCENIC GRN parquet, i.e. a square 0/1
  `genes x genes` matrix with rows/columns as gene names, such as
  `regulon_mat_processed_all_regulons.parquet`; or
- `--pyscenic-loom`: the pySCENIC AUCell output loom (`pyscenic_output.loom`),
  from which the parquet is generated automatically.

`RegVelo_sim.py` needs the simulated GRN:

- scmultisim datasets: `--grn-100-csv` / `--grn-1139-csv`, long-format GRN csv
  files (columns `regulated.gene`, `regulator.gene`, `regulator.effect`) for small
  (~100 regulators) and large (~1100 regulators) networks.
- dyngen datasets: no prior file is used. The GRN is **always rebuilt from
  Spearman correlation** between the known TFs (`*_TF1` gene names) and every
  gene, keeping the `--dyngen-correlation-top-k` most correlated TFs per gene.

If a reliable simulated GRN is available for a dyngen dataset — for example the
`model$feature_network` edge list dumped from the dyngen model, as shipped in
`dyngen_bifurcating_cell10000_gene500_feature_network.csv` — the true edges can be
used instead of the correlation-based GRN. That code path is kept as a
**commented reference** in `RegVelo_sim.load_prior_grn()` and `grn_utils.py`; see
Method Notes for the details.

`pipeline.sh` additionally needs the pySCENIC resources, which are read from
`${AUX_DIR}` (default `/data/simdata/regvelo_aux`). This applies to real data
only — the simulated-data pipeline does not use `AUX_DIR`:

- a TF list: `allTFs_mm.txt` (mouse) or `allTFs_hg38.txt` (human);
- the ranking databases `<db_prefix>*.feather` (`mm10` for mouse, `hg38` for
  human), i.e. the `*.genes_vs_motifs.rankings.feather` files;
- the motif annotation table: `motifs-v9-nr.mgi-m0.001-o0.0.tbl` (mouse) or
  `motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl` (human).

The species is detected from the input file name (`Mm` -> mouse, `Hs` -> human)
and recorded in `species.txt` by `h5ad_to_loom.py`.

## Output

For an input `data.h5ad` and `--dataset-name id_x`, the output structure is:

```text
<output-dir>/
└── id_x/
    ├── data.h5ad          (final AnnData with layers['velocity'])
    ├── pp.h5ad            (preprocessed AnnData with the injected GRN)
    ├── regvelo_model/     (saved RegVelo model)
    └── regvelo_issue_samples.txt   (only when a gene-filter fallback is used)
```

When the real-data pipeline (`pipeline.sh`) is used, the same
`<output-dir>/<id_x>/` folder also holds the preprocessing artefacts: `adata.loom`,
`species.txt`, `adj.csv`, `reg.csv`, `pyscenic_output.loom`,
`regulon_mat_all_regulons.csv` and `regulon_mat_processed_all_regulons.parquet`.

The final `<stem>.h5ad` contains:

- `.layers["velocity"]` — the inferred velocity
- `.layers["Ms"]`, `.layers["Mu"]` — the RegVelo inputs
- `.var["velocity_genes"]`, `.var["TF"]` — the selected velocity genes and regulators
- `.uns["skeleton"]`, `.uns["regulators"]`, `.uns["targets"]` — the prior GRN
- `.uns["regvelo_run"]` — run metadata

## Parameters

### `RegVelo.py` (real data)

- `--input`: input `.h5ad` file (single-file mode). Default: None
- `--metadata-file`: metadata CSV/TSV for batch mode. Default: None
- `--output-dir`: root output directory. Required
- `--dataset-name`: dataset folder name. Default: input file stem
- `--cluster-key`: column in `adata.obs` with cell labels. Required in single-file mode
- `--grn-parquet`: pre-computed pySCENIC GRN parquet. Default: None
- `--pyscenic-loom`: pySCENIC AUCell loom used to build the parquet. Default: None
- `--n-top-genes`: number of highly variable genes. Default: 2000
- `--max-epochs`: maximum number of training epochs. Default: 150
- `--batch-size`: mini-batch size (None = all cells). Default: None
- `--train-size`: fraction of cells used for training. Default: 0.8
- `--soft-constraint` / `--no-soft-constraint`: enable/disable the soft constraint. Default: enabled
- `--lam`: first regularization weight. Default: 1.0
- `--lam2`: second regularization weight. Default: 0.0
- `--no-save-model`: do not save the trained model. Default: save the model
- `--cuda`: CUDA device id; when omitted the freest GPU is auto-selected. Default: None
- `--overwrite`: overwrite existing outputs. Default: False
- `--seed`: random seed. Default: 2024

### `RegVelo_sim.py` (simulated data)

- `--input`: input `.h5ad` file. Default: None
- `--metadata-file`: metadata CSV for batch mode. Default: None
- `--output-dir`: root output directory. Required
- `--dataset-name`: dataset folder name. Default: input file stem
- `--cluster-key`: column in `adata.obs` with cell labels. Default: `vis_annotation`
- `--sim-type`: `scmultisim`, `dyngen` or `auto`. Default: `auto`
- `--grn-100-csv`: scmultisim GRN csv for small networks (~100 regulators). Default: `GRN_params_100.csv`
- `--grn-1139-csv`: scmultisim GRN csv for large networks (~1100 regulators). Default: `GRN_params_1139.csv`
- `--dyngen-correlation-top-k`: top-k correlated TFs kept per gene when rebuilding a dyngen GRN by Spearman correlation. Default: 10
- `--max-epochs`: maximum number of training epochs. Default: 500
- `--batch-size`, `--train-size`, `--soft-constraint`, `--no-soft-constraint`, `--lam`, `--lam2`,
  `--no-save-model`, `--cuda`, `--overwrite`, `--seed`: as above

### `h5ad_to_loom.py` (real-data preprocessing)

- `--input`: input `.h5ad` file. Required
- `--output-dir`: directory for `adata.loom` and `species.txt`. Required
- `--dataset-name`: file name used for species detection. Default: the input file name

### `run_pyscenic_grn_regvelo.py` (real-data preprocessing)

- `--input`: input `.h5ad` file. Required
- `--output-dir`: root output directory. Required
- `--dataset-name`: dataset folder name. Default: the input file stem
- `--cluster-key`: column in `adata.obs` with cell labels. Required
- `--pyscenic-loom`: pySCENIC AUCell loom used to build the GRN parquet. Default: None
- `--grn-parquet`: pre-computed GRN parquet; skips parquet generation. Default: None
- `--n-top-genes`, `--max-epochs`, `--batch-size`, `--cuda`, `--overwrite`, `--seed`: as above

### Pipeline environment variables

`pipeline.sh` and `pipeline_sim.sh` take every path from the environment, so the
same scripts run unchanged inside the Docker image (defaults point at the
`/data/simdata` mount) and on a host (override `DATA_ROOT` / `AUX_DIR`):

- `CONDA_ENV`: conda environment for the Python steps. Default: `regvelo`
- `PYSCENIC_ENV`: conda environment for `pyscenic` (`pipeline.sh` only). Default: `${CONDA_ENV}`
- `CONDA_SH`: conda activation script. Default: `/opt/conda/etc/profile.d/conda.sh`
- `DATA_ROOT`: mounted data root. Default: `/data/simdata`
- `INPUT_CSV`: sample-list CSV with columns `ID` and `path`. Default: `${DATA_ROOT}/inputdata/base_realinput.csv` (real) / `base_siminput.csv` (simulated)
- `SAVE_ROOT`: result root. Default: `${DATA_ROOT}/regvelo` (real) / `${DATA_ROOT}/regvelo_sim` (simulated)
- `AUX_DIR`: pySCENIC resources (`pipeline.sh` only). Default: `${DATA_ROOT}/regvelo_aux`
- `N_WORKERS`: pySCENIC worker count (`pipeline.sh` only). Default: `40`
- `CLUSTER_KEY`: `adata.obs` cluster column. Default: `vis_annotation`
- `MAX_EPOCHS`, `BATCH_THRESHOLD`, `BATCH_SIZE`, `SKIP_KEYWORDS`: training controls (`pipeline_sim.sh` only)

Both pipelines accept optional positional arguments `[start_index] [end_index]`
(`pipeline.sh` also accepts a third `run_ids` list) and skip samples whose final
`<stem>.h5ad` already exists.

The pySCENIC-related variables (`PYSCENIC_ENV`, `AUX_DIR`, `N_WORKERS`) are used
by `pipeline.sh` only; the simulated-data pipeline never invokes pySCENIC.

## Usage

### Real Data — Step 1: preprocessing (loom + pySCENIC GRN prior)

```bash
# 1a. h5ad -> loom
python h5ad_to_loom.py --input data.h5ad --output-dir output/id_x --dataset-name 42_Mm_embryos.h5ad

# 1b. pySCENIC grn -> ctx -> aucell
AUX_DIR=/data/simdata/regvelo_aux
pyscenic grn output/id_x/adata.loom ${AUX_DIR}/allTFs_mm.txt -o output/id_x/adj.csv --num_workers 40
pyscenic ctx output/id_x/adj.csv ${AUX_DIR}/mm10*.feather \
    --annotations_fname ${AUX_DIR}/motifs-v9-nr.mgi-m0.001-o0.0.tbl \
    --expression_mtx_fname output/id_x/adata.loom \
    --output output/id_x/reg.csv --all_modules --num_workers 40
pyscenic aucell output/id_x/adata.loom output/id_x/reg.csv \
    --output output/id_x/pyscenic_output.loom --num_workers 40
```

### Real Data — Step 2: run RegVelo

```bash
# from a pre-computed GRN parquet
python RegVelo.py \
    --input data.h5ad \
    --output-dir ./output \
    --dataset-name id_x \
    --cluster-key celltype \
    --grn-parquet output/id_x/regulon_mat_processed_all_regulons.parquet

# or generate the parquet from the pySCENIC AUCell loom
python RegVelo.py \
    --input data.h5ad \
    --output-dir ./output \
    --dataset-name id_x \
    --cluster-key celltype \
    --pyscenic-loom output/id_x/pyscenic_output.loom
```

### Simulated Data

```bash
python RegVelo_sim.py \
    --input simulated.h5ad \
    --output-dir ./output \
    --cluster-key vis_annotation \
    --grn-100-csv GRN_params_100.csv \
    --grn-1139-csv GRN_params_1139.csv
```

For a scmultisim dataset the matching bundled csv is used; for a dyngen dataset
no prior file is read and the GRN is rebuilt by Spearman correlation
(`--dyngen-correlation-top-k`). pySCENIC is not involved in either case.

Batch, large-cell-count runs (mini-batch training):

```bash
python RegVelo_sim.py \
    --input simulated_cell50000.h5ad \
    --output-dir ./output \
    --cluster-key vis_annotation \
    --max-epochs 150 \
    --batch-size 1000
```

### End-to-End Pipelines

```bash
# real data (loom -> pySCENIC -> GRN parquet -> RegVelo); needs pySCENIC
DATA_ROOT=/data/simdata AUX_DIR=/data/simdata/regvelo_aux bash pipeline.sh 0 10

# simulated data (prior GRN -> RegVelo); no pySCENIC
DATA_ROOT=/data/simdata bash pipeline_sim.sh
```

### Batch Mode

```bash
python RegVelo.py --metadata-file datasets.csv --output-dir ./output
python RegVelo_sim.py --metadata-file sim_datasets.csv --output-dir ./output
```

The batch metadata file requires the columns `dataset_name` and `file_path`; an
optional `cluster_key` column (and, for real data, `grn_parquet` /
`pyscenic_loom`) overrides the command-line values.

## Method Notes

- RegVelo's prior GRN is a binary adjacency matrix. Both scripts inject it with
  `rgv.pp.set_prior_grn` (transposed to rows = targets, columns = regulators), so
  real and simulated data share the same downstream model.
- Velocity-informative genes plus all regulators are kept before training; the
  `set_prior_grn` skeleton is subset accordingly.
- If `rgv.pp.filter_genes` removes every gene (which can happen when the prior GRN
  is very sparse), the script falls back to the pre-filter gene set with min-max
  scaled layers and records the event in `regvelo_issue_samples.txt`, so the
  dataset still completes.
- `--cuda` only sets `CUDA_VISIBLE_DEVICES` when it is not already defined; when
  neither is given, the GPU with the most free memory is selected.
- Every path in the shipped scripts is a CLI argument, an environment variable or
  is resolved relative to the script directory; no host-specific path is
  hard-coded, and `/data/simdata` is the only fixed container path.
- The simulated entry point reads only `GRN_params_100.csv` / `GRN_params_1139.csv`
  (scmultisim) by default. For dyngen datasets it uses no file and always rebuilds
  the GRN from Spearman correlation.
- **Optional dyngen GRN (commented reference).** A trustworthy dyngen GRN can
  replace the correlation-based GRN. The code is written but commented out in two
  places: the constants `DYNGEN_FEATURE_NETWORK_CSV` / `DYNGEN_MODEL_GRN` and the
  loader `load_dyngen_feature_network()` in `grn_utils.py`, and the matching branch
  in `RegVelo_sim.load_prior_grn()`. To enable it, uncomment those definitions, add
  `build_regvelo_grn` (and `DYNGEN_MODEL_GRN` / `load_dyngen_feature_network`) to
  the `grn_utils` import in `RegVelo_sim.py`, and list the dataset in
  `DYNGEN_MODEL_GRN` using the dataset name without the benchmark `_dataset`
  suffix. Datasets absent from that map keep the Spearman fallback.
  `dyngen_bifurcating_cell10000_gene500_feature_network.csv` is the matching
  edge-list dump for the `bifurcating_cell10000_gene500` dataset and is shipped as
  reference data for this branch (it is not read by the default Spearman path).
- **pySCENIC is only needed for real data.** The pySCENIC resources (TF lists,
  ranking databases and motif tables) are not bundled and must be supplied through
  `AUX_DIR`; `pipeline_sim.sh` does not use them.
- The Docker image is built to train the simulated-data workflow. Its pySCENIC
  installation is written but commented out in `Dockerfile/RegVelo/Dockerfile` and
  has not been validated; uncomment it if you need `pipeline.sh` inside the
  container.
