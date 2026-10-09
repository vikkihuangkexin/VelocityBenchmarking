# RegVelo Script

RegVelo is a regulatory-network-aware RNA velocity method: it extends veloVI with
a gene regulatory network (GRN) prior so that transcription rates are explained by
transcription-factor activity.

RegVelo is distributed as two scripts because the GRN prior is obtained
differently for each data type:

- `RegVelo.py` — real data. The prior GRN is computed with pySCENIC
  (grn -> ctx -> aucell) and consumed as a 0/1 regulator-target matrix.
- `RegVelo_sim.py` — simulated data. The a-priori GRN that generated the data
  (scmultisim / dyngen) is used directly.

## Installation

Install the RegVelo environment (Python 3.10) and the auxiliary packages used to
build the GRN prior:

```bash
pip install regvelo scanpy scvelo GPUtil pynvml psutil
pip install pandas numpy scipy scikit-learn loompy pyarrow
```

A GPU-enabled PyTorch build is installed automatically as a dependency of
`regvelo`. The real-data workflow additionally requires pySCENIC for the
`grn` / `ctx` / `aucell` steps.

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

- `--grn-100-csv` / `--grn-1139-csv`: scmultisim long-format GRN csv files
  (columns `regulated.gene`, `regulator.gene`, `regulator.effect`) for small
  (~100 regulators) and large (~1100 regulators) networks; or
- `--dyngen-feature-network-csv`: a dyngen `model$feature_network` edge list
  (`from`/`to`/`effect`/`strength`). When it is absent, a GRN is rebuilt from the
  top Spearman-correlated TFs (`--dyngen-correlation-top-k`).

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
- `--dyngen-feature-network-csv`: dyngen model edge list (true GRN). Default: `dyngen_bifurcating_cell10000_gene500_feature_network.csv`
- `--dyngen-correlation-top-k`: top-k correlated TFs per gene when rebuilding a dyngen GRN. Default: 10
- `--max-epochs`: maximum number of training epochs. Default: 500
- `--batch-size`, `--train-size`, `--soft-constraint`, `--no-soft-constraint`, `--lam`, `--lam2`,
  `--no-save-model`, `--cuda`, `--overwrite`, `--seed`: as above

## Usage

### Real Data — Step 1: pySCENIC GRN prior

```bash
# 1a. h5ad -> loom (the helper is exposed in RegVelo.py)
python - <<'PY'
import RegVelo
RegVelo.convert_h5ad_to_loom("data.h5ad", "output/id_x", "42_Mm_embryos.h5ad")
PY

# 1b. pySCENIC grn -> ctx -> aucell
pyscenic grn output/id_x/adata.loom allTFs_mm.txt -o output/id_x/adj.csv --num_workers 40
pyscenic ctx output/id_x/adj.csv hg38/motifs.feather \
    --annotations_fname motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl \
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

Batch, large-cell-count runs (mini-batch training):

```bash
python RegVelo_sim.py \
    --input simulated_cell50000.h5ad \
    --output-dir ./output \
    --cluster-key vis_annotation \
    --max-epochs 150 \
    --batch-size 1000
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
