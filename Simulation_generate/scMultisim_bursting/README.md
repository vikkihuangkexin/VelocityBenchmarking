# scMultiSim Transctriptional-Bursting Benchmark Simulation Scripts

This repository provides three scripts to generate, export, and visualize simulated single-cell RNA velocity benchmark datasets with **scMultiSim**. The workflow creates a fixed B01–B10 bursting benchmark across different gene regulatory networks, cell counts, gene counts, and kinetic variation regimes, then exports each dataset to `h5ad` and produces tSNE-based diagnostic plots.

## What these scripts do

### `01_generate_bursting_benchmark_and_export_h5ad.R`

Generates the main scMultiSim bursting benchmark datasets.

For each configuration in the B01–B10 simulation plan, the script:

- runs `scMultiSim::sim_true_counts()` with velocity enabled
- supports two built-in GRN settings:
  - `GRN_params_100`
  - `GRN_params_1139`
- varies kinetic parameters through `vary = "s"`, `"kon"`, `"koff"`, or `"all"`
- creates milestone and lineage reference annotations
- summarizes gene roles as regulator, target, regulator-target, or other
- writes both R-native and `h5ad` outputs

The script also patches an internal scMultiSim function to make GRN-based simulations safer when `vary` includes `kon` or `koff`.

### `02_batch_export_all_res_to_h5ad.R`

Recursively searches the benchmark directory for `res.rds` files and converts each one to a scVelo-friendly `res.h5ad`.

For each `res.rds`, the script:

- loads simulated counts, unspliced counts, and ground-truth velocity
- aligns cell metadata to the count matrix
- adds `pseudotime` if it is missing
- adds `pop` if it is missing
- computes one tSNE embedding with `Rtsne`
- exports a `SingleCellExperiment` object to `h5ad` with `zellkonverter`

### `03_batch_plot_all_res_h5ad.py`

Recursively searches the benchmark directory for `res.h5ad` files and generates tSNE-based visualizations for each dataset.

For each `res.h5ad`, the script:

- loads the dataset with `anndata`
- reads tSNE coordinates from `obsm["X_tsne"]` or `obsm["tsne"]`
- uses `obs["pop"]` for population labels
- uses `obs["pseudotime"]`, or falls back to `cell_time`
- reads velocity from `layers["ground_truth_velocity"]` or `layers["velocity"]`
- projects high-dimensional velocity vectors onto the tSNE embedding
- smooths projected velocity vectors locally
- writes population, pseudotime, stream, and arrow plots

## What this workflow produces

For each benchmark dataset, the main output directory contains stable artifacts such as:

### `res.rds`

The raw scMultiSim simulation result object. This is the most complete R-native output and is useful for debugging, re-exporting, or downstream analysis in R.

### `res.h5ad`

The exported AnnData-compatible file for Python workflows such as scVelo. Depending on which export script is used, the file may contain:

- `X`: count matrix
- `layers["ground_truth_velocity"]` or `layers["velocity"]`
- unspliced counts
- tSNE coordinates in `obsm`
- cell metadata in `obs`

### `milestone_graph.csv`

A directed milestone graph describing the reference trajectory structure.

### `cell_milestone_table.csv`

Per-cell trajectory annotation table containing cell IDs, population labels, reference time, milestone IDs, and lineage membership.

### `lineages.csv`

A summary table of root-to-leaf lineage paths.

### `lineage_milestones.csv`

A long-format table mapping each lineage to its ordered milestones.

### `ancestor_descendant_pairs.csv`

All ancestor-descendant milestone pairs within each lineage.

### `gene_role_summary.csv`

Per-gene annotation table marking whether each gene is a regulator, target, regulator-target, or other gene.

### `dataset_summary.csv`

One-row summary of the dataset, including requested and observed dimensions, GRN type, kinetic variation mode, number of milestones, number of lineages, and gene-role counts.

### `dataset_plan.csv`

The full B01–B10 benchmark plan written at the top-level output directory.

### `run_log.csv`

Per-dataset run status table.

### `error_log.csv`

Subset of `run_log.csv` containing failed simulations.

The plotting script additionally writes:

### `scmultisim_tsne_by_pop.png`

tSNE plot colored by population label.

### `scmultisim_tsne_by_pseudotime.png`

tSNE plot colored by pseudotime.

### `scmultisim_tsne_velocity_stream_by_pop.png`

tSNE plot with projected velocity streamlines and cells colored by population.

### `scmultisim_tsne_velocity_arrows_by_pop.png`

tSNE plot with projected per-cell velocity arrows and cells colored by population.

## Requirements

### R

Recommended: R >= 4.1

Required R packages:

```r
scMultiSim
ape
dplyr
readr
purrr
stringr
tibble
SingleCellExperiment
zellkonverter
S4Vectors
SummarizedExperiment
Rtsne
```

### Python

Recommended: Python >= 3.8

Required Python packages:

```bash
numpy
pandas
anndata
scvelo
matplotlib
scikit-learn
```

## Installation

A minimal R installation approach:

```r
install.packages(c(
  "ape",
  "dplyr",
  "readr",
  "purrr",
  "stringr",
  "tibble",
  "Rtsne"
))

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

BiocManager::install(c(
  "SingleCellExperiment",
  "zellkonverter",
  "S4Vectors",
  "SummarizedExperiment"
))

# Install scMultiSim following the instructions for your environment.
```

A minimal Python installation approach:

```bash
pip install numpy pandas anndata scvelo matplotlib scikit-learn
```

## Usage

### 1) Edit the output directory

All three scripts currently use the same hard-coded base directory:

```text
/data/khuang6/simulation/test/scmultisim/bursting_benchmark2
```

Before running the workflow, update this path in each script if needed.

### 2) Generate the benchmark datasets

```bash
Rscript 01_generate_bursting_benchmark_and_export_h5ad.R
```

### 3) Optional: regenerate all `h5ad` files

```bash
Rscript 02_batch_export_all_res_to_h5ad.R
```

By default:

```r
overwrite_h5ad <- TRUE
```

Set this to `FALSE` if you want to skip existing `res.h5ad` files.

### 4) Generate all plots

```bash
python 03_batch_plot_all_res_h5ad.py
```

## Benchmark design

The simulation plan contains ten datasets:

| Dataset | GRN | Cells | Genes | Vary | Description |
|---|---:|---:|---:|---|---|
| B01 | GRN_params_100 | 1000 | 500 | s | size-driven baseline |
| B02 | GRN_params_100 | 1200 | 800 | s | size-driven, mild bimodality |
| B03 | GRN_params_100 | 1500 | 1000 | s | size-driven, stronger mild setting |
| B04 | GRN_params_1139 | 1200 | 1500 | s | size-driven with larger GRN |
| B05 | GRN_params_1139 | 1500 | 2000 | kon | frequency-driven clean setting |
| B06 | GRN_params_1139 | 1500 | 1500 | kon | frequency-driven mild bimodality |
| B07 | GRN_params_100 | 1200 | 800 | koff | duration-driven clean setting |
| B08 | GRN_params_1139 | 1500 | 1500 | koff | duration-driven mild bimodality |
| B09 | GRN_params_1139 | 1800 | 2000 | all | mixed mild regime |
| B10 | GRN_params_1139 | 2200 | 2500 | all | mixed harder regime |

## Output layout

Outputs are saved under:

```text
<base_dir>/
  dataset_plan.csv
  run_log.csv
  error_log.csv

  B01_s_grn100_c1000_g500/
    res.rds
    res.h5ad
    milestone_graph.csv
    cell_milestone_table.csv
    lineages.csv
    lineage_milestones.csv
    ancestor_descendant_pairs.csv
    gene_role_summary.csv
    dataset_summary.csv
    scmultisim_tsne_by_pop.png
    scmultisim_tsne_by_pseudotime.png
    scmultisim_tsne_velocity_stream_by_pop.png
    scmultisim_tsne_velocity_arrows_by_pop.png
```

Dataset directory names follow:

```text
<dataset_id>_<vary>_<grn_short>_c<num_cells>_g<num_genes>
```

For example:

```text
B10_all_grn1139_c2200_g2500
```

## Reproducibility

The main simulation script uses a fixed random seed inside the scMultiSim options:

```r
rand.seed = 1
```

The batch H5AD export script also uses a fixed tSNE seed:

```r
set.seed(1)
```

Exact reproducibility may still depend on package versions, BLAS/LAPACK behavior, and platform-specific numerical differences.

## Performance notes

The workflow can be memory-intensive because several steps densify matrices with `as.matrix()` or `.toarray()`.

Potentially memory-heavy steps include:

- converting count, unspliced, and velocity matrices to dense R matrices
- computing tSNE on log-transformed expression
- loading `h5ad` expression and velocity matrices into dense NumPy arrays
- projecting velocity vectors from gene space to tSNE space with local nearest-neighbor regressions

For larger datasets, consider:

- reducing the number of genes before tSNE
- using sparse-aware preprocessing
- lowering the number of cells or genes in the benchmark plan
- running plotting as a separate step on a compute node
- skipping `03_batch_plot_all_res_h5ad.py` when only benchmark files are needed

## Failure handling

The main simulation loop wraps each dataset in `tryCatch()`. A failed dataset is recorded in `run_log.csv` and `error_log.csv`, and the remaining datasets continue running.

The plotting script also processes files one by one and reports errors without stopping the full batch.
