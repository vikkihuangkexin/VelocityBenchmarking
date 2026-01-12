# dyngen Batch Simulation Script

This repository provides a single R script (`simulate_dyngen.R`) to batch-generate simulated single-cell trajectory datasets with **dyngen** across multiple backbone topologies, cell counts, and gene counts. It also exports a simple 2D embedding and per-cell dominant milestone labels for convenient downstream use.

## What this script produces

For each configuration `(backbone, n_cells, n_genes)` the script writes the following stable artifacts:

- `*_model.rds`  
  The dyngen **model** object used to generate the dataset (useful for reproducibility and debugging).

- `*_dataset.rds`  
  The exported **dyno dataset** object (expression matrix + trajectory annotations + optional velocity fields).

- `*_dimred.rds`  
  A 2D embedding computed with `dyndimred::dimred_landmark_mds()` (Pearson distance).

- `*_obs.rds`  
  A per-cell table containing the **dominant milestone** (argmax over milestone percentages).

The script also triggers dyngen’s internal `generate_dataset()` outputs (stored using dyngen’s output naming conventions).

## Requirements

- R (recommended: R >= 4.1)
- Packages:
  - `dyngen`
  - `dynplot`
  - `dyndimred` (used for 2D embedding)
  - `tidyverse`
  - `Matrix`

> Note: `dyndimred` is called via `dyndimred::...` but does not need `library(dyndimred)`.

## Installation

A minimal installation approach:

```r
install.packages(c("tidyverse", "Matrix"))
# dyngen and dynverse packages are typically installed via:
# See dynverse installation instructions for your environment.
```

If you prefer fully reproducible environments, consider using **renv**:

```r
install.packages("renv")
renv::init()
# install required packages, then:
renv::snapshot()
```

## Usage

### 1) Edit parameters in the script

Open `simulate_dyngen.R` and adjust:

- `base_dir`
- `cell_nums` (vector)
- `gene_nums` (vector)
- `backbone_names` (vector)
- `seed_base`

### 2) Run

```bash
Rscript simulate_dyngen.R
```

### 3) Optional CLI arguments

No extra CLI parser package is required. The script supports simple `--key=value` arguments:

```bash
Rscript simulate_dyngen.R \
  --base_dir=./simulation/test \
  --cells=100,500,1000 \
  --genes=1000,5000,10000 \
  --backbones=bifurcating,linear_simple \
  --seed=123
```

## Parameter semantics

### `gene_nums` controls total genes

`gene_nums` represents the **total number of genes** in the dyngen model:

```
total_genes = num_tfs + num_targets + num_hks
```

The script sets `num_tfs = nrow(backbone$module_info)` and allocates the remaining genes between targets and housekeeping using:

- `target_fraction` (default: 0.5)

If `gene_num < num_tfs + 2`, the configuration is invalid and will fail with a clear error message.

### Reproducibility

The script uses a **deterministic seed per configuration** derived from the configuration `id`, so that:

- rerunning the script yields identical outputs
- different `(backbone, cell, gene)` combinations yield distinct datasets

## Output layout

Outputs are saved under:

```
<base_dir>/<backbone_name>/
  <id>_model.rds
  <id>_dataset.rds
  <id>_dimred.rds
  <id>_obs.rds
  <id>dataset.rds        (dyngen internal output; note the prefix-style naming)
  ...
```

Where:

```
id = paste0(backbone_name, "_cell", n_cell, "_gene", gene_num)
```

## Performance notes

- `dyndimred::dimred_landmark_mds(x = as.matrix(dataset$expression), ...)` densifies the expression matrix.
  - For very large matrices, this may require substantial RAM.
  - If you scale to large `n_cells` or `n_genes`, consider:
    - gene subsampling for dimred
    - a sparse-aware embedding method
    - saving dimred as an optional step

- The batch loop will **skip** a configuration if `*_dataset.rds` already exists.

- Each simulation is wrapped in `tryCatch()` so a single failure does not stop the entire batch.

## Recommended `.gitignore`

If you are committing this to GitHub, you will usually want to ignore generated artifacts:

```gitignore
# R session / misc
.Rhistory
.RData
.Rproj.user/

# Outputs
simulation/
*.rds
```

## Citation

If you use dyngen in academic work, please cite the dyngen and dynverse papers according to their official documentation.
