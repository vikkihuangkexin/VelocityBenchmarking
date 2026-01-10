
## velocity_confidence_plot & ridge_line_plot
### velocity_confidence_plot
This repository provides a utility function to compute and visualize **velocity confidence** for single‑cell RNA velocity analyses. The function supports two workflows:

- **AnnData workflow**: pass an `AnnData` object with precomputed velocities and neighbors; the function will call `scv.tl.velocity_confidence` and produce plots and CSV outputs.
- **Matrix workflow**: pass raw `velocity` and `distances` matrices to compute per‑cell confidence scores directly and generate the same visual outputs.

Visual outputs include:
- Embedding scatter plots (PDF and PNG) colored by velocity confidence.
- Kernel density estimate (KDE) of the velocity confidence distribution (PDF and CSV).
- CSV files with per‑cell confidence values and sampled KDE density values.

---

#### Requirements
- **Python** 3.8+  
- **Libraries**: `scvelo`, `scanpy` (for AnnData workflows), `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`  
Install with conda or pip, for example:

```bash
conda install -c conda-forge scvelo numpy scipy pandas matplotlib seaborn
```

---

#### Usage

**Using an AnnData object**
```python
from your_module import velocity_confidence_plot

# Ensure velocities and neighbors are computed:
# scv.tl.velocity(adata)
# sc.pp.neighbors(adata)

velocity_confidence_plot(adata=adata, plot_save_path='results/', basis='umap')
```

**Using raw matrices**
```python
# velocity: numpy array of shape (n_cells, n_features)
# distances: CSR matrix from sc.pp.neighbors or equivalent
velocity_length, velocity_confidence = velocity_confidence_plot(
    velocity=velocity_matrix,
    distances=distances_csr,
    plot_save_path='results/'
)
```

---

#### Outputs
- **velocity_confidence_scatter.pdf / .png** — embedding scatter colored by confidence  
- **velocity_confidence_KDE.pdf** — KDE plot of confidence distribution  
- **velocity_confidence_row.csv** — per‑cell confidence values (AnnData workflow)  
- **KDE_density_1w.csv** — KDE density values sampled across [0, 1]

---

#### Notes & Recommendations
- For the AnnData workflow, ensure velocities and neighbors are computed beforehand (`scv.tl.velocity`, `sc.pp.neighbors`).  
- For the matrix workflow, confirm that `distances` is a CSR matrix or convertible to one and that `velocity` rows correspond to the same cell ordering used to compute `distances`.  
- The KDE uses a dense grid (`10000` points) for a smooth curve; reduce this if memory or performance is a concern.  
- Ensure `plot_save_path` exists and is writable before calling the function.

---

### ridge_line_plot
This repository contains an R script to aggregate per-cell velocity confidence results from multiple methods and produce ridgeline plots that compare the confidence distributions across methods for each dataset.

## What this script does

- Reads per-method results from a configurable set of directories (`save_dir`).
- Discovers dataset IDs automatically by listing subfolders in the first available method directory.
- Loads `velocity_confidence_row.csv` from each method's dataset folder.
- Computes the density peak for each method and sorts methods by how close their peak is to 1.
- Produces a ridgeline plot per dataset with methods ordered by the computed ranking.
- Saves summary CSVs:
  - `method_order_by_dataset.csv` — ranking of methods per dataset.
  - `peak_location_by_dataset.csv` — density peak locations per method and dataset.

## Requirements

- R (>= 4.0 recommended)
- R packages:
  - `dplyr`, `ggplot2`, `ggridges`, `viridis`, `tibble`
- The script expects each method folder to contain subfolders named by dataset ID, and inside each dataset folder a file named `velocity_confidence_row.csv`.

Install packages in R:

```r
install.packages(c("dplyr", "ggplot2", "viridis", "tibble"))
# ggridges from CRAN
install.packages("ggridges")
```

## How to use

1. Clone this repository or copy the script into your project.
2. Edit the top of `benchmark_ridgeline.R`:
   - Update the `save_dir` named list to point to your method result directories.
   - Optionally set `unitvelo_subset_dir` if UniTVelo results are stored in a separate folder.
   - Optionally set `output_dir` to change where plots and CSVs are written.
3. Ensure each method directory contains subfolders for dataset IDs and that each dataset folder contains `velocity_confidence_row.csv`.
4. Run the script in R:

```bash
Rscript benchmark_ridgeline.R
```

## File/Folder layout expected

```
results/
  velocyto/
    datasetA/
      velocity_confidence_row.csv
    datasetB/
  scvelo_stochastic/
    datasetA/
      velocity_confidence_row.csv
  ...
```

The script will detect dataset IDs from the first existing method directory and then look for matching subfolders under each method.

## Output

- Per-dataset ridgeline plots (PDF and PNG) saved to `output_dir`.
- `method_order_by_dataset.csv` — method ranking per dataset.
- `peak_location_by_dataset.csv` — density peak locations per dataset and method.

## Notes & tips

- If your dataset IDs use a prefix convention (e.g., `ID-1`), the script will also try the prefix (text before `-`) as an alternative folder name for methods that may have used a different naming convention.
- If a method is missing for a dataset, it will be placed after the methods that have data in the final ordering.
- The script is intentionally conservative when computing density peaks: if a method has too few unique points, it uses the mean as a fallback.
- Adjust plot sizes and theme settings in the script to match publication requirements.
