# VeloVAE Velocity Analysis

## Installation

```bash
git clone https://github.com/welch-lab/VeloVAE.git
cd VeloVAE && pip install -e .
pip install scanpy scvelo pandas numpy matplotlib scipy
```

## Usage

### Real Data (Default Configuration)

For real biological data with UMAP embeddings:

```bash
python VeloVAE.py \
    --input data.h5ad \
    --output-dir ./output \
    --fig-dir ./figures \
    --cluster-key celltype
```

### Simulated Data

For simulated single-cell trajectory data with `X_dimred` basis:

```bash
python VeloVAE.py \
    --input simulated.h5ad \
    --output-dir ./output \
    --fig-dir ./figures \
    --cluster-key milestone \
    --dimred-key X_dimred \
    --dim-z 4 \
    --zero-threshold \
    --hvg-scheme simulated
```

> **Simulated data should explicitly set the following options:**
> - `--dimred-key X_dimred`: uses the pre-computed simulation coordinates.
> - `--dim-z 4`: reduces the latent cell state dimension from the default 5 to 4,
>   reflecting the relatively simpler state complexity of these synthetic trajectories
>   compared with real biological datasets.
> - `--zero-threshold`: sets `min_shared_counts = 0` and `min_shared_cells = 0`
>   because dyngen-generated simulated data often have low expression counts and the
>   default VeloVAE filtering thresholds can otherwise be overly strict.
> - `--hvg-scheme simulated`: uses the simulated-data HVG rule
>   (`500 / 2000 / 4000 / 5000`) instead of the real-data rule (`500 / 2000`).

### Batch Mode

```bash
python VeloVAE.py \
    --metadata-file datasets.csv \
    --output-dir ./output \
    --fig-dir ./figures
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | - | Input H5AD file (single-file mode) |
| `--metadata-file` | - | Metadata file for batch processing |
| `--output-dir` | Required | Output directory for H5AD files |
| `--fig-dir` | Required | Output directory for figures |
| `--cluster-key` | Required | Cell type/milestone column name in `adata.obs` |
| `--dimred-key` | `X_umap` | Dimensionality reduction key in `adata.obsm`<br>Use `X_umap` for real data, `X_dimred` for simulated data |
| `--dim-z` | `5` | Latent dimension<br>`5` for real data, `4` for simulated data |
| `--zero-threshold` | `False` | Set `min_shared_counts/cells=0` in preprocessing<br>**REQUIRED** for simulated data |
| `--hvg-scheme` | `real` | HVG selection rule<br>`real`: `500 / 2000`; `simulated`: `500 / 2000 / 4000 / 5000` |
| `--device` | `cuda:0` | PyTorch device (`cuda:0`, `cpu`, etc.) |
| `--n-jobs` | `1` | Number of parallel jobs for post-analysis |

## Metadata File Format

Required columns: `dataset_name`, `file_path`, `cluster_key`

Optional columns with defaults:
- `dimred_key` → `X_umap`
- `dim_z` → `5`
- `zero_threshold` → `False`
- `hvg_scheme` → `real`

### Example CSV

```csv
dataset_name,file_path,cluster_key,dimred_key,dim_z,zero_threshold,hvg_scheme
pancreas_real,/data/pancreas.h5ad,celltype,X_umap,5,False,real
bifurcating_sim,/data/bifurcating_cell1000_gene1000_dataset.h5ad,milestone,X_dimred,4,True,simulated
cycle_simple_sim,/data/cycle-simple_cell5000_gene1000_dataset.h5ad,milestone,X_dimred,4,True,simulated
```

## Output

### H5AD Files
- `{output_dir}/{dataset_name}/VeloVAE_{dataset_name}_plot.h5ad`

### Figures
All figures are saved in both PNG and PDF formats under `{fig_dir}/`:
- `png/VeloVAE_{dataset_name}_{basis}.png` - Scatter plot with velocity overlay
- `png/VeloVAE_{dataset_name}_stream.png` - Stream plot
- `png/VeloVAE_{dataset_name}_grid.png` - Grid plot
- `png/VeloVAE_{dataset_name}_pseudotime.png` - Pseudotime visualization
- `pdf/` - Corresponding PDF versions

### Output Data Keys

The output H5AD file contains:

**`.obs` columns:**
- `vae_time` - Inferred pseudotime
- `vae_time_normalized` - Normalized pseudotime [0, 1]

**`.obsm` keys:**
- `vae_z` - Latent cell state
- `vae_velocity_{basis}` - Velocity embedding in the specified basis

**`.layers`:**
- `vae_velocity` - RNA velocity in gene space
- `vae_uhat`, `vae_shat` - Predicted unspliced/spliced counts

**`.var` columns:**
- `vae_alpha`, `vae_beta`, `vae_gamma` - Kinetic parameters

## Important Notes

### Highly Variable Gene Selection

The number of highly variable genes (`n_gene`) is determined automatically by
`determine_preprocessing_params()` based on the selected `--hvg-scheme` and the
total gene count.

| HVG scheme | Rule |
|------------|------|
| `real` | `n_vars < 1,500` → 500 HVGs; `n_vars ≥ 1,500` → 2,000 HVGs |
| `simulated` | `n_vars < 1,500` → 500; `< 10,000` → 2,000; `< 50,000` → 4,000; `≥ 50,000` → 5,000 |

The `real` scheme reflects the conventional choice of 2,000 HVGs for most real
datasets, with a reduced setting of 500 only when the input itself contains fewer
than 1,500 genes. The `simulated` scheme is intended for synthetic datasets with
much broader gene-count ranges.

### Dimensionality Reduction Basis

The script **does not modify** the original dimensionality reduction key in `adata.obsm`. This preserves data integrity for downstream analysis:

- For real data: Use `--dimred-key X_umap` (default) to use UMAP embeddings
- For simulated data: Use `--dimred-key X_dimred` to use pre-computed simulation coordinates

If the specified key is not found, UMAP will be computed and stored under that key.

### Velocity Embedding Keys

VeloVAE stores velocity embeddings using the naming convention:
```
{method}_velocity_{basis}
```

For example:
- With `--dimred-key X_umap` → `adata.obsm['vae_velocity_umap']`
- With `--dimred-key X_dimred` → `adata.obsm['vae_velocity_dimred']`

This ensures compatibility with downstream evaluation metrics that expect specific basis names.
