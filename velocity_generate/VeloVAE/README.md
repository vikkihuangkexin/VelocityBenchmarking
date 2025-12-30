# VeloVAE Velocity Analysis Usage

### Single File Mode

```bash
python run_velovae.py \
    --input data.h5ad \
    --output-dir ./output \
    --fig-dir ./figures \
    --cluster-key celltype
```

### Batch Mode

```bash
python run_velovae.py \
    --metadata-file datasets.csv \
    --output-dir ./output \
    --fig-dir ./figures
```

### Simulated Data

```bash
python run_velovae.py \
    --input simulated.h5ad \
    --output-dir ./output \
    --fig-dir ./figures \
    --cluster-key milestone \
    --dimred-key X_dimred \
    --dim-z 4 \
    --zero-threshold
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | - | Input H5AD file |
| `--metadata-file` | - | Metadata file for batch processing |
| `--output-dir` | Required | Output directory for H5AD files |
| `--fig-dir` | Required | Output directory for figures |
| `--cluster-key` | Required | Cell type column name |
| `--dimred-key` | `X_umap` | Dimensionality reduction key (computes UMAP if not found) |
| `--dim-z` | `5` | Latent dimension (5 for real data, 4 for simulated) |
| `--zero-threshold` | `False` | Use zero thresholds in preprocessing (for simulated data) |
| `--device` | `cuda:0` | PyTorch device |
| `--n-jobs` | `1` | Parallel jobs for post-analysis |

## Metadata File Format

Required columns: `dataset_name`, `file_path`, `cluster_key`

Optional columns (with defaults): `dimred_key` (X_umap), `dim_z` (5), `zero_threshold` (False)

Example:
```csv
dataset_name,file_path,cluster_key,dimred_key,dim_z,zero_threshold
pancreas,/data/pancreas.h5ad,celltype,X_umap,5,False
simulated,/data/simulated.h5ad,milestone,X_dimred,4,True
```

## Output

- H5AD: `VeloVAE_<input_name>_plot.h5ad`
- Figures: `png/` and `pdf/` subdirectories with umap, stream, grid, pseudotime plots
