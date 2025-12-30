# Evaluation

## Ground Truth Correlation

### Function

```python
result = calculate_groundtruth_correlation(
    adata_or_path,
    method,
    dataset_id,
    output_csv,
    velocity_key='velocity',
    gt_velocity_key='ground_truth_velocity',
    use_low_dim=True,
    gt_npz_base_dir=None,
    apply_tool_preprocessing=True,
    raise_on_gt_failure=True,
    allow_partial_cell_match=True,
    min_cell_match_ratio=0.95
)
```

### Input

An AnnData object or H5AD file path containing predicted velocity information (specified by `velocity_key`). The object must contain velocity in `adata.layers` or low-dimensional velocity embedding in `adata.obsm`. For simulation data, requires precomputed ground truth NPZ files containing ground truth velocity vectors and unified X_dimred coordinates.

**Key parameters:**
- `velocity_key`: Velocity key in adata.layers/obsm. Common values: `'velocity'` (default), `'sck_velocity'` (scKINETICS), `'sde_velocity'` (SDEvelo), `'phylovelo_velocity'` (PhyloVelo)
- `gt_npz_base_dir`: Base directory for ground truth NPZ files (optional)

### Output

Returns a dictionary containing:
- `mean_cosine`: Mean cosine similarity between predicted and ground truth velocity [0-1]
- `n_cells_total`, `n_cells_valid`: Cell counts
- `cell_match_ratio`: Ratio of cells matched to ground truth
- `success`: Boolean indicating success status

Outputs a CSV file in wide format (Method × Datasets) with columns:
- Method name, per-dataset scores, AVG (mean), Reversed_rank (1=worst, N=best)

### Command Line Usage

**Single file:**
```bash
python groundtruth_correlation.py \
    --input result.h5ad \
    --method VeloVAE \
    --dataset-id bifurcating_cell1000_gene1000 \
    --output-csv results.csv \
    --velocity-key velocity
```

**Batch processing:**
```bash
python groundtruth_correlation.py \
    --metadata-csv datasets.csv \
    --output-csv results.csv \
    --verbose
```

Batch CSV format:
```csv
method,id,path,vkey
VeloVAE,bifurcating_cell1000_gene1000,/path/to/result.h5ad,velocity
scKINETICS,cycle-simple_cell5000_gene500,/path/to/result.h5ad,sck_velocity
```

### Notes

- **Tool-specific preprocessing**: Automatically handles PhyloVelo (low-dim only), cellDancer (cluster renaming), TFvelo (velocity computation), SDEvelo (velocity embedding), TopicVelo (key renaming)
- **Reversed_rank**: Rank 1 = worst (lowest correlation), higher rank = better performance
- **Path support**: All paths support absolute or relative paths
- **Ground truth recovery**: Matches cells between predicted results and GT NPZ files, replaces X_dimred with unified coordinates for fair comparison
