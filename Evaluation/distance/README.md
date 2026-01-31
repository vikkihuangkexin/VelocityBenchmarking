# Distance Calculation

This script calculates gene intra-class and inter-class distances based on existing velocity data.

## Required adata Internal Variables

- **Layers**: Mu, Ms (for non-TFvelo methods, or computed via scvelo.pp.moments); for TFvelo: WX, M_total (mapped to Mu, Ms).
- **Obs**: cell_type (cell type column).

## Output Variables/Content

- CSV file with columns: Gene, Intra-class distance, Inter-class distance.

## Usage

```bash
distance.py --h5ad_path /data/TFvelo_data.h5ad --method_name TFvelo --output_dir /data/gene_distance_results
```

Note: For methods other than TFvelo, the h5ad file must contain Mu and Ms layers (run scvelo.pp.moments if not present).