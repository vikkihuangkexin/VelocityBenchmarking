# Multirun Stability Analysis

This script calculates stability of RNA velocity tools based on existing velocity data.

## Required adata Internal Variables

- **Layers**: velocity (for high-dimensional vectors)
- **Obsm**: velocity_umap (for low-dimensional vectors)

## Output Variables/Content

- CSV file with columns: tool_name, dataset_name, group, group_cosine, group_median, average_cosine, average_median