# Result Plot

This script generates corresponding plots based on existing velocity data.

## Required Variables

- `data_dir`: Path to the input h5ad data file.
- `figure_dir`: Directory to save the output figures.
- `method`: Velocity method used (e.g., 'velocyto', 'cellDancer').

## Required adata Internal Variables

- **Layers**: spliced, unspliced, velocity, M_s, M_u.
- **Obsm**: X_umap, velocity_umap;

## Output Files

- UMAP velocity embedding plots: `{method}_{ID}_umap.png` and `{method}_{ID}_umap.pdf`
- Stream plots: `{method}_{ID}_stream.png` and `{method}_{ID}_stream.pdf`
- Grid plots: `{method}_{ID}_grid.png` and `{method}_{ID}_grid.pdf`
- Pseudotime plots: `{method}_{ID}_pseudotime.png` and `{method}_{ID}_pseudotime.pdf`

Files are saved in `figure_dir/png/` and `figure_dir/pdf/` subdirectories.