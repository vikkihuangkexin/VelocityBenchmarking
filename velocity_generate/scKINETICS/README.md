# scKINETICS Velocity Analysis

It is recommended to install scKINETICS from GitHub.

## Usage

### Basic Usage (No Visualization)

```bash
python run_sckinetics.py
```

This uses default parameters:
- Input: `24_mouse_brain.h5ad`
- Peaks: `24_MouseBrain_peaks_for_sckinetics.bed`
- Output: `scKINETICS_24_plot.h5ad`

### With Visualization

```bash
python run_sckinetics.py --fig-dir sckinetics_figures
```

This generates 4 types of plots (PNG + PDF):
- UMAP scatter plot with velocity arrows
- Velocity stream plot
- Velocity grid plot
- Pseudotime heatmap

### Custom Parameters

```bash
python run_sckinetics.py \
    --input-h5ad your_data.h5ad \
    --peaks-bed your_peaks.bed \
    --output-h5ad output.h5ad \
    --genome mm10 \
    --fig-dir figures \
    --n-threads 16 \
    --n-jobs 8
```

## Input Requirements

### H5AD File

Must contain:
- `X`: Gene expression matrix
- `obs['celltype']`: Cell type annotations (will be converted to numeric `cluster`)
- Optional: `obsm['X_umap']` (will be computed if not present)

### BED File

Must be a 3-column BED file (with or without header):
- Column 1: Chromosome name
- Column 2: Start position (integer)
- Column 3: End position (integer)

Example:
```
chrom   chromStart      chromEnd
chr1    3292586 3292976
chr1    3371598 3371961
```

Or without header:
```
chr1    3292586 3292976
chr1    3371598 3371961
```

## Output

### H5AD File

The output file contains:
- `layers['velocity']`: Velocity matrix from scKINETICS
- `obsm['velocity_umap']`: Velocity embedding
- scVelo computed: `velocity_graph`, `velocity_pseudotime`, etc.

### Figures (if --fig-dir specified)

```
figures/
├── png/
│   ├── sckinetics_velocity_umap.png
│   ├── sckinetics_velocity_stream.png
│   ├── sckinetics_velocity_grid.png
│   └── sckinetics_velocity_pseudotime.png
└── pdf/
    ├── sckinetics_velocity_umap.pdf
    ├── sckinetics_velocity_stream.pdf
    ├── sckinetics_velocity_grid.pdf
    └── sckinetics_velocity_pseudotime.pdf
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-h5ad` | `24_mouse_brain.h5ad` | Input scRNA-seq data |
| `--peaks-bed` | `24_MouseBrain_peaks_for_sckinetics.bed` | Peaks BED file |
| `--output-h5ad` | `scKINETICS_24_plot.h5ad` | Output file |
| `--genome` | `mm10` | Genome assembly (mm10, hg38, etc.) |
| `--motif-pvalue` | `1e-10` | P-value threshold for motif calling |
| `--n-neighbors` | `15` | Neighbors for scanpy.pp.neighbors |
| `--n-pcs` | `40` | PCs for scanpy.pp.neighbors |
| `--n-threads` | `15` | Threads for EM algorithm |
| `--maxiter` | `20` | Max iterations for EM |
| `--knn` | `30` | Neighbors for velocity graph |
| `--fig-dir` | `None` | Figure output directory (optional) |
| `--n-jobs` | `1` | Parallel jobs for scVelo (-1=all cores) |

## Notes

- BED file header is auto-detected
- Stream plot PDF generation includes fallback to SVG conversion if needed

## Troubleshooting

### PDF stream plot is corrupted or missing

Install one of the SVG→PDF conversion tools (cairosvg, svglib, or Inkscape).

### Peak annotation fails

Ensure the genome assembly matches your data (`--genome mm10` for mouse, `hg38` for human).
