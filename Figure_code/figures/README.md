# Figure reproduction

Existing Figure2, Figure3, cell-cycle, and metric-heatmap scripts retain their
original dependencies and entry points.

The updated multimodal workflows are:

```bash
python -m pip install -r Figure_code/figures/requirements.txt
python Figure_code/figures/ATAC/run.py
python Figure_code/figures/Spatial/run.py
```

Both use plot_benchmark.py and the current shared calculation engine. They read
Source_data and current independently verified Results, and export PDF, SVG,
300-dpi PNG, numerical figure tables, and validation metadata. Their local
READMEs document panels, methods, auxiliary data, and output options.
