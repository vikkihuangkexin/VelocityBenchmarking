# ATAC figures

From the repository root:

```bash
python -m pip install -r Figure_code/figures/requirements.txt
python Figure_code/figures/ATAC/run.py
```

The entry point computes missing current results, verifies existing results
against Source_data, and writes PDF, SVG, 300-dpi PNG, and figure-data CSV files
to `Results/ATAC/figures/`. Use `--results PATH` and `--output NEW_PATH` to select
other locations. Recompute after changing calculation inputs or configuration.

Figures include five-dimension/Overall rankings, metric distributions, the
Accuracy/Consistency bubble, runtime/RAM heatmaps and scaling, downsampling
distributions and retention curves, usability radars, four-task sparse metric
comparisons, and matched-real-dataset comparisons with TSvelo and both scVelo
modes. All comparisons are recomputed from the source matrices. Method colors
are retained from the supplied plotting package. Main scientific plots use
ordinary simulations; downsampling is displayed separately.

The ten downsampling tasks use means recomputed from the newly supplied
cell-level rescaled cosine file. Its source hash, cell counts, and aggregation
receipt are in the downsampling data directory. The cell-level input is not
needed for normal reruns. To regenerate the summary from that external file:

```bash
python Figure_code/stability/prepare_atac_downsampling.py --raw /path/to/GroundTruthCorrelation_ATAC_sub_01_raw.csv
```

Single comparison panels have square canvases. Composite layouts retain their
panel geometry. The plotting validation records the actual installed font,
using Arial, Liberation Sans, Nimbus Sans, or DejaVu Sans in that order.
