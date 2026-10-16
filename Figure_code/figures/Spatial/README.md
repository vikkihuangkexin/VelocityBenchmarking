# Spatial figures

From the repository root:

```bash
python -m pip install -r Figure_code/figures/requirements.txt
python Figure_code/figures/Spatial/run.py
```

Current Consistency uses STC and SVC. Downsampling uses Angle consistency,
CBDir, Transition V1, STC, and SVC. The main method roster contains seven spatial
methods. Its three RNA comparators remain available for the separate comparison
panels. Chicken-heart perturbations do not enter main scientific scores.

The entry point computes missing current results and independently verifies
results against Source_data before plotting. It writes PDF, SVG, 300-dpi PNG,
and figure-data CSV files to `Results/Spatial/figures/`. Use `--results PATH` and
`--output NEW_PATH` for other locations.

Figures cover current five-dimension/Overall rankings, directional and spatial
metric distributions, task-z ridges with mean points, the Accuracy/Consistency
bubble, runtime/RAM heatmaps and scaling, downsampling, usability radars, and
real/downsampling comparisons with RNA controls. Source-verified ESVC matrices
are separate auxiliary diagnostics and do not enter the selected main ranking.

The supplied method colors and comparison-group styles are retained. Panels
are square; multi-panel overviews use separate composite geometry. Missing
resource values remain labeled NA. Entire resource failures use the configured
display/effective-rank policy. Actual font selection is recorded in validation.
