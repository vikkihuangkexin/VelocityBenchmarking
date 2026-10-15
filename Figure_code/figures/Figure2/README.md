# Figure 2: scRNA velocity measurements

Recompute directional accuracy and reproduce the scientific-metric panels for
28 RNA velocity methods. Python handles data processing and validation; R
generates the main figures using the locked method colors.

## Code

- `run.py`: run input checks, analysis, R plotting and validation.
- `code/parse_original_inputs.py`: clean and organize the supplied measurements.
- `code/prepare_accuracy.py`: compute task z-scores, directional accuracy ranks,
  raw metric means and dataset-level standard errors.
- `code/prepare_normalized_sd.py`: summarize standardized batch-run variability.
- `code/plot_abc.R`: assemble the Methods, Primary metrics and Secondary metrics
  panels using the accompanying R plotting scripts.
- `config/`: method annotations, fixed colors, plotting settings and input hashes.

## Input data

Inputs are located under `Source_data/measurements_raw/scRNA/`:

- `methods.csv` and `real_dataset_ids.csv`.
- `measurement/summary/`: primary metric summaries.
- `measurement/std/`: supplied batch-run SD summaries.
- `measurement/downsampling/`: the separate downsampling measurements.

Directional accuracy combines Angle consistency, CBDir, Transition V1 and
ground-truth correlation using per-task sample z-scores. Shared metrics use
equal real/simulation weights; GT uses simulation only. Available components
receive equal weights, and exact ties receive average ranks. Consistency,
batch-run variability and downsampling do not enter directional accuracy.

Metric bars show raw means with dataset-level SE, rather than z-scores.
Standardized batch variability is lower-is-better. Missing results remain
missing; valid zero and negative metric values are retained.

## Run

From the directory containing the main `run.py`, with Python and R installed:

```bash
python -m pip install -r requirements.txt
Rscript --vanilla code/install_R_packages.R
python run.py
```

The R installer uses `config/r_packages.csv` and a writable `R_LIBS_USER`.
Source installation may require platform build tools and font libraries.
Inputs are found automatically within the repository and checked against the
locked hashes. Use `--data-root PATH` for another input location.

Outputs are written to `results/`: score/rank tables, plotting CSVs,
PDF/SVG/PNG figures, validation results and logs. Full runs require a new or
empty output directory; use `--output PATH` to select one. Use `--mode analysis`
for calculations only, or `--mode plots` to redraw a completed analysis.

## Separate correlation plots

The additional `measurements_corr/plot_correlations.py` module compares
method-average metrics with ground-truth correlation. It reads
`Source_data/measurements_raw/scRNA/measurements_corr/gtc_paired_values.csv` and
has its own `requirements.txt`.

Run it from its own directory with `python plot_correlations.py`. The default
plots Angle consistency, CBDir and Transition score; `--metrics all` includes
ICCoh and Consistency score. Spearman tests and OLS lines with 95% confidence
bands are recomputed from paired values. This module runs separately from the
directional-ranking workflow.
