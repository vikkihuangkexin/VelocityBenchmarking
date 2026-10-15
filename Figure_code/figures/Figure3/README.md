# Figure 3: scalability and usability

Recompute scalability and usability from standalone CSV inputs, reproduce the
resource heatmaps and usability bubble plots, and export per-metric rank tables
for 28 RNA velocity methods. No supplementary workbook is required at runtime.

## Code

- `run.py`: check inputs, run R analysis/plotting, validate results and export ranks.
- `code/plot_fig3.R`: compute scalability/usability scores and generate figures
  using their independent method orders and the locked plotting styles.
- `code/export_ranks.py`: export individual ranks and combined rank tables.
- `code/validate_independent.py`: independently check scores and figure outputs.
- `code/extract_supplementary_inputs.py`: optional source-workbook extraction;
  this is not part of the normal CSV-based run.
- `config/`: fixed colors, input hashes and pinned R package versions.

## Input data

Paths are relative to the repository root:

- `Source_data/scalability/scRNA_runtime.csv`.
- `Source_data/scalability/scRNA_cpu_ram.csv`.
- `Source_data/scalability/scRNA_gpu_vram.csv`.
- `Source_data/usability/scRNA_usability_items.csv`.
- `Source_data/usability/usability_rules.csv`.

Scalability averages runtime and CPU-RAM efficiency components, calculated from
reverse sample z-scores across the 68 main resource tasks. VRAM is retained as
source data and is excluded from this score. Usability combines 24 item scores
into five categories: Installability (25%), Code quality (25%), Usage guidance
(20%), Maintenance (15%) and Publication quality (15%).

## Run

From the directory containing `run.py`, with Python and R installed:

```bash
python -m pip install -r requirements.txt
Rscript --vanilla code/install_R_packages.R
python run.py
```

The R installer uses `config/r_packages.csv` and a writable `R_LIBS_USER`.
Source installation may require platform build tools and font libraries.
Repository inputs are found automatically and checked against locked hashes.
Full runs require a new or empty `results/` directory; select another location
with `--output PATH`.

Figures, plotting CSVs, calculation tables, validation outputs and logs are
saved in `results/`. Figures are exported as PDF, SVG and PNG. Rank tables are
written to the repository's `rank/scRNA/` directory.

## Include Figure 2 ranks

To export all 18 metric rank tables, supply the completed Figure 2 output path:

```bash
python run.py --fig2-results "/absolute/path/to/Figure2/results"
```

Replace the example path with your actual directory. This option is needed if
Figure 2 results are outside the default `Figure_code/Fig2/results` location.
Without Figure 2 results, the exporter produces the nine Figure 3 rank tables.
After Figure 3 has completed, refresh ranks without redrawing:

```bash
python run.py --mode ranks --fig2-results "/absolute/path/to/Figure2/results"
```

Exports include individual `*_rank.csv` files, `all_metric_ranks.csv`,
`rank_matrix.csv` and `rank_index.csv`. Each metric is ranked independently;
there is no Overall rank. Batch-run variability is ranked lower-is-better.
