# Supplementary Note 6: accuracy validation analyses

This folder contains scripts used to validate the revised accuracy aggregation and the directionality/consistency weighting strategy.

## Default input

The scripts expect the accuracy pipeline outputs under:

```bash
PlotData/Results/accuracy
```

The main required table is:

```bash
PlotData/Results/accuracy/metric_scores.csv
```

## Run all validation analyses

From the repository root:

```bash
python Figure_code/Supplementary/Supple.Note6_velocity_validation_suite/05_run_all_validations.py \
  --results_dir PlotData/Results/accuracy \
  --output_root PlotData/Results/validation \
  --topks 5,10 \
  --low_consistency_fraction 0.30
```

## Individual scripts

- `01_old_vs_new_correlation.py`: compares the previous and revised accuracy rankings when an old-ranking table is supplied.
- `02_direction_weight_trajectory_grouped_scaled_bars.py`: evaluates rank trajectories across Directionality/Consistency weights.
- `03_topk_contamination_retention.py`: evaluates top-k method composition under different ranking strategies.
- `04_pareto_dominated_analysis.py`: performs Pareto-front and dominated-tool analyses using Directionality and Consistency scores.
- `plot_rank_trajectory_direct_labels.py`: helper plotting script for rank-trajectory figures.

Figures and tables are written to the selected `--output_root`.
