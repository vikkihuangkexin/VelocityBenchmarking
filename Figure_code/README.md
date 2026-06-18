# Figure_code

All code used for ranking, sensitivity analysis, and figure generation is stored here.

Run the core workflow from the repository root:

```bash
bash Figure_code/run_core_pipeline.sh
```

Subfolders:

- `accuracy/`: accuracy aggregation and reversed-rank formatting.
- `scalability/`: scalability rank calculation.
- `stability/`: stability rank calculation.
- `usability/`: usability rank formatting.
- `overall/`: final overall rank calculation.
- `sensitivity/`: category-weight sensitivity analysis.
- `figures/`: figure plotting scripts.
- `figure4_cell_cycle/`: optional analysis scripts requiring external intermediate files.
