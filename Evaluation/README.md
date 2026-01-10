### Evaluation

Generates standardized evaluation metrics for RNA velocity results produced by multiple methods on both real and simulated datasets.

- **Purpose**  
  Provide comparable quantitative assessments of velocity estimates across methods and datasets to support benchmarking and method selection.

- **Inputs**  
  Per-method velocity outputs and any required ground‑truth or reference data (e.g., simulated trajectories, annotated cell states).

- **Outputs**  
  Numeric metrics and summary tables (e.g., accuracy, concordance, peak/shape statistics), plus plots that visualize method performance across datasets.

- **Usage**  
  Run the evaluation script after collecting each method’s results into the configured result directories. The script reads per-method outputs, computes the metrics, and writes CSV summaries and figures to the designated output folder.
