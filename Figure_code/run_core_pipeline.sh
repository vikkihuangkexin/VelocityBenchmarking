#!/usr/bin/env bash
set -euo pipefail

# Run from repository root:
#   bash Figure_code/run_core_pipeline.sh

mkdir -p PlotData/Results/{accuracy,scalability,stability,reversed_rank,overall,sensitivity,fig3_plotdata} PlotData/Figures

python Figure_code/accuracy/compute_accuracy_metrics.py \
  --input_dir PlotData/accuracy \
  --output_dir PlotData/Results/accuracy

python Figure_code/accuracy/make_accuracy_reversed_rank.py \
  --rank_summary PlotData/Results/accuracy/accuracy_rank_summary.csv \
  --output PlotData/Results/reversed_rank/accuracy_rank.csv

python Figure_code/scalability/compute_scalability_rank.py \
  --input_xlsx PlotData/scalability/Docker_performance_0605.xlsx \
  --output_dir PlotData/Results/scalability \
  --reversed_dir PlotData/Results/reversed_rank

python Figure_code/stability/compute_stability_rank.py \
  --downsampling_csv PlotData/stability/Downsampling_groundtruth_correlation.csv \
  --batchrun_csv PlotData/stability/batchrun.csv \
  --output_dir PlotData/Results/stability \
  --reversed_dir PlotData/Results/reversed_rank

python Figure_code/usability/compute_usability_rank.py \
  --input_csv PlotData/usability/Velocity_Usability_overall_scores.csv \
  --output PlotData/Results/reversed_rank/usability_rank.csv

python Figure_code/overall/compute_overall_rank.py \
  --input_dir PlotData/Results/reversed_rank \
  --output_dir PlotData/Results/overall \
  --w_accuracy 0.625 \
  --w_scalability 0.125 \
  --w_stability 0.125 \
  --w_usability 0.125

if [ -f PlotData/Results/overall/overall_rank_for_plot.csv ]; then
  cp PlotData/Results/overall/overall_rank_for_plot.csv PlotData/Results/overall/final_overall_rank_for_plot.csv
fi

python Figure_code/sensitivity/compute_overall_weight_sensitivity.py \
  --input_dir PlotData/Results/reversed_rank \
  --output_dir PlotData/Results/sensitivity

python Figure_code/sensitivity/find_weight_combinations_matching_boxplot_order.py \
  --sensitivity_dir PlotData/Results/sensitivity

echo "Core benchmark workflow completed."
echo "Main outputs:"
echo "  PlotData/Results/accuracy/"
echo "  PlotData/Results/reversed_rank/"
echo "  PlotData/Results/overall/"
echo "  PlotData/Results/sensitivity/"
