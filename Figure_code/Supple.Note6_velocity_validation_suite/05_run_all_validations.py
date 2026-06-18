#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run all validation analyses with one command.

Default design:
    - Input:  PlotData/Results/accuracy
    - Output: PlotData/Results/validation
    - Directionality excludes groundtruth_correlation
    - low-consistency = bottom 30%
    - top-k = 5 and 10

Each analysis is still available as a standalone script:
    01_old_vs_new_correlation.py
    02_direction_weight_trajectory.py
    03_topk_contamination_retention.py
    04_pareto_dominated_analysis.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

from validation_common import DEFAULT_RESULTS_DIR, DEFAULT_ANALYSIS_DIR

THIS_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: list[str]):
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run all D/C validation analyses.")
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output_root", default=str(DEFAULT_ANALYSIS_DIR / "validation"))
    parser.add_argument("--old_accuracy_csv", default=None)
    parser.add_argument("--old_method_col", default=None)
    parser.add_argument("--old_score_col", default=None)
    parser.add_argument("--include_gt_in_directionality", action="store_true")
    parser.add_argument("--topks", default="5,10")
    parser.add_argument("--low_consistency_fraction", type=float, default=0.30)
    parser.add_argument("--weight_grid", default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    common = ["--results_dir", args.results_dir]
    if args.include_gt_in_directionality:
        common += ["--include_gt_in_directionality"]
    if args.weight_grid:
        common += ["--weight_grid", args.weight_grid]

    cmd1 = [sys.executable, str(THIS_DIR / "01_old_vs_new_correlation.py")] + common + [
        "--output_dir", str(output_root / "01_old_vs_new_correlation")
    ]
    if args.old_accuracy_csv:
        cmd1 += ["--old_accuracy_csv", args.old_accuracy_csv]
        if args.old_method_col:
            cmd1 += ["--old_method_col", args.old_method_col]
        if args.old_score_col:
            cmd1 += ["--old_score_col", args.old_score_col]
    run_cmd(cmd1)

    run_cmd([sys.executable, str(THIS_DIR / "02_direction_weight_trajectory_grouped_scaled_bars.py")] + common + [
        "--output_dir", str(output_root / "02_direction_weight_trajectory")
    ])

    run_cmd([sys.executable, str(THIS_DIR / "03_topk_contamination_retention.py")] + common + [
        "--output_dir", str(output_root / "03_topk_contamination_retention"),
        "--topks", args.topks,
        "--low_consistency_fraction", str(args.low_consistency_fraction),
    ])

    run_cmd([sys.executable, str(THIS_DIR / "04_pareto_dominated_analysis.py")] + common + [
        "--output_dir", str(output_root / "04_pareto_dominated_validation"),
        "--topks", args.topks,
        "--low_consistency_fraction", str(args.low_consistency_fraction),
    ])

    # Combined high-level index report.
    index = output_root / "README_RESULTS.md"
    index.write_text(
        f"""# Velocity accuracy validation outputs

Input results directory: `{args.results_dir}`  
Directionality includes groundtruth_correlation: `{args.include_gt_in_directionality}`  
Top-k values: `{args.topks}`  
Low-consistency fraction: `{args.low_consistency_fraction}`  

## Output folders

1. `01_old_vs_new_correlation/`  
   Old-vs-new rank correlation and family concordance diagnostics. If no old CSV was supplied, this folder still reports new rank vs Directionality/Consistency.

2. `02_direction_weight_trajectory/`  
   Rank trajectories across D/C weights; directionality advantage vs rank improvement.

3. `03_topk_contamination_retention/`  
   D-only vs D+C vs C-only top-k profile; low-consistency contamination; directionality retention vs consistency gain.

4. `04_pareto_dominated_validation/`  
   Pareto-front / dominated-method analysis and replacement analysis for D-only vs D+C.

Each folder contains a `validation_summary.md`, readable CSV tables, and figures under `figures/`.
""",
        encoding="utf-8",
    )
    print(f"\nAll validation analyses finished. Output root: {output_root}")


if __name__ == "__main__":
    main()
