#!/usr/bin/env python3
"""
Plot rank trajectories across Directionality/Consistency weights with direct line labels.

Purpose
-------
The original trajectory plot uses many colored lines and a legend, which can be hard to read
when colors are visually similar. This script improves readability by:

1. Giving each method a direct label at the right endpoint of its trajectory.
2. Using a larger qualitative color set, different markers, and different line styles.
3. Optionally highlighting selected methods while fading the others.
4. Exporting both PNG and PDF files.

Expected input
--------------
A wide CSV table with one row per method and rank columns named like:

    method,D0.5_C0.5,D0.55_C0.45,D0.6_C0.4,...

The script automatically detects columns matching the D*_C* pattern.

Example
-------
python plot_rank_trajectory_direct_labels.py \
  --rank_csv ../../../PlotData/Results/validation/02_direction_weight_trajectory/01_rank_by_weight_wide.csv \
  --output_dir ../../../PlotData/Results/validation/02_direction_weight_trajectory/figures \
  --output_prefix fig1_rank_trajectory_direct_labels
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


WEIGHT_COL_PATTERN = re.compile(r"^D(?P<D>[0-9.]+)_C(?P<C>[0-9.]+)$")


def parse_list_arg(s: str | None) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def detect_weight_columns(df: pd.DataFrame) -> List[Tuple[str, float, float]]:
    """Return list of (column, D weight, C weight), sorted by D weight."""
    out = []
    for col in df.columns:
        m = WEIGHT_COL_PATTERN.match(str(col))
        if m:
            out.append((col, float(m.group("D")), float(m.group("C"))))
    if not out:
        raise ValueError(
            "No weight columns found. Expected column names like D0.5_C0.5 or D0.55_C0.45."
        )
    out = sorted(out, key=lambda x: x[1])
    return out


def make_color_marker_style_cycles(n: int):
    """Generate visually distinct colors, markers, and line styles."""
    # tab20 is usually readable for <=20 methods. For larger n, interpolate turbo.
    if n <= 20:
        cmap = cm.get_cmap("tab20", 20)
        colors = [cmap(i) for i in range(n)]
    else:
        cmap = cm.get_cmap("turbo", n)
        colors = [cmap(i) for i in range(n)]

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H", "8", "p"]
    linestyles = ["-", "--", "-.", ":"]
    return colors, markers, linestyles


def repel_label_positions(y_values: np.ndarray, min_gap: float = 0.45) -> np.ndarray:
    """
    Simple one-dimensional label repulsion for direct labels on the right side.
    Rank axis is inverted later, but here we only ensure labels do not overlap too much.
    """
    y = np.asarray(y_values, dtype=float)
    order = np.argsort(y)
    adjusted = y.copy()
    last = -np.inf
    for idx in order:
        if not np.isfinite(adjusted[idx]):
            continue
        if adjusted[idx] < last + min_gap:
            adjusted[idx] = last + min_gap
        last = adjusted[idx]
    return adjusted


def plot_rank_trajectory(
    df: pd.DataFrame,
    weight_cols: List[Tuple[str, float, float]],
    output_png: Path,
    output_pdf: Path,
    title: str,
    selected_methods: List[str] | None = None,
    label_all: bool = True,
    direct_label: bool = True,
    min_label_gap: float = 0.45,
):
    selected_methods = selected_methods or []
    all_methods = df["method"].tolist()
    methods_to_plot = selected_methods if selected_methods else all_methods
    plot_df = df[df["method"].isin(methods_to_plot)].copy()

    d_weights = [d for _, d, _ in weight_cols]
    rank_cols = [col for col, _, _ in weight_cols]

    n = plot_df.shape[0]
    colors, markers, linestyles = make_color_marker_style_cycles(n)

    fig_h = max(6.0, 0.28 * n + 2.0)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))

    # Draw lines.
    endpoint_y = []
    endpoint_x = max(d_weights)
    line_meta = []
    for i, (_, row) in enumerate(plot_df.iterrows()):
        method = row["method"]
        y = row[rank_cols].astype(float).to_numpy()
        x = np.asarray(d_weights, dtype=float)

        color = colors[i]
        marker = markers[i % len(markers)]
        ls = linestyles[(i // len(markers)) % len(linestyles)]

        alpha = 1.0
        lw = 2.2
        ms = 5.5
        ax.plot(
            x,
            y,
            label=method,
            color=color,
            linestyle=ls,
            marker=marker,
            linewidth=lw,
            markersize=ms,
            alpha=alpha,
        )
        endpoint_y.append(y[-1])
        line_meta.append((method, color, y[-1]))

    # Direct labels on right side.
    if direct_label:
        endpoint_y = np.asarray(endpoint_y, dtype=float)
        label_y = repel_label_positions(endpoint_y, min_gap=min_label_gap)
        x_text = endpoint_x + 0.018
        for (method, color, y_end), y_lab in zip(line_meta, label_y):
            ax.plot([endpoint_x, x_text - 0.004], [y_end, y_lab], color=color, linewidth=0.9, alpha=0.65)
            ax.text(
                x_text,
                y_lab,
                method,
                color=color,
                fontsize=9,
                va="center",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.78),
            )
        ax.set_xlim(min(d_weights) - 0.02, endpoint_x + 0.19)
    else:
        # Legend outside, only if direct labels are disabled.
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
        ax.set_xlim(min(d_weights) - 0.02, endpoint_x + 0.02)

    # Axis settings.
    ax.set_xlabel("Directionality weight")
    ax.set_ylabel("Accuracy rank")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.35)

    # Use the exact weight values as ticks.
    ax.set_xticks(d_weights)
    ax.set_xticklabels([f"{x:.2f}" for x in d_weights])

    # Rank ticks: integer-ish spacing.
    y_values = plot_df[rank_cols].to_numpy(dtype=float)
    y_min = np.nanmin(y_values)
    y_max = np.nanmax(y_values)
    ax.set_ylim(y_max + 0.8, max(0.5, y_min - 0.8))

    # De-emphasize top/right spines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot readable rank trajectory with direct labels.")
    parser.add_argument("--rank_csv", required=True, help="Wide rank table with method and D*_C* columns.")
    parser.add_argument("--output_dir", required=True, help="Output directory for PNG/PDF figures.")
    parser.add_argument("--output_prefix", default="rank_trajectory_direct_labels")
    parser.add_argument("--selected_methods", default=None, help="Comma-separated method list. Default: all methods.")
    parser.add_argument("--exclude_methods", default=None, help="Comma-separated methods to remove.")
    parser.add_argument("--title", default="Rank trajectory across Directionality/Consistency weights")
    parser.add_argument("--min_label_gap", type=float, default=0.45)
    parser.add_argument("--no_direct_label", action="store_true", help="Use legend instead of direct endpoint labels.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.rank_csv)
    if "method" not in df.columns:
        # tolerate Method capitalization
        if "Method" in df.columns:
            df = df.rename(columns={"Method": "method"})
        else:
            raise ValueError("Input rank table must contain a 'method' column.")

    exclude = parse_list_arg(args.exclude_methods)
    if exclude:
        df = df.loc[~df["method"].isin(exclude)].copy()

    selected = parse_list_arg(args.selected_methods)
    if selected:
        # Keep selected order if possible.
        order = {m: i for i, m in enumerate(selected)}
        df = df.loc[df["method"].isin(selected)].copy()
        df["_order"] = df["method"].map(order)
        df = df.sort_values("_order").drop(columns="_order")
    else:
        # If final/mean rank columns exist, use them to order methods; otherwise preserve input order.
        if "mean_accuracy_rank" in df.columns:
            df = df.sort_values("mean_accuracy_rank")
        elif "final_accuracy_rank" in df.columns:
            df = df.sort_values("final_accuracy_rank")

    weight_cols = detect_weight_columns(df)

    plot_rank_trajectory(
        df=df,
        weight_cols=weight_cols,
        output_png=out_dir / f"{args.output_prefix}.png",
        output_pdf=out_dir / f"{args.output_prefix}.pdf",
        title=args.title,
        selected_methods=selected,
        direct_label=not args.no_direct_label,
        min_label_gap=args.min_label_gap,
    )

    print(f"Saved: {out_dir / (args.output_prefix + '.png')}")
    print(f"Saved: {out_dir / (args.output_prefix + '.pdf')}")


if __name__ == "__main__":
    main()
