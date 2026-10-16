#!/usr/bin/env python3
"""Reproduce the metric-versus-ground-truth scatter plots.

Run from any directory after installing requirements.txt (Python >= 3.11):
    python Figure_code/figures/Figure2/measurements_corr/plot_correlations.py
    python Figure_code/figures/Figure2/measurements_corr/plot_correlations.py --metrics all

Input: Source_data/measurements_raw/scRNA/measurements_corr/gtc_paired_values.csv
This CSV is a direct export of the 'GTC paired values' sheet in
scRNA_velocity_metric_analysis.xlsx from AngleConsistency_analysis(1).zip.
Each row is an observed method/dataset/metric pair, not a cell or a run.

Both means for each point use the same paired simulated datasets. Datasets
without supplied true velocity cannot enter these pairs. Missing values are
not imputed. The archive's method scope is retained: Region-Velocity is absent;
PhyloVelo is retained where measured (29 methods for Angle/CBDir, 28 otherwise).
No ranking, task z-score or real/simulation averaging is performed here.
Consistency score denotes the KDE peak location of cell velocity confidence.

Statistics: two-sided Spearman tests across method means; BH adjustment always
covers the same five tests, even when only three metrics are plotted. Plot
annotations show raw P. Lines are OLS fits; shading is the 95% confidence
interval for the fitted mean, not a Spearman or prediction interval.

Colors, sizes, point styles, label offsets, OLS band and axis padding follow
scripts/plot_ground_truth.py in the source archive. Arial is used if installed;
otherwise Matplotlib's bundled DejaVu Sans is used. No private font files are
required. Every plotting area is square; axis units remain metric specific.
Individual plots label all methods; the compact overview labels a subset.

Outputs (created only at runtime): PDF/SVG/600-dpi PNG, method_means.csv,
correlation_statistics.csv and regression_lines.csv in ./results beside this
script. Override locations with --data and --output. No panel-letter mapping
is assumed because the archive does not identify manuscript panels e/f/g.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
METRICS = ("angle_consistency", "cbdir", "transition_score", "iccoh", "consistency_score")
LABELS = ("Angle consistency", "CBDir", "Transition score", "ICCoh", "Consistency score")
X_LABELS = ("Mean angle consistency score (%)", "Mean CBDir", "Mean Transition score",
            "Mean ICCoh", "Mean Consistency peak location")
COLORS = ("#1F6F9A", "#2A9D8F", "#D9982F", "#8B6BAE", "#C65A75")
STYLE = {m: dict(title=a, xlabel=b, color=c)
         for m, a, b, c in zip(METRICS, LABELS, X_LABELS, COLORS)}


def calculate(data_path: Path):
    pairs = pd.read_csv(data_path, float_precision="round_trip")
    required = {"metric", "method", "dataset_id", "metric_value", "groundtruth_correlation"}
    if not required <= set(pairs.columns) or set(pairs.metric) != set(METRICS):
        raise ValueError("Expected paired values for exactly the five supplied metrics.")
    if pairs[list(required)].isna().any().any():
        raise ValueError("The exported pair table must contain complete observations.")
    if pairs.duplicated(["metric", "method", "dataset_id"]).any():
        raise ValueError("Duplicate metric/method/dataset key.")
    normalized = pairs.method.str.lower().str.replace(r"[-_\s]", "", regex=True)
    if normalized.eq("regionvelocity").any():
        raise ValueError("Region-Velocity is outside this analysis scope.")
    for column in ("metric_value", "groundtruth_correlation"):
        pairs[column] = pd.to_numeric(pairs[column], errors="raise")
        if not np.isfinite(pairs[column]).all():
            raise ValueError(f"Nonfinite values in {column}.")

    # Pair first, then average: GTC coverage can differ between metrics.
    means = pairs.groupby(["metric", "method"], sort=True, as_index=False).agg(
        metric_value=("metric_value", "mean"),
        groundtruth_correlation=("groundtruth_correlation", "mean"),
        n_datasets=("dataset_id", "nunique"),
    )
    stats = []
    for metric in METRICS:
        points = means.loc[means.metric.eq(metric)]
        x, y = points.metric_value.to_numpy(), points.groundtruth_correlation.to_numpy()
        if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            raise ValueError(f"Insufficient variation for {metric}.")
        test = spearmanr(x, y, alternative="two-sided")
        stats.append(dict(metric=metric, n_methods=len(points),
                          n_pairs=int(pairs.metric.eq(metric).sum()),
                          n_datasets=pairs.loc[pairs.metric.eq(metric), "dataset_id"].nunique(),
                          spearman_rho=test.statistic, raw_p=test.pvalue))
    stats = pd.DataFrame(stats)
    p = stats.raw_p.to_numpy()
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    bh = np.empty(len(p))
    bh[order] = np.minimum(adjusted, 1.0)
    stats["bh_adjusted_p_five_tests"] = bh
    return means, stats


def regression(points: pd.DataFrame):
    x, y = points.metric_value.to_numpy(), points.groundtruth_correlation.to_numpy()
    coefficients = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 200)
    fitted = np.polyval(coefficients, grid)
    residuals = y - np.polyval(coefficients, x)
    df = len(x) - 2
    residual_scale = np.sqrt(np.sum(residuals ** 2) / df)
    leverage = 1 / len(x) + (grid - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
    margin = t.ppf(0.975, df) * residual_scale * np.sqrt(leverage)
    return pd.DataFrame(dict(metric=points.metric.iloc[0], x=grid, fitted=fitted,
                             ci_lower=fitted - margin, ci_upper=fitted + margin))


def method_labels(axis, x, y, names, color, compact, all_x, all_y):
    """Deterministic source-style candidate offsets; prefer no text/point overlap."""
    size, width = (5.2, 0.25) if compact else (5.8, 0.32)
    candidates = [(5, 6), (5, -8), (-5, 6), (-5, -8), (0, 11), (0, -13),
                  (10, 2), (-10, 2), (12, 10), (12, -11), (-12, 10), (-12, -11),
                  (5, 17), (5, -19), (-5, 17), (-5, -19), (14, 18), (14, -20),
                  (-14, 18), (-14, -20), (18, 5), (18, -8), (-18, 5), (-18, -8),
                  (27, 8), (27, -10), (-27, 8), (-27, -10), (35, 15), (35, -17),
                  (-35, 15), (-35, -17), (42, 4), (42, -5), (-42, 4), (-42, -5)]
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    font = fm.FontProperties(family=plt.rcParams["font.family"], size=size)
    scale = axis.figure.dpi / 72
    radius = 3.2 if compact else 4.8
    point_boxes = [Bbox.from_extents(px-radius, py-radius, px+radius, py+radius).expanded(1.12, 1.12)
                   for px, py in axis.transData.transform(np.column_stack([all_x, all_y]))]
    # Keep labels out of the statistics box as well as away from one another.
    used = [axis.texts[0].get_window_extent(renderer).expanded(1.06, 1.08)]
    bounds = axis.get_window_extent(renderer)
    for i in np.argsort(y, kind="stable"):
        w, h, _ = renderer.get_text_width_height_descent(names[i], font, ismath=False)
        px, py = axis.transData.transform((x[i], y[i]))
        choices = []
        for dx, dy in candidates:
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            left = px + dx * scale - (0 if dx > 0 else w if dx < 0 else w / 2)
            bottom = py + dy * scale - h / 2
            box = Bbox.from_extents(left, bottom, left+w, bottom+h).expanded(1.1, 1.25)
            outside = (max(bounds.x0-box.x0, 0) + max(box.x1-bounds.x1, 0)
                       + max(bounds.y0-box.y0, 0) + max(box.y1-bounds.y1, 0))
            cost = (outside * 1e6 + sum(box.overlaps(p) for p in point_boxes) * 1e5
                    + sum(box.overlaps(p) for p in used) * 1e4 + np.hypot(dx, dy))
            choices.append((cost, dx, dy, ha, box))
        _, dx, dy, ha, box = min(choices, key=lambda row: row[0])
        axis.annotate(names[i], xy=(x[i], y[i]), xytext=(dx, dy), textcoords="offset points",
                      ha=ha, va="center", fontsize=size, color="#263238", annotation_clip=False,
                      arrowprops=dict(arrowstyle="-", color=color, lw=width, alpha=0.60,
                                      shrinkA=2.5, shrinkB=1.5))
        used.append(box)


def draw(axis, metric, means, stats, line, compact=False):
    points = means.loc[means.metric.eq(metric)].reset_index(drop=True)
    row = stats.loc[stats.metric.eq(metric)].iloc[0]
    style = STYLE[metric]
    x, y = points.metric_value.to_numpy(), points.groundtruth_correlation.to_numpy()
    color = style["color"]
    axis.set_box_aspect(1)
    axis.scatter(x, y, s=18 if compact else 28, color=color, edgecolor="white", linewidth=0.45, zorder=3)
    axis.fill_between(line.x, line.ci_lower, line.ci_upper, color=color, alpha=0.24, linewidth=0, zorder=1)
    axis.plot(line.x, line.fitted, color=color, lw=1.05, alpha=0.90, zorder=2)
    lower, upper = min(y.min(), line.ci_lower.min()), max(y.max(), line.ci_upper.max())
    axis.set_ylim(min(0.40, lower-0.02), max(0.72, upper+0.02))
    pad = 0.15 if compact else 0.10 if metric == "transition_score" else 0.06
    axis.set_xlim(x.min()-pad*np.ptp(x), x.max()+pad*np.ptp(x))
    axis.set_title(style["title"], loc="left", pad=4)
    axis.set_xlabel(style["xlabel"])
    axis.set_ylabel("Mean ground-truth correlation")
    p_label = "< 0.001" if row.raw_p < 0.001 else f"= {row.raw_p:.3f}"
    axis.text(0.02, 0.98, f"Spearman's ρ = {row.spearman_rho:.2f}\nP {p_label}; n = {row.n_methods}",
              transform=axis.transAxes, va="top", fontsize=5.7 if compact else 6.2,
              color="#263238", zorder=5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2))
    indices = np.arange(len(points))
    if compact and len(points) > 12:
        ordered = np.argsort(x, kind="stable")
        middle = ordered[4:-4]
        indices = np.unique(np.r_[ordered[:4], ordered[-4:], middle[np.linspace(0, len(middle)-1, 4, dtype=int)]])
    names = points.method.replace({"k-velo": "κ-velo"}).to_numpy()
    method_labels(axis, x[indices], y[indices], names[indices], color, compact, x, y)


def save(figure, stem, formats, dpi):
    for extension in formats:
        metadata = {"Date": None} if extension == "svg" else ({"CreationDate": None, "ModDate": None} if extension == "pdf" else None)
        kwargs = dict(dpi=dpi)
        if metadata is not None:
            kwargs["metadata"] = metadata
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        buffer = io.BytesIO()
        figure.savefig(buffer, format=extension, **kwargs)
        stem.with_suffix("." + extension).write_bytes(buffer.getvalue())
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "Source_data/measurements_raw/scRNA/measurements_corr/gtc_paired_values.csv")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--metrics", nargs="+", default=list(METRICS[:3]), help="Metric keys, or 'all' for all five.")
    parser.add_argument("--formats", nargs="+", choices=["pdf", "svg", "png", "tiff"], default=["pdf", "svg", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--font", default=None, help="Installed font family; default Arial if available, else DejaVu Sans.")
    args = parser.parse_args()
    metrics = list(METRICS) if args.metrics == ["all"] else args.metrics
    if not metrics or len(set(metrics)) != len(metrics) or not set(metrics) <= set(METRICS):
        parser.error("Use unique metric keys from " + ", ".join(METRICS) + ", or 'all'.")
    if args.dpi < 300:
        parser.error("--dpi must be at least 300.")
    installed = {font.name for font in fm.fontManager.ttflist}
    family = args.font or ("Arial" if "Arial" in installed else "DejaVu Sans")
    if family not in installed:
        parser.error(f"Font family not installed: {family}")
    plt.rcParams.update({"font.family": family, "font.size": 6.3, "axes.titlesize": 7,
                         "axes.titleweight": "bold", "axes.labelsize": 7, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "svg.fonttype": "none",
                         "svg.hashsalt": "measurements_corr", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "text.color": "#263238", "axes.labelcolor": "#263238",
                         "xtick.color": "#263238", "ytick.color": "#263238"})
    means, stats = calculate(args.data)
    lines = {metric: regression(means.loc[means.metric.eq(metric)]) for metric in METRICS}
    args.output.mkdir(parents=True, exist_ok=True)
    means.to_csv(args.output / "method_means.csv", index=False, float_format="%.17g")
    stats.to_csv(args.output / "correlation_statistics.csv", index=False, float_format="%.17g")
    pd.concat(lines.values(), ignore_index=True).to_csv(args.output / "regression_lines.csv", index=False, float_format="%.17g")
    for metric in metrics:
        figure, axis = plt.subplots(figsize=(150/25.4, 150/25.4), constrained_layout=True)
        draw(axis, metric, means, stats, lines[metric])
        save(figure, args.output / f"{metric}_vs_ground_truth", args.formats, args.dpi)
    # Three directional metrics form one row; five comparisons use the source's 3x2 layout.
    ncols = 3 if len(metrics) == 3 else min(2, len(metrics))
    nrows = (len(metrics) + ncols - 1) // ncols
    height_mm = 70 if nrows == 1 else 170
    figure, axes = plt.subplots(nrows, ncols, figsize=(180/25.4, height_mm/25.4),
                                constrained_layout=True, squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        draw(axis, metric, means, stats, lines[metric], compact=True)
    for axis in list(axes.flat)[len(metrics):]:
        axis.set_visible(False)
    save(figure, args.output / "metrics_vs_ground_truth_summary", args.formats, args.dpi)
    print(f"Font: {family}; figures: {len(metrics)+1}; output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
