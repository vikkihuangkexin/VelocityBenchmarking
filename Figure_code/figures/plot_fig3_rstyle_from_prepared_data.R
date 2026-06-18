#!/usr/bin/env Rscript

## Fig3 plotting from prepared R-compatible tables.
##
## This script follows the original Fig3 R plotting style:
## - horizontal ggplot bars with black borders
## - geom_errorbar(width = 0.2, color = "black")
## - blank method-label panel on the left
## - theme_minimal with panel/grid/axis lines removed
## - ComplexHeatmap scalability heatmaps using HCL BluGrn with the lightest
##   segment removed and grey90 for NA cells
##
## Required R packages:
##   tidyverse, ggplot2, gridExtra, readxl, ComplexHeatmap, circlize, grid

rm(list = ls())

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(gridExtra)
  library(ComplexHeatmap)
  library(circlize)
  library(grid)
})

plotdata_dir <- "PlotData/Results/reversed_rank/plot/Fig3_R/PlotData"
outdir       <- "PlotData/Results/reversed_rank/plot/Fig3_R/Figures"

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

method_order <- read.csv(file.path(plotdata_dir, "method_order.csv"),
                         check.names = FALSE,
                         stringsAsFactors = FALSE)$Method

## ggplot + coord_flip places the last factor level at the top.
## Reverse factor levels so the first method in method_order, e.g. veloVI, appears at the top.
plot_method_levels <- rev(method_order)

## ggplot + coord_flip draws the first factor level at the bottom.
## Use reversed levels for bar/bubble plots so veloVI appears at the top,
## matching the heatmap row order and the final benchmark rank.
plot_method_levels <- rev(method_order)

## Pink-first palette with one extra final color.
colors_22 <- c(
  "#fccde5",
  "#b3de69", "#fdb468", "#80b1d3", "#fb8072", "#9970ab",
  "lightyellow", "#8dd3c7", "#ffff66", "#cab2d6", "grey",
  "#fb9a99", "#b2df8a", "#a6cee3", "#b15928", "#6a3d9a",
  "#e82b91", "#ff7f00", "#e31a1c", "#33a02c", "#1f78b4",
  "#d9d9d9"
)

make_method_colors <- function(methods) {
  cols <- colors_22[seq_along(methods)]
  if (length(methods) > length(colors_22)) {
    cols <- rep(colors_22, length.out = length(methods))
  }
  names(cols) <- methods
  if ("Region Velocity" %in% methods) {
    cols["Region Velocity"] <- colors_22[length(colors_22)]
  }
  cols
}

method_colors <- make_method_colors(method_order)

theme_method_labels <- theme_minimal() +
  theme(
    legend.position = "none",
    panel.background = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    axis.line = element_blank(),
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y = element_text(size = 14, face = "bold")
  )

theme_metric_bar <- theme_minimal() +
  theme(
    legend.position = "none",
    panel.background = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    axis.line = element_blank(),
    axis.text = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

make_blank_method_panel <- function(methods) {
  df <- data.frame(Method = factor(methods, levels = rev(methods)), y = seq_along(methods))
  ggplot(df, aes(x = Method, y = y)) +
    geom_blank() +
    coord_flip() +
    labs(title = "") +
    theme_method_labels
}

summarise_metric_csv <- function(csv_path) {
  dat <- read.csv(csv_path, check.names = FALSE, stringsAsFactors = FALSE, na.strings = c("", "NA", "NaN"))
  if (!"Method" %in% colnames(dat)) stop("Missing Method column: ", csv_path)
  dat$Method <- factor(dat$Method, levels = plot_method_levels)
  long <- dat %>%
    pivot_longer(-Method, names_to = "Sample", values_to = "Value") %>%
    mutate(Value = as.numeric(Value))

  long %>%
    group_by(Method) %>%
    summarise(
      mean_value = mean(Value, na.rm = TRUE),
      n_valid = sum(!is.na(Value)),
      se = ifelse(n_valid > 1, sd(Value, na.rm = TRUE) / sqrt(n_valid), NA_real_),
      .groups = "drop"
    ) %>%
    mutate(Method = factor(Method, levels = plot_method_levels))
}

make_bar_from_csv <- function(csv_path, title) {
  sm <- summarise_metric_csv(csv_path)
  ggplot(sm, aes(x = Method, y = mean_value, fill = Method)) +
    geom_bar(stat = "identity", color = "black") +
    geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se),
                  width = 0.2, color = "black", na.rm = TRUE) +
    scale_fill_manual(values = method_colors, drop = FALSE) +
    coord_flip() +
    labs(title = title) +
    theme_metric_bar
}

plot_accuracy_group <- function(group_name, metric_files, metric_titles, pdf_width, pdf_height) {
  methods <- method_order
  p0 <- make_blank_method_panel(methods)

  plots <- list(p0)
  for (i in seq_along(metric_files)) {
    f <- file.path(plotdata_dir, "Accuracy", group_name, metric_files[[i]])
    if (file.exists(f)) {
      plots[[length(plots) + 1]] <- make_bar_from_csv(f, metric_titles[[i]])
    }
  }

  if (length(plots) <= 1) {
    message("No accuracy files found for group: ", group_name)
    return(invisible(NULL))
  }

  out_pdf <- file.path(outdir, paste0("Fig3_accuracy_", group_name, "_Rstyle.pdf"))
  out_png <- file.path(outdir, paste0("Fig3_accuracy_", group_name, "_Rstyle.png"))

  pdf(out_pdf, width = pdf_width, height = pdf_height)
  do.call(grid.arrange, c(plots, nrow = 1))
  dev.off()

  png(out_png, width = pdf_width, height = pdf_height, units = "in", res = 300)
  do.call(grid.arrange, c(plots, nrow = 1))
  dev.off()
}

real_metric_files <- c(
  "angle_consistency.csv",
  "CBDir.csv",
  "transition_score.csv",
  "ICCoh.csv",
  "peak_location.csv"
)
real_metric_titles <- c("Angle", "CBDir", "Transition", "ICCoh", "Peak location")

sim_metric_files <- c(
  "angle_consistency.csv",
  "CBDir.csv",
  "transition_score.csv",
  "groundtruth_correlation.csv",
  "ICCoh.csv",
  "peak_location.csv"
)
sim_metric_titles <- c("Angle", "CBDir", "Transition", "Ground truth corr.", "ICCoh", "Peak location")

plot_accuracy_group("real", real_metric_files, real_metric_titles, pdf_width = 14, pdf_height = 12)
plot_accuracy_group("sim",  sim_metric_files,  sim_metric_titles,  pdf_width = 16, pdf_height = 12)

## Stability: downsampling ----------------------------------------------------
stab_path <- file.path(plotdata_dir, "Stability", "Downsampling.csv")
if (file.exists(stab_path)) {
  p0 <- make_blank_method_panel(method_order)
  p_down <- make_bar_from_csv(stab_path, "Downsampling")
  pdf(file.path(outdir, "Fig3_stability_downsampling_Rstyle.pdf"), width = 6, height = 12)
  grid.arrange(p0, p_down, nrow = 1)
  dev.off()
  png(file.path(outdir, "Fig3_stability_downsampling_Rstyle.png"), width = 6, height = 12, units = "in", res = 300)
  grid.arrange(p0, p_down, nrow = 1)
  dev.off()
}

## Usability bubble -----------------------------------------------------------
usa_path <- file.path(plotdata_dir, "Usability", "Velocity_Usability_Detailed_Subscore.csv")
if (file.exists(usa_path)) {
  usa <- read.csv(usa_path, check.names = FALSE, stringsAsFactors = FALSE, na.strings = c("", "NA", "NaN"))
  usa$Method <- factor(usa$Method, levels = plot_method_levels)
  usa_long <- usa %>%
    pivot_longer(-Method, names_to = "Metric", values_to = "Score") %>%
    mutate(
      Score = as.numeric(Score),
      Method = factor(Method, levels = plot_method_levels)
    )

  ## Keep category names visible and compact, but avoid numeric side legends.
  min_s <- min(usa_long$Score, na.rm = TRUE)
  max_s <- max(usa_long$Score, na.rm = TRUE)
  if (!is.finite(min_s)) min_s <- 0
  if (!is.finite(max_s)) max_s <- 1
  usa_long <- usa_long %>%
    mutate(Size = ifelse(is.na(Score), NA_real_, (Score - min_s + 1)^2))

  p0 <- make_blank_method_panel(method_order)
  p_usa <- ggplot(usa_long, aes(x = Metric, y = Method)) +
    geom_point(aes(size = Size, fill = Score), shape = 21, color = "grey30") +
    scale_size_continuous(range = c(2.5, 15), guide = "none") +
    scale_fill_gradient(low = "#fff7bc", high = "#d95f0e", guide = "none") +
    scale_x_discrete(expand = expansion(mult = c(0.15, 0.15))) +
    scale_y_discrete(expand = expansion(mult = c(0.02, 0.02))) +
    labs(title = "Usability") +
    theme_minimal() +
    theme(
      panel.grid = element_blank(),
      axis.text.y = element_blank(),
      axis.title = element_blank(),
      axis.text.x = element_text(size = 10, angle = 45, hjust = 1, vjust = 1),
      plot.margin = margin(10, 30, 10, 10),
      plot.title = element_text(hjust = 0.5)
    )

  pdf(file.path(outdir, "Fig3_usability_bubble_Rstyle.pdf"), width = 7.2, height = 12)
  grid.arrange(p0, p_usa, nrow = 1, widths = c(1.5, 2.6))
  dev.off()

  png(file.path(outdir, "Fig3_usability_bubble_Rstyle.png"), width = 7.2, height = 12, units = "in", res = 300)
  grid.arrange(p0, p_usa, nrow = 1, widths = c(1.5, 2.6))
  dev.off()
}

## Scalability heatmaps -------------------------------------------------------
make_heatmap_from_csv <- function(csv_path, heatmap_name, pdf_stub) {
  df <- read.csv(csv_path, check.names = FALSE, stringsAsFactors = FALSE, na.strings = c("", "NA", "NaN"))
  rownames(df) <- df$Method
  df$Method <- NULL
  mat <- as.matrix(df)
  mat <- apply(mat, 2, as.numeric)
  rownames(mat) <- rownames(df)

  row_order <- method_order[method_order %in% rownames(mat)]
  mat <- mat[row_order, , drop = FALSE]

  vals <- as.vector(mat)
  vals <- vals[!is.na(vals)]

  if (length(vals) == 0) {
    message("No finite values in: ", csv_path)
    return(invisible(NULL))
  }

  vmin <- min(vals, na.rm = TRUE)
  vmax_all <- max(vals, na.rm = TRUE)

  pos_vals <- sort(unique(vals[vals > 0]), decreasing = TRUE)
  if (length(pos_vals) >= 2) {
    vmax_trunc <- pos_vals[2]
  } else {
    vmax_trunc <- vmax_all
  }
  if (!is.finite(vmax_trunc) || vmax_trunc <= vmin) vmax_trunc <- vmax_all

  vals_used <- vals[vals > 0 & vals <= vmax_trunc]
  if (length(vals_used) == 0) vals_used <- vals

  brks <- c(
    vmin,
    quantile(vals_used, probs = c(0.1, 0.3, 0.5, 0.7, 0.9), na.rm = TRUE),
    vmax_trunc
  )
  brks <- sort(unique(brks))

  palette_name <- "BluGrn"
  full_col <- grDevices::hcl.colors(256, palette = palette_name, rev = TRUE)
  full_col <- full_col[41:200]
  cols_for_brks <- full_col[round(seq(1, length(full_col), length.out = length(brks)))]
  col_fun <- circlize::colorRamp2(brks, cols_for_brks)

  legend_breaks <- brks

  ht <- Heatmap(
    mat,
    name = heatmap_name,
    col = col_fun,
    na_col = "grey90",
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    column_names_side = "top",
    column_names_rot = 45,
    column_names_centered = TRUE,
    row_names_gp = gpar(fontsize = 8),
    column_names_gp = gpar(fontsize = 8),
    rect_gp = gpar(col = "white", lwd = 4),
    cell_fun = function(j, i, x, y, w, h, fill) {
      val <- mat[i, j]
      if (is.na(val)) {
        grid.text("NA", x, y, gp = gpar(fontsize = 14, col = "black"))
      } else {
        mid <- (vmin + vmax_trunc) / 2
        txt_col <- ifelse(val > mid, "white", "black")
        grid.text(sprintf("%.2f", val), x, y, gp = gpar(fontsize = 14, col = txt_col))
      }
    },
    heatmap_legend_param = list(
      at = legend_breaks,
      labels = sprintf("%.2f", legend_breaks),
      title = heatmap_name,
      legend_direction = "vertical",
      title_position = "leftcenter-rot"
    ),
    use_raster = TRUE,
    raster_quality = 2
  )

  pdf(file.path(outdir, paste0(pdf_stub, ".pdf")), width = 9.5, height = 14)
  draw(ht)
  dev.off()

  png(file.path(outdir, paste0(pdf_stub, ".png")), width = 9.5, height = 14, units = "in", res = 300)
  draw(ht)
  dev.off()
}

speed_path <- file.path(plotdata_dir, "Scalability", "docker_speed_dim_means.csv")
memory_path <- file.path(plotdata_dir, "Scalability", "docker_memory_dim_means.csv")

if (file.exists(speed_path)) {
  make_heatmap_from_csv(speed_path, "Speed (min)", "Fig3_scalability_speed_Rstyle")
}
if (file.exists(memory_path)) {
  make_heatmap_from_csv(memory_path, "Memory (GB)", "Fig3_scalability_memory_Rstyle")
}

message("Done. R-style Fig3 outputs written to: ", outdir)
