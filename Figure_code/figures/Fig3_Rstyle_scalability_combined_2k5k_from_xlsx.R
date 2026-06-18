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
  library(readxl)
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
## The standard docker scalability heatmaps keep the original BluGrn palette.
## The HVG 2k/5k values are parsed from:
##   PlotData/scalability/HVG_2k5k_Docker_performance_0519.xlsx
##
## Final scalability output:
##   A single combined figure:
##     Speed (green) + 2k/5k time (salmon) + Memory (green) + 2k/5k memory (salmon)
##
## Color meaning:
##   darker color = longer time or larger memory.

hvg_2k5k_xlsx <- "PlotData/scalability/HVG_2k5k_Docker_performance_0519.xlsx"

normalize_scalability_method <- function(x) {
  x <- as.character(x)
  x <- trimws(gsub("\\s+", " ", x))
  key <- tolower(gsub("[^A-Za-z0-9]+", "", x))

  x[key %in% c("scrnakinetics", "scrnakinetic")] <- "scRNAkinetics"
  x[key == "regionvelocity"] <- "Region Velocity"
  x[key == "scvelodynamic"] <- "scVelo dynamical"
  x[key == "scvelostochastic"] <- "scVelo stochastic"
  x[key == "pyrovelocity"] <- "Pyro-Velocity"

  x
}

## Harmonize scalability method order with the benchmark method list.
scalability_method_order <- normalize_scalability_method(method_order)
if (!"Region Velocity" %in% scalability_method_order) {
  scalability_method_order <- c(scalability_method_order, "Region Velocity")
}

coerce_numeric_or_na <- function(x) {
  suppressWarnings(as.numeric(x))
}

is_unit_or_empty_row <- function(x) {
  sx <- trimws(as.character(x))
  if (is.na(sx) || sx == "") return(TRUE)
  if (grepl("^Unit\\s*[:：]", sx, ignore.case = TRUE)) return(TRUE)
  if (tolower(sx) %in% c("unit", "unit seconds", "unit minutes")) return(TRUE)
  FALSE
}

hvg_block_label <- function(x) {
  sx <- as.character(x)
  if (grepl("2000\\s*velocity\\s*genes", sx, ignore.case = TRUE)) return("2k HVG")
  if (grepl("5000\\s*velocity\\s*genes", sx, ignore.case = TRUE)) return("5k HVG")
  NA_character_
}

read_hvg_2k5k_sheet <- function(xlsx_path, sheet_name, convert_time_to_min = FALSE) {
  raw <- readxl::read_excel(
    xlsx_path,
    sheet = sheet_name,
    col_names = FALSE,
    .name_repair = "minimal"
  )
  raw <- as.data.frame(raw, stringsAsFactors = FALSE)

  rows_out <- list()
  i <- 1

  while (i <= nrow(raw)) {
    block_label <- hvg_block_label(raw[i, 1])

    if (is.na(block_label)) {
      i <- i + 1
      next
    }

    header_i <- i + 1
    if (header_i > nrow(raw)) break

    header <- as.character(unlist(raw[header_i, ], use.names = FALSE))
    if (length(header) < 2 || tolower(trimws(header[1])) != "method") {
      i <- i + 1
      next
    }

    j <- header_i + 1
    block_rows <- list()

    while (j <= nrow(raw)) {
      first_val <- raw[j, 1][[1]]

      ## Stop at the next block, blank row, or explicit Unit row.
      ## Do NOT use startsWith("unit"), because UniTVelo starts with "Uni".
      if (!is.na(hvg_block_label(first_val)) || is_unit_or_empty_row(first_val)) {
        break
      }

      row_vec <- as.character(unlist(raw[j, ], use.names = FALSE))
      block_rows[[length(block_rows) + 1]] <- row_vec
      j <- j + 1
    }

    if (length(block_rows) > 0) {
      block <- as.data.frame(do.call(rbind, block_rows), stringsAsFactors = FALSE)
      colnames(block) <- make.names(header, unique = TRUE)
      colnames(block)[1] <- "Method"

      block$Method <- normalize_scalability_method(block$Method)

      value_cols <- setdiff(colnames(block), "Method")
      for (cc in value_cols) {
        block[[cc]] <- coerce_numeric_or_na(block[[cc]])
      }

      value_mat <- as.matrix(block[, value_cols, drop = FALSE])
      storage.mode(value_mat) <- "numeric"

      method_mean <- rowMeans(value_mat, na.rm = TRUE)
      method_mean[is.nan(method_mean)] <- NA_real_

      if (isTRUE(convert_time_to_min)) {
        method_mean <- method_mean / 60
      }

      tmp <- data.frame(
        Method = block$Method,
        Size = block_label,
        Value = method_mean,
        stringsAsFactors = FALSE
      )
      rows_out[[length(rows_out) + 1]] <- tmp
    }

    i <- max(j, i + 1)
  }

  if (length(rows_out) == 0) {
    return(data.frame(Method = character(0), Size = character(0), Value = numeric(0)))
  }

  do.call(rbind, rows_out)
}

make_hvg_2k5k_matrix <- function(xlsx_path, sheet_name, convert_time_to_min = FALSE) {
  long <- read_hvg_2k5k_sheet(
    xlsx_path = xlsx_path,
    sheet_name = sheet_name,
    convert_time_to_min = convert_time_to_min
  )

  if (nrow(long) == 0) {
    mat <- matrix(
      NA_real_,
      nrow = length(scalability_method_order),
      ncol = 2,
      dimnames = list(scalability_method_order, c("2k HVG", "5k HVG"))
    )
    return(mat)
  }

  wide <- long %>%
    group_by(Method, Size) %>%
    summarise(Value = mean(Value, na.rm = TRUE), .groups = "drop") %>%
    mutate(Value = ifelse(is.nan(Value), NA_real_, Value)) %>%
    tidyr::pivot_wider(names_from = Size, values_from = Value)

  if (!"2k HVG" %in% colnames(wide)) wide[["2k HVG"]] <- NA_real_
  if (!"5k HVG" %in% colnames(wide)) wide[["5k HVG"]] <- NA_real_

  wide <- as.data.frame(wide, stringsAsFactors = FALSE)
  rownames(wide) <- wide$Method
  wide$Method <- NULL

  mat <- as.matrix(wide[, c("2k HVG", "5k HVG"), drop = FALSE])
  storage.mode(mat) <- "numeric"

  missing_methods <- setdiff(scalability_method_order, rownames(mat))
  if (length(missing_methods) > 0) {
    miss <- matrix(
      NA_real_,
      nrow = length(missing_methods),
      ncol = ncol(mat),
      dimnames = list(missing_methods, colnames(mat))
    )
    mat <- rbind(mat, miss)
  }

  mat <- mat[scalability_method_order, c("2k HVG", "5k HVG"), drop = FALSE]
  mat
}

read_scalability_csv_matrix <- function(csv_path) {
  df <- read.csv(csv_path, check.names = FALSE, stringsAsFactors = FALSE, na.strings = c("", "NA", "NaN"))
  if (!"Method" %in% colnames(df)) stop("Missing Method column: ", csv_path)

  df$Method <- normalize_scalability_method(df$Method)
  rownames(df) <- df$Method
  df$Method <- NULL

  mat <- as.matrix(df)
  mat <- apply(mat, 2, as.numeric)
  rownames(mat) <- rownames(df)

  missing_methods <- setdiff(scalability_method_order, rownames(mat))
  if (length(missing_methods) > 0) {
    miss <- matrix(
      NA_real_,
      nrow = length(missing_methods),
      ncol = ncol(mat),
      dimnames = list(missing_methods, colnames(mat))
    )
    mat <- rbind(mat, miss)
  }

  mat <- mat[scalability_method_order, , drop = FALSE]
  mat
}

make_scalability_heatmap_object <- function(mat,
                                            heatmap_name,
                                            palette_mode = c("green", "salmon"),
                                            high_color = "#FA5F55",
                                            mid_color = "#FA8072",
                                            show_row_names = TRUE) {
  palette_mode <- match.arg(palette_mode)

  vals <- as.vector(mat)
  vals <- vals[!is.na(vals)]

  if (length(vals) == 0) {
    vals <- c(0, 1)
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
  if (!is.finite(vmax_trunc) || vmax_trunc <= vmin) vmax_trunc <- vmin + 1

  vals_used <- vals[vals > 0 & vals <= vmax_trunc]
  if (length(vals_used) == 0) vals_used <- vals

  brks <- c(
    vmin,
    quantile(vals_used, probs = c(0.1, 0.3, 0.5, 0.7, 0.9), na.rm = TRUE),
    vmax_trunc
  )
  brks <- sort(unique(brks))

  if (length(brks) < 2) {
    brks <- c(vmin, vmax_trunc)
  }

  if (palette_mode == "green") {
    palette_name <- "BluGrn"
    full_col <- grDevices::hcl.colors(256, palette = palette_name, rev = TRUE)
    full_col <- full_col[41:200]
  } else {
    ## Same mapping logic as the green heatmap, but salmon-red color.
    ## Low values are light; high values are dark.
    full_col <- grDevices::colorRampPalette(
      c("#FDEDEC", "#FADBD8", mid_color, high_color, "#B22222")
    )(256)
    full_col <- full_col[41:200]
  }

  cols_for_brks <- full_col[round(seq(1, length(full_col), length.out = length(brks)))]
  col_fun <- circlize::colorRamp2(brks, cols_for_brks)

  Heatmap(
    mat,
    name = heatmap_name,
    col = col_fun,
    na_col = "grey90",
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    show_row_names = show_row_names,
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
      at = brks,
      labels = sprintf("%.2f", brks),
      title = heatmap_name,
      legend_direction = "vertical",
      title_position = "leftcenter-rot"
    ),
    use_raster = TRUE,
    raster_quality = 2
  )
}

make_single_scalability_heatmap <- function(mat, heatmap_name, pdf_stub, palette_mode, show_row_names = TRUE) {
  ht <- make_scalability_heatmap_object(
    mat = mat,
    heatmap_name = heatmap_name,
    palette_mode = palette_mode,
    show_row_names = show_row_names
  )

  pdf(file.path(outdir, paste0(pdf_stub, ".pdf")), width = 9.5, height = 14)
  draw(ht)
  dev.off()

  png(file.path(outdir, paste0(pdf_stub, ".png")), width = 9.5, height = 14, units = "in", res = 300)
  draw(ht)
  dev.off()
}

scalability_dir <- file.path(plotdata_dir, "Scalability")

speed_path <- file.path(scalability_dir, "docker_speed_dim_means.csv")
memory_path <- file.path(scalability_dir, "docker_memory_dim_means.csv")

speed_mat <- NULL
memory_mat <- NULL
hvg_time_mat <- NULL
hvg_memory_mat <- NULL

if (file.exists(speed_path)) {
  speed_mat <- read_scalability_csv_matrix(speed_path)
  make_single_scalability_heatmap(
    speed_mat,
    "Speed (min)",
    "Fig3_scalability_speed_Rstyle",
    palette_mode = "green",
    show_row_names = TRUE
  )
}

if (file.exists(memory_path)) {
  memory_mat <- read_scalability_csv_matrix(memory_path)
  make_single_scalability_heatmap(
    memory_mat,
    "Memory (GB)",
    "Fig3_scalability_memory_Rstyle",
    palette_mode = "green",
    show_row_names = TRUE
  )
}

if (file.exists(hvg_2k5k_xlsx)) {
  hvg_time_mat <- make_hvg_2k5k_matrix(
    xlsx_path = hvg_2k5k_xlsx,
    sheet_name = "time",
    convert_time_to_min = TRUE
  )

  hvg_memory_mat <- make_hvg_2k5k_matrix(
    xlsx_path = hvg_2k5k_xlsx,
    sheet_name = "memory",
    convert_time_to_min = FALSE
  )

  write.csv(
    data.frame(Method = rownames(hvg_time_mat), hvg_time_mat, check.names = FALSE),
    file.path(scalability_dir, "HVG_2k5k_time_dim_means_parsed.csv"),
    row.names = FALSE
  )
  write.csv(
    data.frame(Method = rownames(hvg_memory_mat), hvg_memory_mat, check.names = FALSE),
    file.path(scalability_dir, "HVG_2k5k_memory_dim_means_parsed.csv"),
    row.names = FALSE
  )
} else {
  message("HVG 2k/5k xlsx not found: ", hvg_2k5k_xlsx)
}

## Combined scalability figure:
## Speed + 2k/5k Time + Memory + 2k/5k Memory
ht_list <- NULL

if (!is.null(speed_mat)) {
  ht_speed <- make_scalability_heatmap_object(
    speed_mat,
    "Speed (min)",
    palette_mode = "green",
    show_row_names = TRUE
  )
  ht_list <- ht_speed
}

if (!is.null(hvg_time_mat)) {
  ht_hvg_time <- make_scalability_heatmap_object(
    hvg_time_mat,
    "2k/5k Time (min)",
    palette_mode = "salmon",
    show_row_names = FALSE
  )
  ht_list <- if (is.null(ht_list)) ht_hvg_time else ht_list + ht_hvg_time
}

if (!is.null(memory_mat)) {
  ht_memory <- make_scalability_heatmap_object(
    memory_mat,
    "Memory (GB)",
    palette_mode = "green",
    show_row_names = FALSE
  )
  ht_list <- if (is.null(ht_list)) ht_memory else ht_list + ht_memory
}

if (!is.null(hvg_memory_mat)) {
  ht_hvg_memory <- make_scalability_heatmap_object(
    hvg_memory_mat,
    "2k/5k Memory (GB)",
    palette_mode = "salmon",
    show_row_names = FALSE
  )
  ht_list <- if (is.null(ht_list)) ht_hvg_memory else ht_list + ht_hvg_memory
}

if (!is.null(ht_list)) {
  pdf(file.path(outdir, "Fig3_scalability_combined_with_2k5k_Rstyle.pdf"), width = 18, height = 14)
  draw(ht_list)
  dev.off()

  png(file.path(outdir, "Fig3_scalability_combined_with_2k5k_Rstyle.png"), width = 18, height = 14, units = "in", res = 300)
  draw(ht_list)
  dev.off()
}

message("Done. R-style Fig3 outputs written to: ", outdir)
