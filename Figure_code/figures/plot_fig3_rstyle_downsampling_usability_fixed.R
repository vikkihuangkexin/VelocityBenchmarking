#!/usr/bin/env Rscript

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

summarise_metric_csv <- function(csv_path, methods_filter = NULL) {
  dat <- read.csv(csv_path, check.names = FALSE, stringsAsFactors = FALSE, na.strings = c("", "NA", "NaN"))
  if (!"Method" %in% colnames(dat)) stop("Missing Method column: ", csv_path)

  if (!is.null(methods_filter)) {
    dat <- dat %>% filter(Method %in% methods_filter)
  }

  long <- dat %>%
    pivot_longer(-Method, names_to = "Sample", values_to = "Value") %>%
    mutate(Value = as.numeric(Value))

  sm <- long %>%
    group_by(Method) %>%
    summarise(
      mean_value = mean(Value, na.rm = TRUE),
      n_valid = sum(!is.na(Value)),
      se = ifelse(n_valid > 1, sd(Value, na.rm = TRUE) / sqrt(n_valid), NA_real_),
      .groups = "drop"
    ) %>%
    filter(n_valid > 0)

  if (nrow(sm) == 0) return(sm)

  current_order <- method_order[method_order %in% sm$Method]
  sm$Method <- factor(sm$Method, levels = rev(current_order))
  sm
}

make_bar_from_summary <- function(sm, title) {
  current_methods <- as.character(rev(levels(sm$Method)))
  current_colors <- method_colors[current_methods]
  ggplot(sm, aes(x = Method, y = mean_value, fill = Method)) +
    geom_bar(stat = "identity", color = "black") +
    geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se),
                  width = 0.2, color = "black", na.rm = TRUE) +
    scale_fill_manual(values = current_colors, drop = FALSE) +
    coord_flip() +
    labs(title = title) +
    theme_metric_bar
}

## Downsampling: only plot tools with data
stab_path <- file.path(plotdata_dir, "Stability", "Downsampling.csv")
if (file.exists(stab_path)) {
  sm_down <- summarise_metric_csv(stab_path)
  if (nrow(sm_down) > 0) {
    methods_down <- as.character(rev(levels(sm_down$Method)))
    p0 <- make_blank_method_panel(methods_down)
    p_down <- make_bar_from_summary(sm_down, "Downsampling")

    pdf(file.path(outdir, "Fig3_stability_downsampling_Rstyle.pdf"), width = 6, height = 12)
    grid.arrange(p0, p_down, nrow = 1)
    dev.off()

    png(file.path(outdir, "Fig3_stability_downsampling_Rstyle.png"), width = 6, height = 12, units = "in", res = 300)
    grid.arrange(p0, p_down, nrow = 1)
    dev.off()
  }
}

## Usability: exclude DeepCycle and VeloCycle
usa_path <- file.path(plotdata_dir, "Usability", "Velocity_Usability_Detailed_Subscore.csv")
if (file.exists(usa_path)) {
  usa <- read.csv(usa_path, check.names = FALSE, stringsAsFactors = FALSE, na.strings = c("", "NA", "NaN"))
  usa <- usa %>% filter(!Method %in% c("DeepCycle", "VeloCycle"))

  usa_long <- usa %>%
    pivot_longer(-Method, names_to = "Metric", values_to = "Score") %>%
    mutate(Score = as.numeric(Score)) %>%
    filter(!is.na(Score))

  if (nrow(usa_long) > 0) {
    methods_usa <- method_order[method_order %in% unique(usa_long$Method)]
    usa_long$Method <- factor(usa_long$Method, levels = rev(methods_usa))

    min_s <- min(usa_long$Score, na.rm = TRUE)
    usa_long <- usa_long %>% mutate(Size = (Score - min_s + 1)^2)

    p0 <- make_blank_method_panel(methods_usa)
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


    pdf(file.path(outdir, "Fig3_usability_bubble_Rstyle.pdf"), width = 8, height = 12)
    grid.arrange(p0, p_usa, nrow = 1, widths = c(1.4, 2.2))
    dev.off()

    png(file.path(outdir, "Fig3_usability_bubble_Rstyle.png"), width = 8, height = 12, units = "in", res = 300)
    grid.arrange(p0, p_usa, nrow = 1, widths = c(1.4, 2.2))
    dev.off()
  }
}

message("Patched downsampling/usability figures written to: ", outdir)
