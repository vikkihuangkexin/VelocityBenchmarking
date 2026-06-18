#!/usr/bin/env Rscript

## Final overall ranking bar plots
## Style follows the provided Fig2.r:
## - horizontal bars
## - black bar borders
## - metric-specific gradient colors
## - clean/minimal theme
## - combined panel with method labels on the left
##
## Input:
##   PlotData/Results/reversed_rank/Results/final/final_overall_rank_for_plot.csv
##
## Outputs:
##   PlotData/Results/reversed_rank/Results/final/figures/
##
## Required packages:
##   ggplot2, gridExtra

suppressPackageStartupMessages({
  library(ggplot2)
  library(gridExtra)
})

## -----------------------------
## 1. Paths
## -----------------------------
input_csv <- "PlotData/Results/reversed_rank/Results/final/final_overall_rank_for_plot.csv"
outdir <- "PlotData/Results/reversed_rank/Results/final/figures"

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

## -----------------------------
## 2. Load and check data
## -----------------------------
df <- read.csv(input_csv, check.names = FALSE, stringsAsFactors = FALSE)

required_cols <- c(
  "method",
  "final_overall_rank",
  "R_accuracy",
  "R_scalability",
  "R_stability",
  "R_usability"
)

missing_cols <- setdiff(required_cols, colnames(df))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

## Keep only required columns and coerce numeric values
df <- df[, required_cols]
colnames(df) <- c(
  "Method",
  "Final_overall_rank",
  "Accuracy",
  "Scalability",
  "Stability",
  "Usability"
)

numeric_cols <- c("Final_overall_rank", "Accuracy", "Scalability", "Stability", "Usability")
for (cc in numeric_cols) {
  df[[cc]] <- as.numeric(df[[cc]])
}

## Sort by final rank.
## In ggplot + coord_flip, the last factor level appears at the top.
## Therefore, sorting from worst rank to best rank places the best method on top.
df <- df[order(df$Final_overall_rank, decreasing = TRUE, na.last = TRUE), ]

## For the Overall panel, use the final rank to define a "higher is better"
## plotting value. This keeps visual direction consistent with R_accuracy,
## R_scalability, R_stability, and R_usability, where larger reversed ranks
## indicate better performance.
##
## The rank itself is still shown in the source table and output CSV.
n_methods <- sum(!is.na(df$Final_overall_rank))
df$Overall <- n_methods + 1 - df$Final_overall_rank

## Fix method order for all panels
df$Method <- factor(df$Method, levels = df$Method)

## Save plotting table for reproducibility
write.csv(
  df,
  file.path(outdir, "final_overall_bar_plot_input.csv"),
  row.names = FALSE
)

## -----------------------------
## 3. Gradient palettes
## -----------------------------
n_methods <- nrow(df)

pal_overall     <- colorRampPalette(c("#deebf7", "#08519c"))(n_methods)  # Blue
pal_accuracy    <- colorRampPalette(c("#fee0d2", "#a50f15"))(n_methods)  # Red
pal_scalability <- colorRampPalette(c("#e5f5e0", "#006d2c"))(n_methods)  # Green
pal_stability   <- colorRampPalette(c("#f2f0f7", "#54278f"))(n_methods)  # Purple
pal_usability   <- colorRampPalette(c("#fff7bc", "#d95f0e"))(n_methods)  # Orange

make_metric_colors <- function(data, metric, base_cols) {
  values <- data[[metric]]
  cols <- rep("grey90", nrow(data))

  valid_idx <- which(!is.na(values))
  if (length(valid_idx) > 0) {
    ord_metric <- valid_idx[order(values[valid_idx])]  # low -> high
    gradient <- colorRampPalette(c(base_cols[1], base_cols[length(base_cols)]))(length(valid_idx))
    cols[ord_metric] <- gradient
  }

  names(cols) <- data$Method
  cols
}

cols_overall     <- make_metric_colors(df, "Overall",     pal_overall)
cols_accuracy    <- make_metric_colors(df, "Accuracy",    pal_accuracy)
cols_scalability <- make_metric_colors(df, "Scalability", pal_scalability)
cols_stability   <- make_metric_colors(df, "Stability",   pal_stability)
cols_usability   <- make_metric_colors(df, "Usability",   pal_usability)

## -----------------------------
## 4. Shared themes and helpers
## -----------------------------
theme_method_labels <- theme_minimal(base_size = 12) +
  theme(
    legend.position   = "none",
    panel.background  = element_blank(),
    panel.grid        = element_blank(),
    panel.border      = element_blank(),
    axis.line         = element_blank(),
    axis.text.x       = element_blank(),
    axis.ticks        = element_blank(),
    axis.title.x      = element_blank(),
    axis.title.y      = element_blank(),
    axis.text.y       = element_text(size = 14, face = "bold", color = "black")
  )

theme_metric_bar <- theme_minimal(base_size = 12) +
  theme(
    legend.position   = "none",
    panel.background  = element_blank(),
    panel.grid        = element_blank(),
    panel.border      = element_blank(),
    axis.line         = element_blank(),
    axis.text         = element_blank(),
    axis.ticks        = element_blank(),
    axis.title.x      = element_blank(),
    axis.title.y      = element_blank(),
    plot.title        = element_text(hjust = 0.5, size = 14, face = "bold")
  )

## Left-side method labels only
p_method <- ggplot(df, aes(x = Method, y = Overall)) +
  geom_blank() +
  coord_flip() +
  theme_method_labels

make_bar <- function(data, ycol, title, fill_colors) {
  ggplot(data, aes(x = Method, y = .data[[ycol]], fill = Method)) +
    geom_bar(stat = "identity", color = "black", linewidth = 0.25, na.rm = TRUE) +
    scale_fill_manual(values = fill_colors) +
    coord_flip() +
    labs(title = title, x = NULL, y = NULL) +
    theme_metric_bar
}

## Version for single-panel PDF/PNG, with method names shown
theme_metric_bar_with_labels <- theme_minimal(base_size = 12) +
  theme(
    legend.position   = "none",
    panel.background  = element_blank(),
    panel.grid        = element_blank(),
    panel.border      = element_blank(),
    axis.line         = element_blank(),
    axis.text.x       = element_text(size = 9, color = "black"),
    axis.text.y       = element_text(size = 10, face = "bold", color = "black"),
    axis.ticks        = element_blank(),
    axis.title.x      = element_blank(),
    axis.title.y      = element_blank(),
    plot.title        = element_text(hjust = 0.5, size = 14, face = "bold")
  )

make_bar_single <- function(data, ycol, title, fill_colors) {
  ggplot(data, aes(x = Method, y = .data[[ycol]], fill = Method)) +
    geom_bar(stat = "identity", color = "black", linewidth = 0.25, na.rm = TRUE) +
    scale_fill_manual(values = fill_colors) +
    coord_flip() +
    labs(title = title, x = NULL, y = NULL) +
    theme_metric_bar_with_labels
}

## -----------------------------
## 5. Build plots
## -----------------------------
p_overall <- make_bar(df, "Overall",     "Overall",     cols_overall)
p_acc     <- make_bar(df, "Accuracy",    "Accuracy",    cols_accuracy)
p_scal    <- make_bar(df, "Scalability", "Scalability", cols_scalability)
p_stab    <- make_bar(df, "Stability",   "Stability",   cols_stability)
p_usa     <- make_bar(df, "Usability",   "Usability",   cols_usability)

p_overall_single <- make_bar_single(df, "Overall",     "Overall",     cols_overall)
p_acc_single     <- make_bar_single(df, "Accuracy",    "Accuracy",    cols_accuracy)
p_scal_single    <- make_bar_single(df, "Scalability", "Scalability", cols_scalability)
p_stab_single    <- make_bar_single(df, "Stability",   "Stability",   cols_stability)
p_usa_single     <- make_bar_single(df, "Usability",   "Usability",   cols_usability)

## -----------------------------
## 6. Export combined figure
## -----------------------------
pdf(file.path(outdir, "Figure2_Final_AllMetrics_gradient.pdf"), width = 12, height = 10)
grid.arrange(
  p_method,
  p_overall,
  p_acc,
  p_scal,
  p_stab,
  p_usa,
  nrow = 1,
  widths = c(1.85, 1, 1, 1, 1, 1)
)
dev.off()

png(file.path(outdir, "Figure2_Final_AllMetrics_gradient.png"), width = 12, height = 10, units = "in", res = 300)
grid.arrange(
  p_method,
  p_overall,
  p_acc,
  p_scal,
  p_stab,
  p_usa,
  nrow = 1,
  widths = c(1.85, 1, 1, 1, 1, 1)
)
dev.off()

## -----------------------------
## 7. Export individual panels
## -----------------------------
save_single <- function(plot_obj, filename_stub) {
  pdf(file.path(outdir, paste0(filename_stub, ".pdf")), width = 4, height = 10)
  print(plot_obj)
  dev.off()

  png(file.path(outdir, paste0(filename_stub, ".png")), width = 4, height = 10, units = "in", res = 300)
  print(plot_obj)
  dev.off()
}

save_single(p_overall_single, "Figure2_Final_Overall")
save_single(p_acc_single,     "Figure2_Final_Accuracy")
save_single(p_scal_single,    "Figure2_Final_Scalability")
save_single(p_stab_single,    "Figure2_Final_Stability")
save_single(p_usa_single,     "Figure2_Final_Usability")

message("Done.")
message("Figures saved to: ", outdir)
message("Combined figure: ", file.path(outdir, "Figure2_Final_AllMetrics_gradient.pdf"))
