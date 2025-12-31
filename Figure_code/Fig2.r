rm(list = ls())
library(ggplot2)
library(patchwork)

#Accuracy
acc_rank_df<-read.csv("C:/Users/jliu25/Desktop/velocity-based methods/Manuscript/Materials/PlotData/Benchmarking-RealData/all/Reversed_rank_all.csv")
acc_AVG_rank <- acc_rank_df[, c(1, 9), drop = FALSE]
acc_AVG_rank$Method <- factor(acc_AVG_rank$Method, levels = acc_AVG_rank$Method)

colors_21 <- c(
  "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00","#e82b91", "#6a3d9a",
  "#b15928", "#a6cee3", "#b2df8a", "#fb9a99", "grey",  
  "#cab2d6", "#ffff66", "#8dd3c7", "lightyellow", "#9970ab",                     
  "#fb8072", "#80b1d3", "#fdb468", "#b3de69", "#fccde5"
)

p0<-ggplot(acc_AVG_rank, aes(x = Method, y = AVG, fill = Method)) +
  geom_blank() +   # do not draw bars
  theme_minimal() +
  coord_flip() +
  labs(title = "") +
  theme(
    legend.position = "none",          # remove legend
    panel.background = element_blank(),# remove background
    panel.grid = element_blank(),      # remove gridlines
    panel.border = element_blank(),    # remove border
    axis.line = element_blank(),       # remove axis lines
    axis.text.x = element_blank(),     # remove x-axis tick labels
    axis.title.x = element_blank(),    # remove x-axis title
    axis.title.y = element_blank(),    # remove y-axis title(delete this line if you want to keep it)
    axis.text.y = element_text(size = 14, face = "bold")
  )


p_ACC<-ggplot(acc_AVG_rank, aes(x = Method, y = AVG, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +
  scale_fill_manual(values = colors_21) + 
  theme_minimal() +
  coord_flip() +
  labs(title = "Accuracy") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    #axis.text.y = element_text(size = 14, face = "bold"),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
  )



pdf("C:/Users/jliu25/Desktop/velocity-based methods/Manuscript/Materials/Figures/Figure2_ACC_reversed_rank.pdf",4,10)
p_ACC
dev.off()


####Usability
usa_rank_df<-read.csv("C:/Users/jliu25/Desktop/velocity-based methods/Manuscript/Materials/PlotData/Usability/Velocity_Usability1010.csv")
usa_AVG_rank <- usa_rank_df[, c(1, 7), drop = FALSE]
usa_AVG_rank$Method <- factor(usa_AVG_rank$Method, levels = acc_AVG_rank$Method)
colnames(usa_AVG_rank) <- c('Method', 'AVG')

p_usa<-ggplot(usa_AVG_rank, aes(x = Method, y = AVG, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +
  scale_fill_manual(values = colors_21) + 
  theme_minimal() +
  coord_flip() +
  labs(title = "Usability") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    #axis.text.y = element_text(size = 14, face = "bold"),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
  )

library(gridExtra)
pdf("C:/Users/jliu25/Desktop/velocity-based methods/Manuscript/Materials/Figures/Figure2_ACC_Usability.pdf",6,10)
grid.arrange(p0, p_ACC, p_usa, nrow = 1)
dev.off()



############After all metrics are available：########
library(readxl)
library(ggplot2)
library(gridExtra)

## 1. Load data ------------------------------------------------------------
df <- read_excel("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/bar for fig 2.xlsx")

## Rename columns for safety (following the order in your screenshot)
colnames(df) <- c("Method", "Overall", "Accuracy", "Scalability", "Stability", "Usability")

## 2. Sort by Overall (descending) and fix Method order -------------------------
df <- df[order(df$Overall), ] 
df$Method <- factor(df$Method, levels = df$Method)

## Colors
colors_21 <- c(
  "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00","#e82b91", "#6a3d9a",
  "#b15928", "#a6cee3", "#b2df8a", "#fb9a99", "grey",  
  "#cab2d6", "#ffff66", "#8dd3c7", "lightyellow", "#9970ab",                     
  "#fb8072", "#80b1d3", "#fdb468", "#b3de69", "#fccde5"
)

## 3. Blank plot for method labels (left-side tags) --------------------------------------
p0 <- ggplot(df, aes(x = Method, y = Overall)) +
  geom_blank() +
  coord_flip() +
  theme_minimal() +
  theme(
    legend.position   = "none",
    panel.background  = element_blank(),
    panel.grid        = element_blank(),
    axis.line         = element_blank(),
    axis.text.x       = element_blank(),
    axis.title.x      = element_blank(),
    axis.title.y      = element_blank(),
    axis.text.y       = element_text(size = 14, face = "bold")
  )

## 4. Wrap a helper function to draw bar plots -----------------------------------------
make_bar <- function(data, ycol, title) {
  ggplot(data, aes(x = Method, y = .data[[ycol]], fill = Method)) +
    geom_bar(stat = "identity", color = "black") +
    scale_fill_manual(values = colors_21) +
    coord_flip() +
    theme_minimal() +
    labs(title = title) +
    theme(
      legend.position   = "none",
      panel.background  = element_blank(),
      panel.grid        = element_blank(),
      axis.line         = element_blank(),
      axis.text         = element_blank(),
      axis.title.x      = element_blank(),
      axis.title.y      = element_blank(),
      plot.title        = element_text(hjust = 0.5, size = 14, face = "bold")
    )
}

## 5. Draw one bar plot per metric --------------------------------------------
p_overall <- make_bar(df, "Overall",      "Overall")
p_acc     <- make_bar(df, "Accuracy",     "Accuracy")
p_scal    <- make_bar(df, "Scalability",  "Scalability")
p_stab    <- make_bar(df, "Stability",    "Stability")
p_usa     <- make_bar(df, "Usability",    "Usability")

pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure2_Overall.pdf", 4, 10)
p_overall
dev.off()

pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure2_AllMetrics_gradient_color.pdf", 12, 10)
grid.arrange(p0, p_overall, p_acc, p_scal, p_stab, p_usa, nrow = 1)
dev.off()

###########Gradients######

library(readxl)
library(ggplot2)
library(gridExtra)

## 1. Read Excel ---------------------------------------------------------
df <- read_excel("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/bar for fig 2.xlsx")
colnames(df) <- c("Method", "Overall", "Accuracy", "Scalability", "Stability", "Usability")

## 2. Sort by Overall (ascending) so that after flipping the longest bar is on top -------------------
df <- df[order(df$Overall), ]
df$Method <- factor(df$Method, levels = df$Method)
n_methods <- nrow(df)

## 3. Generate 5 base gradient palettes (light -> dark) ------------------------------------
pal_overall     <- colorRampPalette(c("#deebf7", "#08519c"))(n_methods)  # Blue
pal_accuracy    <- colorRampPalette(c("#fee0d2", "#a50f15"))(n_methods)  # Red
pal_scalability <- colorRampPalette(c("#e5f5e0", "#006d2c"))(n_methods)  # Green
pal_stability   <- colorRampPalette(c("#f2f0f7", "#54278f"))(n_methods)  # Purple
pal_usability   <- colorRampPalette(c("#fff7bc", "#d95f0e"))(n_methods)  # Orange

## 4. Assign colors by metric magnitude: larger values use darker colors ----------------------
make_metric_colors <- function(df, metric, base_cols) {
  # df is already ordered by Overall，this order is used to arrange the bars
  ord_metric <- order(df[[metric]])      # metric order from low to high
  cols <- rep(NA_character_, nrow(df))
  cols[ord_metric] <- base_cols          # use light colors for low values and dark colors for high values
  names(cols) <- df$Method               # name to match scale_fill_manual
  cols
}

cols_overall     <- make_metric_colors(df, "Overall",     pal_overall)
cols_accuracy    <- make_metric_colors(df, "Accuracy",    pal_accuracy)
cols_scalability <- make_metric_colors(df, "Scalability", pal_scalability)
cols_stability   <- make_metric_colors(df, "Stability",   pal_stability)
cols_usability   <- make_metric_colors(df, "Usability",   pal_usability)

## 5. Blank plot showing method names only (left side) ---------------------------------------------
p0 <- ggplot(df, aes(x = Method, y = Overall)) +
  geom_blank() +
  coord_flip() +
  theme_minimal() +
  theme(
    legend.position   = "none",
    panel.background  = element_blank(),
    panel.grid        = element_blank(),
    axis.line         = element_blank(),
    axis.text.x       = element_blank(),
    axis.title.x      = element_blank(),
    axis.title.y      = element_blank(),
    axis.text.y       = element_text(size = 14, face = "bold")
  )

## 6. Unified bar-plot function --------------------------------------------------
make_bar <- function(data, ycol, title, fill_colors) {
  ggplot(data, aes(x = Method, y = .data[[ycol]], fill = Method)) +
    geom_bar(stat = "identity", color = "black") +
    scale_fill_manual(values = fill_colors) +
    coord_flip() +
    theme_minimal() +
    labs(title = title,
         x = NULL,          # or "Method"
         y = ycol) +        # if you want to fix it as "Score" set it to y = "Score"
    theme(
      legend.position   = "none",
      panel.background  = element_blank(),
      panel.grid        = element_blank(),
      axis.line         = element_blank(),
      axis.text.y       = element_text(size = 10, face = "bold"),  # show method names
      axis.text.x       = element_text(size = 9),                  # show numeric ticks
      axis.title.x      = element_blank(),  # keep these two lines if you do not want axis titles
      # axis.title.y   = element_blank(),
      plot.title        = element_text(hjust = 0.5, size = 14, face = "bold")
    )
}

## 7. Plot each metric --------------------------------------------------------
p_overall <- make_bar(df, "Overall",     "Overall",     cols_overall)
p_acc     <- make_bar(df, "Accuracy",    "Accuracy",    cols_accuracy)
p_scal    <- make_bar(df, "Scalability", "Scalability", cols_scalability)
p_stab    <- make_bar(df, "Stability",   "Stability",   cols_stability)
p_usa     <- make_bar(df, "Usability",   "Usability",   cols_usability)

## 8. Export single plots (if needed) -----------------------------------------------
pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure2_Overall.pdf", 4, 10); p_overall; dev.off()
pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure2_Accuracy.pdf", 4, 10); p_acc; dev.off()
pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure2_Scalability.pdf", 4, 10); p_scal; dev.off()
pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure2_Stability.pdf", 4, 10); p_stab; dev.off()
pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure2_Usability.pdf", 4, 10); p_usa; dev.off()

## 9. Combine into one figure (method names on the left + 5 metrics) ------------------------------
pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure2_AllMetrics_gradient.pdf", 12, 10)
grid.arrange(p0, p_overall, p_acc, p_scal, p_stab, p_usa, nrow = 1)
dev.off()
