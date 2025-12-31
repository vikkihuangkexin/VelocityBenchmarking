# Count cell cycle cells
library(dplyr)

# Load data
file_path <- "/data_d/ZXL/RNA_Velocity/velocycle-main/test/phase_result/merged_all_datasets.csv"
df <- read.delim(file_path, sep = ",", stringsAsFactors = FALSE)

# True counts
true_counts <- df %>%
  group_by(dataset, milestone) %>%
  summarise(true_count = n(), .groups = "drop") %>%
  pivot_wider(id_cols = dataset, names_from = milestone, values_from = true_count, values_fill = 0) %>%
  rename_with(~paste0("true_", .x), -dataset)

# Pred counts
pred_counts <- df %>%
  group_by(dataset, predicted_phase) %>%
  summarise(pred_count = n(), .groups = "drop") %>%
  pivot_wider(id_cols = dataset, names_from = predicted_phase, values_from = pred_count, values_fill = 0) %>%
  rename_with(~paste0("pred_", .x), -dataset)

# Merge and save
result <- inner_join(true_counts, pred_counts, by = "dataset")
output_path <- "/data_d/ZXL/RNA_Velocity/velocycle-main/test/phase_result/cell_counts_by_dataset.csv"
write.table(result, file = output_path, sep = ",", row.names = FALSE, quote = FALSE)

cat("Count saved to: ", output_path, "\n")
print(head(result))

##################################################
# Plot cell cycle comparison (batch)
library(ggplot2)
library(dplyr)

# Data
data <- data.frame(
  dataset = c("13", "30", "31", "32", "33", 
              "cycle-simple_cell100000_gene1000", "cycle-simple_cell10000_gene1000", 
              "cycle-simple_cell10000_gene10000", "cycle-simple_cell10000_gene500", 
              "cycle-simple_cell1000_gene1000", "cycle-simple_cell1000_gene10000", 
              "cycle-simple_cell1000_gene100000", "cycle-simple_cell1000_gene500", 
              "cycle-simple_cell1000_gene50000", "cycle-simple_cell50000_gene1000", 
              "cycle-simple_cell500_gene1000", "cycle-simple_cell500_gene10000", 
              "cycle-simple_cell500_gene500", "all"),
  true_G1 = c(667, 1761, 2281, 954, 5367, 32781, 3306, 3388, 3225, 336, 358, 315, 332, 315, 1697.8, 174, 178, 163, 7287.9),
  true_G2M = c(936, 2605, 1519, 1250, 1979, 35292, 3362, 3308, 3398, 343, 332, 343, 331, 343, 1710.4, 166, 175, 172, 7295.8),
  true_S = c(1455, 908, 1567, 1389, 4324, 31927, 3332, 3304, 3377, 321, 310, 342, 337, 342, 1591.8, 160, 147, 165, 6962.5),
  pred_G1 = c(67, 1793, 2323, 771, 6023, 35693, 3287, 5804, 2649, 279, 364, 37, 369, 242, 1910.0, 182, 80, 130, 7919.3),
  pred_G2M = c(213, 2674, 1412, 1229, 1963, 22939, 3485, 1739, 3855, 357, 312, 769, 357, 216, 1179.1, 170, 362, 224, 5406.7),
  pred_S = c(2778, 807, 1632, 1593, 3684, 41368, 3228, 2457, 3496, 364, 324, 194, 274, 542, 1910.9, 148, 58, 146, 8220.2)
)

# Params
labels <- c("G2M", "S", "G1")
colors <- c("#d95f02", "#7570b3", "#1b9e77")
text_size <- 8
plot_width <- 10
plot_height <- 14
dpi <- 300

# Batch plot
for (i in 1:nrow(data)) {
  current_data <- data[i, ]
  dataset_name_raw <- as.character(current_data$dataset)
  dataset_name_clean <- gsub("[^a-zA-Z0-9_.-]", "_", dataset_name_raw)
  
  # Extract values
  true <- c(current_data$true_G2M, current_data$true_S, current_data$true_G1)
  pred <- c(current_data$pred_G2M, current_data$pred_S, current_data$pred_G1)
  
  # Stack positions
  y_pred <- c(0, cumsum(pred)[-length(pred)])
  y_true <- c(0, cumsum(true)[-length(true)])
  
  # Build polygon data
  df <- data.frame()
  for (j in 1:3) {
    polygon_df <- data.frame(
      x = c(0.5, 3.5, 3.5, 0.5),
      y = c(y_pred[j], y_true[j], y_true[j] + true[j], y_pred[j] + pred[j]),
      group = labels[j],
      fill = colors[j]
    )
    df <- rbind(df, polygon_df)
  }
  
  # Label positions
  label_df <- data.frame()
  for (j in 1:3) {
    y_min <- min(y_pred[j], y_true[j])
    y_max <- max(y_pred[j] + pred[j], y_true[j] + true[j])
    label_df <- rbind(label_df, data.frame(x = 2, y = (y_min + y_max)/2, label = labels[j]))
  }
  
  # Title labels
  max_total_y <- max(y_pred + pred, y_true + true)
  title_labels <- data.frame(x = c(0.5, 3.5), y = max_total_y * 1.1, label = c("DeepCycle", "Groundtruth"))
  
  # Aspect ratio
  ratio <- 0.0004 * (5000 / max_total_y)
  ratio <- max(ratio, 0.0001)
  
  # Plot
  p <- ggplot() +
    geom_polygon(data = df, aes(x = x, y = y, group = group, fill = fill), color = "black", alpha = 0.8) +
    scale_fill_identity(guide = "none") +
    geom_text(data = label_df, aes(x = x, y = y, label = label), size = text_size, fontface = "bold", color = "white") +
    geom_text(data = title_labels, aes(x = x, y = y, label = label), size = text_size, fontface = "bold", hjust = c(0, 1), vjust = 1.2) +
    coord_fixed(ratio = ratio) +
    theme_void() +
    theme(plot.margin = margin(80, 40, 20, 40))
  
  # Save
  png_filename <- paste0(dataset_name_clean, ".png")
  pdf_filename <- paste0(dataset_name_clean, ".pdf")
  ggsave(png_filename, p, width = plot_width, height = plot_height, dpi = dpi)
  ggsave(pdf_filename, p, width = plot_width, height = plot_height)
  
  cat("Generated: ", png_filename, " & ", pdf_filename, "\n")
}

# Summary
total_datasets <- nrow(data)
total_plots <- 2 * total_datasets
cat("All plots generated: ", total_datasets, " datasets, ", total_plots, " plots\n")