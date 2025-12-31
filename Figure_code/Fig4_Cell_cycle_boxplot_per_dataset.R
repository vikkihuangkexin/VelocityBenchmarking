# Cell cycle boxplot (per dataset)
library(ggplot2)
library(dplyr)

# Load data
data_path <- "/data_d/ZXL/RNA_Velocity/velocycle-main/test/phase_result/merged_all_datasets.csv"
df <- read.delim(data_path, sep = ",", stringsAsFactors = FALSE)

# Preprocess
df <- df %>%
  filter(milestone %in% c("G1", "G2M", "S")) %>%
  mutate(milestone = factor(milestone, levels = c("G1", "S", "G2M")))

# Output dir
output_dir <- "/data_d/ZXL/RNA_Velocity/velocycle-main/test/phase_result/1114plot/boxplot"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# Batch plot
unique_datasets <- unique(df$dataset)
for (ds in unique_datasets) {
  ds_data <- df %>% filter(dataset == ds)
  
  # Plot
  p <- ggplot(ds_data, aes(x = milestone, y = velocycle_phase, fill = milestone)) +
    geom_boxplot(width = 0.6, alpha = 0.7, color = "black", outlier.shape = NA) +
    geom_jitter(width = 0.2, alpha = 0.5, size = 1, color = "black") +
    labs(title = paste("Dataset:", ds), x = "Cell Cycle Phase (milestone)", y = "velocycle_phase Value", fill = "Phase") +
    scale_fill_manual(values = c("G1" = "#2ECC71", "S" = "#E67E22", "G2M" = "#9B59B6")) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 10),
      legend.position = "bottom",
      legend.key.width = unit(1.5, "cm")
    ) +
    guides(fill = guide_legend(title = "Cell Cycle Phase"))
  
  # Save
  output_path <- file.path(output_dir, paste0(ds, "_velocycle_boxplot.png"))
  ggsave(output_path, p, width = 8, height = 6, dpi = 300)
  cat("Saved: ", output_path, "\n")
}

cat("All boxplots saved to: ", output_dir, "\n")