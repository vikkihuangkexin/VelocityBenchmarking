# Metric boxplot
library(ggplot2)
library(dplyr)
library(tidyr)
library(rstatix)

# Load and combine data
velo <- read.csv("/data_d/ZXL/RNA_Velocity/velocycle-main/test/phase_result/dataset_metrics.csv")
velo$Tool <- "VeloCycle"
deep <- read.csv("/data_d/ZXL/RNA_Velocity/DeepCycle-main/result/phase_result/dataset_metrics.csv")
deep$Tool <- "DeepCycle"
combined <- rbind(velo, deep) %>%
  pivot_longer(cols = -c(dataset, Tool), names_to = "metric", values_to = "value")

# Params
metrics <- c("ari", "nmi", "ri", "accuracy", "precision", "recall", "fscore")
colors <- c("DeepCycle" = "#4682B4", "VeloCycle" = "#5A2E83")

# Statistical test
stat_test <- combined %>%
  group_by(metric) %>%
  wilcox_test(value ~ Tool, paired = TRUE, exact = FALSE) %>%
  adjust_pvalue(method = "fdr") %>%
  mutate(p_label = sprintf("p = %.3f", p))

# Annotation position
annotation <- combined %>%
  group_by(metric) %>%
  summarise(max_val = max(value) + 0.08) %>%
  left_join(stat_test %>% select(metric, p_label), by = "metric")

# Plot
p <- ggplot(combined, aes(x = factor(metric, levels = metrics), y = value, fill = Tool)) +
  geom_boxplot(
    width = 0.7, color = "black", linewidth = 0.7,
    position = position_dodge(width = 0.8),
    outlier.size = 1.2, outlier.alpha = 0.5, outlier.color = "black"
  ) +
  geom_text(
    data = annotation,
    aes(x = metric, y = max_val, label = p_label),
    size = 3.6, position = position_dodge(width = 0.8),
    inherit.aes = FALSE, fontface = "italic", color = "#333333"
  ) +
  scale_fill_manual(values = colors) +
  ylim(0, 1.18) +
  labs(
    x = "Evaluation Metrics (Higher = Better)",
    y = "Metric Value",
    title = "DeepCycle vs VeloCycle Metric Comparison"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5, margin = margin(b = 20), color = "#222222"),
    axis.title.x = element_text(size = 12, margin = margin(t = 12), color = "#222222"),
    axis.title.y = element_text(size = 12, margin = margin(r = 12), color = "#222222"),
    axis.text.x = element_text(size = 10, face = "bold", color = "#222222"),
    axis.text.y = element_text(size = 10, color = "#222222"),
    legend.position = "top", legend.justification = "center",
    legend.title = element_blank(), legend.text = element_text(size = 10, color = "#222222"),
    legend.key = element_blank(), legend.key.width = unit(1.5, "cm"),
    axis.line = element_line(linewidth = 0.7, color = "#222222")
  ) +
  guides(fill = guide_legend(nrow = 1))

# Save plot
ggsave("DeepCycle_VeloCycle_metric_comparison.png", p, dpi = 600, width = 12, height = 6.5, units = "in", bg = "white")
ggsave("DeepCycle_VeloCycle_metric_comparison.pdf", p, width = 12, height = 6.5, units = "in", device = cairo_pdf)

# Print stats
cat("Wilcoxon paired test results (FDR adjusted)\n")
print(stat_test %>% select(metric, statistic, p, p.adj) %>% arrange(metric))