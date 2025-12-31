# Gene heatmap (31 DeepCycle)
library(pheatmap)
library(dplyr)
library(grid)

# Paths
expr_path <- "/data_d/Velocity/silver/exp/31_expression_matrix_fixed.csv"
phase_csv_path <- "/data_d/ZXL/RNA_Velocity/DeepCycle-main/result/phase_result/31_phase_deepcycle_phase.csv"
output_png <- "31_deepcycle_heatmap_with_phase.png"
output_pdf <- "31_deepcycle_heatmap_with_phase.pdf"

# Load data
expr_df <- read.csv(expr_path, row.names = 1, check.names = FALSE)
phase_df <- read.csv(phase_csv_path, stringsAsFactors = FALSE)

# Unify cell IDs
phase_df$cell_id <- trimws(phase_df$cell_id)
rownames(expr_df) <- trimws(rownames(expr_df))
common_cells <- intersect(rownames(expr_df), phase_df$cell_id)
message("Common cells: ", length(common_cells))
if (length(common_cells) == 0) stop("No matching cells")

# Filter genes and cells
target_genes <- c("ORC1","CCNE1","CCNE2","MCM6","WEE1","CDK1","CCNF","NUSAP1","AURKA","CCNA2","CCNB2")
existing_genes <- intersect(target_genes, colnames(expr_df))
if (length(existing_genes) == 0) stop("No matching genes")

# Filter low expression cells
cell_total <- rowSums(expr_df[common_cells, , drop = FALSE])
cell_target_mean <- rowMeans(expr_df[common_cells, existing_genes, drop = FALSE], na.rm = TRUE)
filtered_cells <- names(cell_total)[cell_total >= quantile(cell_total, 0.1) & cell_target_mean >= quantile(cell_target_mean, 0.1)]
message("Filtered cells: ", length(filtered_cells))

# Sort cells by phase
sorted_phase <- phase_df %>% filter(cell_id %in% filtered_cells) %>% arrange(cell_cycle_theta)
sorted_cells <- sorted_phase$cell_id

# Process expression matrix
expr_mat <- t(expr_df[sorted_cells, existing_genes, drop = FALSE])
expr_scaled <- scale(expr_mat)
expr_scaled[is.na(expr_scaled)] <- 0

# Heatmap params
heatmap_params <- list(
  mat = expr_scaled,
  color = colorRampPalette(c("blue", "white", "red"))(100),
  show_rownames = TRUE,
  show_colnames = FALSE,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  treeheight_row = 0,
  treeheight_col = 0,
  main = "Cell Cycle Genes (Sorted by DeepCycle Phase)",
  cellwidth = 0.15,
  cellheight = 12
)

# Save PNG
png(output_png, width = 16, height = 10 + length(existing_genes)*0.3, units = "in", res = 600)
p <- do.call(pheatmap, c(heatmap_params, list(plot = FALSE)))
grid.newpage()
print(p, newpage = FALSE)
grid.text("0", x = 0.185, y = 0.41, just = "left", gp = gpar(fontsize = 12, fontface = "bold"))
grid.text("1", x = 0.755, y = 0.41, just = "right", gp = gpar(fontsize = 12, fontface = "bold"))
grid.text("cell_cycle_theta", x = 0.47, y = 0.4, just = "center", gp = gpar(fontsize = 14, fontface = "bold"))
dev.off()

# Save PDF
pdf(output_pdf, width = 16, height = 10 + length(existing_genes)*0.3)
grid.newpage()
print(p, newpage = FALSE)
grid.text("0", x = 0.185, y = 0.41, just = "left", gp = gpar(fontsize = 12, fontface = "bold"))
grid.text("1", x = 0.755, y = 0.41, just = "right", gp = gpar(fontsize = 12, fontface = "bold"))
grid.text("cell_cycle_theta", x = 0.47, y = 0.4, just = "center", gp = gpar(fontsize = 14, fontface = "bold"))
dev.off()

message("Heatmap saved: ", output_png, " & ", output_pdf)