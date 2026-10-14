#########real dataset######
# ===== Configuration =====
csv_path <- "PlotData/accuracy/real/scRNA_peak_location.csv"            # Change to your CSV path
out_png  <- "PlotData/Figures/example_heatmap.png"
out_pdf  <- "PlotData/Figures/example_heatmap.pdf"

# If you already know the column name of the "method name" column, you can specify it directly (e.g. "Method").
# Otherwise keep NULL and the script will automatically pick the most suitable non-numeric column.
method_col <- "Method"

# ===== Dependencies =====
if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) install.packages("ComplexHeatmap")
if (!requireNamespace("circlize", quietly = TRUE)) install.packages("circlize")
library(ComplexHeatmap)
library(circlize)

# ===== Read and preprocess =====
df <- read.csv(csv_path, check.names = FALSE, stringsAsFactors = FALSE)

# Select the "method name" index column: prefer common names first, otherwise take the first non-numeric column
non_num_cols <- names(df)[!sapply(df, is.numeric)]
if (is.null(method_col)) {
  pref <- c("method","tool","name","method_name","tool_name")
  hit <- intersect(non_num_cols, names(df)[tolower(names(df)) %in% pref])
  if (length(hit) > 0) {
    method_col <- hit[1]
  } else if (length(non_num_cols) > 0) {
    method_col <- non_num_cols[1]
  }
}
if (!is.null(method_col) && method_col %in% names(df)) {
  rownames(df) <- df[[method_col]]
  df[[method_col]] <- NULL
}

# Find the Reversed_rank column and sort in descending order (skip if it does not exist)
rev_col <- names(df)[tolower(names(df)) == "reversed_rank"]
if (length(rev_col) == 0) {
  cand <- names(df)[grepl("reversed", tolower(names(df))) & grepl("rank", tolower(names(df)))]
  if (length(cand) > 0) rev_col <- cand[1]
}
if (length(rev_col) == 1) {
  df <- df[order(df[[rev_col]], decreasing = TRUE, na.last = TRUE), , drop = FALSE]
}

# Keep only numeric columns
num_df <- df[, sapply(df, is.numeric), drop = FALSE]

# Remove the ACG column (if present, case-insensitive)
avg_idx <- which(tolower(colnames(num_df)) == "avg")
if (length(avg_idx) > 0) num_df <- num_df[, -avg_idx, drop = FALSE]

# Remove the ranking column (if it is numeric and present in num_df)
if (length(rev_col) == 1 && rev_col %in% colnames(num_df)) {
  num_df <- num_df[, setdiff(colnames(num_df), rev_col), drop = FALSE]
}

# Set missing values to 0
#num_df[is.na(num_df)] <- 0

# Transpose: rows = data items (original columns), columns = methods (original rows, sorted by Reversed_rank)
mat <- t(as.matrix(num_df))
upper_q <- 0.99
# ===== HCL palette + quantile-enhanced contrast =====
vals <- as.vector(mat)
vmin <- min(mat, na.rm=TRUE); vmax <- max(mat, na.rm=TRUE); legend_breaks <- pretty(c(vmin, vmax), 5)
pos_vals <- vals[vals > 0]
vmax <- if (length(pos_vals) > 0) quantile(pos_vals, probs = upper_q, na.rm = TRUE) else max(vals, na.rm = TRUE)
if (!is.finite(vmax) || vmax <= 0) vmax <- max(vals, na.rm = TRUE)

# Generate continuous colors using the HCL palette (keeping your color scheme)
palette_name <- "Reds 3" 
hcl_col <- grDevices::hcl.colors(256, palette = palette_name, rev = TRUE)

# Use colorRamp2 to map the data range to HCL colors (uniform segments)
col_fun <- circlize::colorRamp2(seq(vmin, vmax, length.out = length(hcl_col)), hcl_col)

legend_breaks <- pretty(c(vmin, vmax), n = 5)

# ===== Plotting (column names on top, rotated 90°) =====

ht <- Heatmap(
  mat,
  name = "Inter_vs_intra_ratio for velocity tools",
  col  = col_fun,                 
  na_col = "grey90",              # ★ NA fill color (does not use the colorbar)
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  column_names_side = "top",
  column_names_rot  = 45,        
  column_names_centered = TRUE,
  row_names_gp     = gpar(fontsize = 8),
  column_names_gp  = gpar(fontsize = 8),
  
  # Use only white edges to form gaps between cells; do not draw black borders
  rect_gp = gpar(col = "white", lwd = 4),
  
  # Write text in cells: two decimals for non-NA; "NA" for NA
  cell_fun = function(j, i, x, y, w, h, fill) {
    val <- mat[i, j]
    if (is.na(val)) {
      grid.text("NA", x, y, gp = gpar(fontsize = 6, col = "black"))
    } else {
      txt_col <- ifelse(val > (vmin + vmax)/2, "white", "black")
      grid.text(sprintf("%.2f", val), x, y, gp = gpar(fontsize = 6, col = txt_col))
    }
  },
  
  heatmap_legend_param = list(
    at = legend_breaks,
    labels = sprintf("%.2f", legend_breaks),
    title = "Inter_vs_intra_ratio value",
    legend_direction = "vertical",
    title_position = "leftcenter-rot"
    # If you only want to reverse the colorbar color order (without changing the mapping), add reverse = TRUE in newer versions
  ),
  
  use_raster = TRUE, raster_quality = 2
)
draw(ht)

png(out_png, width = 2400, height = 3800, res = 300)
draw(ht)
dev.off()

pdf(out_pdf, width = 9.5, height = 14)
draw(ht)
dev.off()


##############################################
##### Has black borders (this version is not used)
ht <- Heatmap(
  mat,
  name = "ICCoh for velocity tools",
  col  = col_fun,
  na_col = "grey85",                 # ★ NA fixed gray, does not use the colorbar
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  column_names_side = "top",
  column_names_rot  = 45,
  column_names_centered = TRUE,
  row_names_gp     = gpar(fontsize = 8),
  column_names_gp  = gpar(fontsize = 8),
  
  cell_fun = function(j, i, x, y, w, h, fill) {
    val <- mat[i, j]
    # 1) First fill with a white background (forming white gaps between cells)
    grid.rect(x, y, width = w, height = h, gp = gpar(fill = "white", col = NA))
    
    shrink <- 0.90
    
    if (is.na(val)) {
      # 2a) NA: gray fill + black border, and write "NA" in the middle
      grid.rect(x, y, width = w*shrink, height = h*shrink,
                gp = gpar(fill = "grey90", col = "#ffffff", lwd = 0.6))
      grid.text("NA", x, y, gp = gpar(fontsize = 8, col = "black", fontface = "bold"))
    } else {
      # 2b) Non-NA: fill with the mapped color + black border, and write two decimals
      grid.rect(x, y, width = w*shrink, height = h*shrink,
                gp = gpar(fill = fill, col = "black", lwd = 0.6))
      mid <- (vmin + vmax)/2
      txt_col <- ifelse(val > mid, "white", "black")
      grid.text(sprintf("%.2f", val), x, y, gp = gpar(fontsize = 6, col = txt_col))
    }
  },
  
  heatmap_legend_param = list(
    at = legend_breaks,
    labels = sprintf("%.2f", legend_breaks),
    title = "ICCoh Value",
    legend_direction = "vertical",
    title_position = "leftcenter-rot"
  ),
  
  use_raster = TRUE, raster_quality = 2
)

draw(ht)

############################################################################################
########simulate dataset#####
# ===== Configuration =====
csv_path <- "PlotData/accuracy/sim/SIM_angle_consistency.csv"            # Change to your CSV path
out_png  <- "PlotData/Figures/sim_velocity_angle_heatmap.png"
out_pdf  <- "PlotData/Figures/sim_velocity_angle_heatmap.pdf"

# If you already know the column name of the "method name" column, you can specify it directly (e.g. "Method").
# Otherwise keep NULL and the script will automatically pick the most suitable non-numeric column.
method_col <- "Method"

# ===== Dependencies =====
if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) install.packages("ComplexHeatmap")
if (!requireNamespace("circlize", quietly = TRUE)) install.packages("circlize")
library(ComplexHeatmap)
library(circlize)

# ===== Read and preprocess =====
df <- read.csv(csv_path, check.names = FALSE, stringsAsFactors = FALSE)

# Select the "method name" index column: prefer common names first, otherwise take the first non-numeric column
non_num_cols <- names(df)[!sapply(df, is.numeric)]
if (is.null(method_col)) {
  pref <- c("method","tool","name","method_name","tool_name")
  hit <- intersect(non_num_cols, names(df)[tolower(names(df)) %in% pref])
  if (length(hit) > 0) {
    method_col <- hit[1]
  } else if (length(non_num_cols) > 0) {
    method_col <- non_num_cols[1]
  }
}
if (!is.null(method_col) && method_col %in% names(df)) {
  rownames(df) <- df[[method_col]]
  df[[method_col]] <- NULL
}

# Find the Reversed_rank column and sort in descending order (skip if it does not exist)
rev_col <- names(df)[tolower(names(df)) == "reversed_rank"]
if (length(rev_col) == 0) {
  cand <- names(df)[grepl("reversed", tolower(names(df))) & grepl("rank", tolower(names(df)))]
  if (length(cand) > 0) rev_col <- cand[1]
}
if (length(rev_col) == 1) {
  df <- df[order(df[[rev_col]], decreasing = TRUE, na.last = TRUE), , drop = FALSE]
}

# Keep only numeric columns
num_df <- df[, sapply(df, is.numeric), drop = FALSE]

# Remove the ACG column (if present, case-insensitive)
avg_idx <- which(tolower(colnames(num_df)) == "avg")
if (length(avg_idx) > 0) num_df <- num_df[, -avg_idx, drop = FALSE]

# Remove the ranking column (if it is numeric and present in num_df)
if (length(rev_col) == 1 && rev_col %in% colnames(num_df)) {
  num_df <- num_df[, setdiff(colnames(num_df), rev_col), drop = FALSE]
}

# Set missing values to 0
#num_df[is.na(num_df)] <- 0

# Transpose: rows = data items (original columns), columns = methods (original rows, sorted by Reversed_rank)
mat <- t(as.matrix(num_df))
upper_q <- 0.99
# ===== HCL palette + quantile-enhanced contrast =====
vals <- as.vector(mat)
vmin <- min(mat, na.rm=TRUE); vmax <- max(mat, na.rm=TRUE); legend_breaks <- pretty(c(vmin, vmax), 5)
pos_vals <- vals[vals > 0]
vmax <- if (length(pos_vals) > 0) quantile(pos_vals, probs = upper_q, na.rm = TRUE) else max(vals, na.rm = TRUE)
if (!is.finite(vmax) || vmax <= 0) vmax <- max(vals, na.rm = TRUE)

# Generate continuous colors using the HCL palette (keeping your color scheme)
palette_name <- "Reds 3" 
hcl_col <- grDevices::hcl.colors(256, palette = palette_name, rev = TRUE)

# Use colorRamp2 to map the data range to HCL colors (uniform segments)
col_fun <- circlize::colorRamp2(seq(vmin, vmax, length.out = length(hcl_col)), hcl_col)

legend_breaks <- pretty(c(vmin, vmax), n = 5)

# ===== Plotting (column names on top, rotated 90°) =====

ht <- Heatmap(
  mat,
  name = "Velocity_Angle for velocity tools",
  col  = col_fun,                 
  na_col = "grey90",              # ★ NA fill color (does not use the colorbar)
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  column_names_side = "top",
  column_names_rot  = 45,        
  column_names_centered = TRUE,
  row_names_gp     = gpar(fontsize = 8),
  column_names_gp  = gpar(fontsize = 8),
  
  # Use only white edges to form gaps between cells; do not draw black borders
  rect_gp = gpar(col = "white", lwd = 4),
  
  # Write text in cells: two decimals for non-NA; "NA" for NA
  cell_fun = function(j, i, x, y, w, h, fill) {
    val <- mat[i, j]
    if (is.na(val)) {
      grid.text("NA", x, y, gp = gpar(fontsize = 6, col = "black"))
    } else {
      txt_col <- ifelse(val > (vmin + vmax)/2, "white", "black")
      grid.text(sprintf("%.2f", val), x, y, gp = gpar(fontsize = 6, col = txt_col))
    }
  },
  
  heatmap_legend_param = list(
    at = legend_breaks,
    labels = sprintf("%.2f", legend_breaks),
    title = "Velocity_Angle value",
    legend_direction = "vertical",
    title_position = "leftcenter-rot"
    # If you only want to reverse the colorbar color order (without changing the mapping), add reverse = TRUE in newer versions
  ),
  
  use_raster = TRUE, raster_quality = 2
)
draw(ht)

png(out_png, width = 2600, height = 4200, res = 300)
draw(ht)
dev.off()

pdf(out_pdf, width = 9.5, height = 20)
draw(ht)
dev.off()
