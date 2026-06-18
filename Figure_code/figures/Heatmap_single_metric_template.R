#########real dataset######
# ===== 配置 =====
csv_path <- "PlotData/accuracy/real/scRNA_peak_location.csv"            # 修改为你的CSV路径
out_png  <- "PlotData/Figures/example_heatmap.png"
out_pdf  <- "PlotData/Figures/example_heatmap.pdf"

# 如果你已经知道“方法名称”那一列的列名，可直接指定（例如 "Method"）。
# 否则保持 NULL，脚本会自动从非数值列里选一个最合适的。
method_col <- "Method"

# ===== 依赖 =====
if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) install.packages("ComplexHeatmap")
if (!requireNamespace("circlize", quietly = TRUE)) install.packages("circlize")
library(ComplexHeatmap)
library(circlize)

# ===== 读取与预处理 =====
df <- read.csv(csv_path, check.names = FALSE, stringsAsFactors = FALSE)

# 选“方法名称”索引列：优先匹配常见命名，其次取第一个非数值列
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

# 找到 Reversed_rank 列并按降序排序（不存在则跳过）
rev_col <- names(df)[tolower(names(df)) == "reversed_rank"]
if (length(rev_col) == 0) {
  cand <- names(df)[grepl("reversed", tolower(names(df))) & grepl("rank", tolower(names(df)))]
  if (length(cand) > 0) rev_col <- cand[1]
}
if (length(rev_col) == 1) {
  df <- df[order(df[[rev_col]], decreasing = TRUE, na.last = TRUE), , drop = FALSE]
}

# 仅保留数值列
num_df <- df[, sapply(df, is.numeric), drop = FALSE]

# 删除 ACG 列（若存在，大小写无关）
avg_idx <- which(tolower(colnames(num_df)) == "avg")
if (length(avg_idx) > 0) num_df <- num_df[, -avg_idx, drop = FALSE]

# 删除排序列（如果它是数值列且存在于 num_df 中）
if (length(rev_col) == 1 && rev_col %in% colnames(num_df)) {
  num_df <- num_df[, setdiff(colnames(num_df), rev_col), drop = FALSE]
}

# 缺失值置 0
#num_df[is.na(num_df)] <- 0

# 转置：行=数据项（原列），列=方法（原行，已按 Reversed_rank 排序）
mat <- t(as.matrix(num_df))
upper_q <- 0.99
# ===== HCL 调色 + 分位数增强对比 =====
vals <- as.vector(mat)
vmin <- min(mat, na.rm=TRUE); vmax <- max(mat, na.rm=TRUE); legend_breaks <- pretty(c(vmin, vmax), 5)
pos_vals <- vals[vals > 0]
vmax <- if (length(pos_vals) > 0) quantile(pos_vals, probs = upper_q, na.rm = TRUE) else max(vals, na.rm = TRUE)
if (!is.finite(vmax) || vmax <= 0) vmax <- max(vals, na.rm = TRUE)

# 用 HCL 调色盘生成连续色（保持你要的配色体系）
palette_name <- "Reds 3" 
hcl_col <- grDevices::hcl.colors(256, palette = palette_name, rev = TRUE)

# 用 colorRamp2 把数据范围映射到 HCL 颜色（均匀分段）
col_fun <- circlize::colorRamp2(seq(vmin, vmax, length.out = length(hcl_col)), hcl_col)

legend_breaks <- pretty(c(vmin, vmax), n = 5)

# ===== 作图（列名顶部、旋转 90°） =====

ht <- Heatmap(
  mat,
  name = "Inter_vs_intra_ratio for velocity tools",
  col  = col_fun,                 
  na_col = "grey90",              # ★ NA 的填充颜色（不走 colorbar）
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  column_names_side = "top",
  column_names_rot  = 45,        
  column_names_centered = TRUE,
  row_names_gp     = gpar(fontsize = 8),
  column_names_gp  = gpar(fontsize = 8),
  
  # 仅用白色边线来形成格子间的空隙；不画黑色边框
  rect_gp = gpar(col = "white", lwd = 4),
  
  # 在格子中写文字：非NA写两位小数；NA写"NA"
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
    # 如果只想把 colorbar 颜色顺序倒过来（不改映射），新版本可加 reverse = TRUE
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
#####有黑色框（没用这一版）
ht <- Heatmap(
  mat,
  name = "ICCoh for velocity tools",
  col  = col_fun,
  na_col = "grey85",                 # ★ NA 固定灰色，不走 colorbar
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  column_names_side = "top",
  column_names_rot  = 45,
  column_names_centered = TRUE,
  row_names_gp     = gpar(fontsize = 8),
  column_names_gp  = gpar(fontsize = 8),
  
  cell_fun = function(j, i, x, y, w, h, fill) {
    val <- mat[i, j]
    # 1) 先铺满白底（形成格子间白缝）
    grid.rect(x, y, width = w, height = h, gp = gpar(fill = "white", col = NA))
    
    shrink <- 0.90
    
    if (is.na(val)) {
      # 2a) NA：灰色填充 + 黑色边框，并在中间写 "NA"
      grid.rect(x, y, width = w*shrink, height = h*shrink,
                gp = gpar(fill = "grey90", col = "#ffffff", lwd = 0.6))
      grid.text("NA", x, y, gp = gpar(fontsize = 8, col = "black", fontface = "bold"))
    } else {
      # 2b) 非 NA：按映射色填充 + 黑边框，并写两位小数
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
# ===== 配置 =====
csv_path <- "PlotData/accuracy/sim/SIM_angle_consistency.csv"            # 修改为你的CSV路径
out_png  <- "PlotData/Figures/sim_velocity_angle_heatmap.png"
out_pdf  <- "PlotData/Figures/sim_velocity_angle_heatmap.pdf"

# 如果你已经知道“方法名称”那一列的列名，可直接指定（例如 "Method"）。
# 否则保持 NULL，脚本会自动从非数值列里选一个最合适的。
method_col <- "Method"

# ===== 依赖 =====
if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) install.packages("ComplexHeatmap")
if (!requireNamespace("circlize", quietly = TRUE)) install.packages("circlize")
library(ComplexHeatmap)
library(circlize)

# ===== 读取与预处理 =====
df <- read.csv(csv_path, check.names = FALSE, stringsAsFactors = FALSE)

# 选“方法名称”索引列：优先匹配常见命名，其次取第一个非数值列
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

# 找到 Reversed_rank 列并按降序排序（不存在则跳过）
rev_col <- names(df)[tolower(names(df)) == "reversed_rank"]
if (length(rev_col) == 0) {
  cand <- names(df)[grepl("reversed", tolower(names(df))) & grepl("rank", tolower(names(df)))]
  if (length(cand) > 0) rev_col <- cand[1]
}
if (length(rev_col) == 1) {
  df <- df[order(df[[rev_col]], decreasing = TRUE, na.last = TRUE), , drop = FALSE]
}

# 仅保留数值列
num_df <- df[, sapply(df, is.numeric), drop = FALSE]

# 删除 ACG 列（若存在，大小写无关）
avg_idx <- which(tolower(colnames(num_df)) == "avg")
if (length(avg_idx) > 0) num_df <- num_df[, -avg_idx, drop = FALSE]

# 删除排序列（如果它是数值列且存在于 num_df 中）
if (length(rev_col) == 1 && rev_col %in% colnames(num_df)) {
  num_df <- num_df[, setdiff(colnames(num_df), rev_col), drop = FALSE]
}

# 缺失值置 0
#num_df[is.na(num_df)] <- 0

# 转置：行=数据项（原列），列=方法（原行，已按 Reversed_rank 排序）
mat <- t(as.matrix(num_df))
upper_q <- 0.99
# ===== HCL 调色 + 分位数增强对比 =====
vals <- as.vector(mat)
vmin <- min(mat, na.rm=TRUE); vmax <- max(mat, na.rm=TRUE); legend_breaks <- pretty(c(vmin, vmax), 5)
pos_vals <- vals[vals > 0]
vmax <- if (length(pos_vals) > 0) quantile(pos_vals, probs = upper_q, na.rm = TRUE) else max(vals, na.rm = TRUE)
if (!is.finite(vmax) || vmax <= 0) vmax <- max(vals, na.rm = TRUE)

# 用 HCL 调色盘生成连续色（保持你要的配色体系）
palette_name <- "Reds 3" 
hcl_col <- grDevices::hcl.colors(256, palette = palette_name, rev = TRUE)

# 用 colorRamp2 把数据范围映射到 HCL 颜色（均匀分段）
col_fun <- circlize::colorRamp2(seq(vmin, vmax, length.out = length(hcl_col)), hcl_col)

legend_breaks <- pretty(c(vmin, vmax), n = 5)

# ===== 作图（列名顶部、旋转 90°） =====

ht <- Heatmap(
  mat,
  name = "Velocity_Angle for velocity tools",
  col  = col_fun,                 
  na_col = "grey90",              # ★ NA 的填充颜色（不走 colorbar）
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  column_names_side = "top",
  column_names_rot  = 45,        
  column_names_centered = TRUE,
  row_names_gp     = gpar(fontsize = 8),
  column_names_gp  = gpar(fontsize = 8),
  
  # 仅用白色边线来形成格子间的空隙；不画黑色边框
  rect_gp = gpar(col = "white", lwd = 4),
  
  # 在格子中写文字：非NA写两位小数；NA写"NA"
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
    # 如果只想把 colorbar 颜色顺序倒过来（不改映射），新版本可加 reverse = TRUE
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
