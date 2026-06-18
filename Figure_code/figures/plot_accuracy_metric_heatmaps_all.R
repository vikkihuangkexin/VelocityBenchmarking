#!/usr/bin/env Rscript

###############################################################################
# Accuracy metric heatmaps for every real/sim metric CSV
#
# Input:
#   PlotData/accuracy/real
#   PlotData/accuracy/sim
#
# Output:
#   PlotData/Results/reversed_rank/plot/Accuracy_metric_heatmaps
#
# Main requirements:
#   1. Draw one heatmap for every metric CSV.
#   2. Real and simulated metrics are handled separately.
#   3. Methods are sorted alphabetically, not by rank.
#   4. Region Velocity is always included. If a metric has no Region Velocity
#      values, its column is shown as grey NA.
#   5. Missing values are shown as grey cells with "NA".
#   6. By default, numeric 0 is treated as missing because many accuracy metric
#      files use 0 as a placeholder for failed / unavailable calculations.
###############################################################################

suppressPackageStartupMessages({
  if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
    BiocManager::install("ComplexHeatmap", ask = FALSE, update = FALSE)
  }
  if (!requireNamespace("circlize", quietly = TRUE)) install.packages("circlize")
  if (!requireNamespace("grid", quietly = TRUE)) install.packages("grid")

  library(ComplexHeatmap)
  library(circlize)
  library(grid)
})

# =========================
# 1. User configuration
# =========================

input_root <- "PlotData/accuracy"
real_dir   <- file.path(input_root, "real")
sim_dir    <- file.path(input_root, "sim")

output_root <- "PlotData/Results/reversed_rank/plot/Accuracy_metric_heatmaps"
real_outdir <- file.path(output_root, "real")
sim_outdir  <- file.path(output_root, "sim")

dir.create(real_outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(sim_outdir,  recursive = TRUE, showWarnings = FALSE)

# If TRUE, numeric 0 values are treated as missing and displayed as NA,
# except for explicitly calculated true-zero cells defined in is_true_zero().
treat_zero_as_na <- TRUE

# If TRUE, use the union of all methods across real and sim files, plus
# Region Velocity, for every heatmap. This makes all heatmaps comparable.
use_global_method_union <- TRUE

# Always force Region Velocity to be present.
forced_methods <- c("PhyloVelo", "Region Velocity")

# Palette and contrast parameters, following the uploaded Heatmap.R style.
palette_name <- "Reds 3"
upper_q <- 0.99
na_fill <- "grey90"
white_grid_lwd <- 4

# Text sizes
cell_text_size <- 6
row_name_size <- 8
column_name_size <- 8

# Output resolution
png_res <- 300


# =========================
# 2. Helper functions
# =========================

normalize_method <- function(x) {
  x <- as.character(x)
  x <- trimws(gsub("\\s+", " ", x))
  key <- tolower(gsub("[^A-Za-z0-9]+", "", x))

  x[key %in% c("regionvelocity", "regionvelo")] <- "Region Velocity"
  x[key %in% c("scrnakinetics", "scrnakinetic")] <- "scRNAkinetics"
  x[key == "scvelodynamic"] <- "scVelo dynamical"
  x[key == "scvelostochastic"] <- "scVelo stochastic"
  x[key == "pyrovelocity"] <- "Pyro-Velocity"
  x[key == "topovelo"] <- "TopoVelo"

  x
}


method_sort <- function(x) {
  x <- unique(normalize_method(x))
  x <- x[!is.na(x) & x != ""]
  x <- sort(x, method = "radix")
  x
}


find_method_col <- function(df, preferred = NULL) {
  if (!is.null(preferred) && preferred %in% colnames(df)) {
    return(preferred)
  }

  lower_names <- tolower(colnames(df))
  preferred_names <- c("method", "tool", "tools", "name", "method_name", "tool_name")

  hit <- which(lower_names %in% preferred_names)
  if (length(hit) > 0) return(colnames(df)[hit[1]])

  non_num_cols <- colnames(df)[!vapply(df, is.numeric, logical(1))]
  if (length(non_num_cols) > 0) return(non_num_cols[1])

  # Fallback: first column
  colnames(df)[1]
}


is_metadata_col <- function(x) {
  lx <- tolower(x)
  lx %in% c(
    "avg", "average", "mean", "median",
    "rank", "reversed_rank", "final_rank", "final_overall_rank",
    "score", "overall", "overall_score"
  ) ||
    grepl("reversed.*rank", lx) ||
    grepl("final.*rank", lx)
}


clean_metric_name <- function(path, data_type) {
  stem <- tools::file_path_sans_ext(basename(path))
  if (data_type == "real") stem <- sub("^scRNA_", "", stem, ignore.case = TRUE)
  if (data_type == "sim")  stem <- sub("^SIM_",   "", stem, ignore.case = TRUE)
  stem
}


safe_filename <- function(x) {
  gsub("[^A-Za-z0-9._-]+", "_", x)
}


extract_dataset_id <- function(x) {
  # Extract the leading numeric ID from dataset names such as
  # "54_xxx", "G_54_xxx", "54", or "4_Mm_visual_cortex".
  sx <- as.character(x)
  sx <- gsub("^G_", "", sx, ignore.case = TRUE)
  m <- regexpr("[0-9]+", sx)
  ifelse(m > 0, regmatches(sx, m), NA_character_)
}


is_angle_metric <- function(metric_name) {
  grepl("angle[_ .-]*consistency|velocity[_ .-]*angle|\\bangle\\b",
        metric_name, ignore.case = TRUE)
}


is_peak_location_metric <- function(metric_name) {
  grepl("peak[_ .-]*location|peaklocation",
        metric_name, ignore.case = TRUE)
}


is_true_zero <- function(metric_name, method, dataset_name) {
  # Most zeros in the accuracy tables are placeholders for missing values.
  # Only the following explicitly calculated zeros are retained as real zeros:
  #
  # angle_consistency:
  #   cell2fate + dataset 54
  #   scRNAkinetics + dataset 54
  #   NeuroVelo + dataset 45
  #   VeloVAE + dataset 54
  #
  # peak_location:
  #   k-velo + dataset 13
  #   TopicVelo + datasets 4 and 31
  method <- normalize_method(method)
  ds_id <- extract_dataset_id(dataset_name)

  if (is_angle_metric(metric_name)) {
    return(
      (method == "cell2fate"      && ds_id == "54") ||
      (method == "scRNAkinetics"  && ds_id == "54") ||
      (method == "NeuroVelo"      && ds_id == "45") ||
      (method == "VeloVAE"        && ds_id == "54")
    )
  }

  if (is_peak_location_metric(metric_name)) {
    return(
      (method == "k-velo"    && ds_id == "13") ||
      (method == "TopicVelo" && ds_id %in% c("4", "31"))
    )
  }

  FALSE
}


apply_zero_to_na_rules <- function(num_df, metric_name) {
  # Convert zeros to NA except for the explicitly listed true-zero cells.
  if (!isTRUE(treat_zero_as_na)) return(num_df)

  if (nrow(num_df) == 0 || ncol(num_df) == 0) return(num_df)

  methods <- rownames(num_df)
  datasets <- colnames(num_df)

  for (ii in seq_along(methods)) {
    for (jj in seq_along(datasets)) {
      val <- num_df[ii, jj]
      if (!is.na(val) && isTRUE(val == 0)) {
        keep_zero <- is_true_zero(
          metric_name = metric_name,
          method = methods[ii],
          dataset_name = datasets[jj]
        )
        if (!isTRUE(keep_zero)) {
          num_df[ii, jj] <- NA_real_
        }
      }
    }
  }

  num_df
}


read_metric_matrix <- function(csv_path,
                               data_type,
                               target_methods = NULL,
                               method_col = NULL,
                               treat_zero = TRUE) {
  df <- read.csv(csv_path, check.names = FALSE, stringsAsFactors = FALSE)

  if (nrow(df) == 0) {
    stop("Empty CSV: ", csv_path)
  }

  method_col <- find_method_col(df, preferred = method_col)
  methods <- normalize_method(df[[method_col]])

  df[[method_col]] <- NULL

  # Keep numeric columns that are not metadata columns.
  numeric_cols <- colnames(df)[vapply(df, is.numeric, logical(1))]
  numeric_cols <- numeric_cols[!vapply(numeric_cols, is_metadata_col, logical(1))]

  # In case numeric columns were read as character because of NA strings or
  # mixed formatting, try to coerce non-metadata columns as well.
  if (length(numeric_cols) == 0) {
    candidate_cols <- colnames(df)[!vapply(colnames(df), is_metadata_col, logical(1))]
    for (cc in candidate_cols) {
      suppressWarnings(df[[cc]] <- as.numeric(df[[cc]]))
    }
    numeric_cols <- colnames(df)[vapply(df, is.numeric, logical(1))]
    numeric_cols <- numeric_cols[!vapply(numeric_cols, is_metadata_col, logical(1))]
  }

  num_df <- df[, numeric_cols, drop = FALSE]
  rownames(num_df) <- methods

  # Collapse duplicated method names by taking the mean per dataset column.
  # This avoids duplicate rowname problems after method-name canonicalization.
  if (any(duplicated(rownames(num_df)))) {
    num_df <- aggregate(
      num_df,
      by = list(method = rownames(num_df)),
      FUN = function(z) mean(z, na.rm = TRUE)
    )
    rownames(num_df) <- num_df$method
    num_df$method <- NULL
  }

  # Convert placeholder zeros to NA according to metric-specific rules.
  # Some cells are explicitly calculated zeros and are retained.
  metric_name <- clean_metric_name(csv_path, data_type)
  if (isTRUE(treat_zero)) {
    num_df <- apply_zero_to_na_rules(num_df, metric_name = metric_name)
  }

  # Determine final method set.
  if (is.null(target_methods)) {
    target_methods <- method_sort(c(rownames(num_df), forced_methods))
  } else {
    target_methods <- method_sort(c(target_methods, forced_methods))
  }

  # Reindex methods: absent methods become NA.
  missing_methods <- setdiff(target_methods, rownames(num_df))
  if (length(missing_methods) > 0) {
    missing_mat <- matrix(
      NA_real_,
      nrow = length(missing_methods),
      ncol = ncol(num_df),
      dimnames = list(missing_methods, colnames(num_df))
    )
    num_df <- rbind(num_df, missing_mat)
  }
  num_df <- num_df[target_methods, , drop = FALSE]

  # Alphabetical method order is now row order. Transpose for the requested
  # visual layout: rows = datasets/items; columns = methods.
  mat <- t(as.matrix(num_df))
  storage.mode(mat) <- "numeric"

  list(
    matrix = mat,
    methods = colnames(mat),
    datasets = rownames(mat),
    metric_name = metric_name
  )
}


collect_methods_from_dir <- function(dir_path) {
  files <- list.files(dir_path, pattern = "\\.csv$", full.names = TRUE)
  methods <- character(0)

  for (ff in files) {
    df <- tryCatch(
      read.csv(ff, check.names = FALSE, stringsAsFactors = FALSE),
      error = function(e) NULL
    )
    if (is.null(df) || nrow(df) == 0) next

    mc <- find_method_col(df)
    methods <- c(methods, normalize_method(df[[mc]]))
  }

  method_sort(c(methods, forced_methods))
}


build_col_fun <- function(mat) {
  vals <- as.vector(mat)
  vals <- vals[is.finite(vals)]

  if (length(vals) == 0) {
    vmin <- 0
    vmax <- 1
    legend_breaks <- pretty(c(vmin, vmax), n = 5)
  } else {
    vmin <- min(vals, na.rm = TRUE)
    pos_vals <- vals[vals > 0]
    vmax <- if (length(pos_vals) > 0) {
      as.numeric(stats::quantile(pos_vals, probs = upper_q, na.rm = TRUE))
    } else {
      max(vals, na.rm = TRUE)
    }

    if (!is.finite(vmax) || vmax <= vmin) {
      vmax <- max(vals, na.rm = TRUE)
    }
    if (!is.finite(vmax) || vmax <= vmin) {
      vmax <- vmin + 1
    }

    legend_breaks <- pretty(c(vmin, vmax), n = 5)
  }

  hcl_col <- grDevices::hcl.colors(256, palette = palette_name, rev = TRUE)
  col_fun <- circlize::colorRamp2(
    seq(vmin, vmax, length.out = length(hcl_col)),
    hcl_col
  )

  list(
    col_fun = col_fun,
    vmin = vmin,
    vmax = vmax,
    legend_breaks = legend_breaks
  )
}


format_cell_value <- function(val) {
  if (is.na(val)) return("NA")
  if (abs(val) >= 100) return(sprintf("%.0f", val))
  if (abs(val) >= 10)  return(sprintf("%.1f", val))
  sprintf("%.2f", val)
}


plot_one_heatmap <- function(mat,
                             metric_name,
                             data_type,
                             outdir) {
  color_info <- build_col_fun(mat)
  col_fun <- color_info$col_fun
  vmin <- color_info$vmin
  vmax <- color_info$vmax
  legend_breaks <- color_info$legend_breaks

  # For peak_location heatmaps, use the same legend title as requested.
  # This only changes the legend title; file names and plotted values are unchanged.
  if (is_peak_location_metric(metric_name)) {
    legend_title <- "Consistency score"
  } else {
    legend_title <- paste0(metric_name, " value")
  }

  ht <- Heatmap(
    mat,
    name = legend_title,
    col = col_fun,
    na_col = na_fill,

    cluster_rows = FALSE,
    cluster_columns = FALSE,

    column_names_side = "top",
    column_names_rot = 45,
    column_names_centered = TRUE,

    row_names_gp = gpar(fontsize = row_name_size),
    column_names_gp = gpar(fontsize = column_name_size),

    # White cell gaps, matching the uploaded Heatmap.R style.
    rect_gp = gpar(col = "white", lwd = white_grid_lwd),

    cell_fun = function(j, i, x, y, w, h, fill) {
      val <- mat[i, j]
      if (is.na(val)) {
        grid.text("NA", x, y, gp = gpar(fontsize = cell_text_size, col = "black"))
      } else {
        txt_col <- ifelse(val > (vmin + vmax) / 2, "white", "black")
        grid.text(
          format_cell_value(val),
          x,
          y,
          gp = gpar(fontsize = cell_text_size, col = txt_col)
        )
      }
    },

    heatmap_legend_param = list(
      at = legend_breaks,
      labels = sprintf("%.2f", legend_breaks),
      title = legend_title,
      legend_direction = "vertical",
      title_position = "leftcenter-rot"
    ),

    use_raster = TRUE,
    raster_quality = 2
  )

  n_rows <- nrow(mat)
  n_cols <- ncol(mat)

  # Dynamic size: enough room for many datasets and many method columns.
  pdf_width <- max(9.5, 0.35 * n_cols + 3.0)
  pdf_height <- max(7.0, 0.22 * n_rows + 3.5)

  png_width <- ceiling(pdf_width * png_res)
  png_height <- ceiling(pdf_height * png_res)

  file_stub <- paste0(data_type, "_", safe_filename(metric_name), "_heatmap")
  out_pdf <- file.path(outdir, paste0(file_stub, ".pdf"))
  out_png <- file.path(outdir, paste0(file_stub, ".png"))

  pdf(out_pdf, width = pdf_width, height = pdf_height)
  draw(ht)
  dev.off()

  png(out_png, width = png_width, height = png_height, res = png_res)
  draw(ht)
  dev.off()

  # Save the exact matrix used for plotting.
  write.csv(
    mat,
    file.path(outdir, paste0(file_stub, "_matrix_used.csv")),
    row.names = TRUE
  )

  data.frame(
    data_type = data_type,
    metric = metric_name,
    n_datasets = n_rows,
    n_methods = n_cols,
    n_non_na = sum(!is.na(mat)),
    n_na = sum(is.na(mat)),
    pdf = out_pdf,
    png = out_png,
    stringsAsFactors = FALSE
  )
}


plot_all_metric_heatmaps <- function(dir_path,
                                     data_type,
                                     outdir,
                                     global_methods = NULL) {
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  files <- list.files(dir_path, pattern = "\\.csv$", full.names = TRUE)
  files <- sort(files)

  if (length(files) == 0) {
    warning("No CSV files found in: ", dir_path)
    return(data.frame())
  }

  qc_list <- list()

  for (ff in files) {
    message("Plotting: ", ff)

    obj <- read_metric_matrix(
      csv_path = ff,
      data_type = data_type,
      target_methods = global_methods,
      method_col = "Method",
      treat_zero = treat_zero_as_na
    )

    qc <- plot_one_heatmap(
      mat = obj$matrix,
      metric_name = obj$metric_name,
      data_type = data_type,
      outdir = outdir
    )
    qc$source_csv <- ff

    qc_list[[length(qc_list) + 1]] <- qc
  }

  do.call(rbind, qc_list)
}


# =========================
# 3. Main run
# =========================

if (use_global_method_union) {
  global_methods <- method_sort(c(
    collect_methods_from_dir(real_dir),
    collect_methods_from_dir(sim_dir),
    forced_methods
  ))
} else {
  global_methods <- NULL
}

qc_real <- plot_all_metric_heatmaps(
  dir_path = real_dir,
  data_type = "real",
  outdir = real_outdir,
  global_methods = global_methods
)

qc_sim <- plot_all_metric_heatmaps(
  dir_path = sim_dir,
  data_type = "sim",
  outdir = sim_outdir,
  global_methods = global_methods
)

qc_all <- rbind(qc_real, qc_sim)
write.csv(
  qc_all,
  file.path(output_root, "accuracy_metric_heatmap_qc_summary.csv"),
  row.names = FALSE
)

if (!is.null(global_methods)) {
  writeLines(global_methods, file.path(output_root, "global_method_order_alphabetical.txt"))
}

message("Done.")
message("Output root: ", output_root)
message("QC summary: ", file.path(output_root, "accuracy_metric_heatmap_qc_summary.csv"))
