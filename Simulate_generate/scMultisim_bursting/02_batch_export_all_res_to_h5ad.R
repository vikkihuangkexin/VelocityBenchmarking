suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(zellkonverter)
  library(Rtsne)
})

base_dir <- "/data/khuang6/simulation/test/scmultisim/bursting_benchmark2"
overwrite_h5ad <- TRUE

export_one_rds_to_h5ad <- function(rds_path, overwrite = TRUE) {
  outdir <- dirname(rds_path)
  h5ad_path <- file.path(outdir, "res.h5ad")

  if (file.exists(h5ad_path) && !overwrite) {
    message("[skip] ", h5ad_path, " already exists")
    return(invisible(h5ad_path))
  }

  message("Processing: ", rds_path)

  results <- readRDS(rds_path)

  counts <- as.matrix(results$counts)                # gene x cell
  unspliced <- as.matrix(results$unspliced_counts)   # gene x cell
  velocity <- as.matrix(results$velocity)            # gene x cell
  meta <- as.data.frame(results$cell_meta)

  # names
  if (is.null(colnames(counts))) {
    colnames(counts) <- paste0("Cell_", seq_len(ncol(counts)))
  }
  if (is.null(rownames(counts))) {
    rownames(counts) <- paste0("Gene_", seq_len(nrow(counts)))
  }
  if (is.null(colnames(unspliced))) colnames(unspliced) <- colnames(counts)
  if (is.null(rownames(unspliced))) rownames(unspliced) <- rownames(counts)
  if (is.null(colnames(velocity))) colnames(velocity) <- colnames(counts)
  if (is.null(rownames(velocity))) rownames(velocity) <- rownames(counts)

  # robust metadata alignment
  if (!"cell_id" %in% colnames(meta)) {
    if (!is.null(rownames(meta)) && all(nzchar(rownames(meta)))) {
      meta$cell_id <- rownames(meta)
    } else if (nrow(meta) == ncol(counts)) {
      meta$cell_id <- colnames(counts)
    } else {
      meta$cell_id <- NA_character_
    }
  }

  idx <- match(colnames(counts), meta$cell_id)
  if (all(!is.na(idx))) {
    meta <- meta[idx, , drop = FALSE]
  } else if (nrow(meta) == ncol(counts)) {
    message("  cell_id does not match; using row order to align metadata.")
    meta <- meta[seq_len(ncol(counts)), , drop = FALSE]
    meta$cell_id <- colnames(counts)
  } else {
    stop("metadata cannot be aligned to counts for: ", rds_path)
  }

  if (!"pseudotime" %in% colnames(meta)) {
    if ("cell_time" %in% colnames(meta)) {
      meta$pseudotime <- meta$cell_time
    } else if (!is.null(results$cell_time)) {
      meta$pseudotime <- results$cell_time
    } else {
      meta$pseudotime <- seq_len(ncol(counts))
    }
  }

  if (!"pop" %in% colnames(meta)) {
    meta$pop <- "cells"
  }

  rownames(meta) <- colnames(counts)

  # ONE tSNE only
  expr <- t(log2(counts + 1))  # cell x gene
  set.seed(1)
  perp <- min(30, floor((nrow(expr) - 1) / 3))
  perp <- max(perp, 5)

  tsne_out <- Rtsne(
    expr,
    dims = 2,
    pca = TRUE,
    perplexity = perp,
    check_duplicates = FALSE,
    verbose = TRUE
  )

  tsne_mat <- tsne_out$Y
  rownames(tsne_mat) <- colnames(counts)
  colnames(tsne_mat) <- c("tSNE1", "tSNE2")

  sce <- SingleCellExperiment(
    assays = list(
      counts = counts,
      spliced = counts,
      unspliced = unspliced,
      ground_truth_velocity = velocity
    ),
    colData = meta
  )

  reducedDims(sce)$tsne <- tsne_mat

  zellkonverter::writeH5AD(
    sce,
    file = h5ad_path,
    X_name = "counts"
  )

  message("  saved: ", h5ad_path)
  invisible(h5ad_path)
}

rds_files <- list.files(
  path = base_dir,
  pattern = "^res\\.rds$",
  recursive = TRUE,
  full.names = TRUE
)

message("Found ", length(rds_files), " res.rds files.")

for (f in rds_files) {
  tryCatch(
    export_one_rds_to_h5ad(f, overwrite = overwrite_h5ad),
    error = function(e) {
      message("[ERROR] ", f)
      message("        ", conditionMessage(e))
    }
  )
}

message("Done.")