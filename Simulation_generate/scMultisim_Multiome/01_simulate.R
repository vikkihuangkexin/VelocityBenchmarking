#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE, warn = 1)

get_arg <- function(flag, default = NULL) {
  args <- commandArgs(trailingOnly = TRUE)
  hit <- which(args == flag)
  if (!length(hit)) return(default)
  if (hit[1] == length(args)) stop("Missing value after ", flag)
  args[hit[1] + 1L]
}

dataset_name <- get_arg("--dataset-name")
outdir <- get_arg("--outdir")
topology <- get_arg("--topology")
num_cells <- as.integer(get_arg("--cells"))
num_genes <- as.integer(get_arg("--genes"))
seed <- as.integer(get_arg("--seed"))

if (any(vapply(list(dataset_name, outdir, topology), is.null, logical(1)))) {
  stop("Required: --dataset-name --outdir --topology --cells --genes --seed")
}
outdir <- normalizePath(outdir, mustWork = FALSE)
if (any(lengths(list(num_cells, num_genes, seed)) != 1L) ||
    anyNA(c(num_cells, num_genes, seed))) stop("cells, genes, and seed must be integers")
if (!(topology %in% c("linear", "bifurcation3", "bifurcation5"))) {
  stop("Unsupported topology: ", topology)
}
if (num_cells < 100L || num_genes < 100L) stop("cells/genes are unexpectedly small")

required <- c("scMultiSim", "Matrix")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing R packages: ", paste(missing, collapse = ", "))

suppressPackageStartupMessages({
  library(scMultiSim)
  library(Matrix)
})

intermediate <- file.path(outdir, "intermediate")
dir.create(intermediate, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(outdir, "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(outdir, "logs"), recursive = TRUE, showWarnings = FALSE)

section <- function(label) {
  cat("\n", paste(rep("=", 96), collapse = ""), "\n", label, "\n", sep = "")
}

rescale01 <- function(x) {
  x <- as.numeric(x)
  limits <- range(x, finite = TRUE)
  if (!all(is.finite(limits)) || diff(limits) == 0) return(rep(0, length(x)))
  (x - limits[1]) / diff(limits)
}

orient_sparse <- function(x, n_rows, n_cols = NULL, label) {
  if (is.null(x)) stop("Missing ", label)
  value <- Matrix::Matrix(x, sparse = TRUE)
  if (nrow(value) == n_rows && (is.null(n_cols) || ncol(value) == n_cols)) return(value)
  if (ncol(value) == n_rows && (is.null(n_cols) || nrow(value) == n_cols)) return(Matrix::t(value))
  stop(label, " has incompatible shape ", paste(dim(value), collapse = " x "))
}

write_names <- function(values, path) {
  write.table(values, path, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
}

write_mtx_gz <- function(value, stem) {
  path <- file.path(intermediate, paste0(stem, ".mtx"))
  gz_path <- paste0(path, ".gz")
  if (file.exists(path)) unlink(path)
  if (file.exists(gz_path)) unlink(gz_path)
  Matrix::writeMM(Matrix::Matrix(value, sparse = TRUE), path)
  status <- system2("gzip", c("-f", shQuote(path)))
  if (status != 0L || !file.exists(gz_path)) stop("gzip failed for ", path)
}

write_dense_f32 <- function(value, filename) {
  con <- file(file.path(intermediate, filename), open = "wb")
  on.exit(close(con), add = TRUE)
  writeBin(as.numeric(value), con, size = 4L, endian = "little")
  close(con)
  on.exit(NULL, add = FALSE)
}

write_tsv_gz <- function(value, filename, row_names = FALSE) {
  con <- gzfile(file.path(intermediate, filename), open = "wt")
  on.exit(close(con), add = TRUE)
  write.table(value, con, sep = "\t", row.names = row_names, col.names = TRUE, quote = FALSE)
  close(con)
  on.exit(NULL, add = FALSE)
}


safe_save_sim_rds <- function(sim_obj, path) {
  if (identical(Sys.getenv("SAVE_SIM_RDS", "1"), "0")) {
    cat("Skipping scmultisim_result.rds cache because SAVE_SIM_RDS=0\n")
    return(invisible(FALSE))
  }
  ok <- tryCatch({
    saveRDS(sim_obj, path, compress = FALSE)
    TRUE
  }, error = function(e) {
    warning("Could not write scmultisim_result.rds cache; continuing without cache: ",
            conditionMessage(e))
    if (file.exists(path)) unlink(path)
    FALSE
  })
  invisible(ok)
}

section("01 scMultiSim simulation")
set.seed(seed)
data("GRN_params_100", package = "scMultiSim")
tree <- switch(
  topology,
  linear = scMultiSim::Phyla1(),
  bifurcation3 = scMultiSim::Phyla3(),
  bifurcation5 = scMultiSim::Phyla5()
)

# Keeping the supplied 100-gene GRN and setting num.genes gives an exact feature
# count while avoiding construction of an unmanageably dense 20k-gene GRN.
base_num_cifs <- as.integer(Sys.getenv(
  "SCMULTISIM_NUM_CIFS",
  if (num_genes >= 5000L) "100" else "50"
))
if (!is.finite(base_num_cifs) || base_num_cifs < 5L) {
  stop("SCMULTISIM_NUM_CIFS must be an integer >= 5")
}
cat("Requested scMultiSim num.cifs: ", base_num_cifs, "\n", sep = "")
base_atac_p_zero <- as.numeric(Sys.getenv("ATAC_P_ZERO", "0.80"))
if (!is.finite(base_atac_p_zero) || base_atac_p_zero < 0 || base_atac_p_zero >= 1) {
  stop("ATAC_P_ZERO must be in [0, 1)")
}
cat("Requested ATAC p_zero: ", base_atac_p_zero, "\n", sep = "")
request_signature <- list(
  dataset_name = dataset_name, topology = topology,
  num_cells = num_cells, num_genes = num_genes, seed = seed,
  num_cifs = base_num_cifs, atac_p_zero = base_atac_p_zero,
  scMultiSim_version = as.character(packageVersion("scMultiSim"))
)

make_sim_options <- function(actual_seed, n_cifs, cif_sigma, diff_fraction, atac_effect, atac_p_zero) {
  list(
    rand.seed = actual_seed,
    GRN = GRN_params_100,
    num.genes = num_genes,
    num.cells = num_cells,
    num.cifs = n_cifs,
    cif.sigma = cif_sigma,
    diff.cif.fraction = diff_fraction,
    tree = tree,
    discrete.cif = FALSE,
    do.velocity = TRUE,
    speed.up = TRUE,
    atac.effect = atac_effect,
    atac.p_zero = atac_p_zero
  )
}

sim_attempts <- data.frame(
  label = c("primary", "retry_seed", "retry_low_noise", "retry_conservative"),
  actual_seed = c(seed, seed + 10000L, seed + 20000L, seed + 30000L),
  n_cifs = c(base_num_cifs, base_num_cifs, min(base_num_cifs, 60L), min(base_num_cifs, 40L)),
  cif_sigma = c(0.30, 0.25, 0.18, 0.12),
  diff_fraction = c(0.85, 0.75, 0.60, 0.50),
  atac_effect = c(0.70, 0.60, 0.45, 0.35),
  atac_p_zero = rep(base_atac_p_zero, 4L),
  stringsAsFactors = FALSE
)

sim_rds_path <- file.path(intermediate, "scmultisim_result.rds")
simulation_reused <- FALSE
simulation_attempt_label <- NA_character_
simulation_actual_seed <- seed
simulation_error_messages <- character()

if (file.exists(sim_rds_path)) {
  candidate <- tryCatch(readRDS(sim_rds_path), error = function(e) e)
  if (inherits(candidate, "error")) {
    cat("Existing scmultisim_result.rds is unreadable; rerunning simulation: ",
        conditionMessage(candidate), "\n", sep = "")
  } else {
    candidate_shape <- dim(candidate$counts)
    candidate_requested_seed <- tryCatch(
      as.integer(candidate$.pipeline_simulation_meta$requested_seed),
      error = function(e) NA_integer_
    )
    candidate_seed <- tryCatch(
      as.integer(candidate$.options$rand.seed),
      error = function(e) NA_integer_
    )
    if (length(candidate_requested_seed) != 1L) candidate_requested_seed <- NA_integer_
    if (length(candidate_seed) != 1L) candidate_seed <- NA_integer_
    seed_compatible <- identical(candidate_requested_seed, seed) ||
      (is.na(candidate_requested_seed) &&
         (is.na(candidate_seed) || identical(candidate_seed, seed)))
    if (
      identical(as.integer(candidate_shape), c(num_genes, num_cells)) &&
        seed_compatible &&
        identical(candidate$.pipeline_simulation_meta$request_signature, request_signature)
    ) {
      cat("Reusing existing compatible scmultisim_result.rds\n")
      sim <- candidate
      simulation_reused <- TRUE
      simulation_attempt_label <- tryCatch(
        paste0("reused:", candidate$.pipeline_simulation_meta$attempt_label),
        error = function(e) "reused_existing_rds"
      )
      simulation_actual_seed <- tryCatch(
        as.integer(candidate$.pipeline_simulation_meta$actual_seed),
        error = function(e) seed
      )
    } else {
      cat("Existing scmultisim_result.rds is incompatible; rerunning simulation\n")
    }
  }
}

if (!simulation_reused) {
  sim <- NULL
  for (i in seq_len(nrow(sim_attempts))) {
    row <- sim_attempts[i, ]
    sim_options <- make_sim_options(
      actual_seed = row$actual_seed,
      n_cifs = row$n_cifs,
      cif_sigma = row$cif_sigma,
      diff_fraction = row$diff_fraction,
      atac_effect = row$atac_effect,
      atac_p_zero = row$atac_p_zero
    )
    cat(
      "Simulation attempt ", i, "/", nrow(sim_attempts),
      ": ", row$label,
      " seed=", row$actual_seed,
      " num.cifs=", row$n_cifs,
      " cif.sigma=", row$cif_sigma,
      " diff.cif.fraction=", row$diff_fraction,
      " atac.effect=", row$atac_effect,
      " atac.p_zero=", row$atac_p_zero,
      "\n", sep = ""
    )
    attempt <- tryCatch(
      scMultiSim::sim_true_counts(sim_options),
      error = function(e) e
    )
    if (!inherits(attempt, "error")) {
      sim <- attempt
      simulation_attempt_label <- row$label
      simulation_actual_seed <- row$actual_seed
      sim$.pipeline_simulation_meta <- list(
        requested_seed = seed,
        actual_seed = simulation_actual_seed,
        attempt_label = simulation_attempt_label,
        sim_options = sim_options,
        request_signature = request_signature,
        previous_errors = simulation_error_messages
      )
      safe_save_sim_rds(sim, sim_rds_path)
      break
    }
    msg <- conditionMessage(attempt)
    simulation_error_messages <- c(
      simulation_error_messages,
      paste0(row$label, ": ", msg)
    )
    cat("Simulation attempt failed: ", msg, "\n", sep = "")
  }
  if (is.null(sim)) {
    stop("All scMultiSim simulation attempts failed:\n",
         paste(simulation_error_messages, collapse = "\n"))
  }
}


spliced <- Matrix::Matrix(sim$counts, sparse = TRUE)
unspliced <- Matrix::Matrix(sim$unspliced_counts, sparse = TRUE)
true_velocity <- as.matrix(sim$velocity)
atac_counts <- Matrix::Matrix(round(pmax(sim$atac_counts, 0)), sparse = TRUE)
atac_true <- Matrix::Matrix(pmax(sim$atacseq_data, 0), sparse = TRUE)

if (!identical(dim(spliced), dim(unspliced)) || !identical(dim(spliced), dim(true_velocity))) {
  stop("spliced/unspliced/velocity dimensions differ")
}
if (nrow(spliced) != num_genes || ncol(spliced) != num_cells) {
  stop("scMultiSim did not honor requested shape: ", paste(dim(spliced), collapse = " x "))
}
if (ncol(atac_counts) != num_cells) stop("RNA/ATAC cell counts differ")

region_to_gene <- orient_sparse(sim$region_to_gene, nrow(atac_counts), num_genes, "region_to_gene")
region_to_tf <- orient_sparse(sim$region_to_tf, nrow(atac_counts), NULL, "region_to_tf")

# Keep only observed peaks that participate in the paired region-to-gene graph;
# cells and RNA genes are never removed.
raw_region_count <- nrow(atac_counts)
region_has_signal <- Matrix::rowSums(atac_counts) > 0
region_has_gene_link <- Matrix::rowSums(region_to_gene) > 0
region_keep <- region_has_signal & region_has_gene_link
cat("Raw ATAC peaks: ", raw_region_count, "\n", sep = "")
cat("All-zero observed ATAC peaks: ", sum(!region_has_signal), "\n", sep = "")
cat("ATAC peaks without a gene link: ", sum(!region_has_gene_link), "\n", sep = "")
cat("Retained linked, observed ATAC peaks: ", sum(region_keep), "\n", sep = "")
if (!any(region_keep)) stop("No observed gene-linked ATAC peaks remain")
if (!all(region_keep)) {
  cat("Dropping ATAC peaks unusable for paired gene-region benchmarking: ", sum(!region_keep), "\n", sep = "")
  atac_counts <- atac_counts[region_keep, , drop = FALSE]
  atac_true <- atac_true[region_keep, , drop = FALSE]
  region_to_gene <- region_to_gene[region_keep, , drop = FALSE]
  region_to_tf <- region_to_tf[region_keep, , drop = FALSE]
}

n_regions <- nrow(atac_counts)
gene_names <- sprintf("gene_%05d", seq_len(num_genes))
cell_names <- sprintf("cell_%06d", seq_len(num_cells))

# One coordinate system is shared by RNA genes and linked ATAC peaks.
gene_index <- seq_len(num_genes)
gene_chrom <- paste0("chr", ((gene_index - 1L) %% 22L) + 1L)
gene_start <- 100000L + ((gene_index - 1L) %/% 22L) * 100000L
gene_end <- gene_start + 19999L

mapping_summary <- Matrix::summary(region_to_gene)
primary_gene <- rep(NA_integer_, n_regions)
first_link <- !duplicated(mapping_summary$i)
primary_gene[mapping_summary$i[first_link]] <- mapping_summary$j[first_link]
if (anyNA(primary_gene)) stop("Every retained ATAC peak must link to at least one gene")
rank_within_gene <- ave(seq_len(n_regions), primary_gene, FUN = seq_along)
peak_chrom <- gene_chrom[primary_gene]
peak_start <- pmax(1L, gene_start[primary_gene] - 5000L + 500L * (rank_within_gene - 1L))
peak_end <- peak_start + 399L
region_names <- paste0(peak_chrom, ":", peak_start, "-", peak_end)
if (anyDuplicated(region_names)) stop("Synthetic peak coordinates are not unique")

rownames(spliced) <- gene_names
rownames(unspliced) <- gene_names
rownames(true_velocity) <- gene_names
colnames(spliced) <- cell_names
colnames(unspliced) <- cell_names
colnames(true_velocity) <- cell_names
rownames(atac_counts) <- region_names
rownames(atac_true) <- region_names
colnames(atac_counts) <- cell_names
colnames(atac_true) <- cell_names
rownames(region_to_gene) <- region_names
colnames(region_to_gene) <- gene_names
rownames(region_to_tf) <- region_names
tf_names <- colnames(region_to_tf)
if (is.null(tf_names) || any(!nzchar(tf_names))) {
  tf_names <- sprintf("TF_%03d", seq_len(ncol(region_to_tf)))
}
colnames(region_to_tf) <- tf_names

region_to_gene <- 1L * (region_to_gene > 0)
region_to_tf <- 1L * (region_to_tf > 0)
chromatin <- Matrix::t(region_to_gene) %*% atac_counts

gene_metadata <- data.frame(
  gene = gene_names, chrom = gene_chrom, chromStart = gene_start,
  chromEnd = gene_end, gene_type = "protein_coding", stringsAsFactors = FALSE
)
peak_metadata <- data.frame(
  region = region_names, chrom = peak_chrom, chromStart = peak_start,
  chromEnd = peak_end, linked_gene = gene_names[primary_gene],
  linked_gene_index = primary_gene, coordinate_system = "synthetic_human_like_v1",
  stringsAsFactors = FALSE
)

cell_meta <- as.data.frame(sim$cell_meta)
if (nrow(cell_meta) != num_cells) stop("cell_meta does not have requested cells")
rownames(cell_meta) <- cell_names
type_col <- intersect(c("pop", "cell_type", "celltype", "branch", "lineage"), names(cell_meta))
cell_meta$true_cell_type <- if (length(type_col)) as.character(cell_meta[[type_col[1]]]) else "trajectory"
cell_meta$true_lineage <- cell_meta$true_cell_type
true_time <- sim$cell_time
if (is.null(true_time)) true_time <- seq_len(num_cells)
if (!is.vector(true_time)) {
  true_time <- if (nrow(true_time) == num_cells) rowMeans(true_time) else colMeans(true_time)
}
if (length(true_time) != num_cells) stop("Cannot obtain one true_time value per cell")
cell_meta$true_time <- as.numeric(true_time)
pt_col <- intersect(c("pseudotime", "pseudo_time", "time", "step"), names(cell_meta))
cell_meta$true_pseudotime <- if (length(pt_col)) rescale01(cell_meta[[pt_col[1]]]) else rescale01(true_time)
cell_meta$simulation_cell_id <- cell_names

section("02 Write scalable intermediate matrices")
write_mtx_gz(spliced, "spliced_genes_cells")
write_mtx_gz(unspliced, "unspliced_genes_cells")
write_mtx_gz(atac_counts, "atac_counts_regions_cells")
write_mtx_gz(atac_true, "atac_true_regions_cells")
write_mtx_gz(region_to_gene, "region_to_gene_regions_genes")
write_mtx_gz(region_to_tf, "region_to_tf_regions_tfs")
write_mtx_gz(chromatin, "chromatin_genes_cells")
write_dense_f32(true_velocity, "true_velocity_genes_cells.f32")
write_names(gene_names, file.path(intermediate, "gene_names.tsv"))
write_names(cell_names, file.path(intermediate, "cell_names.tsv"))
write_names(region_names, file.path(intermediate, "region_names.tsv"))
write_names(tf_names, file.path(intermediate, "tf_names.tsv"))
write_tsv_gz(gene_metadata, "gene_metadata.tsv.gz")
write_tsv_gz(peak_metadata, "peak_metadata.tsv.gz")
write_tsv_gz(cbind(cell = rownames(cell_meta), cell_meta), "cell_metadata.tsv.gz")

section("03 chromVAR (synthetic annotation mode)")
run_chromvar <- identical(Sys.getenv("RUN_CHROMVAR", "1"), "1")
if (!run_chromvar) {
  cat("Skipping chromVAR because RUN_CHROMVAR=0\n")
}

chromvar_status <- data.frame(
  available = FALSE,
  mode = "synthetic_region_to_tf_with_pseudo_gc_background_no_reference_genome_scan",
  reason = "not_attempted",
  n_motifs = 0L,
  stringsAsFactors = FALSE
)

chromvar_packages <- c("chromVAR", "SummarizedExperiment", "GenomicRanges", "IRanges")
if (run_chromvar && all(vapply(chromvar_packages, requireNamespace, logical(1), quietly = TRUE))) {
  tryCatch({
    motif_keep <- Matrix::colSums(region_to_tf) >= 3L & Matrix::colSums(region_to_tf) <= (n_regions - 3L)
    if (sum(motif_keep) < 2L) stop("fewer than two usable region-to-TF annotations")
    annotations <- region_to_tf[, motif_keep, drop = FALSE]
    ranges <- GenomicRanges::GRanges(
      seqnames = peak_chrom,
      ranges = IRanges::IRanges(start = peak_start, end = peak_end)
    )
    se <- SummarizedExperiment::SummarizedExperiment(
      assays = list(counts = atac_counts), rowRanges = ranges
    )
    rownames(se) <- region_names
    colnames(se) <- cell_names
    SummarizedExperiment::rowData(se)$bias <-
      0.35 + 0.30 * ((seq_len(n_regions) * 0.618033988749895) %% 1)
    background <- chromVAR::getBackgroundPeaks(se, niterations = 50)
    deviations <- chromVAR::computeDeviations(se, annotations, background_peaks = background)
    z <- as.matrix(SummarizedExperiment::assay(deviations, "z"))
    cell_motif <- if (ncol(z) == num_cells) t(z) else z
    if (nrow(cell_motif) != num_cells) stop("unexpected chromVAR dimensions")
    finite <- apply(cell_motif, 2L, function(x) all(is.finite(x)))
    cell_motif <- cell_motif[, finite, drop = FALSE]
    if (ncol(cell_motif) < 2L) stop("fewer than two finite chromVAR motifs")
    rownames(cell_motif) <- cell_names
    write_dense_f32(cell_motif, "chromvar_cells_motifs.f32")
    write_names(colnames(cell_motif), file.path(intermediate, "chromvar_motif_names.tsv"))
    chromvar_status$available <- TRUE
    chromvar_status$reason <- "success"
    chromvar_status$n_motifs <- ncol(cell_motif)
  }, error = function(e) {
    chromvar_status$reason <<- conditionMessage(e)
    warning("chromVAR unavailable: ", conditionMessage(e))
  })
} else if (!run_chromvar) {
  chromvar_status$reason <- "skipped_by_RUN_CHROMVAR_0"
} else {
  chromvar_status$reason <- paste0(
    "missing_packages:",
    paste(chromvar_packages[!vapply(chromvar_packages, requireNamespace, logical(1), quietly = TRUE)], collapse = ",")
  )
}
write_tsv_gz(chromvar_status, "chromvar_status.tsv.gz")

manifest <- data.frame(
  dataset_name = dataset_name, topology = topology, seed = seed,
  requested_cells = num_cells, final_cells = ncol(spliced),
  requested_genes = num_genes, final_genes = nrow(spliced),
  final_regions = nrow(atac_counts),
  atac_p_zero = base_atac_p_zero, coordinate_system = "synthetic_human_like_v1",
  chromvar_available = chromvar_status$available,
  simulation_actual_seed = simulation_actual_seed,
  simulation_attempt_label = simulation_attempt_label,
  simulation_rds_reused = simulation_reused,
  simulation_retry_errors = paste(simulation_error_messages, collapse = " || "),
  simulation_num_cifs = sim$.pipeline_simulation_meta$sim_options$num.cifs,
  simulation_cif_sigma = sim$.pipeline_simulation_meta$sim_options$cif.sigma,
  simulation_diff_fraction = sim$.pipeline_simulation_meta$sim_options$diff.cif.fraction,
  simulation_atac_effect = sim$.pipeline_simulation_meta$sim_options$atac.effect,
  scMultiSim_version = as.character(packageVersion("scMultiSim")),
  stringsAsFactors = FALSE
)
write_tsv_gz(manifest, "simulation_manifest.tsv.gz")
writeLines(capture.output(sessionInfo()), file.path(intermediate, "R_sessionInfo.txt"))

section("04 R stage complete")
cat("Dataset: ", dataset_name, "\n", sep = "")
cat("RNA genes x cells: ", nrow(spliced), " x ", ncol(spliced), "\n", sep = "")
cat("ATAC peaks x cells: ", nrow(atac_counts), " x ", ncol(atac_counts), "\n", sep = "")
cat("chromVAR available: ", chromvar_status$available, "\n", sep = "")
