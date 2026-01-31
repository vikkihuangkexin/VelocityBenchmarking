suppressPackageStartupMessages({
  library(Matrix)
  library(fastTopics)
  library(optparse)
})

option_list <- list(
  make_option("--counts_mtx", type = "character", default = NULL, help = "Path to *_Counts.mtx (default: auto-detect in ./example/output/topic-velo)"),
  make_option("--genes_csv",  type = "character", default = NULL, help = "Path to *_Genes_names.csv (default: auto-detect in ./example/output/topic-velo)"),
  make_option("--save_dir",   type = "character", default = "./example/output/topic-velo", help = "Output directory (default: ./example/output/topic-velo)"),
  make_option("--K",          type = "integer",   default = 8, help = "Number of topics [default %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))

dir.create(opt$save_dir, showWarnings = FALSE, recursive = TRUE)

# Auto-detect counts and genes files if not provided
if (is.null(opt$counts_mtx)) {
  candidates <- list.files(opt$save_dir, pattern = "_scNT_HH_filtered_SU_Counts.mtx$", full.names = TRUE)
  if (length(candidates) == 0) stop("No *_Counts.mtx found in save_dir and --counts_mtx not provided")
  opt$counts_mtx <- candidates[1]
}
if (is.null(opt$genes_csv)) {
  candidates <- list.files(opt$save_dir, pattern = "_scNT_HH_filtered_SU_Genes_names.csv$", full.names = TRUE)
  if (length(candidates) == 0) stop("No *_Genes_names.csv found in save_dir and --genes_csv not provided")
  opt$genes_csv <- candidates[1]
}

fname <- basename(opt$counts_mtx)
num_id <- sub("_scNT_HH_filtered_SU_Counts.mtx$", "", fname)

message("Processing sample: ", num_id)


scNTHH_counts <- as(
  Matrix::readMM(opt$counts_mtx),
  "CsparseMatrix"
)


scNTHH_counts <- scNTHH_counts[, Matrix::colSums(scNTHH_counts) > 0]

scNTHH_geneNames <- read.csv(
  opt$genes_csv,
  header = TRUE,
  stringsAsFactors = FALSE
)[[2]]


K <- opt$K

scNTHH_fit <- fit_topic_model(scNTHH_counts, k = K)
scNTHH_de  <- de_analysis(scNTHH_fit, scNTHH_counts)


saveRDS(
  scNTHH_fit,
  file = file.path(opt$save_dir,
                   paste0(num_id, "_fastTopics_fit_k=", K, ".rds"))
)

write.csv(
  scNTHH_fit$L,
  file = file.path(opt$save_dir,
                   paste0(num_id, "_fastTopics_CellWeights_k=", K, ".csv"))
)

saveRDS(
  scNTHH_de,
  file = file.path(opt$save_dir,
                   paste0(num_id, "_fastTopics_de_k=", K, ".rds"))
)

write.csv(
  scNTHH_de$postmean,
  file = file.path(opt$save_dir,
                   paste0(num_id, "_fastTopics_de_postmean_k=", K, ".csv"))
)

write.csv(
  scNTHH_de$lfsr,
  file = file.path(opt$save_dir,
                   paste0(num_id, "_fastTopics_de_lfsr_k=", K, ".csv"))
)

write.csv(
  scNTHH_de$z,
  file = file.path(opt$save_dir,
                   paste0(num_id, "_fastTopics_de_z_k=", K, ".csv"))
)

message("Done.")
