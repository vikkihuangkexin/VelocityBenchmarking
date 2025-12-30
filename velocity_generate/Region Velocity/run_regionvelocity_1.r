#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Regionvelocity)
  library(data.table)
  library(optparse)
})

# --------------------------------------------------
# CLI options
# --------------------------------------------------
option_list <- list(
  make_option(
    c("-o", "--outdir"),
    type = "character",
    default = "regionvelocity_csv",
    help = "Output directory for RegionVelocity CSV files [default: %default]"
  ),
  make_option(
    c("-n", "--n-cores"),
    type = "integer",
    default = 8,
    help = "Number of cores for RegionVelocity computation [default: %default]"
  )
)

opt <- parse_args(OptionParser(option_list = option_list))

outdir <- opt$outdir
n.cores <- opt$`n-cores`

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

cat("Output directory:", outdir, "\n")
cat("Using n.cores =", n.cores, "\n")

# --------------------------------------------------
# load built-in data
# --------------------------------------------------
cat("Loading RegionVelocity built-in data...\n")

data(cell_e_gexp_sper_pb)
data(cell_i_gexp_sper_pb)
data(cell_RNA_gexp_sper_pb)

# --------------------------------------------------
# run RegionVelocity
# --------------------------------------------------
cat("Running steady-state region velocity...\n")

region_vel <- gene.region.velocity.estimates(
  cell_e_gexp_sper_pb,
  cell_i_gexp_sper_pb,
  theta.s = TRUE,
  RNA_mat = cell_RNA_gexp_sper_pb,
  fit.quantile = 0.05,
  kCells = 10,
  n.cores = n.cores
)

cat("Running EM-based region velocity...\n")

region_vel_EM <- gene.EM.velocity.estimates(
  region_vel,
  n.cores = n.cores
)

# --------------------------------------------------
# helper: gene x cell -> cell x gene with cell column
# --------------------------------------------------
write_cell_gene_matrix <- function(mat, file) {
  mat <- as.matrix(mat)
  dt <- as.data.table(t(mat), keep.rownames = "cell")
  fwrite(dt, file)
}

# --------------------------------------------------
# layers (cell x gene)
# --------------------------------------------------
cat("Writing layer CSV files...\n")

write_cell_gene_matrix(region_vel$deltaE,
                       file.path(outdir, "region_velocity.csv"))
write_cell_gene_matrix(region_vel_EM$deltaE,
                       file.path(outdir, "region_velocity_EM.csv"))
write_cell_gene_matrix(region_vel$projected,
                       file.path(outdir, "region_projected.csv"))
write_cell_gene_matrix(region_vel_EM$projected,
                       file.path(outdir, "region_projected_EM.csv"))
write_cell_gene_matrix(region_vel$current,
                       file.path(outdir, "region_Ms.csv"))
write_cell_gene_matrix(region_vel_EM$time,
                       file.path(outdir, "region_EM_time.csv"))

# --------------------------------------------------
# gene-level metadata
# --------------------------------------------------
cat("Writing gene-level metadata...\n")

colnames(region_vel_EM$para.new) <- c(
  "region_EM_alpha",
  "region_EM_beta",
  "region_EM_gamma",
  "region_EM_ts",
  "region_EM_ITR"
)

fwrite(
  data.table(
    gene = rownames(region_vel_EM$para.new),
    region_vel_EM$para.new
  ),
  file.path(outdir, "region_EM_parameters.csv")
)

fwrite(
  data.table(
    gene = names(region_vel_EM$gamma),
    region_velocity_gamma = as.numeric(region_vel_EM$gamma)
  ),
  file.path(outdir, "region_velocity_gamma.csv")
)

fwrite(
  data.table(
    gene = rownames(region_vel$projected),
    used_in_EM_model = rownames(region_vel$projected) %in%
                       rownames(region_vel_EM$para.new)
  ),
  file.path(outdir, "used_in_EM_model.csv")
)

# --------------------------------------------------
# cell-level metadata
# --------------------------------------------------
cat("Writing cell-level metadata...\n")

fwrite(
  data.table(
    cell = names(region_vel$cellsize),
    region_velocity_cellsize = as.numeric(region_vel$cellsize)
  ),
  file.path(outdir, "region_velocity_cellsize.csv")
)

cat("RegionVelocity export finished successfully.\n")
