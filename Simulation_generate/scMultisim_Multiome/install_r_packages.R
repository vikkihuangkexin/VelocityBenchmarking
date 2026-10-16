#!/usr/bin/env Rscript
options(repos = c(CRAN = "https://cloud.r-project.org"))
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(
  c("scMultiSim", "chromVAR", "SummarizedExperiment", "GenomicRanges", "IRanges"),
  ask = FALSE, update = FALSE
)
required <- c("scMultiSim", "Matrix", "chromVAR", "SummarizedExperiment", "GenomicRanges", "IRanges")
stopifnot(all(vapply(required, requireNamespace, logical(1), quietly = TRUE)))
print(sessionInfo())
