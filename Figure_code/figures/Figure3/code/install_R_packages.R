#!/usr/bin/env Rscript
# Install the exact tested R package versions in dependency order.
# Usage: Rscript --vanilla code/install_R_packages.R [--check]
# Source installation needs R development tools and, for systemfonts/svglite,
# the platform fontconfig/freetype development libraries (Rtools on Windows).
args <- commandArgs(trailingOnly=TRUE)
script <- grep('^--file=', commandArgs(FALSE), value=TRUE)
if(!length(script)) stop('Run this script using Rscript.')
script_path <- sub('^--file=', '', script[[1]])
# Some Rscript versions encode spaces in the --file argument as ~+~.
if(!file.exists(script_path)) script_path <- gsub('~+~', ' ', script_path, fixed=TRUE)
root <- dirname(dirname(normalizePath(script_path, mustWork=TRUE)))
requirements <- read.csv(file.path(root, 'config/r_packages.csv'), stringsAsFactors=FALSE)
requirements <- requirements[order(requirements$install_order), ]
version_ok <- function(package, version) {
  tryCatch(isTRUE(packageVersion(package) == package_version(version)), error=function(e) FALSE)
}
if('--check' %in% args) {
  ok <- mapply(version_ok, requirements$package, requirements$version)
  if(!all(ok)) stop('Missing or mismatched R packages: ', paste(requirements$package[!ok], collapse=', '))
  message('All pinned R package versions match.')
} else {
  libs <- strsplit(Sys.getenv('R_LIBS_USER'), .Platform$path.sep, fixed=TRUE)[[1]]
  if(!length(libs)) stop('Set R_LIBS_USER to a writable package library.')
  lib <- path.expand(libs[[1]])
  if(!nzchar(lib)) stop('Set R_LIBS_USER to a writable package library.')
  dir.create(lib, recursive=TRUE, showWarnings=FALSE)
  .libPaths(c(lib, .libPaths()))
  for(i in seq_len(nrow(requirements))) {
    package <- requirements$package[[i]];version <- requirements$version[[i]]
    if(version_ok(package, version)) next
    archive <- tempfile(fileext='.tar.gz')
    urls <- if(requirements$repository[[i]]=='Bioconductor_3.18') c(
      sprintf('https://bioconductor.org/packages/3.18/bioc/src/contrib/%s_%s.tar.gz',package,version),
      sprintf('https://bioconductor.org/packages/3.18/bioc/src/contrib/Archive/%s/%s_%s.tar.gz',package,package,version)) else c(
      sprintf('https://cran.r-project.org/src/contrib/%s_%s.tar.gz',package,version),
      sprintf('https://cran.r-project.org/src/contrib/Archive/%s/%s_%s.tar.gz',package,package,version))
    downloaded <- FALSE
    for(url in urls) {
      status <- tryCatch(suppressWarnings(download.file(url, archive, mode='wb', quiet=TRUE)), error=function(e) 1L)
      if(identical(status, 0L)) {downloaded <- TRUE;break}
    }
    if(!downloaded) stop('Could not download ', package, ' ', version)
    install.packages(archive, repos=NULL, type='source', lib=lib)
    unlink(archive)
    if(!version_ok(package, version)) stop('Installation/version check failed: ', package)
  }
  message('All pinned R package versions are installed.')
}
