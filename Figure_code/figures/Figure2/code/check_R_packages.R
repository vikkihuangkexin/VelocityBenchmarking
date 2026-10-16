# Loaded after PROJECT is set by plot_accuracy_panels.R.
requirements <- read.csv(file.path(PROJECT, 'config/r_packages.csv'), stringsAsFactors=FALSE)
for(i in seq_len(nrow(requirements))) {
  package <- requirements$package[[i]];version <- requirements$version[[i]]
  installed <- tryCatch(as.character(packageVersion(package)), error=function(e) '')
  matches <- tryCatch(isTRUE(packageVersion(package) == package_version(version)), error=function(e) FALSE)
  if(!matches) stop('Expected ', package, ' ', version,
    '; found ', installed, '. Run code/install_R_packages.R before plotting.')
}
