#!/usr/bin/env Rscript
# ==============================================================================
# simulate_dyngen.R
#
# Purpose
#   Batch-generate dyngen simulation datasets (dyno format) across multiple
#   trajectory backbones, cell counts, and gene counts, and save:
#     1) the dyngen model object            (*_model.rds)
#     2) the dyno dataset object            (*_dataset.rds)
#     3) a 2D dimred embedding (MDS)        (*_dimred.rds)
#     4) per-cell dominant milestone labels (*_obs.rds)
#
# Notes
#   - This script is based on your original version and keeps the same output
#     conventions (including dyngen's "output_dir" behaving like a prefix).
#   - Compared to the original, it makes 'gene_num' effective by allocating
#     num_targets/num_hks so that:
#         total_genes ~= num_tfs + num_targets + num_hks
#   - Seeds are made deterministic per (backbone, cell, gene) configuration,
#     so different runs produce reproducible but distinct datasets.
#
# Typical usage (edit parameters below):
#   Rscript simulate_dyngen.R
#
# Optional CLI (no extra packages needed):
#   Rscript simulate_dyngen.R \
#     --base_dir=./simulation/test \
#     --cells=100,500,1000 \
#     --genes=1000,5000,10000 \
#     --backbones=bifurcating,linear_simple \
#     --seed=123
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(dyngen)
  library(dynplot)
  library(Matrix)
})

# ---- Minimal command-line parsing (key=value) --------------------------------
parse_args <- function(args) {
  # Accept arguments like: --key=value
  out <- list()
  for (a in args) {
    if (!grepl("^--[^=]+=", a)) next
    key <- sub("^--([^=]+)=.*$", "\\1", a)
    val <- sub("^--[^=]+=(.*)$", "\\1", a)
    out[[key]] <- val
  }
  out
}

as_int_vec <- function(x) {
  # Convert "100,500,1000" -> c(100L, 500L, 1000L)
  if (is.null(x) || nchar(x) == 0) return(NULL)
  as.integer(strsplit(x, ",", fixed = TRUE)[[1]])
}

as_chr_vec <- function(x) {
  if (is.null(x) || nchar(x) == 0) return(NULL)
  strsplit(x, ",", fixed = TRUE)[[1]]
}

cli <- parse_args(commandArgs(trailingOnly = TRUE))

# ---- User parameters (defaults; override via CLI) -----------------------------
base_dir <- if (!is.null(cli$base_dir)) cli$base_dir else "./simulation/test"

# IMPORTANT: use vectors if you want multiple configurations
cell_nums <- if (!is.null(cli$cells)) as_int_vec(cli$cells) else c(1000L)

# 'gene_nums' here refers to the *total genes* (TFs + targets + housekeeping).
# It must be >= num_tfs + 2 (to leave at least 2 non-TF genes).
gene_nums <- if (!is.null(cli$genes)) as_int_vec(cli$genes) else c(1000L)

# Default backbones to simulate
backbone_names <- if (!is.null(cli$backbones)) {
  as_chr_vec(cli$backbones)
} else {
  c(
    "bifurcating",
    "bifurcating_loop",
    "consecutive_bifurcating",
    "trifurcating",
    "disconnected",
    "linear_simple",
    "cycle_simple"
  )
}

# Base seed; the actual seed becomes deterministic per configuration.
seed_base <- if (!is.null(cli$seed)) as.integer(cli$seed) else 123L

# Feature-network and simulation knobs (keep in one place for reproducibility)
target_resampling <- 1000L
max_in_degree <- 3L
census_interval <- 1
num_simulations <- 1200L
burn_multiplier <- 1.5

# Allocate non-TF genes between targets and housekeeping genes.
# Adjust target_fraction if you want more/less housekeeping.
target_fraction <- 0.5

# ---- Backbone map -------------------------------------------------------------
backbones <- list(
  bifurcating = dyngen::backbone_bifurcating,
  bifurcating_loop = dyngen::backbone_bifurcating_loop,
  consecutive_bifurcating = dyngen::backbone_consecutive_bifurcating,
  trifurcating = dyngen::backbone_trifurcating,
  disconnected = function() dyngen::backbone_disconnected("linear", "cycle"),
  linear_simple = dyngen::backbone_linear_simple,
  cycle_simple = dyngen::backbone_cycle_simple
)

# ---- Helpers -----------------------------------------------------------------
allocate_gene_counts <- function(total_genes, num_tfs, target_fraction = 0.5) {
  # dyngen total genes = num_tfs + num_targets + num_hks
  # Ensure we leave at least 1 target and 1 housekeeping gene.
  remaining <- as.integer(total_genes - num_tfs)
  if (remaining < 2L) {
    stop(
      sprintf(
        "gene_num (%d) too small for backbone TFs (%d). Need >= num_tfs + 2.",
        total_genes, num_tfs
      )
    )
  }

  num_targets <- max(1L, as.integer(floor(remaining * target_fraction)))
  num_hks <- remaining - num_targets
  if (num_hks < 1L) {
    num_hks <- 1L
    num_targets <- remaining - num_hks
  }
  list(num_targets = num_targets, num_hks = num_hks)
}

seed_from_id <- function(id, seed_base = 123L) {
  # Deterministic seed per id without extra dependencies:
  # sum of UTF-8 code points, then bring into R's integer range.
  s <- sum(utf8ToInt(id))
  as.integer((seed_base + s) %% .Machine$integer.max)
}

# ---- Main loop ---------------------------------------------------------------
dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)

for (backbone_name in backbone_names) {
  if (!backbone_name %in% names(backbones)) {
    warning("Unknown backbone '", backbone_name, "'. Skipping.")
    next
  }

  for (gene_num in gene_nums) {
    for (n_cell in cell_nums) {

      id <- paste0(backbone_name, "_cell", n_cell, "_gene", gene_num)
      outdir <- file.path(base_dir, backbone_name)
      dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

      # Final artifacts we keep regardless of dyngen internal outputs
      dataset_file <- file.path(outdir, paste0(id, "_dataset.rds"))
      model_file   <- file.path(outdir, paste0(id, "_model.rds"))
      dimred_file  <- file.path(outdir, paste0(id, "_dimred.rds"))
      obs_file     <- file.path(outdir, paste0(id, "_obs.rds"))

      # dyngen uses output_dir as a prefix in generate_dataset():
      #   paste0(output_prefix, "dataset.rds")
      output_prefix <- file.path(outdir, id)

      if (file.exists(dataset_file)) {
        message("exists, skip: ", id)
        next
      }

      message("processing: ", id)

      # Use deterministic but distinct seeds across configurations
      set.seed(seed_from_id(id, seed_base = seed_base))

      # Build the topology backbone (milestones + transitions)
      backbone <- backbones[[backbone_name]]()

      # Determine TF count from the backbone and allocate the remaining genes
      num_tfs <- nrow(backbone$module_info)
      alloc <- allocate_gene_counts(
        total_genes = gene_num,
        num_tfs = num_tfs,
        target_fraction = target_fraction
      )

      # Wrap the whole simulation in tryCatch so one failure doesn't kill the batch
      tryCatch({

        model <- initialise_model(
          id = id,
          num_tfs = num_tfs,
          num_targets = alloc$num_targets,
          num_hks = alloc$num_hks,
          backbone = backbone,
          num_cells = n_cell,
          feature_network = feature_network_default(
            target_resampling = target_resampling,
            max_in_degree = max_in_degree
          ),
          simulation_params = simulation_default(
            burn_time = simtime_from_backbone(backbone, burn = TRUE) * burn_multiplier,
            census_interval = census_interval,
            compute_rna_velocity = TRUE,
            store_reaction_propensities = TRUE,
            experiment_params = simulation_type_wild_type(num_simulations = num_simulations)
          ),
          verbose = FALSE
        )

        # This will write several files; for dyno format the dataset is typically:
        #   paste0(output_prefix, "dataset.rds")
        generate_dataset(
          model,
          format = "dyno",
          output_dir = output_prefix,
          make_plots = FALSE
        )

        # Save the full dyngen model for reproducibility / debugging
        saveRDS(model, model_file)

        # Read the dyno dataset and save a clean copy under a stable filename
        dataset <- readr::read_rds(paste0(output_prefix, "dataset.rds"))
        saveRDS(dataset, dataset_file)

        # 2D embedding for quick visualization / downstream plotting
        # Note: as.matrix() will densify; for very large matrices, consider a
        # sparse-aware method or a gene subsampling strategy.
        dimred <- dyndimred::dimred_landmark_mds(
          x = as.matrix(dataset$expression),
          ndim = 2,
          distance_method = "pearson"
        )
        saveRDS(dimred, dimred_file)

        # Create an "obs" table: assign each cell to its dominant milestone
        dominant_milestone <- dataset$milestone_percentages %>%
          dplyr::group_by(cell_id) %>%
          dplyr::slice_max(order_by = percentage, n = 1, with_ties = FALSE) %>%
          dplyr::ungroup() %>%
          dplyr::select(cell_id, milestone = milestone_id)

        obs_df <- data.frame(
          cell_id = rownames(dataset$expression),
          row.names = rownames(dataset$expression),
          stringsAsFactors = FALSE
        )
        obs_df$milestone <- dominant_milestone$milestone[
          match(rownames(obs_df), dominant_milestone$cell_id)
        ]
        saveRDS(obs_df, obs_file)

        message("saved: ", id)

        # Clean up memory between runs
        rm(model, dataset, dimred, obs_df)
        gc()

      }, error = function(e) {
        warning("FAILED: ", id, " | ", conditionMessage(e))
      })
    }
  }
}

message("Done.")
