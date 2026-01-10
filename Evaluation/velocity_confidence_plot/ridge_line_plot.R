# ridge_line_plot.R
# Reduced comments; configurable paths and reads dataset IDs from save_dir subfolders.

library(dplyr)
library(ggplot2)
library(ggridges)
library(viridis)
library(tibble)

# Configuration
save_dir <- list(
  'velocyto' = 'results/velocyto',
  'scVelo stochastic' = 'results/scvelo_stochastic',
  'scVelo dynamical' = 'results/scvelo_dynamical',
  'veloVI' = 'results/velovi',
  'UniTVelo' = 'results/unitvelo',
  'CellDancer' = 'results/celldancer',
  'k-velo' = 'results/k-velo',
  'VeloAE' = 'results/veloae',
  'TopicVelo' = 'results/topicvelo',
  'DeepVelo' = 'results/deepvelo',
  'NeuroVelo' = 'results/neurvelo',
  'TFvelo' = 'results/tfvelo',
  'Latentvelo' = 'results/latentvelo',
  'SDEvelo' = 'results/sdevelo',
  'STT' = 'results/stt',
  'VeloVAE' = 'results/velovae',
  'Pyro-Velocity' = 'results/pyrovelocity',
  'scRNAKinetics' = 'results/scrnakinetics',
  'cell2fate' = 'results/cell2fate',
  'Region_Velocity' = 'results/regionvelocity'
)

unitvelo_subset_dir <- 'results/unitvelo_subset'
output_dir <- 'results/figure_benchmark/ridgeline'
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Find dataset IDs from the first existing method directory
existing_method_dirs <- vapply(save_dir, dir.exists, logical(1))
if (!any(existing_method_dirs)) stop("No existing directories in save_dir. Update paths.")
first_existing <- names(save_dir)[which(existing_method_dirs)[1]]
first_dir <- save_dir[[first_existing]]
dataset_ids <- list.dirs(first_dir, full.names = FALSE, recursive = FALSE)
dataset_ids <- dataset_ids[dataset_ids != "" & !grepl("^\\.", dataset_ids)]
if (length(dataset_ids) == 0) stop("No dataset subfolders found in: ", first_dir)

save_dir <- save_dir[order(names(save_dir))]
n_datasets <- length(dataset_ids)
n_methods <- length(save_dir)

order_table <- data.frame(matrix(0, nrow = n_datasets, ncol = n_methods),
                          stringsAsFactors = FALSE)
colnames(order_table) <- names(save_dir)
rownames(order_table) <- dataset_ids

peak_location_table <- data.frame(matrix(NA, nrow = n_datasets, ncol = n_methods),
                                  stringsAsFactors = FALSE)
colnames(peak_location_table) <- names(save_dir)
rownames(peak_location_table) <- dataset_ids

# Main processing loop
for (ID in dataset_ids) {
  message(sprintf("Processing Dataset ID: %s", ID))
  all_rows <- list()

  for (method_name in names(save_dir)) {
    base_dir <- save_dir[[method_name]]

    if (method_name == "UniTVelo") {
      dir_path <- file.path(base_dir, ID)
      if (!dir.exists(dir_path) && dir.exists(unitvelo_subset_dir)) {
        alt_dir <- file.path(unitvelo_subset_dir, ID)
        if (dir.exists(alt_dir)) {
          dir_path <- alt_dir
          message(sprintf("Using UniTVelo subset data for %s", ID))
        }
      }
    } else {
      if (grepl("-", ID, fixed = TRUE)) {
        ID_prefix <- strsplit(ID, "-")[[1]][1]
        alt_path <- file.path(base_dir, ID_prefix)
        dir_path <- if (dir.exists(alt_path)) alt_path else file.path(base_dir, ID)
      } else {
        dir_path <- file.path(base_dir, ID)
      }
    }

    if (!dir.exists(dir_path)) next
    row_file <- file.path(dir_path, "velocity_confidence_row.csv")
    if (!file.exists(row_file)) next

    tryCatch({
      row_data <- read.csv(row_file, stringsAsFactors = FALSE) %>%
        mutate(method = method_name)
      all_rows[[method_name]] <- row_data
    }, error = function(e) {
      message(sprintf("Error reading %s for %s: %s", method_name, ID, e$message))
    })
  }

  if (length(all_rows) == 0) {
    message(sprintf("Skipping ID: %s - No data found.", ID))
    next
  }

  for (i in seq_along(all_rows)) {
    colnames(all_rows[[i]]) <- c("X", "confidence", "method")
    all_rows[[i]]$X <- as.character(all_rows[[i]]$X)
  }
  final_rows <- bind_rows(all_rows)
  final_rows$method <- as.character(final_rows$method)

  method_order_df <- final_rows %>%
    group_by(method) %>%
    summarise(
      peak_location = {
        if (n() > 1 && length(unique(confidence)) > 1) {
          d <- density(confidence, na.rm = TRUE)
          d$x[which.max(d$y)]
        } else {
          mean(confidence, na.rm = TRUE)
        }
      }
    ) %>%
    mutate(dist_to_one = abs(peak_location - 1)) %>%
    arrange(dist_to_one)

  ordered_method_names <- na.omit(method_order_df$method)
  all_possible_methods <- names(save_dir)
  missing_methods <- setdiff(all_possible_methods, ordered_method_names)
  missing_methods <- missing_methods[missing_methods != 'UniTVelo subset']
  final_y_axis_order <- c(ordered_method_names, missing_methods)

  for (rank in seq_along(ordered_method_names)) {
    method_name <- ordered_method_names[rank]
    order_table[ID, method_name] <- rank
  }

  for (i in seq_len(nrow(method_order_df))) {
    method_name <- method_order_df$method[i]
    peak_val <- method_order_df$peak_location[i]
    if (method_name %in% colnames(peak_location_table)) {
      peak_location_table[ID, method_name] <- peak_val
    }
  }

  if (nrow(final_rows) > 0) {
    final_rows$method <- factor(final_rows$method, levels = final_y_axis_order)

    p <- ggplot(final_rows, aes(x = confidence, y = method, fill = stat(x))) +
      geom_density_ridges_gradient(rel_min_height = 0.01, scale = 1.5) +
      scale_fill_viridis_c(name = "", option = "C") +
      scale_y_discrete(limits = rev(final_y_axis_order), name = "") +
      labs(title = NULL, x = "Velocity Confidence") +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, size = 18),
        axis.title.x = element_text(size = 12),
        axis.text.x = element_text(size = 10, angle = 45, hjust = 1),
        axis.title.y = element_text(size = 12),
        axis.text.y = element_text(size = 10)
      )

    pdf_file <- file.path(output_dir, paste0(ID, '_row_kde_full_method_sorted.pdf'))
    png_file <- file.path(output_dir, paste0(ID, '_row_kde_full_method_sorted.png'))
    ggsave(plot = p, filename = pdf_file, width = 10, height = 8)
    ggsave(plot = p, filename = png_file, width = 10, height = 8)
  }
}

final_order_table <- order_table %>% rownames_to_column("Dataset_ID")
final_peak_location_table <- peak_location_table %>% rownames_to_column("Dataset_ID")

write.csv(final_order_table,
          file = file.path(output_dir, "method_order_by_dataset.csv"),
          row.names = FALSE)

write.csv(final_peak_location_table,
          file = file.path(output_dir, "peak_location_by_dataset.csv"),
          row.names = FALSE)

message("Processing complete. Results saved to: ", normalizePath(output_dir))
