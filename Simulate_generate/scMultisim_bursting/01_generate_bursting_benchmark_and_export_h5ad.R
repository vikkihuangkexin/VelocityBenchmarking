suppressPackageStartupMessages({
  library(scMultiSim)
  library(ape)
  library(dplyr)
  library(readr)
  library(purrr)
  library(stringr)
  library(tibble)
})

# =========================================================
# 0. output dir
# =========================================================
OUTDIR <- "./bursting_benchmark"
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

# =========================================================
# 0.1 fixed tree for current B01-B10 benchmark
# IMPORTANT:
# replace ONLY this line if your current benchmark uses another tree object
# =========================================================
TREE_OBJ <- Phyla5()

# =========================================================
# 0.2 patch scMultiSim internal bug:
#     GRN + vary in c("kon", "koff")
# =========================================================
patch_scmultisim_geneIdentifyVectors <- function(verbose = TRUE) {
  ns <- asNamespace("scMultiSim")

  patched_fun <- function(seed, sim, options) {
    GRN <- sim$GRN
    N <- sim$N

    param_names <- c("kon", "koff", "s")

    giv <- lapply(seq_len(3), function(i) {
      .identityVectors(
        N$gene, N$cif,
        prob = OP("giv.prob"),
        mean = OP("giv.mean"),
        sd   = OP("giv.sd")
      )
    })
    names(giv) <- param_names

    has_grn <- !(is.null(GRN) || (length(GRN) == 1 && all(is.na(GRN))))

    if (has_grn) {
      giv$kon <- cbind(giv$kon, matrix(0, N$gene, N$reg_cif))
      giv$koff <- cbind(giv$koff, matrix(0, N$gene, N$reg_cif))

      if (isTRUE(sim$do_spatial)) {
        n_reg <- N$regulator + N$sp_regulators
        regu_list <- c(GRN$regulators, sim$sp_regulators)
        tgt_list  <- c(GRN$targets, sim$sp_targets)
      } else {
        n_reg <- N$regulator
        regu_list <- GRN$regulators
        tgt_list  <- GRN$targets
      }

      non_grn_gene <- setdiff(seq(N$gene), c(regu_list, tgt_list))
      giv$s[-non_grn_gene, ] <- 0

      s_total_cif <- ncol(giv$s)
      s_nd_cif    <- N$nd.cif[3]
      s_diff_cif  <- N$diff.cif[3]

      if (s_total_cif < 2) {
        stop("scMultiSim patch: total number of s CIF columns is < 2.")
      }

      regu_s_cif <- matrix(0, nrow = n_reg, ncol = s_total_cif)

      if (!is.na(s_diff_cif) && s_diff_cif >= 2) {
        s_pool <- s_nd_cif + seq_len(s_diff_cif)
      } else {
        s_pool <- seq_len(s_total_cif)
      }

      if (length(s_pool) < 2) {
        stop("scMultiSim patch: eligible s-CIF pool has < 2 columns.")
      }

      indices <- replicate(
        n_reg,
        sample(s_pool, 2, replace = FALSE)
      ) %>% as.vector()

      regu_s_cif[cbind(rep(seq_len(n_reg), each = 2), indices)] <- 2
      giv$s[regu_list, ] <- regu_s_cif

      regu_counts <- rowSums(GRN$geff != 0)
      stopifnot(sum(regu_counts > 0) == GRN$n_tgt)

      grn_eff <- na.omit(GRN$geff %*% giv$s[GRN$regulators, ] / regu_counts / 2)

      rg_row <- which(rowSums(giv$s[GRN$targets, ]) > 0)
      grn_eff[rg_row, ] <- grn_eff[rg_row, ] * 0.5 + giv$s[GRN$targets[rg_row], ] * 0.5
      grn_target <- giv$s[GRN$targets, ] <- grn_eff

      if (isTRUE(sim$do_spatial)) {
        sp_eff <- sim$sp_effect[sim$sp_targets, seq(N$sp_regulators)]
        sp_target <- na.omit(sp_eff %*% giv$s[sim$sp_regulators, ] / 2)

        sp_factor <- mean(grn_target[grn_target > 0]) / mean(sp_target[sp_target > 0])
        if (is.nan(sp_factor)) sp_factor <- 0

        sp_eff <- sp_target * sp_factor

        rg_row <- which(rowSums(giv$s[sim$sp_targets, ]) > 0)
        sp_eff[rg_row, ] <- sp_eff[rg_row, ] * 0.5 + giv$s[sim$sp_targets[rg_row], ] * 0.5
        giv$s[sim$sp_targets, ] <- sp_eff
      }

      giv$s[tgt_list, ][abs(giv$s[tgt_list, ]) < 0.2] <- 0
      giv$s[non_grn_gene, ] <- giv$s[non_grn_gene, ] * 3
    }

    giv
  }

  environment(patched_fun) <- ns
  assignInNamespace(".geneIdentifyVectors", patched_fun, ns = "scMultiSim")

  if (isTRUE(verbose)) {
    message("[PATCH] scMultiSim::.geneIdentifyVectors patched for GRN + kon/koff safety.")
  }

  invisible(TRUE)
}

patch_scmultisim_geneIdentifyVectors()

# =========================================================
# 1. helper: choose num.cifs
# =========================================================
choose_num_cifs <- function(num_genes) {
  if (num_genes <= 1000) return(35L)
  if (num_genes <= 2000) return(50L)
  if (num_genes <= 4000) return(80L)
  if (num_genes <= 8000) return(100L)
  return(120L)
}

# =========================================================
# 2. helper: dataset dir name
#    example: B10_all_grn1139_c2200_g2500
# =========================================================
make_dataset_dirname <- function(dataset_id, vary, grn_name, num_cells, num_genes) {
  grn_short <- case_when(
    grn_name == "GRN_params_100"  ~ "grn100",
    grn_name == "GRN_params_1139" ~ "grn1139",
    TRUE ~ str_replace_all(grn_name, "[^A-Za-z0-9]+", "")
  )

  paste0(
    dataset_id, "_", vary, "_", grn_short,
    "_c", num_cells, "_g", num_genes
  )
}

# =========================================================
# 3. GRN factory
# =========================================================
data(GRN_params_100)
data(GRN_params_1139)

make_grn <- function(grn_name) {
  if (grn_name == "GRN_params_100")  return(GRN_params_100)
  if (grn_name == "GRN_params_1139") return(GRN_params_1139)
  stop("Unknown grn_name: ", grn_name)
}

# =========================================================
# 4. helper: readable reference for trajectory evaluation
# =========================================================
make_reference <- function(results, linear_bins = 5, branch_bins = 3) {
  meta <- as.data.frame(results$cell_meta)

  if (!"pop" %in% colnames(meta)) {
    stop("results$cell_meta must contain column 'pop'")
  }

  if (!"cell_id" %in% colnames(meta)) {
    meta$cell_id <- colnames(results$counts)
  }

  if ("pseudotime" %in% colnames(meta)) {
    meta$cell_time_ref <- meta$pseudotime
  } else if ("cell_time" %in% colnames(meta)) {
    meta$cell_time_ref <- meta$cell_time
  } else if (!is.null(results$cell_time)) {
    meta$cell_time_ref <- results$cell_time
  } else {
    stop("Cannot find pseudotime/cell_time in results")
  }

  meta$pop_raw <- as.character(meta$pop)

  edge_ref <- meta %>%
    group_by(pop_raw) %>%
    summarise(
      n_cells = n(),
      t_min = min(cell_time_ref),
      t_med = median(cell_time_ref),
      t_max = max(cell_time_ref),
      .groups = "drop"
    ) %>%
    arrange(t_med) %>%
    mutate(edge_id = sprintf("edge%02d", row_number()))

  meta <- meta %>%
    left_join(edge_ref, by = "pop_raw")

  n_edges <- nrow(edge_ref)
  is_linear_like <- n_edges <= 2

  if (is_linear_like) {
    label_pool <- c("sA", "sB", "sBmid", "sC", "sD")
    nb <- min(linear_bins, length(label_pool))

    ord <- order(meta$cell_time_ref, meta$cell_id)
    rank_vec <- integer(nrow(meta))
    rank_vec[ord] <- seq_len(nrow(meta))

    qbreaks <- quantile(
      rank_vec,
      probs = seq(0, 1, length.out = nb + 1),
      type = 1,
      na.rm = TRUE
    )
    qbreaks[1] <- min(rank_vec)
    qbreaks[length(qbreaks)] <- max(rank_vec)
    qbreaks <- unique(qbreaks)

    if (length(qbreaks) < 3) {
      qbreaks <- seq(min(rank_vec), max(rank_vec), length.out = nb + 1)
    }

    meta$milestone_ref <- cut(
      rank_vec,
      breaks = qbreaks,
      include.lowest = TRUE,
      labels = label_pool[seq_len(length(qbreaks) - 1)]
    )

    milestone_levels <- levels(meta$milestone_ref)
    milestone_graph <- tibble(
      from = milestone_levels[-length(milestone_levels)],
      to   = milestone_levels[-1]
    )

  } else {
    meta <- meta %>%
      group_by(pop_raw) %>%
      arrange(cell_time_ref, cell_id, .by_group = TRUE) %>%
      mutate(within_edge_rank = row_number()) %>%
      ungroup()

    meta <- meta %>%
      group_by(pop_raw) %>%
      mutate(
        within_edge_bin = {
          n <- n()
          k <- min(branch_bins, max(2L, floor(n / 20)))
          rk <- row_number()
          qbreaks <- quantile(
            rk,
            probs = seq(0, 1, length.out = k + 1),
            type = 1,
            na.rm = TRUE
          )
          qbreaks[1] <- min(rk)
          qbreaks[length(qbreaks)] <- max(rk)
          qbreaks <- unique(qbreaks)
          if (length(qbreaks) < 3) {
            qbreaks <- seq(min(rk), max(rk), length.out = k + 1)
          }
          as.integer(cut(rk, breaks = qbreaks, include.lowest = TRUE, labels = FALSE))
        }
      ) %>%
      ungroup()

    edge_alias <- edge_ref %>%
      transmute(pop_raw, edge_alias = edge_id)

    meta <- meta %>%
      left_join(edge_alias, by = "pop_raw") %>%
      mutate(
        milestone_ref = paste0(edge_alias, "_m", within_edge_bin)
      )

    within_edge_graph <- meta %>%
      distinct(edge_alias, within_edge_bin, milestone_ref) %>%
      arrange(edge_alias, within_edge_bin) %>%
      group_by(edge_alias) %>%
      mutate(next_m = lead(milestone_ref)) %>%
      filter(!is.na(next_m)) %>%
      transmute(from = milestone_ref, to = next_m) %>%
      ungroup()

    parts_mat <- str_split_fixed(edge_ref$pop_raw, "_", 2)
    edge_nodes <- edge_ref %>%
      mutate(
        parent_node = parts_mat[, 1],
        child_node  = parts_mat[, 2]
      ) %>%
      select(pop_raw, edge_id, parent_node, child_node)

    edge_first_last <- meta %>%
      group_by(pop_raw, edge_alias) %>%
      summarise(
        first_m = milestone_ref[which.min(within_edge_bin)],
        last_m  = milestone_ref[which.max(within_edge_bin)],
        .groups = "drop"
      )

    edge_nodes <- edge_nodes %>%
      left_join(edge_first_last, by = "pop_raw")

    between_edge_graph <- list()
    idx <- 1L

    for (i in seq_len(nrow(edge_nodes))) {
      this_child <- edge_nodes$child_node[i]
      this_last  <- edge_nodes$last_m[i]

      downstream <- edge_nodes %>%
        filter(parent_node == this_child)

      if (nrow(downstream) > 0) {
        between_edge_graph[[idx]] <- tibble(
          from = rep(this_last, nrow(downstream)),
          to   = downstream$first_m
        )
        idx <- idx + 1L
      }
    }

    between_edge_graph <- bind_rows(between_edge_graph)

    milestone_graph <- bind_rows(within_edge_graph, between_edge_graph) %>%
      distinct()
  }

  meta$milestone_ref <- as.character(meta$milestone_ref)

  list(
    cell_meta = meta,
    edge_ref = edge_ref,
    milestone_graph = milestone_graph
  )
}

# =========================================================
# 5. lineage helpers
# =========================================================
enumerate_lineages <- function(milestone_graph) {
  mg <- milestone_graph %>% distinct(from, to)

  if (nrow(mg) == 0) {
    empty_lineages <- tibble(
      lineage_id = character(),
      root_milestone = character(),
      leaf_milestone = character(),
      n_milestones = integer(),
      lineage = character()
    )
    empty_lineage_milestones <- tibble(
      lineage_id = character(),
      milestone_order = integer(),
      milestone_id = character()
    )
    empty_pairs <- tibble(
      lineage_id = character(),
      ancestor = character(),
      descendant = character(),
      step_distance = integer()
    )
    return(list(
      lineages = empty_lineages,
      lineage_milestones = empty_lineage_milestones,
      ancestor_descendant_pairs = empty_pairs
    ))
  }

  nodes <- sort(unique(c(mg$from, mg$to)))

  indeg <- setNames(integer(length(nodes)), nodes)
  outdeg <- setNames(integer(length(nodes)), nodes)

  for (i in seq_len(nrow(mg))) {
    indeg[mg$to[i]] <- indeg[mg$to[i]] + 1L
    outdeg[mg$from[i]] <- outdeg[mg$from[i]] + 1L
  }

  roots <- names(indeg[indeg == 0])
  if (length(roots) == 0) roots <- nodes[1]

  children <- split(mg$to, mg$from)

  dfs_paths <- function(node, path) {
    nxt <- children[[node]]
    if (is.null(nxt) || length(nxt) == 0) {
      return(list(path))
    }
    out <- list()
    for (kid in nxt) {
      out <- c(out, dfs_paths(kid, c(path, kid)))
    }
    out
  }

  path_list <- list()
  idx <- 1L
  for (r in roots) {
    rr <- dfs_paths(r, c(r))
    for (p in rr) {
      path_list[[idx]] <- p
      idx <- idx + 1L
    }
  }

  lineage_ids <- sprintf("L%02d", seq_along(path_list))

  lineages <- tibble(
    lineage_id = lineage_ids,
    root_milestone = vapply(path_list, function(x) x[1], character(1)),
    leaf_milestone = vapply(path_list, function(x) x[length(x)], character(1)),
    n_milestones = lengths(path_list),
    lineage = vapply(path_list, function(x) paste(x, collapse = " -> "), character(1))
  )

  lineage_milestones <- map2_dfr(
    lineage_ids, path_list,
    ~ tibble(
      lineage_id = .x,
      milestone_order = seq_along(.y),
      milestone_id = .y
    )
  )

  ancestor_descendant_pairs <- map2_dfr(
    lineage_ids, path_list,
    function(id, p) {
      if (length(p) < 2) {
        return(tibble(
          lineage_id = character(),
          ancestor = character(),
          descendant = character(),
          step_distance = integer()
        ))
      }

      out <- list()
      k <- 1L
      for (i in seq_len(length(p) - 1L)) {
        for (j in (i + 1L):length(p)) {
          out[[k]] <- tibble(
            lineage_id = id,
            ancestor = p[i],
            descendant = p[j],
            step_distance = j - i
          )
          k <- k + 1L
        }
      }
      bind_rows(out)
    }
  )

  list(
    lineages = lineages,
    lineage_milestones = lineage_milestones,
    ancestor_descendant_pairs = ancestor_descendant_pairs
  )
}

# =========================================================
# 6. cell table / gene role / summary / h5ad
# =========================================================
make_cell_milestone_table <- function(ref, lineage_info) {
  milestone_to_lineages <- lineage_info$lineage_milestones %>%
    group_by(milestone_id) %>%
    summarise(
      lineage_id = paste(unique(lineage_id), collapse = ";"),
      .groups = "drop"
    )

  ref$cell_meta %>%
    transmute(
      cell_id = cell_id,
      pop_raw = pop_raw,
      edge_id = edge_id,
      cell_time_ref = cell_time_ref,
      milestone_id = milestone_ref
    ) %>%
    left_join(milestone_to_lineages, by = "milestone_id")
}

make_gene_role_summary <- function(results, grn_obj) {
  gene_ids <- rownames(results$counts)
  if (is.null(gene_ids)) {
    gene_ids <- paste0("G", seq_len(nrow(results$counts)))
  }

  n_gene <- length(gene_ids)
  is_regulator <- rep(FALSE, n_gene)
  is_target <- rep(FALSE, n_gene)

  if (!(length(grn_obj) == 1 && all(is.na(grn_obj)))) {
    reg_idx <- intersect(as.integer(grn_obj$regulators), seq_len(n_gene))
    tgt_idx <- intersect(as.integer(grn_obj$targets), seq_len(n_gene))
    is_regulator[reg_idx] <- TRUE
    is_target[tgt_idx] <- TRUE
  }

  role <- case_when(
    is_regulator & is_target ~ "regulator_target",
    is_regulator ~ "regulator",
    is_target ~ "target",
    TRUE ~ "other"
  )

  tibble(
    gene_index = seq_len(n_gene),
    gene_id = gene_ids,
    is_regulator = is_regulator,
    is_target = is_target,
    role = role
  )
}

make_dataset_summary <- function(plan_row, ds_name, results, ref, lineage_info, gene_role_summary) {
  tibble(
    dataset_name = ds_name,
    dataset_id = plan_row$dataset_id,
    grn_name = plan_row$grn_name,
    num_cells_requested = plan_row$num_cells,
    num_genes_requested = plan_row$num_genes,
    num_cells_observed = ncol(results$counts),
    num_genes_observed = nrow(results$counts),
    vary = plan_row$vary,
    bimod = plan_row$bimod,
    scale_s = plan_row$scale_s,
    n_edges = nrow(ref$edge_ref),
    n_milestones = dplyr::n_distinct(ref$cell_meta$milestone_ref),
    n_lineages = nrow(lineage_info$lineages),
    n_regulators = sum(gene_role_summary$is_regulator),
    n_targets = sum(gene_role_summary$is_target),
    note = plan_row$note
  )
}

write_h5ad_export <- function(results, cell_milestone_table, gene_role_summary, outfile) {
  needed <- c("zellkonverter", "SingleCellExperiment", "S4Vectors", "SummarizedExperiment")
  missing_pkgs <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]

  if (length(missing_pkgs) > 0) {
    stop(
      "To write res.h5ad, please install packages: ",
      paste(missing_pkgs, collapse = ", ")
    )
  }

  counts_mat <- as.matrix(results$counts)

  # ---------- make cell metadata order consistent with counts ----------
  cell_df <- as.data.frame(cell_milestone_table)

  if (!"cell_id" %in% colnames(cell_df)) {
    stop("cell_milestone_table must contain column 'cell_id'.")
  }
  cell_df$cell_id <- as.character(cell_df$cell_id)

  if (!is.null(colnames(counts_mat))) {
    miss_cells <- setdiff(colnames(counts_mat), cell_df$cell_id)
    if (length(miss_cells) > 0) {
      stop("cell_milestone_table is missing ", length(miss_cells), " cells from results$counts.")
    }
    cell_df <- cell_df[match(colnames(counts_mat), cell_df$cell_id), , drop = FALSE]
  } else {
    if (nrow(cell_df) != ncol(counts_mat)) {
      stop("nrow(cell_milestone_table) does not match ncol(results$counts).")
    }
  }

  # ---------- make gene metadata order consistent with counts ----------
  gene_df <- as.data.frame(gene_role_summary)

  if (!"gene_id" %in% colnames(gene_df)) {
    stop("gene_role_summary must contain column 'gene_id'.")
  }
  gene_df$gene_id <- as.character(gene_df$gene_id)

  if (!is.null(rownames(counts_mat))) {
    miss_genes <- setdiff(rownames(counts_mat), gene_df$gene_id)
    if (length(miss_genes) > 0) {
      stop("gene_role_summary is missing ", length(miss_genes), " genes from results$counts.")
    }
    gene_df <- gene_df[match(rownames(counts_mat), gene_df$gene_id), , drop = FALSE]
  } else {
    if (nrow(gene_df) != nrow(counts_mat)) {
      stop("nrow(gene_role_summary) does not match nrow(results$counts).")
    }
  }

  # ---------- force identical dimnames ----------
  gene_ids <- gene_df$gene_id
  cell_ids <- cell_df$cell_id
  dimnames(counts_mat) <- list(gene_ids, cell_ids)

  rownames(cell_df) <- cell_ids
  rownames(gene_df) <- gene_ids

  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = counts_mat),
    colData = S4Vectors::DataFrame(cell_df),
    rowData = S4Vectors::DataFrame(gene_df)
  )

  add_assay_if_present <- function(sce_obj, assay_name, candidate_name) {
    mat <- results[[candidate_name]]
    if (!is.null(mat) && length(dim(mat)) == 2 && all(dim(mat) == dim(counts_mat))) {
      mat <- as.matrix(mat)
      dimnames(mat) <- dimnames(counts_mat)
      SummarizedExperiment::assay(sce_obj, assay_name) <- mat
    }
    sce_obj
  }

  sce <- add_assay_if_present(sce, "spliced", "counts_s")
  sce <- add_assay_if_present(sce, "unspliced", "counts_u")
  sce <- add_assay_if_present(sce, "velocity", "velocity")

  zellkonverter::writeH5AD(sce, outfile)
}
# =========================================================
# 7. simulation plan
# =========================================================
dataset_plan <- tribble(
  ~dataset_id, ~grn_name,          ~num_cells, ~num_genes, ~vary,   ~bimod, ~scale_s, ~note,
  "B01",       "GRN_params_100",      1000L,      500L,    "s",      0.00,    1.00,   "size-driven baseline, smallest stable setting",
  "B02",       "GRN_params_100",      1200L,      800L,    "s",      0.08,    1.05,   "size-driven, mild bimod",
  "B03",       "GRN_params_100",      1500L,     1000L,    "s",      0.15,    1.10,   "size-driven, stronger but still mild",
  "B04",       "GRN_params_1139",     1200L,     1500L,    "s",      0.10,    1.05,   "size-driven with larger GRN",
  "B05",       "GRN_params_1139",     1500L,     2000L,    "kon",    0.00,    1.00,   "frequency-driven clean",
  "B06",       "GRN_params_1139",     1500L,     1500L,    "kon",    0.08,    1.05,   "frequency-driven mild bimod",
  "B07",       "GRN_params_100",      1200L,      800L,    "koff",   0.00,    1.00,   "duration-driven clean",
  "B08",       "GRN_params_1139",     1500L,     1500L,    "koff",   0.08,    1.05,   "duration-driven mild bimod",
  "B09",       "GRN_params_1139",     1800L,     2000L,    "all",    0.08,    1.05,   "mixed mild regime",
  "B10",       "GRN_params_1139",     2200L,     2500L,    "all",    0.12,    1.10,   "mixed mild harder regime"
)

# =========================================================
# 8. helper: finished dataset?
# =========================================================
is_dataset_done <- function(ds_dir) {
  required_files <- c(
    "ancestor_descendant_pairs.csv",
    "cell_milestone_table.csv",
    "dataset_summary.csv",
    "gene_role_summary.csv",
    "lineage_milestones.csv",
    "lineages.csv",
    "milestone_graph.csv",
    "res.h5ad",
    "res.rds"
  )
  all(file.exists(file.path(ds_dir, required_files)))
}

# =========================================================
# 9. one simulation
# =========================================================
simulate_one <- function(plan_row, outdir = OUTDIR) {
  stopifnot(nrow(plan_row) == 1)

  ds_name <- make_dataset_dirname(
    dataset_id = plan_row$dataset_id,
    vary = plan_row$vary,
    grn_name = plan_row$grn_name,
    num_cells = plan_row$num_cells,
    num_genes = plan_row$num_genes
  )

  ds_dir <- file.path(outdir, ds_name)
  dir.create(ds_dir, showWarnings = FALSE, recursive = TRUE)

  if (is_dataset_done(ds_dir)) {
    message("[SKIP] already finished: ", ds_name)
    return(invisible(NULL))
  }

  grn_obj <- make_grn(plan_row$grn_name)
  ncif <- choose_num_cifs(plan_row$num_genes)

  options <- list(
    rand.seed = 1,
    GRN = grn_obj,
    num.cells = plan_row$num_cells,
    num.genes = plan_row$num_genes,
    num.cifs = ncif,
    tree = TREE_OBJ,
    vary = plan_row$vary,
    bimod = plan_row$bimod,
    scale.s = plan_row$scale_s,
    do.velocity = TRUE,
    beta = 0.4,
    d = 1,
    num.cycles = 4,
    cycle.len = 1
  )

  message("====================================================")
  message("Running: ", ds_name)
  message("cells = ", plan_row$num_cells,
          ", genes = ", plan_row$num_genes,
          ", vary = ", plan_row$vary,
          ", GRN = ", plan_row$grn_name,
          ", num.cifs = ", ncif)

  results <- sim_true_counts(options)

  ref <- make_reference(results)
  lineage_info <- enumerate_lineages(ref$milestone_graph)
  cell_milestone_table <- make_cell_milestone_table(ref, lineage_info)
  gene_role_summary <- make_gene_role_summary(results, grn_obj)
  dataset_summary <- make_dataset_summary(plan_row, ds_name, results, ref, lineage_info, gene_role_summary)

  saveRDS(results, file.path(ds_dir, "res.rds"))

  write_csv(ref$milestone_graph, file.path(ds_dir, "milestone_graph.csv"))
  write_csv(cell_milestone_table, file.path(ds_dir, "cell_milestone_table.csv"))
  write_csv(lineage_info$lineages, file.path(ds_dir, "lineages.csv"))
  write_csv(lineage_info$lineage_milestones, file.path(ds_dir, "lineage_milestones.csv"))
  write_csv(lineage_info$ancestor_descendant_pairs, file.path(ds_dir, "ancestor_descendant_pairs.csv"))
  write_csv(gene_role_summary, file.path(ds_dir, "gene_role_summary.csv"))
  write_csv(dataset_summary, file.path(ds_dir, "dataset_summary.csv"))

  write_h5ad_export(
    results = results,
    cell_milestone_table = cell_milestone_table,
    gene_role_summary = gene_role_summary,
    outfile = file.path(ds_dir, "res.h5ad")
  )

  invisible(results)
}

# =========================================================
# 10. safe runner
# =========================================================
run_one_safe <- function(i, outdir = OUTDIR) {
  plan_row <- dataset_plan[i, , drop = FALSE]

  tryCatch(
    {
      simulate_one(plan_row, outdir = outdir)
      tibble(
        dataset_id = plan_row$dataset_id,
        status = "OK",
        message = NA_character_
      )
    },
    error = function(e) {
      msg <- conditionMessage(e)
      message("[ERROR] ", plan_row$dataset_id, " :: ", msg)

      tibble(
        dataset_id = plan_row$dataset_id,
        status = "ERROR",
        message = msg
      )
    }
  )
}

# =========================================================
# 11. run all datasets
# =========================================================
write_csv(dataset_plan, file.path(OUTDIR, "dataset_plan.csv"))

run_log <- purrr::map_dfr(seq_len(nrow(dataset_plan)), run_one_safe, outdir = OUTDIR)
write_csv(run_log, file.path(OUTDIR, "run_log.csv"))

error_log <- run_log %>% filter(status != "OK")
write_csv(error_log, file.path(OUTDIR, "error_log.csv"))

message("All simulations finished.")
