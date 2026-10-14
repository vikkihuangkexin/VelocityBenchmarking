#!/usr/bin/env Rscript
# Independent Scalability and Usability, using standalone CSV inputs only.
# Sc: equal means of reverse runtime/RAM sample-z scores over 68 top tasks.
# U: 24 item scores, S7 subweights, category weights .25/.25/.20/.15/.15.
# The two panels use independent descending orders and exact average ties.
# Original BluGrn heatmaps, orange bubbles, dimensions and geometry are kept.
# U = .25 Installability + .25 Code quality + .20 Usage guidance + .15 Maintenance + .15 Publication quality。

args <- commandArgs(trailingOnly = TRUE)
PROJECT <- normalizePath(if (length(args)) args[[1]] else ".", winslash = "/", mustWork = TRUE)
OUT <- if (length(args) >= 2L) args[[2]] else file.path(PROJECT, "results_run")
OUT <- normalizePath(OUT, winslash = "/", mustWork = FALSE)
PANEL_HEIGHT_IN <- 16
if (length(args) > 2L) stop("Usage: Rscript SCRIPT PROJECT_OR_SOURCE_ROOT [NEW_OUTPUT]")
if (dir.exists(OUT) && length(list.files(OUT, all.files=TRUE, no..=TRUE))) stop("Use a new empty output directory.")

find_input <- function(local) {
  path <- file.path(PROJECT, "data", local)
  if (!file.exists(path)) stop("Missing locked input: ", local)
  normalizePath(path, winslash = "/", mustWork = TRUE)
}
input_resource <- find_input("resource_clean_long.csv")
input_items <- find_input("usability_item_contributions.csv")
input_rules <- find_input("usability_rules_S7.csv")
input_palette <- find_input("method_colors.csv")
read_csv <- function(path) read.csv(path, check.names=FALSE, stringsAsFactors=FALSE, na.strings=c("NA","NaN",""),fileEncoding="UTF-8-BOM")
palette <- read_csv(input_palette)
palette <- palette[order(palette$method_order), ]
methods <- palette$method
N <- length(methods)
normalize_method <- function(x) gsub("[^a-z0-9]+", "", tolower(trimws(x)))
method_keys <- normalize_method(methods)
excluded_keys <- c("regionvelocity", "phylovelo")
stopifnot(N == 28L, identical(palette$method_order, seq_len(N)),
          !anyDuplicated(methods), !anyDuplicated(method_keys),
          !anyDuplicated(toupper(palette$hex)),
          !any(method_keys %in% excluded_keys))
method_colors <- setNames(toupper(palette$hex), methods)
resource_all <- read_csv(input_resource)
resource_all$method_key <- normalize_method(resource_all$method)
resource_excluded <- resource_all[resource_all$method_key %in% excluded_keys, ]
resource_unknown <- resource_all[
  !resource_all$method_key %in% c(method_keys, excluded_keys), ]
if (nrow(resource_unknown)) {
  stop("Unknown resource methods: ",
       paste(sort(unique(resource_unknown$method)), collapse=", "))
}
resource_all <- resource_all[resource_all$method_key %in% method_keys, ]
resource <- resource_all[resource_all$measure %in% c("runtime_seconds","ram_gb"), ]
stopifnot(!anyDuplicated(resource[c("method","protocol","measure","dataset_id")]),
          is.numeric(resource$clean_value), !any(resource$clean_value < 0, na.rm=TRUE),
          all(resource$unit[resource$measure=="runtime_seconds"]=="seconds"),
          all(resource$unit[resource$measure=="ram_gb"]=="GB"))
if (any(!is.finite(resource$clean_value) & !is.na(resource$clean_value))) stop("Non-finite cost values")
resource$display_value <- ifelse(resource$measure=="runtime_seconds", resource$clean_value/60, resource$clean_value)

task_z <- resource[resource$protocol=="top_scRNA", ]
stopifnot(length(unique(task_z$dataset_id))==68L,
          setequal(unique(task_z$measure), c("runtime_seconds","ram_gb")))
task_z$directed_value <- -task_z$clean_value
task_z$z <- NA_real_; task_z$task_mean_directed <- NA_real_; task_z$task_sd <- NA_real_
task_z$task_n_valid <- 0L; task_z$z_status <- "missing_input"
task_indices <- split(seq_len(nrow(task_z)), paste(task_z$measure,task_z$dataset_id,sep="::"))
for (idx in task_indices) {
  good <- idx[is.finite(task_z$clean_value[idx])]; n <- length(good)
  directed <- -task_z$clean_value[good]
  mu <- if (n) mean(directed) else NA_real_; sigma <- if (n>=2L) sd(directed) else NA_real_
  task_z$task_n_valid[idx] <- n; task_z$task_mean_directed[idx] <- mu; task_z$task_sd[idx] <- sigma
  if (n < 2L) task_z$z_status[good] <- "too_few_valid_methods"
  else if (!is.finite(sigma) || sigma==0) task_z$z_status[good] <- "zero_task_sd"
  else {task_z$z[good] <- (directed-mu)/sigma; task_z$z_status[good] <- "valid_z"}
}
valid_mean <- function(v) {v <- v[is.finite(v)]; if (length(v)) mean(v) else NA_real_}
sc <- do.call(rbind,lapply(seq_along(methods),function(i) {
  m <- methods[[i]]; p <- task_z[task_z$method==m, ]
  t <- p$z[p$measure=="runtime_seconds"]; r <- p$z[p$measure=="ram_gb"]
  t_ids <- p$dataset_id[p$measure=="runtime_seconds"]
  r_ids <- p$dataset_id[p$measure=="ram_gb"]
  T <- valid_mean(t); R <- valid_mean(r); comps <- c(T,R); n <- sum(is.finite(comps))
  data.frame(method=m,fixed_method_order=i,T=T,R=R,Sc=valid_mean(comps),
    T_n_valid=sum(is.finite(t)),T_n_expected=length(t),
    R_n_valid=sum(is.finite(r)),R_n_expected=length(r),
    n_valid_tasks_total=sum(is.finite(t))+sum(is.finite(r)),
    n_expected_tasks_total=length(t)+length(r),Sc_n_components=n,
    effective_weight_T=if (is.finite(T)) 1/n else NA_real_,
    effective_weight_R=if (is.finite(R)) 1/n else NA_real_,
    missing_runtime_tasks=paste(t_ids[!is.finite(t)],collapse=";"),
    missing_ram_tasks=paste(r_ids[!is.finite(r)],collapse=";"),
    score_status=if (n==2L) "complete_T_R" else if (n==1L) "one_component_available" else "missing_T_R",
    missing_reason=if (n) "" else "no_valid_runtime_or_ram_component")
}))
sc$Sc_rank <- rank(-sc$Sc,ties.method="average",na.last="keep")
sc$Sc_reversed_rank <- N+1-sc$Sc_rank
sc <- sc[order(is.na(sc$Sc),-sc$Sc,sc$fixed_method_order), ]
sc$display_order <- seq_len(N)
sc$score <- sc$Sc
sc$rank <- sc$Sc_rank
sc_order <- sc$method

items_all <- read_csv(input_items)
items_all$method_key <- normalize_method(items_all$method)
items_excluded <- items_all[items_all$method_key %in% excluded_keys, ]
items_unknown <- items_all[
  !items_all$method_key %in% c(method_keys, excluded_keys), ]
if (nrow(items_unknown)) {
  stop("Unknown usability methods: ",
       paste(sort(unique(items_unknown$method)), collapse=", "))
}
items <- items_all[items_all$method_key %in% method_keys, ]
rules <- read_csv(input_rules)
category_order <- c("Installability","Code quality","Usage guidance","Maintenance","Publication quality")
category_weights <- setNames(c(.25,.25,.20,.15,.15),category_order)
expected_counts <- setNames(c(6L,5L,5L,5L,3L),category_order)
stopifnot(!anyDuplicated(items[c("method","category","subcategory")]),
          !anyDuplicated(rules[c("category","subcategory")]),
          all(items$category %in% category_order),is.numeric(items$subscore),is.numeric(items$subcategory_weight),
          !any(items$subscore < 0 | items$subscore > 10,na.rm=TRUE),
          !any(items$subcategory_weight < 0,na.rm=TRUE),
          setequal(rules$category, category_order))
for (c in category_order) {
  rule <- rules[rules$category==c, ]
  stopifnot(nrow(rule)==expected_counts[[c]],
            abs(sum(rule$subcategory_weight)-1)<=1e-12,
            all(abs(rule$category_weight-category_weights[[c]])<=1e-12))
}
items$recomputed_contribution <- items$subscore*items$subcategory_weight
categories <- do.call(rbind,lapply(methods,function(m) do.call(rbind,lapply(category_order,function(c) {
  p <- items[items$method==m & items$category==c, ]
  rule <- rules[rules$category==c, ]
  matched <- match(rule$subcategory, p$subcategory)
  subcategories_match <- nrow(p)==nrow(rule) && !anyNA(matched) &&
    all(abs(p$subcategory_weight[matched]-rule$subcategory_weight)<=1e-12)
  weight_ok <- !anyNA(p$subcategory_weight) && abs(sum(p$subcategory_weight)-1)<=1e-12
  if (nrow(p) && any(abs(p$category_weight-category_weights[[c]])>1e-12,na.rm=TRUE)) stop("Category weight mismatch: ",m,"/",c)
  complete <- nrow(p)==expected_counts[[c]] && !anyNA(p$subscore) &&
    weight_ok && subcategories_match
  old_scores <- unique(p$old_category_score[is.finite(p$old_category_score)])
  if (length(old_scores)>1L) stop("Conflicting old category scores: ",m,"/",c)
  old_score <- if (length(old_scores)) old_scores[[1]] else NA_real_
  score <- if (complete) sum(p$recomputed_contribution) else NA_real_
  missing_subcategories <- setdiff(rule$subcategory,p$subcategory)
  unexpected_subcategories <- setdiff(p$subcategory,rule$subcategory)
  reason <- if (complete) "" else paste(c(
    if (length(missing_subcategories)) paste0("missing:",paste(missing_subcategories,collapse="|")),
    if (length(unexpected_subcategories)) paste0("unexpected:",paste(unexpected_subcategories,collapse="|")),
    if (!weight_ok) "subcategory_weights_not_one",
    if (!subcategories_match) "S7_rule_mismatch",
    if (anyNA(p$subscore)) "missing_subscore"),collapse=";")
  data.frame(method=m,category=c,expected_items=expected_counts[[c]],observed_items=nrow(p),
    valid_subscores=sum(is.finite(p$subscore)),
    subcategory_weight_sum=if (nrow(p)) sum(p$subcategory_weight) else NA_real_,
    score=score,old_category_score=old_score,
    old_category_score_difference=score-old_score,
    category_weight=category_weights[[c]],category_complete=complete,
    missing_subcategories=paste(missing_subcategories,collapse=";"),
    unexpected_subcategories=paste(unexpected_subcategories,collapse=";"),
    missing_reason=reason)
}))))
u <- do.call(rbind,lapply(seq_along(methods),function(i) {
  m<-methods[[i]];p<-categories[categories$method==m, ]; p<-p[match(category_order,p$category), ]
  complete<-nrow(p)==5L && all(is.finite(p$score))
  missing_categories<-p$category[!is.finite(p$score)]
  data.frame(method=m,fixed_method_order=i,
    U=if (complete) sum(p$score*p$category_weight) else NA_real_,
    U_complete=complete,n_valid_categories=sum(is.finite(p$score)),
    n_expected_categories=5L,
    missing_categories=paste(missing_categories,collapse=";"),
    score_status=if (complete) "complete_24_items" else "incomplete_items",
    missing_reason=if (complete) "" else paste0("missing_categories:",paste(missing_categories,collapse="|")))
}))
u$U_rank <- rank(-u$U,ties.method="average",na.last="keep")
u <- u[order(is.na(u$U),-u$U,u$fixed_method_order), ]
u$display_order<-seq_len(N)
u$score<-u$U
u$rank<-u$U_rank
u_order<-u$method
categories$display_order<-match(categories$method,u_order)
categories<-categories[order(categories$display_order,match(categories$category,category_order)), ]
min_score <- min(categories$score,na.rm=TRUE); categories$Size<-(categories$score-min_score+1)^2

structure_order <- c("linear-simple","bifurcating","consecutive-bifurcating","bifurcating-loop","trifurcating","cycle-simple","disconnected")
heat_source_detail <- list()
structure_source_detail <- list()
annotate_source_rows <- function(rows,balance_columns,condition,condition_label,group,measure) {
  balance_unit <- do.call(paste,c(rows[balance_columns],sep="::"))
  data.frame(
    method=rows$method,
    display_order=match(rows$method,sc_order),
    condition=condition,
    column_label=condition_label,
    condition_group=group,
    measure=measure,
    unit=if (measure=="runtime_seconds") "min" else "GB",
    balance_unit=balance_unit,
    protocol=rows$protocol,
    dataset_id=rows$dataset_id,
    structure=rows$structure,
    cells=rows$cells,
    genes=rows$genes,
    raw_cost_value=rows$clean_value,
    display_value=rows$display_value,
    value_status=rows$value_status,
    missing_reason=rows$missing_reason,
    source_path=rows$source_path,
    source_locator=rows$source_locator,
    is_numeric=is.finite(rows$clean_value),
    stringsAsFactors=FALSE)
}
summarise_cost <- function(rows,balance_columns,condition,condition_label,group,measure) {
  if (!nrow(rows)) stop("No tasks for ",condition)
  groups <- do.call(paste,c(rows[balance_columns],sep="::"))
  expected_groups <- sort(unique(groups))
  expected_task_ids <- sort(unique(rows$dataset_id))
  do.call(rbind,lapply(sc_order,function(m) {
    q<-which(rows$method==m)
    v<-vapply(expected_groups,function(g) valid_mean(rows$display_value[q[groups[q]==g]]),numeric(1))
    ok<-is.finite(v)
    numeric_q<-q[is.finite(rows$display_value[q])]
    data.frame(method=m,display_order=match(m,sc_order),condition=condition,column_label=condition_label,
      condition_group=group,measure=measure,unit=if (measure=="runtime_seconds") "min" else "GB",
      value=valid_mean(v),n_balance_valid=sum(ok),n_balance_expected=length(expected_groups),
      n_records_numeric=length(numeric_q),n_records_expected=length(expected_task_ids),
      structures_expected=paste(sort(unique(rows$structure)),collapse=";"),
      structures_numeric=paste(sort(unique(rows$structure[q[is.finite(rows$display_value[q])]])),collapse=";"),
      balance_units_numeric=paste(expected_groups[ok],collapse=";"),
      task_ids_expected=paste(expected_task_ids,collapse=";"),
      task_ids_numeric=paste(sort(unique(rows$dataset_id[numeric_q])),collapse=";"),
      source_coordinates_numeric=paste(
        paste(rows$source_path[numeric_q],rows$source_locator[numeric_q],sep="::"),
        collapse=";"),
      status=if (!any(ok)) "NA_no_values" else if (all(ok)) "complete" else "partial_available_mean")
  }))
}
scales<-data.frame(cells=c(1000L,10000L,200000L),genes=c(10000L,1000L,1000L),
  id=c("1k_x_10k","10k_x_1k","200k_x_1k"),label=c("1k x 10k","10k x 1k","200k x 1k"))
heat_data<-list()
for (measure in c("runtime_seconds","ram_gb")) {
  for (i in seq_len(nrow(scales))) {
    s<-scales[i, ];rows<-resource[resource$protocol=="top_scRNA" & resource$measure==measure & resource$cells==s$cells & resource$genes==s$genes, ]
    heat_source_detail[[length(heat_source_detail)+1L]] <- annotate_source_rows(
      rows,"structure",s$id,s$label,"Data size",measure)
    heat_data[[length(heat_data)+1L]]<-summarise_cost(rows,"structure",s$id,s$label,"Data size",measure)
  }
  for (ng in c(2000L,5000L)) {
    rows<-resource[resource$protocol==paste0("velocity_genes_",ng) & resource$measure==measure, ]
    stopifnot(setequal(unique(rows$structure),structure_order),all(rows$cells==1000L),all(rows$genes==20000L))
    heat_source_detail[[length(heat_source_detail)+1L]] <- annotate_source_rows(
      rows,"structure",paste0("genes_",ng),paste0(ng/1000,"k"),
      "Velocity genes",measure)
    heat_data[[length(heat_data)+1L]]<-summarise_cost(rows,"structure",paste0("genes_",ng),paste0(ng/1000,"k"),"Velocity genes",measure)
  }
}
heat_data<-do.call(rbind,heat_data)
heat_source_detail<-do.call(rbind,heat_source_detail)
structure_time<-do.call(rbind,lapply(structure_order,function(s) {
  rows<-resource[resource$protocol=="top_scRNA" & resource$measure=="runtime_seconds" & resource$structure==s, ]
  structure_source_detail[[length(structure_source_detail)+1L]] <<- annotate_source_rows(
    rows,c("cells","genes"),s,s,"Topology","runtime_seconds")
  summarise_cost(rows,c("cells","genes"),s,s,"Topology","runtime_seconds")
}))
structure_source_detail<-do.call(rbind,structure_source_detail)

scope_rows <- list()
for (measure in c("runtime_seconds","ram_gb")) {
  for (condition in c(scales$id,"genes_2000","genes_5000")) {
    p <- heat_source_detail[
      heat_source_detail$measure==measure &
      heat_source_detail$condition==condition, ]
    tasks <- unique(p[c("protocol","dataset_id","structure","cells","genes")])
    scope_rows[[length(scope_rows)+1L]] <- data.frame(
      measure=measure,condition=condition,
      n_unique_tasks=nrow(tasks),
      n_structures=length(unique(tasks$structure)),
      structures=paste(sort(unique(tasks$structure)),collapse=";"),
      cells=paste(sort(unique(tasks$cells)),collapse=";"),
      genes=paste(sort(unique(tasks$genes)),collapse=";"),
      scope_note=if (condition=="200k_x_1k")
        "single extra-large task; not a seven-topology mean"
      else if (grepl("^genes_",condition))
        "velocity-gene protocol; 1k cells x 20k genes"
      else "top-scRNA tasks balanced over available structures")
  }
}
resource_scope_audit <- do.call(rbind,scope_rows)
stopifnot(resource_scope_audit$n_unique_tasks[
  resource_scope_audit$condition=="200k_x_1k"]==1L)

known_structure_scope <- do.call(rbind,lapply(structure_order,function(s) {
  p <- unique(structure_source_detail[
    structure_source_detail$structure==s,
    c("dataset_id","cells","genes")])
  data.frame(scope="known_topology",structure=s,n_unique_tasks=nrow(p),
    scales=paste(sort(unique(paste(p$cells,p$genes,sep="x"))),collapse=";"),
    included_in_structure_figure=TRUE,
    note="mean within cell x gene scale, then equal mean over valid scales")
}))
extra_large_resource_audit <- resource_all[
  resource_all$protocol=="top_scRNA" &
  resource_all$cells==200000L & resource_all$genes==1000L, ]
structure_scope_audit <- rbind(
  known_structure_scope,
  data.frame(scope="extra_large",structure="unverified_topology",
    n_unique_tasks=length(unique(extra_large_resource_audit$dataset_id)),
    scales="200000x1000",included_in_structure_figure=FALSE,
    note="kept in range audit; not presented as an eighth known topology"))

vram_source_audit <- resource_all[
  resource_all$measure=="vram_gb" & resource_all$method %in% methods, ]
vram_source_audit$Sc_display_order <- match(vram_source_audit$method,sc_order)
vram_summary <- do.call(rbind,lapply(methods,function(m) {
  do.call(rbind,lapply(sort(unique(vram_source_audit$protocol)),function(protocol) {
    p <- vram_source_audit[
      vram_source_audit$method==m &
      vram_source_audit$protocol==protocol, ]
    data.frame(method=m,Sc_display_order=match(m,sc_order),protocol=protocol,
      n_numeric=sum(is.finite(p$clean_value)),n_expected=nrow(p),
      mean_vram_gb=valid_mean(p$clean_value),
      status=if (!any(is.finite(p$clean_value))) "NA_no_values"
      else if (all(is.finite(p$clean_value))) "complete"
      else "partial_available_mean")
  }))
}))

make_exclusion_audit <- function(data,source) {
  if (!nrow(data)) {
    return(data.frame(source=source,method=NA_character_,
      normalized_method=NA_character_,rows_excluded=0L))
  }
  counts <- as.data.frame(table(data$method),stringsAsFactors=FALSE)
  names(counts) <- c("method","rows_excluded")
  counts$source <- source
  counts$normalized_method <- normalize_method(counts$method)
  counts[c("source","method","normalized_method","rows_excluded")]
}
exclusion_audit <- rbind(
  make_exclusion_audit(resource_excluded,"resource"),
  make_exclusion_audit(items_excluded,"usability"))

packages<-c("ggplot2","gridExtra","ComplexHeatmap","circlize","svglite","systemfonts")
missing<-packages[!vapply(packages,requireNamespace,logical(1),quietly=TRUE)]
if (length(missing)) stop("Missing R packages: ",paste(missing,collapse=", "))
# Use the device-generic sans alias so Cairo, svglite and ComplexHeatmap share
# one font backend. In this clean environment it resolves to DejaVu Sans;
# explicitly naming Liberation Sans triggers PostScript-metric warnings in
# ComplexHeatmap even though the TTF exists.
font_family <- "sans"
font_match <- if ("match_fonts" %in% getNamespaceExports("systemfonts"))
  systemfonts::match_fonts(font_family) else systemfonts::match_font(font_family)
font_resolved <- systemfonts::font_info(path=font_match$path[[1]])$family[[1]]
for (d in c("tables","figures","logs","validation")) dir.create(file.path(OUT,d),recursive=TRUE,showWarnings=FALSE)
layout_file<-tempfile(fileext=".pdf")
grDevices::cairo_pdf(layout_file,width=2,height=2,family=font_family)
layout_id<-grDevices::dev.cur()
writeLines(c(paste0("device_family=",font_family),
  paste0("resolved_family=",font_resolved)),
  file.path(OUT,"logs/font_family.txt"))
publish_bytes <- function(data,path) {
  writeBin(data,path)
  observed <- readBin(path,"raw",n=file.info(path)$size)
  stopifnot(length(data)>0L,identical(data,observed))
}
write_table<-function(data,filename) {
  tmp <- tempfile(fileext=".csv")
  write.csv(data,tmp,row.names=FALSE,na="NA",fileEncoding="UTF-8")
  publish_bytes(readBin(tmp,"raw",n=file.info(tmp)$size),file.path(OUT,filename))
  unlink(tmp)
}
# Compute component/category ranks before CSV formatting can round scores.
sc$T_rank <- rank(-sc$T,ties.method="average",na.last="keep")
sc$R_rank <- rank(-sc$R,ties.method="average",na.last="keep")
categories$category_rank <- ave(categories$score,categories$category,
  FUN=function(v) rank(-v,ties.method="average",na.last="keep"))
write_table(sc,"tables/scalability_independent_rank.csv")
write_table(sc[c("method","score","rank","display_order","fixed_method_order",
  "T","R","T_n_valid","T_n_expected","R_n_valid","R_n_expected",
  "Sc_n_components","effective_weight_T","effective_weight_R",
  "score_status","missing_reason")],"tables/scalability_order.csv")
write_table(u,"tables/usability_independent_rank.csv")
write_table(u[c("method","score","rank","display_order","fixed_method_order",
  "n_valid_categories","n_expected_categories","U_complete",
  "score_status","missing_categories","missing_reason")],
  "tables/usability_order.csv")
write_table(task_z,"tables/scalability_task_z.csv")
write_table(categories,"tables/usability_categories_recomputed.csv")
write_table(items,"tables/usability_item_contributions_recomputed.csv")
write_table(rules,"tables/usability_rules_S7_verified.csv")
write_table(heat_data,"tables/scalability_heatmaps.csv")
write_table(heat_source_detail,"tables/scalability_heatmap_source_detail.csv")
write_table(resource_scope_audit,"tables/resource_scope_audit.csv")
write_table(structure_time,"tables/structure_mean_time.csv")
write_table(structure_source_detail,"tables/structure_mean_time_source_detail.csv")
write_table(structure_scope_audit,"tables/structure_scope_audit.csv")
write_table(extra_large_resource_audit,"tables/extra_large_resource_audit.csv")
write_table(vram_source_audit,"tables/vram_source_audit.csv")
write_table(vram_summary,"tables/vram_summary.csv")
write_table(exclusion_audit,"tables/method_exclusion_audit.csv")
locked_manifest_path <- file.path(PROJECT,"config","input_sha256.csv")
if (!file.exists(locked_manifest_path)) stop("Missing config/input_sha256.csv")
write_table(read_csv(locked_manifest_path),"tables/input_manifest.csv")

BODY_BOTTOM <- .95; BODY_TOP <- PANEL_HEIGHT_IN-1.25; BODY_HEIGHT <- BODY_TOP-BODY_BOTTOM
y_fraction <- (N+.6-seq_len(N))/(N+.2)
row_height_in <- BODY_HEIGHT/(N+.2)
row_y <- BODY_BOTTOM+BODY_HEIGHT*y_fraction
SC_WIDTH <- 10.5; U_WIDTH <- 7.5
resource_condition_order <- c(scales$id,"genes_2000","genes_5000")
resource_labels <- c(scales$label,"2k","5k")
cost_matrix <- function(measure,conditions=resource_condition_order) {
  mat<-matrix(NA_real_,nrow=N,ncol=length(conditions),dimnames=list(sc_order,conditions))
  for (j in seq_along(conditions)) {
    p<-heat_data[heat_data$measure==measure & heat_data$condition==conditions[[j]], ]
    stopifnot(nrow(p)==N,!anyDuplicated(p$method));mat[,j]<-p$value[match(sc_order,p$method)]
  }
  mat
}
time_mat<-cost_matrix("runtime_seconds");memory_mat<-cost_matrix("ram_gb")
color_info<-function(mat) {
  values<-as.vector(mat);values<-values[is.finite(values)]
  if (!length(values)) return(list(
    fun=circlize::colorRamp2(c(0,1),c("#F7FCF0","#00441B")),
    breaks=c(0,1),midpoint=.5,data_min=NA_real_,
    display_cap=NA_real_,data_max=NA_real_))
  minimum<-min(values);maximum_all<-max(values);positives<-sort(unique(values[values>0]),decreasing=TRUE)
  maximum<-if (length(positives)>=2L) positives[[2]] else maximum_all
  if (!is.finite(maximum)||maximum<=minimum) maximum<-maximum_all
  used<-values[values>0 & values<=maximum]
  quantiles<-if (length(used)) quantile(used,probs=c(.1,.3,.5,.7,.9),na.rm=TRUE,names=FALSE) else numeric(0)
  breaks<-sort(unique(c(minimum,quantiles,maximum)))
  if (length(breaks)<2L) breaks<-c(minimum,minimum+max(abs(minimum)*.01,1e-6))
  full<-grDevices::hcl.colors(256,palette="BluGrn",rev=TRUE)[41:200]
  cols<-full[round(seq(1,length(full),length.out=length(breaks)))]
  list(fun=circlize::colorRamp2(breaks,cols),breaks=breaks,
    midpoint=(minimum+maximum)/2,data_min=minimum,
    display_cap=maximum,data_max=maximum_all)
}
time_colors<-color_info(time_mat);memory_colors<-color_info(memory_mat)
write_table(data.frame(
  measure=c("runtime_seconds","ram_gb"),
  unit=c("min","GB"),
  data_min=c(time_colors$data_min,memory_colors$data_min),
  second_largest_unique_positive_display_cap=c(
    time_colors$display_cap,memory_colors$display_cap),
  data_max=c(time_colors$data_max,memory_colors$data_max),
  palette="BluGrn",
  saturation_changes_data=FALSE),
  "tables/scalability_color_scale_config.csv")
sc_bar_colors<-circlize::colorRamp2(c(1,N),c("#e5f5e0","#006d2c"))
u_bar_colors<-circlize::colorRamp2(c(0,10),c("#fff7bc","#d95f0e"))
make_bar_geometry <- function(table,dimension,x=1.55,width=.82) {
  values <- if (dimension=="Sc") table$Sc_reversed_rank else table$U
  domain <- if (dimension=="Sc") N else 10
  fills <- if (dimension=="Sc") sc_bar_colors(values) else u_bar_colors(values)
  data.frame(
    dimension=dimension,method=table$method,
    display_order=table$display_order,
    raw_score=if (dimension=="Sc") table$Sc else table$U,
    rank=if (dimension=="Sc") table$Sc_rank else table$U_rank,
    plotted_value=values,domain=domain,
    x_left_in=x,y_center_in=row_y,
    background_width_in=width,
    plotted_width_in=ifelse(is.finite(values),width*values/domain,NA_real_),
    bar_height_in=row_height_in*.86,
    fill_hex=fills,
    label=ifelse(is.finite(values),
      if (dimension=="Sc") sprintf("%.1f",values) else sprintf("%.2f",values),
      "NA"))
}
sc_bar_geometry <- make_bar_geometry(sc,"Sc")
u_bar_geometry <- make_bar_geometry(u,"U")
make_heat_geometry <- function(mat,info,measure,x,width) {
  rows <- list(); nc <- ncol(mat); cw <- width/nc
  for (j in seq_len(nc)) for (i in seq_len(N)) {
    v <- mat[i,j]
    rows[[length(rows)+1L]] <- data.frame(
      measure=measure,method=rownames(mat)[i],
      display_order=i,condition=colnames(mat)[j],
      value=v,label=if (is.finite(v)) sprintf("%.2f",v) else "NA",
      x_center_in=x+(j-.5)*cw,y_center_in=row_y[[i]],
      width_in=cw,height_in=row_height_in,
      fill_hex=if (is.finite(v)) info$fun(v) else "#E5E5E5",
      text_hex=if (is.finite(v)&&v>info$midpoint) "#FFFFFF" else "#000000")
  }
  do.call(rbind,rows)
}
heat_render_geometry <- rbind(
  make_heat_geometry(time_mat,time_colors,"runtime_seconds",2.7,3.7),
  make_heat_geometry(memory_mat,memory_colors,"ram_gb",6.6,3.7))
write_table(rbind(sc_bar_geometry,u_bar_geometry),
  "tables/main_bar_render_geometry.csv")
write_table(heat_render_geometry,
  "tables/scalability_heatmap_render_geometry.csv")
text_in<-function(label,x,y,size=10,bold=FALSE,just="centre",rot=0,col="black") {
  grid::grid.text(label,x=grid::unit(x,"in"),y=grid::unit(y,"in"),just=just,rot=rot,
    gp=grid::gpar(fontfamily=font_family,fontsize=size,fontface=if (bold) "bold" else "plain",col=col))
}
rect_in<-function(x,y,w,h,fill,col="white",lwd=1) {
  grid::grid.rect(x=grid::unit(x,"in"),y=grid::unit(y,"in"),width=grid::unit(w,"in"),height=grid::unit(h,"in"),
    gp=grid::gpar(fill=fill,col=col,lwd=lwd))
}
draw_method_labels<-function(order) for (i in seq_len(N)) text_in(order[[i]],.10,row_y[[i]],size=11,just="left")
draw_score_bars<-function(table,dimension,x=1.55,width=.82) {
  values<-if (dimension=="Sc") table$Sc_reversed_rank else table$U
  domain<-if (dimension=="Sc") N else 10
  fills<-if (dimension=="Sc") sc_bar_colors(values) else u_bar_colors(values)
  for (i in seq_len(N)) {
    rect_in(x+width/2,row_y[[i]],width,row_height_in*.86,fill="grey94",col=NA)
    if (is.finite(values[[i]])) {
      w<-width*values[[i]]/domain
      if (w>0) rect_in(x+w/2,row_y[[i]],w,row_height_in*.86,fill=fills[[i]],col="grey30",lwd=.5)
      if (dimension=="U") text_in(sprintf("%.2f",values[[i]]),x+w+.05,row_y[[i]],size=8,just="left")
      if (values[[i]]==0) text_in("0",x,row_y[[i]],size=8,just="left")
    } else text_in("NA",x+width/2,row_y[[i]],size=8)
  }
  text_in(if (dimension=="Sc") "Sc reversed rank" else "Weighted U total",
    x+width/2,BODY_TOP+.13,size=10,bold=TRUE)
  ticks<-if (dimension=="Sc") c(1,14,28) else c(0,5,10)
  for (t in ticks) text_in(as.character(t),x+width*t/domain,BODY_BOTTOM-.09,size=8)
  text_in(if (dimension=="Sc") "29 - Sc rank" else "Score (0-10)",
    x+width/2,BODY_BOTTOM-.29,size=8)
}
draw_cost_block<-function(mat,info,x,width,title) {
  nc<-ncol(mat);cw<-width/nc
  for (j in seq_len(nc)) for (i in seq_len(N)) {
    v<-mat[i,j]
    rect_in(x+(j-.5)*cw,row_y[[i]],cw,row_height_in,fill=if (is.finite(v)) info$fun(v) else "grey90",col="white",lwd=2)
    text_in(if (is.finite(v)) sprintf("%.2f",v) else "NA",x+(j-.5)*cw,row_y[[i]],size=10,
      col=if (is.finite(v)&&v>info$midpoint) "white" else "black")
  }
  text_in(title,x+width/2,PANEL_HEIGHT_IN-.56,size=12,bold=TRUE)
  text_in("Data size (cells x genes)",x+1.5*cw,PANEL_HEIGHT_IN-.79,size=9)
  text_in("Velocity genes",x+4*cw,PANEL_HEIGHT_IN-.79,size=9)
  for (j in seq_len(nc)) text_in(resource_labels[[j]],x+(j-.5)*cw,BODY_TOP+.02,size=8,just="left",rot=45)
  legend_y<-.34;legend_width<-min(2,width*.72);legend_x<-x+(width-legend_width)/2
  for (k in 1:100) {
    v<-min(info$breaks)+(max(info$breaks)-min(info$breaks))*(k-.5)/100
    rect_in(legend_x+(k-.5)*legend_width/100,legend_y,legend_width/100,.10,info$fun(v),col=NA)
  }
  for (v in c(min(info$breaks),(min(info$breaks)+max(info$breaks))/2,max(info$breaks))) {
    frac<-(v-min(info$breaks))/(max(info$breaks)-min(info$breaks))
    text_in(sprintf("%.2f",v),legend_x+frac*legend_width,legend_y-.15,size=8)
  }
}
draw_sc_panel<-function() {
  text_in("Scalability",SC_WIDTH/2,PANEL_HEIGHT_IN-.24,size=16,bold=TRUE)
  draw_method_labels(sc_order);draw_score_bars(sc,"Sc")
  draw_cost_block(time_mat,time_colors,x=2.7,width=3.7,title="Time (min)")
  draw_cost_block(memory_mat,memory_colors,x=6.6,width=3.7,title="CPU RAM (GB)")
}

make_bubble_plot<-function(data,legend_example=FALSE) {
  if (!legend_example) {
    data$method<-factor(data$method,levels=rev(u_order))
    data$category<-factor(data$category,levels=category_order)
    plot<-ggplot2::ggplot(data,ggplot2::aes(x=category,y=method))+
      ggplot2::scale_y_discrete(limits=rev(u_order),drop=FALSE)
  } else plot<-ggplot2::ggplot(data,ggplot2::aes(x=factor(score),y=1))
  plot<-plot+
    ggplot2::geom_point(ggplot2::aes(size=Size,fill=score),shape=21,color="grey30")+
    ggplot2::scale_size_continuous(range=c(2,16),limits=range(categories$Size,na.rm=TRUE),
      breaks=(4:9-min_score+1)^2,labels=4:9,name="Score")+
    ggplot2::scale_fill_gradient(low="#fff7bc",high="#d95f0e",limits=range(categories$score,na.rm=TRUE),name="Score")+
    ggplot2::labs(title="Usability",x=NULL,y=NULL)+ggplot2::theme_minimal(base_family=font_family)+
    ggplot2::theme(panel.grid=ggplot2::element_blank(),axis.text=ggplot2::element_blank(),axis.title=ggplot2::element_blank(),
      legend.title=ggplot2::element_text(size=12,face="bold"),legend.text=ggplot2::element_text(size=10),
      plot.margin=ggplot2::margin(10,20,10,10),legend.position="none")
  plot
}
bubble_plot<-make_bubble_plot(categories)
bubble_layer<-ggplot2::ggplot_build(bubble_plot)$data[[1]]
stopifnot(nrow(bubble_layer)==N*5L,all(bubble_layer$shape==21),min(bubble_layer$size)==2,max(bubble_layer$size)==16)
bubble_gt<-ggplot2::ggplotGrob(bubble_plot)
bubble_panel<-bubble_gt$grobs[[which(bubble_gt$layout$name=="panel")]]
legend_scores<-4:9
legend_data<-data.frame(score=legend_scores,Size=(legend_scores-min_score+1)^2)
legend_gt<-ggplot2::ggplotGrob(make_bubble_plot(legend_data,TRUE))
legend_panel<-legend_gt$grobs[[which(legend_gt$layout$name=="panel")]]
bubble_x<-2.75;bubble_width<-U_WIDTH-bubble_x-.15
x_category<-(seq_len(5)-.4)/(5+.2)
draw_u_panel<-function() {
  text_in("Usability",U_WIDTH/2,PANEL_HEIGHT_IN-.24,size=16,bold=TRUE)
  draw_method_labels(u_order);draw_score_bars(u,"U")
  grid::pushViewport(grid::viewport(x=grid::unit(bubble_x,"in"),y=grid::unit(BODY_BOTTOM,"in"),
    width=grid::unit(bubble_width,"in"),height=grid::unit(BODY_HEIGHT,"in"),just=c("left","bottom"),clip="off"))
  grid::grid.draw(bubble_panel);grid::popViewport()
  for (j in seq_len(5)) text_in(category_order[[j]],bubble_x+bubble_width*x_category[[j]],PANEL_HEIGHT_IN-.50,size=9,just="right",rot=30)
  grid::pushViewport(grid::viewport(x=grid::unit(bubble_x+.1,"in"),y=grid::unit(.20,"in"),
    width=grid::unit(bubble_width-.2,"in"),height=grid::unit(.45,"in"),just=c("left","bottom"),clip="off"))
  grid::grid.draw(legend_panel);grid::popViewport()
  for (j in seq_along(legend_scores)) {
    x<-bubble_x+.1+(bubble_width-.2)*(j-.4)/(length(legend_scores)+.2)
    text_in(as.character(legend_scores[[j]]),x,.13,size=10)
  }
  text_in("Score",bubble_x-.03,.43,size=12,bold=TRUE,just="right")
}

figure_registry<-list()
save_plot<-function(name,draw_fun,width,height,dimension,plot_csv) {
  stem<-file.path(OUT,"figures",name)
  with_device<-function(open) {open();id<-grDevices::dev.cur();on.exit(if (id %in% grDevices::dev.list()) grDevices::dev.off(id),add=TRUE);grid::grid.newpage();draw_fun()}
  for(ext in c("pdf","svg","png")) {
    tmp <- tempfile(fileext=paste0(".",ext))
    if(ext=="pdf") with_device(function() grDevices::cairo_pdf(tmp,width=width,height=height,family=font_family))
    if(ext=="svg") with_device(function() svglite::svglite(tmp,width=width,height=height,system_fonts=list(sans=font_resolved)))
    if(ext=="png") with_device(function() grDevices::png(tmp,width=width,height=height,units="in",res=300,type="cairo",family=font_family))
    bytes <- if(ext=="png") png::writePNG(png::readPNG(tmp,native=TRUE),target=raw(),dpi=300) else readBin(tmp,"raw",n=file.info(tmp)$size)
    if(ext=="pdf") stopifnot(identical(tail(bytes,6L),charToRaw("%%EOF\n")))
    if(ext=="svg") stopifnot(grepl("</svg>",rawToChar(tail(bytes,100))))
    publish_bytes(bytes,paste0(stem,".",ext));unlink(tmp)
  }
  order<-if (dimension=="Sc") sc_order else if (dimension=="U") u_order else character(0)
  figure_registry[[length(figure_registry)+1L]]<<-data.frame(figure=name,dimension=dimension,plotting_csv=plot_csv,
    pdf=paste0("figures/",name,".pdf"),
    svg=paste0("figures/",name,".svg"),
    png=paste0("figures/",name,".png"),
    width_inches=width,height_inches=height,n_methods=N,
    method_sequence=paste(order,collapse=";"),
    sc_method_sequence=if (dimension %in% c("Sc","independent_Sc_and_U"))
      paste(sc_order,collapse=";") else "",
    u_method_sequence=if (dimension %in% c("U","independent_Sc_and_U"))
      paste(u_order,collapse=";") else "")
}
write_table(heat_data,"figures/FigureA_Scalability_plotting.csv")
write_table(sc,"figures/FigureA_Scalability_bar_plotting.csv")
write_table(categories,"figures/FigureB_Usability_plotting.csv")
write_table(u,"figures/FigureB_Usability_bar_plotting.csv")
combined_plotting <- rbind(
  data.frame(panel="A",element="Sc_bar",method=sc$method,
    display_order=sc$display_order,variable="Sc_reversed_rank",
    value=sc$Sc_reversed_rank,rank=sc$Sc_rank,unit="29 - Sc rank",
    status=sc$score_status),
  data.frame(panel="A",element="resource_heatmap",method=heat_data$method,
    display_order=heat_data$display_order,
    variable=paste(heat_data$measure,heat_data$condition,sep="::"),
    value=heat_data$value,rank=NA_real_,unit=heat_data$unit,
    status=heat_data$status),
  data.frame(panel="B",element="U_bar",method=u$method,
    display_order=u$display_order,variable="weighted_U",
    value=u$U,rank=u$U_rank,unit="score_0_10",status=u$score_status),
  data.frame(panel="B",element="category_bubble",method=categories$method,
    display_order=categories$display_order,variable=categories$category,
    value=categories$score,rank=NA_real_,unit="score_0_10",
    status=ifelse(categories$category_complete,"complete","incomplete")))
write_table(combined_plotting,
  "figures/FigureAB_Scalability_Usability_independent_plotting.csv")
save_plot("FigureA_Scalability_independent_rank",draw_sc_panel,SC_WIDTH,PANEL_HEIGHT_IN,"Sc","figures/FigureA_Scalability_plotting.csv")
save_plot("FigureB_Usability_weighted_rank",draw_u_panel,U_WIDTH,PANEL_HEIGHT_IN,"U","figures/FigureB_Usability_plotting.csv")

combo_gap<-.35;combo_width<-SC_WIDTH+U_WIDTH+combo_gap
save_plot("FigureAB_Scalability_Usability_independent",function() {
  grid::pushViewport(grid::viewport(x=grid::unit(0,"in"),y=grid::unit(0,"in"),width=grid::unit(SC_WIDTH,"in"),height=grid::unit(PANEL_HEIGHT_IN,"in"),just=c("left","bottom")))
  draw_sc_panel();grid::popViewport()
  grid::pushViewport(grid::viewport(x=grid::unit(SC_WIDTH+combo_gap,"in"),y=grid::unit(0,"in"),width=grid::unit(U_WIDTH,"in"),height=grid::unit(PANEL_HEIGHT_IN,"in"),just=c("left","bottom")))
  draw_u_panel();grid::popViewport()
  text_in("A",.06,PANEL_HEIGHT_IN-.30,size=16,bold=TRUE,just="left")
  text_in("B",SC_WIDTH+combo_gap+.06,PANEL_HEIGHT_IN-.30,
    size=16,bold=TRUE,just="left")
},combo_width,PANEL_HEIGHT_IN,"independent_Sc_and_U",
"figures/FigureAB_Scalability_Usability_independent_plotting.csv")

make_standard_heatmap<-function(mat,title,show_rows=TRUE) {
  info<-color_info(mat)
  ComplexHeatmap::Heatmap(mat,name=title,col=info$fun,na_col="grey90",cluster_rows=FALSE,cluster_columns=FALSE,
    column_names_side="top",column_names_rot=45,column_names_centered=TRUE,show_row_names=show_rows,
    row_names_side="left",row_names_gp=grid::gpar(fontsize=8,fontfamily=font_family),column_names_gp=grid::gpar(fontsize=8,fontfamily=font_family),
    rect_gp=grid::gpar(col="white",lwd=4),use_raster=FALSE,
    cell_fun=function(j,i,x,y,width,height,fill) {
      v<-mat[i,j];grid::grid.text(if (is.finite(v)) sprintf("%.2f",v) else "NA",x,y,
        gp=grid::gpar(fontsize=14,fontfamily=font_family,col=if (!is.finite(v)) "black" else if (v>info$midpoint) "white" else "black"))
    },heatmap_legend_param=list(at=info$breaks,labels=sprintf("%.2f",info$breaks),title=title,
      legend_direction="vertical",title_position="leftcenter-rot"))
}
for (measure in c("runtime_seconds","ram_gb")) {
  label<-if (measure=="runtime_seconds") "Time (min)" else "RAM (GB)"
  name<-if (measure=="runtime_seconds") "FigureS_Sc_three_scales_time" else "FigureS_Sc_three_scales_memory"
  mat<-cost_matrix(measure,scales$id);colnames(mat)<-scales$label
  plotting<-heat_data[heat_data$measure==measure & heat_data$condition %in% scales$id, ]
  write_table(plotting,paste0("figures/",name,"_plotting.csv"))
  local({M<-mat;TITLE<-label;saved_name<-name
    save_plot(saved_name,function() ComplexHeatmap::draw(make_standard_heatmap(M,TITLE),newpage=FALSE),9.5,14,"Sc",paste0("figures/",saved_name,"_plotting.csv"))})
}
genes_time<-cost_matrix("runtime_seconds",c("genes_2000","genes_5000"));genes_ram<-cost_matrix("ram_gb",c("genes_2000","genes_5000"))
colnames(genes_time)<-colnames(genes_ram)<-c("2k genes","5k genes")
name<-"FigureS_Sc_2k_5k_time_memory"
write_table(heat_data[heat_data$condition %in% c("genes_2000","genes_5000"), ],paste0("figures/",name,"_plotting.csv"))
save_plot(name,function() ComplexHeatmap::draw(make_standard_heatmap(genes_time,"Time (min)")+make_standard_heatmap(genes_ram,"RAM (GB)",FALSE),
  merge_legends=FALSE,ht_gap=grid::unit(4,"mm"),newpage=FALSE),9.5,14,"Sc",paste0("figures/",name,"_plotting.csv"))

labels<-c("linear-simple"="Linear","bifurcating"="Bifurcating","consecutive-bifurcating"="Consecutive\nbifurcating",
  "bifurcating-loop"="Bifurcating\nloop","trifurcating"="Trifurcating","cycle-simple"="Cycle","disconnected"="Disconnected")
plots<-lapply(structure_order,function(s) {
  d<-structure_time[structure_time$condition==s, ];d$method<-factor(d$method,levels=rev(sc_order))
  p<-ggplot2::ggplot(d,ggplot2::aes(x=method,y=value,fill=method))+
    ggplot2::geom_bar(data=d[is.finite(d$value), ],stat="identity",color="black",linewidth=.5)+
    ggplot2::scale_fill_manual(values=method_colors,drop=FALSE)+ggplot2::scale_x_discrete(limits=rev(sc_order),drop=FALSE)+
    ggplot2::scale_y_continuous(n.breaks=3)+ggplot2::coord_flip()+ggplot2::labs(title=labels[[s]],x=NULL,y="Time (min)")+
    ggplot2::theme_minimal(base_family=font_family)+ggplot2::theme(panel.grid=ggplot2::element_blank(),legend.position="none",
      axis.text.y=if (s==structure_order[[1]]) ggplot2::element_text(size=14,face="bold") else ggplot2::element_blank(),
      axis.text.x=ggplot2::element_text(size=9),plot.title=ggplot2::element_text(size=14,hjust=.5))
  if (anyNA(d$value)) p<-p+ggplot2::geom_text(data=d[is.na(d$value), ],ggplot2::aes(x=method,y=0,label="NA"),inherit.aes=FALSE,size=2.3)
  p
})
grobs<-lapply(plots,ggplot2::ggplotGrob);heights<-Reduce(grid::unit.pmax,lapply(grobs,`[[`,"heights"));grobs<-lapply(grobs,function(g){g$heights<-heights;g})
structure_grob<-gridExtra::arrangeGrob(grobs=grobs,nrow=1,widths=c(1.6,rep(1,6)))
name<-"FigureS_Sc_mean_time_by_structure";write_table(structure_time,paste0("figures/",name,"_plotting.csv"))
save_plot(name,function() grid::grid.draw(structure_grob),14,10,"Sc",paste0("figures/",name,"_plotting.csv"))

write_table(do.call(rbind,figure_registry),"figures/figure_registry.csv")
write_table(data.frame(plot_row=seq_len(nrow(bubble_layer)),method=categories$method,category=categories$category,score=categories$score,Size=categories$Size,category_x=as.numeric(bubble_layer$x),method_y=as.numeric(bubble_layer$y),size_mm=bubble_layer$size,shape=bubble_layer$shape,
  fill_hex=bubble_layer$fill,stroke=bubble_layer$stroke),"tables/usability_actual_ggplot_layer.csv")
capture.output(sessionInfo(),file=file.path(OUT,"logs/R_sessionInfo.txt"))
writeLines(c(
  "Sc and U have independent descending orders; no Overall order is used.",
  "Sc scores recomputed from 68 top-scRNA runtime/RAM tasks; supplementary 2k/5k blocks are not double-counted.",
  "Usability recomputed from 24 item subscores and the five category weights .25/.25/.20/.15/.15.",
  "Sc bar shows 29-rank; U bar shows the actual 0-10 weighted total.",
  "True zero is retained; missing data are not zero; available structures are equally weighted with n/N exported.",
  "200k x 1k has one extra-large task, not seven confirmed topologies.",
  "2k/5k are velocity genes in the input; original datasets are 1k cells x 20k genes.",
  "Fig3 bubble parameters are preserved; its ggplot panel is reused inside the new layout.",
  "Both main panels are 16 inches high to retain 2-16 mm bubbles without overlap for 28 methods.",
  "Structure mean-time panels use 3 numeric ticks to prevent overlapping long time labels in narrow panels.",
  "New composite resource cell text is 10 pt and grid width 2 to fit ten columns plus a ranking bar; standalone original heatmaps retain 14 pt/grid width 4.",
  "Bubble legend uses six scores 4-9 at the bottom, with the same global Size/fill limits as the actual dots.",
  "PDF and SVG are vector exports, use_raster=FALSE; PNG is 300 dpi.",
  "Sc reflects recorded resource performance; hardware conditions not independently verified."
),file.path(OUT,"logs/plot_changes_and_assumptions.txt"))
warning_lines<-capture.output(warnings())
if (!length(warning_lines)) warning_lines<-"No R warnings."
writeLines(warning_lines,file.path(OUT,"logs/R_warnings.txt"))
if (layout_id %in% grDevices::dev.list()) grDevices::dev.off(layout_id)
unlink(layout_file)
message("Completed independent Sc and weighted U ranking with actual R figures: ",OUT)
