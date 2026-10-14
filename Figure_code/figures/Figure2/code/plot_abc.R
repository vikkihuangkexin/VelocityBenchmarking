#!/usr/bin/env Rscript
# Reuse every baseline plot and add the requested aligned A-C composition.
# Usage: Rscript --vanilla code/plot_abc.R /absolute/analysis_output
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(if(length(args)) args[[1]] else '.', mustWork=TRUE)
source(file.path(root, 'code/plot_normalized_sd.R'), local=.GlobalEnv)
annotations <- read.csv(file.path(PROJECT, 'config/method_annotations.csv'),
                        check.names=FALSE, na.strings=c('', 'NA'), stringsAsFactors=FALSE)
stopifnot(!anyDuplicated(annotations$method), setequal(annotations$method, order))
annotations <- annotations[match(order, annotations$method), ]
fields <- c('platform', 'deep_learning', 'input', 'additional_input')
stopifnot(all(c(fields, 'verification_status') %in% names(annotations)))
for(field in fields) annotations[[field]][is.na(annotations[[field]])] <- 'Unverified'
not_verified <- is.na(annotations$verification_status) | annotations$verification_status != 'verified'
# Unverified draft metadata cannot silently appear as an established fact.
for(field in fields) annotations[[field]][not_verified] <- 'Unverified'
annotations$display_order <- seq_len(N)
write.csv(annotations, file.path(PROJECT, 'tables/method_annotations_display.csv'), row.names=FALSE, na='NA')

META_WIDTH <- c(.88, .52, 1.25, 1.62)
METHODS_W <- LABEL_W + sum(META_WIDTH) + 3*COL_GAP
metadata_draw <- function() {
  heading('A  Methods', 0, METHODS_W, '')
  txt('RNA velocity\nmethod', LABEL_W/2, H-1.12, 12, TRUE)
  method_labels()
  x <- LABEL_W
  headings <- c('Platform', 'DL?', 'Input', 'Additional\ninput')
  yy <- row_positions()
  for(j in seq_along(fields)) {
    width <- META_WIDTH[[j]]
    txt(headings[[j]], x+width/2, H-1.12, 11, TRUE)
    for(i in seq_len(N)) {
      label <- paste(strwrap(annotations[[fields[[j]]]][[i]], width=max(7, floor(width*12))), collapse='\n')
      txt(label, x+width/2, yy[[i]], 9)
    }
    x <- x+width+COL_GAP
  }
}

for(view in view_order) local({
  v <- view
  pm <- if(v=='Real') PRIMARY[PRIMARY!='Ground-truth correlation'] else PRIMARY
  xp <- METHODS_W+GAP
  xm <- xp+RANK_W+GAP
  pw <- RANK_W+GAP+block_span(length(pm))
  xs <- xm+block_span(length(pm))+GAP
  xc <- xs+block_span(length(CONSISTENCY))+GAP
  xd <- xc+VAR_W+GAP
  end <- if(v=='Real') xc+VAR_W else xd+METRIC_W
  width <- end+PAD
  vd <- var_data(v)
  vd$PDF_x_center_points <- (xc+VAR_W/2)*72
  vd$panel_group <- 'C'
  frames <- list()
  for(j in seq_along(pm)) {
    m <- pm[[j]];d <- metric_data(m,v);d$plot_row <- match(d$method,order)
    d$panel_left_inches <- xm+(j-1)*STEP;d$panel_width_inches <- METRIC_W
    d$scale_min <- panels[[paste(v,m,sep='|')]]$range[[1]]
    d$scale_max <- panels[[paste(v,m,sep='|')]]$range[[2]]
    d$panel_group <- 'B';frames[[length(frames)+1L]] <- d
  }
  for(j in seq_along(CONSISTENCY)) {
    m <- CONSISTENCY[[j]];d <- metric_data(m,v);d$plot_row <- match(d$method,order)
    d$panel_left_inches <- xs+(j-1)*STEP;d$panel_width_inches <- METRIC_W
    d$scale_min <- panels[[paste(v,m,sep='|')]]$range[[1]]
    d$scale_max <- panels[[paste(v,m,sep='|')]]$range[[2]]
    d$panel_group <- 'C';frames[[length(frames)+1L]] <- d
  }
  frames[[length(frames)+1L]] <- vd
  if(v!='Real') {
    d <- metric_data(DOWN,v);d$plot_row <- match(d$method,order)
    d$panel_left_inches <- xd;d$panel_width_inches <- METRIC_W
    d$scale_min <- panels[[paste(v,DOWN,sep='|')]]$range[[1]]
    d$scale_max <- panels[[paste(v,DOWN,sep='|')]]$range[[2]]
    d$panel_group <- 'C';frames[[length(frames)+1L]] <- d
  }
  plotting <- fill_bind(frames)
  export_new(paste0('Figure_ABC_',v),function() {
    metadata_draw()
    heading('B  Primary scientific metrics',xp,pw,'Directional accuracy',fill='#E5B5D6')
    draw_rank(xp)
    for(j in seq_along(pm)) draw_metric(pm[[j]],v,xm+(j-1)*STEP)
    heading('C  Secondary scientific metrics',xs,end-xs,'',fill='#BCE3C8')
    txt('Consistency',xs+block_span(2)/2,H-.73,10)
    txt('Stability',xc+(end-xc)/2,H-.73,10)
    for(j in seq_along(CONSISTENCY)) draw_metric(CONSISTENCY[[j]],v,xs+(j-1)*STEP)
    txt('Batch-run\nstandardized SD',xc+VAR_W/2,H-1.12,11,TRUE)
    yy <- row_positions()
    for(i in seq_len(N)) if(nzchar(vd$label[[i]])) txt(vd$label[[i]],xc+VAR_W/2,yy[[i]],11)
    if(v!='Real') draw_metric(DOWN,v,xd)
    txt(paste(v,'bars: raw means +/- dataset SE. Shared Combined metrics: real/simulation 50:50; GT/DS: simulation only.'),width/2,.65,9)
    txt('All rows: Combined directional Accuracy rank. Standardized batch variability: lower = less relative variability; missing is blank.',width/2,.43,9)
    if(any(not_verified)) txt('Unverified: method metadata were not supplied or verified; edit config/method_annotations.csv.',width/2,.21,8)
  },width,view=v,plotting=plotting)
})
write.csv(do.call(rbind,registry),file.path(PROJECT,'figure_index.csv'),row.names=FALSE)
message('Completed ',length(registry),' figures including A-C; no Overall score.')
