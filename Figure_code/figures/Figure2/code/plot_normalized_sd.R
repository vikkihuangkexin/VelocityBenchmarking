#!/usr/bin/env Rscript
# Keep all 21 earlier figures; add per-type unitless STD strips and full panels.
args<-commandArgs(trailingOnly=TRUE)
root<-normalizePath(if(length(args)) args[[1]] else ".",mustWork=TRUE)
source(file.path(root,"code/plot_accuracy_panels.R"),local=.GlobalEnv)
nsd<-read.csv(file.path(PROJECT,"tables/batch_SD_normalized_method_summary.csv"),check.names=FALSE)
VAR_W<-1.16
var_data<-function(view,methods=order) {
 d<-nsd[match(methods,nsd$method), ];d$value<-d[[switch(view,Combined="combined_score",Real="real_score",Simulation="simulation_score")]]
 d$label<-ifelse(is.finite(d$value),ifelse(d$value==0,"0",sprintf("%.3g",d$value)),"")
 d$metric<-"Standardized batch variability";d$view<-view;d$plot_row<-seq_along(methods);d$plot_column<-1L
 d$hex<-unname(colors[d$method]);d$plot_role<-"normalized_SD";d
}
var_draw<-function(d,x0,view,h=H,methods=order) {
 heading("Variability",x0,VAR_W,view,h,fill="#BCE3C8")
 txt("Standardized\nSD score",x0+VAR_W/2,h-1.12,11,TRUE)
 yy<-row_positions(length(methods),h)
 for(i in seq_len(nrow(d))) if(nzchar(d$label[[i]])) txt(d$label[[i]],x0+VAR_W/2,yy[[d$plot_row[[i]]]],11)
}
fill_bind<-function(frames) {
 cols<-unique(unlist(lapply(frames,names)))
 do.call(rbind,lapply(frames,function(d){for(c in setdiff(cols,names(d))) d[[c]]<-NA;d[,cols]}))
}
export_new<-function(name,draw,width,height=H,view="Combined",nrows=N,show_methods=TRUE,plotting) {
 stem<-file.path(OUT,name)
 with_device<-function(open) {open();id<-dev.cur();on.exit(if(id %in% dev.list()) dev.off(id));grid.newpage();draw()}
 for(ext in c("pdf","svg")) {
  tmp<-tempfile(fileext=paste0(".",ext))
  if(ext=="pdf") with_device(function() cairo_pdf(tmp,width=width,height=height,family=font)) else
   with_device(function() svglite::svglite(tmp,width=width,height=height,system_fonts=list(sans=font)))
  b<-readBin(tmp,"raw",n=file.info(tmp)$size)
  if(ext=="pdf") stopifnot(identical(tail(b,6L),charToRaw("%%EOF\n")))
  if(ext=="svg") stopifnot(grepl("</svg>",rawToChar(tail(b,100))))
  writeBin(b,paste0(stem,".",ext));unlink(tmp)
 }
 tmp<-tempfile(fileext=".png")
 with_device(function() png(tmp,width=width,height=height,units="in",res=300,type="cairo-png",family=font))
 writeBin(png::writePNG(png::readPNG(tmp,native=TRUE),target=raw(),dpi=300),paste0(stem,".png"));unlink(tmp)
 plotting$figure<-name;plotting$y_inches<-row_positions(nrows,height)[plotting$plot_row]
 plotting$PDF_y_center_points<-(height-plotting$y_inches)*72
 write.csv(plotting,paste0(stem,"_plotting.csv"),row.names=FALSE,na="NA")
 registry[[length(registry)+1L]]<<-data.frame(figure=name,view=view,width_inches=width,height_inches=height,n_rows=nrows,show_methods=show_methods,plotting_csv=paste0(name,"_plotting.csv"))
}
for(view in view_order) for(with_names in c(TRUE,FALSE)) local({v<-view;names_on<-with_names;d<-var_data(v)
 x0<-if(names_on) LABEL_W else PAD;width<-x0+VAR_W+PAD
 d$PDF_x_center_points<-(x0+VAR_W/2)*72
 name<-paste0("Stability_normalizedSD_",v,if(names_on) "" else "_strip")
 export_new(name,function(){if(names_on) method_labels();var_draw(d,x0,v)
  txt(if(names_on) "Lower = less relative variability.\nNot raw SD; unmeasured rows are blank." else "Lower = less\nrelative variability.\nNot raw SD.",width/2,.48,8)},width,view=v,show_methods=names_on,plotting=d)
})
measured<-order[order %in% nsd$method[is.finite(nsd$combined_score)]]
compact_h<-2.55+(length(measured)+.2)*(H-2.55)/(N+.2);d<-var_data("Combined",measured)
d$PDF_x_center_points<-(LABEL_W+VAR_W/2)*72;width<-LABEL_W+VAR_W+PAD
export_new("Stability_normalizedSD_measured_only",function(){method_labels(measured,compact_h);var_draw(d,LABEL_W,"Combined",compact_h,measured)
 txt("Lower = less relative variability.\nNot raw SD; real/simulation 50:50.",width/2,.48,8)},width,height=compact_h,nrows=length(measured),plotting=d)

# Main figures: Accuracy-only order; a single unitless SD score after the bars.
for(view in view_order) local({v<-view
 pm<-if(v=="Real") PRIMARY[PRIMARY!="Ground-truth correlation"] else PRIMARY
 sm<-if(v=="Real") CONSISTENCY else c(CONSISTENCY,DOWN)
 xp<-LABEL_W+RANK_W+GAP;xs<-xp+block_span(length(pm))+GAP;xv<-xs+block_span(length(sm))+GAP
 width<-xv+VAR_W+PAD;mm<-c(pm,sm)
 frames<-lapply(seq_along(mm),function(j){m<-mm[[j]];d<-metric_data(m,v);d$plot_row<-match(d$method,order)
   d$panel_left_inches<-if(j<=length(pm)) xp+(j-1)*STEP else xs+(j-length(pm)-1)*STEP
   d$panel_width_inches<-METRIC_W;d$scale_min<-panels[[paste(v,m,sep="|")]]$range[[1]];d$scale_max<-panels[[paste(v,m,sep="|")]]$range[[2]];d})
 vd<-var_data(v);vd$PDF_x_center_points<-(xv+VAR_W/2)*72
 plotting<-fill_bind(c(frames,list(vd)))
 name<-paste0("Accuracy_Primary_Secondary_Variability_",v)
 export_new(name,function(){method_labels();heading("Accuracy",LABEL_W,RANK_W,"Combined rank");draw_rank(LABEL_W)
   heading("Primary scientific metrics",xp,block_span(length(pm)),v,fill="#E5B5D6")
   heading("Secondary scientific metrics",xs,block_span(length(sm)),v,fill="#BCE3C8")
   for(j in seq_along(pm)) draw_metric(pm[[j]],v,xp+(j-1)*STEP)
   for(j in seq_along(sm)) draw_metric(sm[[j]],v,xs+(j-1)*STEP)
   var_draw(vd,xv,v)
   txt(if(v=="Combined") "Bars: raw means ± SE; shared metrics real/simulation 50:50. GT/downsampling: simulation only." else paste(v,"bars: raw means ± SE. All rows follow the same 50:50 Accuracy ranking."),width/2,.62,9)
   txt("SD score: within-task SD z, then equal metrics and datasets; lower = less relative variability. Missing scores are blank.",width/2,.38,9)
 },width,view=v,plotting=plotting)
})
write.csv(do.call(rbind,registry),file.path(PROJECT,"figure_index.csv"),row.names=FALSE)
message("Completed ",length(registry)," figures: raw means unchanged; normalized STD is descriptive only.")
