#!/usr/bin/env Rscript
# Actual R rendering. Fixed palette A; every full block has identical row centres.
args<-commandArgs(trailingOnly=TRUE)
PROJECT<-normalizePath(if(length(args)) args[[1]] else ".",mustWork=TRUE)
source(file.path(PROJECT,"code/check_R_packages.R"),local=.GlobalEnv)
suppressPackageStartupMessages({library(ggplot2);library(grid);library(circlize)})
stopifnot(all(vapply(c("png","svglite","systemfonts"),requireNamespace,logical(1),quietly=TRUE)))
OUT<-file.path(PROJECT,"figures");dir.create(OUT,showWarnings=FALSE)
rank<-read.csv(file.path(PROJECT,"tables/accuracy_scores_ranks.csv"),check.names=FALSE)
summary<-read.csv(file.path(PROJECT,"tables/metric_summary_5050.csv"),check.names=FALSE)
sd<-read.csv(file.path(PROJECT,"tables/batch_sd_numeric_accuracy_order.csv"),check.names=FALSE)
palette<-read.csv(file.path(PROJECT,"config/method_colors.csv"))
order<-rank$method;N<-length(order)
stopifnot(N==28L,!anyDuplicated(order),setequal(palette$method,order),!anyDuplicated(palette$method),length(unique(palette$hex))==N)
# Palette row order is historical; colours are bound by method name.
colors<-setNames(palette$hex,palette$method)
fonts<-unique(systemfonts::system_fonts()$family)
choices<-c("Liberation Sans","Arial","Nimbus Sans","DejaVu Sans")
font<-if(any(choices %in% fonts)) choices[choices %in% fonts][[1]] else "sans"
layout_file<-tempfile(fileext=".pdf");cairo_pdf(layout_file,width=2,height=2,family=font);layout_id<-dev.cur()
H<-12;LABEL_W<-2.05;RANK_W<-1.0;METRIC_W<-1.42;GAP<-.22;COL_GAP<-.14;STEP<-METRIC_W+COL_GAP;PAD<-.12;SD_W<-.88
block_span<-function(n) n*METRIC_W+(n-1)*COL_GAP
PRIMARY<-c("Angle consistency","CBDir","Transition V1","Ground-truth correlation")
CONSISTENCY<-c("Consistency score","ICCoh")
DOWN<-"Performance under downsampling"
title_names<-c("Angle consistency"="Angle\nconsistency","CBDir"="CBDir","Transition V1"="Transition\nV1",
 "Ground-truth correlation"="Ground-truth\ncorrelation","Consistency score"="Consistency\nscore","ICCoh"="ICCoh",
 "Performance under downsampling"="Downsampling\nGT correlation")
registry<-list();audit<-list();error_audit<-list();panels<-list()
view_order<-c("Combined","Real","Simulation")
row_positions<-function(n=N,h=H) {
 bottom<-1.15;top<-h-1.4
 bottom+(top-bottom)*(n+.6-seq_len(n))/(n+.2)
}
txt<-function(label,x,y,size=12,bold=FALSE,just="centre",col="black",rot=0) {
 grid.text(label,x=unit(x,"in"),y=unit(y,"in"),just=just,rot=rot,
  gp=gpar(fontfamily=font,fontsize=size,fontface=if(bold) "bold" else "plain",col=col))
}
box<-function(x,y,w,h,fill,col=NA,lwd=.5) grid.rect(x=unit(x,"in"),y=unit(y,"in"),width=unit(w,"in"),height=unit(h,"in"),gp=gpar(fill=fill,col=col,lwd=lwd))
heading<-function(title,x,width,view,h=H,fill="grey94") {
 box(x+width/2,h-.42,width,.33,fill,"grey30",.6)
 txt(title,x+width/2,h-.42,13,TRUE)
 txt(view,x+width/2,h-.73,10)
}
method_labels<-function(methods=order,h=H) {
 yy<-row_positions(length(methods),h)
 for(i in seq_along(methods)) txt(methods[[i]],.08,yy[[i]],14,TRUE,"left")
}
metric_data<-function(metric,view) {
 d<-summary[summary$metric==metric, ];d<-d[match(order,d$method), ]
 prefix<-switch(view,Combined="combined",Real="real",Simulation="simulation")
 d$value<-d[[paste0(prefix,"_raw_mean")]];d$se<-d[[paste0(prefix,"_raw_se")]]
 d$view<-view;d$hex<-unname(colors[d$method]);d$display_order<-seq_len(N)
 d$metric_label<-unname(title_names[metric]);d$plot_role<-if(metric %in% PRIMARY) "primary" else "secondary"
 d
}
for(view in view_order) for(metric in c(PRIMARY,CONSISTENCY,DOWN)) {
 if(view=="Real" && metric %in% c("Ground-truth correlation",DOWN)) next
 d<-metric_data(metric,view);d$method_factor<-factor(d$method,levels=rev(order))
 p<-ggplot(d,aes(x=method_factor,y=value,fill=method_factor))+
  geom_bar(data=d[is.finite(d$value), ],stat="identity",color="black",linewidth=.5)+
  geom_errorbar(data=d[is.finite(d$value)&is.finite(d$se), ],aes(ymin=value-se,ymax=value+se),width=.2,color="black",linewidth=.4)+
  scale_fill_manual(values=colors,drop=FALSE)+scale_x_discrete(limits=rev(order),drop=FALSE)+coord_flip()+
  theme_minimal(base_family=font)+theme(panel.grid=element_blank(),axis.text=element_blank(),axis.title=element_blank(),legend.position="none")
 if(anyNA(d$value)) p<-p+geom_text(data=d[is.na(d$value), ],aes(x=method_factor,y=0,label="NA"),inherit.aes=FALSE,size=3,family=font,hjust=0)
 if(any(d$value==0,na.rm=TRUE)) p<-p+geom_text(data=d[is.finite(d$value)&d$value==0, ],aes(x=method_factor,y=0,label="0"),inherit.aes=FALSE,size=3,family=font,hjust=0)
 build<-ggplot_build(p);g<-ggplotGrob(p);body<-g$grobs[[which(g$layout$name=="panel")]]
 range<-build$layout$panel_params[[1]]$x.range
 key<-paste(view,metric,sep="|");panels[[key]]<-list(grob=body,range=range,data=d)
 el<-build$data[[2]]
 if(nrow(el)) error_audit[[length(error_audit)+1L]]<-data.frame(view=view,metric=metric,method=order[N+1-as.integer(el$x)],lower=el$ymin,upper=el$ymax,cap_width=el$xmax-el$xmin)
 layer<-build$data[[1]]
 if(nrow(layer)) {
  audit[[length(audit)+1L]]<-data.frame(view=view,metric=metric,method=order[N+1-as.integer(layer$x)],
   actual_mean=ifelse(layer$ymin<0,layer$ymin,layer$ymax),actual_factor_position=as.numeric(layer$x),actual_fill=layer$fill)
 }
}
draw_metric<-function(metric,view,x,h=H) {
 q<-panels[[paste(view,metric,sep="|")]]
 txt(title_names[[metric]],x+METRIC_W/2,h-1.12,12,TRUE)
 pushViewport(viewport(x=unit(x,"in"),y=unit(1.15,"in"),width=unit(METRIC_W,"in"),height=unit(h-2.55,"in"),just=c("left","bottom"),clip="on"))
 grid.draw(q$grob);popViewport()
 ticks<-pretty(q$range,n=3);ticks<-ticks[ticks>=q$range[[1]] & ticks<=q$range[[2]]]
 for(t in ticks) txt(format(t,trim=TRUE,digits=3),x+METRIC_W*(t-q$range[[1]])/diff(q$range),.99,10)
}
rank_cols<-circlize::colorRamp2(c(1,N),c("#fee0d2","#a50f15"))
draw_rank<-function(x,h=H) {
 yy<-row_positions(N,h);rowh<-(h-2.55)/(N+.2)
 txt("Accuracy\nrank",x+RANK_W/2,h-1.12,12,TRUE)
 for(i in seq_len(N)) {
  v<-rank$Accuracy_reversed_rank[[i]]
  if(is.finite(v)) {
   w<-(RANK_W-.23)*v/N;box(x+.18+w/2,yy[[i]],w,rowh*.9,rank_cols(v),"black",.5)
   txt(format(rank$Accuracy_rank[[i]],trim=TRUE),x+.1,yy[[i]],9)
  } else txt("NA",x+.12,yy[[i]],10,just="left")
 }
 txt("29 − rank",x+RANK_W/2,.99,10)
}
footnote<-function(width,view,kind="metrics",h=H) {
 label<-if(kind=="sd") "Mean batch-run SD per metric. Three significant digits; missing values are blank." else if(view=="Combined")
  "Shared metrics: real/simulation 50:50. GT and downsampling: simulation only. Bars: raw mean ± SE." else
  paste(view,"datasets. Bars: raw mean ± SE.")
 txt(label,width/2,.54,9)
}
save_fig<-function(name,draw,width,height=H,view="Combined",nrows=N,show_methods=TRUE,plotting) {
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
 b<-png::writePNG(png::readPNG(tmp,native=TRUE),target=raw(),dpi=300);writeBin(b,paste0(stem,".png"));unlink(tmp)
 plotting$figure<-name
  if(!"plot_row" %in% names(plotting)) plotting$plot_row<-match(plotting$method,order)
  plotting$y_inches<-row_positions(nrows,height)[plotting$plot_row]
  plotting$PDF_y_center_points<-(height-plotting$y_inches)*72
  if("metric" %in% names(plotting) && "se" %in% names(plotting)) {
    mm<-unique(plotting$metric)
    x0<-if(name=="Accuracy_Primary_Secondary_Combined") start_primary else if(show_methods) LABEL_W else PAD
    xx<-x0+(match(plotting$metric,mm)-1)*STEP
    if(name=="Accuracy_Primary_Secondary_Combined") xx<-xx+ifelse(match(plotting$metric,mm)>4,GAP-COL_GAP,0)
    plotting$panel_left_inches<-xx;plotting$panel_width_inches<-METRIC_W
    plotting$scale_min<-vapply(plotting$metric,function(m) panels[[paste(view,m,sep="|")]]$range[[1]],numeric(1))
    plotting$scale_max<-vapply(plotting$metric,function(m) panels[[paste(view,m,sep="|")]]$range[[2]],numeric(1))
  }
  if("label" %in% names(plotting)) {
    x0<-if(show_methods) LABEL_W else PAD
    plotting$PDF_x_center_points<-(x0+(plotting$plot_column-.5)*SD_W)*72
  }
  write.csv(plotting,paste0(stem,"_plotting.csv"),row.names=FALSE,na="NA")
 registry[[length(registry)+1L]]<<-data.frame(figure=name,view=view,width_inches=width,height_inches=height,
  n_rows=nrows,show_methods=show_methods,plotting_csv=paste0(name,"_plotting.csv"))
}

# Standalone blocks and strips: identical full-row geometry, ready for manual assembly.
for(view in view_order) {
 metrics<-if(view=="Real") PRIMARY[PRIMARY!="Ground-truth correlation"] else PRIMARY
 for(with_names in c(TRUE,FALSE)) local({v<-view;mm<-metrics;names_on<-with_names
  x0<-if(names_on) LABEL_W else PAD;width<-x0+block_span(length(mm))+PAD
  name<-paste0("Primary_Directionality_",v,if(names_on) "" else "_strip")
  plotting<-do.call(rbind,lapply(mm,metric_data,view=v))
  save_fig(name,function(){heading("Primary scientific metrics",x0,width-x0-PAD,v,fill="#E5B5D6");if(names_on) method_labels()
    for(j in seq_along(mm)) draw_metric(mm[[j]],v,x0+(j-1)*STEP);footnote(width,v)},width,view=v,show_methods=names_on,plotting=plotting)
 })
 metrics<-if(view=="Real") CONSISTENCY else c(CONSISTENCY,DOWN)
 for(with_names in c(TRUE,FALSE)) local({v<-view;mm<-metrics;names_on<-with_names
  x0<-if(names_on) LABEL_W else PAD;width<-x0+block_span(length(mm))+PAD
  name<-paste0("Secondary_Consistency_Stability_",v,if(names_on) "" else "_strip")
  plotting<-do.call(rbind,lapply(mm,metric_data,view=v))
  save_fig(name,function(){heading("Secondary scientific metrics",x0,width-x0-PAD,v,fill="#BCE3C8");if(names_on) method_labels()
    for(j in seq_along(mm)) draw_metric(mm[[j]],v,x0+(j-1)*STEP);footnote(width,v)},width,view=v,show_methods=names_on,plotting=plotting)
 })
}
width<-LABEL_W+RANK_W+PAD
save_fig("Accuracy_rank",function(){heading("Accuracy",LABEL_W,RANK_W,"Combined");method_labels();draw_rank(LABEL_W)
 txt("Rank 1 = best; bar length = 29 − rank.",width/2,.54,9)},width,plotting=rank)

start_primary<-LABEL_W+RANK_W+GAP;start_secondary<-start_primary+block_span(length(PRIMARY))+GAP
main_width<-start_secondary+block_span(3)+PAD
main_data<-do.call(rbind,lapply(c(PRIMARY,CONSISTENCY,DOWN),metric_data,view="Combined"))
save_fig("Accuracy_Primary_Secondary_Combined",function(){
 method_labels();heading("Accuracy",LABEL_W,RANK_W,"Rank");draw_rank(LABEL_W)
 heading("Primary scientific metrics",start_primary,block_span(4),"Directionality",fill="#E5B5D6")
 heading("Secondary scientific metrics",start_secondary,block_span(3),"Consistency and Stability",fill="#BCE3C8")
 for(j in seq_along(PRIMARY)) draw_metric(PRIMARY[[j]],"Combined",start_primary+(j-1)*STEP)
 for(j in seq_along(c(CONSISTENCY,DOWN))) draw_metric(c(CONSISTENCY,DOWN)[[j]],"Combined",start_secondary+(j-1)*STEP)
 footnote(main_width,"Combined")},main_width,plotting=main_data)

sd_data<-function(view,methods=order) {
 metrics<-if(view=="Real") c(PRIMARY[PRIMARY!="Ground-truth correlation"],CONSISTENCY) else c(PRIMARY,CONSISTENCY)
 d<-sd[sd$method %in% methods & sd$metric %in% metrics, ]
 value_col<-switch(view,Combined="combined_mean_sd",Real="real_mean_sd",Simulation="simulation_mean_sd")
 d$value<-d[[value_col]];d$label<-ifelse(is.finite(d$value),ifelse(d$value==0,"0",sprintf("%.3g",d$value)),"")
 d$plot_row<-match(d$method,methods);d$plot_column<-match(d$metric,metrics);d$view<-view
 d$hex<-unname(colors[d$method]);d[order(d$plot_row,d$plot_column), ]
}
sd_title_names<-c("Angle consistency"="Angle", "CBDir"="CBDir", "Transition V1"="Transition\nV1", "Ground-truth correlation"="GT corr.", "Consistency score"="Consistency\nscore", "ICCoh"="ICCoh")
draw_sd<-function(d,view,x0,methods=order,h=H) {
 metrics<-unique(d$metric[order(d$plot_column)]);yy<-row_positions(length(methods),h)
 heading("Stability: batch-run SD",x0,length(metrics)*SD_W,view,h,fill="#BCE3C8")
 for(j in seq_along(metrics)) txt(sd_title_names[[metrics[[j]]]],x0+(j-.5)*SD_W,h-1.12,10,TRUE)
 for(i in seq_len(nrow(d))) if(nzchar(d$label[[i]])) txt(d$label[[i]],x0+(d$plot_column[[i]]-.5)*SD_W,yy[[d$plot_row[[i]]]],11)
}
for(view in view_order) for(with_names in c(TRUE,FALSE)) local({v<-view;names_on<-with_names;d<-sd_data(v)
 x0<-if(names_on) LABEL_W else PAD;width<-x0+max(d$plot_column)*SD_W+PAD
 name<-paste0("Stability_batchSD_",v,if(names_on) "" else "_strip")
 save_fig(name,function(){if(names_on) method_labels();draw_sd(d,v,x0);footnote(width,v,"sd")},width,view=v,show_methods=names_on,plotting=d)
})
measured<-order[order %in% sd$method[is.finite(sd$combined_mean_sd)]]
compact_h<-2.55+(length(measured)+.2)*(H-2.55)/(N+.2);d<-sd_data("Combined",measured)
width<-LABEL_W+6*SD_W+PAD
save_fig("Stability_batchSD_measured_only",function(){method_labels(measured,compact_h);draw_sd(d,"Combined",LABEL_W,measured,compact_h)
 footnote(width,"Combined","sd",compact_h)},width,compact_h,nrows=length(measured),plotting=d)

write.csv(do.call(rbind,registry),file.path(PROJECT,"figure_index.csv"),row.names=FALSE)
write.csv(do.call(rbind,audit),file.path(PROJECT,"tables/actual_ggplot_bar_layers.csv"),row.names=FALSE)
write.csv(do.call(rbind,error_audit),file.path(PROJECT,"tables/actual_ggplot_errorbar_layers.csv"),row.names=FALSE)
row_layout<-data.frame(method=order,display_order=seq_len(N),y_inches=row_positions(),PDF_y_points=(H-row_positions())*72,
 body_bottom_inches=1.15,body_top_inches=H-1.4,height_inches=H)
write.csv(row_layout,file.path(PROJECT,"tables/full_row_layout.csv"),row.names=FALSE)
writeLines(font,file.path(PROJECT,"logs/font_family.txt"))
capture.output(sessionInfo(),file=file.path(PROJECT,"logs/R_sessionInfo.txt"))
if(layout_id %in% dev.list()) dev.off(layout_id);unlink(layout_file)
warning_lines<-capture.output(warnings());if(!length(warning_lines)) warning_lines<-"No R warnings."
writeLines(warning_lines,file.path(PROJECT,"logs/R_warnings.txt"))
message("Rendered ",length(registry)," actual R figures with fixed palette A and Accuracy-only order.")
