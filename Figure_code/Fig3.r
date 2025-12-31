rm(list = ls())
library(ggplot2)
library(tidyverse)

####### 1. Inter/Intra Ratio###########
ratio_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-RealData/all_new_data_id/Inter_vs_intra_ratio.csv")
ratio_data <- ratio_data[,1:55]
ratio_data[is.na(ratio_data)] <- 0

ratio_df<- ratio_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
ratio_summary_df <- ratio_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

#Method order is same as it is in Rank list, to make sure color for each method is same in Figure 2 and 3.
rank_df<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-RealData/all/Reversed_rank_all.csv")
ratio_summary_df$Method <- factor(ratio_summary_df$Method, levels = rank_df$Method)

colors_21 <- c(
  "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00","#e82b91", "#6a3d9a",
  "#b15928", "#a6cee3", "#b2df8a", "#fb9a99", "grey",  
  "#cab2d6", "#ffff66", "#8dd3c7", "lightyellow", "#9970ab",                     
  "#fb8072", "#80b1d3", "#fdb468", "#b3de69", "#fccde5"
)

p0 <- ggplot(ratio_summary_df, aes(x = Method, y = mean_value, fill = mean_value)) +
  geom_blank() +   # do not draw bars
  theme_minimal() +
  coord_flip() +
  labs(title = "") +
  theme(
    legend.position = "none",          # remove legend
    panel.background = element_blank(),# remove background
    panel.grid = element_blank(),      # remove gridlines
    panel.border = element_blank(),    # remove border
    axis.line = element_blank(),       # remove axis lines
    axis.text.x = element_blank(),     # remove x-axis tick labels
    axis.title.x = element_blank(),    # remove x-axis title
    axis.title.y = element_blank(),    # remove y-axis title(delete this line if you want to keep it)
    axis.text.y = element_text(size = 14, face = "bold")
  )

p_ratio <- ggplot(ratio_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  #labs(title = "Inter/Intra Ratio") +
  labs(title = "Ratio") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )


######## 2. Inter-class Distance###########
inter_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-RealData/all_new_data_id/Inter_class_distance.csv")
inter_data <- inter_data[,1:55]
inter_data[is.na(inter_data)] <- 0

inter_df<- inter_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
inter_summary_df <- inter_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

inter_summary_df$Method <- factor(inter_summary_df$Method, levels = rank_df$Method)

p_inter <- ggplot(inter_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  #labs(title = "Inter-class Distance") +
  labs(title = "Inter") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )


######## 3. Intra-class Distance###########
intra_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-RealData/all_new_data_id/Intra_class_distance.csv")
intra_data <- intra_data[,1:55]
intra_data[is.na(intra_data)] <- 0

intra_df<- intra_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
intra_summary_df <- intra_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

intra_summary_df$Method <- factor(intra_summary_df$Method, levels = rank_df$Method)

p_intra <- ggplot(intra_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  #labs(title = "Intra-class Distance") +
  labs(title = "Intra") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

######## 4. Consistency Score###########
cs_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-RealData/all_new_data_id/Consistency_Score.csv")
cs_data <- cs_data[,1:55]
cs_data[is.na(cs_data)] <- 0

cs_df<- cs_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
cs_summary_df <- cs_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

cs_summary_df$Method <- factor(cs_summary_df$Method, levels = rank_df$Method)

p_cs <- ggplot(cs_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  #labs(title = "Consistency Score") +
  labs(title = "Consistency") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

######## 5. Velocity Angle###########
angle_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-RealData/all_new_data_id/Velocity_Angle.csv")
angle_data <- angle_data[,1:46]
angle_data[is.na(angle_data)] <- 0

angle_df<- angle_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
angle_summary_df <- angle_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

angle_summary_df$Method <- factor(angle_summary_df$Method, levels = rank_df$Method)

p_angle <- ggplot(angle_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  #labs(title = "Velocity Angle") +
  labs(title = "Angle") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

######## 6. ICCoh ###########
ICCoh_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-RealData/all_new_data_id/ICCoh.csv")
ICCoh_data <- ICCoh_data[,1:55]
ICCoh_data[is.na(ICCoh_data)] <- 0

ICCoh_df<- ICCoh_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
ICCoh_summary_df <- ICCoh_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

ICCoh_summary_df$Method <- factor(ICCoh_summary_df$Method, levels = rank_df$Method)

p_ICCoh <- ggplot(ICCoh_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  labs(title = "ICCoh") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )


######## 7. CBDir ###########
CBDir_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-RealData/all_new_data_id/CBDir.csv")
CBDir_data <- CBDir_data[,1:46]
CBDir_data[is.na(CBDir_data)] <- 0

CBDir_df<- CBDir_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
CBDir_summary_df <- CBDir_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

CBDir_summary_df$Method <- factor(CBDir_summary_df$Method, levels = rank_df$Method)

p_CBDir <- ggplot(CBDir_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  labs(title = "CBDir") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

library(gridExtra)
pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/SupplFigure_ACC_real.pdf", 14, 12)
grid.arrange(p0, p_cs, p_angle, p_inter, p_intra, p_ratio, p_CBDir, p_ICCoh, nrow = 1)
#grid.arrange(p0, p_ratio, nrow = 1)
dev.off()

## Usability
usa_rank_df<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Usability/Velocity_Usability1010.csv")
usa_score <- usa_rank_df[,1:6, drop = FALSE]

usa_score_df<- usa_score %>%
  pivot_longer(-Method, names_to = "Metric", values_to = "Score")

usa_score_df$Method <- factor(usa_score_df$Method, levels = method_levels)

p <- ggplot(usa_score_df, aes(x = Metric, y = Method)) +
  # determine shape by Score: <0.8 circles, ≥0.8 squares
  geom_point(aes(size = Score, fill = Score, shape = Score >= 8.5), color = "grey30") +
  scale_shape_manual(values = c(21, 22)) +   # 21=circle, 22=square
  scale_size_continuous(range = c(4, 18)) +
  # high values are lighter; low values are darker
  scale_fill_gradient(low = "#fff7bc", high = "#d95f0e") +
  labs(title = "Usability") +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.text = element_blank(),
    axis.title = element_blank(),
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 10),
    #panel.border = element_rect(fill = NA, color = "grey70"),
    plot.margin = margin(10, 20, 10, 10)
  ) +
  labs(fill = "Score", size = "Score", shape = "High Score (≥8.5)")
###################################################################################
usa_rank_df <- read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Usability/Velocity_Usability1010.csv")
usa_score   <- usa_rank_df[,1:6, drop = FALSE]

usa_score_df <- usa_score %>%
  pivot_longer(-Method, names_to = "Metric", values_to = "Score")

usa_score_df$Method <- factor(usa_score_df$Method, levels = method_levels)

## key: apply a magnification transform so high scores appear larger
min_s <- min(usa_score_df$Score, na.rm = TRUE)
usa_score_df$Size <- (usa_score_df$Score - min_s + 1)^2   # square scaling

p <- ggplot(usa_score_df, aes(x = Metric, y = Method)) +
  geom_point(
    aes(size = Size, fill = Score),   # use Size for point size; keep original Score for color
    shape = 21,
    color = "grey30"
  ) +
  # control actual point-size range; tune as needed
  scale_size_continuous(
    range  = c(2, 16),
    breaks = (usa_score_df$Score - min_s + 1)^2 %>% unique() %>% sort(),
    labels = usa_score_df$Score %>% unique() %>% sort(),
    name   = "Score"
  ) +
  scale_fill_gradient(low = "#fff7bc", high = "#d95f0e", name = "Score") +
  labs(title = "Usability") +
  theme_minimal() +
  theme(
    panel.grid   = element_blank(),
    axis.text    = element_blank(),
    axis.title   = element_blank(),
    legend.title = element_text(size = 12, face = "bold"),
    legend.text  = element_text(size = 10),
    plot.margin  = margin(10, 20, 10, 10)
  )
p


library(gridExtra)
pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure3_Usability2.pdf", 37, 12)
grid.arrange(p0, p_cs, p_angle, p_inter, p_intra, p_ratio, p_CBDir, p_ICCoh, p, nrow = 1)
dev.off()

pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure3_Usability3roll.pdf",20, 12)
grid.arrange(p0, p_cs, p_angle, p, nrow = 1)
dev.off()



###############################
###############################
##########Simulated data######

####### 1. Inter/Intra Ratio###########
df <- read_excel("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/bar for fig 2.xlsx")

df <- df[order(df$Overall), ]
df$Method <- factor(df$Method, levels = df$Method)
n_methods <- nrow(df)

method_levels <- as.character(df$Method)


ratio_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-Simulate/new/Inter_vs_Intra_ratio.csv")
ratio_data <- ratio_data[,1:90]
ratio_data[is.na(ratio_data)] <- 0

ratio_df<- ratio_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
ratio_summary_df <- ratio_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

#Method order is same as it is in Rank list, to make sure color for each method is same in Figure 2 and 3.
rank_df<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-Simulate/new/Reversed_rank_all_simulate.csv")
ratio_summary_df$Method <- factor(ratio_summary_df$Method, levels = method_levels)

colors_21 <- c(
  "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00","#e82b91", "#6a3d9a",
  "#b15928", "#a6cee3", "#b2df8a", "#fb9a99", "grey",  
  "#cab2d6", "#ffff66", "#8dd3c7", "lightyellow", "#9970ab",                     
  "#fb8072", "#80b1d3", "#fdb468", "#b3de69", "#fccde5"
)

p0 <- ggplot(ratio_summary_df, aes(x = Method, y = mean_value, fill = mean_value)) +
  geom_blank() +   # do not draw bars
  theme_minimal() +
  coord_flip() +
  labs(title = "") +
  theme(
    legend.position = "none",          # remove legend
    panel.background = element_blank(),# remove background
    panel.grid = element_blank(),      # remove gridlines
    panel.border = element_blank(),    # remove border
    axis.line = element_blank(),       # remove axis lines
    axis.text.x = element_blank(),     # remove x-axis tick labels
    axis.title.x = element_blank(),    # remove x-axis title
    axis.title.y = element_blank(),    # remove y-axis title(delete this line if you want to keep it)
    axis.text.y = element_text(size = 14, face = "bold")
  )

p_ratio <- ggplot(ratio_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  #labs(title = "Inter/Intra Ratio") +
  labs(title = "Ratio") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )
p_ratio

######## 2. Inter-class Distance###########
inter_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-Simulate/new/Inter_class_distance.csv")
inter_data <- inter_data[,1:90]
inter_data[is.na(inter_data)] <- 0

inter_df<- inter_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
inter_summary_df <- inter_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

inter_summary_df$Method <- factor(inter_summary_df$Method, levels = method_levels)

p_inter <- ggplot(inter_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  #labs(title = "Inter-class Distance") +
  labs(title = "Inter") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )


######## 3. Intra-class Distance###########
intra_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-Simulate/new/Intra_class_distance.csv")
intra_data <- intra_data[,1:90]
intra_data[is.na(intra_data)] <- 0

intra_df<- intra_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
intra_summary_df <- intra_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

intra_summary_df$Method <- factor(intra_summary_df$Method, levels = method_levels)

p_intra <- ggplot(intra_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  #labs(title = "Intra-class Distance") +
  labs(title = "Intra") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

######## 4. Consistency Score###########
cs_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-Simulate/new/Consistency_Score.csv")
cs_data <- cs_data[,1:90]
cs_data[is.na(cs_data)] <- 0

cs_df<- cs_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
cs_summary_df <- cs_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

cs_summary_df$Method <- factor(cs_summary_df$Method, levels = method_levels)

p_cs <- ggplot(cs_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  #labs(title = "Consistency Score") +
  labs(title = "Consistency") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

######## 5. Velocity Angle###########
angle_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-Simulate/new/Velocity_Angle.csv")
angle_data <- angle_data[,1:90]
angle_data[is.na(angle_data)] <- 0

angle_df<- angle_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
angle_summary_df <- angle_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

angle_summary_df$Method <- factor(angle_summary_df$Method, levels = method_levels)

p_angle <- ggplot(angle_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  #labs(title = "Velocity Angle") +
  labs(title = "Angle") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

######## 6. ICCoh ###########
ICCoh_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-Simulate/new/ICCoh.csv")
ICCoh_data <- ICCoh_data[,1:90]
ICCoh_data[is.na(ICCoh_data)] <- 0

ICCoh_df<- ICCoh_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
ICCoh_summary_df <- ICCoh_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

ICCoh_summary_df$Method <- factor(ICCoh_summary_df$Method, levels = method_levels)

p_ICCoh <- ggplot(ICCoh_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  labs(title = "ICCoh") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )


######## 7. CBDir ###########
CBDir_data<-read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-Simulate/new/CBDir.csv")
CBDir_data <- CBDir_data[,1:90]
CBDir_data[is.na(CBDir_data)] <- 0

CBDir_df<- CBDir_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

# compute mean and standard error
CBDir_summary_df <- CBDir_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se = sd(Value, na.rm = TRUE) / sqrt(n())
  )

CBDir_summary_df$Method <- factor(CBDir_summary_df$Method, levels = method_levels)

p_CBDir <- ggplot(CBDir_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +         # bars
  geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                width = 0.2, color = "black") +          # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  labs(title = "CBDir") +
  theme(
    legend.position = "none",                     # remove legend
    panel.background = element_blank(),           # remove background
    panel.grid = element_blank(),                 # remove gridlines
    #panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # outer black frame
    axis.line = element_blank(),                   # remove axis lines
    axis.text = element_blank(),                  # remove axis tick labels
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )


######## 8. Groundtruth correlation ###########
## 1. Read ground-truth correlation data
GTcorr_data <- read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/Benchmarking-Simulate/new/ALL_groundtruth_correlation.csv")
GTcorr_data <- GTcorr_data[,1:90]
GTcorr_data[is.na(GTcorr_data)] <- 0

## 2. Pivot to long format: Method × Sample → Value
GTcorr_df <- GTcorr_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

## 3. Compute mean and standard error
GTcorr_summary_df <- GTcorr_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se         = sd(Value, na.rm = TRUE) / sqrt(n())
  )

## 4. Keep method order consistent with rank_df
GTcorr_summary_df$Method <- factor(GTcorr_summary_df$Method,
                                   levels = method_levels)

## 5. Plot with the same style as p_CBDir
p_GTcorr <- ggplot(GTcorr_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +   # bars
  geom_errorbar(aes(ymin = mean_value - se,
                    ymax = mean_value + se),
                width = 0.2, color = "black") +    # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  labs(title = "Groundtruth correlation") +
  theme(
    legend.position   = "none",          # remove legend
    panel.background  = element_blank(), # remove background
    panel.grid        = element_blank(), # remove gridlines
    axis.line         = element_blank(), # remove axis lines
    axis.text         = element_blank(), # remove tick labels
    axis.title.x      = element_blank(),
    axis.title.y      = element_blank(),
    plot.title        = element_text(hjust = 0.5)
  )

########9. downsampling####
Down_data <- read.csv("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/scalability/Downsampling_groundtruth_correlation.csv")
Down_data <- Down_data[, 1:11]          # same as other simulated metrics: take the first 11 columns
Down_data[is.na(Down_data)] <- 0

## pivot to long format:Method × Sample → Value
Down_df <- Down_data %>%
  pivot_longer(-Method, names_to = "Sample", values_to = "Value")

## compute mean and standard error
Down_summary_df <- Down_df %>%
  group_by(Method) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    se         = sd(Value, na.rm = TRUE) / sqrt(n())
  )

## Method order is method_levels / rank_df kept consistent
Down_summary_df$Method <- factor(Down_summary_df$Method,
                                 levels = method_levels)

## Plot style is consistent with p_CBDir / p_GTcorr exactly the same
p_Down <- ggplot(Down_summary_df, aes(x = Method, y = mean_value, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +   # bars
  geom_errorbar(aes(ymin = mean_value - se,
                    ymax = mean_value + se),
                width = 0.2, color = "black") +                 # error bars
  scale_fill_manual(values = colors_21) +
  theme_minimal() +
  coord_flip() +
  labs(title = "Downsampling") +
  theme(
    legend.position   = "none",
    panel.background  = element_blank(),
    panel.grid        = element_blank(),
    axis.line         = element_blank(),
    axis.text         = element_blank(),
    axis.title.x      = element_blank(),
    axis.title.y      = element_blank(),
    plot.title        = element_text(hjust = 0.5)
  )

########10. barchrun####

#shown as table


###########scalability#####

library(ComplexHeatmap)
library(circlize)
library(grid)
#######1. speed----------
speed_csv_path <- "C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/scalability/docker_speed_dim_means.csv"

spd_df <- read.csv(speed_csv_path,
                   check.names = FALSE,
                   stringsAsFactors = FALSE)

# assume the first column is the method name
method_col <- 1
rownames(spd_df) <- spd_df[[method_col]]
spd_df[[method_col]] <- NULL

# ensure the column order is the 5 structures (adjust here if you need a specific order)
# for example:
# spd_df <- spd_df[, c("1000x1000","1000x10000","1000x200000","10000x1000","200000x1000")]

# convert seconds to minutes（(comment this out if already converted)）
spd_mat <- as.matrix(spd_df) / 60

# 2. reorder rows by method_levels ---------------------------------
# note: method_levels should be defined beforehand to keep consistent with previous figures
# e.g.,
# method_levels <- c("LatentVelo","TopicVelo", ...,"veloVI")

method_levels_chr <- as.character(method_levels)

# keep only methods present in the matrix and preserve the method_levels order
row_order <- method_levels_chr[method_levels_chr %in% rownames(spd_mat)]
row_order <- rev(row_order)
# reorder rows by row_order
mat_sp <- spd_mat[row_order, , drop = FALSE]

# 3. Buildcolor mapping:quantiles + truncate extreme values -----------------------

vals <- as.vector(mat_sp)
vals <- vals[!is.na(vals)]

vmin <- min(vals, na.rm = TRUE)
vmax_all <- max(vals, na.rm = TRUE)

# lock extreme large values (top two) as the darkest colors; use the 3rd largest value as upper limit
pos_vals  <- sort(unique(vals[vals > 0]), decreasing = TRUE)

if (length(pos_vals) >= 2) {
  vmax_trunc <- pos_vals[2]
} else {
  vmax_trunc <- vmax_all
}
if (!is.finite(vmax_trunc) || vmax_trunc <= vmin) vmax_trunc <- vmax_all

# compute quantiles only within (0, vmax_trunc]
vals_used <- vals[vals > 0 & vals <= vmax_trunc]

# adjust quantile points as needed (more points = finer)
brks <- c(
  vmin,
  quantile(vals_used, probs = c(0.1, 0.3, 0.5, 0.7, 0.9), na.rm = TRUE),
  vmax_trunc
)
brks <- sort(unique(brks))

# choose palette (BluGrn) and drop the lightest segment to avoid confusion with NA grey
palette_name <- "BluGrn"
full_col  <- grDevices::hcl.colors(256, palette = palette_name, rev = TRUE)
full_col  <- full_col[41:200]   # drop the lightest 40 colors

# select a color for each breakpoint
cols_for_brks <- full_col[round(seq(1, length(full_col), length.out = length(brks)))]

# mapping function: value → color
col_fun <- circlize::colorRamp2(brks, cols_for_brks)

# use these breakpoints for legend ticks
legend_breaks <- brks

# 4. Draw heatmap ---------------------------------------------------
ht_speed <- Heatmap(
  mat_sp,
  name = "Speed (min)",
  col  = col_fun,
  na_col = "grey90",
  cluster_rows    = FALSE,
  cluster_columns = FALSE,
  column_names_side     = "top",
  column_names_rot      = 45,
  column_names_centered = TRUE,
  row_names_gp          = gpar(fontsize = 8),
  column_names_gp       = gpar(fontsize = 8),
  
  # separate cells with white grid lines; no outer border
  rect_gp = gpar(col = "white", lwd = 4),
  
  # write values in each cell (two decimals); NA as "NA"
  cell_fun = function(j, i, x, y, w, h, fill) {
    val <- mat_sp[i, j]
    if (is.na(val)) {
      grid.text("NA", x, y, gp = gpar(fontsize = 14, col = "black"))
    } else {
      # use white text for high values and black for low values
      mid <- (vmin + vmax_trunc) / 2
      txt_col <- ifelse(val > mid, "white", "black")
      grid.text(sprintf("%.2f", val),
                x, y,
                gp = gpar(fontsize = 14, col = txt_col))
    }
  },
  
  heatmap_legend_param = list(
    at     = legend_breaks,
    labels = sprintf("%.2f", legend_breaks),
    title  = "Speed (min)",
    legend_direction = "vertical",
    title_position   = "leftcenter-rot"
  ),
  
  use_raster       = TRUE,
  raster_quality   = 2
)

# 5. Output / display --------------------------------------------


pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure3_docker_speed_heatmap.pdf",
    width = 9.5, height = 14)
draw(ht_speed)
dev.off()

#####2.memory--------
mem_csv_path <- "C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/PlotData/scalability/docker_memory_dim_means.csv"

mem_df <- read.csv(mem_csv_path,
                   check.names = FALSE,
                   stringsAsFactors = FALSE)

# the first column is the method name
method_col <- 1
rownames(mem_df) <- mem_df[[method_col]]
mem_df[[method_col]] <- NULL

# rows = methods; columns = 5 structures (reorder columns here if needed)
mat_mem <- as.matrix(mem_df)

## 2. Reorder rows by method_levels and reverse (same as the speed heatmap) ----
method_levels_chr <- as.character(method_levels)

row_order <- method_levels_chr[method_levels_chr %in% rownames(mat_mem)]
row_order <- rev(row_order)   # reverse so veloVI is at the top and SDEvelo below

mat_mem <- mat_mem[row_order, , drop = FALSE]

## 3. Color mapping: use the same logic as the speed heatmap ----------------------------

vals <- as.vector(mat_mem)
vals <- vals[!is.na(vals)]

vmin     <- min(vals, na.rm = TRUE)
vmax_all <- max(vals, na.rm = TRUE)

# truncate extreme large values: lock top two as darkest; use the 2nd largest value as upper limit
pos_vals <- sort(unique(vals[vals > 0]), decreasing = TRUE)
if (length(pos_vals) >= 2) {
  vmax_trunc <- pos_vals[2]
} else {
  vmax_trunc <- vmax_all
}
if (!is.finite(vmax_trunc) || vmax_trunc <= vmin) vmax_trunc <- vmax_all

# compute quantiles within (0, vmax_trunc]
vals_used <- vals[vals > 0 & vals <= vmax_trunc]

brks <- c(
  vmin,
  quantile(vals_used, probs = c(0.1, 0.3, 0.5, 0.7, 0.9), na.rm = TRUE),
  vmax_trunc
)
brks <- sort(unique(brks))

# palette: BluGrn; drop the lightest segment to avoid confusion with NA grey
palette_name <- "BluGrn"
full_col  <- grDevices::hcl.colors(256, palette = palette_name, rev = TRUE)
full_col  <- full_col[41:200]

cols_for_brks <- full_col[round(seq(1, length(full_col), length.out = length(brks)))]
col_fun <- circlize::colorRamp2(brks, cols_for_brks)

legend_breaks <- brks

## 4. Draw the memory heatmap ---------------------------------------------
ht_memory <- Heatmap(
  mat_mem,
  name = "Memory (GB)",           # if not GB, change the title
  col  = col_fun,
  na_col = "grey90",
  cluster_rows    = FALSE,
  cluster_columns = FALSE,
  column_names_side     = "top",
  column_names_rot      = 45,
  column_names_centered = TRUE,
  row_names_gp          = gpar(fontsize = 8),
  column_names_gp       = gpar(fontsize = 8),
  rect_gp = gpar(col = "white", lwd = 4),
  
  cell_fun = function(j, i, x, y, w, h, fill) {
    val <- mat_mem[i, j]
    if (is.na(val)) {
      grid.text("NA", x, y, gp = gpar(fontsize = 14, col = "black"))
    } else {
      mid <- (vmin + vmax_trunc) / 2
      txt_col <- ifelse(val > mid, "white", "black")
      grid.text(sprintf("%.2f", val),
                x, y,
                gp = gpar(fontsize = 14, col = txt_col))
    }
  },
  
  heatmap_legend_param = list(
    at     = legend_breaks,
    labels = sprintf("%.2f", legend_breaks),
    title  = "Memory (GB)",
    legend_direction = "vertical",
    title_position   = "leftcenter-rot"
  ),
  
  use_raster     = TRUE,
  raster_quality = 2
)

## 5. Output --------------------------------------------------------
pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure3_docker_memory_heatmap.pdf",
    width = 9.5, height = 14)
draw(ht_memory)
dev.off()

####Combine plots---------
pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure3_AllMetrics_simulate4.pdf", 50, 12)
grid.arrange(p0, p_cs, p_angle, p_inter, p_intra, p_ratio, p_CBDir, p_ICCoh, p_GTcorr, p_Down, p, nrow = 1)
dev.off()


pdf("C:/Users/khuang6/Documents/2024Velocity/Fig/Velocity_Fig_2_3_KH/Figures/Figure3_AllMetrics_simulate4.pdf", 50, 12)
grid.arrange(p0, p_cs, p_angle, p_inter, p_intra, p_ratio, p_CBDir, p_ICCoh, p_GTcorr, p_Down, p, nrow = 1)
dev.off()












