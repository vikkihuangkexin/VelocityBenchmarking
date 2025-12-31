# Predict cell cycle
rm(list = ls())
library(mclust)
library(readr)

# Read data
data <- read_delim(
  "/data_d/ZXL/RNA_Velocity/DeepCycle-main/result/phase_result/cycle-simple_cell500_gene500_phase_deepcycle_phase.csv",
  delim = ",",
  show_col_types = FALSE
)

# Extract features and labels
pseudotime <- data$cell_cycle_theta
groundtruth <- data$milestone

# Train model and predict
mclust_result <- MclustDA(pseudotime, groundtruth)
pred <- predict(mclust_result, newdata = pseudotime)
data$predicted_phase <- pred$classification

# Save result
write_delim(
  data,
  file = "./cycle-simple_cell500_gene500_phase_deepcycle_phase.csv",
  delim = ",",
  col_names = TRUE
)

cat("Data saved, first 5 rows:\n")
print(head(data))