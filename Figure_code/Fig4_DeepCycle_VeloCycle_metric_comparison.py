import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, rand_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# Read data
file_path = "/data_d/ZXL/RNA_Velocity/DeepCycle-main/result/phase_result/merged_all_datasets.csv"
merged_df = pd.read_csv(file_path, sep=",")

# Calculate metrics function
def calculate_metrics(group):
    true_labels = group["milestone"]
    pred_labels = group["predicted_phase"]
    
    # Compute metrics
    ari = adjusted_rand_score(true_labels, pred_labels) if len(set(true_labels)) > 1 else None
    nmi = normalized_mutual_info_score(true_labels, pred_labels) if len(set(true_labels)) > 1 else None
    ri = rand_score(true_labels, pred_labels) if len(set(true_labels)) > 1 else None
    accuracy = accuracy_score(true_labels, pred_labels) if len(set(true_labels)) > 1 else None
    precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0) if len(set(true_labels)) > 1 else None
    recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0) if len(set(true_labels)) > 1 else None
    fscore = f1_score(true_labels, pred_labels, average="macro", zero_division=0) if len(set(true_labels)) > 1 else None
    
    return pd.Series({
        "ari": ari, "nmi": nmi, "ri": ri,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "fscore": fscore
    })

# Group and calculate
metrics_df = merged_df.groupby("dataset").apply(calculate_metrics).reset_index()

# Save metrics
output_path = "/data_d/ZXL/RNA_Velocity/DeepCycle-main/result/phase_result/dataset_metrics.csv"
metrics_df.to_csv(output_path, sep=",", index=False)

print(f"Metrics saved to: {output_path}")
print(metrics_df.head())