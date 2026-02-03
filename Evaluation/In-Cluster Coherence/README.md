# In-Cluster Coherence (ICCoh) 

In-Cluster Coherence (ICCoh) metric assesses the smoothness and local consistency of the estimated RNA velocity vector field. It calculates the average cosine similarity between a cell's velocity vector and the velocity vectors of its neighbors within the same cluster.

A higher score implies that cells of the same type share a consistent direction of differentiation or state change, indicating a high-quality, non-chaotic velocity field.

## adata Requirements

The input .h5ad file must contain:

adata.layers[vkey]: The velocity matrix (e.g., 'velocity').

adata.obs[celltype_key]: Cell type or cluster annotations.

Neighbors Graph: While the script recalculates this via scv.pp.neighbors, the input data should ideally be preprocessed (PCA, etc.).

## Output Interpretation

File: iccoh.tsv (Saved in save_dir)

Format: Tab-separated values (TSV).

Content:

Columns represent cell types/clusters.

Rows contain coherence scores for individual cells within that cluster.

High Score (~1.0): Very high consistency; all neighbors move in the same direction.

Low Score (~0.0): Random or noisy velocity vectors within the cluster.