# The Cross-Boundary Direction Correctness 

CBDir metric evaluates the accuracy of RNA velocity vector fields at cellular transition boundaries. It measures the cosine similarity between the inferred velocity vector of a cell and the actual directional vector pointing toward its neighbors in the target destination cluster.

A higher score indicates that the velocity correctly points from a progenitor state to the differentiated state defined in the lineage.

## adata Requirements

The input .h5ad file must contain:

adata.X: Expression matrix (or specific layer) used for neighbor calculation.

adata.layers[vkey]: The velocity matrix (e.g., 'velocity').

adata.obs[celltype_key]: Cell type or cluster annotations.

adata.obsm[umap_key]: Low-dimensional embedding coordinates (e.g., 'X_umap') used to calculate the spatial direction of the transition.

## Output Interpretation

File: cbdir.txt (Saved in save_dir)

Format: Tab-separated values (TSV).

Content:

Columns represent specific transitions (e.g., Stem -> Progenitor).

Rows represent individual cell scores (if return_raw=True) or aggregate metrics.

High Score (>0): Velocity vectors align with the transition direction.

Low/Negative Score: Velocity vectors are orthogonal or opposite to the expected lineage trajectory.