# Chromatin Velocity Preparation

This script prepares chromatin velocity data by fusing multiple datasets and creating a velocity-ready AnnData object.

## Required adata Internal Variables

- **Input h5ad files**: DHS_tn5_CH.h5ad, DHS_tnH_CH.h5ad, complDHS_tn5_CH.h5ad, complDHS_tnH_CH.h5ad
  - X: Expression matrix
  - obs: Metadata including 'batch' (optional)
  - var: Feature metadata

## Output Variables/Content

- **Fused_data.h5ad**: Fused dataset with embeddings and graphs.
- **ChromatinVelocity_ready.h5ad**: Velocity-ready AnnData with layers 'spliced' and 'unspliced', transferred embeddings, and computed moments.