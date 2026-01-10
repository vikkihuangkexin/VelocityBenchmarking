#!/usr/bin/env python3
# step7_prepare_chromatin_velocity.py
# Prepare AnnData object for chromatin velocity from fused data

import scanpy as sc
import anndata
import scvelo as scv

# ---------------------------
# Load processed objects
# ---------------------------
print("Loading processed objects...")
fdata = sc.read("Fused_data.h5ad")
adata_tn5 = sc.read("adata_tn5_processed.h5ad")
adata_tnH = sc.read("adata_tnH_processed.h5ad")

# ---------------------------
# Find intersecting features and cells
# ---------------------------
var_names = list(set(fdata.var_names)
                 .intersection(adata_tnH.var_names)
                 .intersection(adata_tn5.var_names))
obs_names = list(set(fdata.obs_names)
                 .intersection(adata_tnH.obs_names)
                 .intersection(adata_tn5.obs_names))

print(f"Common genes/features: {len(var_names)}")
print(f"Common cells: {len(obs_names)}")

if len(var_names) == 0 or len(obs_names) == 0:
    raise RuntimeError("No common features or cells found across datasets.")

# subset tn5/tnH to common cells
adata_tn5 = adata_tn5[obs_names]
adata_tnH = adata_tnH[obs_names]

# ---------------------------
# Build new AnnData with spliced/unspliced layers
# ---------------------------
print("Building velocity-ready AnnData...")
adata = anndata.AnnData(adata_tn5.raw[:, var_names].X)
adata.layers['spliced'] = adata_tnH.raw[:, var_names].X
adata.layers['unspliced'] = adata_tn5.raw[:, var_names].X
adata.obs_names = obs_names
adata.var_names = var_names

# ---------------------------
# Subset fdata and transfer embeddings / graphs
# ---------------------------
fdata = fdata[:, var_names]
fdata = fdata[obs_names]

print("Transferring embeddings and graphs...")
for c in fdata.obsm.keys():
    adata.obsm[c] = fdata.obsm[c]
for c in fdata.obsp.keys():
    adata.obsp[c] = fdata.obsp[c]
for c in ['neighbors', 'pca', 'umap']:
    if c in fdata.uns:
        adata.uns[c] = fdata.uns[c]

# ---------------------------
# Transfer annotations
# ---------------------------
print("Transferring metadata...")
for c in ['batch', 'sum_peaks', 'coverage']:
    if c in fdata.obs.columns:
        adata.obs[c] = fdata.obs[c]
for c in fdata.var.columns:
    adata.var[c] = fdata.var[c]

# ---------------------------
# Calculate moments (without recomputing kNN)
# ---------------------------
print("Calculating moments...")
scv.pp.moments(adata, method="umap")

# ---------------------------
# Save velocity-ready AnnData
# ---------------------------
adata.write("ChromatinVelocity_ready.h5ad")
print("Saved ChromatinVelocity_ready.h5ad ?")
