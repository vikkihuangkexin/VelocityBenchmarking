#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import scanpy as sc
import anndata as ad

# ---------------------------
# Arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Prepare chromatin velocity data.")
parser.add_argument("--tn5", required=True, help="Path to DHS_tn5_CH.h5ad")
parser.add_argument("--tnH", required=True, help="Path to DHS_tnH_CH.h5ad")
parser.add_argument("--ctn5", required=True, help="Path to complDHS_tn5_CH.h5ad")
parser.add_argument("--ctnH", required=True, help="Path to complDHS_tnH_CH.h5ad")
parser.add_argument("--out-dir", default=".", help="Directory to save output files")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

fn_tn5 = Path(args.tn5)
fn_tnH = Path(args.tnH)
fn_c_tn5 = Path(args.ctn5)
fn_c_tnH = Path(args.ctnH)

# thresholds
COMMONNESS_PCT = 95
SUM_PEAKS_THRESHOLD = 300

# ---------------------------
# Load data
# ---------------------------
print("Loading files...")
adata_tn5 = sc.read(fn_tn5)
adata_tnH = sc.read(fn_tnH)
cdata_tn5 = sc.read(fn_c_tn5)
cdata_tnH = sc.read(fn_c_tnH)
print("Loaded AnnData shapes:")
print("adata_tn5:", adata_tn5.shape)
print("adata_tnH:", adata_tnH.shape)
print("cdata_tn5:", cdata_tn5.shape)
print("cdata_tnH:", cdata_tnH.shape)

datasets = {
    "adata_tn5": adata_tn5,
    "adata_tnH": adata_tnH,
    "cdata_tn5": cdata_tn5,
    "cdata_tnH": cdata_tnH
}

# ---------------------------
# Per-dataset QC & safe filtering
# ---------------------------
for name, ds in datasets.items():
    # ensure var_names
    if ds.var_names is None or len(ds.var_names) == 0:
        ds.var_names = [f"feature_{i}" for i in range(ds.shape[1])]

    # drop obs starting with 'Bac'
    keep_cells = [x for x in ds.obs_names if not str(x).startswith('Bac')]
    ds = ds[keep_cells].copy()

    # compute sum_peaks and coverage
    ds.obs['sum_peaks'] = np.asarray((ds.X > 0).sum(axis=1)).ravel()
    ds.obs['coverage'] = np.asarray(ds.X.sum(axis=1)).ravel()

    # compute commonness
    ds.var['commonness'] = np.asarray((ds.X > 0).sum(axis=0)).ravel()

    # safe commonness threshold: ensure at least 100 features kept
    var_commonness = ds.var['commonness'].values
    c_thr = np.percentile(var_commonness, COMMONNESS_PCT)
    keep_vars = var_commonness > c_thr
    if keep_vars.sum() < 100:
        # lower threshold if too few features
        c_thr = np.percentile(var_commonness, 50)
        keep_vars = var_commonness > c_thr
    ds = ds[:, keep_vars].copy()

    # filter cells by sum_peaks
    keep_cells2 = ds.obs['sum_peaks'].values > SUM_PEAKS_THRESHOLD
    ds = ds[keep_cells2].copy()

    datasets[name] = ds
    print(f"{name} shape after filtering: {ds.shape}")

# reassign cleaned datasets
adata_tn5 = datasets['adata_tn5']
adata_tnH = datasets['adata_tnH']
cdata_tn5 = datasets['cdata_tn5']
cdata_tnH = datasets['cdata_tnH']

# ---------------------------
# Preprocessing: normalize, log, regress_out, scale, PCA, BBKNN
# ---------------------------
def preprocess(ds, batch_key='batch'):
    sc.pp.normalize_total(ds)
    sc.pp.log1p(ds)
    ds.raw = ds
    sc.pp.regress_out(ds, keys=['sum_peaks'])
    sc.pp.scale(ds, max_value=10)
    sc.tl.pca(ds, svd_solver='arpack')
    try:
        bbknn_neighbors = int(np.sqrt(ds.shape[0]) / 2 / max(1, len(ds.obs[batch_key].unique())))
        sc.external.pp.bbknn(ds, batch_key=batch_key, neighbors_within_batch=bbknn_neighbors, n_pcs=30)
    except Exception as e:
        print("  BBKNN failed:", e)
    return ds

# estimate sample count from 'batch'
samples_count = max([len(ds.obs['batch'].unique()) if 'batch' in ds.obs.columns else 1
                     for ds in [adata_tn5, adata_tnH, cdata_tn5, cdata_tnH]])

adata_tn5 = preprocess(adata_tn5)
adata_tnH = preprocess(adata_tnH)
cdata_tn5 = preprocess(cdata_tn5)
cdata_tnH = preprocess(cdata_tnH)

# ---------------------------
# Save processed intermediate files
# ---------------------------
processed_tn5 = out_dir / "adata_tn5_processed.h5ad"
processed_tnH = out_dir / "adata_tnH_processed.h5ad"
processed_ctn5 = out_dir / "cdata_tn5_processed.h5ad"
processed_ctnH = out_dir / "cdata_tnH_processed.h5ad"

adata_tn5.write(processed_tn5)
adata_tnH.write(processed_tnH)
cdata_tn5.write(processed_ctn5)
cdata_tnH.write(processed_ctnH)
print("Processed files saved.")

# ---------------------------
# Take common cells
# ---------------------------
cells = set(cdata_tnH.obs_names) & set(cdata_tn5.obs_names) & set(adata_tnH.obs_names) & set(adata_tn5.obs_names)
cells = sorted(list(cells))
print(f"Number of common cells across all 4 datasets: {len(cells)}")
if len(cells) == 0:
    raise RuntimeError("No overlapping cells found.")

for name in ['cdata_tnH','cdata_tn5','adata_tnH','adata_tn5']:
    locals()[name] = locals()[name][cells].copy()

# ---------------------------
# Compute UMAP20
# ---------------------------
def compute_umap20(ds):
    sc.pp.neighbors(ds, n_pcs=30)
    sc.tl.umap(ds, n_components=20)
    ds.obsm['X_umap20'] = ds.obsm['X_umap']

for ds in [cdata_tnH, cdata_tn5, adata_tnH, adata_tn5]:
    compute_umap20(ds)

# ---------------------------
# Fusion: simple average of UMAP embeddings
# ---------------------------
X_fusion = np.mean(
    np.stack([
        cdata_tnH.obsm['X_umap20'],
        cdata_tn5.obsm['X_umap20'],
        adata_tnH.obsm['X_umap20'],
        adata_tn5.obsm['X_umap20']
    ], axis=0),
    axis=0
)
for ds in [cdata_tnH, cdata_tn5, adata_tnH, adata_tn5]:
    ds.obsm['X_fusion'] = X_fusion.copy()

fdata = adata_tn5.copy()
fdata.obsm['X_fusion'] = X_fusion.copy()
print("Fusion done, shape:", fdata.obsm['X_fusion'].shape)

# ---------------------------
# BBKNN on fusion
# ---------------------------
bbknn_neighbors = int(np.sqrt(fdata.shape[0]) / 2 / max(1, samples_count))
try:
    sc.external.pp.bbknn(fdata, use_rep='X_fusion', batch_key='batch',
                         neighbors_within_batch=bbknn_neighbors, n_pcs=20)
    fdata.uns['neighbors']['params']['metric'] = 'cosine'
except Exception as e:
    print("WARNING: BBKNN on fusion failed:", e)

# ---------------------------
# PAGA & UMAP
# ---------------------------
group_key = 'batch' if 'batch' in fdata.obs.columns else fdata.obs.columns[0]
try:
    sc.tl.paga(fdata, groups=group_key)
    sc.pl.paga(fdata, show=False)
except Exception as e:
    print("WARNING: PAGA failed:", e)

try:
    sc.tl.umap(fdata, init_pos='paga')
except Exception:
    sc.tl.umap(fdata)

# ---------------------------
# Save fused data
# ---------------------------
fused_file = out_dir / "Fused_data.h5ad"
fdata.write(fused_file)
print(f"{fused_file} saved.")

# ---------------------------
# Prepare for chromatin velocity (from step2)
# ---------------------------
import anndata
import scvelo as scv

print("Loading processed objects...")
fused_file = out_dir / "Fused_data.h5ad"
processed_tn5 = out_dir / "adata_tn5_processed.h5ad"
processed_tnH = out_dir / "adata_tnH_processed.h5ad"

fdata = sc.read(fused_file)
adata_tn5 = sc.read(processed_tn5)
adata_tnH = sc.read(processed_tnH)

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
ready_file = out_dir / "ChromatinVelocity_ready.h5ad"
adata.write(ready_file)
print(f"{ready_file} saved. Done.")
