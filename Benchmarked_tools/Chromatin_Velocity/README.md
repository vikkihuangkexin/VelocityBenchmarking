# Chromatin_Velocity Script

This script prepares chromatin velocity data by fusing multiple datasets and creating a velocity-ready AnnData object.

## Installation

```bash
pip install numpy
pip install scanpy
pip install anndata
pip install scvelo
pip install bbknn
```

## Usage

```bash
python prepare_chromatin_velocity.py \
  --tn5 path/to/DHS_tn5_CH.h5ad \
  --tnH path/to/DHS_tnH_CH.h5ad \
  --ctn5 path/to/complDHS_tn5_CH.h5ad \
  --ctnH path/to/complDHS_tnH_CH.h5ad \
  --out-dir path/to/output_dir
```

## Input Requirements

- **Input h5ad files**: DHS_tn5_CH.h5ad, DHS_tnH_CH.h5ad, complDHS_tn5_CH.h5ad, complDHS_tnH_CH.h5ad
  - X: Expression matrix
  - obs: Metadata including 'batch' (optional)
  - var: Feature metadata

## Output

- `Fused_data.h5ad`: fused dataset with embeddings and graphs.
- `ChromatinVelocity_ready.h5ad`: velocity-ready AnnData with layers `spliced` and `unspliced`, transferred embeddings, and computed moments.