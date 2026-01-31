# TFvelo Script

This script performs velocity analysis using TFvelo on single-cell RNA sequencing data.

## Installation

```bash
pip install pandas==1.2.3 
pip install anndata==0.8.0 
pip install scanpy==1.8.2
pip install numpy==1.21.6
pip install scipy==1.10.1 
pip install numba==0.57.0 
pip install matplotlib==3.3.4
pip install scvelo==0.2.4
pip install typing_extensions
```

## Input Requirements

The input h5ad file must contain 'spliced' and 'unspliced' layers.

## Output

Running the script produces pp.h5ad and rc.h5ad files. To obtain the velocity layer in rc.h5ad, run the following code:

```python
import numpy as np
import anndata as ad

adata = ad.read_h5ad('rc.h5ad')
n_cells = adata.shape[0]
expanded_scaling_y = np.expand_dims(np.array(adata.var['fit_scaling_y']), 0).repeat(n_cells, axis=0)
adata.layers['velocity'] = adata.layers['velo_hat'] / expanded_scaling_y
```

## Parameters

- `--dataset_name`: Dataset name (e.g., pancreas, gastrulation_erythroid, 10x_mouse_brain). Default: pancreas
- `--n_jobs`: Number of CPUs to use. Default: 28
- `--var_names`: Variable names (all or highly_variable_genes). Default: all
- `--init_weight_method`: Initialization method for weights. Default: correlation
- `--WX_method`: Weight method. Default: lsq_linear
- `--n_neighbors`: Number of neighbors. Default: 30
- `--WX_thres`: Weight threshold. Default: 20
- `--n_top_genes`: Number of top genes. Default: 2000
- `--TF_databases`: TF databases. Default: ENCODE ChEA
- `--max_n_TF`: Max number of TFs. Default: 99
- `--max_iter`: Max iterations. Default: 20
- `--n_time_points`: Number of time points. Default: 1000
- `--save_name`: Save name suffix. Default: _demo
- `--use_raw`: Use raw data. Default: 0
- `--basis`: Basis for plotting. Default: umap
- `--simulate`: Whether the data is simulation data. If true, adjusts preprocessing and sets clusters to 'milestone', and sets X_umap from X_dimred if available.

## Usage

```bash
python TFvelo.py --dataset_name pancreas --simulate
```

For more details, see https://github.com/xiaoyeye/TFvelo