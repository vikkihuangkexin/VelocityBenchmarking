# MultiVelo Script

## Installation

```bash
pip install multivelo
```

For detailed information, see [MultiVelo GitHub](https://github.com/welch-lab/MultiVelo/) and [documentation](https://multivelo.readthedocs.io/en/latest/).

## Parameters

- `--rna_dir`: Input RNA h5ad data file path. Default: ./adata_postpro.h5ad
- `--atac_dir`: Input ATAC h5ad data file path. Default: ./adata_atac_postpro.h5ad
- `--save_dir`: Result saving directory. Default: ./test
- `--max_iter`: Maximum iterations for recover_dynamics_chrom. Default: 5
- `--n_jobs`: Number of jobs for parallel processing in recover_dynamics_chrom. Default: 15
- `--n_anchors`: Number of anchors for recover_dynamics_chrom. Default: 500
- `--simulate`: Whether the data is simulation data. If true, sets color key to 'milestone' and sets X_umap from X_dimred if available.

## Usage

```bash
python multivelo.py --rna_dir rna.h5ad --atac_dir atac.h5ad --save_dir results --max_iter 10 --n_jobs 8 --n_anchors 300 --simulate
```
