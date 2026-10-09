# MultiVelo Script

MultiVelo estimates chromatin-informed RNA velocity from paired RNA and chromatin accessibility data.

## Installation

```bash
pip install multivelo
```

MultiVelo does not declare PyTorch, but it imports `torch` at module level, so a compatible CPU wheel has to be installed separately:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

For detailed information, see the [MultiVelo GitHub repository](https://github.com/welch-lab/MultiVelo/) and the [documentation](https://multivelo.readthedocs.io/en/latest/).

## Input Requirements

The RNA H5AD file must contain:

- `adata.layers["spliced"]`
- `adata.layers["unspliced"]`
- `adata.obs[cluster_key]` (real data), or use `--simulate` and the `milestone` label is set automatically
- `adata.obsm["X_umap"]`, or a source embedding that is remapped to `X_umap`
- `adata.obsm["X_dimred"]` for simulated data, copied to `X_umap` when present

The ATAC modality is read from the same RNA object through a gene activity layer, selected with `--atac_layer` (`chromatin` by default, with `Mc` as a fallback). The `--atac_dir` argument is validated when provided, mirroring the two-modality input layout of the original driver.

## Output

For an RNA input file `test.h5ad`, the output is written to:

```text
<save-dir>/
└── <dataset-name>/
    ├── test_MultiVelo.h5ad    # native name, layers['velocity']
    └── rc.h5ad                # alias of the file above
```

The output H5AD contains the exported velocity in `layers["velocity"]` together with the native MultiVelo result keys. Run metadata is stored in `uns["multivelo_run"]`.

## Parameters

- `--rna_dir`: Input RNA h5ad data file path. Default: `./adata_postpro.h5ad`
- `--atac_dir`: Input ATAC h5ad data file path. Default: `./adata_atac_postpro.h5ad`
- `--save_dir`: Result saving directory. Default: `./test`
- `--metadata_file`: Metadata CSV/TSV file for batch processing; when set, single-file arguments are ignored.
- `--dataset_name`: Dataset folder name in single-file mode. Default: RNA input file stem.
- `--cluster-key`: Column name in `adata.obs` used as the color key; use `milestone` for simulated data.
- `--atac_layer`: `adata.layers` key holding the ATAC gene activity matrix; falls back to `Mc` when unset. Default: `chromatin`
- `--max_iter`: Maximum iterations for `recover_dynamics_chrom`. Default: `5`
- `--n_jobs`: Number of jobs for parallel processing in `recover_dynamics_chrom`. Default: `1`
- `--n_anchors`: Number of anchors for `recover_dynamics_chrom`. Default: `500`
- `--init_mode`: Initialization mode passed to `recover_dynamics_chrom`. Default: `invert`
- `--simulate`: Simulated-data branch, sets the `milestone` label, copies `X_dimred` to `X_umap`, and relaxes the filtering. Default: `False`
- `--min_shared_counts`: Minimum shared counts used by `filter_and_normalize` in the real-data branch. Default: `10`
- `--n_top_genes`: Number of highly variable genes retained; when unset (default) the `filter_and_normalize` kwarg is omitted. Default: unset
- `--overwrite`: Overwrite existing outputs. Default: `False`
- `--seed`: Random seed. Default: `2024`

## Usage

```bash
python MultiVelo.py --rna_dir rna.h5ad --atac_dir atac.h5ad --save_dir results --cluster-key celltype
python MultiVelo.py --rna_dir sim.h5ad --atac_dir sim_atac.h5ad --save_dir results --cluster-key milestone --simulate
python MultiVelo.py --metadata_file datasets.csv --save_dir results
```
