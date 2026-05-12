# scKINETICS Script

This directory provides the `scKINETICS.py` script used to run `scKINETICS` for velocity benchmarking with the maintained local fork and the fixed team plotting outputs.

## Installation

Use the maintained fork and environment instructions from:

- `https://github.com/sd68515/scKINETICS_py311_r441.git`

This benchmark script is intended to use that local fork only. A recommended setup is:

```bash
git clone https://github.com/sd68515/scKINETICS_py311_r441.git
cd scKINETICS_py311_r441
# create / activate the environment according to that repository
pip install -e . --no-deps
```

After that, run the benchmark script from this repository.

## Usage

Single-dataset mode:

```bash
python scKINETICS.py \
  --input-h5ad your_data.h5ad \
  --peaks-bed your_peaks.bed \
  --output-dir ./output \
  --cluster-key celltype \
  --genome mm10 \
  --embedding-basis X_umap
```

Batch mode:

```bash
python scKINETICS.py \
  --metadata-file datasets.csv \
  --output-dir ./output \
  --fig-dir ./figures
```

Example metadata columns:

- `dataset_name`
- `file_path`
- `peaks_bed`
- optional: `cluster_key`, `embedding_basis`, `genome`

Example `datasets.csv`:

```csv
dataset_name,file_path,peaks_bed,cluster_key,embedding_basis,genome
MouseBrain,/path/to/mouse_brain.h5ad,/path/to/mouse_brain_peaks.bed,celltype,X_umap,mm10
Pancreas,/path/to/pancreas.h5ad,/path/to/pancreas_peaks.bed,celltype,X_umap,mm10
HumanSample,/path/to/human_sample.h5ad,/path/to/human_sample_peaks.bed,celltype,X_umap,hg38
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-h5ad` | optional | Input AnnData file in single-dataset mode |
| `--metadata-file` | optional | Metadata table for batch mode |
| `--peaks-bed` | required in single mode | Peaks BED file |
| `--output-dir` | required | Root output directory |
| `--dataset-name` | input stem | Dataset name in single-dataset mode |
| `--cluster-key` | `celltype` | Grouping column for scKINETICS |
| `--embedding-basis` | `X_umap` | Basis used for final velocity embedding and plotting |
| `--genome` | `mm10` | Genome assembly |
| `--peak-width-max` | `2000` | Maximum peak width kept |
| `--min-genes` | `200` | Basic cell filter threshold |
| `--min-cells` | `3` | Basic gene filter threshold |
| `--target-sum` | `1e4` | Normalization target sum |
| `--skip-normalize` | off | Skip normalize_total |
| `--skip-log1p` | off | Skip log1p |
| `--pca-n-comps` | `50` | PCA dimensions if `X_pca` missing |
| `--motif-pvalue` | `1e-10` | Motif calling p-value |
| `--threads` | `1` | EM thread count; default kept conservative for shared servers |
| `--maxiter` | `20` | EM max iterations |
| `--tol` | `0.005` | EM tolerance |
| `--model-knn` | `50` | kNN used inside EM |
| `--graph-knn` | `30` | kNN used for VelocityGraph |
| `--sigma` | `5.0` | EM sigma |
| `--sigma-prior` | `1.0` | EM sigma prior |
| `--fig-dir` | `None` | Optional root directory for the fixed benchmark figures |
| `--n-jobs` | `1` | Jobs for scVelo graph-related plotting calculations |

## Input requirements

### AnnData

Required:

- `adata.X`: expression matrix
- `adata.obs[cluster_key]`: fitting groups, default `obs['celltype']`

Recommended:

- `adata.obsm['X_umap']`: unified benchmark embedding used for final low-dimensional projection
- `adata.obsm['X_pca']`: if absent, the script will compute PCA automatically
- existing layers such as `spliced`, `unspliced`, `Ms`, `Mu` can be retained and will be subset together with the modeled genes

The script performs:

- `obs_names_make_unique()`
- `var_names_make_unique()`
- basic filtering:
  - `filter_cells(min_genes=200)`
  - `filter_genes(min_cells=3)`
- `normalize_total(target_sum=1e4)` unless `--skip-normalize`
- `log1p()` unless `--skip-log1p`

### BED file

The peaks file may be either:

- a BED with header columns `chrom`, `chromStart`, `chromEnd`
- or a plain 3-column BED without header

Only the first three BED columns are used. Peak width is filtered by `--peak-width-max` (default `2000`).

## Output

The final exported `h5ad` contains at least:

- `layers['velocity']`
- `obsm['velocity_umap']` if `--embedding-basis X_umap`
- `obsp['sckinetics_T']`
- `obsp['sckinetics_T_backward']`
- `obsp['sckinetics_knn_graph']`
- `uns['sckinetics_params']`
- `uns['sckinetics_velocity_genes']`
- `uns['sckinetics_velocity_genes_upper']`

The final output restores the original gene symbols as `var_names`, while preserving the internal uppercase gene list used by scKINETICS matching logic in `uns['sckinetics_velocity_genes_upper']`.

Auxiliary files saved alongside the exported `h5ad`:

- `sckinetics_model.pickle`
- `sckinetics_runtime_adata.pickle`
- `velocity_<basis>.npy`

If `--fig-dir` is provided, the script additionally writes the fixed benchmark figures in both `png/` and `pdf/` subdirectories:

- scatter
- stream
- grid
- pseudotime

## Notes

- The official `sckinetics` PyPI package only contains the initial release and is not sufficient for the maintained workflow here.
- `scKINETICS` internally uppercases gene symbols for TF/target matching; this script restores the original gene symbols in the final export.
- The script does not run TF ablation by default because benchmark generation primarily needs velocity inference rather than downstream regulator perturbation analysis.
- If your benchmark datasets already contain unified `X_umap`, it is recommended to keep using that as the exported visualization basis.
- This workflow targets real datasets with an accompanying ATAC differential-accessibility peak BED file, so it is not applicable to standard simulated RNA velocity datasets that do not provide such peak inputs.
- The core scKINETICS code does not expose a user-facing seed parameter. For practical reproducibility on shared compute, this script keeps `--threads 1` by default.
