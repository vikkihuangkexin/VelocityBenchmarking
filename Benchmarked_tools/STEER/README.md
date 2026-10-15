# STEER Script

STEER infers spatial RNA velocity with a graph neural network that is guided by a
spatial graph and an expression-expert prior. It is a spatial method: the model
needs the expression layers, the spatial coordinates and a cell-type column.

## Installation

```bash
git clone https://github.com/lzygenomics/STEER.git /opt/STEER
cd /opt/STEER
pip install -r requirements.txt
pip install --no-deps .
```

STEER additionally requires a GPU-enabled PyTorch (CUDA 12.4), PyTorch Geometric
with `torch-scatter` / `torch-sparse`, and R with `mclust` for the clustering-based
priors. All of these are installed by the repository Dockerfile.

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install torch-geometric
mamba install -c conda-forge r-base=4.3.3 r-mclust=6.1.1
```

## Input Requirements

The input H5AD file must contain:

- `adata.layers["spliced"]` and `adata.layers["unspliced"]` (or the layers named by
  `--spliced-layer` / `--unspliced-layer`)
- `adata.obsm["spatial"]` — the 2D spatial coordinates (or the key named by
  `--spatial-key`)
- `adata.obs[<cluster-key>]` — **required**, the cell-type labels used by the
  expert prior

The wrapper runs the two-stage upstream workflow: `filter_and_normalize`
(`min_shared_counts=20`, `n_top_genes=2000`), joint-moments smoothing, an
expression graph merged with the spatial graph (`--graph union` or `overlap`),
stage-1 training, prior inference (fine clusters, gene filter, convexity), and
stage-2 fine-tuning on the selected velocity genes. There is no simulated-data
branch because both real and simulated spatial datasets share the same layout.

## Output

For an input `test.h5ad` with `dataset_name=id_test` the result is written to:

```text
output/
└── id_test/
    ├── test.h5ad     # result with layers['velocity']
    └── (model and prior artefacts written by STEER into the same folder)
```

The naming is exactly the one of the original driver: its `run_all` loop computes
`name = path.stem`, `folder = root / name` and `out = folder / f"{name}.h5ad"`, and the
wrapper reproduces those statements, so the result file keeps the input stem verbatim
(no suffix rewriting). Because the native file name is already the benchmark name, no
extra alias file is produced.

The output H5AD contains:

- `.layers["velocity"]` — the inferred velocity, aliased from the native
  `pred_vs_norm` layer (the predicted spliced velocity that STEER itself passes as
  the `vkey` of `steer.velocity_graph`)
- `.layers["pred_vs_norm"]` — the native STEER velocity layer, preserved
- `.layers["Ms"]`, `.layers["Mu"]`, `.layers["model_Ms"]` — model inputs
- `.obsm["X_refine_embed_t"]` — refined stage-2 embedding
- `.uns["steer_run"]` — run metadata (input path, parameters, native velocity key)

STEER itself only builds the velocity graph (`uns['pred_vs_norm_graph']`) and does not
write `layers['velocity']`; the wrapper performs that single alias so the benchmark can
read the result, and raises if `pred_vs_norm` is absent instead of writing an unusable
file.

## Parameters

- `--input`: input H5AD file (single-file mode). Default: none
- `--metadata-file`: metadata CSV/TSV file for batch processing. Default: none
- `--output-dir`: root output directory. Required
- `--dataset-name`: dataset folder name in single-file mode. Default: input file stem
- `--cluster-key`: column in `adata.obs` used for the cell-type labels. Default: `celltype`
- `--spatial-key`: spatial coordinates key in `adata.obsm`. Default: `spatial`
- `--spliced-layer`: spliced counts layer. Default: `spliced`
- `--unspliced-layer`: unspliced counts layer. Default: `unspliced`
- `--device`: torch device, for example `cuda:0` or `cpu`. Default: `cuda:0`
- `--expert`: maximum number of expert clusters. Default: 10
- `--smooth-neigh`: neighbours used when smoothing the counts. Default: 50
- `--spatial-neighbors`: neighbours used for the spatial graph. Default: 8
- `--n-top-genes`: number of highly variable genes retained. Default: 2000
- `--training-mode`: training preset, `full` or `fast`. Default: `full`
- `--finetune-epochs`: stage-2 fine-tuning epochs. Default: 2000
- `--graph`: how the expression graph is combined with the spatial graph, `union` or `overlap`. Default: `union`
- `--corr-mode`: correlation direction used by the expert model, `u` or `s`. Default: `u`
- `--neighbor-metric`: distance metric for the neighbour graphs. Default: `cosine`
- `--use-us` / `--no-use-us`: smooth and build the graph from the joint moments (`Ms`+`Mu`) or from `Ms` only. Default: joint moments
- `--use-filter` / `--no-use-filter`: enable or disable the prior gene filter. Default: enabled
- `--fine-method`: fine-cluster method used by the prior inference manager. Default: `none`
- `--target-size`: target fine-cluster size. Default: 300
- `--direction-base`: basis used for the convexity direction. Default: `expert`
- `--overwrite`: overwrite existing outputs. Default: False
- `--seed`: random seed. Default: 618

## Usage

```bash
python STEER.py --input spatial.h5ad --output-dir ./output --cluster-key celltype
python STEER.py --input spatial.h5ad --output-dir ./output --cluster-key celltype \
    --graph overlap --training-mode fast --finetune-epochs 1000
python STEER.py --metadata-file datasets.csv --output-dir ./output
```

## Metadata File Format

Required columns:

- `dataset_name`
- `file_path`
- `cluster_key`

Optional columns:

- `spatial_key` → defaults to `spatial`

```csv
dataset_name,file_path,cluster_key,spatial_key
cortex,/data/spatial/cortex.h5ad,celltype,spatial
```

## Notes

- STEER needs a cell-type column: `--cluster-key` is validated in both single-file
  and batch mode.
- The spatial graph is combined with the expression graph before training, so
  `obsm["spatial"]` is required.
- `--training-mode fast` selects the reduced-epoch preset (`expert_mode="slim"`,
  500 pretrain / 200 cluster epochs) that the upstream driver defines.
- STEER does not plot in this workflow, so the wrapper carries no plotting code or
  figure palette.
- Verification scope: this wrapper was checked statically (AST parse, CLI construction,
  parameter-by-parameter comparison with `velocity_spatial_code/steer_run.py`). No image
  was built and no dataset was run here, so the two-stage training was not re-confirmed
  numerically.
- The numba cache directory is configurable through `NUMBA_CACHE_DIR` instead of a
  fixed `/tmp` path.
