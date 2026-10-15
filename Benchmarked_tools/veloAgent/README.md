# veloAgent Script

veloAgent infers spatial RNA velocity with an agent-based model that is steered by a
cell-state VAE and a protein-interaction-constrained gene network. It is a spatial
method: the model needs the expression layers, the spatial coordinates and the
upstream protein interaction tables.

## Installation

```bash
git clone https://github.com/mcgilldinglab/veloAgent.git /opt/veloAgent
cd /opt/veloAgent
pip install --no-deps .
```

veloAgent pins a Python 3.8 / PyTorch 2.2.2 (CUDA 12.1) environment and depends on
the upstream VeloAE package:

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install 'git+https://github.com/qiaochen/VeloAE.git'
pip install 'numpy==1.24.4' 'pandas==2.0.3' 'anndata==0.9.2' 'scipy==1.10.1' \
    'scikit-learn==1.1.3' 'seaborn==0.13.2' 'statsmodels==0.14.1' 'matplotlib==3.7.3' \
    'tqdm==4.66.5' 'scanpy==1.9.8' 'scvelo==0.3.2' 'leidenalg==0.10.2' \
    'mesa==2.1.5' 'mplscience==0.0.7' 'statannotations==0.7.2'
```

## Input Requirements

The input H5AD file must contain:

- `adata.layers["spliced"]` and `adata.layers["unspliced"]` (or the layers named by
  `--spliced-layer` / `--unspliced-layer`)
- `adata.obsm["spatial"]` — the 2D spatial coordinates (or the key named by
  `--spatial-key`)

`adata.obs[<cluster-key>]` is optional for veloAgent and only validated when passed.

The wrapper then runs the upstream workflow: `veloagent.preprocess`, cell-state VAE
training, cell embedding extraction, the protein-constrained `GeneNet` trained
first gene-gene then neighbour-wise, and finally the agent-based `CellModel` step.
There is no simulated-data branch because both real and simulated spatial datasets
share the same input layout.

## Output

For an input `test.h5ad` with `dataset_name=id_test` the result is written to:

```text
output/
└── id_test/
    └── test.h5ad    # result with layers['velocity']
```

The naming is exactly the one of the original driver: its `run_all` loop computes
`name = path.stem`, `folder = root / name` and `out = folder / f"{name}.h5ad"`, and the
wrapper reproduces those statements, so the result file keeps the input stem verbatim
(no suffix rewriting). Because the native file name is already the benchmark name, no
extra alias file is produced.

The output H5AD contains:

- `.layers["velocity"]` — the inferred velocity, written directly by the agent-based
  step (`veloagent.CellModel.step`), so no alias is needed
- `.layers["velocity_u"]` — the matching unspliced velocity
- `.obsm["cell_embed"]` — the VAE cell embedding used by the gene network
- `.obsm["spatial"]`, `.layers["Ms"]`, `.layers["Mu"]` — model inputs
- `.uns["veloagent_run"]` — run metadata (input path, parameters, native velocity key)

The wrapper raises if `layers['velocity']` is missing instead of writing an unusable
file.

`veloagent.CellModel` reads `layers['velocity']` from every cell when it is
constructed, so the gene-network stages must run before the agent-based step. The
wrapper keeps that order (`train_gg` → `train_nbr` → `CellModel.step`).

## Parameters

- `--input`: input H5AD file (single-file mode). Default: none
- `--metadata-file`: metadata CSV/TSV file for batch processing. Default: none
- `--output-dir`: root output directory. Required
- `--dataset-name`: dataset folder name in single-file mode. Default: input file stem
- `--spatial-key`: spatial coordinates key in `adata.obsm`. Default: `spatial`
- `--cluster-key`: column in `adata.obs` used for cell labels. Default: none
- `--spliced-layer`: spliced counts layer. Default: `spliced`
- `--unspliced-layer`: unspliced counts layer. Default: `unspliced`
- `--protein-data-dir`: directory containing the veloAgent protein interaction files. Default: `$VELOAGENT_PROTEIN_DIR` or `/opt/veloAgent`
- `--species`: species used to pick the protein interaction tables. Default: `human`
- `--device`: torch device, for example `cuda:0` or `cpu`. Default: `cuda:0`
- `--z-dim`: latent dimension of the VAE. Default: 5
- `--vae-lr`: VAE learning rate. Default: 1e-2
- `--genenet-lr`: GeneNet learning rate. Default: 1e-4
- `--batch-size`: batch size. Default: 256
- `--patience`: early-stopping patience. Default: 25
- `--vae-epochs`: VAE training epochs. Default: 1000
- `--gg-epochs`: gene-gene training epochs. Default: 500
- `--nbr-epochs`: neighbour training epochs. Default: 50
- `--abm-steps`: agent-based model steps. Default: 100
- `--overwrite`: overwrite existing outputs. Default: False
- `--seed`: random seed. Default: 144

## Usage

```bash
python veloAgent.py --input spatial.h5ad --output-dir ./output --spatial-key spatial
python veloAgent.py --input spatial.h5ad --output-dir ./output --species mouse --z-dim 10
python veloAgent.py --metadata-file datasets.csv --output-dir ./output
```

## Metadata File Format

Required columns:

- `dataset_name`
- `file_path`

Optional columns:

- `cluster_key` → defaults to empty (veloAgent does not require labels)
- `spatial_key` → defaults to `spatial`

```csv
dataset_name,file_path,cluster_key,spatial_key
cortex,/data/spatial/cortex.h5ad,,spatial
```

## Notes

- The protein interaction tables come from the upstream checkout; `--protein-data-dir`
  replaces the hard-coded path of the original script and defaults to the cloned
  repository inside the container.
- The agent-based model writes the native `layers['velocity']` layer itself
  (together with `velocity_u`), and the wrapper verifies it before writing the result.
- veloAgent does not plot in this workflow, so the wrapper carries no plotting code
  or figure palette.
- Verification scope: this wrapper was checked statically (AST parse, CLI construction,
  parameter-by-parameter comparison with `velocity_spatial_code/veloagent_run.py`). No
  image was built and no dataset was run here, so the VAE / gene-network / agent steps
  were not re-confirmed numerically.
