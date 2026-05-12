# InterVelo Velocity Analysis

## Installation

Install InterVelo from the public compatibility fork used for benchmark integration.

```bash
pip install git+https://github.com/sd68515/InterVelo_py311.git
pip install "numpy==1.26.4" "numba==0.59.1"
```

Install a GPU-enabled PyTorch build according to your own hardware and CUDA
setup. The example workflow was validated with `torch 2.5.1 + CUDA 12.1`.

## Usage

### Real Data

```bash
python InterVelo.py \
    --input data.h5ad \
    --output-dir ./output \
    --cluster-key celltype
```

### Simulated Data

```bash
python InterVelo.py \
    --input simulated.h5ad \
    --output-dir ./output \
    --cluster-key milestone \
    --dimred-key X_dimred \
    --zero-threshold
```

### Multi-omic or Auxiliary-layer Data

InterVelo uses the same `train(adata, inputdata, configs)` entry point for
RNA-only and multi-omic runs. The difference is the input matrix. RNA-only
runs use `[Ms, Mu]`; runs with additional omic information use
`[Ms, Mu, O]`.

Use `--extra-layers` to append existing `adata.layers` matrices after `Ms`
and `Mu`. For example, `Mc` is commonly used for a chromatin accessibility
gene activity layer, and `Ma` appears in the InterVelo simulated multi-omic
example:

```bash
python InterVelo.py \
    --input multiome.h5ad \
    --output-dir ./output \
    --cluster-key celltype \
    --extra-layers Mc
```

Multiple layers can be supplied as a comma-separated list:

```bash
python InterVelo.py \
    --input multiome.h5ad \
    --output-dir ./output \
    --cluster-key celltype \
    --extra-layers Mc,Ma
```

### Batch Mode

```bash
python InterVelo.py \
    --metadata-file datasets.csv \
    --output-dir ./output
```

### Python Function Call

```python
import sys
from pathlib import Path

script_dir = Path("/path/to/VelocityBenchmarking/velocity_generate/InterVelo")
sys.path.insert(0, str(script_dir))

import InterVelo

InterVelo.run_intervelo_analysis(
    input_path="data.h5ad",
    output_dir="./output",
    cluster_key="celltype",
)

InterVelo.run_batch_intervelo(
    metadata_file="datasets.csv",
    output_dir="./output",
)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | - | Input H5AD file for single-file mode |
| `--metadata-file` | - | Metadata CSV/TSV file for batch processing |
| `--output-dir` | Required | Root output directory |
| `--dataset-name` | Input file stem | Dataset folder name in single-file mode |
| `--cluster-key` | Required in single-file mode | Column name in `adata.obs` used for labels and plots |
| `--dimred-key` | `X_umap` | Embedding key in `adata.obsm` |
| `--zero-threshold` | `False` | Set `min_shared_counts=0` and `min_shared_cells=0` during preprocessing |
| `--extra-layers` | Empty | Comma-separated `adata.layers` keys for additional cell-aligned omic or reference matrices appended after `Ms` and `Mu` |
| `--save-pdf` | `False` | Also save PDF figures in addition to PNG |
| `--overwrite` | `False` | Overwrite existing outputs |
| `--seed` | `2024` | Random seed |

## Metadata File Format

Required columns:

- `dataset_name`
- `file_path`
- `cluster_key`

Optional columns:

- `dimred_key` → defaults to `X_umap`
- `zero_threshold` → defaults to `False`
- `extra_layers` → defaults to empty; use comma-separated layer names such as `Mc` or `Ma`

### Example CSV

```csv
dataset_name,file_path,cluster_key,dimred_key,zero_threshold,extra_layers
1,/data/real/pancreas.h5ad,celltype,X_umap,False,
2,/data/real/neuron.h5ad,celltype,X_umap,False,
bifurcation_sim,/data/sim/bifurcation_dataset.h5ad,milestone,X_dimred,True,
cycle_sim,/data/sim/cycle_dataset.h5ad,milestone,X_dimred,True,
multiome_cortex,/data/multiome/cortex.h5ad,celltype,X_umap,False,Mc
```

## Expected Input

### Required H5AD Content

- `adata.layers["spliced"]`
- `adata.layers["unspliced"]`
- `adata.obs[cluster_key]`

The wrapper computes `adata.layers["Ms"]` and `adata.layers["Mu"]` during
preprocessing and uses them as the required InterVelo RNA input.

### Optional Auxiliary Layers

When `--extra-layers` is set, each listed key must already exist in
`adata.layers` before running the script. These layers are scaled together
with `Ms` and `Mu`, then concatenated into the model input:

```text
inputdata = [Ms, Mu, extra_layer_1, extra_layer_2, ...]
```

This follows the InterVelo paper's formulation where the input can be
`(S, U)` for RNA-only data or `(S, U, O)` when an additional omic matrix is
available. `Mc` is a common chromatin accessibility gene activity example,
but the wrapper does not require this exact name. The main velocity output
is still RNA velocity in `adata.layers["velocity"]`.

### Real Data

- Recommended embedding key: `adata.obsm["X_umap"]`
- If `X_umap` is missing, the script computes UMAP and stores it under the requested `dimred_key`

### Simulated Data

- Recommended cluster key: `milestone`
- Recommended embedding key: `adata.obsm["X_dimred"]`
- Use `--zero-threshold` when you want to disable the default real-data filter (`min_shared_counts=20`)

## Output

For an input dataset named `test.h5ad` with `dataset_name=id_test`, the output structure is:

```text
output/
└── id_test/
    ├── test_plot.h5ad
    ├── saved/
    └── plot/
        ├── test_umap.png
        ├── test_stream.png
        ├── test_grid.png
        └── test_pseudotime.png
```

If `--save-pdf` is enabled, matching `.pdf` files are written to the same `plot/` directory.

## Output Data Keys

The output H5AD file contains the standard InterVelo results together with plotting-ready keys:

**`.layers`**

- `velocity`
- `velocity_unspliced` (if generated by the model)
- `pred_alpha` (only when the model is configured to predict unspliced rates)

**`.obs`**

- `pseudotime` - the primary pseudotime estimated directly by InterVelo
- `velocity_pseudotime` - a scVelo-derived pseudotime computed from the predicted velocity field and used by InterVelo as a direction-consistency check during training
- `intervelo_pseudotime_normalized`

**`.obsm`**

- `X_TNODE`
- `X_VF`
- `velocity_umap` or `velocity_dimred` (depending on `dimred_key`)

**`.var`**

- `pred_*` kinetic parameters returned by InterVelo when available, such as `pred_beta` and `pred_gamma`

**`.uns`**

- `intervelo_run` with input and run metadata

## Notes

- The public example script does not generate `ground_truth_velocity_graph` or `ground_truth_velocity_dimred`.
- The batch metadata file is intended to be a public-facing CSV/TSV manifest; dataset IDs do not need any special filename parsing logic.
- The `extra_layers` option is explicit by design. The wrapper does not automatically use `Mc` or other layers just because they are present, which keeps benchmark runs reproducible across datasets.
- The script keeps the configuration surface intentionally small and does not expose the local runtime-only InterVelo patches that were used in private server workflows.
