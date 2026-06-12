# Evaluation

## Ground Truth Correlation

This metric compares predicted velocity vectors against ground truth velocity vectors on a shared low-dimensional basis.

Default workflow:

- Predicted velocity is read from `adata.layers[vkey]`.
- Ground truth velocity is read from `adata.layers['ground_truth_velocity']` by default.
- The script uses `scvelo.tl.velocity_graph` and `scvelo.tl.velocity_embedding` to project both into the same low-dimensional basis.
- The final score is the cosine similarity between low-dimensional velocity vectors.

For the public simulation datasets provided by the team, all datasets except `lineage-tracing` are expected to include `adata.layers['ground_truth_velocity']`, so the default path should work out of the box.
Datasets such as `lineage-tracing` that do not provide usable ground truth follow the same default failure path as any other unsupported input: the script raises a clear error and does not hard-code any skip behavior.

## Function

```python
result = calculate_groundtruth_correlation(
    adata_or_path,
    method,
    dataset_id,
    output_csv,
    velocity_key="velocity",
    gt_velocity_key="ground_truth_velocity",
    basis_name="dimred",
    gt_npz_base_dir=None,
    reference_gt_key="gt_dimred",
    reference_basis_key="X_basis",
    raise_on_gt_failure=True,
    min_cell_match_ratio=0.95,
)
```

## Input

The input can be either an AnnData object or an H5AD file path.

By default, the script expects:

- `adata.layers[velocity_key]`: high-dimensional predicted velocity
- `adata.layers[gt_velocity_key]`: high-dimensional ground truth velocity

Optional:

- An external reference directory `gt_npz_base_dir`
  - File naming convention: `*_reference_data.npz`
  - Default low-dimensional GT key: `gt_dimred`
  - Default reference basis key: `X_basis`

A reference is not required by default. It is only needed when:

- the input H5AD file has lost visualization coordinates,
- the low-dimensional coordinates were recomputed,
- or the comparison must be corrected to a fixed reference basis.

If a reference directory is provided explicitly but the requested `reference_gt_key` is missing from a reference npz file, the script prints a warning and falls back to projecting `adata.layers[gt_velocity_key]` if that layer is available.

Important note:

- `gt_dimred` in the reference files is the precomputed low-dimensional projection of `ground_truth_velocity` onto a fixed basis.
- For the public simulation datasets, projecting `adata.layers['ground_truth_velocity']` on the fly should be numerically equivalent to using `gt_dimred` from the reference files.

If the input has neither usable ground truth velocity nor an external reference, the script stops with a clear error.

## Basis Behavior

Users specify the low-dimensional basis with `--basis-name`. The default is `dimred`.

Rules:

- If the user explicitly requests a basis and it does not exist, the script fails immediately.
- If the default `dimred` is used and `X_dimred` is missing, the script prints a warning and falls back in the order `X_umap -> X_tsne`.
- `X_pca` is always ignored as a comparison basis.
- No `X_dimred -> X_umap` mirroring or copy logic is kept.
- If no usable non-PCA basis exists, the script raises an error.

## Output

The function returns a dictionary containing:

- `mean_cosine`
- `n_cells_total`
- `n_cells_valid`
- `cell_match_ratio`
- `basis_name`
- `success`

The output CSV is a wide table containing only:

- `Method`
- one column per dataset

Rows are sorted alphabetically by `Method`. `AVG` and `Reversed_rank` are no longer produced.

## Command Line Usage

### Single file

```bash
python groundtruth_correlation.py \
    --input result.h5ad \
    --method VeloVAE \
    --dataset-id bifurcating_cell1000_gene1000 \
    --output-csv results.csv \
    --velocity-key velocity
```

### Batch processing

```bash
python groundtruth_correlation.py \
    --metadata-csv datasets.csv \
    --output-csv results.csv \
    --basis-name dimred \
    --verbose
```

Batch CSV format:

```csv
method,id,path,vkey
VeloVAE,bifurcating_cell1000_gene1000,/path/to/result.h5ad,velocity
MyMethod,cycle-simple_cell5000_gene500,/path/to/result.h5ad,my_velocity
```

### Optional reference override

```bash
python groundtruth_correlation.py \
    --input result.h5ad \
    --method MyMethod \
    --dataset-id bifurcating_cell1000_gene1000 \
    --output-csv results.csv \
    --gt-npz-dir simdata_reference \
    --reference-gt-key gt_dimred \
    --reference-basis-key X_basis
```

## Notes

- The script does not contain any tool-specific special handling.
- `method` is used only as a display label in the result table.
- Generic fallbacks are retained: cell name normalization, partial matching, position-based fallback after failed name matching, neighbor rebuilding, and embedding recomputation.
- Reference file lookup keeps topology normalization logic, including underscore/hyphen variants.
