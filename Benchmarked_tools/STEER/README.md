# STEER

This standalone entry follows the two-stage spatial workflow used in the
benchmark. Install the [team STEER source fork](https://github.com/sd68515/STEER_benchmark) and its
dependencies in a suitable environment. Expert estimation and the default
`mclust` refinement require R, `mclust`, and `rpy2`; follow the upstream
installation guide for the PyTorch and PyG packages matching your hardware.

```bash
git clone https://github.com/sd68515/STEER_benchmark.git ../STEER_benchmark
python -m pip install --no-deps -e ../STEER_benchmark
```

## Inputs and run

Commands below run from the repository root. The input H5AD needs `spliced` and `unspliced` layers, a cell annotation in
`obs`, and finite two-dimensional spatial coordinates in `obsm`. The
coordinate key defaults to `spatial`; use `--coordinate-key` or CSV
`dimred_key` for another key.

The script provides `obsm['X_spatial']` for scVelo's `basis='spatial'` calls,
copying `obsm['spatial']` when the alias is missing. It checks the plotting
object again before projecting velocity or drawing spatial plots.

Declare the annotation key explicitly:

```bash
python Benchmarked_tools/STEER/STEER.py \
  --input data/tissue.h5ad --dataset-id tissue \
  --annotation-key cell_type \
  --output-dir results/STEER
```

For several inputs, use a CSV. Required columns are
`dataset_name,file_path,modality,cluster_key`; `modality` must be `spatial`.
Optional `dimred_key` defaults to `spatial`. Relative file paths resolve from the CSV
directory; each result is `<output-dir>/<dataset_id>.h5ad`.

```csv
dataset_name,file_path,modality,cluster_key,dimred_key
tissue,data/tissue.h5ad,spatial,cell_type,
```

```bash
python Benchmarked_tools/STEER/STEER.py \
  --metadata-file datasets.csv --output-dir results/STEER
```

## Preprocessing and training

The team fork accepts both sparse and dense count layers. This entry converts
them to sparse `float64` matrices to retain the benchmark's input representation.
It makes duplicate gene names unique, applies
scVelo filtering with `min_shared_counts=20` and 2,000 top genes, and combines
expression and spatial neighbors by union. The initial expert count is
estimated from candidates 2-10. The first training stage is followed by
`mclust` refinement, prior construction, selection of up to 1,000 velocity
genes, and the final training stage. The result receives a STEER velocity
graph based on `pred_vs_norm` and `model_Ms`.

The benchmark used seed 618, which STEER also sets internally during training,
and the method's standard training settings.
Choose hardware and worker counts with `--device cpu|cuda|cuda:<index>` and
`--threads`; the defaults are CPU and up to 16 threads. `--skip-mclust-refine`
changes the benchmark workflow and should be used only for a deliberate
alternative analysis. `--overwrite` permits replacing an existing H5AD.
`--plot-dir` optionally saves annotation, spatial velocity stream, and
predicted-time PNGs; these are simple views of the saved result, not the
manuscript figure-generation pipeline.

## Outputs

The output retains `layers['pred_vs_norm']`, `layers['model_Ms']`,
`obs['Pred Time']`, spatial coordinates, the velocity graph, and run settings
in `uns['steer_run']`. Expert-estimation and training artifacts are written to
`<output-dir>/<dataset_id>/training/`.
