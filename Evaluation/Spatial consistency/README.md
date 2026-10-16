# Spatial velocity and time consistency

Compute local velocity coherence and spatial time autocorrelation from result
H5AD files. This entry reads inputs without changing them. Its metric graph
is the symmetric, binary union of each cell's 14 nearest **other** cells in
spatial coordinates, independent of H5AD expression/velocity graphs.
`--n-neighbors` changes that definition; keep 14 for the second-revision metric.

## Inputs

Install NumPy, pandas, SciPy, h5py, and Matplotlib. Supply a CSV with:

| Field | Requirement |
| --- | --- |
| `method,id,path,vkey,data_type` | Required: method, dataset ID, result H5AD, velocity layer, and `real` or `sim` |
| `basis_name` | Optional, default `spatial`: reads `obsm['spatial']`; `foo` reads `X_foo`, and `X_foo` is an exact key |
| `tkey` | Optional, default `<vkey>_pseudotime`: numeric column in `obs` |
| `cluster_key` | Optional annotation column in `obs`, needed only for annotation-matched ESVC |
| `modality`, `run_id` | Optional: modality must be `spatial`; run ID defaults to `single` |

The H5AD needs finite spatial coordinates with at least two columns and three
cells, and a cell-by-gene `layers['<vkey>']` matrix with at least two genes.
Velocity rows with any non-finite value or zero variance are excluded.
Non-finite times are omitted from the time calculation. Missing time or
velocity can leave the other score available (`status=partial`). A constant
time vector gives an undefined time score.

```csv
method,id,path,vkey,data_type,tkey,cluster_key,basis_name
MyMethod,55,data/spatial_real.h5ad,velocity,real,velocity_pseudotime,cell_type,spatial
MyMethod,cellsub-chickenheart_60pct,data/spatial_subset.h5ad,velocity,sim,,cell_type,spatial
```

Relative H5AD paths resolve from the manifest directory. Custom dataset IDs
are accepted; no differentiation-path table is needed.

## Run

From the repository root:

```bash
python "Evaluation/Spatial consistency/spatial_consistency.py" \
  --csv-file inputs.csv --output-dir results/spatial_consistency
```

`--jobs` controls parallel H5AD workers (default 2); `--no-plots` writes CSVs
only. Each worker loads dense velocity matrices, including sparse layers
converted to dense; choose the worker count to fit memory.

`--esvc` adds an exploratory diagnostic. It is **off by default and was not
used in the second-revision paper**.

## Outputs and interpretation

| File | Contents |
| --- | --- |
| `spatial_consistency_results.csv` | One row per input, scores, coverage, status, and errors; checkpointed as inputs finish |
| `spatial_consistency_failures.csv` | Missing, failed, and partial rows |
| `spatial_velocity_consistency_<real_or_sim>.csv` | Method-by-dataset velocity consistency |
| `spatial_time_consistency_<real_or_sim>.csv` | Method-by-dataset time consistency |
| Matching `.svg` and `.png` | Heatmaps, unless `--no-plots` is set |

Wide tables average repeated `run_id` results; the long table preserves each
run. Missing scores appear as `NA`. The process exits nonzero if any input is
missing or failed. Partial rows remain available.

| Score | Definition and coverage |
| --- | --- |
| `velocity_consistency` | Pearson correlation across velocity genes for each spatial neighbor pair, averaged over valid neighbors per cell and then over cells. `velocity_cells` counts cells with a valid comparison. |
| `time_consistency` | Global Moran's I of numeric time on the same binary graph, restricted to finite times. `time_cells` counts finite-time cells. The score can be negative or greater than one. |

Velocity consistency measures local coherence, not biological direction or
accuracy. Methods that smooth neighboring velocities can score highly even
when distant cells are also highly correlated.

## Optional excess spatial velocity consistency (ESVC)

ESVC subtracts an exact non-neighbor baseline from the paired spatial term:
`ESVC = VC_spatial - VC_null`. Both nulls exclude the anchor cell and its
spatial neighbors. No random draw or seed is used.

| Output | Null comparison |
| --- | --- |
| `esvc_label_matched` | For each edge `i -> j`, compare `i` with valid non-neighbors sharing `j`'s annotation. The anchor and neighbor can have different annotations. Edges with no eligible null are excluded from both terms. |
| `esvc_global` | Compare each anchor with all valid non-neighbors, regardless of annotation, using only anchors with an eligible null. |

With `--esvc`, the wide CSV/heatmap naming pattern also produces
`spatial_esvc_label_matched_<real_or_sim>` and `spatial_esvc_global_<real_or_sim>`.
The long table adds each paired spatial term (`vc_spatial_matched` or
`vc_spatial_global`), null term (`vc_null_label_matched` or `vc_null_global`),
cell/edge counts, and `matched_edge_fraction`. No ESVC outputs are written
unless requested. Without `cluster_key`, global ESVC can remain available,
while annotation-matched ESVC is `NA` and the row is partial.

Compare the matching spatial/null columns: `0.995 - 0.990 = 0.005` indicates
little additional local coherence above that baseline. ESVC does not establish
biological direction, accuracy, or statistical significance. Annotation
matching can remove spatial signal associated with annotation boundaries;
the global null can reflect differences in cell-type composition. Read the
paired terms and coverage alongside either difference.
