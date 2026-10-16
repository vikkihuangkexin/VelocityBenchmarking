# Groundtruth correlation

This metric compares predicted and true velocity per cell. It computes their
cosine similarity, maps it from `[-1, 1]` to `[0, 1]`, and reports the mean of
cells with nonzero velocity in both inputs. The comparison space matters:
gene-space and projected-embedding scores can differ.

## Run

Install Scanpy, scVelo, NumPy, pandas, and SciPy. Commands run from the
repository root.

```bash
python Evaluation/Groundtruth-correlation/groundtruth_correlation.py \
  --manifest inputs.csv --output-dir groundtruth_results
```

The input CSV requires `modality,method,dataset_id,path,vkey`. Each row can
set `score_space=gene` or `score_space=embedding`; `gene` is the default except
for ATAC, which defaults to `embedding`.
`modality` accepts `real`, `sim`, `atac`, `spatial`, or `lineage`, but does not
trigger hidden dataset-ID rules. Unlike the path-based metrics, this entry
uses `dataset_id` and a modality field that also accepts `real` and `sim`. `path` points to the result H5AD and resolves
relative to the input CSV. Optional `run_id` distinguishes repeated results
and defaults to `single`.

For `gene`, `vkey` and optional `gt_vkey` name cell-by-gene layers in the
result H5AD. The default `gt_vkey` is `ground_truth_velocity`. Both layers
must use the same cells and genes. A reference NPZ cannot substitute a
high-dimensional truth layer.

For `embedding`, `coord_key` names the coordinate matrix in `adata.obsm` and
is required except for the ATAC default described below. The default
projected velocity keys are `<vkey>_<basis>` and
`<gt_vkey>_<basis>`, where `basis` is `coord_key` without a leading `X_`.
Set `pred_embedding_key` or `gt_embedding_key` for other names. Existing
embeddings and graphs are reused. If an embedding is absent, the matching
high-dimensional velocity layer is projected with scVelo. Optional `xkey`
names the expression layer for a velocity graph that must be built.
Projection densifies sparse velocity layers; use saved projections when that
conversion would exceed available memory.

For the study's simulated ATAC data, `modality=atac` defaults to
`coord_key=X_tsne` and `gt_embedding_key=true_velocity_tsne`. The true
velocity is read directly from `obsm['true_velocity_tsne']`; no true
cell-by-gene velocity layer is needed. Set `pred_embedding_key` when a method
stores its projected prediction under another name, such as `gv_tsne`.

```csv
modality,method,dataset_id,run_id,path,score_space,vkey,gt_vkey,coord_key
sim,InterVelo,bifurcating_cell1000_gene1000,single,results/method.h5ad,gene,velocity,ground_truth_velocity,
sim,InterVelo,bifurcating_cell1000_gene1000,single,results/method.h5ad,embedding,velocity,ground_truth_velocity,X_dimred
```

An ATAC simulation row may omit `score_space` and `coord_key`:

```csv
modality,method,dataset_id,path,vkey,pred_embedding_key
atac,GraphVelo,03_bifurcation5_cell1000_gene1000,results/graphvelo.h5ad,velocity_gv,gv_tsne
```

The two example rows produce separate scores. A result without actual ground
truth velocity is unsupported; a coordinate or label reference alone is not
truth. In particular, the supplied lineage-tracing reference files contain no
true velocity.

## Outputs

`groundtruth_correlation.csv` contains one row per input with the mean score,
`score_space`, reference mode, and input/matched/valid cell counts.
Its legacy `mean_cosine` column holds the mapped `[0,1]` mean, not raw cosine.
`groundtruth_correlation_cells.csv` contains individual cell scores. Cells
with a zero predicted or true vector, or an entirely NaN velocity row, have
an empty score and are excluded from the mean. The totals report zero and NaN
rows separately. Partially non-finite velocity rows and missing truth produce
explicit errors.

## Optional embedding reference

Set `reference_npz` on an embedding row to compare on an explicit reference
basis. The NPZ needs `cell_names` and `X_basis`; `cell_names_unique` is accepted
as an alternate exact cell ID key. The optional true projected velocity array
defaults to `gt_dimred`. Set `reference_basis_key` or `reference_gt_key` when
these array names differ. If true projected velocity is absent from the NPZ,
the result H5AD must provide a usable true velocity layer or embedding.

The input can select a cell ID column with `reference_cell_key`. Name matching
needs at least 95% of result cells by default; set `min_match_ratio` in the
CSV to change that threshold. A partial match removes unmatched cells and
records the counts. If names do not match, position matching is allowed only
when result and reference coordinates already agree. When reference
coordinates differ, the old predicted projection is discarded and rebuilt
from the high-dimensional predicted velocity layer. Neighbor and velocity
graphs are rebuilt only when absent/invalid or invalidated by a coordinate change or cell subset.
The matching positive and negative velocity graphs are validated together.

```csv
modality,method,dataset_id,path,score_space,vkey,coord_key,reference_npz
sim,InterVelo,bifurcating_cell1000_gene1000,results/method.h5ad,embedding,velocity,X_dimred,simdata_reference/bifurcating/bifurcating_cell1000_gene1000_reference_data.npz
```

Relative result and reference paths inside the CSV resolve relative to that
CSV. The reference is never guessed from the dataset ID. A user can supply a
different NPZ for a new simulation without editing the script.
The study's reference NPZ files are already provided under
`simdata_reference/`; their former server-specific export script is not
needed to score the supplied inputs.
