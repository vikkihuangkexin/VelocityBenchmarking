# Cross-Boundary Direction Correctness (CBDir)

CBDir measures whether projected velocity points from a source state toward
neighboring cells in a declared target state. It supports RNA, ATAC, spatial,
and lineage results with cell annotations and projected velocity. Install
Scanpy, scVelo, NumPy, pandas, SciPy, and scikit-learn in the metric environment.

## Inputs

Supply a CSV with these fields. Relative H5AD paths resolve from the CSV directory.

| Field | Required value |
| --- | --- |
| `method`, `id`, `path`, `vkey`, `data_type` | Method name, dataset ID, result H5AD, velocity key, and `real` or `sim` |
| `cluster_key` | Annotation column in `obs`; may be blank for simulated ATAC, which selects `pseudo_celltype`, then `celltype` |
| `modality` | Optional: `rna` (default), `spatial`, `atac`, or `lineage` |
| `basis_name` | Optional coordinate basis; defaults are listed below |

| Input | Default coordinates |
| --- | --- |
| Real RNA, ATAC, or lineage | `obsm['X_umap']` |
| RNA or lineage simulation | `obsm['X_dimred']` |
| ATAC simulation | `obsm['X_tsne']` |
| Spatial | `obsm['spatial']` |

`basis_name=umap` selects `X_umap`; `spatial` selects the literal `spatial`
key. An explicit `X_` key is accepted. The H5AD needs finite coordinates and
the path labels in the declared annotation. It must contain
`obsm['<vkey>_<basis>']` or `layers['<vkey>']` for projection. A method with
only 2D velocity can use its saved projection. Missing fields cause an error;
the script does not restore metadata from an NPZ.
Projection densifies a sparse velocity layer; a saved projection avoids that
memory cost.

```csv
method,id,path,vkey,data_type,basis_name,cluster_key,modality
InterVelo,1,results/intervelo_1.h5ad,velocity,real,umap,ClusterName,rna
TopoVelo,55,results/topovelo_55.h5ad,gat_velocity,real,spatial,cell_type,spatial
```

The default path table is [`../differentiation_paths.csv`](../differentiation_paths.csv).
For a new ID, supply a matching table with `--paths`. Its required header is
`dataset_id,path_id,position,label,closed`. `dataset_id` matches `id`;
positions start at zero and follow differentiation order. Give branches
distinct `path_id` values. `closed=true` adds the last-to-first edge. Shared
directed edges are counted once per dataset. Simulated cycles close; the
included real cell-cycle paths are open. Unknown IDs need explicit paths.

For real spatial dataset `69`, values matching `Neuron_<digits>` in the
selected annotation are grouped as `Neuron`, retaining the second-revision
annotation rule. Other datasets keep their supplied annotations.

## Run

From the repository root:

```bash
python "Evaluation/Cross-Boundary Direction Correctness/cbdir.py" \
  --manifest inputs.csv --output-dir results/cbdir
```

`--paths` resolves relative to the working directory; H5AD paths inside the
manifest resolve relative to the manifest. `--n-neighbors` defaults to 30 and
is used only when a neighbor graph is missing or invalid.

## Outputs and score interpretation

| File | Contents |
| --- | --- |
| `cbdir_cells.csv` | Source-cell scores for each directed edge, including unscored cells |
| `cbdir_edges.csv` | Edge means and eligible/scored cell counts |
| `cbdir_totals.csv` | Dataset mean, number of contributions, and invalid velocity count |

A cell score is its mean cosine to target-state neighbor displacements.
The total pools finite cell scores across unique directed edges. A cell can
contribute once to each outgoing edge; the total is not an equal-weight mean
of edge means. Positive values point toward the target; negative values point
away. Cells without target neighbors or with zero/entirely NaN velocity have
blank scores and do not enter the total. Zero displacement contributes cosine
zero under scikit-learn's convention. Partially non-finite velocity rows fail.

## Projection options

`run_id` distinguishes repetitions (default `single`).
`velocity_embedding_key` overrides the projected `obsm` key. `xkey` selects
the expression layer when a velocity graph must be built; otherwise missing
`Ms` can be computed from spliced/unspliced counts on valid existing neighbors.
Existing valid connectivities are reused for every modality, including spatial.
Projection reuses the declared `vkey`'s valid positive/negative graph pair;
missing or invalid graphs are rebuilt from that velocity layer. scVelo also
needs usable distance rows to build a graph.

`invalid_embedding_recompute_ratio` defaults to `0.15`. At or above this
fraction of entirely NaN rows, a saved projection is recomputed and therefore
requires the high-dimensional velocity layer. Remaining entirely NaN rows are
excluded. No common cleanup threshold is applied to other metrics.
