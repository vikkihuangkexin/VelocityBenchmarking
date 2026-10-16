# Transition score

`run_trans.py` measures predicted velocity-graph values along declared
differentiation edges. It supports RNA, ATAC, spatial, and lineage results
with annotations and a velocity graph or high-dimensional velocity layer.
Install Scanpy, scVelo, NumPy, pandas, and SciPy in the metric environment.

## Inputs

| CSV field | Required value |
| --- | --- |
| `method`, `id`, `path`, `vkey`, `data_type` | Method, dataset ID, result H5AD, velocity key, and `real` or `sim` |
| `cluster_key` | Annotation column in `obs`; may be blank for simulated ATAC, which selects `pseudo_celltype`, then `celltype` |
| `modality` | Optional: `rna` (default), `spatial`, `atac`, or `lineage` |
| `basis_name` | Optional coordinate basis for rebuilding missing/invalid neighbors |

The H5AD must contain the path labels and either `uns['<vkey>_graph']`,
`obsp['<vkey>_graph']`, or `layers['<vkey>']` for graph construction.
Existing valid neighbor connectivities and velocity graphs are reused.
Missing/invalid neighbors require finite coordinates: defaults are `X_umap`
for real RNA/ATAC/lineage, `X_dimred` for other simulations, `X_tsne` for
simulated ATAC, and `spatial` for spatial inputs. `basis_name=umap` selects
`obsm['X_umap']`; `spatial` selects the literal `obsm['spatial']`. Exact
`X_` keys are accepted. Rebuilding a velocity graph also needs an expression
layer (`Ms`, `spliced`, or optional CSV `xkey`) and usable distance rows.
Missing moments can be calculated from spliced/unspliced counts on existing
neighbors. No NPZ metadata repair or transition-probability conversion is used.

```csv
method,id,path,vkey,data_type,basis_name,cluster_key,modality,run_id,xkey
scVelo,12,data/cell_cycle.h5ad,velocity,real,umap,cell_cycle_phase,rna,single,
InterVelo,cycle-simple_cell1000_gene1000,data/cycle.h5ad,velocity,sim,dimred,milestone,rna,single,Ms
```

Relative H5AD paths resolve from the manifest directory. `run_id` is optional
and defaults to `single`.

The default path table is [`../differentiation_paths.csv`](../differentiation_paths.csv).
For another dataset, supply `--paths-csv` with the required header
`dataset_id,path_id,position,label,closed`. `dataset_id` matches `id`;
positions start at zero in differentiation order. Branches have distinct
`path_id` values. `closed=true` adds the last-to-first edge. Identical
directed edges are counted once per dataset. Included simulated cycles close;
real cell-cycle paths remain open. Unknown IDs need explicit paths.

For real spatial dataset `69`, values matching `Neuron_<digits>` in the
selected annotation are grouped as `Neuron`, retaining the second-revision
annotation rule. Other datasets keep their supplied annotations.

## Run

From the repository root:

```bash
python "Evaluation/Transition score/run_trans.py" \
  --manifest inputs.csv --output-dir results/transition
```

Relative `--paths-csv` values resolve from the manifest directory.
`--n-neighbors` defaults to 30 when neighbors must be rebuilt.

## Outputs and score interpretation

| File | Contents |
| --- | --- |
| `transition_score.csv` | Dataset totals and number of contributions |
| `transition_score_by_edge.csv` | Directed edge means and coverage |
| `transition_score_cells.csv` | Eligible source cells and their scores |

For each source cell, average the **input velocity-graph values** to its
target-state neighbors. A missing edge in a valid sparse velocity graph
contributes zero. Cells without target neighbors are unscored. The dataset
total pools finite source-cell scores across unique directed edges; it is
not an equal-weight mean of edge means. A cell may contribute to more than
one outgoing edge. Unscored cell/edge means are blank.
