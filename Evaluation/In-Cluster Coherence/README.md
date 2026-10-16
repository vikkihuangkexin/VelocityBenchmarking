# In-Cluster Coherence (ICCoh)

ICCoh measures cosine similarity between a cell's high-dimensional velocity
and the velocities of neighbors with the same annotation. It supports RNA,
ATAC, spatial, and lineage results with a cell-by-gene velocity layer; 2D
velocity alone is insufficient. Install Scanpy, NumPy, pandas, SciPy, and
scikit-learn in the metric environment.

## Inputs

| CSV field | Required value |
| --- | --- |
| `method`, `id`, `path`, `vkey`, `data_type` | Method, dataset ID, result H5AD, velocity layer, and `real` or `sim` |
| `cluster_key` | Annotation column in `obs`; may be blank for simulated ATAC, which selects `pseudo_celltype`, then `celltype` |
| `modality` | Optional: `rna` (default), `spatial`, `atac`, or `lineage` |
| `basis_name` | Optional coordinates for rebuilding missing/invalid neighbors |

The H5AD needs `layers['<vkey>']` and the specified annotation. Existing valid
neighbor connectivities are reused for every modality, including spatial.
When neighbors need rebuilding, coordinates must exist: defaults are
`X_umap` for real RNA/ATAC/lineage, `X_dimred` for other simulations,
`X_tsne` for simulated ATAC, and `spatial` for spatial inputs.
`basis_name=umap` selects `obsm['X_umap']`; `spatial` selects the literal
`obsm['spatial']`. Exact `X_` keys are also accepted.

```csv
method,id,path,vkey,data_type,basis_name,cluster_key,modality
InterVelo,1,results/intervelo_1.h5ad,velocity,real,umap,ClusterName,rna
TopoVelo,55,results/topovelo_55.h5ad,gat_velocity,real,spatial,cell_type,spatial
```

Relative H5AD paths resolve from the manifest directory. There is no metadata
restoration step, differentiation-path table, or velocity-graph requirement.

For real spatial dataset `69`, values matching `Neuron_<digits>` in the
selected annotation are grouped as `Neuron`, retaining the second-revision
annotation rule. Other datasets keep their supplied annotations.

## Run

From the repository root:

```bash
python "Evaluation/In-Cluster Coherence/iccoh.py" \
  --manifest inputs.csv --output-dir results/iccoh
```

`--n-neighbors` defaults to 30 and applies when neighbors must be rebuilt.
Optional CSV `run_id` distinguishes repetitions (default `single`).

## Outputs and calculation details

| File | Contents |
| --- | --- |
| `iccoh_cells.csv` | Labeled cells, same-cluster neighbor counts, and scores |
| `iccoh_clusters.csv` | Cluster means and coverage |
| `iccoh_totals.csv` | Pooled cell mean, cell counts, and selected gene count |

A cell score averages cosine similarity over its same-cluster neighbors.
The dataset total averages finite cell scores, rather than weighting clusters
equally. Cells without same-cluster neighbors have blank scores. Unlabeled
cells are omitted and counted. A finite zero velocity vector contributes
cosine zero under scikit-learn's convention.

Optional `gene_mask_policy=not_nan` excludes genes with NaN in the **first**
velocity row. Simulated ATAC defaults to `finite`, which also excludes
infinities in that row. These rules preserve the source implementations;
neither repairs non-finite values in later rows. Missing/nonnumeric velocity,
an empty gene mask, or non-finite values in a compared cell cause an error.
