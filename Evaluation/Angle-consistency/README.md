# Velocity Angle Consistency

Compare predicted 2D velocity with the tangent of a reference differentiation
trajectory in the same coordinates. The score is the percentage of scored
cells with an angle from 0 to 60 degrees. RNA, ATAC, spatial, and lineage
inputs use distinct reference rules described below.

Install Scanpy, scVelo, NumPy, pandas, SciPy, scikit-learn, and Matplotlib.
Commands below run from the repository root.

## Inputs

The result H5AD needs finite 2D coordinates, cell annotations matching the
path, and either `obsm['<vkey>_<basis>']` or `layers['<vkey>']` for projection.
Coordinates are never computed. Building a missing projection needs matching
velocity graphs or expression data and neighbors to construct them.
Projection densifies sparse velocity layers; a saved projection avoids that
memory cost.

| Input | Default coordinates | Annotation/reference |
| --- | --- | --- |
| Real RNA, ATAC, lineage | `X_umap` | Explicit `cluster_key` or a column matching path labels |
| Real spatial | `spatial` | Spatial annotations matching the path |
| Ordinary RNA simulation | `X_dimred` | `milestone`, or an explicit annotation key |
| ATAC simulation | `X_tsne` | `pseudo_celltype`, then `celltype`, then `cell_type`, or an explicit key |
| Spatial simulation | `spatial` | `cell_type`, or an explicit annotation key |
| Lineage-tracing/Bursting-tree RNA simulation | NPZ `X_basis` | Matching public reference NPZ under `--reference-dir` |

Coordinate keys are in `obsm`; annotations are in `obs`. `basis_name` uses a
bare name such as `umap`, `dimred`, or `spatial`. `spatial` selects the literal
key; other names select `X_<basis_name>`. ATAC simulations use `tsne`; the
dedicated lineage simulation uses `umap`.

The default [path table](../differentiation_paths.csv) contains study paths.
For another dataset, pass `--path-csv` with the following required header:

```csv
dataset_id,path_id,position,label,closed
my_real_dataset,1,0,Progenitor,false
my_real_dataset,1,1,Intermediate,false
my_real_dataset,1,2,Mature,false
```

`dataset_id` matches manifest `id` or single-input `--dataset`. Positions
start at zero, with at least two labels per path; branches use distinct
`path_id` values. Labels match annotation values exactly, except ordinary RNA
simulation normalizes underscores/hyphens to spaces. Real paths are open.
`closed=true` applies to supported specialized simulation builders; custom
and real splines remain open. Inline `--differentiation-paths "A|B|C;A|B|D"`
also creates open paths.

## Run

For one real dataset:

```bash
python Evaluation/Angle-consistency/angle_consistency.py \
  --input results/sample.h5ad --data-type real --modality rna \
  --method MyMethod --dataset my_real_dataset --cluster-key cell_type \
  --basis-name umap --velocity-key velocity --path-csv my_paths.csv \
  --output-dir results/angle
```

For an unfamiliar simulation, supply paths and `--topology-type custom`.
For a simulation annotation override use `--milestone-key` or `--cluster-key`.

The standard manifest fields are `method,id,path,vkey,data_type`.
`method,id,path` are mandatory; `data_type=real|sim` must be present or supplied
by `--data-type`. Omitted/blank `vkey` uses `--velocity-key` (default `velocity`).
Declare `modality,basis_name,cluster_key` for custom IDs; known study IDs have
compatibility defaults.

```csv
method,id,path,vkey,data_type,modality,basis_name,cluster_key,topology_type
MyMethod,my_real_dataset,results/sample.h5ad,velocity,real,rna,umap,cell_type,
MyMethod,my_sim_dataset,results/simulation.h5ad,velocity,sim,rna,dimred,milestone,custom
```

```bash
python Evaluation/Angle-consistency/angle_consistency.py \
  --csv-file inputs.csv --path-csv my_paths.csv --output-dir results/angle \
  --save-detailed-csv --plot-reference-curves
```

Relative H5AD, `reference_dir`, and `annotation_h5ad` paths inside the manifest
resolve from its directory. `--path-csv` resolves from the working directory
and replaces the default table. Each method/ID pair must be unique within a
command; run repetitions into separate output directories.

## Outputs and limits

| Output | Contents |
| --- | --- |
| `benchmark_total.csv` | Method rows and dataset columns, with the 0-60 degree percentage |
| `benchmark.csv` | Per-label angle-bin statistics, with `--save-detailed-csv` |
| `<method>/pdf/`, `<method>/png/` | Rose plots for successful inputs |
| Reference-curve PNG | Simulation QA plot with `--plot-reference-curves` |
| `errors.log` | Processing errors |

The denominator includes only cells on retained paths with finite predicted
and reference vectors whose individual norms are at least `1e-6`. Other cells
are excluded. Angle-bin summaries overlap (for example `60-90` and `60-80`)
and should not be summed together. Real paths can retain a contiguous observed
portion with missing endpoints; paths with internal missing labels are dropped.

For real spatial dataset `69`, `cell_type` is the default annotation and
`Neuron_<digits>` labels are grouped as `Neuron`. An explicit annotation key
takes precedence. Spatial simulations have no supplied true velocity: their
angle score uses trajectory geometry and is an auxiliary score.

## Reference construction and advanced options

| Input | Reference rule |
| --- | --- |
| Real data | Open splines through declared annotation centers, including cell-cycle paths |
| Ordinary RNA simulation | Topology-specific guide points, spline degree, and smoothing; CSV paths give segments/direction |
| Custom simulation | Basic open spline through declared label centers |
| ATAC simulation | One directed KNN/Dijkstra curve per unique CSV edge; shared labels use their nearest curve |
| Spatial chicken-heart simulation | Path-based trajectory without true-velocity assistance |
| Lineage-tracing/Bursting-tree RNA simulation | Dedicated geometry and metadata; CSV paths do not replace these rules |

Branching ATAC references follow observed 2D cell distributions by default.
They keep label clouds separate until the declared transition and retain
bends with arc-length interpolation. The single-edge ATAC reference uses
`obsm['true_velocity_tsne']` for guidance by default. No gene-space truth layer
is required. Use `--no-gt-velocity-enhancement` for geometry-only linear curves,
or `--use-gt-velocity-enhancement` to guide branching curves with 2D truth.
Geometry-only ATAC scoring does not need `true_velocity_tsne`.

An ATAC `--annotation-h5ad` supplies missing standard labels by exact cell
IDs. The raw `01_linear_cell1000_gene1000` ATAC input, for example, needs
`early/late` from its paired RNA H5AD. Explicit nonstandard annotation keys
must exist in the result H5AD.

The scRNA lineage reference retains its separate reciprocal KNN distance
sum, arc-length anchors, and path-length-scaled spline smoothing. ATAC uses
its own graph and branch rules. Bursting-tree references use edge/time NPZ
metadata. Available NPZ truth enables ordinary RNA guide-point enhancement
by default; `--no-gt-velocity-enhancement` disables it. Explicit enhancement
requests fail when truth is unavailable.

Optional manifest fields include `milestone_key,topology_type,annotation_h5ad,
reference_dir,differentiation_paths,gt_velocity_enhancement,plot_reference_curves,
reference_curves_dir,n_jobs`. `cluster_key` overrides `milestone_key` in
simulation mode. `n_jobs` defaults to 1 for graph construction. Valid saved
projections and matching graph pairs are reused. Missing/invalid graph
construction preserves cells and computes missing moments on valid neighbors
where possible. For custom topology, the QA illustration may differ from the
basic scoring spline. ATAC and dedicated lineage QA plots show scoring curves.
