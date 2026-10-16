# Evaluation

Choose an entry below after collecting the methods' result H5AD files.
Each metric has its own input contract and output files. Install its listed
dependencies in a metric environment; tool-specific environments can use
different package versions.

| Entry | Use | Main inputs beyond the result H5AD |
| --- | --- | --- |
| [CBDir](Cross-Boundary%20Direction%20Correctness/README.md) | Direction toward a neighboring target state | Coordinates, annotations, projected velocity, directed paths |
| [ICCoh](In-Cluster%20Coherence/README.md) | Velocity coherence within annotated states | Cell-by-gene velocity and neighbor connectivities |
| [Transition score](Transition%20score/README.md) | Velocity-graph values along directed state edges | Velocity graph, neighbor connectivities, annotations, directed paths |
| [Angle consistency](Angle-consistency/README.md) | Agreement with reference trajectory tangents | 2D coordinates, projected velocity, paths; reference NPZ for dedicated simulation models |
| [Groundtruth correlation](Groundtruth-correlation/README.md) | Predicted versus true velocity | Actual true velocity in the chosen comparison space |
| [Spatial velocity/time consistency](Spatial%20consistency/README.md) | Local velocity coherence and spatial time autocorrelation | Spatial coordinates, velocity layer, numeric time column |
| [Consistency score](Consistency%20score/README.md) | Consistency analysis | See the entry's own configuration |
| [Multirun stability](Multirun%20stability%20analysis/README.md) | Compare repeated model fits | Repeated results; see the entry's README |
| [Result plots](Result-plot/README.md) | Plot method results | Result H5AD; see the entry's README |

## Prepare a manifest and run

CBDir, ICCoh, Transition score, and Angle consistency use the common manifest
fields `method,id,path,vkey,data_type`, with `data_type=real|sim`. Declare
`cluster_key` and `basis_name` explicitly when possible. `modality` uses
`rna|atac|spatial|lineage`; defaults and additional required fields differ by
metric. Relative result paths resolve from the manifest directory.
Groundtruth correlation uses a separate schema, including `dataset_id` and
`score_space`; follow its README.

```csv
method,id,path,vkey,data_type,basis_name,cluster_key,modality
MyMethod,1,results/sample.h5ad,velocity,real,umap,ClusterName,rna
```

From the repository root, for example:

```bash
python "Evaluation/Cross-Boundary Direction Correctness/cbdir.py" \
  --manifest inputs.csv --output-dir results/cbdir
```

The included [`differentiation_paths.csv`](differentiation_paths.csv) supplies
ordered study paths for CBDir, Transition score, and Angle consistency. For
new IDs, pass your own path table using the option documented by each entry.
The required columns are `dataset_id,path_id,position,label,closed`; positions
start at zero. Real cell-cycle paths are open; simulated cycles may close.

## Interpret outputs

Read per-cell/edge coverage alongside aggregate scores. Metrics differ in
their denominators, zero-vector rules, and handling of NaN. Scores from
different coordinate bases or neighbor graphs need care when compared.
CBDir, ICCoh, and Transition score reuse valid input neighbors. Spatial
velocity/time consistency instead defines its own symmetric, binary
14-nearest-neighbor spatial graph. Its optional `--esvc` diagnostic is off by
default and was not used in the second-revision paper.

## Profile a tool with Scalene

[`run_scalene.sh`](run_scalene.sh) is a separate, legacy profiling driver.
It requires `--data-csv`, `--results-csv`, `--target-script`, `--target-method`,
and `--save-dir`; its dataset CSV uses `ID,path` and its existing-result CSV
uses `method,ID,result_path`. It passes one positional H5AD path to the target
script, so entries using named CLI options need a wrapper. `--enable-gpu` and
`--scalene-args` control profiling. Inspect its default `--profile-only` list
before targeting another tool.

The driver's cleanup currently calls `pkill -9 python`. Run it only in an
isolated container/process namespace, or adapt cleanup before use. This driver
is not part of the metric workflow above. See [Docker instructions](../Dockerfile/README.md)
for container setup.
