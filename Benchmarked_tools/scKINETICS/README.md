# scKINETICS

Run scKINETICS on real multiome data or paired scMultiSim data. Benchmark data
archives are linked in the [repository README](../../README.md#datasets).

## Setup

Create the environment using the maintained fork's
[installation instructions](https://github.com/sd68515/scKINETICS_py311_r441#environment-and-installation).
Then run these commands from the VelocityBenchmarking repository root:

```bash
git clone https://github.com/sd68515/scKINETICS_py311_r441.git ../scKINETICS_py311_r441
git -C ../scKINETICS_py311_r441 checkout 980118ea3b9dcb858bffa9db538b67eeb601d6e1
export PYTHONPATH="$PWD/../scKINETICS_py311_r441${PYTHONPATH:+:$PYTHONPATH}"
```

Keep `scmultisim_grn_params_100.csv` alongside `scKINETICS.py`.

## Prepare inputs and run

| Data | Files to provide |
| --- | --- |
| Real data | RNA h5ad and differential-accessibility peak BED |
| scMultiSim data | Paired RNA and ATAC h5ad files |

Edit [inputs.example.csv](inputs.example.csv) with your local file paths.
Each row describes one dataset, with both simulated modalities on the same row.
`file_path` is the RNA file; fill exactly one of `peaks_bed` or `atac_h5ad`.
For real data, set `genome` to the matching assembly (`mm10`, `mm39`, or `hg38`).
Paths are relative to the working directory.

```bash
python Benchmarked_tools/scKINETICS/scKINETICS.py \
  --metadata-file Benchmarked_tools/scKINETICS/inputs.example.csv \
  --output-dir ./output \
  --fig-dir ./figures
```

Omit `--fig-dir` if figures are not needed. For your own datasets, check the input
requirements below before running.

## Results

The main result is `output/<dataset_name>/scKINETICS_<dataset_name>.h5ad`.
It contains the modeled genes, `layers['velocity']`, and
`obsm['velocity_<basis>']` (for example, `velocity_tsne`). Original gene names
are preserved. Optional scatter, stream, grid, and pseudotime figures are saved
under `figures/<dataset_name>/`. Input files are not modified.

## Additional usage

<details>
<summary>Single-dataset commands and annotation defaults</summary>

Real data:

```bash
python Benchmarked_tools/scKINETICS/scKINETICS.py \
  --input-h5ad /data/sample_rna.h5ad \
  --peaks-bed /data/sample_peaks.bed \
  --genome mm10 \
  --output-dir ./output
```

Paired scMultiSim data:

```bash
python Benchmarked_tools/scKINETICS/scKINETICS.py \
  --input-h5ad /data/01_linear_cell1000_gene1000_rna.h5ad \
  --atac-h5ad /data/01_linear_cell1000_gene1000_atac.h5ad \
  --dataset-name 01_linear_cell1000_gene1000 \
  --output-dir ./output
```

| Data | Default grouping in `obs` | Default embedding in `obsm` |
| --- | --- | --- |
| Real data | `celltype` | `X_umap` |
| Simulation ID containing `linear` | `pseudo_celltype` | `X_tsne` |
| Other simulation | `celltype` | `X_tsne` |

In single-dataset mode, `--dataset-name` defaults to the RNA filename stem.
For your own data, use `--cluster-key` and `--embedding-basis` to select the
appropriate annotations. In a CSV, use `cluster_key` and `dimred_key` instead;
blank fields use the defaults above. Provide the published dataset ID when
using the benchmark's naming-based defaults.

</details>

<details>
<summary>Requirements for your own data</summary>

### RNA and preprocessing

Use cells by genes, with unique gene identifiers in `var_names` and non-missing
group labels in the selected `obs` column. Real data uses `X` as expression;
scMultiSim data uses `layers['spliced']`. `layers['Mc']` is not used.
For real data, gene identifiers must match the reference annotation's gene
symbols.

By default, the script retains cells expressing at least 200 genes and genes
expressed in at least 3 cells, removes genes with zero total expression,
normalizes each cell to 10,000 counts, and applies `log1p`. Use
`--skip-normalize` and/or `--skip-log1p` when those steps have already been
applied to the expression matrix used for fitting. More than 50 cells and genes
must remain after filtering.

The selected embedding must exist in `obsm`; `X_umap` is computed if missing.
Other embeddings, including `X_tsne`, must be supplied. Existing `X_pca` is used
when available; otherwise PCA is computed.

### Real-data peaks

Provide a tab-separated BED whose first three columns are chromosome, start,
and end. A header with `chrom`, `chromStart`, and `chromEnd` is also accepted.
Use peak coordinates and a reference genome from the same assembly. The default
width filter retains peaks with positive width strictly below 2,000 bp.

### Simulated ATAC annotations

RNA and ATAC must contain the same cells. The script aligns them using unique
`obs['simulation_cell_id']` values when present in both files, otherwise unique
`obs_names`. Cell sets that differ are rejected.

The ATAC h5ad must contain:

| Field | Content |
| --- | --- |
| `var['chrom']`, `var['chromStart']`, `var['chromEnd']` | Valid peak coordinates |
| `var['sckinetics_diff_peak']` | Boolean peak-selection mask |
| `varm['region_to_tf']` | Peak-by-TF link matrix |
| `uns['region_to_tf_names']` | TF identifiers in matrix-column order |
| `varm['region_to_gene']` | Peak-by-gene link matrix |
| `uns['region_to_gene_genes']` | Gene identifiers in matrix-column order |

Link matrices may be dense or sparse and must contain finite, nonnegative
values. Each name array must contain unique identifiers and match its matrix's
columns. RNA gene names must also remain unique after uppercasing. TFs and
targets are matched to RNA genes by name; subsetting RNA can therefore remove
regulators or targets from the fitted network.

`--diff-peak-key` selects another mask column. Alternatively, the mask may be in
`uns['sckinetics_diff_peaks']['selected']`. The same width filter applies as for
real peaks. `--all-peaks` explicitly uses all peaks passing that filter. A
selection with no RNA-compatible TF-target pairs raises an error.

The simulation workflow fits RNA expression using the supplied regulatory
annotations. ATAC count values are not fitted directly, and synthetic
coordinates are not scanned against a real genome. Simulator-derived network
information is part of the model input and should be reported in comparisons.

</details>

<details>
<summary>TF identities for simulated data</summary>

For annotations generated with scMultiSim's
[`GRN_params_100` reference](https://github.com/ZhangLabGT/scMultiSim/blob/563c3b542ada289fb4d9789bb1bc944229e78699/data/GRN_params_100.RData),
the script can recover TF identities using the adjacent
[reference CSV](scmultisim_grn_params_100.csv). This requires
`uns['simulation']['simulator']` to be `scMultiSim`, matching TF labels/order and
template gene identifiers, and complete agreement between the stored
region-to-TF links and those reconstructed from the reference and region-to-gene
links across all peaks. Every TF column must have nonzero links. A mismatch
raises an error.

For another regulatory network, use TF identifiers that are already RNA gene
names, or provide an explicit mapping through `--tf-map tf_map.csv`. The CSV
must have columns `tf,gene`, with one RNA gene per TF label. A batch input table
can supply a different mapping file per row through an optional `tf_map` column.
The same dictionary can instead be stored in the ATAC h5ad's
`uns['sckinetics_tf_to_gene']`; a mapping CSV takes precedence. All TF labels
that are not already RNA gene names must be mapped to genes in the RNA input.

</details>

<details>
<summary>Parameters and additional outputs</summary>

List all parameters and their defaults:

```bash
python Benchmarked_tools/scKINETICS/scKINETICS.py --help
```

Common adjustments are `--min-genes`, `--min-cells`, `--target-sum`,
`--peak-width-max`, `--maxiter`, and `--tol`. EM, nearest-neighbor, and plotting
job counts default to one and are controlled by `--threads`,
`--sklearn-n-jobs`, and `--n-jobs`, respectively.

The result also includes transition and neighbor graphs in `obsp`, model
settings in `uns['sckinetics_params']`, and, for simulations, the TF mapping and
peak-selection provenance in `uns['sckinetics_simulation']`. Each output folder
also contains the fitted model and runtime AnnData pickles, plus the projected
velocity array `velocity_<basis>.npy`.

Stream-PDF rendering may use SVG conversion. If that conversion is needed,
install `cairosvg`, `svglib`/`reportlab`, or Inkscape.

</details>
