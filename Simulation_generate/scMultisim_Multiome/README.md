# scMultiSim Multiome simulation and preprocessing

This folder generates paired synthetic RNA and ATAC data with RNA velocity
ground truth, then prepares the fields used by the multiome benchmark.
Place it at `VelocityBenchmarking/Simulation_generate/scMultisim_Multiome/`.
Only code and task manifests are distributed. Task manifests describe planned
runs, not a claim that every listed dataset has been generated successfully.

## Files

| File | Purpose |
|---|---|
| `01_simulate.R` | Simulate paired counts and ground truth; write matrices and optional chromVAR deviations |
| `02_compute_diff_peaks.py` | Select differential peaks from observed counts; retain separate latent accessibility comparisons |
| `03_run_cistopic.py` | Fit the original pycisTopic CGS model and retain both topic factors |
| `04_build_pair.py` | Build paired H5AD files, gene-level chromatin, TF-IDF/smoothing, and reference figures |
| `05_add_tf_activity.py` | Optional, explicitly labeled synthetic TF-activity proxy for missing annotations |
| `run.py` | Run selected datasets, check dependencies, record settings, resume matching runs, and validate outputs |
| `downsample.py` | Generate the ten paired cell-retention tasks |
| `validate_pair.py` | Check required fields, cell order, topic reconstruction, and peak-to-gene aggregation |
| `common.py` | Shared matrix I/O and differentiation-tree helpers |
| `datasets_*.tsv` | Original task names, sizes, seeds, and sampling plans |
| `requirements.txt`, `install_r_packages.R` | Python and R dependencies |
| `smoke_test.py` | Small artificial-fixture processing checks; no scientific simulation results |
| `validation.json` | Scope and outcome of release checks |

## Retained simulation tasks

Original IDs are preserved; gaps are intentional.

| IDs | Cells | RNA genes | Topologies | ATAC zero-probability parameter |
|---|---:|---:|---|---|
| 01-12 | 1,000 | 1,000 / 5,000 / 10,000 / 20,000 | linear, bifurcation3, bifurcation5 | 0.80 |
| 13-15 | 5,000 | 1,000 | linear, bifurcation3, bifurcation5 | 0.80 |
| 16-18 | 10,000 | 1,000 | linear, bifurcation3, bifurcation5 | 0.80 |
| 19-21 | 10,000 | 20,000 | linear, bifurcation3, bifurcation5 | 0.80 |
| 25-26 | 10,000 | 20,000 | linear | 0.90 / 0.97 |
| 27-28 | 10,000 | 20,000 | bifurcation3 | 0.90 / 0.97 |

There are 21 standard tasks and four sparse tasks. `atac.p_zero` is the
requested simulator parameter; filtering, linkage, and simulation noise can
change the observed zero fraction. Peak counts are determined by the simulator,
not by the requested RNA gene count. Only observed, gene-linked peaks are kept;
RNA genes and paired cells are retained.

## Dependencies

Use Linux or macOS with `gzip`, R, and Python 3.10-3.12. The original
workflow uses two Conda environments: `scmulti_velo` for R and `crak_cistopic`
for Python. Their names can be changed with `--r-env` and `--python-env`.
Use `current` to use Rscript/Python from the current environment instead.

Install the R packages in a compatible R/Bioconductor environment:

```bash
conda run --no-capture-output -n scmulti_velo Rscript install_r_packages.R
```

Install Python processing dependencies in the Python environment:

```bash
conda run --no-capture-output -n crak_cistopic python -m pip install -r requirements.txt
```

Reuse the working pycisTopic installation from the source workflow. This code
requires `pycisTopic.cistopic_class.create_cistopic_object` and
`pycisTopic.lda_models.run_cgs_models`. For a new installation, follow the
[official pycisTopic instructions](https://pycistopic.readthedocs.io/en/latest/installation.html)
and run the preflight check below. API compatibility is checked explicitly;
the pipeline does not silently substitute another topic-modeling backend.
The original environment lockfile was not supplied, so dependency lower bounds
are installation guidance rather than a historical environment lock.

```bash
python run.py --preflight
```

## Generate and process datasets

Run these commands from this folder. Output paths are user-selected and contain
no fixed server directories. Start with one task before launching larger runs:

```bash
python run.py --list
python run.py --outdir output --ids 1 --dry-run
python run.py --outdir output --ids 1
python run.py --outdir output --ids 2-12
python run.py --outdir output --ids 13-18
python run.py --outdir output --ids 19-21
python run.py --outdir output --ids 25-28
```

For different environment names:

```bash
python run.py --outdir output --ids 1 --r-env my_r_env --python-env my_python_env
```

The driver runs sequentially and preserves the source defaults:

| Setting | Standard tasks | Sparse tasks |
|---|---|---|
| Initial number of CIFs | 50 for 1,000 genes; otherwise 100 | 40 |
| cisTopic topics | 20 | 20 |
| cisTopic iterations | 150 | 60 |
| cisTopic CPUs | 1 | 8 |
| Attempt R chromVAR | Yes | No |
| Save simulator RDS cache | Yes | No |

The source also contained queue launchers using a different fast preset:
40 CIFs, 60 cisTopic iterations, four CPUs, R chromVAR disabled, and no RDS
cache. Preserve that alternative explicitly with `--preset queue`, for example:

```bash
python run.py --outdir output_queue --ids 20-21,26-28 --preset queue
```

Which preset an existing historical dataset actually used cannot be determined
from scripts alone; consult its recorded manifest or original run logs.

`--topics`, `--iterations`, and `--cpus` override the topic-model settings.
The original environment variables remain supported: `SCMULTISIM_NUM_CIFS`,
`RUN_CHROMVAR`, `SAVE_SIM_RDS`, `SAVE_CISTOPIC_OBJECT`, `N_TOPICS`,
`CISTOPIC_ITERATIONS`, and `CISTOPIC_CPUS`. `ATAC_P_ZERO` overrides standard
tasks only; sparse probabilities come from their manifest. For example:

```bash
RUN_CHROMVAR=1 python run.py --outdir output_with_deviations --ids 25-28 --require-chromvar
```

The four original simulation attempts are retained. Retry seeds are the
requested seed plus 0 / 10,000 / 20,000 / 30,000. Their CIF noise, differential
CIF fraction, ATAC effect, and CIF count differ as specified in `01_simulate.R`.
The actual successful seed and settings are written to
`intermediate/simulation_manifest.tsv.gz`; a requested seed alone does not
fully describe a run that used a fallback attempt. R session information,
Python package versions, resolved settings, and code hashes are recorded.

The Python requirements bound NumPy/SciPy to versions compatible with the
AnnData 0.10/0.11 backed sparse-matrix interfaces used here.

Rerun the same command to resume matching completed stages. Changing code or
settings requires a new output root. Old server outputs without `run_config.json`
are not adopted automatically. RDS caches are reused only when their request
signature matches, including topology, sparsity, CIF count, and simulator version.

## TF activity and chromVAR

The R stage can compute actual chromVAR deviations using the simulated
region-to-TF annotations and a deterministic pseudo-GC background. These are
synthetic annotations, not motif matches from a real reference genome. If the
R packages are absent or the computation fails, availability and the reason
are retained explicitly; no zero-filled replacement is created.

The source archive also supplied a faster TF-activity approximation. It sums
counts over the region-to-TF graph, normalizes each cell to 10,000 counts,
applies log1p, and standardizes each TF across cells with sample SD (`ddof=1`).
It does not perform chromVAR background correction. Enable it explicitly:

```bash
python run.py --outdir output_with_proxy --ids 25-28 --tf-activity-proxy
```

For compatibility with the source workflow, both modes use
`atac.obsm['chromVAR']`. The mode and proxy status are stored in
`atac.uns['chromVAR_status']` and the validation report. Existing values are
never replaced. `--require-chromvar` rejects missing values and proxy-only
activity. A standalone backfill is also available:

```bash
python 05_add_tf_activity.py --atac output/25_sparse90_linear_c10000_g20000/25_sparse90_linear_c10000_g20000_atac.h5ad
```

## Paired cell downsampling

After the source dataset `20_bifurcation3_c10000_g20000` completes:

```bash
conda run --no-capture-output -n crak_cistopic python downsample.py --root output
```

The manifest defines IDs 31-40: retain 60%, 70%, 80%, 90%, or 95% of cells,
with two independently seeded replicates at each fraction. In this supplied
generation script, `r1` and `r2` are replicate draws; both sample **cells**.
They do not indicate cell versus gene downsampling. The same ordered cell
selection is applied to RNA and ATAC. All genes and peaks remain.

As in the source code, these derivatives inherit the source topic factors,
TF activity, differential peak selections, chromatin smoothing, embeddings,
and subset graphs. They are not refitted or redetected after subsetting.
Source hashes, selected cells, fraction, seed, and inherited-processing scope
are saved in each downsampling manifest. The differential peak BED is copied
from the source and has the same inherited interpretation. Use a new output
root if the source pair changes. `--ids 31-32` selects a subset of the plan.

## Output fields

Each dataset directory contains `<name>_rna.h5ad`, `<name>_atac.h5ad`,
`sckinetics_differential_peaks.bed`, validation reports, intermediate matrices,
and logs. `output/h5ad/` contains relative links to the paired H5AD files.
Reference figures show cell types, projected velocity arrows/grid, and
pseudotime, in PDF and 300-dpi PNG. The velocity projection uses true time
to prefer forward neighbors and is a reference visualization, not an estimated
velocity result or a benchmark score.

| Object | Main fields |
|---|---|
| RNA `layers` | `spliced`, `unspliced`, `true_velocity`, `chromatin`, `atac_counts`, `Mc` |
| ATAC `layers` | `atac_counts`, `true_accessibility` |
| ATAC `obsm` | `cisTopic` = cells x peaks (`theta @ psi`); `cisTopic_topics` = cells x topics; optional `chromVAR` |
| ATAC `varm` / `uns` | Region-to-gene and region-to-TF graphs, feature names, topic-to-peak factors, complete differential peak tables |
| Both `obs` / `uns` | True cell types, branches, time, pseudotime, directed partial-order metadata, simulation settings |

Synthetic coordinates are used only to align genes and peaks. They are not
real genomic positions. Gene-level chromatin is exactly
`ATAC counts @ region_to_gene`; `Mc` is the supplied MultiVelo-style TF-IDF
followed by RNA-neighbor smoothing. No Signac peak-to-gene inference is run.
RNA `X` contains library-normalized log1p spliced counts; ATAC `X` contains
peak counts. SVD is used for the RNA reference embedding without gene centering.

Observed differential peaks use two-proportion tests on detection rates,
within-comparison Benjamini-Hochberg correction, and the source effect-size
thresholds. A small top-effect fallback is explicitly marked in
`selection_rule`. Comparisons use simulated branch/time annotations. Latent
accessibility selections are kept separately and are never substituted for
observed selections. The exported BED uses the original eight-column layout:
chromosome, 0-based start, end, region, comparison, log2 fold change, adjusted
p-value, selection rule; columns after the first three are workflow metadata,
not a standard BED8 track definition.

## Validation

```bash
python smoke_test.py
python validate_pair.py --rna output/01_linear_c1000_g1000/01_linear_c1000_g1000_rna.h5ad --atac output/01_linear_c1000_g1000/01_linear_c1000_g1000_atac.h5ad
```

The smoke check creates temporary artificial inputs and exercises differential
peak processing, H5AD construction, TF-activity backfill, and all downsampling
tasks. It does not run scMultiSim or fit a real pycisTopic model. Use preflight
and the per-dataset reports to validate actual simulation runs. The release
does not include generated datasets or benchmark ranking calculations.

## Upstream references

- [scMultiSim](https://github.com/ZhangLabGT/scMultiSim)
- [scMultiSim simulation output documentation](https://zhanglabgt.github.io/scMultiSim/reference/sim_true_counts.html)
- [scMultiSim parameter guide](https://zhanglabgt.github.io/scMultiSim/articles/options.html)
- [pycisTopic](https://github.com/aertslab/pycisTopic)
- [chromVAR](https://bioconductor.org/packages/chromVAR/)

The scripts were organized from the supplied `pipeline.zip`. Server-specific
launch/repair scripts and backup copies were consolidated into the portable
driver and optional TF-activity step. Retained manifest values, simulation
attempt settings, differential peak thresholds, and preprocessing formulas
follow that source. Paired downsampling explicitly retains backed `X` matrices
to avoid missing-matrix exports in affected AnnData versions.
