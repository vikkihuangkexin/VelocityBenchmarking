# Simulation pipeline of lineage-tracing datasets by using scReadSim

This repository contains a command-line pipeline for generating synthetic scRNA-seq reads from a selected cell subset and attaching synthetic cells to a donor lineage scaffold. The workflow is designed for GitHub use: code is path-agnostic, all local paths are supplied by command-line arguments, and large input/output files are excluded by `.gitignore`.

## Workflow

```text
loom + reference real CSV data + genome reference + raw BAM
        |
        v
00_prepare_inputs.py
        |
        v
manifest.json + selected barcodes/cell labels + cleaned reference
        |
        v
01_subset_bam_to_selected_cells.py
        |
        v
cell-subset BAM with CB-prefixed read names
        |
        v
02_run_screadsim_fast.py
        |
        v
synthetic count matrices, synthetic barcodes, BED/FASTQ outputs
        |
        v
03_build_lineage_scaffold.py + 04_assign_synthetic_cells_to_donor_lineage.py
        |
        v
synthetic cells annotated with donor lineage metadata
```

## Required input files

The pipeline requires the following external files. These files should not be committed to GitHub.

| Input | Required | Description |
|---|---:|---|
| Loom file | Yes | Must contain `CellID` in column attributes. `_X`, `_Y`, and `Clusters` are optional and are copied to metadata when available. |
| Selected-cell CSV | Yes | Must contain one barcode-like column overlapping loom `CellID`; must contain a cluster/label column such as `cluster`, `Cluster`, `Clusters`, `seurat_clusters`, `leiden`, or `louvain`. |
| Raw BAM | Yes | Must contain cell barcode `CB` tags. The scReadSim count step is configured to use UMI tag `UB:Z`. |
| Genome FASTA | Yes | Reference genome FASTA used for FASTQ generation. |
| Genome FASTA index | Yes | `.fai` file for the reference genome. |
| Gene annotation GTF | Yes | `.gtf` or `.gtf.gz`. The prepare step harmonizes `chr`/non-`chr` chromosome style to the FASTA index. |
| Chromosome sizes | Optional | Two-column `chrom size` file. If omitted, it is generated from the FASTA index. |
| Newick tree | Yes for lineage step | Tree leaf names must contain 10x-style barcodes that overlap the selected-cell barcodes. |
| fgbio JAR | Optional | Only needed if sequencing-error modeling is requested through `--fgbio-jar` in `02_run_screadsim_fast.py`. |

## Software requirements

Install the Python packages and command-line tools listed in `environment.yml` or `requirements.txt`.

```bash
conda env create -f environment.yml
conda activate simulation-generate
```

The following executables must be available in `PATH` when running scReadSim:

```text
samtools
bedtools
seqtk
```

## Run the full pipeline

Use placeholder paths in scripts and documentation, then supply actual local paths at runtime.

```bash
bash run_full_pipeline.sh \
  --project-root /path/to/output/project \
  --sample sample_name \
  --mode-name selected_cells \
  --loom /path/to/input.loom \
  --csv /path/to/selected_cells.csv \
  --raw-bam /path/to/raw.bam \
  --genome-fa /path/to/genome.fa \
  --genome-fai /path/to/genome.fa.fai \
  --genes-gtf /path/to/genes.gtf.gz \
  --tree /path/to/tree.newick \
  --threads 12
```

Optional arguments:

```bash
--chrom-sizes /path/to/chrom.sizes
--symlink-reference
--skip-fastq
--force
```

## Run step by step

### 1. Prepare selected-cell inputs

```bash
python scripts/00_prepare_inputs.py \
  --loom /path/to/input.loom \
  --csv /path/to/selected_cells.csv \
  --project-root /path/to/output/project \
  --sample sample_name \
  --mode-name selected_cells \
  --raw-bam /path/to/raw.bam \
  --genome-fa /path/to/genome.fa \
  --genome-fai /path/to/genome.fa.fai \
  --genes-gtf /path/to/genes.gtf.gz
```

Main outputs:

```text
/path/to/output/project/prepared/selected_cells/manifest.json
/path/to/output/project/prepared/selected_cells/barcode/barcodes.tsv
/path/to/output/project/prepared/selected_cells/barcode/sample_name.celllabels.txt
/path/to/output/project/prepared/selected_cells/metadata/barcode_mapping.tsv
/path/to/output/project/prepared/selected_cells/reference/
```

### 2. Subset the BAM and rewrite read names

```bash
python scripts/01_subset_bam_to_selected_cells.py \
  --manifest /path/to/output/project/prepared/selected_cells/manifest.json \
  --threads 12
```

Main outputs:

```text
/path/to/output/project/rawdata/bam/sample_name.cells_only.bam
/path/to/output/project/rawdata/bam/sample_name.cells_only.CBinReadName.bam
```

### 3. Generate one scReadSim replicate

```bash
python scripts/02_run_screadsim_fast.py \
  --manifest /path/to/output/project/prepared/selected_cells/manifest.json \
  --outdir /path/to/output/project/screadsim/rep01 \
  --sample sample_name \
  --n-cores 12 \
  --read-len 98 \
  --jitter-size 5 \
  --n-cell-new 2477 \
  --total-count-new 2500000
```

`02_run_screadsim_fast.py` includes a gene-only fallback. If the selected subset has an empty intergenic count matrix, it skips intergenic synthetic-count and BED-coordinate generation and continues with gene-only BED files.

### 4. Build the lineage scaffold

```bash
python scripts/03_build_lineage_scaffold.py \
  --tree /path/to/tree.newick \
  --subset-barcodes /path/to/output/project/prepared/selected_cells/barcode/barcodes.tsv \
  --subset-labels /path/to/output/project/prepared/selected_cells/barcode/sample_name.celllabels.txt \
  --outdir /path/to/output/project/lineage/rep01/truth_tree \
  --sample sample_name \
  --target-major-clades 8
```

### 5. Assign synthetic cells to donor lineages

```bash
python scripts/04_assign_synthetic_cells_to_donor_lineage.py \
  --donor-metadata /path/to/output/project/lineage/rep01/truth_tree/sample_name.donor_metadata.tsv \
  --synthetic-barcodes /path/to/output/project/screadsim/rep01/sample_name.synthetic_cell_barcode.txt \
  --synthetic-labels /path/to/output/project/screadsim/rep01/sample_name.gene.countmatrix.scDesign2Simulated.CellTypeLabel.txt \
  --outdir /path/to/output/project/lineage/rep01 \
  --sample sample_name \
  --seed 1001
```

## Run ten predefined replicates

After `00_prepare_inputs.py` and `01_subset_bam_to_selected_cells.py`, run:

```bash
bash run_replicates.sh \
  /path/to/output/project \
  sample_name \
  /path/to/tree.newick \
  12 \
  selected_cells
```

The predefined replicate settings are:

| Replicates | Synthetic cells | Total count | Read length | Jitter |
|---|---:|---:|---:|---:|
| rep01-rep02 | 2477 | 2500000 | 98 | 5 |
| rep03-rep04 | 2477 | 3500000 | 98 | 5 |
| rep05-rep06 | 2477 | 5000000 | 98 | 7 |
| rep07-rep08 | 3000 | 6000000 | 126 | 7 |
| rep09-rep10 | 3500 | 8000000 | 126 | 9 |

## Notes from code review

- No Chinese comments were present in the submitted scripts.
- The repository version removes hard-coded local paths from the runner scripts.
- The original BAM-subsetting step expected `manifest["bam"]["raw_bam"]`, but the prepare step did not write it. This version adds `--raw-bam` to `00_prepare_inputs.py` and records it in `manifest.json`.
- The original replicate runner used a sandbox-specific absolute path as the default script directory. This version resolves scripts relative to the repository root.
- The patched scReadSim script is used as the main `02_run_screadsim_fast.py`, so empty intergenic count matrices will not stop the pipeline.

## Expected repository layout

```text
simulation_generate_github/
├── README.md
├── environment.yml
├── requirements.txt
├── run_full_pipeline.sh
├── run_replicates.sh
├── config/
│   └── example_paths.env
└── scripts/
    ├── 00_prepare_inputs.py
    ├── 01_subset_bam_to_selected_cells.py
    ├── 02_run_screadsim_fast.py
    ├── 03_build_lineage_scaffold.py
    └── 04_assign_synthetic_cells_to_donor_lineage.py
```
