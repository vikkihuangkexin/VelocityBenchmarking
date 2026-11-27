# Prepare peak BED files for real data

Use [`prepare_real_da_peaks.py`](prepare_real_da_peaks.py) to select differentially
accessible ATAC peaks and create a BED for the [scKINETICS runner](../README.md).
Provide an RNA H5AD with unique cell identifiers and labels in `obs` (default
`celltype`), along with raw ATAC peak counts. The script uses the RNA file for
cell identifiers and labels. UMAP coordinates and the `Mc` layer are optional
and unused.

## Installation

```bash
python -m pip install anndata numpy pandas scipy
```

## Input formats

Choose the input format based on how the peak counts, coordinates, and
barcodes are stored.

### Cell Ranger ARC matrix directory

Use the Cell Ranger ARC `filtered_feature_bc_matrix` directory containing
`matrix.mtx`, `features.tsv`, and `barcodes.tsv`. Plain and gzip-compressed
files are supported. The Matrix Market file contains integer counts with
features in rows and cells in columns. Only rows with feature
type `Peaks` are used, and the coordinates come from their ATAC feature
identifiers.

RNA cell identifiers are matched to the matrix barcodes by exact match first.
For unmatched 10x identifiers, the script recognizes prefixes such as
`sample:` and treats the velocyto suffix `x` as GEM group `-1`. A barcode
without a GEM suffix can match only when its sequence identifies one matrix
column. Ambiguous or duplicate matches raise an error.

Run the following from the repository root for data aligned to mm39.

```bash
python Benchmarked_tools/scKINETICS/real_peak_preprocessing/prepare_real_da_peaks.py \
  --input-format 10x \
  --input-h5ad data/multiome/rna.h5ad \
  --matrix-dir data/multiome/outs/filtered_feature_bc_matrix \
  --sample-name multiome_sample \
  --genome mm39 \
  --cluster-key celltype \
  --output-dir output/peaks
```

### Peak matrix and barcode map

Provide a peak-by-cell Matrix Market file with integer counts, a one-column
barcode list in matrix column order, and a headerless BED in matrix row
order. Supply a tab-separated one-to-one barcode map with `atac.bc` and
`rna.bc` columns. Plain and gzip-compressed inputs are accepted. Choose
`--input-format share-seq` for this layout, which is used in some SHARE-seq
deposits.

Check which barcode namespace each file uses. Some SHARE-seq datasets use
RNA barcodes as ATAC matrix column names. If the RNA H5AD stores ATAC
barcodes in `obs_names` and the matrix columns use RNA barcodes, set
`--rna-id-kind atac --matrix-id-kind rna`. Both options default to `rna`.
Period- and comma-delimited SHARE-seq barcode forms are supported. Labels
come from the specified RNA `obs` column.

```bash
python Benchmarked_tools/scKINETICS/real_peak_preprocessing/prepare_real_da_peaks.py \
  --input-format share-seq \
  --input-h5ad data/paired/rna.h5ad \
  --atac-counts data/paired/peak_counts.mtx.gz \
  --atac-barcodes data/paired/matrix_barcodes.tsv.gz \
  --atac-peaks data/paired/peaks.bed.gz \
  --barcode-map data/paired/barcode_map.tsv.gz \
  --rna-id-kind atac \
  --matrix-id-kind rna \
  --sample-name paired_sample \
  --genome mm10 \
  --cluster-key celltype \
  --output-dir output/peaks
```

For either input format, the script uses one-to-one matched cells. By default,
every RNA cell must have a matching ATAC matrix column. Use
`--allow-partial-overlap` to analyze the intersection when the difference in
cell sets is expected. Check the cell alignment table before using the peaks.
Set `--genome` to the input assembly. The script records this label and
retains the original coordinates. Use the same assembly in scKINETICS,
whose benchmark runner supports `mm10`, `mm39`, and `hg38`.

## Batch processing

Copy [`inputs.example.csv`](inputs.example.csv) to `peak_inputs.csv` and replace
the example paths. Input paths are relative to the CSV's location. Use a
unique `sample_name` in each row and leave fields for the other input format
empty. All samples write to the supplied output directory, with filenames
prefixed by `sample_name`.

```bash
python Benchmarked_tools/scKINETICS/real_peak_preprocessing/prepare_real_da_peaks.py \
  --metadata-file peak_inputs.csv \
  --output-dir output/peaks
```

## Peak selection

Use the final cell-type or cell-state annotations for the states you intend to
model. For each label, the script compares cells with that label against the
other matched cells. A peak is
open in a cell if its raw ATAC count is greater than zero. The one-sided
hypergeometric enrichment test is equivalent to a one-sided Fisher exact test
for greater accessibility. Benjamini–Hochberg correction is applied across
the tested peaks within each comparison.

Defaults retain peaks with width greater than zero and strictly below 2,000 bp,
adjusted `p <= 0.05`, and `log2` open-fraction ratio at least 1. That ratio uses
a pseudocount of 0.5 added to both the open and closed counts in each group.
The script ranks qualifying peaks by
`-log10(max(adjusted p, 1e-300)) * log2 fold change` and retains up to 1,000 per
group. Ties favor smaller adjusted p-values, then larger fold changes, then
the input peak order. Adjust these settings with `--max-peak-width`, `--padj-cutoff`,
`--log2fc-cutoff`, `--pseudocount`, and `--top-n-per-cluster`. In batch mode,
the settings apply to all rows.

The [scKINETICS pancreas tutorial](https://github.com/dpeerlab/scKINETICS/blob/main/Demo_Pancreas_Regeneration.ipynb)
uses peaks selected with DESeq2 from bulk ATAC-seq and lists Fisher's exact
test as an option for scATAC-seq. Here the test ranks candidate peaks for
annotated cell states within each dataset. Sample-level pseudobulk inference
requires independent biological replicates.

## Outputs

Each filename starts with `<sample_name>_sckinetics_da_`.

| Suffix | Contents |
| --- | --- |
| `peaks.bed` | Three-column, headerless BED with the union of selected peaks, deduplicated across labels |
| `peaks_by_cluster.tsv` | Selected coordinates and labels |
| `peak_table.tsv` | Statistics for the selected peaks |
| `peak_summary.tsv` | Cell counts and numbers of qualifying and retained peaks per label |
| `cell_alignment.tsv` | Original RNA H5AD `obs_names`, labels, match status, and corresponding ATAC matrix barcodes |
| `input_audit.json` | Input, alignment, and filtering summary |

Pass the union BED to `scKINETICS.py --peaks-bed` or use it as the `peaks_bed`
value in the runner's input CSV. The public runner uses the union peak set for
all groups. If no peaks pass the filters, the script reports an error.
