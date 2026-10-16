# veloAgent

This independent spatial entry uses the [team veloAgent source fork](https://github.com/sd68515/veloAgent_benchmark).
The fork corrects the order of distances and indices returned by scikit-learn's
radius-neighbor search in `CellModel`; this can change the inferred velocity.
Install compatible dependencies, then install the fork:

```bash
git clone https://github.com/sd68515/veloAgent_benchmark.git ../veloAgent_benchmark
python -m pip install --no-deps -e ../veloAgent_benchmark
```

## Inputs and run

Commands below run from the repository root. Each input H5AD needs `spliced` and `unspliced` layers, an annotation column in
`obs`, and finite two-dimensional spatial coordinates. The fallback coordinate
key defaults to `obsm['spatial']`; set `--coordinate-key` or CSV `dimred_key` for
another key.

`obsm['X_spatial']` is also supplied for scVelo's `basis='spatial'` interface:
the script copies `obsm['spatial']` when this alias is missing, including in
the saved result.

Download the matching
`protein.aliases`, `protein.info`, and standard three-column `protein.links`
files yourself from
[STRING downloads](https://string-db.org/cgi/download). Decompress `.gz`
files before passing them to this script; use matching species/version files. This study used STRING
v12.0 files for human, mouse, and chicken (*Gallus gallus*). The script never
chooses a species from a dataset ID. For the three *Ambystoma mexicanum*
datasets `64_Am_TelencephalonS44`, `65_Am_TelencephalonS54`, and
`66_Am_TelencephalonS57`, gene names were mapped to human homologs and the
human STRING files were used.

```bash
python Benchmarked_tools/veloAgent/veloAgent.py \
  --input data/tissue.h5ad --dataset-id tissue \
  --annotation-key cell_type \
  --protein-info data/9606.protein.info.v12.0.txt \
  --protein-aliases data/9606.protein.aliases.v12.0.txt \
  --protein-links data/9606.protein.links.v12.0.txt \
  --output-dir results/veloAgent
```

For several inputs, use a CSV. Required columns are
`dataset_name,file_path,modality,cluster_key,protein_info,protein_aliases,protein_links`;
`modality` must be `spatial`. `dimred_key` is optional and defaults to `spatial`.
Relative paths resolve from the CSV directory:

```csv
dataset_name,file_path,modality,cluster_key,dimred_key,protein_info,protein_aliases,protein_links,homolog_key,radius_reference,nbr_radius
tissue,data/tissue.h5ad,spatial,cell_type,,data/protein.info.txt,data/protein.aliases.txt,data/protein.links.txt,,,
```

```bash
python Benchmarked_tools/veloAgent/veloAgent.py \
  --metadata-file datasets.csv --output-dir results/veloAgent
```

## STRING mapping and radius options

The optional `homolog_key` names a `var` column with human homolog symbols
marked `[hs]`; leave it empty when the gene names themselves match STRING
symbols. Protein info symbols are preferred over aliases, and ambiguous
matches choose a stable protein ID. Unmapped genes are removed. Use the standard
STRING `protein.links` download: by default, each undirected interaction is
listed in both directions. The script builds a symmetric prior from these links.
If using a de-duplicated STRING file or another interaction source, confirm that
its records represent undirected protein interactions; directional interaction
data are not suitable for this prior. After optional score filtering, mapped genes
with no retained interactions are connected to all mapped genes, consistent with
the benchmark prior. `--confidence-threshold` optionally filters links by STRING
combined score; by default no score threshold is applied.

The paper specifies a radius giving about 15-20 spatial neighbors per cell.
The script targets the midpoint, 17.5, excluding each cell itself. It uses
`obs[['x_raw', 'y_raw']]` when both columns exist; otherwise it reads the
declared `dimred_key` from `obsm`. A KD-tree and binary search find the radius
whose mean neighbor count is closest to 17.5. To reproduce a shared
parent-tissue radius, provide
`radius_reference` (a parent-tissue H5AD with the same coordinate convention), or
provide an explicit positive `nbr_radius`. These options are mutually exclusive.
For a single input, use `--radius-reference` or `--nbr-radius`.

## Outputs and training settings

The entry uses seed 618, 2,000 initial genes, a STRING prior, VAE fitting,
gene-network training, neighbor refinement, and the spatial agent model.
`--device` defaults to CPU; `--threads` defaults to at most 16. Each dataset
writes `<output-dir>/<dataset_id>.h5ad` with `layers['velocity']`,
`obsm['spatial']`, the learned `obsm['cell_embed']`, and run
settings in `uns['veloagent_run']`. Existing outputs require `--overwrite`. The script does not export a
projected velocity or scVelo pseudotime; metrics needing these fields must
build them separately. Automatic radius selection needs at least 19 cells.
`--nbr-radius` bypasses that radius search for smaller inputs, subject to the
upstream model's own cell-count requirements.
