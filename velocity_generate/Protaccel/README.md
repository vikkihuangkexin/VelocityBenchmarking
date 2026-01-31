# ProtAccel Analysis

This script performs single-cell protein and RNA velocity analysis using ProtAccel, generating corresponding plots based on existing velocity data.

## Required adata Internal Variables

- **Loom file layers**: spliced, unspliced
- **Protein matrix CSV**: Corresponding protein expression matrix

## Output Variables/Content

- Various SVG plot files in `save_dir/fig/`, including phase portraits, velocity projections, grid arrows, and combined plots.

## Usage

```bash
python protaccel.py --protein_dir /data/protein_matrix.csv --loom_dir /data/test.loom --save_dir /data/protaccel_results
```

For detailed information, see [ProtAccel GitHub](https://github.com/pachterlab/GSP_2019/tree/master).