# TopicVelo Script
=========

Purpose
- Compute topic-based velocity using a three-step pipeline involving data preprocessing, topic modeling with fastTopics (R), and velocity inference.

Workflow Overview
1. **Step 1 (topic_velo_step1.py)**: Preprocess AnnData and export spliced/unspliced counts to matrix market format for topic modeling.
2. **Step 2 (topic_velo_step2.R)**: Run fastTopics to fit topic models and perform differential expression analysis.
3. **Step 3 (topic_velo_step3.py)**: Integrate topic assignments, compute velocity, and save results.

Inputs
- AnnData `.h5ad` file with `layers['spliced']` and `layers['unspliced']`.
- For Step 1: `--data_dir` (default: './example')
- For Step 2: `--counts_mtx`, `--genes_csv` (outputs from Step 1)
- For Step 3: `--adata_path`, `--step2_dir`, `--input_dir` (Step 1 outputs and Step 2 dir)

Outputs
- Step 1: `{num_id}_scNT_HH_filtered_SU_Counts.mtx`, `{num_id}_scNT_HH_filtered_SU_Genes_names.csv`, `{num_id}_scNT_HH_filtered_SU_Cells_names.csv` in `--save_dir`
- Step 2: Various CSV files (CellWeights, de_postmean, de_lfsr, etc.) and RDS files in `--save_dir`
- Step 3: `{sample_id}_topicvelo.h5ad` in `--save_dir/{sample_id}`

CLI Options
- **Step 1**:
  - `--data_dir`: input .h5ad file (default: './example')
  - `--save_dir`: output directory (default: './example/output/topic-velo')
- **Step 2**:
  - `--counts_mtx`: path to Counts.mtx
  - `--genes_csv`: path to Genes_names.csv
  - `--save_dir`: output directory
  - `--K`: number of topics (default: 8)
- **Step 3**:
  - `--adata_path`: input .h5ad file
  - `--step2_dir`: directory with Step 2 outputs
  - `--input_dir`: directory with Step 1 outputs
  - `--save_dir`: output directory

Notes
- Ensure R and fastTopics are installed for Step 2.
- The pipeline assumes specific file naming conventions for intermediate files.
- Default paths are set for example usage; adjust for actual data.