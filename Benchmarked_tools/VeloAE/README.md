# VeloAE Script

## Installation

```bash
git clone https://github.com/qiaochen/VeloAE.git && \
cd VeloAE && \
conda install python=3.7 && \
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --retries 10 --timeout 300 && \
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html && \
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html && \
pip install torch-geometric && \
pip install .
```

## Purpose
- Helper script to run the VeloAE `veloproj` command for training/projecting a VeloAE model on an AnnData dataset.

Notes & Reference
- This repository includes a small runner `veloae.sh` that invokes `veloproj` with recommended defaults.
- Upstream project: https://github.com/qiaochen/VeloAE (refer there for full model details and options).

Dependencies
- `veloproj` must be installed and available on PATH (from VeloAE project or package providing the binary).
- CUDA (optional): script defaults to `--device cuda:0`; change to CPU if needed.

Inputs
- `input.h5ad`: AnnData file containing the dataset (must include layers/obs expected by VeloAE).
- `output_dir`: directory where model and outputs will be saved.
- `celltype_key`: obs column used as `vis_type_col` for visualization/conditioning.
- `vis_key`: embedding key used for visualization (e.g., `umap`).

Outputs
- Model checkpoint: `<output_dir>/<input_prefix>_model.cpt`
- Other outputs written to `<output_dir>` as produced by `veloproj`.

Usage

```bash
bash veloae.sh /path/to/input.h5ad ./example/output/veloae cell_type umap
```

Parameters used by the script
- Learning rate: `--lr 1e-5`
- Gumbel-softmax temperature: `--gumbsoft_tau 5`
- Number of scVelo jobs: `--scv_n_jobs 64`
- Device: `--device cuda:0` (change to CPU if you don't have GPU)
- Epochs: `--n-epochs 10000`

Customisation
- Edit `veloae.sh` to adjust `veloproj` parameters (learning rate, epochs, device, etc.).
- For reproducible runs, set seeds in the model's configuration or in `veloproj` arguments if supported.
