#!/bin/bash
set -euo pipefail

# ------------------------
# Usage check
# ------------------------
if [ "$#" -lt 4 ]; then
    echo "Usage: bash veloae.sh <input.h5ad> <output_dir> <celltype_key> <vis_key>"
    exit 1
fi

# ------------------------
# Parse positional arguments
# ------------------------
INPUT_ADATA="$1"
OUTPUT_DIR="$2"
CELLTYPE_KEY="$3"
VIS_KEY="$4"

# ------------------------
# Derive filename prefix
# ------------------------
FILENAME=$(basename "$INPUT_ADATA")
PREFIX="${FILENAME%.h5ad}"

MODEL_NAME="${OUTPUT_DIR}/${PREFIX}_model.cpt"

mkdir -p "$OUTPUT_DIR"

# ------------------------
# Run veloproj with recommended defaults
# ------------------------
veloproj \
    --lr 1e-5 \
    --nb_g_src X \
    --gumbsoft_tau 5 \
    --fit_offset_pred true \
    --vis_type_col "$CELLTYPE_KEY" \
    --vis-key "$VIS_KEY" \
    --scv_n_jobs 64 \
    --refit true \
    --adata "$INPUT_ADATA" \
    --device cuda:0 \
    --model-name "$MODEL_NAME" \
    --output "$OUTPUT_DIR" \
    --n-epochs 10000