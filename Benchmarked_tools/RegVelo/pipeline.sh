#!/bin/bash
# =============================================================================
# RegVelo real-data pipeline
# =============================================================================
# Per sample:
#   1. h5ad -> loom                          (h5ad_to_loom.py)
#   2. pySCENIC grn -> ctx -> aucell         (pyscenic CLI)
#   3. GRN parquet + RegVelo velocity        (run_pyscenic_grn_regvelo.py)
#
# Usage:
#   bash pipeline.sh [start_index] [end_index] [run_ids]
#     bash pipeline.sh 0 10            # process sample index 0..9
#     bash pipeline.sh                 # process every sample
#     bash pipeline.sh "" "" "4,5,13"  # process only these sample ids
#
# Every path comes from an environment variable; the defaults point at the
# container data mount (/data/simdata) so the script also runs unmodified
# inside the Docker image. Override them for a local run, e.g.
#   DATA_ROOT=/path/to/data INPUT_CSV=/path/to/samples.csv bash pipeline.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------- Configuration -------------------------------
CONDA_ENV="${CONDA_ENV:-regvelo}"
PYSCENIC_ENV="${PYSCENIC_ENV:-${CONDA_ENV}}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
DATA_ROOT="${DATA_ROOT:-/data/simdata}"
INPUT_CSV="${INPUT_CSV:-${DATA_ROOT}/inputdata/base_realinput.csv}"
SAVE_ROOT="${SAVE_ROOT:-${DATA_ROOT}/regvelo}"
AUX_DIR="${AUX_DIR:-${DATA_ROOT}/regvelo_aux}"
N_WORKERS="${N_WORKERS:-40}"
# -----------------------------------------------------------------------------

# conda is a shell function defined by conda.sh, not a stand-alone executable;
# it must be sourced in this shell.
# shellcheck source=/dev/null
source "${CONDA_SH}"

echo "============================================"
echo "  RegVelo real-data pipeline"
echo "  time          : $(date)"
echo "  script dir    : ${SCRIPT_DIR}"
echo "  input csv     : ${INPUT_CSV}"
echo "  save root     : ${SAVE_ROOT}"
echo "  aux dir       : ${AUX_DIR}"
echo "============================================"

START_IDX="${1:-}"
END_IDX="${2:-}"
RUN_IDS="${3:-${RUN_IDS:-}}"

mkdir -p "${SAVE_ROOT}"

# --- Build the sample list from the CSV -------------------------------------
# Written to a temporary file because `conda run` mangles quoting in inline
# scripts. Columns used: ID and path; the file name is taken from the path.
PY_SAMPLES="$(mktemp "${TMPDIR:-/tmp}/regvelo_samples_XXXXXX.py")"
trap 'rm -f "${PY_SAMPLES}"' EXIT
cat > "${PY_SAMPLES}" << 'PYEOF'
import os
import sys

import pandas as pd

datalist = pd.read_csv(sys.argv[1])
for i, row in datalist.iterrows():
    print(f"{i}|{row['ID']}|{os.path.basename(row['path'])}|{row['path']}")
PYEOF

if ! SAMPLES="$(conda run -n "${CONDA_ENV}" python "${PY_SAMPLES}" "${INPUT_CSV}")"; then
    echo "[ERROR] failed to read ${INPUT_CSV}"
    exit 1
fi

IDX=0
while IFS='|' read -r CSV_IDX SAMPLE_ID DATA_FILE DATA_PATH; do
    # index filter
    if [ -n "${START_IDX}" ] && [ "${IDX}" -lt "${START_IDX}" ]; then
        IDX=$((IDX + 1))
        continue
    fi
    if [ -n "${END_IDX}" ] && [ "${IDX}" -ge "${END_IDX}" ]; then
        break
    fi

    # id-list filter (only active when RUN_IDS is set)
    if [ -n "${RUN_IDS}" ] && ! echo ",${RUN_IDS}," | grep -q ",${SAMPLE_ID},"; then
        IDX=$((IDX + 1))
        continue
    fi

    SAVE_DIR="${SAVE_ROOT}/${SAMPLE_ID}"
    # the wrapper writes <stem>.h5ad where the benchmark `_dataset` suffix is stripped
    STEM="${DATA_FILE%.h5ad}"
    STEM="${STEM%_dataset}"
    VELO_OUT="${SAVE_DIR}/${STEM}.h5ad"

    # skip completed samples (the final velocity file is the marker)
    if [ -s "${VELO_OUT}" ]; then
        echo "[SKIP] ${SAMPLE_ID} already done."
        IDX=$((IDX + 1))
        continue
    fi

    echo ""
    echo "============================================================"
    echo " [${IDX}] processing: ${SAMPLE_ID}  |  ${DATA_FILE}"
    echo " data file  : ${DATA_PATH}"
    echo " save dir   : ${SAVE_DIR}"
    echo "============================================================"

    mkdir -p "${SAVE_DIR}"

    # ================================================================
    # Step 1: h5ad -> loom
    # ================================================================
    echo "[Step 1] h5ad -> loom ..."
    LOOM_PATH="${SAVE_DIR}/adata.loom"
    if [ -s "${LOOM_PATH}" ]; then
        echo "  adata.loom already exists, skipping conversion."
    else
        if ! conda run --no-capture-output -n "${CONDA_ENV}" python "${SCRIPT_DIR}/h5ad_to_loom.py" \
                --input "${DATA_PATH}" \
                --output-dir "${SAVE_DIR}" \
                --dataset-name "${DATA_FILE}"; then
            echo "[ERROR] loom conversion failed for ${SAMPLE_ID}, skipping."
            IDX=$((IDX + 1))
            continue
        fi
    fi

    # species -> pySCENIC resources (species.txt is written by h5ad_to_loom.py)
    SPECIES="$(cat "${SAVE_DIR}/species.txt" 2>/dev/null || true)"
    case "${SPECIES}" in
        mm)
            TF_LIST="allTFs_mm.txt"
            DB_PREFIX="mm10"
            MOTIF_FILE="motifs-v9-nr.mgi-m0.001-o0.0.tbl"
            ;;
        hs)
            TF_LIST="allTFs_hg38.txt"
            DB_PREFIX="hg38"
            MOTIF_FILE="motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
            ;;
        *)
            echo "[ERROR] unknown species '${SPECIES}' for ${SAMPLE_ID}, skipping."
            IDX=$((IDX + 1))
            continue
            ;;
    esac
    echo "  species: ${SPECIES}  |  TF list: ${TF_LIST}  |  ranking db: ${DB_PREFIX}"

    TF_PATH="${AUX_DIR}/${TF_LIST}"
    MOTIF_PATH="${AUX_DIR}/${MOTIF_FILE}"

    ADJ_PATH="${SAVE_DIR}/adj.csv"
    REG_PATH="${SAVE_DIR}/reg.csv"
    AUC_OUT="${SAVE_DIR}/pyscenic_output.loom"

    # ================================================================
    # Step 2: pySCENIC grn -> ctx -> aucell
    # ================================================================
    echo ""
    echo "[Step 2] running pySCENIC ..."

    # --- grn ---
    echo "  [2a] pySCENIC grn ..."
    if [ -s "${ADJ_PATH}" ]; then
        echo "    adj.csv already exists, skipping."
    else
        conda run --no-capture-output -n "${PYSCENIC_ENV}" pyscenic grn \
            "${LOOM_PATH}" \
            "${TF_PATH}" \
            -o "${ADJ_PATH}" \
            --num_workers "${N_WORKERS}"
        echo "  [2a] done -> ${ADJ_PATH}"
    fi

    # --- ctx ---
    echo "  [2b] pySCENIC ctx ..."
    if [ -s "${REG_PATH}" ]; then
        echo "    reg.csv already exists, skipping."
    else
        DB_FILES="$(ls "${AUX_DIR}/${DB_PREFIX}"*.feather 2>/dev/null | tr '\n' ' ' || true)"
        if [ -z "${DB_FILES}" ]; then
            echo "[ERROR] no ranking database found at ${AUX_DIR}/${DB_PREFIX}*.feather"
            echo "  download the pySCENIC ranking databases into ${AUX_DIR}/"
            IDX=$((IDX + 1))
            continue
        fi
        # shellcheck disable=SC2086 - DB_FILES is a space-separated file list
        conda run --no-capture-output -n "${PYSCENIC_ENV}" pyscenic ctx \
            "${ADJ_PATH}" \
            ${DB_FILES} \
            --annotations_fname "${MOTIF_PATH}" \
            --expression_mtx_fname "${LOOM_PATH}" \
            --output "${REG_PATH}" \
            --all_modules \
            --num_workers "${N_WORKERS}"
        echo "  [2b] done -> ${REG_PATH}"
    fi

    # --- aucell ---
    echo "  [2c] pySCENIC aucell ..."
    if [ -s "${AUC_OUT}" ]; then
        echo "    pyscenic_output.loom already exists, skipping."
    else
        conda run --no-capture-output -n "${PYSCENIC_ENV}" pyscenic aucell \
            "${LOOM_PATH}" \
            "${REG_PATH}" \
            --output "${AUC_OUT}" \
            --num_workers "${N_WORKERS}"
        echo "  [2c] done -> ${AUC_OUT}"
    fi

    # ================================================================
    # Step 3: GRN parquet + RegVelo velocity
    # ================================================================
    echo ""
    echo "[Step 3] GRN parquet + RegVelo velocity ..."

    if [ ! -s "${AUC_OUT}" ]; then
        echo "[SKIP] ${SAMPLE_ID} has no valid pyscenic_output.loom, skipping Step 3."
        IDX=$((IDX + 1))
        continue
    fi

    conda run --no-capture-output -n "${CONDA_ENV}" python "${SCRIPT_DIR}/run_pyscenic_grn_regvelo.py" \
        --input "${DATA_PATH}" \
        --output-dir "${SAVE_ROOT}" \
        --dataset-name "${SAMPLE_ID}" \
        --cluster-key "${CLUSTER_KEY:-vis_annotation}" \
        --pyscenic-loom "${AUC_OUT}"

    if [ -f "${VELO_OUT}" ]; then
        echo "[DONE] ${SAMPLE_ID} finished."
    else
        echo "[WARN] ${SAMPLE_ID} may not have completed; check the logs."
    fi

    IDX=$((IDX + 1))
done <<< "${SAMPLES}"

echo ""
echo "============================================================"
echo "  pipeline finished"
echo "  time: $(date)"
echo "============================================================"
