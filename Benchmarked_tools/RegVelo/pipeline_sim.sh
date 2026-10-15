#!/bin/bash
# =============================================================================
# RegVelo simulated-data pipeline
# =============================================================================
# Simulated datasets have no real species database, so pySCENIC is skipped and
# the a-priori GRN (scmultisim params / dyngen model) or a Spearman-correlation
# GRN is used instead, followed by RegVelo velocity estimation.
#
# Per sample:
#   RegVelo_sim.py    (prior GRN + RegVelo velocity)
#
# Usage:
#   bash pipeline_sim.sh [start_index] [end_index]
#     bash pipeline_sim.sh 0 10     # process sample index 0..9
#     bash pipeline_sim.sh          # process every sample
#
# Every path comes from an environment variable; the defaults point at the
# container data mount (/data/simdata).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------- Configuration -------------------------------
CONDA_ENV="${CONDA_ENV:-regvelo}"
CONDA_SH="${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
DATA_ROOT="${DATA_ROOT:-/data/simdata}"
INPUT_CSV="${INPUT_CSV:-${DATA_ROOT}/inputdata/base_siminput.csv}"
SAVE_ROOT="${SAVE_ROOT:-${DATA_ROOT}/regvelo_sim}"
# large-cell-count datasets are trained with mini-batches to avoid GPU OOM
BATCH_THRESHOLD="${BATCH_THRESHOLD:-10000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
MAX_EPOCHS="${MAX_EPOCHS:-150}"
CLUSTER_KEY="${CLUSTER_KEY:-vis_annotation}"
# samples whose id contains one of these keywords are skipped
SKIP_KEYWORDS=(${SKIP_KEYWORDS:-cell100000 cell50000})
# -----------------------------------------------------------------------------

# shellcheck source=/dev/null
source "${CONDA_SH}"

echo "============================================"
echo "  RegVelo simulated-data pipeline"
echo "  time        : $(date)"
echo "  script dir  : ${SCRIPT_DIR}"
echo "  input csv   : ${INPUT_CSV}"
echo "  save root   : ${SAVE_ROOT}"
echo "============================================"

mkdir -p "${SAVE_ROOT}"

START_IDX="${1:-}"
END_IDX="${2:-}"

# --- Build the sample list from the CSV -------------------------------------
# Columns used: ID, path and the cell count parsed from the file name.
PY_SAMPLES="$(mktemp "${TMPDIR:-/tmp}/regvelo_sim_samples_XXXXXX.py")"
trap 'rm -f "${PY_SAMPLES}"' EXIT
cat > "${PY_SAMPLES}" << 'PYEOF'
import os
import re
import sys

import pandas as pd

datalist = pd.read_csv(sys.argv[1])
for i, row in datalist.iterrows():
    name = os.path.basename(row['path'])
    match = re.search(r"cell(\d+)", name)
    ncells = match.group(1) if match else ""
    print(f"{i}|{row['ID']}|{name}|{row['path']}|{ncells}")
PYEOF

if ! SAMPLES="$(conda run -n "${CONDA_ENV}" python "${PY_SAMPLES}" "${INPUT_CSV}")"; then
    echo "[ERROR] failed to read ${INPUT_CSV}"
    exit 1
fi

IDX=0
while IFS='|' read -r CSV_IDX SAMPLE_ID DATA_FILE DATA_PATH NCELLS; do
    # index filter
    if [ -n "${START_IDX}" ] && [ "${IDX}" -lt "${START_IDX}" ]; then
        IDX=$((IDX + 1))
        continue
    fi
    if [ -n "${END_IDX}" ] && [ "${IDX}" -ge "${END_IDX}" ]; then
        break
    fi

    SAVE_DIR="${SAVE_ROOT}/${SAMPLE_ID}"
    STEM="${DATA_FILE%.h5ad}"
    STEM="${STEM%_dataset}"
    VELO_OUT="${SAVE_DIR}/${STEM}.h5ad"

    # skip by keyword (see SKIP_KEYWORDS above)
    SKIP_THIS=0
    for kw in "${SKIP_KEYWORDS[@]}"; do
        if [[ "${SAMPLE_ID}" == *"${kw}"* ]]; then
            SKIP_THIS=1
            break
        fi
    done
    if [ "${SKIP_THIS}" -eq 1 ]; then
        echo "[SKIP] ${SAMPLE_ID} matched a skip keyword."
        IDX=$((IDX + 1))
        continue
    fi

    # skip completed samples
    if [ -s "${VELO_OUT}" ]; then
        echo "[SKIP] ${SAMPLE_ID} already done."
        IDX=$((IDX + 1))
        continue
    fi

    echo ""
    echo "============================================================"
    echo " [${IDX}] processing: ${SAMPLE_ID}  |  ${DATA_FILE}"
    echo " data file : ${DATA_PATH}"
    echo " save dir  : ${SAVE_DIR}"
    echo " cells     : ${NCELLS}"
    echo "============================================================"

    mkdir -p "${SAVE_DIR}"

    # large datasets use mini-batch training to avoid GPU OOM
    BS_ARGS=()
    if [ -n "${NCELLS}" ] && [ "${NCELLS}" -ge "${BATCH_THRESHOLD}" ]; then
        BS_ARGS=(--batch-size "${BATCH_SIZE}")
        echo "[INFO] ${NCELLS} cells >= ${BATCH_THRESHOLD}; using batch_size=${BATCH_SIZE}"
    fi

    echo ""
    echo "[Step] prior GRN + RegVelo velocity ..."

    # PYTHONUNBUFFERED=1 streams print output in real time.
    conda run --no-capture-output -n "${CONDA_ENV}" \
        env PYTHONUNBUFFERED=1 python "${SCRIPT_DIR}/RegVelo_sim.py" \
        --input "${DATA_PATH}" \
        --output-dir "${SAVE_ROOT}" \
        --dataset-name "${SAMPLE_ID}" \
        --cluster-key "${CLUSTER_KEY}" \
        --max-epochs "${MAX_EPOCHS}" \
        "${BS_ARGS[@]}"

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
