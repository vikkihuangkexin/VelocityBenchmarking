#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_replicates.sh <project_root> <sample> <newick_tree> [threads] [mode_name] [extra_scReadSim_args...]

Example:
  bash run_replicates.sh /path/to/project sample_name /path/to/tree.newick 12 selected_cells --skip-fastq
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PROJECT_ROOT="${1:?Missing project root}"
SAMPLE="${2:-sample}"
TREE="${3:?Missing Newick tree path}"
THREADS="${4:-12}"
MODE_BASE="${5:-selected_cells}"
shift $(( $# >= 5 ? 5 : $# )) || true
EXTRA_SCREADSIM_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${REPO_ROOT}/scripts"

PREP="${PROJECT_ROOT}/prepared/${MODE_BASE}"
MANIFEST="${PREP}/manifest.json"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  echo "Run scripts/00_prepare_inputs.py first, or use run_full_pipeline.sh." >&2
  exit 1
fi

for i in $(seq -w 1 10); do
  MODE="rep${i}"
  OUT_SCR="${PROJECT_ROOT}/screadsim/${MODE}"
  OUT_LIN="${PROJECT_ROOT}/lineage/${MODE}"

  case "${i}" in
    01|02) NNEW=2477; TOTAL=2500000; READLEN=98;  JITTER=5 ;;
    03|04) NNEW=2477; TOTAL=3500000; READLEN=98;  JITTER=5 ;;
    05|06) NNEW=2477; TOTAL=5000000; READLEN=98;  JITTER=7 ;;
    07|08) NNEW=3000; TOTAL=6000000; READLEN=126; JITTER=7 ;;
    09|10) NNEW=3500; TOTAL=8000000; READLEN=126; JITTER=9 ;;
  esac

  python "${SCRIPT_DIR}/02_run_screadsim_fast.py" \
    --manifest "${MANIFEST}" \
    --outdir "${OUT_SCR}" \
    --sample "${SAMPLE}" \
    --n-cores "${THREADS}" \
    --read-len "${READLEN}" \
    --jitter-size "${JITTER}" \
    --n-cell-new "${NNEW}" \
    --total-count-new "${TOTAL}" \
    "${EXTRA_SCREADSIM_ARGS[@]}"

  python "${SCRIPT_DIR}/03_build_lineage_scaffold.py" \
    --tree "${TREE}" \
    --subset-barcodes "${PREP}/barcode/barcodes.tsv" \
    --subset-labels "${PREP}/barcode/${SAMPLE}.celllabels.txt" \
    --outdir "${OUT_LIN}/truth_tree" \
    --sample "${SAMPLE}" \
    --target-major-clades 8

  python "${SCRIPT_DIR}/04_assign_synthetic_cells_to_donor_lineage.py" \
    --donor-metadata "${OUT_LIN}/truth_tree/${SAMPLE}.donor_metadata.tsv" \
    --synthetic-barcodes "${OUT_SCR}/${SAMPLE}.synthetic_cell_barcode.txt" \
    --synthetic-labels "${OUT_SCR}/${SAMPLE}.gene.countmatrix.scDesign2Simulated.CellTypeLabel.txt" \
    --outdir "${OUT_LIN}" \
    --sample "${SAMPLE}" \
    --seed $((1000 + 10#${i}))
done
