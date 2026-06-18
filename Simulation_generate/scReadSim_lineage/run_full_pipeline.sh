#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_full_pipeline.sh \
    --project-root /path/to/project \
    --sample sample_name \
    --mode-name selected_cells \
    --loom /path/to/input.loom \
    --csv /path/to/selected_cells.csv \
    --raw-bam /path/to/raw.bam \
    --genome-fa /path/to/genome.fa \
    --genome-fai /path/to/genome.fa.fai \
    --genes-gtf /path/to/genes.gtf[.gz] \
    --tree /path/to/tree.newick \
    --threads 12

Optional:
  --chrom-sizes /path/to/chrom.sizes
  --symlink-reference
  --skip-fastq
  --force
EOF
}

PROJECT_ROOT=""
SAMPLE="sample"
MODE_NAME="selected_cells"
LOOM=""
CSV=""
RAW_BAM=""
GENOME_FA=""
GENOME_FAI=""
GENES_GTF=""
CHROM_SIZES=""
TREE=""
THREADS="12"
SYMLINK_REFERENCE="false"
SKIP_FASTQ="false"
FORCE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --sample) SAMPLE="$2"; shift 2 ;;
    --mode-name) MODE_NAME="$2"; shift 2 ;;
    --loom) LOOM="$2"; shift 2 ;;
    --csv) CSV="$2"; shift 2 ;;
    --raw-bam) RAW_BAM="$2"; shift 2 ;;
    --genome-fa) GENOME_FA="$2"; shift 2 ;;
    --genome-fai) GENOME_FAI="$2"; shift 2 ;;
    --genes-gtf) GENES_GTF="$2"; shift 2 ;;
    --chrom-sizes) CHROM_SIZES="$2"; shift 2 ;;
    --tree) TREE="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --symlink-reference) SYMLINK_REFERENCE="true"; shift ;;
    --skip-fastq) SKIP_FASTQ="true"; shift ;;
    --force) FORCE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

required=(PROJECT_ROOT LOOM CSV RAW_BAM GENOME_FA GENOME_FAI GENES_GTF TREE)
for var in "${required[@]}"; do
  if [[ -z "${!var}" ]]; then
    echo "Missing required argument: --$(echo "${var}" | tr '[:upper:]_' '[:lower:]-')" >&2
    usage
    exit 1
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${REPO_ROOT}/scripts"

PREPARE_ARGS=(
  --loom "${LOOM}"
  --csv "${CSV}"
  --project-root "${PROJECT_ROOT}"
  --sample "${SAMPLE}"
  --mode-name "${MODE_NAME}"
  --raw-bam "${RAW_BAM}"
  --genome-fa "${GENOME_FA}"
  --genome-fai "${GENOME_FAI}"
  --genes-gtf "${GENES_GTF}"
)

if [[ -n "${CHROM_SIZES}" ]]; then
  PREPARE_ARGS+=(--chrom-sizes "${CHROM_SIZES}")
fi
if [[ "${SYMLINK_REFERENCE}" == "true" ]]; then
  PREPARE_ARGS+=(--symlink-reference)
fi
if [[ "${FORCE}" == "true" ]]; then
  PREPARE_ARGS+=(--force)
fi

python "${SCRIPT_DIR}/00_prepare_inputs.py" "${PREPARE_ARGS[@]}"
python "${SCRIPT_DIR}/01_subset_bam_to_selected_cells.py" \
  --manifest "${PROJECT_ROOT}/prepared/${MODE_NAME}/manifest.json" \
  --threads "${THREADS}"

EXTRA_ARGS=()
if [[ "${SKIP_FASTQ}" == "true" ]]; then
  EXTRA_ARGS+=(--skip-fastq)
fi

bash "${REPO_ROOT}/run_replicates.sh" \
  "${PROJECT_ROOT}" \
  "${SAMPLE}" \
  "${TREE}" \
  "${THREADS}" \
  "${MODE_NAME}" \
  "${EXTRA_ARGS[@]}"
