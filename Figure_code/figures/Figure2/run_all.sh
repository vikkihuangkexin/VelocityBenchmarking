#!/usr/bin/env bash
# Usage: bash run_all.sh [new_output_directory] [Rscript_executable]
set -euo pipefail
project_dir="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "${project_dir}/run.py" --output "${1:-${project_dir}/results}" --rscript "${2:-Rscript}"
