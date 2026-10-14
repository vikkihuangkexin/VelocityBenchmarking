#!/usr/bin/env bash
# Usage: bash Figure_code/Fig3/run_all.sh [arguments passed to run.py]
set -euo pipefail
project_dir="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "${project_dir}/run.py" "$@"
