#!/bin/bash

set -euo pipefail

# ================= Configuration =================
DATA_CSV=""
RESULTS_CSV=""
TARGET_SCRIPT=""
TARGET_METHOD=""
SAVE_DIR=""
ENABLE_GPU=0
SHOW_PYTHON_LOGS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-csv)
      DATA_CSV="$2"
      shift 2
      ;;
    --results-csv)
      RESULTS_CSV="$2"
      shift 2
      ;;
    --target-script)
      TARGET_SCRIPT="$2"
      shift 2
      ;;
    --target-method)
      TARGET_METHOD="$2"
      shift 2
      ;;
    --save-dir)
      SAVE_DIR="$2"
      shift 2
      ;;
    --enable-gpu)
      ENABLE_GPU=1
      shift
      ;;
    --show-logs)
      SHOW_PYTHON_LOGS=1
      shift
      ;;
    --scalene-args)
      SCALENE_ARGS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check required parameters
if [[ -z "$DATA_CSV" || -z "$RESULTS_CSV" || -z "$TARGET_SCRIPT" || -z "$TARGET_METHOD" || -z "$SAVE_DIR" ]]; then
  echo "Usage: $0 --data-csv <path> --results-csv <path> --target-script <path> --target-method <method> --save-dir <path> [--enable-gpu] [--show-logs] [--scalene-args <args>]"
  exit 1
fi

# --- Sub-save paths ---
METHOD_SAVE_DIR="$SAVE_DIR/$TARGET_METHOD"
mkdir -p "$METHOD_SAVE_DIR"

INPUT_DATASETS=()

# Default Scalene args if not provided
SCALENE_ARGS="${SCALENE_ARGS:-"--memory --malloc-threshold 1048576 --profile-only unitvelo_sim,unitvelo,scvelo,scanpy,tensorflow"}"

# ================= Load successful business records =================
declare -A SUCCESSFUL_BUSINESS_TASKS

if [ -f "$RESULTS_CSV" ]; then
    echo "Loading analysis success records ($RESULTS_CSV)..."

    while IFS='|' read -r method id result_path; do
        if [ -n "$result_path" ] && [ -n "$method" ] && [ -n "$id" ]; then
            SUCCESSFUL_BUSINESS_TASKS["${method}_${id}"]=1
        fi
    done < <(awk -F',' '
        NR==1 {
            for(i=1; i<=NF; i++) {
                col=$i; gsub(/^[ \t"]+|[ \t"\r]+$/, "", col)
                if(col == "method") method_idx=i
                if(col == "ID") id_idx=i
                if(col == "result_path") path_idx=i
            }
        }
        NR>1 {
            if(method_idx > 0 && id_idx > 0 && path_idx > 0) {
                m_val=$method_idx; id_val=$id_idx; p_val=$path_idx
                gsub(/^[ \t"]+|[ \t"\r]+$/, "", m_val)
                gsub(/^[ \t"]+|[ \t"\r]+$/, "", id_val)
                gsub(/^[ \t"]+|[ \t"\r]+$/, "", p_val)
                if(p_val != "") print m_val "|" id_val "|" p_val
            }
        }
    ' "$RESULTS_CSV")
else
    echo "Warning: Business result record not found ($RESULTS_CSV), will perform full analysis."
fi

# ================= Task execution and cleanup functions =================

cleanup_processes() {
    if [ -n "${TAIL_PID:-}" ]; then
        kill "$TAIL_PID" 2>/dev/null || true
    fi

    if [ -n "${ACTIVE_SCALENE_PID:-}" ]; then
        kill -TERM "-$ACTIVE_SCALENE_PID" 2>/dev/null || kill -TERM "$ACTIVE_SCALENE_PID" 2>/dev/null || true
        sleep 1
        kill -KILL "-$ACTIVE_SCALENE_PID" 2>/dev/null || kill -KILL "$ACTIVE_SCALENE_PID" 2>/dev/null || true
    fi
    pkill -9 python || true
    sleep 1
}
trap cleanup_processes EXIT INT TERM

run_scalene_analysis() {
    local id="$1"
    local data_path="$2"
    local mode="$3"

    local json_out="$METHOD_SAVE_DIR/${TARGET_METHOD}_${id}_${mode}.json"
    local html_out="$METHOD_SAVE_DIR/${TARGET_METHOD}_${id}_${mode}.html"
    local log_out="$METHOD_SAVE_DIR/${TARGET_METHOD}_${id}_${mode}.log"

    TEMP_JSON="$SAVE_DIR/scalene-profile.json"
    TEMP_HTML="$SAVE_DIR/scalene-profile.html"

    mkdir -p "$METHOD_SAVE_DIR"
    rm -f "$TEMP_JSON" "$TEMP_HTML"

    echo "   -> [$mode] Capturing snapshot..."

    local current_args="$SCALENE_ARGS"

    if [ "$mode" == "gpu" ]; then
        current_args="$current_args --gpu"
        export PYTHONMALLOC=malloc
        export CUPTI_ERROR_IGNORE=1
    fi

    export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1

    cleanup_processes

    local cmd="python -m scalene run $current_args --outfile $TEMP_JSON $TARGET_SCRIPT $data_path"

    > "$log_out"

    if command -v setsid >/dev/null 2>&1; then
        setsid $cmd > "$log_out" 2>&1 &
    else
        $cmd > "$log_out" 2>&1 &
    fi
    ACTIVE_SCALENE_PID="$!"

    if [ "$SHOW_PYTHON_LOGS" -eq 1 ]; then
        echo "   [DEBUG CMD]: $cmd"
        echo "   ==================== Real-time run logs ===================="
        tail -f "$log_out" &
        TAIL_PID=$!
    fi

    wait "$ACTIVE_SCALENE_PID" || true
    ACTIVE_SCALENE_PID=""

    if [ "$SHOW_PYTHON_LOGS" -eq 1 ]; then
        kill "$TAIL_PID" 2>/dev/null || true
        wait "$TAIL_PID" 2>/dev/null || true
        TAIL_PID=""
        echo -e "\n   ======================================================"
    fi

    if [ -f "$TEMP_JSON" ] && [ -s "$TEMP_JSON" ]; then
        echo "   -> [$mode] Generating offline web report..."

        (
            cd "$SAVE_DIR"
            python3 -m scalene view --standalone scalene-profile.json > /dev/null 2>&1 || scalene view scalene-profile.json --standalone > /dev/null 2>&1
        )

        if [ -f "$TEMP_HTML" ]; then
            mv "$TEMP_JSON" "$json_out"
            mv "$TEMP_HTML" "$html_out"
            echo "   ✅ [$mode] Analysis results ready: $(basename "$json_out")"
        else
            echo "   ⚠️ [$mode] Web report generation failed, keeping JSON only: $(basename "$json_out")"
            mv "$TEMP_JSON" "$json_out"
        fi
    else
        echo "   ❌ [$mode] Scalene capture failed or file is empty. Check log file: $(basename "$log_out")"
        if [ "$SHOW_PYTHON_LOGS" -eq 0 ]; then
            echo "   --- Error log summary ---"
            tail -n 10 "$log_out" | sed 's/^/   /'
        fi
        rm -f "$TEMP_JSON"
    fi
}

# ================= Main iteration process =================

process_item() {
    local id="$1"
    local data_path="$2"

    if [ "${SUCCESSFUL_BUSINESS_TASKS["${TARGET_METHOD}_${id}"]:-0}" == "1" ]; then
        echo "⏭️  [Business skip] Business result exists: ID=$id, no need to analyze."
        return
    fi

    local NEED_CPU=0
    local NEED_GPU=0

    if [ "$ENABLE_GPU" -eq 1 ]; then
        NEED_GPU=1
        if [ -f "$METHOD_SAVE_DIR/${TARGET_METHOD}_${id}_gpu.json" ] && [ -f "$METHOD_SAVE_DIR/${TARGET_METHOD}_${id}_gpu.html" ]; then
            NEED_GPU=0
        fi
    else
        NEED_CPU=1
        if [ -f "$METHOD_SAVE_DIR/${TARGET_METHOD}_${id}_cpu.json" ] && [ -f "$METHOD_SAVE_DIR/${TARGET_METHOD}_${id}_cpu.html" ]; then
            NEED_CPU=0
        fi
    fi

    if [ "$NEED_CPU" -eq 0 ] && [ "$NEED_GPU" -eq 0 ]; then
        echo "⏭️  [Analysis skip] Results complete: ID=$id"
        return
    fi

    echo "========================================================"
    echo "▶️  Found incomplete analysis task | ID: $id"

    if [ "$NEED_CPU" -eq 1 ]; then
        run_scalene_analysis "$id" "$data_path" "cpu"
    fi

    if [ "$NEED_GPU" -eq 1 ]; then
        run_scalene_analysis "$id" "$data_path" "gpu"
    fi
    
    echo "⏹️  Processing complete: ID=$id"
    echo ""
}

# ================= Read task data =================
if [ ${#INPUT_DATASETS[@]} -gt 0 ]; then
    echo "INPUT_DATASETS not empty, ignoring CSV file, processing array data..."
    for data_path in "${INPUT_DATASETS[@]}"; do
        filename=$(basename "$data_path")
        id="${filename%.*}"
        process_item "$id" "$data_path"
    done
else
    if [ ! -f "$DATA_CSV" ]; then
        echo "Error: Data file $DATA_CSV does not exist and INPUT_DATASETS is empty!"
        exit 1
    fi
    echo "Reading CSV file data for analysis..."
    
    while IFS='|' read -r id data_path; do
        
        if [ -z "$id" ] || [ -z "$data_path" ]; then continue; fi
        
        process_item "$id" "$data_path"
        
    done < <(awk -F',' '
        NR==1 {
            for(i=1; i<=NF; i++) {
                col=$i; gsub(/^[ \t"]+|[ \t"\r]+$/, "", col)
                if(col == "ID") id_idx=i
                if(col == "path") path_idx=i
            }
        }
        NR>1 {
            if(id_idx > 0 && path_idx > 0) {
                id_val=$id_idx
                path_val=$path_idx
                gsub(/^[ \t"]+|[ \t"\r]+$/, "", id_val)
                gsub(/^[ \t"]+|[ \t"\r]+$/, "", path_val)
                
                if(id_val != "") print id_val "|" path_val
            }
        }
    ' "$DATA_CSV")
fi

cleanup_processes
echo "🎉 All performance analysis tasks for abnormal data completed!"