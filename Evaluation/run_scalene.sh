#!/bin/bash

set -euo pipefail

# ================= Configuration =================
DATA_CSV=""
RESULTS_CSV=""
TARGET_SCRIPT=""
TARGET_METHOD=""
SAVE_DIR=""
ENABLE_GPU=0

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
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check required parameters
if [[ -z "$DATA_CSV" || -z "$RESULTS_CSV" || -z "$TARGET_SCRIPT" || -z "$TARGET_METHOD" || -z "$SAVE_DIR" ]]; then
  echo "Usage: $0 --data-csv <path> --results-csv <path> --target-script <path> --target-method <method> --save-dir <path> [--enable-gpu]"
  exit 1
fi

# --- Sub-save paths ---
METHOD_SAVE_DIR="$SAVE_DIR/$TARGET_METHOD"
mkdir -p "$METHOD_SAVE_DIR"

INPUT_DATASETS=()

# Core parameters (16MB sampling window)
SCALENE_ARGS="--memory --malloc-threshold 16777216"

# ================= Load successful business records =================
declare -A SUCCESSFUL_BUSINESS_TASKS

if [ -f "$RESULTS_CSV" ]; then
    echo "Loading analysis success records ($RESULTS_CSV)..."
    while IFS=',' read -r method id result_path; do
        # Enhanced cleanup: remove leading/trailing spaces, all quotes, \r
        method=$(echo "$method" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/"//g' -e 's/\r//g')
        id=$(echo "$id" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/"//g' -e 's/\r//g')
        result_path=$(echo "$result_path" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/"//g' -e 's/\r//g')

        # If path is not empty after cleanup, consider successful
        if [ -n "$result_path" ]; then
            SUCCESSFUL_BUSINESS_TASKS["${method}_${id}"]=1
        fi
    done < <(tail -n +2 "$RESULTS_CSV" || true)
fi

# ================= Task execution and cleanup functions =================

cleanup_processes() {
    echo "   [Cleanup] Terminating remaining Python processes..."
    if [ -n "${ACTIVE_SCALENE_PID:-}" ]; then
        kill -TERM "-$ACTIVE_SCALENE_PID" 2>/dev/null || kill -TERM "$ACTIVE_SCALENE_PID" 2>/dev/null || true
        sleep 2
        kill -KILL "-$ACTIVE_SCALENE_PID" 2>/dev/null || kill -KILL "$ACTIVE_SCALENE_PID" 2>/dev/null || true
    fi
    pkill -9 python || true
    sleep 2
}
trap cleanup_processes EXIT INT TERM

run_scalene_analysis() {
    local id="$1"
    local data_path="$2"
    local mode="$3"
    
    local json_out="$METHOD_SAVE_DIR/${TARGET_METHOD}_${id}_${mode}.json"
    local html_out="$METHOD_SAVE_DIR/${TARGET_METHOD}_${id}_${mode}.html"
    
    # Keep these fixed intermediate parameters
    TEMP_JSON="$SAVE_DIR/scalene-profile.json"
    TEMP_HTML="$SAVE_DIR/scalene-profile.html"
    
    mkdir -p "$METHOD_SAVE_DIR"
    rm -f "$TEMP_JSON" "$TEMP_HTML"

    echo "   -> [$mode] Capturing snapshot..."
    
    local current_args="$SCALENE_ARGS"
    if [ "$mode" == "gpu" ]; then
        current_args="$current_args --gpu"
    else
        export CUDA_VISIBLE_DEVICES=""
    fi

    cleanup_processes

    # Run scalene
    if command -v setsid >/dev/null 2>&1; then
        setsid python3 -m scalene run $current_args --outfile "$TEMP_JSON" "$TARGET_SCRIPT" "$data_path" > /dev/null 2>&1 &
    else
        python3 -m scalene run $current_args --outfile "$TEMP_JSON" "$TARGET_SCRIPT" "$data_path" > /dev/null 2>&1 &
    fi
    ACTIVE_SCALENE_PID="$!"
    wait "$ACTIVE_SCALENE_PID" || true
    ACTIVE_SCALENE_PID=""

    # Verify if JSON was generated successfully
    if [ -f "$TEMP_JSON" ] && [ -s "$TEMP_JSON" ]; then
        echo "   -> [$mode] Generating offline web report..."
        
        # Core fix: Enter TEMP_JSON directory to generate HTML, avoid empty output redirection
        (
            cd "$SAVE_DIR"
            python3 -m scalene view --standalone scalene-profile.json > /dev/null 2>&1 || scalene view scalene-profile.json --standalone > /dev/null 2>&1
        )

        # Check if HTML was generated in current directory
        if [ -f "$TEMP_HTML" ]; then
            mv "$TEMP_JSON" "$json_out"
            mv "$TEMP_HTML" "$html_out"
            echo "   ✅ [$mode] Analysis results ready: $(basename "$json_out")"
        else
            echo "   ⚠️ [$mode] Web report generation failed, keeping JSON only: $(basename "$json_out")"
            mv "$TEMP_JSON" "$json_out"
        fi
    else
        echo "   ❌ [$mode] Scalene capture failed or file is empty."
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

    local NEED_CPU=1
    local NEED_GPU=0

    if [ -f "$METHOD_SAVE_DIR/${TARGET_METHOD}_${id}_cpu.json" ] && [ -f "$METHOD_SAVE_DIR/${TARGET_METHOD}_${id}_cpu.html" ]; then
        NEED_CPU=0
    fi

    if [ "$ENABLE_GPU" -eq 1 ]; then
        NEED_GPU=1
        if [ -f "$METHOD_SAVE_DIR/${TARGET_METHOD}_${id}_gpu.json" ] && [ -f "$METHOD_SAVE_DIR/${TARGET_METHOD}_${id}_gpu.html" ]; then
            NEED_GPU=0
        fi
    fi

    if [ "$NEED_CPU" -eq 0 ] && [ "$NEED_GPU" -eq 0 ]; then
        echo "⏭️  [Analysis skip] Results complete: ID=$id"
        return
    fi

    echo "========================================================"
    echo "▶️  Found incomplete analysis task | ID: $id"

    if [ "$NEED_CPU" -eq 1 ]; then run_scalene_analysis "$id" "$data_path" "cpu"; else echo "   ⏭️  [CPU] Results exist, skipping."; fi
    if [ "$NEED_GPU" -eq 1 ]; then run_scalene_analysis "$id" "$data_path" "gpu"; elif [ "$ENABLE_GPU" -eq 1 ]; then echo "   ⏭️  [GPU] Results exist, skipping."; fi
    
    echo "⏹️  Processing complete: ID=$id"
    echo ""
}

# Core judgment logic
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
    while IFS=',' read -r id data_path; do
        id=$(echo "$id" | tr -d '\r')
        data_path=$(echo "$data_path" | tr -d '\r')

        if [ -z "$id" ] || [ -z "$data_path" ] || [ "$id" == "ID" ]; then continue; fi
        process_item "$id" "$data_path"
    done < <(tail -n +2 "$DATA_CSV")
fi

cleanup_processes
echo "🎉 All performance analysis tasks for abnormal data completed!"