# Evaluation

Run the evaluation script after collecting each method’s results into the configured result directories. The script reads per-method outputs, computes the metrics, and writes CSV summaries and figures to the designated output folder.
- **Inputs**  
  Per-method velocity outputs and any required ground‑truth or reference data (e.g., simulated trajectories, annotated cell states).

- **Outputs**  
  Numeric metrics and summary tables (e.g., accuracy, concordance, peak/shape statistics), plus plots that visualize method performance across datasets.

# Profiling with Scalene
You can run the performance profiling using either a local environment or our provided Docker images.
### Option 1: Local Environment
We recommend referring to the official Scalene GitHub repository for installation guidelines (`pip install scalene`). Please ensure all dependencies and the specific single-cell tools you wish to profile (e.g., scVelo, UnitVelo, Velocyto) are installed according to their respective guidelines before running our profiling scripts.
### Option 2: Docker (Recommended)
Alternatively, you can use our provided Docker images to run the profiling in an isolated container environment. This is highly recommended to prevent background host processes from interfering with CPU and memory measurements.

**Steps to run with Docker:**
1. Ensure the Docker image is loaded and the container is running with appropriate permissions.
2. Open the `scalene.sh` script.
3. Modify the `TARGET_METHOD` and `TARGET_SCRIPT` variables to match the specific tool you intend to profile.
4. Update the `DATA_CSV` or `INPUT_DATASETS` arrays to point to your target `.h5ad` files.

**Configuration Example (`scalene.sh`):**

```bash
# Define the target method and execution script
TARGET_METHOD="scVelo-dynamical"
TARGET_SCRIPT=".../scalene/scvelo_D_sim.py"
# Define data inputs and output directory
DATA_CSV=".../sim_data_forDocker.csv"
SAVE_DIR=".../scalene/output"
```
**Profiling Parameters in `scalene.sh`**

This project includes customizable profiling arguments inside `scalene.sh` to help standardize CPU, memory, and GPU tracking across different tool evaluations. Adjusting these parameters makes it easier to balance profiling precision with execution overhead, ensuring fair comparisons between algorithms.
---
**Why adjust profiling parameters**
- **Precision vs. Overhead**: High-frequency sampling (e.g., 1 Byte) captures exact memory peaks but severely slows down execution. Adjusted thresholds keep profiling times reasonable.
- **Hardware Targeting**: Different algorithms utilize hardware differently. Toggling GPU tracking ensures accurate VRAM allocation monitoring for deep learning tools (like VeloVI or UnitVelo) without burdening pure CPU tools.
- **Reproducibility**: Fixed tracking windows make memory and time experiments easier to reproduce consistently across different datasets.
---
**How to configure parameters**
Before you run `scalene.sh`, open the script and adjust the core variables and `SCALENE_ARGS` to match your hardware and desired profiling granularity.
---
**Example configuration**

Edit the variables in `scalene.sh` to reflect your profiling needs. Example:

```bash
# GPU profiling switch (1: enable GPU profiling, 0: CPU only)
ENABLE_GPU=1
# Scalene core tracking arguments
# Note: 16777216 Bytes = 16MB sampling window
SCALENE_ARGS="--memory --malloc-threshold 16777216"
```
- **ENABLE_GPU**: Set to `1` to include GPU memory and utilization tracking, or `0` to isolate CPU metrics.
- **SCALENE_ARGS**: Command-line arguments passed directly to the python profiler module.
- **--memory**: Instructs Scalene to profile memory usage and identify line-by-line memory leaks.
- **--malloc-threshold**: Sets the allocation sampling window in bytes (e.g., `1048576` for 1MB). Lower values increase accuracy; higher values improve profiling execution speed.