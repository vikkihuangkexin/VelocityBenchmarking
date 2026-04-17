# Benchmarking algorithms for RNA velocity inference


We evaluated the performance, stability, scalability and usability of 29 velocity inference methods, including 20 RNA velocity inference methods, 7 multi-omics velocity inference methods, and 2 velocity-based cell cycle inference methods across 176 datasets. 

<img width="2194" height="2298" alt="velocity benchmarking (2)" src="https://github.com/user-attachments/assets/2fd81762-b0c2-499c-900c-e09b985cb842" />

---
# File description
/velocity_generate: Code for run each velocity inference tool.

/example: Example datasets for running the code.

/Evaluation: Code of accurary evaluation metrics used in the study.

/batch_run: Code of stability evaluation used in the study.

/Dockerfile: Docker for each tool.

/Figure_code: Code for Figure 2, 3, and 4 in the paper.

/PlotData: Input data used for the code in ./Figure_code.

/Simulate_generate: a R script to batch-generate simulated single-cell datasets with dyngen across multiple backbone topologies, cell counts, and gene counts.

---

# Installation and Usage

You can run the tools using either a local installation or our provided Docker images.

### Option 1: Local Installation
We recommend referring to the official GitHub repositories or API documentation of the specific tools you wish to use. Please ensure all dependencies are installed according to their respective guidelines before running our scripts.

### Option 2: Docker (Recommended)
Alternatively, you can use our provided Docker images to run the tools in an isolated container environment.

**Steps to run with Docker:**

1. Ensure the Docker image is loaded into your computing environment.
2. Open the `docker.sh` script.
3. Modify the `IMAGE_NAME` variable to match the specific image you intend to run.
4. Update the `SCRIPT_CMD` variable with the execution command for the specific tool.

**Configuration Example (`docker.sh`):**

```bash
# Define the image name
IMAGE_NAME=${INPUT_IMAGE_NAME:-scVelo}

# Define the execution command
SCRIPT_CMD="source ~/.bashrc && \
            conda activate $IMAGE_NAME && \
            python /path/to/script/${IMAGE_NAME}.py \
            --save_dir ${CONTAINER_PATH}/${IMAGE_NAME} \
            --data_dir ${CONTAINER_PATH}/data/example.h5ad"
```
**Resource limits in `docker.sh`**

This project includes optional resource‑limiting code inside `docker.sh` to help standardize CPU, memory, and GPU usage across different tool containers. Enabling these limits makes it easier to compare resource consumption and execution speed between tools, and it helps prevent a single container from monopolizing host resources.

---

**Why enable resource limits**

- **Fair benchmarking**: Ensures each tool runs under comparable constraints for meaningful performance comparisons.  
- **Stability on shared hosts**: Prevents runaway processes from affecting other workloads.  
- **Reproducibility**: Fixed limits make experiments easier to reproduce across machines.

---

**How to enable limits**

When you run `docker.sh`, you will be prompted:

```
Apply resource limits?
```

Type **`y`** to enable the resource‑limiting logic, then adjust the variables in the script to match your hardware and desired constraints.

---

**Example configuration**

Edit the variables in `docker.sh` to reflect your system. Example:

```bash
# Define your limits here
CPU_CORES=8
MEMORY_GB="32g"

# Bind to specific CPU cores (update based on `lscpu`)
CPU_SET_VAL="0-7"

# GPU device index (e.g., "device=1")
GPU_DEVICE_VAL="device=0"
```

- **CPU_CORES**: Number of CPU cores to reserve for the container.  
- **MEMORY_GB**: Memory limit in a format accepted by your tooling (e.g., `32g`).  
- **CPU_SET_VAL**: CPU affinity range or list; verify available cores with `lscpu`.  
- **GPU_DEVICE_VAL**: GPU device selector used by your runtime (adjust if you have multiple GPUs).

---

# Webserver
The detailed results for the benchmarking and comparison are available on the https://relab.xidian.edu.cn/RNAVelocity/#/

# Datasets
Real datasets: https://zenodo.org/records/18205008

Simulated datasets: https://zenodo.org/records/18276904


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