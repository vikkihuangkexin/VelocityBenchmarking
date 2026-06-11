# Benchmarking algorithms for RNA velocity inference


We evaluated the performance, stability, scalability and usability of 28 velocity inference methods, including 21 RNA velocity inference methods, 9 multi-omics velocity inference methods, and 2 velocity-based cell cycle inference methods across 176 datasets.Among these, 4 methods (InterVelo, SDEvelo, STT, and cell2fate) are applicable to the inference of both RNA velocity and specific multi-omics velocities.

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

All benchmarking datasets used in this study can be downloaded from our webserver at [https://relab.xidian.edu.cn/RNAVelocity/#/](https://relab.xidian.edu.cn/RNAVelocity/#/). You can use `wget` to download them via the command line:

**Real datasets:**
```bash
wget https://ccsm.uth.edu/Benchmarking/VelocityBenchmarking/RealData.zip
```

**Simulated datasets:**
```bash
wget https://ccsm.uth.edu/Benchmarking/VelocityBenchmarking/SimulatedData.zip
```


