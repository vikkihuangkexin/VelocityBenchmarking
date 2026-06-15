# Benchmarking algorithms for RNA velocity inference


We evaluated the performance, stability, scalability and usability of 28 velocity inference methods, including 21 RNA velocity inference methods, 9 multi-omics velocity inference methods, and 2 velocity-based cell cycle inference methods across 209 datasets (69 real and 140 simulated). Among these, 4 methods (InterVelo, SDEvelo, STT, and cell2fate) are applicable to the inference of both RNA velocity and specific multi-omics velocities.
<p align="center">
  <img width="75%" alt="Fig 1" src="https://github.com/user-attachments/assets/8d42bba3-e98d-41ec-b6ae-e8fc9890cb68" />
</p>


---
# Repository Structure
- **`Batch_run/`**
Scripts for stability assessment across multiple runs (deep learning-based tools only).

- **`Benchmarked_tools/`**
Source code for running RNA velocity inference methods and generating velocity estimation results.

- **`Dockerfile/`**
Dockerfiles for each benchmarked velocity inference tool.

- **`Evaluation/`**
Scripts of accurary evaluation metrics used in the study.

- **`Example_data/`**
Example datasets and input files for reproducing analyses.

- **`Figure_code/`**
Scripts used to generate Figures 2–4 presented in the manuscript.

- **`PlotData/`**
Input data used for the code in **`Figure_code/`**.

- **`Simulation_generate/`**
Scripts for generating simulated single-cell datasets using [dyngen](https://github.com/dynverse/dyngen) across diverse trajectory backbone topologies, cell counts, and gene counts, as well as transcriptional bursting simulations using [scMultiSim](https://github.com/ZhangLabGT/scMultiSim).

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

# Website
The detailed results for this benchmarking studt are available on the https://relab.xidian.edu.cn/RNAVelocity/#/

# Datasets

All benchmarking datasets used in this study can be downloaded from our websiter at [https://relab.xidian.edu.cn/RNAVelocity/#/download](https://relab.xidian.edu.cn/RNAVelocity/#/download).<br>

You can also use `wget` to download them via the command line:
**Real datasets:**
```bash
wget https://ccsm.uth.edu/Benchmarking/VelocityBenchmarking/RealData.zip
```

**Simulated datasets:**
```bash
wget https://ccsm.uth.edu/Benchmarking/VelocityBenchmarking/SimulatedData.zip
```

# Citation
Huang K, Zhou Y, Wang T, Li X, Zhao X, Liu X, Huang L, Zhou X, Liu J. Benchmarking algorithms for RNA velocity inference. bioRxiv [Preprint]. 2026. doi: 10.64898/2026.01.03.697314.

