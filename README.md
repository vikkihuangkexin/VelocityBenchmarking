# Benchmarking algorithms for RNA velocity inference
We evaluated the performance, stability, scalability and usability of 29 velocity inference methods, including 20 RNA velocity inference methods, 7 multi-omics velocity inference methods, and 2 velocity-based cell cycle inference methods across 176 datasets. 

<img width="2194" height="2298" alt="velocity benchmarking (2)" src="https://github.com/user-attachments/assets/2fd81762-b0c2-499c-900c-e09b985cb842" />

# Installation and Usage

You can run the tools using either a local installation or our provided Docker images.

### Option 1: Local Installation
We recommend referring to the official GitHub repositories or API documentation of the specific tools you wish to use. Please ensure all dependencies are installed according to their respective guidelines before running our scripts.

### Option 2: Docker (Recommended)
Alternatively, you can use our provided Docker images to run the tools in an isolated container environment.

**Steps to run with Docker:**

1. Ensure the Docker image is loaded into your computing environment.
2. Open the `data.sh` script.
3. Modify the `IMAGE_NAME` variable to match the specific image you intend to run.
4. Update the `SCRIPT_CMD` variable with the execution command for the specific tool.

**Configuration Example (`data.sh`):**

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
# Webserver
The detailed results for the benchmarking and comparison are available on the https://relab.xidian.edu.cn/RNAVelocity/#/

# Datasets
https://zenodo.org/records/18102832

