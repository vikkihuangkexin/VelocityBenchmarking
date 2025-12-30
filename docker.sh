#!/bin/bash

# =======================================================================
# Script Configuration
# =======================================================================

# --- 1. Image and Container Naming ---
# Default to scVelo if no input is provided
read -p "Enter IMAGE_NAME (default: scVelo): " INPUT_IMAGE_NAME
IMAGE_NAME=${INPUT_IMAGE_NAME:-scVelo}

CONTAINER_NAME="${IMAGE_NAME}_container"

echo "-> Image: $IMAGE_NAME"
echo "-> Container: $CONTAINER_NAME"
echo ""

# --- 2. Resource Limit Configuration ---
# Options to apply resource limits (CPU/Memory/GPU)
# Default is 'n' (No limits)
read -p "Apply resource limits? (y/n, default: n): " LIMIT_CONFIRM
LIMIT_CONFIRM=${LIMIT_CONFIRM:-n}

# Initialize variables as empty strings
CPU_FLAG=""
MEMORY_FLAG=""
CPU_SET_FLAG=""
GPU_FLAG=""

if [[ "$LIMIT_CONFIRM" == "y" ]]; then
    echo "-> Resource limits ENABLED."
    
    # Define your limits here
    CPU_CORES=8
    MEMORY_GB="32g"
    # Bind to specific CPU cores (Update based on lscpu)
    CPU_SET_VAL="0-7" 
    # GPU Device Index (e.g., "device=1")
    GPU_DEVICE_VAL="device=0"

    # Set the flags
    CPU_FLAG="--cpus=$CPU_CORES"
    MEMORY_FLAG="--memory=$MEMORY_GB --memory-swap=128g"
    CPU_SET_FLAG="--cpuset-cpus=$CPU_SET_VAL"
    GPU_FLAG="--gpus \"$GPU_DEVICE_VAL\""
else
    echo "-> Resource limits DISABLED (Using defaults)."
fi

# --- 3. Data Mount Configuration ---
# Format: -v /host/absolute/path:/container/absolute/path
HOST_PATH="$(pwd)/example"
CONTAINER_PATH="/example"
VOLUME_MOUNT="-v ${HOST_PATH}:${CONTAINER_PATH}"


# =======================================================================
# Script Execution
# =======================================================================

echo ""
echo "--- Step 1: Starting Container ($CONTAINER_NAME) ---"
echo "Data Mount:"
echo "  -> Host: '${HOST_PATH}'"
echo "  -> Container: '${CONTAINER_PATH}'"
echo ""

# Run the container in background
# Using 'eval' to handle the quoting of GPU/CPU flags properly if they exist
eval "docker run -d \
    -p 8808:22 \
    --name $CONTAINER_NAME \
    $CPU_FLAG \
    $GPU_FLAG \
    $MEMORY_FLAG \
    $CPU_SET_FLAG \
    $VOLUME_MOUNT \
    $IMAGE_NAME \
    tail -f /dev/null"

# Check if container started successfully
if [ $? -eq 0 ]; then
    echo "Container started successfully in the background."
    echo ""
    
    echo "--- Step 2: Running Code Inside Container ---"
    echo "Executing: conda activate $IMAGE_NAME"
    echo "Executing: python script..."

    # Define the python execution command
    # Note: We use 'bash -c' to ensure we can chain commands and source bashrc for conda
    # Adjust the paths (...../) below to match your actual structure inside the container
    
    SCRIPT_CMD="source ~/.bashrc && \
                conda activate $IMAGE_NAME && \
                python /path/to/script/${IMAGE_NAME}.py \
                --save_dir ${CONTAINER_PATH}/${IMAGE_NAME} \
                --data_dir ${CONTAINER_PATH}/data/example.h5ad"

    # Execute the command inside the container
    docker exec -it $CONTAINER_NAME /bin/bash -c "$SCRIPT_CMD"

    echo ""
    echo "--- Execution Finished ---"
    echo "Container $CONTAINER_NAME is still running in the background."
    echo "To stop it, run: docker stop $CONTAINER_NAME"
    echo "To enter it, run: docker exec -it $CONTAINER_NAME bash"
    echo "To remove it, run: docker rm -f $CONTAINER_NAME"
else
    echo "Failed to start the container!"
    exit 1
fi