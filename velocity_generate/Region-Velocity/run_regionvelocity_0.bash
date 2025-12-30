#!/usr/bin/env bash

set -e  # Exit on error

# Parse arguments
ENV_FILE="${1:-regionvelocity-environment.yml}"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file '$ENV_FILE' not found!"
    echo "Usage: $0 [path/to/environment.yml]"
    exit 1
fi

echo "Creating conda environment from $ENV_FILE..."
conda env create -f "$ENV_FILE"

echo ""
echo "Installing RegionVelocity R package from GitHub..."
eval "$(conda shell.bash hook)"
conda activate regionvelocity

Rscript --vanilla -e "
library(devtools)
install_github('Dekayzc/Regionvelocity')
"

echo ""
echo "Setup complete! Run 'conda activate regionvelocity' to start."
