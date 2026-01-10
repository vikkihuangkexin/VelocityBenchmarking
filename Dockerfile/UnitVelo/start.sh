#!/bin/bash
# set -e: exit immediately if any command fails
set -e

# Ensure ownership of mounted data directory
chown -R user1:user1 /data/simdata

# Insert import and replace scv.read with sc.read in unitvelo utils
sed -i -e '1i import scanpy as sc' -e 's/scv.read/sc.read/g' /opt/conda/envs/unitvelo/lib/python3.8/site-packages/unitvelo/utils.py

# 1. Start the SSH daemon in service mode (background)
echo "Starting SSH service..."
service ssh start

# 2. Run bash shell in the foreground
# The exec command replaces the current shell process with /bin/bash.
# This is important because it makes bash the container's main process (PID 1),
# allowing the container to correctly receive and handle signals from `docker stop`.
echo "Starting bash shell..."
exec bash
