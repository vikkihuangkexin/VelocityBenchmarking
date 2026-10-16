#!/bin/bash
set -e

# Start SSH service in background
service ssh start

# Run bash in foreground (PID 1)
exec bash