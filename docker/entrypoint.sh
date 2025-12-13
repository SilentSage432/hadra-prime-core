#!/bin/bash
set -e

echo "=============================================="
echo " ADRAE Prime-Core — Container Runtime Start"
echo "=============================================="

# Create persistent data directories
mkdir -p /data/logs
mkdir -p /data/memory
mkdir -p /data/continuity
mkdir -p /data/stability
mkdir -p /data/substrate

# Activate environment variables
if [ -f "/app/docker/adrae.env" ]; then
    export $(grep -v '^#' /app/docker/adrae.env | xargs)
fi

# Start ADRAE continuous runtime loop
exec python3 main.py

