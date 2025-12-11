#!/bin/bash
set -e

echo "=============================================="
echo " ADRAE Prime-Core — Container Runtime Start"
echo "=============================================="

# Create persistent data directories
mkdir -p /data/logs
mkdir -p /data/stability
mkdir -p /data/substrate

# Activate environment variables
if [ -f "/app/docker/adrae.env" ]; then
    export $(grep -v '^#' /app/docker/adrae.env | xargs)
fi

if [ "$1" = "server" ]; then
    echo "[*] Starting ADRAE Health Server..."
    exec uvicorn prime_core.health_server:app --host 0.0.0.0 --port 8080
else
    exec python3 -m prime_core.runtime
fi

