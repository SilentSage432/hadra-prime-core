#!/bin/bash
set -e

echo "=============================================="
echo " ADRAE Prime-Core — Container Runtime Start"
echo "=============================================="

# Activate environment variables
if [ -f "/app/docker/adrae.env" ]; then
    export $(grep -v '^#' /app/docker/adrae.env | xargs)
fi

python3 -m prime_core.runtime

