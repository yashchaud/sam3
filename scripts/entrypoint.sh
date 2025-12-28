#!/bin/bash
# =============================================================================
# RunPod Entrypoint Script
# =============================================================================
# This script determines the run mode based on environment variables and
# starts either the serverless handler or the API server.
#
# Environment Variables:
#   RUNPOD_SERVERLESS: Set to "true" for serverless mode (default: false)
#   API_HOST: API server host (default: 0.0.0.0)
#   API_PORT: API server port (default: 8000)
#
# Usage:
#   # Pod mode (API server)
#   docker run -e RUNPOD_SERVERLESS=false ...
#
#   # Serverless mode
#   docker run -e RUNPOD_SERVERLESS=true ...
# =============================================================================

set -e

echo "=================================================="
echo "  Anomaly Detection Pipeline"
echo "  RF-DETR + SAM3"
echo "=================================================="
echo ""

# Print configuration
echo "Configuration:"
echo "  DETECTOR_VARIANT: ${DETECTOR_VARIANT:-medium}"
echo "  SAM_MODEL_PATH: ${SAM_MODEL_PATH:-/models/sam3.pt}"
echo "  CONFIDENCE_THRESHOLD: ${CONFIDENCE_THRESHOLD:-0.3}"
echo "  RUNPOD_SERVERLESS: ${RUNPOD_SERVERLESS:-false}"
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
    echo ""
fi

# Determine run mode
if [ "${RUNPOD_SERVERLESS:-false}" = "true" ]; then
    echo "Starting in SERVERLESS mode..."
    echo ""
    exec python -u /app/src/handler.py
else
    echo "Starting in POD mode (API server)..."
    echo "  Host: ${API_HOST:-0.0.0.0}"
    echo "  Port: ${API_PORT:-8000}"
    echo ""
    exec python -u /app/src/api_server.py
fi
