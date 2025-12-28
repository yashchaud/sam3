# =============================================================================
# Anomaly Detection Pipeline - RunPod Dockerfile (Optimized)
# =============================================================================
# Optimized for fast builds - heavy model downloads happen at RUNTIME.
#
# Build time: ~2-3 minutes (just base + lightweight deps)
# First run: +2-3 minutes (downloads RF-DETR, SAM3, model weights)
# Subsequent runs: Fast (cached in volume)
#
# Models:
#   - RF-DETR Medium: ~1.5GB VRAM, auto-downloads on first use
#   - SAM3: ~4GB VRAM, mounted from volume
#
# Build:
#   docker build -t anomaly-detection:latest .
#
# Run:
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/models:/models \
#     -v anomaly-cache:/root/.cache \
#     anomaly-detection:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Base Image: RunPod PyTorch (pre-built with CUDA + PyTorch)
# -----------------------------------------------------------------------------
# SAM3 requires Python 3.12+ and PyTorch 2.7+
# Using latest RunPod image with Python 3.11 + upgrading PyTorch at runtime
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# -----------------------------------------------------------------------------
# System Dependencies (minimal - most already in base)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# -----------------------------------------------------------------------------
# Working Directory
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# Python Dependencies - LIGHTWEIGHT ONLY
# -----------------------------------------------------------------------------
# Only install small, fast packages at build time
# Heavy packages (RF-DETR, SAM2) are installed at runtime

# Core lightweight dependencies + SAM3 prerequisites
RUN pip install --no-cache-dir \
    "numpy>=1.26.0,<2" \
    opencv-python-headless>=4.8.0 \
    Pillow>=10.0.0 \
    supervision>=0.21.0 \
    fastapi>=0.109.0 \
    uvicorn>=0.27.0 \
    python-multipart>=0.0.6 \
    pydantic>=2.0.0 \
    runpod>=1.6.0 \
    # SAM3 core dependencies (install at build time for faster runtime)
    timm>=1.0.17 \
    tqdm \
    ftfy==6.1.1 \
    regex \
    iopath>=0.1.10 \
    typing_extensions \
    huggingface_hub \
    decord \
    einops

# -----------------------------------------------------------------------------
# Application Code (changes frequently - keep near end)
# -----------------------------------------------------------------------------
COPY setup.py /app/
COPY anomaly_detection/ /app/anomaly_detection/
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Make scripts executable and install package
RUN chmod +x /app/scripts/*.sh && pip install --no-cache-dir -e .

# -----------------------------------------------------------------------------
# Runtime Installation Script
# -----------------------------------------------------------------------------
COPY <<'EOF' /app/scripts/install_models.sh
#!/bin/bash
# =============================================================================
# Runtime Model Installation
# Installs heavy dependencies on first run, caches in volume
# =============================================================================
set -e

CACHE_DIR="${CACHE_DIR:-/root/.cache}"
INSTALL_MARKER="$CACHE_DIR/.anomaly_detection_installed_v2"

# Skip if already installed
if [ -f "$INSTALL_MARKER" ]; then
    echo "Dependencies already installed (cached)"
    exit 0
fi

echo "Installing heavy dependencies (first run only)..."

# Upgrade PyTorch to 2.7+ for SAM3 compatibility
echo "Upgrading PyTorch to 2.7+ for SAM3..."
pip install --no-cache-dir torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install RF-DETR
echo "Installing RF-DETR..."
pip install --no-cache-dir rfdetr>=1.2.0

# Install SAM3 (dependencies already installed at build time)
echo "Installing SAM3..."
pip install --no-cache-dir git+https://github.com/facebookresearch/sam3.git

# Mark as installed
mkdir -p "$CACHE_DIR"
date > "$INSTALL_MARKER"
echo "Installation complete!"
EOF
RUN chmod +x /app/scripts/install_models.sh

# -----------------------------------------------------------------------------
# Updated Entrypoint
# -----------------------------------------------------------------------------
COPY <<'EOF' /app/scripts/entrypoint.sh
#!/bin/bash
# =============================================================================
# RunPod Entrypoint - Optimized
# =============================================================================
set -e

echo "=================================================="
echo "  Anomaly Detection Pipeline"
echo "  RF-DETR + SAM3"
echo "=================================================="

# Install heavy deps on first run
/app/scripts/install_models.sh

echo ""
echo "Configuration:"
echo "  DETECTOR_VARIANT: ${DETECTOR_VARIANT:-medium}"
echo "  SAM_MODEL_PATH: ${SAM_MODEL_PATH:-/models/sam3.pt}"
echo "  CONFIDENCE_THRESHOLD: ${CONFIDENCE_THRESHOLD:-0.3}"
echo "  RUNPOD_SERVERLESS: ${RUNPOD_SERVERLESS:-false}"
echo ""

# GPU info
if command -v nvidia-smi &> /dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Start appropriate mode
if [ "${RUNPOD_SERVERLESS:-false}" = "true" ]; then
    echo "Starting SERVERLESS mode..."
    exec python -u /app/src/handler.py
else
    echo "Starting POD mode (API server)..."
    echo "  http://${API_HOST:-0.0.0.0}:${API_PORT:-8000}"
    exec python -u /app/src/api_server.py
fi
EOF
RUN chmod +x /app/scripts/entrypoint.sh

# -----------------------------------------------------------------------------
# Create Directories
# -----------------------------------------------------------------------------
RUN mkdir -p /models /root/.cache

# -----------------------------------------------------------------------------
# Environment Variables
# -----------------------------------------------------------------------------
ENV RUNPOD_SERVERLESS=false
ENV DETECTOR_VARIANT=medium
ENV CONFIDENCE_THRESHOLD=0.3
ENV SAM_MODEL_PATH=/models/sam3.pt
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV CACHE_DIR=/root/.cache

# -----------------------------------------------------------------------------
# Expose & Health Check
# -----------------------------------------------------------------------------
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
