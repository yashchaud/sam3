#!/bin/bash

# SAM3 API - Quick Setup Script
# This script sets up the SAM3 API server on your local machine

set -e  # Exit on error

echo "=================================="
echo "SAM3 API - Quick Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.12"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${YELLOW}Warning: Python $REQUIRED_VERSION+ recommended, you have $PYTHON_VERSION${NC}"
    echo "Continuing anyway..."
else
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install API dependencies
echo ""
echo "Installing API dependencies..."
pip install -r requirements.txt
echo -e "${GREEN}✓ API dependencies installed${NC}"

# Check for CUDA
echo ""
echo "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo -e "${GREEN}✓ CUDA $CUDA_VERSION detected${NC}"

    # Install PyTorch with CUDA
    echo ""
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    echo -e "${GREEN}✓ PyTorch with CUDA installed${NC}"
else
    echo -e "${YELLOW}⚠ CUDA not detected${NC}"
    echo "Installing PyTorch (CPU version)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo -e "${YELLOW}⚠ PyTorch (CPU) installed - inference will be slower${NC}"
fi

# Check if SAM3 is installed
echo ""
echo "Checking for SAM3..."
if python3 -c "import sam3" 2>/dev/null; then
    echo -e "${GREEN}✓ SAM3 already installed${NC}"
else
    echo -e "${YELLOW}SAM3 not found. Installing...${NC}"
    echo ""
    echo "SAM3 requires access to the model on Hugging Face."
    echo "Please ensure you have:"
    echo "  1. Requested access at: https://huggingface.co/facebook/sam3"
    echo "  2. Created a Hugging Face token at: https://huggingface.co/settings/tokens"
    echo ""

    read -p "Have you completed these steps? (y/n): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Install huggingface_hub
        echo ""
        echo "Installing huggingface_hub..."
        pip install -U huggingface_hub

        # Login to Hugging Face using Python
        echo ""
        echo "==================================="
        echo "Hugging Face Login"
        echo "==================================="
        echo "You'll be prompted to enter your Hugging Face token."
        echo "Get your token at: https://huggingface.co/settings/tokens"
        echo ""

        python3 << 'PYEOF'
from huggingface_hub import login
import sys

try:
    login()
    print("\n✓ Successfully logged in to Hugging Face!")
except Exception as e:
    print(f"\n✗ Login failed: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to login to Hugging Face${NC}"
            exit 1
        fi

        # Clone and install SAM3
        if [ ! -d "sam3_repo" ]; then
            echo ""
            echo "Cloning SAM3 repository..."
            git clone https://github.com/facebookresearch/sam3.git sam3_repo
        fi

        echo "Installing SAM3..."
        cd sam3_repo
        pip install -e .
        cd ..
        echo -e "${GREEN}✓ SAM3 installed${NC}"
    else
        echo -e "${YELLOW}⚠ Skipping SAM3 installation${NC}"
        echo "You can install it later by running:"
        echo "  git clone https://github.com/facebookresearch/sam3.git sam3_repo"
        echo "  cd sam3_repo && pip install -e . && cd .."
    fi
fi

# Create output directory
mkdir -p output

echo ""
echo "=================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=================================="
echo ""
echo "To start the server, run:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Or use the quick start script:"
echo "  ./start_server.sh"
echo ""
echo "API will be available at: http://localhost:8000"
echo "Documentation: http://localhost:8000/docs"
echo ""
