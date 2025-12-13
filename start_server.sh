#!/bin/bash

# SAM3 API - Server Starter Script
# This script activates the virtual environment and starts the API server

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "SAM3 API - Starting Server"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please run setup.sh first to set up the environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Check if SAM3 is installed
echo ""
echo "Checking SAM3 installation..."
if ! python3 -c "import sam3" 2>/dev/null; then
    echo -e "${RED}Error: SAM3 is not installed${NC}"
    echo "Please run setup.sh to complete the installation"
    exit 1
fi
echo -e "${GREEN}✓ SAM3 is installed${NC}"

# Start the server
echo ""
echo "=================================="
echo -e "${GREEN}Starting SAM3 API Server${NC}"
echo "=================================="
echo ""
echo "API will be available at: http://localhost:8000"
echo "Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
