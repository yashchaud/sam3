#!/bin/bash

# SAM3 API - Server Starter Script
# This script activates the virtual environment and starts the API server

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
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to activate virtual environment${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Check if SAM3 is installed
echo ""
echo "Checking SAM3 installation..."

# Try to import SAM3 and capture both output and exit code
sam3_test_output=$(python3 -c "import sam3; print('SUCCESS')" 2>&1)
sam3_test_exit=$?

if [ $sam3_test_exit -eq 0 ] && echo "$sam3_test_output" | grep -q "SUCCESS"; then
    echo -e "${GREEN}✓ SAM3 is ready${NC}"
else
    echo -e "${YELLOW}Warning: SAM3 import issue detected${NC}"
    echo "Error message: $sam3_test_output"
    echo ""

    # Check for specific missing dependencies
    if echo "$sam3_test_output" | grep -q "No module named 'einops'"; then
        echo "Installing missing dependency: einops..."
        pip install einops timm ftfy --quiet
    elif echo "$sam3_test_output" | grep -q "numpy"; then
        echo "Attempting to fix numpy compatibility..."
        pip install "numpy>=1.26.0,<2.0" --force-reinstall --no-deps --quiet
    else
        echo "Installing missing SAM3 dependencies..."
        pip install -r requirements.txt --quiet
    fi

    # Check again
    sam3_test_output2=$(python3 -c "import sam3; print('SUCCESS')" 2>&1)
    sam3_test_exit2=$?

    if [ $sam3_test_exit2 -eq 0 ] && echo "$sam3_test_output2" | grep -q "SUCCESS"; then
        echo -e "${GREEN}✓ SAM3 is ready (after fixing dependencies)${NC}"
    else
        echo -e "${RED}Error: SAM3 is not installed properly${NC}"
        echo ""
        echo "Error details:"
        echo "$sam3_test_output2"
        echo ""
        echo "Please run setup.sh to install SAM3:"
        echo "  ./setup.sh"
        exit 1
    fi
fi

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
