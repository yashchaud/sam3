# SAM3 API - Troubleshooting Guide

Common issues and their solutions.

## Numpy Version Conflict

### Problem
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 1.26.0 which is incompatible.
```

Or when running the server:
```
Error: SAM3 is not installed
```

### Cause
SAM3 requires `numpy 1.26.0` while newer versions of opencv-python require `numpy >= 2.0`, creating a dependency conflict.

### Solution
The setup scripts automatically handle this by:
1. Installing compatible versions specified in `requirements.txt`
2. Pinning numpy to `>=1.26.0,<2.0`
3. Using opencv-python `>=4.8.0,<4.10.0`

**Manual fix if needed:**
```bash
source venv/bin/activate  # or venv\Scripts\activate.bat on Windows
pip install "numpy>=1.26.0,<2.0" --force-reinstall --no-deps
```

## HuggingFace Login Issues

### Problem
```
huggingface-cli: command not found
```

### Cause
The `huggingface-cli` command may not be in PATH after installation.

### Solution
Our setup scripts use Python directly for login instead of the CLI:
```bash
python -c "from huggingface_hub import login; login()"
```

This is built into `setup.sh` and `setup.bat` - no manual intervention needed!

## SAM3 Import Errors

### Problem
```python
ModuleNotFoundError: No module named 'sam3'
```

### Cause
1. SAM3 not installed
2. Wrong Python environment
3. Installation failed silently

### Solution

**Check installation:**
```bash
source venv/bin/activate
python -c "import sam3; print(sam3.__version__)"
```

**Reinstall SAM3:**
```bash
source venv/bin/activate
cd sam3_repo
pip install -e .
cd ..
pip install "numpy>=1.26.0,<2.0" --force-reinstall --no-deps
```

## CUDA Not Detected

### Problem
Server runs but uses CPU, slow inference.

### Cause
1. NVIDIA drivers not installed
2. CUDA not installed
3. Wrong PyTorch version

### Solution

**Check CUDA:**
```bash
nvidia-smi
```

**Reinstall PyTorch with CUDA:**
```bash
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Verify PyTorch CUDA:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Virtual Environment Issues

### Problem
```
Error: Virtual environment not found
```

### Cause
Virtual environment wasn't created or was deleted.

### Solution
Run the setup script again:
```bash
./setup.sh  # Linux/macOS
setup.bat   # Windows
```

## Port Already in Use

### Problem
```
ERROR: Address already in use
```

### Cause
Port 8000 is being used by another application.

### Solution

**Find and kill the process:**
```bash
# Linux/macOS
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Or use a different port:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

## Model Download Failures

### Problem
```
Failed to download model from HuggingFace
```

### Cause
1. No HuggingFace access to SAM3
2. Not logged in
3. Network issues

### Solution

**1. Request access:**
Visit https://huggingface.co/facebook/sam3 and request access

**2. Login:**
```bash
source venv/bin/activate
python -c "from huggingface_hub import login; login()"
```

**3. Verify login:**
```bash
python -c "from huggingface_hub import whoami; print(whoami())"
```

## Docker Issues

### Problem
GPU not accessible in Docker container.

### Cause
1. NVIDIA Docker runtime not installed
2. Wrong docker-compose configuration

### Solution

**Install NVIDIA Docker:**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Test GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## Permission Denied Errors

### Problem
```bash
./setup.sh: Permission denied
```

### Cause
Script not executable.

### Solution
```bash
chmod +x setup.sh start_server.sh
./setup.sh
```

## API Returns 503 Service Unavailable

### Problem
```json
{"detail": "SAM3 model not loaded"}
```

### Cause
1. Model failed to load
2. Insufficient GPU memory
3. SAM3 not installed properly

### Solution

**Check server logs for detailed error:**
Look at the console output when starting the server.

**Common fixes:**
1. Ensure SAM3 is installed: `python -c "import sam3"`
2. Check GPU memory: `nvidia-smi`
3. Try CPU mode: `export CUDA_VISIBLE_DEVICES=-1`

## Image Size Errors

### Problem
```json
{"detail": "Image too large"}
```

### Cause
Image exceeds size limits.

### Solution
Images are automatically resized to 4096x4096 max. If you're using URL input, the limit is 50MB.

For larger images, preprocess them:
```python
from PIL import Image

img = Image.open("large.jpg")
img.thumbnail((4096, 4096))
img.save("resized.jpg")
```

## Getting More Help

1. **Check server logs**: Look at console output for detailed errors
2. **Verify installation**: Run `python -c "import sam3; import torch; print('OK')"`
3. **Test API**: Visit http://localhost:8000/health
4. **Read documentation**: Check README.md and INSTALL.md

## Quick Diagnostic Script

Save as `diagnose.sh`:
```bash
#!/bin/bash
echo "=== SAM3 API Diagnostics ==="
echo ""
echo "Python version:"
python3 --version
echo ""
echo "Virtual environment:"
[ -d "venv" ] && echo "✓ exists" || echo "✗ missing"
echo ""
echo "SAM3 installation:"
python3 -c "import sam3; print('✓ installed')" 2>/dev/null || echo "✗ not installed"
echo ""
echo "PyTorch:"
python3 -c "import torch; print(f'✓ installed, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "✗ not installed"
echo ""
echo "CUDA:"
command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader || echo "✗ not available"
```

Run with:
```bash
chmod +x diagnose.sh
./diagnose.sh
```
