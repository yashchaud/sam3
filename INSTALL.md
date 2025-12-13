# SAM3 API - Installation Guide

Complete installation instructions for the SAM3 Segmentation API.

## Prerequisites

Before you begin, ensure you have:

- **Operating System**: Linux (Ubuntu 22.04+ recommended) or Windows with WSL2
- **Python**: 3.12 or higher
- **CUDA**: 12.6 or higher (for GPU support)
- **GPU**: NVIDIA GPU with at least 8GB VRAM (recommended)
- **RAM**: At least 16GB system RAM
- **Disk Space**: At least 10GB free space for model weights and dependencies
- **Hugging Face Account**: Required for SAM3 model access

## Step-by-Step Installation

### 1. Request SAM3 Model Access

1. Go to [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. Click "Request access to this model"
3. Wait for approval (usually within a few hours)

### 2. Set Up Hugging Face Authentication

Install Hugging Face CLI:

```bash
pip install huggingface_hub
```

Login to Hugging Face:

```bash
huggingface-cli login
```

When prompted, paste your Hugging Face token. You can create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 3. Install System Dependencies (Linux)

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y \
    python3.12 \
    python3-pip \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1
```

**CentOS/RHEL:**

```bash
sudo yum install -y \
    python3.12 \
    python3-pip \
    git \
    gcc \
    gcc-c++ \
    make
```

### 4. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3.12 -m venv sam3-env

# Activate (Linux/Mac)
source sam3-env/bin/activate

# Activate (Windows)
sam3-env\Scripts\activate
```

### 5. Install PyTorch with CUDA Support

**For CUDA 12.6:**

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**For CPU only (not recommended):**

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify PyTorch installation:

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.7.0
CUDA available: True
```

### 6. Install SAM3

Clone the official SAM3 repository:

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
```

Install SAM3:

```bash
pip install -e .
```

Verify installation:

```python
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 installed successfully')"
```

### 7. Install API Dependencies

Navigate back to your API directory:

```bash
cd /path/to/sam3-api
```

Install requirements:

```bash
pip install -r requirements.txt
```

### 8. Verify Installation

Run the health check:

```bash
python app.py &
sleep 10
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

## Docker Installation (Alternative)

### Prerequisites for Docker

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit (for GPU support)

### Install NVIDIA Container Toolkit (Linux)

```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### Build and Run with Docker

1. **Set up Hugging Face token in environment:**

```bash
export HF_TOKEN="your_hugging_face_token"
```

2. **Build and start:**

```bash
docker-compose up --build
```

3. **Access API:**

```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Issue: "Model not loading"

**Solution:**
1. Ensure you have access to the SAM3 model on Hugging Face
2. Run `huggingface-cli whoami` to verify authentication
3. Check server logs for detailed errors

### Issue: "CUDA out of memory"

**Solution:**
1. Reduce image size (images are auto-resized to max 4096x4096)
2. Process one image at a time
3. Use a GPU with more VRAM
4. Switch to CPU mode (set `CUDA_VISIBLE_DEVICES=-1`)

### Issue: "ImportError: cannot import name 'Sam3Processor'"

**Solution:**
1. Ensure SAM3 is installed from the official repository
2. Reinstall: `cd sam3 && pip install -e . --force-reinstall`
3. Check Python version: `python --version` (must be 3.12+)

### Issue: "Connection refused" when accessing API

**Solution:**
1. Ensure the server is running: `python app.py`
2. Check if port 8000 is available: `lsof -i :8000`
3. Try different port: `uvicorn app:app --port 8080`

### Issue: Docker build fails

**Solution:**
1. Ensure Docker has sufficient disk space
2. Clear Docker cache: `docker system prune -a`
3. Build with more memory: Add to docker-compose.yml:
   ```yaml
   build:
     context: .
     shm_size: '2gb'
   ```

## Performance Optimization

### For Production Use

1. **Use GPU**: CUDA-enabled GPU provides 10-50x speedup over CPU
2. **Batch processing**: Process multiple images in parallel (separate requests)
3. **Model caching**: Models are cached after first load
4. **Image preprocessing**: Resize images before sending to API
5. **Use nginx**: Set up nginx as reverse proxy for load balancing

### Example nginx Configuration

```nginx
upstream sam3_api {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://sam3_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 50M;
    }
}
```

## Next Steps

1. Read the [README.md](README.md) for API usage examples
2. Try the example client: `python example_client.py --help`
3. Test with your images
4. Set up monitoring and logging for production use

## Getting Help

- SAM3 Issues: [https://github.com/facebookresearch/sam3/issues](https://github.com/facebookresearch/sam3/issues)
- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Hugging Face: [https://huggingface.co/facebook/sam3/discussions](https://huggingface.co/facebook/sam3/discussions)
