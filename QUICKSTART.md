# SAM3 API - Quick Start Guide

Get up and running with the SAM3 API in minutes!

## Prerequisites

Before you start, make sure you have:

1. **Python 3.12+** installed
   ```bash
   python3 --version
   ```

2. **Git** installed
   ```bash
   git --version
   ```

3. **CUDA 12.6+** (optional, for GPU support)
   ```bash
   nvidia-smi
   ```

4. **Hugging Face Account** with SAM3 access
   - Request access at: https://huggingface.co/facebook/sam3
   - Create token at: https://huggingface.co/settings/tokens

## Installation (3 Simple Steps)

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd sam3
```

### Step 2: Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Check Python version
- Create virtual environment
- Install dependencies
- Detect CUDA and install PyTorch
- Guide you through SAM3 installation
- Set up Hugging Face authentication

**Note**: You'll be prompted to login to Hugging Face during setup. Have your token ready!

### Step 3: Start the Server

```bash
chmod +x start_server.sh
./start_server.sh
```

**That's it!** Your API is now running at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Quick Test

Once the server is running, test it with:

### cURL (Terminal)
```bash
curl http://localhost:8000/health
```

### Python Script
```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
```

Expected response:
```json
{
  "status": "running",
  "model": "SAM3",
  "model_loaded": true,
  "device": "cuda",
  "version": "1.0.0"
}
```

## Example Usage

### Text-based Segmentation

```bash
curl -X POST "http://localhost:8000/segment/text" \
  -F "image=@photo.jpg" \
  -F "text=person" \
  -F "return_visualization=false"
```

### Using Python

```python
import requests

url = "http://localhost:8000/segment/text"
files = {"image": open("photo.jpg", "rb")}
data = {"text": "person", "return_visualization": False}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Found {result['num_masks']} objects")
print(f"Scores: {result['scores']}")
```

### Run Example Scripts

```bash
# Activate virtual environment first
source venv/bin/activate

# File upload examples
python example_client.py --image photo.jpg --all

# Base64/JSON examples
python example_base64_client.py --image photo.jpg --all
```

## Troubleshooting

### Server won't start
```bash
# Check if virtual environment is activated
source venv/bin/activate

# Verify SAM3 is installed
python -c "import sam3; print('SAM3 OK')"

# Check app.py exists
ls app.py
```

### Model not loading
1. Ensure you've requested access to SAM3 on Hugging Face
2. Login to Hugging Face:
   ```bash
   source venv/bin/activate
   huggingface-cli login
   ```
3. Restart the server

### CUDA errors
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=-1
./start_server.sh
```

### Port already in use
```bash
# Change port in app.py or run with custom port
uvicorn app:app --host 0.0.0.0 --port 8001
```

## Docker Alternative

If you prefer Docker:

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t sam3-api .
docker run --gpus all -p 8000:8000 sam3-api
```

## Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs for interactive documentation
2. **Read Examples**: Check out the example scripts and documentation
3. **Integration**: See README.md for integration examples in Python, JavaScript, and cURL

## API Endpoints Summary

| Method | Endpoint | Input Types |
|--------|----------|-------------|
| Text | `/segment/text` | File, Base64, URL |
| Box | `/segment/box` | File, Base64, URL |
| Points | `/segment/points` | File, Base64, URL |
| Auto | `/segment/auto` | File, Base64, URL |

**Total**: 12 endpoints (4 methods × 3 input modes)

## Getting Help

- **API Documentation**: http://localhost:8000/docs
- **Examples**: See `example_client.py` and `example_base64_client.py`
- **Full Guide**: See `README.md` and `INSTALL.md`
- **URL Examples**: See `example_url_usage.md`

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

## Restarting

Just run the start script again:
```bash
./start_server.sh
```

No need to run setup.sh again unless you want to reinstall dependencies!
