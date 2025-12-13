# SAM3 Image Segmentation API

A production-ready FastAPI service for SAM3 (Segment Anything Model 3) image segmentation with comprehensive error handling, validation, and all supported features.

## Quick Start

**Want to get started immediately?** See [QUICKSTART.md](QUICKSTART.md) for a 3-step guide:

```bash
git clone <your-repo>
cd sam3
chmod +x setup.sh && ./setup.sh    # Automated setup
chmod +x start_server.sh && ./start_server.sh    # Start server
```

**Windows users**: Use `setup.bat` and `start_server.bat` instead.

**Docker users**: See Docker installation section below.

## Features

- **🎯 Real-time Click/Touch Segmentation** - NEW! Interactive web interface with live segmentation ([docs](REALTIME_SEGMENTATION.md))
- **Text-based segmentation** - Open-vocabulary segmentation using natural language
- **Bounding box segmentation** - Segment objects within specified boxes
- **Point-based segmentation** - Segment with positive/negative point prompts
- **Automatic segmentation** - Segment all objects without prompts
- **Dual input modes** - File upload AND base64 JSON endpoints for web apps
- **WebSocket API** - Real-time bidirectional communication for interactive apps
- **Visualization support** - Get overlaid mask visualizations
- **Comprehensive error handling** - Detailed error messages and validation
- **CORS enabled** - Ready for web integration
- **Health checks** - Monitor service status
- **Docker support** - Easy deployment with GPU support

## Prerequisites

- Python 3.12+
- PyTorch 2.7+
- CUDA 12.6+ (for GPU support)
- Hugging Face account with SAM3 model access

## Installation

### Option 1: Quick Setup (Recommended)

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
./start_server.sh
```

**Windows:**
```batch
setup.bat
start_server.bat
```

The setup script automatically handles:
- Python environment setup
- Dependency installation
- CUDA detection and PyTorch installation
- SAM3 installation with Hugging Face authentication

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

### Option 2: Manual Installation

1. **Request SAM3 model access**

   Visit [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and request access to the model.

2. **Login to Hugging Face**

   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

3. **Install PyTorch with CUDA**

   ```bash
   pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

4. **Clone and install SAM3**

   ```bash
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   pip install -e .
   cd ..
   ```

5. **Install API dependencies**

   ```bash
   pip install -r requirements.txt
   ```

6. **Run the server**

   ```bash
   python app.py
   ```

### Option 3: Docker Installation

1. **Build and run with Docker Compose**

   ```bash
   docker-compose up --build
   ```

2. **Or build manually**

   ```bash
   docker build -t sam3-api .
   docker run --gpus all -p 8000:8000 sam3-api
   ```

The API will be available at `http://localhost:8000`

## Real-time Interactive Segmentation 🎯

**NEW!** Try the interactive web interface for real-time click-based segmentation:

**[http://localhost:8000/app](http://localhost:8000/app)**

Features:
- Click on objects to segment them instantly
- Real-time mask updates with each click
- Add multiple points to refine segmentation
- Support for foreground (left-click) and background (right-click) points
- Touch support for mobile devices

See [REALTIME_SEGMENTATION.md](REALTIME_SEGMENTATION.md) for full documentation and WebSocket API details.

## API Documentation

Once running, visit these URLs for interactive documentation:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## API Endpoints

The API provides **three input modes** for each endpoint:
1. **File upload** endpoints (multipart/form-data) - `/segment/{method}`
2. **Base64 JSON** endpoints (application/json) - `/segment/{method}/base64`
3. **Image URL** endpoints (application/json) - `/segment/{method}/url`

Choose file upload for traditional applications, base64 JSON for web apps with client-side images, or URL for processing images from the web.

### Health Check

**GET /** or **GET /health**

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "status": "running",
  "model": "SAM3",
  "model_loaded": true,
  "device": "cuda",
  "version": "1.0.0"
}
```

### 1. Text-based Segmentation

#### POST /segment/text (File Upload)

Segment objects using natural language descriptions.

**Parameters:**
- `image` (file, required): Image file to segment
- `text` (string, required): Description of objects to segment
- `return_visualization` (bool, optional): Return visualization instead of masks (default: false)
- `mask_threshold` (float, optional): Mask confidence threshold 0.0-1.0 (default: 0.5)

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/segment/text" \
  -F "image=@photo.jpg" \
  -F "text=person" \
  -F "return_visualization=false" \
  -F "mask_threshold=0.5"
```

**Python Example:**
```python
import requests
import base64
from PIL import Image
import io

url = "http://localhost:8000/segment/text"
files = {"image": open("photo.jpg", "rb")}
data = {
    "text": "person",
    "return_visualization": False,
    "mask_threshold": 0.5
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Success: {result['success']}")
print(f"Found {result['num_masks']} objects")
print(f"Scores: {result['scores']}")

# Save masks
for i, mask_b64 in enumerate(result['masks']):
    mask_data = base64.b64decode(mask_b64)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_image.save(f"mask_{i}.png")
```

#### POST /segment/text/base64 (JSON/Base64)

Same functionality as above, but accepts base64-encoded images in JSON format.

**Python Example:**
```python
import requests
import base64

url = "http://localhost:8000/segment/text/base64"

# Read and encode image
with open("photo.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

payload = {
    "image_base64": image_base64,
    "text": "person",
    "return_visualization": False,
    "mask_threshold": 0.5
}

response = requests.post(url, json=payload)
result = response.json()
```

**JavaScript Example:**
```javascript
async function segmentImage(imageFile, text) {
    // Convert file to base64
    const reader = new FileReader();
    const imageBase64 = await new Promise((resolve) => {
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(imageFile);
    });

    const response = await fetch('http://localhost:8000/segment/text/base64', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image_base64: imageBase64,
            text: text,
            return_visualization: false,
            mask_threshold: 0.5
        })
    });

    const result = await response.json();
    console.log(`Found ${result.num_masks} objects`);
    return result;
}
```

**Response:**
```json
{
  "success": true,
  "num_masks": 3,
  "masks": ["base64_encoded_mask_1", "base64_encoded_mask_2", "base64_encoded_mask_3"],
  "boxes": [[100, 150, 400, 600], [500, 200, 800, 650], [50, 100, 300, 500]],
  "scores": [0.95, 0.88, 0.92],
  "prompt": "person"
}
```

#### POST /segment/text/url (Image URL)

Segment using an image URL - perfect for web scraping, automation, or processing images already hosted online.

**Python Example:**
```python
import requests

url = "http://localhost:8000/segment/text/url"

payload = {
    "image_url": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=800",
    "text": "person",
    "return_visualization": False,
    "mask_threshold": 0.5
}

response = requests.post(url, json=payload)
result = response.json()
print(f"Found {result['num_masks']} people")
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/segment/text/url" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/photo.jpg",
    "text": "car",
    "return_visualization": false
  }'
```

See [example_url_usage.md](example_url_usage.md) for complete URL API documentation with request/response examples.

### 2. All Available Endpoints

All segmentation methods support three input modes:

| Method | File Upload | Base64 JSON | Image URL |
|--------|------------|-------------|-----------|
| Text | POST /segment/text | POST /segment/text/base64 | POST /segment/text/url |
| Box | POST /segment/box | POST /segment/box/base64 | POST /segment/box/url |
| Points | POST /segment/points | POST /segment/points/base64 | POST /segment/points/url |
| Auto | POST /segment/auto | POST /segment/auto/base64 | POST /segment/auto/url |

**Example (Box with file upload):**
```python
import requests

url = "http://localhost:8000/segment/box"
files = {"image": open("photo.jpg", "rb")}
data = {"x1": 100, "y1": 150, "x2": 500, "y2": 450}

response = requests.post(url, files=files, data=data)
result = response.json()
```

**Example (Box with base64):**
```python
import requests
import base64

url = "http://localhost:8000/segment/box/base64"

with open("photo.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

payload = {
    "image_base64": image_base64,
    "x1": 100, "y1": 150,
    "x2": 500, "y2": 450
}

response = requests.post(url, json=payload)
```

**Example (Box with URL):**
```python
import requests

url = "http://localhost:8000/segment/box/url"

payload = {
    "image_url": "https://example.com/image.jpg",
    "x1": 100, "y1": 150,
    "x2": 500, "y2": 450
}

response = requests.post(url, json=payload)
```

### 3. Complete Documentation

For detailed parameters and more examples:

- **Interactive API docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **URL endpoint examples**: [example_url_usage.md](example_url_usage.md)

**Run example scripts:**
```bash
# File upload examples
python example_client.py --image photo.jpg --all

# Base64/JSON examples
python example_base64_client.py --image photo.jpg --all
```

## Visualization Mode

All endpoints support `return_visualization=true` which returns a PNG image with masks overlaid:

```python
import requests
from PIL import Image
import io

url = "http://localhost:8000/segment/text"
files = {"image": open("photo.jpg", "rb")}
data = {
    "text": "car",
    "return_visualization": True
}

response = requests.post(url, files=files, data=data)

# Save visualization
viz_image = Image.open(io.BytesIO(response.content))
viz_image.save("visualization.png")
```

## Error Handling

The API provides detailed error responses:

```json
{
  "detail": "SAM3 model not loaded. Please check server logs and ensure SAM3 is properly installed.",
  "status_code": 503
}
```

Common error codes:
- **400**: Bad request (invalid parameters, image format, coordinates out of bounds)
- **500**: Internal server error (segmentation failure, encoding errors)
- **503**: Service unavailable (model not loaded)

## Response Format

**JSON Response (when return_visualization=false):**
```json
{
  "success": true,
  "num_masks": 2,
  "masks": ["base64_png_1", "base64_png_2"],
  "boxes": [[x1, y1, x2, y2], ...],
  "scores": [0.95, 0.87],
  "prompt": "text prompt used",
  "message": "optional message"
}
```

**Image Response (when return_visualization=true):**
- Content-Type: `image/png`
- Binary PNG image with masks overlaid

## Configuration

Environment variables (optional):
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=-1

# Specify GPU
export CUDA_VISIBLE_DEVICES=0

# API host and port
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Performance Notes

- **First request**: May take longer due to model initialization
- **GPU recommended**: CPU inference is significantly slower
- **Large images**: Images larger than 4096x4096 are automatically resized
- **Memory**: Model requires ~4GB GPU memory
- **Concurrent requests**: FastAPI handles multiple requests, but GPU memory limits concurrent processing

## Troubleshooting

### Model not loading

1. Ensure you've requested and received access to the SAM3 model on Hugging Face
2. Run `huggingface-cli login` and provide your token
3. Check server logs for detailed error messages

### CUDA errors

1. Verify CUDA version: `nvidia-smi`
2. Ensure PyTorch CUDA version matches your system CUDA
3. Set `CUDA_VISIBLE_DEVICES=-1` to use CPU mode

### Import errors

1. Ensure SAM3 is properly installed: `pip list | grep sam3`
2. Reinstall SAM3 from the official repository
3. Check Python version (3.12+ required)

## Development

Run in development mode with auto-reload:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## License

Check the SAM3 model license at [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)

## Links

- SAM3 Model: [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
- SAM3 Repository: [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- FastAPI Documentation: [https://fastapi.tiangolo.com](https://fastapi.tiangolo.com)
