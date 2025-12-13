# SAM3 Image Segmentation API

A simple FastAPI-based REST API for image segmentation using the SAM3 (Segment Anything Model 3) foundation model.

## Features

- Text-based segmentation (segment objects by description)
- Bounding box-based segmentation
- Point-based segmentation (with positive/negative points)
- REST API with easy-to-use endpoints
- Returns base64-encoded PNG masks

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The model will be automatically downloaded on first run (~900MB)

## Running the Server

```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```bash
GET /
```

### 1. Segment with Text Prompt

Segment objects using natural language descriptions.

**Endpoint:** `POST /segment/text`

**Parameters:**
- `image` (file): Image file to segment
- `text` (string): Description of objects to segment (e.g., "person", "car", "tree")
- `threshold` (float, optional): Confidence threshold (default: 0.5)
- `mask_threshold` (float, optional): Mask threshold (default: 0.5)

**Example:**
```bash
curl -X POST "http://localhost:8000/segment/text" \
  -F "image=@test_image.jpg" \
  -F "text=person" \
  -F "threshold=0.5"
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/segment/text"
files = {"image": open("test_image.jpg", "rb")}
data = {"text": "person", "threshold": 0.5}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Found {result['num_objects']} objects")
# Masks are base64-encoded PNG images
for i, mask in enumerate(result['masks']):
    # Decode and save mask
    import base64
    mask_data = base64.b64decode(mask)
    with open(f"mask_{i}.png", "wb") as f:
        f.write(mask_data)
```

### 2. Segment with Bounding Box

Segment objects within a specified bounding box.

**Endpoint:** `POST /segment/box`

**Parameters:**
- `image` (file): Image file to segment
- `x1` (int): Top-left x coordinate
- `y1` (int): Top-left y coordinate
- `x2` (int): Bottom-right x coordinate
- `y2` (int): Bottom-right y coordinate

**Example:**
```bash
curl -X POST "http://localhost:8000/segment/box" \
  -F "image=@test_image.jpg" \
  -F "x1=100" \
  -F "y1=150" \
  -F "x2=500" \
  -F "y2=450"
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/segment/box"
files = {"image": open("test_image.jpg", "rb")}
data = {"x1": 100, "y1": 150, "x2": 500, "y2": 450}

response = requests.post(url, files=files, data=data)
result = response.json()
```

### 3. Segment with Points

Segment objects using point prompts (positive and negative points).

**Endpoint:** `POST /segment/points`

**Parameters:**
- `image` (file): Image file to segment
- `points_x` (string): Comma-separated x coordinates (e.g., "100,200,300")
- `points_y` (string): Comma-separated y coordinates (e.g., "150,250,350")
- `labels` (string): Comma-separated labels - 1 for positive point, 0 for negative (e.g., "1,1,0")

**Example:**
```bash
curl -X POST "http://localhost:8000/segment/points" \
  -F "image=@test_image.jpg" \
  -F "points_x=200,300" \
  -F "points_y=150,250" \
  -F "labels=1,1"
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/segment/points"
files = {"image": open("test_image.jpg", "rb")}
data = {
    "points_x": "200,300",
    "points_y": "150,250",
    "labels": "1,1"  # Both positive points
}

response = requests.post(url, files=files, data=data)
result = response.json()
```

## Response Format

All endpoints return JSON with the following structure:

```json
{
  "success": true,
  "num_objects": 2,
  "masks": ["base64_encoded_mask_1", "base64_encoded_mask_2"],
  "scores": [0.95, 0.87],
  "prompt": "person"
}
```

## Interactive API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- ~4GB RAM (minimum)
- ~1GB disk space for model weights

## Model Information

- Model: SAM3 (facebook/sam3)
- Parameters: 0.9B
- License: Check model card for details
- Source: [Hugging Face](https://huggingface.co/DiffusionWave/sam3)

## Notes

- First run will download the model weights (~900MB)
- GPU recommended for faster inference
- Masks are returned as base64-encoded PNG images
- Supports all common image formats (JPEG, PNG, etc.)
