# RunPod Serverless Deployment Guide

Complete guide for deploying SAM3 segmentation as a serverless function on RunPod.

## Quick Deploy

### 1. Build Docker Image (GitHub Actions)

The Docker image is automatically built and pushed to GitHub Container Registry when you push changes.

**Trigger build:**
```bash
git add .
git commit -m "Update serverless"
git push origin main
```

**Or manually trigger:**
- Go to GitHub Actions → "Build and Push Serverless Docker Image" → "Run workflow"

**Image will be available at:**
```
ghcr.io/YOUR_USERNAME/sam3-serverless:latest
```

### 2. Deploy to RunPod

1. **Go to RunPod**: https://www.runpod.io/console/serverless

2. **Create New Endpoint**:
   - Click "New Endpoint"
   - Name: `sam3-segmentation`
   - Container Image: `ghcr.io/YOUR_USERNAME/YOUR_REPO/sam3-serverless:latest`
   - Container Disk: `10 GB`
   - GPU: `NVIDIA RTX 4090` or `A40`

3. **Environment Variables**:
   ```
   HF_TOKEN=your_huggingface_token_here
   ```

4. **Advanced Options**:
   - Max Workers: `3`
   - Idle Timeout: `5` seconds
   - Execution Timeout: `300` seconds

5. **Deploy**: Click "Deploy"

## Usage

### API Endpoint

Once deployed, you'll get an endpoint URL like:
```
https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run
```

### Request Format

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "data:image/png;base64,iVBORw0KG...",
      "points": [[100, 200], [150, 250]],
      "labels": [1, 1]
    }
  }'
```

### Response Format

```json
{
  "id": "request-id",
  "status": "COMPLETED",
  "output": {
    "mask": "base64_png_string",
    "mask_shape": [1024, 768],
    "num_points": 2
  }
}
```

## Python Client Example

```python
import requests
import base64
from PIL import Image
import io

RUNPOD_API_KEY = "your_api_key"
ENDPOINT_ID = "your_endpoint_id"
ENDPOINT_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def segment_image(image_path, points, labels):
    """
    Segment image using SAM3 serverless endpoint.

    Args:
        image_path: Path to image file
        points: List of [x, y] coordinates
        labels: List of 1 (foreground) or 0 (background)

    Returns:
        Base64 encoded mask image
    """
    image_b64 = image_to_base64(image_path)

    payload = {
        "input": {
            "image": f"data:image/png;base64,{image_b64}",
            "points": points,
            "labels": labels
        }
    }

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
    result = response.json()

    if result["status"] == "COMPLETED":
        return result["output"]["mask"]
    else:
        raise Exception(f"Segmentation failed: {result}")

# Example usage
mask_b64 = segment_image(
    "image.jpg",
    points=[[500, 300], [520, 310]],
    labels=[1, 1]
)

# Decode and save mask
mask_bytes = base64.b64decode(mask_b64)
mask_image = Image.open(io.BytesIO(mask_bytes))
mask_image.save("mask.png")
```

## JavaScript Client Example

```javascript
const RUNPOD_API_KEY = "your_api_key";
const ENDPOINT_ID = "your_endpoint_id";
const ENDPOINT_URL = `https://api.runpod.ai/v2/${ENDPOINT_ID}/run`;

async function segmentImage(imageBase64, points, labels) {
    const response = await fetch(ENDPOINT_URL, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${RUNPOD_API_KEY}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            input: {
                image: imageBase64,
                points: points,
                labels: labels
            }
        })
    });

    const result = await response.json();

    if (result.status === 'COMPLETED') {
        return result.output.mask;
    } else {
        throw new Error(`Segmentation failed: ${JSON.stringify(result)}`);
    }
}

// Example usage
const maskBase64 = await segmentImage(
    'data:image/png;base64,iVBORw0KG...',
    [[500, 300], [520, 310]],
    [1, 1]
);

// Display mask
const img = new Image();
img.src = `data:image/png;base64,${maskBase64}`;
document.body.appendChild(img);
```

## Testing Locally

Build and test the Docker image locally:

```bash
# Build image
docker build -f Dockerfile.serverless -t sam3-serverless .

# Run container
docker run --rm --gpus all \
  -e HF_TOKEN=your_token \
  -p 8000:8000 \
  sam3-serverless

# Test with curl
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "data:image/png;base64,...",
      "points": [[100, 200]],
      "labels": [1]
    }
  }'
```

## Cost Optimization

### Cold Start Optimization
- Image is cached after first run
- Subsequent requests are much faster
- Set appropriate idle timeout (5-10 seconds)

### GPU Selection
- **RTX 4090**: ~$0.40/hour, faster inference
- **A40**: ~$0.60/hour, more VRAM
- **RTX 3090**: ~$0.30/hour, budget option

### Scaling
- Start with 1-2 max workers
- Increase based on usage patterns
- Monitor queue time vs cost

## Monitoring

View logs and metrics:
1. Go to RunPod Console → Endpoints
2. Click your endpoint
3. View "Logs" and "Analytics" tabs

## Troubleshooting

### Model Loading Fails
- Check HF_TOKEN is set correctly
- Verify you have access to facebook/sam3
- Check container logs for errors

### Out of Memory
- Reduce max workers
- Use GPU with more VRAM (A40)
- Optimize batch size

### Slow Response Times
- Check cold start time
- Increase min workers to keep instances warm
- Optimize Docker image size

## Files

- `Dockerfile.serverless` - Docker image definition
- `serverless_handler.py` - RunPod handler function
- `.github/workflows/build-serverless.yml` - Auto-build workflow

## GitHub Container Registry

Make your image public (optional):

1. Go to GitHub → Packages
2. Find `sam3-serverless` package
3. Package settings → Change visibility → Public

Or keep private and use Personal Access Token for pulls.

## Environment Variables

Required:
- `HF_TOKEN` - Hugging Face token for model access

Optional:
- `HUGGING_FACE_HUB_TOKEN` - Alternative token variable name

## Limits

- Max execution time: 300 seconds (configurable)
- Max request size: 100 MB (RunPod limit)
- Concurrent requests: Based on max workers setting

## Next Steps

1. Push code to GitHub
2. Wait for Actions to build image
3. Deploy to RunPod with image URL
4. Test with example requests
5. Monitor and optimize costs
