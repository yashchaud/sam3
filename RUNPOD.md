# RunPod Deployment Guide

## VRAM Requirements

| Model | VRAM Usage | Notes |
|-------|------------|-------|
| RF-DETR Nano | ~1.0 GB | Fastest, lowest accuracy |
| RF-DETR Small | ~1.2 GB | Good balance |
| RF-DETR Medium | ~1.5 GB | **Recommended** |
| RF-DETR Large | ~2.5 GB | Best accuracy |
| SAM2 Large | ~3.5 GB | Fallback segmenter |
| SAM3 | ~4.0 GB | Latest, best quality |

### Total VRAM Recommendations

| Configuration | VRAM Required | Recommended GPU |
|---------------|---------------|-----------------|
| RF-DETR Nano + SAM2 | ~5 GB | RTX 3060 (12GB) |
| RF-DETR Medium + SAM2 | ~6 GB | RTX 3070/4060 |
| RF-DETR Medium + SAM3 | **~6-7 GB** | **RTX 3080/4070** |
| RF-DETR Large + SAM3 | ~8 GB | RTX 3090/4080 |

### RunPod GPU Tiers

For **RF-DETR Medium + SAM3** (recommended):

| GPU | VRAM | Cost/hr | Recommendation |
|-----|------|---------|----------------|
| RTX 3070 | 8 GB | ~$0.15 | Minimum viable |
| RTX 3080 | 10 GB | ~$0.20 | **Good choice** |
| RTX 3090 | 24 GB | ~$0.30 | Comfortable headroom |
| RTX 4080 | 16 GB | ~$0.35 | Fast + headroom |
| RTX 4090 | 24 GB | ~$0.55 | Overkill for Phase-1 |
| A10 | 24 GB | ~$0.30 | Enterprise option |

**Recommendation: RTX 3080 (10GB) or RTX 4070 (12GB)** - provides ~6-7GB for models with headroom for batch processing.

---

## Deployment Modes

### 1. Pod Mode (Persistent GPU)

Best for:
- Development and testing
- Consistent workloads
- Low-latency requirements

```bash
# Deploy to RunPod Pod
docker build -t your-registry/anomaly-detection:latest .
docker push your-registry/anomaly-detection:latest

# On RunPod, use:
# Image: your-registry/anomaly-detection:latest
# Environment:
#   RUNPOD_SERVERLESS=false
#   SAM_MODEL_PATH=/models/sam3.pt
```

### 2. Serverless Mode (Pay-per-request)

Best for:
- Variable/unpredictable workloads
- Cost optimization
- Auto-scaling needs

```bash
# Build and push serverless image
docker build -t your-registry/anomaly-detection:serverless .
docker push your-registry/anomaly-detection:serverless

# On RunPod Serverless:
# Image: your-registry/anomaly-detection:serverless
# Environment:
#   RUNPOD_SERVERLESS=true
#   SAM_MODEL_PATH=/runpod-volume/models/sam3.pt
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNPOD_SERVERLESS` | `false` | `true` for serverless, `false` for pod |
| `DETECTOR_VARIANT` | `medium` | `nano`, `small`, `medium`, or `large` |
| `DETECTOR_WEIGHTS` | - | Path to custom fine-tuned weights |
| `DETECTOR_NUM_CLASSES` | - | Number of classes (if custom weights) |
| `SAM_MODEL_PATH` | `/models/sam3.pt` | Path to SAM model weights |
| `CONFIDENCE_THRESHOLD` | `0.3` | Detection confidence threshold |
| `API_HOST` | `0.0.0.0` | API server host (pod mode) |
| `API_PORT` | `8000` | API server port (pod mode) |

---

## API Endpoints (Pod Mode)

### Health Check
```bash
GET /health
```

### Model Info
```bash
GET /info
```

### Single Image Detection
```bash
POST /detect
Content-Type: application/json

{
  "image": "<base64-encoded-image>",
  "frame_id": "optional-id",
  "include_masks": false,
  "confidence_threshold": 0.3
}
```

### Batch Detection
```bash
POST /detect/batch
Content-Type: application/json

{
  "images": ["<base64-image-1>", "<base64-image-2>"],
  "frame_ids": ["frame1", "frame2"],
  "include_masks": false
}
```

### File Upload
```bash
POST /detect/upload
Content-Type: multipart/form-data

file: <image-file>
include_masks: false
```

---

## Serverless Request Format

```json
{
  "input": {
    "image": "<base64-encoded-image>",
    "frame_id": "optional-id",
    "include_masks": false,
    "confidence_threshold": 0.3
  }
}
```

### Response Format

```json
{
  "frame_id": "frame_abc123",
  "image_size": {"width": 1920, "height": 1080},
  "anomalies": [
    {
      "anomaly_id": "anom_xyz789",
      "defect_type": "crack",
      "structure_type": "beam",
      "bbox": {"x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400},
      "geometry": {
        "area_pixels": 5000,
        "length_pixels": 150.5,
        "width_pixels": 33.2,
        "orientation_degrees": 45.0
      },
      "confidence": {
        "detection": 0.92,
        "segmentation": 0.88,
        "combined": 0.90
      }
    }
  ],
  "anomaly_count": 1,
  "structures": [...],
  "processing_time_ms": 125.5
}
```

---

## Model Download

### SAM3 (Recommended)
1. Request access: https://huggingface.co/facebook/sam3
2. Download `sam3.pt`
3. Place in `/models/` or your RunPod volume

### SAM2 (Fallback)
```bash
# Auto-downloaded, or manually:
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### RF-DETR
- Auto-downloads COCO pretrained weights on first use
- No manual download needed unless using custom weights

---

## Quick Start

### Local Testing
```bash
# Build
docker build -t anomaly-detection:test .

# Run (API mode)
docker run --gpus all -p 8000:8000 \
  -e RUNPOD_SERVERLESS=false \
  -v $(pwd)/models:/models \
  anomaly-detection:test

# Test
curl http://localhost:8000/health
```

### RunPod Pod Deployment
1. Push image to Docker Hub or RunPod registry
2. Create new Pod with GPU (RTX 3080+ recommended)
3. Set environment variables
4. Mount volume with SAM weights at `/models`
5. Expose port 8000

### RunPod Serverless Deployment
1. Push image to registry
2. Create new Serverless Endpoint
3. Set `RUNPOD_SERVERLESS=true`
4. Configure network volume with SAM weights
5. Set container start command (uses entrypoint by default)
